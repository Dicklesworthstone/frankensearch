//! Model download system with progress reporting and atomic installation.
//!
//! Downloads model files from `HuggingFace` with SHA-256 verification,
//! atomic installation (rename-over), and progress callbacks.
//!
//! Gated behind the `download` feature flag to keep the core crate network-free.

use std::collections::BTreeSet;
use std::fmt;
use std::fs::{File, OpenOptions, TryLockError};
use std::future::{Future, poll_fn};
use std::io::{BufReader, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::Poll;
use std::time::{Duration, Instant};

use asupersync::bytes::Buf;
use asupersync::channel::oneshot;
use asupersync::http::body::{Body, Frame};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use asupersync::Cx;
use asupersync::http::h1::{
    ClientError, HttpClient, HttpClientConfig, Method, ParsedUrl, RedirectPolicy, Scheme,
};
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::QuantizationFormat;

use crate::model_manifest::{
    FrozenModelArtifactManifestV1, ModelArtifactFileV1, ModelArtifactRoleV1, ModelFile,
    ModelLifecycle, ModelManifest,
};
use crate::proxy::{ProxySetting, proxy_from_environment};

static STAGING_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);
const LOCAL_COPY_BUFFER_BYTES: usize = 1024 * 1024;
const IN_MEMORY_F32_STORAGE_FORMAT_V1: &str = "in-memory-f32-v1";

/// Schema version for durable acquisition receipts.
pub const MODEL_ACQUISITION_RECEIPT_SCHEMA_V1: u16 = 1;
/// Schema version for path-free acquisition progress records.
pub const MODEL_ACQUISITION_PROGRESS_SCHEMA_V1: u16 = 1;
/// Schema version for path-free acquisition recovery diagnostics.
pub const MODEL_ACQUISITION_RECOVERY_SCHEMA_V1: u16 = 1;

/// Default per-artifact response body cap for model downloads.
///
/// The underlying HTTP codec defaults to a 16 MiB body limit, which is smaller
/// than several production-ready model artifacts in the built-in catalog (e.g.
/// the default Potion tokenizer is ~17.8 MiB and its model weights ~489 MiB).
/// Files are streamed to disk with bounded memory and verified against the
/// manifest's declared size and SHA-256, so this cap serves only as a finite
/// resource guard. 2 GiB comfortably covers the current catalog while keeping a
/// bounded ceiling; callers may override it via [`DownloadConfig`].
pub const DEFAULT_MAX_MODEL_ARTIFACT_BYTES: usize = 2 * 1024 * 1024 * 1024;

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for model downloads.
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    /// Maximum retries per file on transient failure.
    pub max_retries: u32,
    /// Base delay for exponential backoff between retries.
    pub retry_base_delay: Duration,
    /// Maximum wait for each request's connection, proxy handshake and response
    /// headers, including redirect hops individually. Defaults to 60 seconds.
    pub request_timeout: Duration,
    /// Maximum time without receiving nonempty model bytes after the response
    /// headers. Defaults to 60 seconds. Progressing large downloads have no
    /// total wall-clock limit; the caller's shorter deadline still applies.
    pub idle_timeout: Duration,
    /// User-Agent header value.
    pub user_agent: String,
    /// Maximum redirects to follow. Proxy selection and bypass rules are
    /// reevaluated for each destination. HTTPS-to-HTTP downgrades are rejected.
    pub max_redirects: u32,
    /// Maximum size, in bytes, of a single downloaded model artifact.
    ///
    /// Forwarded to the HTTP client's body-size limit. The codec default of
    /// 16 MiB is too small for real model artifacts, so this defaults to
    /// [`DEFAULT_MAX_MODEL_ARTIFACT_BYTES`]. Downloads are streamed to disk and
    /// verified against the manifest's declared size and SHA-256, so this bound
    /// only guards against an unexpectedly large response.
    pub max_response_bytes: usize,
    /// How downloads reach the network. Defaults to the conventional proxy
    /// environment variables (`https_proxy`, `all_proxy`, `no_proxy`, ...); see
    /// [`ProxySetting`].
    pub proxy: ProxySetting,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_base_delay: Duration::from_secs(1),
            request_timeout: Duration::from_secs(60),
            idle_timeout: Duration::from_secs(60),
            user_agent: format!("frankensearch/{}", env!("CARGO_PKG_VERSION")),
            max_redirects: 5,
            max_response_bytes: DEFAULT_MAX_MODEL_ARTIFACT_BYTES,
            proxy: ProxySetting::FromEnvironment,
        }
    }
}

// ─── Progress ───────────────────────────────────────────────────────────────

/// Progress information for an in-flight model download.
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Name of the file currently being downloaded.
    pub file_name: String,
    /// Bytes downloaded so far (current file).
    pub bytes_downloaded: u64,
    /// Total bytes expected (current file), if known.
    pub total_bytes: Option<u64>,
    /// Number of files completed so far.
    pub files_completed: usize,
    /// Total number of files to download.
    pub files_total: usize,
    /// Estimated download speed in bytes per second.
    pub speed_bytes_per_sec: f64,
    /// Estimated time remaining in seconds, if calculable.
    pub eta_seconds: Option<f64>,
}

/// Authorized byte source for one explicit acquisition.
pub enum ModelAcquisitionSource<'a> {
    /// Download the immutable URLs carried by the frozen manifest.
    Network,
    /// Copy the complete artifact tree rooted at this local directory.
    LocalBundle(&'a Path),
}

/// Inputs for one explicit frozen-model acquisition transaction.
pub struct ModelAcquisitionRequest<'a> {
    /// Frozen manifest that authorizes every accepted byte.
    pub frozen_manifest: &'a FrozenModelArtifactManifestV1,
    /// Explicit transport selected by the caller.
    pub source: ModelAcquisitionSource<'a>,
    /// Final generation directory.
    pub destination_dir: &'a Path,
}

impl fmt::Debug for ModelAcquisitionRequest<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelAcquisitionRequest")
            .field("manifest_fingerprint", &self.frozen_manifest.fingerprint)
            .field("source", &self.source)
            .field("destination_dir", &"<redacted-path>")
            .finish()
    }
}

impl fmt::Debug for ModelAcquisitionSource<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Network => formatter.write_str("Network"),
            Self::LocalBundle(_) => formatter.write_str("LocalBundle(<redacted-path>)"),
        }
    }
}

/// Content source recorded in the path-free durable receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAcquisitionSourceKindV1 {
    /// Immutable HTTPS artifacts from the manifest.
    Network,
    /// Explicit operator-supplied local bundle.
    LocalBundle,
    /// Existing destination was already complete and verified.
    WarmCache,
}

/// Publication result recorded in the durable receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAcquisitionOutcomeV1 {
    /// A newly verified generation was published.
    Published,
    /// The existing generation was reverified and reused without transport.
    VerifiedWarmCache,
}

/// Why acquisition did or did not reuse the destination cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAcquisitionCacheReasonV1 {
    /// No destination generation was present.
    DestinationMissing,
    /// A destination existed but failed frozen-manifest verification.
    DestinationRejected,
    /// The complete destination generation verified and was reused.
    FrozenGenerationVerified,
}

/// Current phase represented by an acquisition progress record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAcquisitionStageV1 {
    /// Registered artifact bytes are being streamed.
    Transporting,
    /// The complete staged artifact set passed size and SHA-256 verification.
    StagedVerified,
    /// The verified artifact set passed the caller's load self-test.
    LoadSelfTestPassed,
    /// The verified generation was durably published.
    Published,
    /// The existing destination passed verification and load self-test.
    WarmCacheVerified,
    /// Acquisition failed without declaring semantic readiness.
    Failed,
    /// Structured cancellation was observed.
    Cancelled,
}

/// Verification result represented by an acquisition progress record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAcquisitionVerificationResultV1 {
    /// Streaming is still in progress.
    Pending,
    /// The phase's verification boundary passed.
    Passed,
    /// The transaction failed.
    Failed,
    /// The transaction observed structured cancellation.
    Cancelled,
}

/// Bounded, path-free progress for one frozen-model acquisition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelAcquisitionProgressV1 {
    /// Progress schema version.
    pub schema_version: u16,
    /// Logical semantic model ID.
    pub logical_model_id: String,
    /// Immutable upstream revision.
    pub upstream_revision: String,
    /// Full frozen-manifest fingerprint.
    pub artifact_manifest_fingerprint: String,
    /// Actual or requested transport.
    pub source: ModelAcquisitionSourceKindV1,
    /// Credential-free source host for the current network artifact.
    pub source_host: Option<String>,
    /// Semantic role of the current artifact, if transport is active.
    pub artifact_role: Option<ModelArtifactRoleV1>,
    /// Exact registered SHA-256 for the current artifact.
    pub artifact_sha256: Option<String>,
    /// Transaction phase.
    pub stage: ModelAcquisitionStageV1,
    /// Bounded bytes processed for the current artifact or complete set.
    pub bytes_processed: u64,
    /// Exact declared bytes for the current artifact or complete set.
    pub total_bytes: u64,
    /// Completed artifact count.
    pub files_completed: usize,
    /// Registered artifact count.
    pub files_total: usize,
    /// Saturating duration since transaction start.
    pub duration_millis: u64,
    /// Cache disposition, once known.
    pub cache_reason: Option<ModelAcquisitionCacheReasonV1>,
    /// Verification outcome for this phase.
    pub verification_result: ModelAcquisitionVerificationResultV1,
}

/// Path-free inventory of retained acquisition recovery artifacts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelAcquisitionRecoveryV1 {
    /// Recovery diagnostic schema version.
    pub schema_version: u16,
    /// Whether the requested destination currently exists.
    pub destination_present: bool,
    /// Unique streaming stages retained after incomplete attempts.
    pub orphan_staging_generations: usize,
    /// Verified stages renamed into the publication lane but not committed.
    pub interrupted_installing_generations: usize,
    /// Prior generations preserved by replacement attempts.
    pub preserved_backup_generations: usize,
    /// Whether the persistent advisory lock file has been initialized.
    pub advisory_lock_present: bool,
}

/// Diagnose retained acquisition state without deleting or opening artifacts.
///
/// The result contains counts only, never paths or content. Garbage collection
/// is intentionally not implicit; a separate operator action must supply an
/// explicit retention policy.
///
/// # Errors
///
/// Returns a typed I/O or configuration error if the destination parent cannot
/// be inspected safely.
pub fn diagnose_model_acquisition(
    destination_dir: &Path,
) -> SearchResult<ModelAcquisitionRecoveryV1> {
    let parent = destination_dir
        .parent()
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "model_acquisition.destination".to_owned(),
            value: "redacted".to_owned(),
            reason: "destination must have a parent directory".to_owned(),
        })?;
    let destination_name = destination_dir
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "model_acquisition.destination".to_owned(),
            value: "redacted".to_owned(),
            reason: "destination must end in a valid UTF-8 directory name".to_owned(),
        })?;
    let staging_prefix = format!(".{destination_name}-download-");
    let installing_prefix = format!(".{destination_name}.installing.");
    let backup_prefix = format!("{destination_name}.backup.");
    let lock_name = format!(".{destination_name}.acquisition.lock");
    let mut recovery = ModelAcquisitionRecoveryV1 {
        schema_version: MODEL_ACQUISITION_RECOVERY_SCHEMA_V1,
        destination_present: destination_dir.exists(),
        orphan_staging_generations: 0,
        interrupted_installing_generations: 0,
        preserved_backup_generations: 0,
        advisory_lock_present: false,
    };
    if !parent.exists() {
        return Ok(recovery);
    }
    for entry in std::fs::read_dir(parent).map_err(SearchError::from)? {
        let entry = entry.map_err(SearchError::from)?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name.starts_with(&staging_prefix) {
            recovery.orphan_staging_generations =
                recovery.orphan_staging_generations.saturating_add(1);
        } else if name.starts_with(&installing_prefix) {
            recovery.interrupted_installing_generations = recovery
                .interrupted_installing_generations
                .saturating_add(1);
        } else if name.starts_with(&backup_prefix) {
            recovery.preserved_backup_generations =
                recovery.preserved_backup_generations.saturating_add(1);
        } else if name == lock_name {
            recovery.advisory_lock_present = true;
        }
    }
    Ok(recovery)
}

/// Path-free evidence that a frozen model generation was acquired.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelAcquisitionReceiptV1 {
    /// Receipt schema version.
    pub schema_version: u16,
    /// Logical semantic model ID.
    pub logical_model_id: String,
    /// Immutable upstream revision.
    pub upstream_revision: String,
    /// Pinned SPDX license assertion.
    pub license_spdx: String,
    /// Fingerprint of the full frozen artifact manifest.
    pub artifact_manifest_fingerprint: String,
    /// Mathematical embedding-space fingerprint.
    pub space_fingerprint: String,
    /// Local implementation/producer fingerprint.
    pub producer_fingerprint: String,
    /// Actual acquisition source.
    pub source: ModelAcquisitionSourceKindV1,
    /// Credential-free source hosts; empty for local/warm sources.
    pub source_hosts: Vec<String>,
    /// Publication or warm-cache outcome.
    pub outcome: ModelAcquisitionOutcomeV1,
    /// Exact registered artifact bytes verified.
    pub bytes_verified: u64,
    /// Bounded operation duration.
    pub duration_millis: u64,
    /// Whether an earlier destination generation was preserved as a backup.
    pub prior_generation_preserved: bool,
    /// Acquisition never implies compatibility with an existing index.
    pub requires_reindex: bool,
}

/// Verified but unpublished staging evidence.
pub struct VerifiedModelStageV1 {
    staging_dir: PathBuf,
    manifest_fingerprint: String,
    source: ModelAcquisitionSourceKindV1,
    source_hosts: Vec<String>,
    bytes_verified: u64,
}

impl VerifiedModelStageV1 {
    /// Authorized staging directory used by the load self-test and publisher.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.staging_dir
    }
}

impl fmt::Debug for VerifiedModelStageV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedModelStageV1")
            .field("staging_dir", &"<redacted-path>")
            .field("manifest_fingerprint", &self.manifest_fingerprint)
            .field("source", &self.source)
            .field("source_hosts", &self.source_hosts)
            .field("bytes_verified", &self.bytes_verified)
            .finish()
    }
}

impl fmt::Display for DownloadProgress {
    #[allow(clippy::cast_precision_loss)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pct = self
            .total_bytes
            .filter(|&t| t > 0)
            .map(|t| self.bytes_downloaded as f64 / t as f64 * 100.0);

        if let Some(pct) = pct {
            write!(
                f,
                "[{}/{}] {} {:.0}% ({}/{})",
                self.files_completed + 1,
                self.files_total,
                self.file_name,
                pct,
                format_bytes(self.bytes_downloaded),
                format_bytes(self.total_bytes.unwrap_or(0)),
            )
        } else {
            write!(
                f,
                "[{}/{}] {} {}",
                self.files_completed + 1,
                self.files_total,
                self.file_name,
                format_bytes(self.bytes_downloaded),
            )
        }
    }
}

// ─── Downloader ─────────────────────────────────────────────────────────────

/// Downloads model files from `HuggingFace` with verification and progress reporting.
pub struct ModelDownloader {
    config: DownloadConfig,
}

impl ModelDownloader {
    /// Create a new downloader with the given configuration.
    #[must_use]
    pub const fn new(config: DownloadConfig) -> Self {
        Self { config }
    }

    /// The HTTP client for a request to `url`. The proxy depends on the URL (its scheme
    /// selects the environment variable and `no_proxy` matches its host), and the client
    /// fixes its proxy at construction. Building one is cheap: TLS and connections are
    /// set up per request, and each artifact is one large streamed request.
    fn http_client_for(&self, url: &str) -> HttpClient {
        let mut client_config = HttpClientConfig::default();
        // Follow redirects here rather than inside a fixed-route client: a CDN
        // or mirror can have a different scheme or `no_proxy` policy.
        client_config.redirect_policy = RedirectPolicy::None;
        client_config.user_agent = Some(self.config.user_agent.clone());
        // Lift the codec's 16 MiB default body cap; model artifacts routinely
        // exceed it. Streaming-to-disk keeps memory bounded regardless.
        client_config.max_body_size = Some(self.config.max_response_bytes);
        client_config.request_timeout = Some(self.config.request_timeout);
        client_config.proxy_url = proxy_from_environment(&self.config.proxy, url);
        HttpClient::with_config(client_config)
    }

    /// Create a downloader with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(DownloadConfig::default())
    }

    /// Acquire, verify, load-test, and publish one frozen model generation.
    ///
    /// The destination is protected by a persistent sibling lock file. A valid
    /// warm generation is load-tested and returned without consulting the
    /// requested transport. New bytes are streamed into a unique sibling stage,
    /// verified against the frozen manifest, load-tested, synced, and published.
    /// An earlier destination is preserved as a uniquely named backup.
    ///
    /// Acquisition deliberately ends in
    /// [`crate::model_manifest::ModelState::AcquiredNeedsReindex`]. Selecting or
    /// rebuilding a compatible vector index is a separate readiness decision.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid manifests, lock contention,
    /// cancellation, transport failures, filesystem failures, integrity drift,
    /// load-test failures, or publication failures.
    pub async fn acquire_frozen_model(
        &self,
        cx: &Cx,
        request: ModelAcquisitionRequest<'_>,
        lifecycle: &mut ModelLifecycle,
        on_progress: impl Fn(&ModelAcquisitionProgressV1) + Send + Sync,
        load_self_test: impl Fn(&Path, &FrozenModelArtifactManifestV1) -> SearchResult<()> + Send + Sync,
    ) -> SearchResult<ModelAcquisitionReceiptV1> {
        let started = Instant::now();
        let ModelAcquisitionRequest {
            frozen_manifest,
            source,
            destination_dir,
        } = request;
        let requested_source = match &source {
            ModelAcquisitionSource::Network => ModelAcquisitionSourceKindV1::Network,
            ModelAcquisitionSource::LocalBundle(_) => ModelAcquisitionSourceKindV1::LocalBundle,
        };
        let mut observed_cache_reason = None;
        let operation = async {
            frozen_manifest.validate()?;
            validate_lifecycle_manifest(frozen_manifest, lifecycle.manifest())?;
            check_acquisition_cancel(cx, "acquisition-start")?;

            let _lock = AcquisitionLock::acquire(destination_dir)?;
            check_acquisition_cancel(cx, "acquisition-lock")?;

            let cache_reason = if destination_dir.is_dir()
                && frozen_manifest.manifest.verify_dir(destination_dir).is_ok()
            {
                ModelAcquisitionCacheReasonV1::FrozenGenerationVerified
            } else if destination_dir.exists() {
                ModelAcquisitionCacheReasonV1::DestinationRejected
            } else {
                ModelAcquisitionCacheReasonV1::DestinationMissing
            };
            observed_cache_reason = Some(cache_reason);

            if cache_reason == ModelAcquisitionCacheReasonV1::FrozenGenerationVerified {
                check_acquisition_cancel(cx, "warm-cache-self-test")?;
                load_self_test(destination_dir, frozen_manifest)?;
                check_acquisition_cancel(cx, "warm-cache-complete")?;
                lifecycle.mark_acquired_needs_reindex()?;
                on_progress(&acquisition_phase_progress(
                    frozen_manifest,
                    ModelAcquisitionSourceKindV1::WarmCache,
                    ModelAcquisitionStageV1::WarmCacheVerified,
                    started.elapsed(),
                    Some(cache_reason),
                    ModelAcquisitionVerificationResultV1::Passed,
                )?);
                return acquisition_receipt(
                    frozen_manifest,
                    ModelAcquisitionSourceKindV1::WarmCache,
                    Vec::new(),
                    ModelAcquisitionOutcomeV1::VerifiedWarmCache,
                    total_frozen_bytes(frozen_manifest)?,
                    started.elapsed(),
                    false,
                );
            }

            let stage = match source {
                ModelAcquisitionSource::Network => {
                    let download_manifest = legacy_manifest_from_frozen(frozen_manifest)?;
                    let forward_progress = |progress: &DownloadProgress| {
                        if let Ok(progress) = acquisition_transport_progress(
                            frozen_manifest,
                            ModelAcquisitionSourceKindV1::Network,
                            progress,
                            started.elapsed(),
                            Some(cache_reason),
                        ) {
                            on_progress(&progress);
                        }
                    };
                    let staging_dir = self
                        .download_model(
                            cx,
                            &download_manifest,
                            destination_dir,
                            lifecycle,
                            forward_progress,
                        )
                        .await?;
                    VerifiedModelStageV1 {
                        staging_dir,
                        manifest_fingerprint: frozen_manifest.fingerprint.clone(),
                        source: ModelAcquisitionSourceKindV1::Network,
                        source_hosts: frozen_source_hosts(frozen_manifest)?,
                        bytes_verified: total_frozen_bytes(frozen_manifest)?,
                    }
                }
                ModelAcquisitionSource::LocalBundle(source_dir) => {
                    let forward_progress = |progress: &DownloadProgress| {
                        if let Ok(progress) = acquisition_transport_progress(
                            frozen_manifest,
                            ModelAcquisitionSourceKindV1::LocalBundle,
                            progress,
                            started.elapsed(),
                            Some(cache_reason),
                        ) {
                            on_progress(&progress);
                        }
                    };
                    Self::stage_local_bundle(
                        cx,
                        frozen_manifest,
                        source_dir,
                        destination_dir,
                        lifecycle,
                        &forward_progress,
                    )?
                }
            };

            if stage.manifest_fingerprint != frozen_manifest.fingerprint {
                return Err(SearchError::InvalidConfig {
                    field: "model_acquisition.stage_manifest".to_owned(),
                    value: "mismatch".to_owned(),
                    reason: "verified stage does not bind the requested frozen manifest".to_owned(),
                });
            }
            frozen_manifest.manifest.verify_dir(stage.path())?;
            on_progress(&acquisition_phase_progress(
                frozen_manifest,
                stage.source,
                ModelAcquisitionStageV1::StagedVerified,
                started.elapsed(),
                Some(cache_reason),
                ModelAcquisitionVerificationResultV1::Passed,
            )?);
            check_acquisition_cancel(cx, "load-self-test")?;
            load_self_test(stage.path(), frozen_manifest)?;
            on_progress(&acquisition_phase_progress(
                frozen_manifest,
                stage.source,
                ModelAcquisitionStageV1::LoadSelfTestPassed,
                started.elapsed(),
                Some(cache_reason),
                ModelAcquisitionVerificationResultV1::Passed,
            )?);
            check_acquisition_cancel(cx, "atomic-publication")?;
            let backup = frozen_manifest
                .manifest
                .promote_verified_installation(stage.path(), destination_dir)?;
            lifecycle.mark_acquired_needs_reindex()?;
            on_progress(&acquisition_phase_progress(
                frozen_manifest,
                stage.source,
                ModelAcquisitionStageV1::Published,
                started.elapsed(),
                Some(cache_reason),
                ModelAcquisitionVerificationResultV1::Passed,
            )?);

            acquisition_receipt(
                frozen_manifest,
                stage.source,
                stage.source_hosts,
                ModelAcquisitionOutcomeV1::Published,
                stage.bytes_verified,
                started.elapsed(),
                backup.is_some(),
            )
        }
        .await;

        if let Err(error) = &operation {
            let cancelled = matches!(error, SearchError::Cancelled { .. });
            if cancelled {
                lifecycle.cancel();
            } else {
                lifecycle.fail_verification(bounded_download_failure_reason(error));
            }
            if let Ok(progress) = acquisition_phase_progress(
                frozen_manifest,
                requested_source,
                if cancelled {
                    ModelAcquisitionStageV1::Cancelled
                } else {
                    ModelAcquisitionStageV1::Failed
                },
                started.elapsed(),
                observed_cache_reason,
                if cancelled {
                    ModelAcquisitionVerificationResultV1::Cancelled
                } else {
                    ModelAcquisitionVerificationResultV1::Failed
                },
            ) {
                on_progress(&progress);
            }
        }
        operation
    }

    fn stage_local_bundle(
        cx: &Cx,
        frozen: &FrozenModelArtifactManifestV1,
        source_dir: &Path,
        destination_dir: &Path,
        lifecycle: &mut ModelLifecycle,
        on_progress: &(impl Fn(&DownloadProgress) + Send + Sync),
    ) -> SearchResult<VerifiedModelStageV1> {
        let total_bytes = total_frozen_bytes(frozen)?;
        lifecycle.begin_download(total_bytes)?;
        check_acquisition_cancel(cx, "local-stage-start")?;
        let staging_dir = create_unique_staging_dir(destination_dir)?;
        let files_total = frozen.manifest.artifacts.len();
        let mut cumulative_bytes = 0_u64;

        for (file_idx, artifact) in frozen.manifest.artifacts.iter().enumerate() {
            check_acquisition_cancel(cx, "local-stage-copy")?;
            copy_local_artifact(
                cx,
                source_dir,
                &staging_dir,
                artifact,
                file_idx,
                files_total,
                on_progress,
            )?;
            cumulative_bytes = cumulative_bytes.checked_add(artifact.size).ok_or_else(|| {
                SearchError::InvalidConfig {
                    field: "artifacts[].size".to_owned(),
                    value: "overflow".to_owned(),
                    reason: "total registered artifact bytes must fit in u64".to_owned(),
                }
            })?;
            lifecycle.update_download_progress(cumulative_bytes)?;
        }

        check_acquisition_cancel(cx, "local-stage-verification")?;
        lifecycle.begin_verification()?;
        frozen.manifest.verify_dir(&staging_dir)?;
        lifecycle.mark_staged_verified()?;
        Ok(VerifiedModelStageV1 {
            staging_dir,
            manifest_fingerprint: frozen.fingerprint.clone(),
            source: ModelAcquisitionSourceKindV1::LocalBundle,
            source_hosts: Vec::new(),
            bytes_verified: total_bytes,
        })
    }

    /// Download all files for a model manifest into a staging directory.
    ///
    /// A unique staging directory is created under `{dest_dir}` for each call
    /// (for example, `.download-<pid>-<counter>`), and files are placed there
    /// during download. After all files are verified, the
    /// caller should use [`ModelManifest::promote_verified_installation`] to
    /// atomically install the model.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` on network failure, hash mismatch, or I/O error.
    pub async fn download_model(
        &self,
        cx: &Cx,
        manifest: &ModelManifest,
        dest_dir: &Path,
        lifecycle: &mut ModelLifecycle,
        on_progress: impl Fn(&DownloadProgress) + Send + Sync,
    ) -> SearchResult<PathBuf> {
        if let Err(error) = validate_download_config(&self.config) {
            lifecycle.fail_verification(bounded_download_failure_reason(&error));
            return Err(error);
        }
        if let Err(error) = check_acquisition_cancel(cx, "download-start") {
            lifecycle.cancel();
            return Err(error);
        }
        let total_bytes = manifest.total_size_bytes();
        lifecycle.begin_download(total_bytes.max(1))?;

        if let Err(err) = manifest.validate() {
            lifecycle.fail_verification(format!(
                "manifest validation failed for '{}': {err}",
                manifest.id
            ));
            return Err(err);
        }

        if !manifest.is_production_ready() {
            let reason = format!(
                "manifest for '{}' must be production-ready (pinned revision + verified checksums) before download",
                manifest.id
            );
            lifecycle.fail_verification(reason.clone());
            return Err(SearchError::InvalidConfig {
                field: "manifest".to_owned(),
                value: manifest.id.clone(),
                reason,
            });
        }

        // Fail fast with an actionable error if any declared artifact exceeds the
        // configured response cap, rather than failing cryptically mid-stream with
        // a `BodyTooLarge` codec error after exhausting all retries.
        let cap = self.config.max_response_bytes as u64;
        if let Some(oversized) = manifest.files.iter().find(|file| file.size > cap) {
            let reason = format!(
                "model artifact '{}' declares {} bytes, which exceeds the configured \
                 download response cap of {cap} bytes; raise DownloadConfig.max_response_bytes",
                oversized.name, oversized.size
            );
            lifecycle.fail_verification(reason.clone());
            return Err(SearchError::InvalidConfig {
                field: "download.max_response_bytes".to_owned(),
                value: manifest.id.clone(),
                reason,
            });
        }

        let staging_dir = match create_unique_staging_dir(dest_dir) {
            Ok(path) => path,
            Err(err) => {
                lifecycle.fail_verification(format!(
                    "failed to create staging directory for '{}': {err}",
                    manifest.id
                ));
                return Err(err);
            }
        };

        let files_total = manifest.files.len();
        let mut cumulative_bytes: u64 = 0;

        for (idx, file) in manifest.files.iter().enumerate() {
            if let Err(error) = check_acquisition_cancel(cx, "download-file-start") {
                lifecycle.cancel();
                return Err(error);
            }
            let url = file.download_url(&manifest.repo, &manifest.revision);
            let file_dest = staging_dir.join(&file.name);

            // Create parent directories for nested paths (e.g., "onnx/model.onnx").
            if let Some(parent) = file_dest.parent()
                && let Err(err) = std::fs::create_dir_all(parent).map_err(SearchError::from)
            {
                lifecycle.fail_verification(format!(
                    "failed to create parent directory for '{}': {err}",
                    file.name
                ));
                return Err(err);
            }

            info!(
                file = %file.name,
                size = file.size,
                "downloading model file"
            );

            if let Err(err) = self
                .download_file_with_retry(
                    cx,
                    &url,
                    &file_dest,
                    file,
                    idx,
                    files_total,
                    &on_progress,
                )
                .await
            {
                if matches!(err, SearchError::Cancelled { .. }) {
                    lifecycle.cancel();
                } else {
                    lifecycle.fail_verification(format!(
                        "download failed for '{}': {}",
                        file.name,
                        bounded_download_failure_reason(&err)
                    ));
                }
                return Err(err);
            }

            cumulative_bytes = cumulative_bytes.saturating_add(file.size);
            if let Err(err) = lifecycle.update_download_progress(cumulative_bytes) {
                lifecycle.fail_verification(format!("failed to update download progress: {err}"));
                return Err(err);
            }
        }

        // Verify all files.
        if let Err(error) = check_acquisition_cancel(cx, "download-verification") {
            lifecycle.cancel();
            return Err(error);
        }
        if let Err(err) = lifecycle.begin_verification() {
            lifecycle.fail_verification(format!(
                "failed to transition model lifecycle into verification: {err}"
            ));
            return Err(err);
        }
        info!(model = %manifest.id, "verifying downloaded files");
        match manifest.verify_dir(&staging_dir) {
            Ok(()) => {
                lifecycle.mark_staged_verified()?;
                info!(model = %manifest.id, "model download stage complete and verified");
                Ok(staging_dir)
            }
            Err(e) => {
                lifecycle.fail_verification(bounded_download_failure_reason(&e));
                Err(e)
            }
        }
    }

    /// Download a single file with retry logic.
    async fn download_file_with_retry(
        &self,
        cx: &Cx,
        url: &str,
        dest: &Path,
        file: &ModelFile,
        file_idx: usize,
        files_total: usize,
        on_progress: &(impl Fn(&DownloadProgress) + Send + Sync),
    ) -> SearchResult<()> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            check_acquisition_cancel(cx, "download-retry")?;
            if attempt > 0 {
                let delay = self
                    .config
                    .retry_base_delay
                    .saturating_mul(2_u32.saturating_pow(attempt - 1));
                warn!(
                    file = %file.name,
                    attempt,
                    delay_ms = delay.as_millis(),
                    "retrying download after failure"
                );
                wait_download_retry(cx, url, delay).await?;
            }

            match self
                .download_single_file(cx, url, dest, file, file_idx, files_total, on_progress)
                .await
            {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if matches!(e, SearchError::Cancelled { .. }) {
                        return Err(e);
                    }
                    warn!(
                        file = %file.name,
                        attempt,
                        reason = bounded_download_failure_reason(&e),
                        "download attempt failed"
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| SearchError::ModelLoadFailed {
            path: dest.to_path_buf(),
            source: "download failed after all retries".into(),
        }))
    }

    /// Download a single file (one attempt).
    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    async fn download_single_file(
        &self,
        cx: &Cx,
        url: &str,
        dest: &Path,
        file: &ModelFile,
        file_idx: usize,
        files_total: usize,
        on_progress: &(impl Fn(&DownloadProgress) + Send + Sync),
    ) -> SearchResult<()> {
        validate_download_config(&self.config)?;
        let start = Instant::now();
        check_acquisition_cancel(cx, "download-request")?;

        // Report start.
        on_progress(&DownloadProgress {
            file_name: file.name.clone(),
            bytes_downloaded: 0,
            total_bytes: if file.size > 0 { Some(file.size) } else { None },
            files_completed: file_idx,
            files_total,
            speed_bytes_per_sec: 0.0,
            eta_seconds: None,
        });

        // Every hop gets its own route. No cookies or origin credentials are
        // carried between clients; proxy authentication belongs to the newly
        // selected proxy only. Drop redirect bodies without buffering them.
        let mut request_url = url.to_owned();
        let mut redirects = 0_u32;
        let response = loop {
            check_acquisition_cancel(cx, "download-request")?;
            let response =
                run_download_request(cx, &request_url, self.config.request_timeout, async {
                    self.http_client_for(&request_url)
                        .request_streaming(cx, Method::Get, &request_url, Vec::new(), Vec::new())
                        .await
                        .map_err(|e| client_error_to_search(e, &request_url))
                })
                .await?;
            check_acquisition_cancel(cx, "download-response")?;
            if !matches!(response.head.status, 301 | 302 | 303 | 307 | 308) {
                break response;
            }
            let Some((_, location)) = response
                .head
                .headers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case("location"))
            else {
                break response;
            };
            if redirects >= self.config.max_redirects {
                return Err(client_error_to_search(
                    ClientError::TooManyRedirects {
                        count: redirects.saturating_add(1),
                        max: self.config.max_redirects,
                    },
                    &request_url,
                ));
            }
            let next_url = resolve_download_redirect(&request_url, location)
                .map_err(|e| client_error_to_search(e, &request_url))?;
            drop(response);
            request_url = next_url;
            redirects = redirects.saturating_add(1);
        };

        // Check HTTP status.
        if response.head.status < 200 || response.head.status >= 300 {
            return Err(SearchError::ModelLoadFailed {
                path: PathBuf::from("<redacted-model-stage>"),
                source: format!(
                    "HTTP {} from host {}",
                    response.head.status,
                    diagnostic_source_host(url)
                )
                .into(),
            });
        }
        if let Some(content_length) = response_content_length(&response.head.headers)
            && file.size > 0
            && content_length != file.size
        {
            return Err(SearchError::HashMismatch {
                path: dest.to_path_buf(),
                expected: format!("content-length={}", file.size),
                actual: format!("content-length={content_length}"),
            });
        }

        let tmp_path = dest.with_extension("tmp");
        let mut tmp_guard = TempFileGuard::new(tmp_path.clone());
        let mut tmp_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)
            .map_err(SearchError::from)?;

        let total_bytes = if file.size > 0 {
            Some(file.size)
        } else {
            response_content_length(&response.head.headers)
        };
        let mut hasher = Sha256::new();
        let mut bytes_downloaded: u64 = 0;

        stream_download_body(cx, url, response.body, self.config.idle_timeout, |bytes| {
            let byte_count = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
            let next_size = bytes_downloaded.checked_add(byte_count).ok_or_else(|| {
                SearchError::HashMismatch {
                    path: dest.to_path_buf(),
                    expected: format!("size={}", file.size),
                    actual: "size-overflow".to_owned(),
                }
            })?;
            if file.size > 0 && next_size > file.size {
                return Err(SearchError::HashMismatch {
                    path: dest.to_path_buf(),
                    expected: format!("size={}", file.size),
                    actual: format!("size-at-least={next_size}"),
                });
            }
            tmp_file.write_all(bytes).map_err(SearchError::from)?;
            hasher.update(bytes);
            bytes_downloaded = next_size;
            let elapsed = start.elapsed();
            let speed = if elapsed.as_secs_f64() > 0.0 {
                bytes_downloaded as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };
            let eta_seconds = total_bytes.and_then(|total| {
                if speed <= f64::EPSILON || bytes_downloaded >= total {
                    None
                } else {
                    Some((total.saturating_sub(bytes_downloaded)) as f64 / speed)
                }
            });
            on_progress(&DownloadProgress {
                file_name: file.name.clone(),
                bytes_downloaded,
                total_bytes,
                files_completed: file_idx,
                files_total,
                speed_bytes_per_sec: speed,
                eta_seconds,
            });
            Ok(())
        })
        .await?;

        check_acquisition_cancel(cx, "download-file-sync")?;
        // ubs:ignore — artifact byte counts are public integrity metadata.
        if file.size > 0 && bytes_downloaded != file.size {
            return Err(SearchError::HashMismatch {
                path: dest.to_path_buf(),
                expected: format!("size={}", file.size),
                actual: format!("size={bytes_downloaded}"),
            });
        }

        // Verify SHA-256 only when the manifest provides a concrete checksum.
        if file.has_verified_checksum() {
            let actual_hash = sha256_digest_hex(hasher.finalize().as_slice());
            // ubs:ignore — pinned artifact digests are public integrity metadata.
            if actual_hash != file.sha256 {
                return Err(SearchError::HashMismatch {
                    path: dest.to_path_buf(),
                    expected: format!("sha256={},size={}", file.sha256, file.size),
                    actual: format!("sha256={actual_hash},size={bytes_downloaded}"),
                });
            }
        }

        tmp_file.flush().map_err(SearchError::from)?;
        tmp_file.sync_all().map_err(SearchError::from)?;
        drop(tmp_file);
        check_acquisition_cancel(cx, "download-file-publish")?;
        std::fs::rename(&tmp_path, dest).map_err(SearchError::from)?;
        tmp_guard.disarm();

        let elapsed = start.elapsed();
        let speed = if elapsed.as_secs_f64() > 0.0 {
            bytes_downloaded as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        on_progress(&DownloadProgress {
            file_name: file.name.clone(),
            bytes_downloaded,
            total_bytes: if file.size > 0 {
                Some(file.size)
            } else {
                total_bytes
            },
            files_completed: file_idx,
            files_total,
            speed_bytes_per_sec: speed,
            eta_seconds: Some(0.0),
        });

        info!(
            file = %file.name,
            bytes = bytes_downloaded,
            elapsed_ms = elapsed.as_millis(),
            "file saved"
        );

        Ok(())
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn validate_download_config(config: &DownloadConfig) -> SearchResult<()> {
    for (field, value) in [
        ("download.request_timeout", config.request_timeout),
        ("download.idle_timeout", config.idle_timeout),
    ] {
        if value.is_zero() {
            return Err(SearchError::InvalidConfig {
                field: field.to_owned(),
                value: "0ms".to_owned(),
                reason: "must be greater than zero".to_owned(),
            });
        }
    }
    if config.max_response_bytes == 0 {
        return Err(SearchError::InvalidConfig {
            field: "download.max_response_bytes".to_owned(),
            value: "0".to_owned(),
            reason: "must be greater than zero".to_owned(),
        });
    }
    Ok(())
}

/// Bound a pending request independently of socket activity. The transport's
/// own timeout covers response headers, but its I/O cannot wake for every Cx
/// cancellation (notably a silent CONNECT or SOCKS handshake).
async fn run_download_request<T: Send>(
    cx: &Cx,
    url: &str,
    timeout: Duration,
    work: impl Future<Output = SearchResult<T>> + Send,
) -> SearchResult<T> {
    let started = cx.now();
    let ambient_started = asupersync::time::wall_now();
    let wall_started = Instant::now();
    let progress = AtomicU64::new(0);
    run_download_io_with_elapsed(cx, url, timeout, &progress, work, || {
        Duration::from_nanos(cx.now().duration_since(started))
            .max(Duration::from_nanos(
                asupersync::time::wall_now().duration_since(ambient_started),
            ))
            .max(wall_started.elapsed())
    })
    .await
}

/// A backoff belongs to the caller's cancellation/deadline, not another HTTP
/// attempt. Keeping its timer inside the same guard also unregisters its wake
/// when the caller abandons the download.
async fn wait_download_retry(cx: &Cx, url: &str, delay: Duration) -> SearchResult<()> {
    run_download_request(cx, url, Duration::MAX, async {
        asupersync::time::sleep(asupersync::time::wall_now(), delay).await;
        Ok(())
    })
    .await
}

fn check_download_wait_cancel(cx: &Cx) -> SearchResult<()> {
    check_acquisition_cancel(cx, "download-wait")?;
    // Sleep also observes the ambient task. A caller may pass a separate Cx;
    // cancellation of either owner must end the wait without repolling a
    // timer that completed early because its ambient task was cancelled.
    if Cx::with_current(Cx::is_cancel_requested).unwrap_or(false) {
        return Err(SearchError::Cancelled {
            phase: "download-wait".to_owned(),
            reason: "model download task cancelled".to_owned(),
        });
    }
    Ok(())
}

/// One cancellation registration and timer cover a complete request or body.
/// Only nonempty body bytes advance `progress`; empty frames and trailers do
/// not keep a stalled transfer alive. The timer is rearmed only when its old
/// deadline fires, avoiding timer allocation/registration for every chunk.
/// `elapsed` uses runtime/wall time in production and controlled time in tests.
async fn run_download_io_with_elapsed<T: Send>(
    cx: &Cx,
    url: &str,
    timeout: Duration,
    progress: &AtomicU64,
    work: impl Future<Output = SearchResult<T>> + Send,
    elapsed: impl Fn() -> Duration + Send + Sync,
) -> SearchResult<T> {
    check_download_wait_cancel(cx)?;
    let explicit_remaining = cx
        .budget()
        .deadline
        .map(|deadline| Duration::from_nanos(deadline.duration_since(cx.now())));
    let ambient_remaining = Cx::with_current(|ambient| {
        ambient
            .budget()
            .deadline
            .map(|deadline| Duration::from_nanos(deadline.duration_since(ambient.now())))
    })
    .flatten();
    let caller_remaining = [explicit_remaining, ambient_remaining]
        .into_iter()
        .flatten()
        .min();
    let remaining = || {
        let elapsed = elapsed();
        let last_progress = Duration::from_nanos(progress.load(Ordering::Relaxed));
        let idle = timeout.saturating_sub(elapsed.saturating_sub(last_progress));
        caller_remaining.map_or(idle, |budget| idle.min(budget.saturating_sub(elapsed)))
    };
    let timed_out = || client_error_to_search(ClientError::DeadlineExceeded, url);
    let (sender, mut receiver) = oneshot::channel::<()>();
    let outcome = {
        let mut cancelled = std::pin::pin!(receiver.recv(cx));
        // wall_now follows the ambient driver used by an unbound Sleep. The
        // explicit Cx can have a different clock; its budget was converted to
        // a remaining duration above before meeting the two owners' limits.
        let mut timer = asupersync::time::sleep(asupersync::time::wall_now(), remaining());
        let mut work = std::pin::pin!(work);
        poll_fn(|task| {
            check_download_wait_cancel(cx)?;
            if cancelled.as_mut().poll(task).is_ready() {
                return Poll::Ready(Err(SearchError::Cancelled {
                    phase: "download-wait".to_owned(),
                    reason: "model download wait cancelled".to_owned(),
                }));
            }
            let left = remaining();
            if left.is_zero() {
                return Poll::Ready(Err(timed_out()));
            }
            if Pin::new(&mut timer).poll(task).is_ready() {
                // Actual progress may have extended the inactivity deadline
                // since the previous registration. A caller deadline never moves.
                timer.reset_after(asupersync::time::wall_now(), left);
                if Pin::new(&mut timer).poll(task).is_ready() {
                    check_download_wait_cancel(cx)?;
                    return Poll::Ready(Err(timed_out()));
                }
            }
            let outcome = work.as_mut().poll(task);
            check_download_wait_cancel(cx)?;
            if matches!(&outcome, Poll::Ready(Err(_))) {
                return outcome;
            }
            if remaining().is_zero() {
                return Poll::Ready(Err(timed_out()));
            }
            outcome
        })
        .await
    };
    drop(receiver);
    drop(sender);
    // Work (and therefore any owned response/socket) is dropped before return.
    check_download_wait_cancel(cx)?;
    outcome
}

const DOWNLOAD_BODY_BYTES_PER_POLL: usize = 64 * 1024;
const DOWNLOAD_BODY_STEPS_PER_POLL: usize = 32;

async fn stream_download_body<B>(
    cx: &Cx,
    url: &str,
    body: B,
    idle_timeout: Duration,
    on_chunk: impl FnMut(&[u8]) -> SearchResult<()> + Send,
) -> SearchResult<()>
where
    B: Body + Unpin + Send,
    B::Data: Send,
    B::Error: Send,
{
    let started = cx.now();
    let ambient_started = asupersync::time::wall_now();
    let wall_started = Instant::now();
    let elapsed = || {
        Duration::from_nanos(cx.now().duration_since(started))
            .max(Duration::from_nanos(
                asupersync::time::wall_now().duration_since(ambient_started),
            ))
            .max(wall_started.elapsed())
    };
    let progress = AtomicU64::new(0);
    let work = copy_download_body(cx, url, body, &progress, &elapsed, on_chunk);
    run_download_io_with_elapsed(cx, url, idle_timeout, &progress, work, &elapsed).await
}

/// Bound synchronous copy/hash/callback work per poll so even continuously
/// ready empty frames or a large fragmented buffer cannot starve cancellation
/// or the inactivity timer. Unconsumed bytes remain owned across yields.
async fn copy_download_body<B>(
    cx: &Cx,
    url: &str,
    mut body: B,
    progress: &AtomicU64,
    elapsed: &(impl Fn() -> Duration + Send + Sync),
    mut on_chunk: impl FnMut(&[u8]) -> SearchResult<()> + Send,
) -> SearchResult<()>
where
    B: Body + Unpin + Send,
    B::Data: Send,
    B::Error: Send,
{
    let mut pending: Option<B::Data> = None;
    poll_fn(|task| {
        let mut bytes_left = DOWNLOAD_BODY_BYTES_PER_POLL;
        for _ in 0..DOWNLOAD_BODY_STEPS_PER_POLL {
            check_acquisition_cancel(cx, "download-stream")?;
            if bytes_left == 0 {
                break;
            }
            if let Some(chunk) = pending.as_mut() {
                if !chunk.has_remaining() {
                    pending = None;
                    continue;
                }
                let bytes = chunk.chunk();
                if bytes.is_empty() || bytes.len() > chunk.remaining() {
                    return Poll::Ready(Err(SearchError::ModelLoadFailed {
                        path: PathBuf::from("<redacted-model-stage>"),
                        source: "HTTP body buffer violated its byte contract".into(),
                    }));
                }
                let count = bytes.len().min(bytes_left);
                on_chunk(&bytes[..count])?;
                chunk.advance(count);
                bytes_left -= count;
                progress.store(
                    u64::try_from(elapsed().as_nanos()).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
                continue;
            }
            match Pin::new(&mut body).poll_frame(task) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => return Poll::Ready(Ok(())),
                Poll::Ready(Some(Ok(Frame::Data(chunk)))) => pending = Some(chunk),
                Poll::Ready(Some(Ok(Frame::Trailers(_)))) => {}
                Poll::Ready(Some(Err(_error))) => {
                    return Poll::Ready(Err(SearchError::ModelLoadFailed {
                        path: PathBuf::from("<redacted-model-stage>"),
                        source: format!(
                            "stream read failed via host {} (http-body-failure)",
                            diagnostic_source_host(url)
                        )
                        .into(),
                    }));
                }
            }
        }
        task.waker().wake_by_ref();
        Poll::Pending
    })
    .await
}

struct AcquisitionLock {
    _file: File,
}

impl AcquisitionLock {
    fn acquire(destination_dir: &Path) -> SearchResult<Self> {
        let parent = destination_dir
            .parent()
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "model_acquisition.destination".to_owned(),
                value: "redacted".to_owned(),
                reason: "destination must have a parent directory".to_owned(),
            })?;
        std::fs::create_dir_all(parent).map_err(SearchError::from)?;
        let destination_name = destination_dir
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "model_acquisition.destination".to_owned(),
                value: "redacted".to_owned(),
                reason: "destination must end in a valid UTF-8 directory name".to_owned(),
            })?;
        let lock_path = parent.join(format!(".{destination_name}.acquisition.lock"));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)
            .map_err(SearchError::from)?;
        match file.try_lock() {
            Ok(()) => Ok(Self { _file: file }),
            Err(TryLockError::WouldBlock) => Err(SearchError::InvalidConfig {
                field: "model_acquisition.lock".to_owned(),
                value: "busy".to_owned(),
                reason: "another process is acquiring this model generation".to_owned(),
            }),
            Err(TryLockError::Error(error)) => Err(SearchError::from(error)),
        }
    }
}

fn validate_lifecycle_manifest(
    frozen: &FrozenModelArtifactManifestV1,
    lifecycle_manifest: &ModelManifest,
) -> SearchResult<()> {
    let expected = legacy_manifest_from_frozen(frozen)?;
    let same_files = expected.files.len() == lifecycle_manifest.files.len()
        && expected.files.iter().all(|expected_file| {
            lifecycle_manifest.files.iter().any(|actual_file| {
                actual_file.name == expected_file.name
                    && actual_file.size == expected_file.size
                    && actual_file.sha256 == expected_file.sha256
                    && lifecycle_manifest.download_url(actual_file)
                        == expected_file.url.as_deref().unwrap_or_default()
            })
        });
    if lifecycle_manifest.id != expected.id
        || lifecycle_manifest.repo != expected.repo
        || lifecycle_manifest.revision != expected.revision
        || lifecycle_manifest.license != expected.license
        || lifecycle_manifest.dimension != expected.dimension
        || !same_files
    {
        return Err(SearchError::InvalidConfig {
            field: "model_acquisition.lifecycle_manifest".to_owned(),
            value: lifecycle_manifest.id.clone(),
            reason: "lifecycle manifest must exactly bind the requested frozen artifact set"
                .to_owned(),
        });
    }
    Ok(())
}

fn legacy_manifest_from_frozen(
    frozen: &FrozenModelArtifactManifestV1,
) -> SearchResult<ModelManifest> {
    frozen.validate()?;
    let total_bytes = total_frozen_bytes(frozen)?;
    Ok(ModelManifest {
        id: frozen.manifest.logical_model_id.clone(),
        version: frozen.manifest.upstream_revision.clone(),
        display_name: None,
        description: None,
        repo: frozen.manifest.upstream_repository.clone(),
        revision: frozen.manifest.upstream_revision.clone(),
        files: frozen
            .manifest
            .artifacts
            .iter()
            .map(|artifact| ModelFile {
                name: artifact.relative_path.clone(),
                sha256: artifact.sha256.clone(),
                size: artifact.size,
                url: Some(artifact.upstream_url.clone()),
            })
            .collect(),
        license: frozen.manifest.license_spdx.clone(),
        dimension: Some(frozen.manifest.dimension),
        tier: None,
        download_size_bytes: total_bytes,
    })
}

fn total_frozen_bytes(frozen: &FrozenModelArtifactManifestV1) -> SearchResult<u64> {
    frozen
        .manifest
        .artifacts
        .iter()
        .try_fold(0_u64, |total, artifact| {
            total
                .checked_add(artifact.size)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "artifacts[].size".to_owned(),
                    value: "overflow".to_owned(),
                    reason: "total registered artifact bytes must fit in u64".to_owned(),
                })
        })
}

fn check_acquisition_cancel(cx: &Cx, phase: &str) -> SearchResult<()> {
    if cx.is_cancel_requested() {
        return Err(SearchError::Cancelled {
            phase: phase.to_owned(),
            reason: "structured cancellation requested".to_owned(),
        });
    }
    Ok(())
}

fn frozen_source_hosts(frozen: &FrozenModelArtifactManifestV1) -> SearchResult<Vec<String>> {
    let mut hosts = BTreeSet::new();
    for artifact in &frozen.manifest.artifacts {
        let authority = artifact
            .upstream_url
            .strip_prefix("https://")
            .and_then(|remainder| remainder.split('/').next())
            .filter(|authority| !authority.is_empty())
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "artifacts[].upstream_url".to_owned(),
                value: "redacted".to_owned(),
                reason: "expected a credential-free HTTPS authority".to_owned(),
            })?;
        let host = authority_host(authority)?;
        hosts.insert(host.to_owned());
    }
    Ok(hosts.into_iter().collect())
}

fn authority_host(authority: &str) -> SearchResult<&str> {
    let host = if let Some(rest) = authority.strip_prefix('[') {
        let closing = rest.find(']').ok_or_else(|| SearchError::InvalidConfig {
            field: "artifacts[].upstream_url".to_owned(),
            value: "redacted".to_owned(),
            reason: "malformed bracketed HTTPS host".to_owned(),
        })?;
        &rest[..closing]
    } else {
        authority.split(':').next().unwrap_or_default()
    };
    if host.is_empty()
        || !host.is_ascii()
        || host
            .bytes()
            .any(|byte| !(byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b':')))
    {
        return Err(SearchError::InvalidConfig {
            field: "artifacts[].upstream_url".to_owned(),
            value: "redacted".to_owned(),
            reason: "HTTPS source host contains invalid characters".to_owned(),
        });
    }
    Ok(host)
}

fn diagnostic_source_host(url: &str) -> &str {
    url.split_once("://")
        .and_then(|(_, remainder)| remainder.split('/').next())
        .and_then(|authority| authority_host(authority).ok())
        .unwrap_or("<invalid-host>")
}

fn acquisition_receipt(
    frozen: &FrozenModelArtifactManifestV1,
    source: ModelAcquisitionSourceKindV1,
    source_hosts: Vec<String>,
    outcome: ModelAcquisitionOutcomeV1,
    bytes_verified: u64,
    duration: Duration,
    prior_generation_preserved: bool,
) -> SearchResult<ModelAcquisitionReceiptV1> {
    let declared_bytes = total_frozen_bytes(frozen)?;
    if bytes_verified != declared_bytes {
        return Err(SearchError::InvalidConfig {
            field: "model_acquisition.bytes_verified".to_owned(),
            value: bytes_verified.to_string(),
            reason: "verified byte count must equal the frozen artifact total".to_owned(),
        });
    }
    let identity = frozen
        .manifest
        .declared_identity_bundle(QuantizationFormat::F32, IN_MEMORY_F32_STORAGE_FORMAT_V1)?;
    Ok(ModelAcquisitionReceiptV1 {
        schema_version: MODEL_ACQUISITION_RECEIPT_SCHEMA_V1,
        logical_model_id: frozen.manifest.logical_model_id.clone(),
        upstream_revision: frozen.manifest.upstream_revision.clone(),
        license_spdx: frozen.manifest.license_spdx.clone(),
        artifact_manifest_fingerprint: frozen.fingerprint.clone(),
        space_fingerprint: identity.space.fingerprint(),
        producer_fingerprint: identity.producer.fingerprint(),
        source,
        source_hosts,
        outcome,
        bytes_verified,
        duration_millis: saturating_duration_millis(duration),
        prior_generation_preserved,
        requires_reindex: true,
    })
}

fn acquisition_transport_progress(
    frozen: &FrozenModelArtifactManifestV1,
    source: ModelAcquisitionSourceKindV1,
    progress: &DownloadProgress,
    duration: Duration,
    cache_reason: Option<ModelAcquisitionCacheReasonV1>,
) -> SearchResult<ModelAcquisitionProgressV1> {
    let artifact = frozen
        .manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.relative_path == progress.file_name)
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "model_acquisition.progress_artifact".to_owned(),
            value: "unregistered".to_owned(),
            reason: "transport progress must reference a registered frozen artifact".to_owned(),
        })?;
    let source_host = if source == ModelAcquisitionSourceKindV1::Network {
        Some(
            artifact
                .upstream_url
                .strip_prefix("https://")
                .and_then(|remainder| remainder.split('/').next())
                .and_then(|authority| authority_host(authority).ok())
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "artifacts[].upstream_url".to_owned(),
                    value: "redacted".to_owned(),
                    reason: "network progress requires a valid credential-free HTTPS host"
                        .to_owned(),
                })?
                .to_owned(),
        )
    } else {
        None
    };
    Ok(ModelAcquisitionProgressV1 {
        schema_version: MODEL_ACQUISITION_PROGRESS_SCHEMA_V1,
        logical_model_id: frozen.manifest.logical_model_id.clone(),
        upstream_revision: frozen.manifest.upstream_revision.clone(),
        artifact_manifest_fingerprint: frozen.fingerprint.clone(),
        source,
        source_host,
        artifact_role: Some(artifact.role),
        artifact_sha256: Some(artifact.sha256.clone()),
        stage: ModelAcquisitionStageV1::Transporting,
        bytes_processed: progress.bytes_downloaded.min(artifact.size),
        total_bytes: artifact.size,
        files_completed: progress.files_completed.min(progress.files_total),
        files_total: frozen.manifest.artifacts.len(),
        duration_millis: saturating_duration_millis(duration),
        cache_reason,
        verification_result: ModelAcquisitionVerificationResultV1::Pending,
    })
}

fn acquisition_phase_progress(
    frozen: &FrozenModelArtifactManifestV1,
    source: ModelAcquisitionSourceKindV1,
    stage: ModelAcquisitionStageV1,
    duration: Duration,
    cache_reason: Option<ModelAcquisitionCacheReasonV1>,
    verification_result: ModelAcquisitionVerificationResultV1,
) -> SearchResult<ModelAcquisitionProgressV1> {
    let total_bytes = total_frozen_bytes(frozen)?;
    let complete = matches!(
        stage,
        ModelAcquisitionStageV1::StagedVerified
            | ModelAcquisitionStageV1::LoadSelfTestPassed
            | ModelAcquisitionStageV1::Published
            | ModelAcquisitionStageV1::WarmCacheVerified
    );
    Ok(ModelAcquisitionProgressV1 {
        schema_version: MODEL_ACQUISITION_PROGRESS_SCHEMA_V1,
        logical_model_id: frozen.manifest.logical_model_id.clone(),
        upstream_revision: frozen.manifest.upstream_revision.clone(),
        artifact_manifest_fingerprint: frozen.fingerprint.clone(),
        source,
        source_host: None,
        artifact_role: None,
        artifact_sha256: None,
        stage,
        bytes_processed: if complete { total_bytes } else { 0 },
        total_bytes,
        files_completed: if complete {
            frozen.manifest.artifacts.len()
        } else {
            0
        },
        files_total: frozen.manifest.artifacts.len(),
        duration_millis: saturating_duration_millis(duration),
        cache_reason,
        verification_result,
    })
}

fn saturating_duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
thread_local! {
    static TEST_LOCAL_COPY_ERROR: std::cell::Cell<Option<ErrorKind>> =
        const { std::cell::Cell::new(None) };
}

#[allow(clippy::unnecessary_wraps)] // Test builds inject typed filesystem failures here.
fn local_copy_write_boundary() -> SearchResult<()> {
    #[cfg(test)]
    if let Some(kind) = TEST_LOCAL_COPY_ERROR.with(std::cell::Cell::get) {
        return Err(SearchError::from(std::io::Error::from(kind)));
    }
    Ok(())
}

#[cfg(test)]
struct LocalCopyErrorGuard;

#[cfg(test)]
impl LocalCopyErrorGuard {
    fn install(kind: ErrorKind) -> Self {
        TEST_LOCAL_COPY_ERROR.with(|fault| {
            assert!(
                fault.replace(Some(kind)).is_none(),
                "local-copy test error already installed"
            );
        });
        Self
    }
}

#[cfg(test)]
impl Drop for LocalCopyErrorGuard {
    fn drop(&mut self) {
        TEST_LOCAL_COPY_ERROR.with(|fault| fault.set(None));
    }
}

fn copy_local_artifact(
    cx: &Cx,
    source_dir: &Path,
    staging_dir: &Path,
    artifact: &ModelArtifactFileV1,
    file_idx: usize,
    files_total: usize,
    on_progress: &(impl Fn(&DownloadProgress) + Send + Sync),
) -> SearchResult<()> {
    let source_path = source_dir.join(&artifact.relative_path);
    let metadata = std::fs::symlink_metadata(&source_path).map_err(SearchError::from)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(SearchError::InvalidConfig {
            field: "model_acquisition.local_artifact".to_owned(),
            value: "redacted".to_owned(),
            reason: "registered local artifacts must be regular files, never symlinks".to_owned(),
        });
    }

    let destination = staging_dir.join(&artifact.relative_path);
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).map_err(SearchError::from)?;
    }
    let source = File::open(source_path).map_err(SearchError::from)?;
    let mut reader = BufReader::with_capacity(LOCAL_COPY_BUFFER_BYTES, source);
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&destination)
        .map_err(SearchError::from)?;
    let mut buffer = vec![0_u8; LOCAL_COPY_BUFFER_BYTES];
    let mut hasher = Sha256::new();
    let mut bytes_copied = 0_u64;
    let started = Instant::now();
    on_progress(&DownloadProgress {
        file_name: artifact.relative_path.clone(),
        bytes_downloaded: 0,
        total_bytes: Some(artifact.size),
        files_completed: file_idx,
        files_total,
        speed_bytes_per_sec: 0.0,
        eta_seconds: None,
    });

    loop {
        check_acquisition_cancel(cx, "local-artifact-read")?;
        let count = reader.read(&mut buffer).map_err(SearchError::from)?;
        if count == 0 {
            break;
        }
        let count_u64 = u64::try_from(count).unwrap_or(u64::MAX);
        let next_size =
            bytes_copied
                .checked_add(count_u64)
                .ok_or_else(|| SearchError::HashMismatch {
                    path: PathBuf::from("<redacted-local-artifact>"),
                    expected: format!("size={}", artifact.size),
                    actual: "size-overflow".to_owned(),
                })?;
        if next_size > artifact.size {
            return Err(SearchError::HashMismatch {
                path: PathBuf::from("<redacted-local-artifact>"),
                expected: format!("size={}", artifact.size),
                actual: format!("size-at-least={next_size}"),
            });
        }
        local_copy_write_boundary()?;
        output
            .write_all(&buffer[..count])
            .map_err(SearchError::from)?;
        hasher.update(&buffer[..count]);
        bytes_copied = next_size;
        report_local_progress(
            artifact,
            file_idx,
            files_total,
            bytes_copied,
            started.elapsed(),
            on_progress,
        );
    }

    if bytes_copied != artifact.size {
        return Err(SearchError::HashMismatch {
            path: PathBuf::from("<redacted-local-artifact>"),
            expected: format!("size={}", artifact.size),
            actual: format!("size={bytes_copied}"),
        });
    }
    let actual_hash = sha256_digest_hex(hasher.finalize().as_slice());
    // ubs:ignore — pinned artifact digests are public integrity metadata.
    if actual_hash != artifact.sha256 {
        return Err(SearchError::HashMismatch {
            path: PathBuf::from("<redacted-local-artifact>"),
            expected: format!("sha256={},size={}", artifact.sha256, artifact.size),
            actual: format!("sha256={actual_hash},size={bytes_copied}"),
        });
    }
    check_acquisition_cancel(cx, "local-artifact-sync")?;
    output.flush().map_err(SearchError::from)?;
    output.sync_all().map_err(SearchError::from)?;
    report_local_progress(
        artifact,
        file_idx,
        files_total,
        bytes_copied,
        started.elapsed(),
        on_progress,
    );
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn report_local_progress(
    artifact: &ModelArtifactFileV1,
    file_idx: usize,
    files_total: usize,
    bytes_copied: u64,
    elapsed: Duration,
    on_progress: &(impl Fn(&DownloadProgress) + Send + Sync),
) {
    let speed = if elapsed.as_secs_f64() > 0.0 {
        bytes_copied as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    let eta_seconds = if speed <= f64::EPSILON || bytes_copied >= artifact.size {
        None
    } else {
        Some((artifact.size.saturating_sub(bytes_copied)) as f64 / speed)
    };
    on_progress(&DownloadProgress {
        file_name: artifact.relative_path.clone(),
        bytes_downloaded: bytes_copied,
        total_bytes: Some(artifact.size),
        files_completed: file_idx,
        files_total,
        speed_bytes_per_sec: speed,
        eta_seconds,
    });
}

/// Build a `HuggingFace` CDN URL for a model file.
#[cfg(test)]
fn huggingface_url(repo: &str, revision: &str, file_name: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/{revision}/{file_name}")
}

fn create_unique_staging_dir(dest_dir: &Path) -> SearchResult<PathBuf> {
    // Prefer creating staging dir as a sibling to avoid dirtying the target dir.
    let (base_dir, prefix) = dest_dir.parent().map_or((dest_dir, "download"), |parent| {
        (
            parent,
            dest_dir
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("model"),
        )
    });

    std::fs::create_dir_all(base_dir).map_err(SearchError::from)?;

    let pid = std::process::id();
    for _ in 0..64 {
        let counter = STAGING_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let candidate = base_dir.join(format!(".{prefix}-download-{pid}-{counter:016x}"));
        match std::fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(err) if err.kind() == ErrorKind::AlreadyExists => {}
            Err(err) => return Err(SearchError::from(err)),
        }
    }

    Err(SearchError::ModelLoadFailed {
        path: PathBuf::from("<redacted-model-cache>"),
        source: "failed to allocate unique staging directory".into(),
    })
}

/// Resolve an HTTP redirect without carrying an origin's query or credentials
/// into a different origin. The transport parser validates the resulting
/// authority; relative references need path/query resolution before parsing.
fn resolve_download_redirect(current: &str, location: &str) -> Result<String, ClientError> {
    let invalid = || ClientError::InvalidUrl("invalid model download redirect".to_owned());
    let current = ParsedUrl::parse(current)?;
    let location = location.trim();
    if location
        .chars()
        .any(|ch| ch.is_control() || ch.is_whitespace())
    {
        return Err(invalid());
    }
    // Fragments never belong in an HTTP request target.
    let reference = location.split('#').next().unwrap_or_default();
    let scheme = match current.scheme {
        Scheme::Http => "http",
        Scheme::Https => "https",
    };
    let origin = format!("{scheme}://{}", current.authority());
    let current_path = current.path.split('#').next().unwrap_or("/");
    let path_only = current_path.split('?').next().unwrap_or("/");
    let path_only = if path_only.is_empty() { "/" } else { path_only };
    let next = if reference.starts_with("//") {
        format!("{scheme}:{reference}")
    } else if reference
        .split(['/', '?'])
        .next()
        .is_some_and(|first| first.contains(':'))
    {
        let (next_scheme, rest) = reference.split_once("://").ok_or_else(invalid)?;
        if !next_scheme.eq_ignore_ascii_case("http") && !next_scheme.eq_ignore_ascii_case("https") {
            return Err(invalid());
        }
        format!("{}://{rest}", next_scheme.to_ascii_lowercase())
    } else if reference.is_empty() {
        format!("{origin}{current_path}")
    } else if reference.starts_with('?') {
        format!("{origin}{path_only}{reference}")
    } else if reference.starts_with('/') {
        format!("{origin}{reference}")
    } else {
        let directory = path_only.rfind('/').map_or("/", |idx| &path_only[..=idx]);
        format!("{origin}{directory}{reference}")
    };
    let next = ParsedUrl::parse(&next)?;
    if current.scheme == Scheme::Https && next.scheme == Scheme::Http {
        return Err(ClientError::InvalidUrl(
            "model download redirect would downgrade HTTPS".to_owned(),
        ));
    }
    let next_scheme = match next.scheme {
        Scheme::Http => "http",
        Scheme::Https => "https",
    };
    let (path, query) = next
        .path
        .split_once('?')
        .map_or((next.path.as_str(), None), |(path, query)| {
            (path, Some(query))
        });
    // Remove literal dot segments (RFC 3986 section 5.2.4), preserving empty
    // segments and escaped bytes so signed CDN paths are not rewritten.
    let trailing_dot = path.ends_with("/.") || path.ends_with("/..");
    let mut segments = Vec::new();
    for segment in path.split('/') {
        match segment {
            "." => {}
            ".." => {
                if segments.len() > 1 {
                    segments.pop();
                }
            }
            other => segments.push(other),
        }
    }
    if trailing_dot {
        segments.push("");
    }
    let mut path = segments.join("/");
    if !path.starts_with('/') {
        path.insert(0, '/');
    }
    if let Some(query) = query {
        path.push('?');
        path.push_str(query);
    }
    Ok(format!("{next_scheme}://{}{path}", next.authority()))
}

fn response_content_length(headers: &[(String, String)]) -> Option<u64> {
    headers.iter().find_map(|(name, value)| {
        name.eq_ignore_ascii_case("content-length")
            .then(|| value.trim().parse::<u64>().ok())
            .flatten()
    })
}

fn sha256_digest_hex(data: &[u8]) -> String {
    let mut out = String::with_capacity(64);
    for byte in data {
        use std::fmt::Write;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn bounded_download_failure_reason(error: &SearchError) -> &'static str {
    match error {
        SearchError::HashMismatch { .. } => "artifact-integrity-mismatch",
        SearchError::ModelNotFound { .. } => "registered-artifact-missing",
        SearchError::ModelLoadFailed { .. } | SearchError::Io(_) => {
            "transport-or-filesystem-failure"
        }
        SearchError::Cancelled { .. } => "operation-cancelled",
        SearchError::InvalidConfig { .. } => "invalid-download-contract",
        _ => "model-download-failed",
    }
}

/// Compute lowercase hex SHA-256 of a byte slice.
#[cfg(test)]
fn sha256_hex(data: &[u8]) -> String {
    sha256_digest_hex(Sha256::digest(data).as_slice())
}

struct TempFileGuard {
    path: PathBuf,
    armed: bool,
}

impl TempFileGuard {
    const fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    const fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if self.armed
            && let Err(e) = std::fs::remove_file(&self.path)
        {
            tracing::warn!(
                error_kind = ?e.kind(),
                "failed to clean up model download temp file"
            );
        }
    }
}

/// Format bytes as a human-readable string.
#[allow(clippy::cast_precision_loss)]
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Convert asupersync `ClientError` to `SearchError`.
fn client_error_to_search(error: ClientError, url: &str) -> SearchError {
    if matches!(error, ClientError::Cancelled) {
        return SearchError::Cancelled {
            phase: "model-download-request".to_owned(),
            reason: "HTTP client observed structured cancellation".to_owned(),
        };
    }
    let detail = match error {
        ClientError::InvalidUrl(_) => "invalid-url".to_owned(),
        ClientError::DnsError(error)
        | ClientError::ConnectError(error)
        | ClientError::Io(error) => format!("io-{:?}", error.kind()),
        ClientError::TlsError(_) => "tls-handshake-failure".to_owned(),
        ClientError::HttpError(_) => "http-protocol-failure".to_owned(),
        ClientError::TooManyRedirects { count, max } => {
            format!("redirect-limit-{count}-of-{max}")
        }
        ClientError::DeadlineExceeded => "request-deadline-exceeded".to_owned(),
        ClientError::ConnectTunnelRefused { status, .. } => {
            format!("proxy-tunnel-http-{status}")
        }
        ClientError::InvalidConnectInput(_) => "invalid-proxy-connect-input".to_owned(),
        ClientError::ProxyError(_) => "proxy-failure".to_owned(),
        ClientError::PoolExhausted { port, .. } => {
            format!("connection-pool-exhausted-port-{port}")
        }
        ClientError::Cancelled => unreachable!("handled above"),
    };
    SearchError::ModelLoadFailed {
        path: PathBuf::from("<redacted-model-source>"),
        source: format!(
            "transport failure via host {} ({detail})",
            diagnostic_source_host(url)
        )
        .into(),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_manifest::{
        ConsentSource, DownloadConsent, ModelArtifactManifestV1, ModelArtifactRoleV1, ModelState,
        PLACEHOLDER_PINNED_REVISION, PLACEHOLDER_VERIFY_AFTER_DOWNLOAD,
    };
    use std::collections::VecDeque;
    use std::io::{Read, Write};
    use std::net::{Shutdown, TcpListener, TcpStream};
    use std::process::{Command, Stdio};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;

    use asupersync::test_utils::run_test_with_cx;

    const LOCK_CHILD_DESTINATION_ENV: &str =
        "FRANKENSEARCH_TEST_ACQUISITION_LOCK_CHILD_DESTINATION";
    const LOCK_CHILD_READY_ENV: &str = "FRANKENSEARCH_TEST_ACQUISITION_LOCK_CHILD_READY";
    const LOCK_CHILD_RELEASE_ENV: &str = "FRANKENSEARCH_TEST_ACQUISITION_LOCK_CHILD_RELEASE";

    #[derive(Debug, Clone)]
    struct TestHttpResponse {
        status: u16,
        reason: &'static str,
        body: Vec<u8>,
    }

    fn spawn_test_http_server(
        responses: Vec<TestHttpResponse>,
    ) -> (String, Arc<AtomicUsize>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let queue = Arc::new(Mutex::new(VecDeque::from(responses)));
        let served = Arc::new(AtomicUsize::new(0));
        let served_for_thread = Arc::clone(&served);
        let queue_for_thread = Arc::clone(&queue);

        let handle = thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                if read_http_headers(&mut stream).is_err() {
                    break;
                }

                let response = {
                    let mut guard = queue_for_thread.lock().unwrap();
                    guard.pop_front()
                };
                let Some(response) = response else {
                    break;
                };

                served_for_thread.fetch_add(1, Ordering::SeqCst);
                if write_http_response(&mut stream, &response).is_err() {
                    break;
                }
                let _ = stream.shutdown(Shutdown::Both);
                if queue_for_thread.lock().unwrap().is_empty() {
                    break;
                }
            }
        });

        (format!("http://{addr}"), served, handle)
    }

    fn read_http_headers(stream: &mut TcpStream) -> std::io::Result<()> {
        let mut buf = [0_u8; 1024];
        let mut request = Vec::new();
        loop {
            let read = stream.read(&mut buf)?;
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buf[..read]);
            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
            if request.len() > 64 * 1024 {
                break;
            }
        }
        Ok(())
    }

    fn write_http_response(
        stream: &mut TcpStream,
        response: &TestHttpResponse,
    ) -> std::io::Result<()> {
        write!(
            stream,
            "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            response.status,
            response.reason,
            response.body.len()
        )?;
        stream.write_all(&response.body)?;
        stream.flush()?;
        Ok(())
    }

    fn acquisition_fixture() -> FrozenModelArtifactManifestV1 {
        let mut manifest = ModelArtifactManifestV1::potion_128m_native().unwrap();
        manifest.logical_model_id = "acquisition-fixture".to_owned();
        manifest.artifacts.retain(|artifact| {
            matches!(
                artifact.role,
                ModelArtifactRoleV1::Weights | ModelArtifactRoleV1::Tokenizer
            )
        });
        for artifact in &mut manifest.artifacts {
            let (path, bytes) = match artifact.role {
                ModelArtifactRoleV1::Weights => ("model.safetensors", fixture_weights()),
                ModelArtifactRoleV1::Tokenizer => ("tokenizer.json", fixture_tokenizer()),
                _ => unreachable!("fixture retains only required roles"),
            };
            artifact.relative_path = path.to_owned();
            artifact.upstream_url = format!("https://models.example.invalid/{path}");
            artifact.size = u64::try_from(bytes.len()).unwrap();
            artifact.sha256 = sha256_hex(bytes);
        }
        manifest.freeze().unwrap()
    }

    const fn fixture_weights() -> &'static [u8] {
        b"fixture-weights-v1"
    }

    const fn fixture_tokenizer() -> &'static [u8] {
        br#"{"fixture":"tokenizer-v1"}"#
    }

    fn fixture_bytes(role: ModelArtifactRoleV1) -> &'static [u8] {
        match role {
            ModelArtifactRoleV1::Weights => fixture_weights(),
            ModelArtifactRoleV1::Tokenizer => fixture_tokenizer(),
            _ => unreachable!("fixture retains only required roles"),
        }
    }

    fn write_acquisition_fixture(source_dir: &Path, frozen: &FrozenModelArtifactManifestV1) {
        for artifact in &frozen.manifest.artifacts {
            let path = source_dir.join(&artifact.relative_path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, fixture_bytes(artifact.role)).unwrap();
        }
    }

    fn fixture_lifecycle(frozen: &FrozenModelArtifactManifestV1) -> ModelLifecycle {
        ModelLifecycle::new(
            legacy_manifest_from_frozen(frozen).unwrap(),
            DownloadConsent::granted(ConsentSource::Programmatic),
        )
    }

    fn verify_fixture_load(
        path: &Path,
        frozen: &FrozenModelArtifactManifestV1,
    ) -> SearchResult<()> {
        frozen.manifest.verify_dir(path).map(|_| ())
    }

    #[test]
    fn huggingface_url_format() {
        let url = huggingface_url(
            "sentence-transformers/all-MiniLM-L6-v2",
            "abc123",
            "onnx/model.onnx",
        );
        assert_eq!(
            url,
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/abc123/onnx/model.onnx"
        );
    }

    #[test]
    fn sha256_hex_known_value() {
        let hash = sha256_hex(b"hello world");
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn format_bytes_units() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1_048_576), "1.0 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.0 GB");
    }

    #[test]
    fn download_config_defaults() {
        let config = DownloadConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_base_delay, Duration::from_secs(1));
        assert_eq!(config.max_redirects, 5);
        assert_eq!(config.max_response_bytes, DEFAULT_MAX_MODEL_ARTIFACT_BYTES);
        assert!(config.max_response_bytes > 16 * 1024 * 1024);
        assert!(config.user_agent.starts_with("frankensearch/"));
    }

    #[test]
    fn invalid_download_timeouts_fail_before_network_or_staging() {
        for (field, config) in [
            (
                "download.request_timeout",
                DownloadConfig {
                    request_timeout: Duration::ZERO,
                    ..DownloadConfig::default()
                },
            ),
            (
                "download.idle_timeout",
                DownloadConfig {
                    idle_timeout: Duration::ZERO,
                    ..DownloadConfig::default()
                },
            ),
            (
                "download.max_response_bytes",
                DownloadConfig {
                    max_response_bytes: 0,
                    ..DownloadConfig::default()
                },
            ),
        ] {
            let error = validate_download_config(&config).unwrap_err();
            assert!(matches!(
                error,
                SearchError::InvalidConfig { ref field: actual, .. } if actual == field
            ));
        }
    }

    #[test]
    fn frozen_local_acquisition_publishes_then_reuses_warm_cache_without_network() {
        let temp = tempfile::tempdir().unwrap();
        let source_dir = temp.path().join("private-source");
        let destination_dir = temp.path().join("cache").join("model");
        let frozen = acquisition_fixture();
        write_acquisition_fixture(&source_dir, &frozen);
        let downloader = ModelDownloader::with_defaults();

        run_test_with_cx(|cx| async move {
            let mut lifecycle = fixture_lifecycle(&frozen);
            let local_events = Arc::new(Mutex::new(Vec::new()));
            let local_events_for_callback = Arc::clone(&local_events);
            let receipt = downloader
                .acquire_frozen_model(
                    &cx,
                    ModelAcquisitionRequest {
                        frozen_manifest: &frozen,
                        source: ModelAcquisitionSource::LocalBundle(&source_dir),
                        destination_dir: &destination_dir,
                    },
                    &mut lifecycle,
                    move |progress| {
                        local_events_for_callback
                            .lock()
                            .unwrap()
                            .push(progress.clone());
                    },
                    verify_fixture_load,
                )
                .await
                .unwrap();

            assert_eq!(receipt.source, ModelAcquisitionSourceKindV1::LocalBundle);
            assert_eq!(receipt.outcome, ModelAcquisitionOutcomeV1::Published);
            assert!(receipt.requires_reindex);
            assert!(!receipt.prior_generation_preserved);
            assert!(matches!(
                lifecycle.state(),
                ModelState::AcquiredNeedsReindex
            ));
            frozen.manifest.verify_dir(&destination_dir).unwrap();

            let serialized = serde_json::to_string(&receipt).unwrap();
            assert!(!serialized.contains(&source_dir.to_string_lossy().to_string()));
            assert!(!serialized.contains(&destination_dir.to_string_lossy().to_string()));
            assert!(!serialized.contains("fixture-weights-v1"));
            assert!(serialized.contains(&frozen.fingerprint));
            let events = local_events.lock().unwrap().clone();
            assert!(events.iter().any(|event| {
                event.stage == ModelAcquisitionStageV1::Published
                    && event.verification_result == ModelAcquisitionVerificationResultV1::Passed
            }));
            assert!(
                events
                    .iter()
                    .filter(|event| event.stage == ModelAcquisitionStageV1::Transporting)
                    .all(|event| {
                        event.source_host.is_none()
                            && event.artifact_role.is_some()
                            && event.artifact_sha256.is_some()
                            && event.bytes_processed <= event.total_bytes
                    })
            );
            let serialized_events = serde_json::to_string(&events).unwrap();
            assert!(!serialized_events.contains(&source_dir.to_string_lossy().to_string()));
            assert!(!serialized_events.contains("fixture-weights-v1"));

            let mut warm_lifecycle = fixture_lifecycle(&frozen);
            let warm_events = Arc::new(Mutex::new(Vec::new()));
            let warm_events_for_callback = Arc::clone(&warm_events);
            let warm_receipt = downloader
                .acquire_frozen_model(
                    &cx,
                    ModelAcquisitionRequest {
                        frozen_manifest: &frozen,
                        source: ModelAcquisitionSource::Network,
                        destination_dir: &destination_dir,
                    },
                    &mut warm_lifecycle,
                    move |progress| {
                        warm_events_for_callback
                            .lock()
                            .unwrap()
                            .push(progress.clone());
                    },
                    verify_fixture_load,
                )
                .await
                .unwrap();
            assert_eq!(warm_receipt.source, ModelAcquisitionSourceKindV1::WarmCache);
            assert_eq!(
                warm_receipt.outcome,
                ModelAcquisitionOutcomeV1::VerifiedWarmCache
            );
            assert!(warm_receipt.source_hosts.is_empty());
            let events = warm_events.lock().unwrap().clone();
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].stage, ModelAcquisitionStageV1::WarmCacheVerified);
            assert_eq!(
                events[0].cache_reason,
                Some(ModelAcquisitionCacheReasonV1::FrozenGenerationVerified)
            );
            assert!(events[0].source_host.is_none());
            assert!(events[0].artifact_sha256.is_none());
            let serialized_progress = serde_json::to_string(&events[0]).unwrap();
            assert!(!serialized_progress.contains(&destination_dir.to_string_lossy().to_string()));
            assert!(matches!(
                warm_lifecycle.state(),
                ModelState::AcquiredNeedsReindex
            ));
        });
    }

    #[test]
    fn corrupt_warm_cache_is_rejected_and_preserved_during_replacement() {
        let temp = tempfile::tempdir().unwrap();
        let source_dir = temp.path().join("source");
        let destination_dir = temp.path().join("model");
        let frozen = acquisition_fixture();
        write_acquisition_fixture(&source_dir, &frozen);
        let downloader = ModelDownloader::with_defaults();

        run_test_with_cx(|cx| async move {
            let mut initial_lifecycle = fixture_lifecycle(&frozen);
            downloader
                .acquire_frozen_model(
                    &cx,
                    ModelAcquisitionRequest {
                        frozen_manifest: &frozen,
                        source: ModelAcquisitionSource::LocalBundle(&source_dir),
                        destination_dir: &destination_dir,
                    },
                    &mut initial_lifecycle,
                    |_| {},
                    verify_fixture_load,
                )
                .await
                .unwrap();
            std::fs::write(
                destination_dir.join("model.safetensors"),
                b"corrupt-cached-weights",
            )
            .unwrap();

            let events = Arc::new(Mutex::new(Vec::new()));
            let events_for_callback = Arc::clone(&events);
            let mut replacement_lifecycle = fixture_lifecycle(&frozen);
            let receipt = downloader
                .acquire_frozen_model(
                    &cx,
                    ModelAcquisitionRequest {
                        frozen_manifest: &frozen,
                        source: ModelAcquisitionSource::LocalBundle(&source_dir),
                        destination_dir: &destination_dir,
                    },
                    &mut replacement_lifecycle,
                    move |progress| {
                        events_for_callback.lock().unwrap().push(progress.clone());
                    },
                    verify_fixture_load,
                )
                .await
                .unwrap();

            assert_eq!(receipt.outcome, ModelAcquisitionOutcomeV1::Published);
            assert!(receipt.prior_generation_preserved);
            frozen.manifest.verify_dir(&destination_dir).unwrap();
            assert!(events.lock().unwrap().iter().all(|event| {
                event.cache_reason == Some(ModelAcquisitionCacheReasonV1::DestinationRejected)
            }));
            let backup = std::fs::read_dir(temp.path())
                .unwrap()
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .find(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with("model.backup."))
                })
                .expect("corrupt cache retained as prior generation");
            assert_eq!(
                std::fs::read(backup.join("model.safetensors")).unwrap(),
                b"corrupt-cached-weights"
            );
        });
    }

    #[test]
    fn network_progress_exposes_only_registered_host_and_integrity_metadata() {
        let frozen = acquisition_fixture();
        let artifact = &frozen.manifest.artifacts[0];
        let raw = DownloadProgress {
            file_name: artifact.relative_path.clone(),
            bytes_downloaded: artifact.size.saturating_add(100),
            total_bytes: Some(artifact.size),
            files_completed: 0,
            files_total: frozen.manifest.artifacts.len(),
            speed_bytes_per_sec: 42.0,
            eta_seconds: None,
        };
        let progress = acquisition_transport_progress(
            &frozen,
            ModelAcquisitionSourceKindV1::Network,
            &raw,
            Duration::from_millis(7),
            Some(ModelAcquisitionCacheReasonV1::DestinationMissing),
        )
        .unwrap();
        assert_eq!(
            progress.source_host.as_deref(),
            Some("models.example.invalid")
        );
        assert_eq!(progress.bytes_processed, artifact.size);
        assert_eq!(progress.artifact_sha256.as_deref(), Some(&*artifact.sha256));
        let serialized = serde_json::to_string(&progress).unwrap();
        assert!(!serialized.contains(&artifact.upstream_url));
        assert!(!serialized.contains(&artifact.relative_path));
    }

    #[test]
    fn local_acquisition_rejects_truncation_and_overrun_before_publication() {
        for replacement in [
            b"short".as_slice(),
            b"fixture-weights-v1-with-unregistered-tail".as_slice(),
        ] {
            let temp = tempfile::tempdir().unwrap();
            let source_dir = temp.path().join("source");
            let destination_dir = temp.path().join("model");
            let frozen = acquisition_fixture();
            write_acquisition_fixture(&source_dir, &frozen);
            std::fs::write(source_dir.join("model.safetensors"), replacement).unwrap();
            let downloader = ModelDownloader::with_defaults();

            run_test_with_cx(|cx| async move {
                let mut lifecycle = fixture_lifecycle(&frozen);
                let error = downloader
                    .acquire_frozen_model(
                        &cx,
                        ModelAcquisitionRequest {
                            frozen_manifest: &frozen,
                            source: ModelAcquisitionSource::LocalBundle(&source_dir),
                            destination_dir: &destination_dir,
                        },
                        &mut lifecycle,
                        |_| {},
                        verify_fixture_load,
                    )
                    .await
                    .unwrap_err();
                assert!(matches!(error, SearchError::HashMismatch { .. }));
                assert!(!destination_dir.exists());
                assert!(matches!(
                    lifecycle.state(),
                    ModelState::VerificationFailed { .. }
                ));
            });
        }
    }

    #[test]
    fn failed_local_acquisition_preserves_prior_generation_and_retains_stage() {
        let temp = tempfile::tempdir().unwrap();
        let source_dir = temp.path().join("source");
        let destination_dir = temp.path().join("model");
        let frozen = acquisition_fixture();
        write_acquisition_fixture(&source_dir, &frozen);
        std::fs::write(source_dir.join("model.safetensors"), b"fixture-weights-v2").unwrap();
        std::fs::create_dir_all(&destination_dir).unwrap();
        std::fs::write(destination_dir.join("prior-generation"), b"prior").unwrap();
        let downloader = ModelDownloader::with_defaults();

        run_test_with_cx(|cx| async move {
            let mut lifecycle = fixture_lifecycle(&frozen);
            let error = downloader
                .acquire_frozen_model(
                    &cx,
                    ModelAcquisitionRequest {
                        frozen_manifest: &frozen,
                        source: ModelAcquisitionSource::LocalBundle(&source_dir),
                        destination_dir: &destination_dir,
                    },
                    &mut lifecycle,
                    |_| {},
                    verify_fixture_load,
                )
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::HashMismatch { .. }));
            assert_eq!(
                std::fs::read(destination_dir.join("prior-generation")).unwrap(),
                b"prior"
            );
            let recovery = diagnose_model_acquisition(&destination_dir).unwrap();
            assert_eq!(recovery.orphan_staging_generations, 1);
            assert!(recovery.destination_present);
            assert!(recovery.advisory_lock_present);
            let serialized = serde_json::to_string(&recovery).unwrap();
            assert!(!serialized.contains(&destination_dir.to_string_lossy().to_string()));
        });
    }

    #[test]
    fn recovery_diagnostics_inventory_every_retained_generation_kind() {
        let temp = tempfile::tempdir().unwrap();
        let destination_dir = temp.path().join("model");
        std::fs::create_dir_all(&destination_dir).unwrap();
        std::fs::create_dir(temp.path().join(".model-download-orphan")).unwrap();
        std::fs::create_dir(temp.path().join(".model.installing.interrupted")).unwrap();
        std::fs::create_dir(temp.path().join("model.backup.prior")).unwrap();
        std::fs::write(temp.path().join(".model.acquisition.lock"), b"").unwrap();

        let recovery = diagnose_model_acquisition(&destination_dir).unwrap();

        assert!(recovery.destination_present);
        assert_eq!(recovery.orphan_staging_generations, 1);
        assert_eq!(recovery.interrupted_installing_generations, 1);
        assert_eq!(recovery.preserved_backup_generations, 1);
        assert!(recovery.advisory_lock_present);
        let serialized = serde_json::to_string(&recovery).unwrap();
        assert!(!serialized.contains(&temp.path().to_string_lossy().to_string()));
    }

    #[test]
    fn permission_and_storage_exhaustion_preserve_prior_generation() {
        for injected_kind in [ErrorKind::PermissionDenied, ErrorKind::StorageFull] {
            let temp = tempfile::tempdir().unwrap();
            let source_dir = temp.path().join("source");
            let destination_dir = temp.path().join("model");
            let frozen = acquisition_fixture();
            write_acquisition_fixture(&source_dir, &frozen);
            std::fs::create_dir_all(&destination_dir).unwrap();
            std::fs::write(destination_dir.join("prior-generation"), b"prior").unwrap();
            let downloader = ModelDownloader::with_defaults();

            run_test_with_cx(|cx| async move {
                let _fault = LocalCopyErrorGuard::install(injected_kind);
                let mut lifecycle = fixture_lifecycle(&frozen);
                let error = downloader
                    .acquire_frozen_model(
                        &cx,
                        ModelAcquisitionRequest {
                            frozen_manifest: &frozen,
                            source: ModelAcquisitionSource::LocalBundle(&source_dir),
                            destination_dir: &destination_dir,
                        },
                        &mut lifecycle,
                        |_| {},
                        verify_fixture_load,
                    )
                    .await
                    .unwrap_err();

                assert!(matches!(
                    error,
                    SearchError::Io(ref io_error) if io_error.kind() == injected_kind
                ));
                assert_eq!(
                    std::fs::read(destination_dir.join("prior-generation")).unwrap(),
                    b"prior"
                );
                assert!(matches!(
                    lifecycle.state(),
                    ModelState::VerificationFailed { .. }
                ));
            });
        }
    }

    #[test]
    fn successful_replacement_preserves_prior_generation_as_backup() {
        let temp = tempfile::tempdir().unwrap();
        let source_dir = temp.path().join("source");
        let destination_dir = temp.path().join("model");
        let frozen = acquisition_fixture();
        write_acquisition_fixture(&source_dir, &frozen);
        std::fs::create_dir_all(&destination_dir).unwrap();
        std::fs::write(destination_dir.join("prior-generation"), b"prior").unwrap();
        let downloader = ModelDownloader::with_defaults();

        run_test_with_cx(|cx| async move {
            let mut lifecycle = fixture_lifecycle(&frozen);
            let receipt = downloader
                .acquire_frozen_model(
                    &cx,
                    ModelAcquisitionRequest {
                        frozen_manifest: &frozen,
                        source: ModelAcquisitionSource::LocalBundle(&source_dir),
                        destination_dir: &destination_dir,
                    },
                    &mut lifecycle,
                    |_| {},
                    verify_fixture_load,
                )
                .await
                .unwrap();
            assert!(receipt.prior_generation_preserved);
            frozen.manifest.verify_dir(&destination_dir).unwrap();
            let recovery = diagnose_model_acquisition(&destination_dir).unwrap();
            assert_eq!(recovery.preserved_backup_generations, 1);

            let backup = std::fs::read_dir(temp.path())
                .unwrap()
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .find(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with("model.backup."))
                })
                .expect("prior generation backup");
            assert_eq!(
                std::fs::read(backup.join("prior-generation")).unwrap(),
                b"prior"
            );
        });
    }

    #[test]
    fn load_self_test_failure_never_replaces_prior_generation() {
        let temp = tempfile::tempdir().unwrap();
        let source_dir = temp.path().join("source");
        let destination_dir = temp.path().join("model");
        let frozen = acquisition_fixture();
        write_acquisition_fixture(&source_dir, &frozen);
        std::fs::create_dir_all(&destination_dir).unwrap();
        std::fs::write(destination_dir.join("prior-generation"), b"prior").unwrap();
        let downloader = ModelDownloader::with_defaults();

        run_test_with_cx(|cx| async move {
            let mut lifecycle = fixture_lifecycle(&frozen);
            let error = downloader
                .acquire_frozen_model(
                    &cx,
                    ModelAcquisitionRequest {
                        frozen_manifest: &frozen,
                        source: ModelAcquisitionSource::LocalBundle(&source_dir),
                        destination_dir: &destination_dir,
                    },
                    &mut lifecycle,
                    |_| {},
                    |_, _| {
                        Err(SearchError::InvalidConfig {
                            field: "model_acquisition.load_self_test".to_owned(),
                            value: "failed".to_owned(),
                            reason: "fixture rejected by load self-test".to_owned(),
                        })
                    },
                )
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { .. }));
            assert_eq!(
                std::fs::read(destination_dir.join("prior-generation")).unwrap(),
                b"prior"
            );
        });
    }

    #[test]
    fn concurrent_acquisition_lock_fails_closed_without_touching_destination() {
        let temp = tempfile::tempdir().unwrap();
        let source_dir = temp.path().join("source");
        let destination_dir = temp.path().join("model");
        let frozen = acquisition_fixture();
        write_acquisition_fixture(&source_dir, &frozen);
        let downloader = ModelDownloader::with_defaults();

        run_test_with_cx(|cx| async move {
            let _held_lock = AcquisitionLock::acquire(&destination_dir).unwrap();
            let mut lifecycle = fixture_lifecycle(&frozen);
            let error = downloader
                .acquire_frozen_model(
                    &cx,
                    ModelAcquisitionRequest {
                        frozen_manifest: &frozen,
                        source: ModelAcquisitionSource::LocalBundle(&source_dir),
                        destination_dir: &destination_dir,
                    },
                    &mut lifecycle,
                    |_| {},
                    verify_fixture_load,
                )
                .await
                .unwrap_err();
            assert!(matches!(
                error,
                SearchError::InvalidConfig { ref field, .. }
                    if field == "model_acquisition.lock"
            ));
            assert!(!destination_dir.exists());
        });
    }

    #[test]
    fn acquisition_lock_process_helper() {
        let Some(destination) = std::env::var_os(LOCK_CHILD_DESTINATION_ENV) else {
            return;
        };
        let ready =
            PathBuf::from(std::env::var_os(LOCK_CHILD_READY_ENV).expect("child ready signal path"));
        let release = PathBuf::from(
            std::env::var_os(LOCK_CHILD_RELEASE_ENV).expect("child release signal path"),
        );
        let _lock = AcquisitionLock::acquire(Path::new(&destination)).unwrap();
        std::fs::write(&ready, b"locked").unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !release.exists() {
            assert!(
                Instant::now() < deadline,
                "parent did not release child acquisition lock"
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn two_process_acquisition_race_fails_closed_and_preserves_prior_generation() {
        let temp = tempfile::tempdir().unwrap();
        let source_dir = temp.path().join("source");
        let destination_dir = temp.path().join("model");
        let ready = temp.path().join("child-ready");
        let release = temp.path().join("child-release");
        let frozen = acquisition_fixture();
        write_acquisition_fixture(&source_dir, &frozen);
        std::fs::create_dir_all(&destination_dir).unwrap();
        std::fs::write(destination_dir.join("prior-generation"), b"prior").unwrap();

        let mut child = Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("model_download::tests::acquisition_lock_process_helper")
            .arg("--nocapture")
            .env(LOCK_CHILD_DESTINATION_ENV, &destination_dir)
            .env(LOCK_CHILD_READY_ENV, &ready)
            .env(LOCK_CHILD_RELEASE_ENV, &release)
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !ready.exists() {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(
                    status.success() && ready.exists(),
                    "lock helper exited before readiness with {status}"
                );
            }
            assert!(
                Instant::now() < deadline,
                "lock helper did not become ready"
            );
            thread::sleep(Duration::from_millis(10));
        }

        let downloader = ModelDownloader::with_defaults();
        let observed_lock_failure = Arc::new(AtomicBool::new(false));
        let observed_lock_failure_for_task = Arc::clone(&observed_lock_failure);
        run_test_with_cx(|cx| async move {
            let mut lifecycle = fixture_lifecycle(&frozen);
            let result = downloader
                .acquire_frozen_model(
                    &cx,
                    ModelAcquisitionRequest {
                        frozen_manifest: &frozen,
                        source: ModelAcquisitionSource::LocalBundle(&source_dir),
                        destination_dir: &destination_dir,
                    },
                    &mut lifecycle,
                    |_| {},
                    verify_fixture_load,
                )
                .await;
            observed_lock_failure_for_task.store(
                matches!(
                    result,
                    Err(SearchError::InvalidConfig { ref field, .. })
                        if field == "model_acquisition.lock"
                ),
                Ordering::SeqCst,
            );
        });
        std::fs::write(&release, b"release").unwrap();
        assert!(child.wait().unwrap().success());

        assert!(observed_lock_failure.load(Ordering::SeqCst));
        assert_eq!(
            std::fs::read(temp.path().join("model").join("prior-generation")).unwrap(),
            b"prior"
        );
    }

    #[test]
    fn structured_cancellation_prevents_stage_or_publication() {
        let temp = tempfile::tempdir().unwrap();
        let source_dir = temp.path().join("source");
        let destination_dir = temp.path().join("model");
        let frozen = acquisition_fixture();
        write_acquisition_fixture(&source_dir, &frozen);
        let downloader = ModelDownloader::with_defaults();

        run_test_with_cx(|cx| async move {
            cx.set_cancel_requested(true);
            let mut lifecycle = fixture_lifecycle(&frozen);
            let error = downloader
                .acquire_frozen_model(
                    &cx,
                    ModelAcquisitionRequest {
                        frozen_manifest: &frozen,
                        source: ModelAcquisitionSource::LocalBundle(&source_dir),
                        destination_dir: &destination_dir,
                    },
                    &mut lifecycle,
                    |_| {},
                    verify_fixture_load,
                )
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::Cancelled { .. }));
            assert!(matches!(lifecycle.state(), ModelState::Cancelled));
            assert!(!destination_dir.exists());
        });
    }

    #[test]
    fn structured_cancellation_is_observed_at_every_precommit_stage() {
        for cancellation_stage in [
            ModelAcquisitionStageV1::Transporting,
            ModelAcquisitionStageV1::StagedVerified,
            ModelAcquisitionStageV1::LoadSelfTestPassed,
        ] {
            let temp = tempfile::tempdir().unwrap();
            let source_dir = temp.path().join("source");
            let destination_dir = temp.path().join("model");
            let frozen = acquisition_fixture();
            write_acquisition_fixture(&source_dir, &frozen);
            std::fs::create_dir_all(&destination_dir).unwrap();
            std::fs::write(destination_dir.join("prior-generation"), b"prior").unwrap();
            let downloader = ModelDownloader::with_defaults();

            run_test_with_cx(|cx| async move {
                let cancellation_cx = cx.clone();
                let mut lifecycle = fixture_lifecycle(&frozen);
                let error = downloader
                    .acquire_frozen_model(
                        &cx,
                        ModelAcquisitionRequest {
                            frozen_manifest: &frozen,
                            source: ModelAcquisitionSource::LocalBundle(&source_dir),
                            destination_dir: &destination_dir,
                        },
                        &mut lifecycle,
                        move |progress| {
                            if progress.stage == cancellation_stage {
                                cancellation_cx.set_cancel_requested(true);
                            }
                        },
                        verify_fixture_load,
                    )
                    .await
                    .unwrap_err();

                assert!(matches!(error, SearchError::Cancelled { .. }));
                assert!(matches!(lifecycle.state(), ModelState::Cancelled));
                assert_eq!(
                    std::fs::read(destination_dir.join("prior-generation")).unwrap(),
                    b"prior"
                );
            });
        }
    }

    #[test]
    fn cancellation_after_publication_does_not_revoke_committed_generation() {
        let temp = tempfile::tempdir().unwrap();
        let source_dir = temp.path().join("source");
        let destination_dir = temp.path().join("model");
        let frozen = acquisition_fixture();
        write_acquisition_fixture(&source_dir, &frozen);
        let downloader = ModelDownloader::with_defaults();

        run_test_with_cx(|cx| async move {
            let cancellation_cx = cx.clone();
            let mut lifecycle = fixture_lifecycle(&frozen);
            let receipt = downloader
                .acquire_frozen_model(
                    &cx,
                    ModelAcquisitionRequest {
                        frozen_manifest: &frozen,
                        source: ModelAcquisitionSource::LocalBundle(&source_dir),
                        destination_dir: &destination_dir,
                    },
                    &mut lifecycle,
                    move |progress| {
                        if progress.stage == ModelAcquisitionStageV1::Published {
                            cancellation_cx.set_cancel_requested(true);
                        }
                    },
                    verify_fixture_load,
                )
                .await
                .unwrap();

            assert_eq!(receipt.outcome, ModelAcquisitionOutcomeV1::Published);
            assert!(matches!(
                lifecycle.state(),
                ModelState::AcquiredNeedsReindex
            ));
            frozen.manifest.verify_dir(&destination_dir).unwrap();
        });
    }

    #[test]
    fn download_progress_display_with_total() {
        let progress = DownloadProgress {
            file_name: "model.onnx".to_owned(),
            bytes_downloaded: 524_288,
            total_bytes: Some(1_048_576),
            files_completed: 0,
            files_total: 3,
            speed_bytes_per_sec: 1_048_576.0,
            eta_seconds: Some(0.5),
        };
        let display = progress.to_string();
        assert!(display.contains("[1/3]"));
        assert!(display.contains("model.onnx"));
        assert!(display.contains("50%"));
    }

    #[test]
    fn download_progress_display_without_total() {
        let progress = DownloadProgress {
            file_name: "config.json".to_owned(),
            bytes_downloaded: 1024,
            total_bytes: None,
            files_completed: 2,
            files_total: 3,
            speed_bytes_per_sec: 0.0,
            eta_seconds: None,
        };
        let display = progress.to_string();
        assert!(display.contains("[3/3]"));
        assert!(display.contains("config.json"));
        assert!(display.contains("1.0 KB"));
    }

    #[test]
    fn client_error_converts_to_search_error() {
        let err = client_error_to_search(
            ClientError::InvalidUrl("bad".to_owned()),
            "https://example.com",
        );
        assert!(matches!(err, SearchError::ModelLoadFailed { .. }));
    }

    #[test]
    fn tls_and_cancellation_errors_are_typed_bounded_and_redacted() {
        let tls_error = client_error_to_search(
            ClientError::TlsError("certificate detail from /private/cert.pem".to_owned()),
            "https://models.example.invalid/private/model?token=secret",
        );
        let rendered = tls_error.to_string();
        assert!(matches!(tls_error, SearchError::ModelLoadFailed { .. }));
        assert!(rendered.contains("models.example.invalid"));
        assert!(rendered.contains("tls-handshake-failure"));
        assert!(!rendered.contains("/private"));
        assert!(!rendered.contains("secret"));
        assert!(!rendered.contains("certificate detail"));

        let cancelled = client_error_to_search(
            ClientError::Cancelled,
            "https://models.example.invalid/model",
        );
        assert!(matches!(cancelled, SearchError::Cancelled { .. }));
    }

    #[test]
    fn download_single_file_success_writes_file_and_reports_progress() {
        let body = b"hello-model".to_vec();
        let file = ModelFile {
            name: "model.onnx".to_owned(),
            sha256: sha256_hex(&body),
            size: u64::try_from(body.len()).unwrap(),
            url: None,
        };

        let (base_url, served, handle) = spawn_test_http_server(vec![TestHttpResponse {
            status: 200,
            reason: "OK",
            body: body.clone(),
        }]);
        let url = format!("{base_url}/model.onnx");
        let dest_dir = tempfile::tempdir().unwrap();
        let dest = dest_dir.path().join("model.onnx");
        let dest_for_task = dest.clone();
        let progress = Arc::new(Mutex::new(Vec::<DownloadProgress>::new()));
        let progress_for_cb = Arc::clone(&progress);
        let downloader = ModelDownloader::new(DownloadConfig {
            max_retries: 0,
            retry_base_delay: Duration::from_millis(1),
            user_agent: "frankensearch-test".to_owned(),
            max_redirects: 0,
            max_response_bytes: DEFAULT_MAX_MODEL_ARTIFACT_BYTES,
            proxy: ProxySetting::Disabled,
            ..DownloadConfig::default()
        });

        run_test_with_cx(|cx| async move {
            downloader
                .download_single_file(&cx, &url, &dest_for_task, &file, 0, 1, &|p| {
                    progress_for_cb.lock().unwrap().push(p.clone());
                })
                .await
                .unwrap();
        });

        handle.join().unwrap();
        assert_eq!(served.load(Ordering::SeqCst), 1);
        assert_eq!(std::fs::read(dest).unwrap(), body);

        let events = progress.lock().unwrap();
        assert!(events.len() >= 2);
        assert_eq!(events[0].bytes_downloaded, 0);
        assert_eq!(events[0].file_name, "model.onnx");
        let last = events.last().unwrap();
        let expected_size = u64::try_from(body.len()).unwrap();
        assert_eq!(last.bytes_downloaded, expected_size);
        assert_eq!(last.total_bytes, Some(expected_size));
        drop(events);
    }

    #[test]
    fn download_config_debug_redacts_proxy_credentials() {
        let config = DownloadConfig {
            proxy: ProxySetting::Url("http://user:hunter2@proxy.corp:3128".to_owned()),
            ..DownloadConfig::default()
        };
        let rendered = format!("{config:?}");
        assert!(!rendered.contains("hunter2"), "{rendered}");
        assert!(rendered.contains("Url(<redacted>)"), "{rendered}");
    }

    #[test]
    fn download_single_file_routes_through_the_configured_proxy() {
        // A local listener stands in for an HTTP proxy. The target host does not resolve,
        // so the download can only succeed if the client sends it to the proxy, and an
        // http:// target must arrive in absolute form.
        let body = b"proxied model bytes".to_vec();
        let file = ModelFile {
            name: "model.onnx".to_owned(),
            sha256: sha256_hex(&body),
            size: u64::try_from(body.len()).unwrap(),
            url: None,
        };
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        let proxy = recording_download_server(
            listener,
            vec![format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                std::str::from_utf8(&body).unwrap(),
            )],
        );

        let url = "http://model-host.invalid/model.onnx".to_owned();
        let dest_dir = tempfile::tempdir().unwrap();
        let dest = dest_dir.path().join("model.onnx");
        let dest_for_task = dest.clone();
        let downloader = ModelDownloader::new(DownloadConfig {
            max_retries: 0,
            retry_base_delay: Duration::from_millis(1),
            user_agent: "frankensearch-test".to_owned(),
            max_redirects: 0,
            max_response_bytes: DEFAULT_MAX_MODEL_ARTIFACT_BYTES,
            proxy: ProxySetting::Url(format!("http://{proxy_addr}")),
            ..DownloadConfig::default()
        });

        run_test_with_cx(|cx| async move {
            downloader
                .download_single_file(&cx, &url, &dest_for_task, &file, 0, 1, &|_| {})
                .await
                .unwrap();
        });

        let requests = proxy.join().unwrap();
        assert_eq!(
            requests[0].lines().next(),
            Some("GET http://model-host.invalid/model.onnx HTTP/1.1")
        );
        assert_eq!(std::fs::read(dest).unwrap(), body);
    }

    const PROXY_TEST_BODY: &[u8] = b"verified proxy download";

    fn recording_download_server(
        listener: TcpListener,
        responses: Vec<String>,
    ) -> thread::JoinHandle<Vec<String>> {
        thread::spawn(move || {
            listener.set_nonblocking(true).unwrap();
            let mut requests = Vec::new();
            for response in responses {
                let deadline = Instant::now() + Duration::from_secs(10);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error) if error.kind() == ErrorKind::WouldBlock => {
                            assert!(
                                Instant::now() < deadline,
                                "download test server saw no request"
                            );
                            thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("download test accept failed: {error}"),
                    }
                };
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut request = Vec::new();
                while !request.ends_with(b"\r\n\r\n") {
                    let mut byte = [0];
                    stream.read_exact(&mut byte).unwrap();
                    request.push(byte[0]);
                    assert!(
                        request.len() <= 64 * 1024,
                        "request headers exceeded test bound"
                    );
                }
                requests.push(String::from_utf8(request).unwrap());
                stream.write_all(response.as_bytes()).unwrap();
            }
            requests
        })
    }

    fn proxy_download_response() -> String {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            PROXY_TEST_BODY.len(),
            std::str::from_utf8(PROXY_TEST_BODY).unwrap(),
        )
    }

    fn run_proxy_download_child(url: &str, disabled: bool, environment: &[(&str, String)]) {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command.args([
            "--exact",
            "model_download::tests::download_proxy_environment_child",
            "--nocapture",
        ]);
        for variable in [
            "http_proxy",
            "HTTP_PROXY",
            "https_proxy",
            "HTTPS_PROXY",
            "all_proxy",
            "ALL_PROXY",
            "no_proxy",
            "NO_PROXY",
        ] {
            command.env_remove(variable);
        }
        command
            .env("FRANKENSEARCH_TEST_DOWNLOAD_PROXY_URL", url)
            .env(
                "FRANKENSEARCH_TEST_DOWNLOAD_PROXY_DISABLED",
                if disabled { "1" } else { "0" },
            )
            .env_remove("FRANKENSEARCH_TEST_DOWNLOAD_PROXY_EXPECT_ERROR");
        for (key, value) in environment {
            command.env(key, value);
        }
        let mut child = command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        let mut timed_out = false;
        while child.try_wait().unwrap().is_none() {
            if Instant::now() >= deadline {
                timed_out = true;
                child.kill().unwrap();
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        let output = child.wait_with_output().unwrap();
        assert!(
            !timed_out && output.status.success(),
            "download child failed (timeout={timed_out}): {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    #[test]
    fn download_proxy_environment_child() {
        let Ok(url) = std::env::var("FRANKENSEARCH_TEST_DOWNLOAD_PROXY_URL") else {
            return;
        };
        let disabled = std::env::var("FRANKENSEARCH_TEST_DOWNLOAD_PROXY_DISABLED").unwrap() == "1";
        let expect_error = std::env::var("FRANKENSEARCH_TEST_DOWNLOAD_PROXY_EXPECT_ERROR")
            .is_ok_and(|value| value == "1");
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("model.bin");
        let file = ModelFile {
            name: "model.bin".to_owned(),
            sha256: sha256_hex(PROXY_TEST_BODY),
            size: u64::try_from(PROXY_TEST_BODY.len()).unwrap(),
            url: None,
        };
        let downloader = ModelDownloader::new(DownloadConfig {
            max_retries: 0,
            proxy: if disabled {
                ProxySetting::Disabled
            } else {
                ProxySetting::FromEnvironment
            },
            ..DownloadConfig::default()
        });
        run_test_with_cx(|cx| async move {
            let result = downloader
                .download_single_file(&cx, &url, &destination, &file, 0, 1, &|_| {})
                .await;
            if expect_error {
                let error = result.expect_err("the configured HTTPS proxy must refuse CONNECT");
                let rendered = format!("{error:?} {error}");
                assert!(rendered.contains("proxy-tunnel-http-407"), "{rendered}");
                for secret in [
                    "proxy-user-canary",
                    "proxy-password-canary",
                    "proxy-reason-canary",
                ] {
                    assert!(!rendered.contains(secret), "proxy error exposed {secret}");
                }
                assert!(!destination.exists());
                assert_eq!(
                    std::fs::read_dir(destination.parent().unwrap())
                        .unwrap()
                        .count(),
                    0
                );
            } else {
                result.unwrap();
                assert_eq!(std::fs::read(destination).unwrap(), PROXY_TEST_BODY);
            }
        });
    }

    #[test]
    fn download_proxy_environment_reaches_an_unresolvable_destination() {
        for variable in ["http_proxy", "all_proxy", "ALL_PROXY"] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let server = recording_download_server(listener, vec![proxy_download_response()]);
            let mut environment = vec![(variable, format!("http://{address}"))];
            // Windows cannot store both spellings as distinct variables.
            if !cfg!(windows) {
                environment.push(("HTTP_PROXY", "invalid://ignored".to_owned()));
            }
            run_proxy_download_child("http://model-host.invalid/model.bin", false, &environment);
            let requests = server.join().unwrap();
            assert!(
                requests[0].starts_with("GET http://model-host.invalid/model.bin HTTP/1.1\r\n")
            );
        }
    }

    #[test]
    fn download_proxy_https_environment_precedence_reaches_connect_proxy() {
        // A refusal proves HTTPS routing reached the selected proxy without
        // weakening certificate validation or needing a resolvable destination.
        for selected in ["https_proxy", "HTTPS_PROXY", "ALL_PROXY"] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let response = "HTTP/1.1 407 proxy-reason-canary\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned();
            let server = recording_download_server(listener, vec![response]);
            let mut environment = vec![
                (
                    selected,
                    format!("http://proxy-user-canary:proxy-password-canary@{address}"),
                ),
                (
                    "FRANKENSEARCH_TEST_DOWNLOAD_PROXY_EXPECT_ERROR",
                    "1".to_owned(),
                ),
            ];
            if selected != "ALL_PROXY" {
                environment.push(("ALL_PROXY", "invalid://must-not-be-used".to_owned()));
            }
            // Windows cannot hold lower/upper spellings as distinct values.
            if selected == "https_proxy" && !cfg!(windows) {
                environment.push(("HTTPS_PROXY", "invalid://must-not-be-used".to_owned()));
            }
            run_proxy_download_child("https://model-host.invalid/model.bin", false, &environment);
            let requests = server.join().unwrap();
            assert_eq!(requests.len(), 1);
            assert!(requests[0].starts_with("CONNECT model-host.invalid:443 HTTP/1.1\r\n"));
            assert!(requests[0].lines().any(|line| {
                line.split_once(':').is_some_and(|(name, value)| {
                    name.eq_ignore_ascii_case("proxy-authorization")
                        && value.trim().starts_with("Basic ")
                })
            }));
        }
    }

    #[test]
    fn download_proxy_disabled_bypass_and_unset_connect_directly() {
        for (disabled, environment) in [
            (
                true,
                vec![(
                    "http_proxy",
                    "invalid://user:secret@proxy.invalid".to_owned(),
                )],
            ),
            (
                false,
                vec![
                    (
                        "http_proxy",
                        "invalid://user:secret@proxy.invalid".to_owned(),
                    ),
                    ("no_proxy", "127.0.0.1".to_owned()),
                ],
            ),
            (
                false,
                vec![
                    (
                        "ALL_PROXY",
                        "invalid://user:secret@proxy.invalid".to_owned(),
                    ),
                    ("NO_PROXY", "*".to_owned()),
                ],
            ),
            (
                false,
                vec![(
                    "HTTP_PROXY",
                    "invalid://user:secret@proxy.invalid".to_owned(),
                )],
            ),
            (false, vec![]),
        ] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let server = recording_download_server(listener, vec![proxy_download_response()]);
            run_proxy_download_child(
                &format!("http://{address}/model.bin"),
                disabled,
                &environment,
            );
            let requests = server.join().unwrap();
            assert!(requests[0].starts_with("GET /model.bin HTTP/1.1\r\n"));
            assert!(
                !requests[0]
                    .to_ascii_lowercase()
                    .contains("proxy-authorization")
            );
        }
    }

    #[test]
    fn download_proxy_redirects_reselect_routes_in_both_directions() {
        let origin = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin_address = origin.local_addr().unwrap();
        let proxy = TcpListener::bind("127.0.0.1:0").unwrap();
        let proxy_address = proxy.local_addr().unwrap();
        let origin_server = recording_download_server(origin, vec![
            "HTTP/1.1 302 Found\r\nLocation: http://download-cdn.invalid/model.bin\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned(),
            proxy_download_response(),
        ]);
        let proxy_server = recording_download_server(
            proxy,
            vec![format!(
                "HTTP/1.1 307 Temporary Redirect\r\nLocation: http://{origin_address}/verified.bin\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )],
        );
        run_proxy_download_child(
            &format!("http://{origin_address}/start?origin-token=private"),
            false,
            &[
                (
                    "http_proxy",
                    format!("http://user:password@{proxy_address}"),
                ),
                ("no_proxy", "127.0.0.1".to_owned()),
            ],
        );
        let direct_requests = origin_server.join().unwrap();
        let proxy_requests = proxy_server.join().unwrap();
        assert!(direct_requests[0].starts_with("GET /start?origin-token=private HTTP/1.1\r\n"));
        assert!(direct_requests[1].starts_with("GET /verified.bin HTTP/1.1\r\n"));
        assert!(
            direct_requests
                .iter()
                .all(|request| !request.to_ascii_lowercase().contains("proxy-authorization"))
        );
        assert!(
            proxy_requests[0].starts_with("GET http://download-cdn.invalid/model.bin HTTP/1.1\r\n")
        );
        assert!(
            proxy_requests[0]
                .to_ascii_lowercase()
                .contains("proxy-authorization:")
        );
        assert!(!proxy_requests[0].contains("origin-token"));
    }

    #[test]
    fn download_proxy_malformed_configuration_fails_closed_and_redacts() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let destination = temporary.path().join("model.bin");
        let downloader = ModelDownloader::new(DownloadConfig {
            proxy: ProxySetting::Url(
                "unsupported://secret-user:secret-password@private-proxy.invalid".to_owned(),
            ),
            max_retries: 0,
            ..DownloadConfig::default()
        });
        let file = ModelFile {
            name: "model.bin".to_owned(),
            sha256: sha256_hex(PROXY_TEST_BODY),
            size: 23,
            url: None,
        };
        run_test_with_cx(|cx| async move {
            let error = downloader
                .download_single_file(
                    &cx,
                    &format!("http://{address}/model.bin"),
                    &destination,
                    &file,
                    0,
                    1,
                    &|_| {},
                )
                .await
                .unwrap_err();
            let rendered = format!("{error}\n{error:?}");
            for secret in ["secret-user", "secret-password", "private-proxy.invalid"] {
                assert!(!rendered.contains(secret));
            }
            assert!(!destination.exists());
            assert!(!destination.with_extension("tmp").exists());
        });
        assert_eq!(listener.accept().unwrap_err().kind(), ErrorKind::WouldBlock);
    }

    #[test]
    fn download_redirect_references_preserve_query_and_path_semantics() {
        let base = "https://models.example/a/b/model.bin?old=token";
        for (location, expected) in [
            ("../weights.bin", "https://models.example/a/weights.bin"),
            (
                "?next=https://cdn.invalid/x",
                "https://models.example/a/b/model.bin?next=https://cdn.invalid/x",
            ),
            (
                "#fragment",
                "https://models.example/a/b/model.bin?old=token",
            ),
            ("//cdn.invalid/model.bin", "https://cdn.invalid/model.bin"),
            (
                "HTTPS://cdn.invalid/model.bin#fragment",
                "https://cdn.invalid/model.bin",
            ),
            ("/a//.", "https://models.example/a//"),
            ("/a///..", "https://models.example/a//"),
            ("/../../x", "https://models.example/x"),
            (
                "/%2e%2e/signed//object?x=1",
                "https://models.example/%2e%2e/signed//object?x=1",
            ),
        ] {
            assert_eq!(
                resolve_download_redirect(base, location).unwrap(),
                expected,
                "{location}"
            );
        }
        assert_eq!(
            resolve_download_redirect("https://[::1]:8443/a/model", "../weights").unwrap(),
            "https://[::1]:8443/weights",
        );
        for location in [
            "http://cdn.invalid/model",
            "ftp://cdn.invalid/model",
            "file:///etc/passwd",
            "https://user:secret@cdn.invalid/x",
            "https://cdn.invalid:bad/x",
            "https://cdn.invalid/x\r\nInjected: secret",
        ] {
            assert!(resolve_download_redirect(base, location).is_err());
        }
    }

    #[test]
    fn download_redirect_limit_stops_before_publishing_bytes() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let redirect = "HTTP/1.1 302 Found\r\nLocation: /loop\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned();
        let server = recording_download_server(listener, vec![redirect.clone(), redirect]);
        let downloader = ModelDownloader::new(DownloadConfig {
            max_redirects: 1,
            max_retries: 0,
            proxy: ProxySetting::Disabled,
            ..DownloadConfig::default()
        });
        let temporary = tempfile::tempdir().unwrap();
        let destination = temporary.path().join("model.bin");
        let file = ModelFile {
            name: "model.bin".to_owned(),
            sha256: sha256_hex(PROXY_TEST_BODY),
            size: 23,
            url: None,
        };
        run_test_with_cx(|cx| async move {
            let error = downloader
                .download_single_file(
                    &cx,
                    &format!("http://{address}/loop"),
                    &destination,
                    &file,
                    0,
                    1,
                    &|_| {},
                )
                .await
                .unwrap_err();
            assert!(error.to_string().contains("redirect-limit-2-of-1"));
            assert!(!destination.exists());
            assert!(!destination.with_extension("tmp").exists());
        });
        assert_eq!(server.join().unwrap().len(), 2);
    }

    #[test]
    fn download_single_file_succeeds_beyond_legacy_16mib_codec_cap() {
        // Regression for #27: the default HTTP codec caps response bodies at
        // 16 MiB, which rejected production model artifacts (the default Potion
        // tokenizer is ~17.8 MiB). A body just over the legacy cap must now
        // stream to disk and verify.
        let body = vec![0x41_u8; 16 * 1024 * 1024 + 4096];
        let file = ModelFile {
            name: "big.onnx".to_owned(),
            sha256: sha256_hex(&body),
            size: u64::try_from(body.len()).unwrap(),
            url: None,
        };

        let (base_url, served, handle) = spawn_test_http_server(vec![TestHttpResponse {
            status: 200,
            reason: "OK",
            body: body.clone(),
        }]);
        let url = format!("{base_url}/big.onnx");
        let dest_dir = tempfile::tempdir().unwrap();
        let dest = dest_dir.path().join("big.onnx");
        let dest_for_task = dest.clone();
        // Default config carries the 2 GiB cap.
        let downloader = ModelDownloader::with_defaults();

        run_test_with_cx(|cx| async move {
            downloader
                .download_single_file(&cx, &url, &dest_for_task, &file, 0, 1, &|_| {})
                .await
                .unwrap();
        });

        handle.join().unwrap();
        assert_eq!(served.load(Ordering::SeqCst), 1);
        assert_eq!(std::fs::read(dest).unwrap().len(), body.len());
    }

    #[test]
    fn download_model_rejects_artifact_larger_than_response_cap() {
        // A manifest whose declared artifact exceeds the configured cap must
        // fail fast with an actionable InvalidConfig error, not a cryptic
        // mid-stream BodyTooLarge after retries.
        let mut manifest = ModelManifest::minilm_v2();
        let oversized = u64::try_from(manifest_cap_test_bytes()).unwrap() + 1;
        manifest.files[0].size = oversized;
        // Skip the sum-of-sizes cross-check so the oversized size reaches the cap guard.
        manifest.download_size_bytes = 0;
        let consent = crate::model_manifest::DownloadConsent::granted(
            crate::model_manifest::ConsentSource::Environment,
        );
        let mut lifecycle = ModelLifecycle::new(manifest.clone(), consent);
        let dest = tempfile::tempdir().unwrap();
        let downloader = ModelDownloader::new(DownloadConfig {
            max_retries: 0,
            retry_base_delay: Duration::from_millis(1),
            user_agent: "frankensearch-test".to_owned(),
            max_redirects: 0,
            max_response_bytes: manifest_cap_test_bytes(),
            proxy: ProxySetting::Disabled,
            ..DownloadConfig::default()
        });

        run_test_with_cx(|cx| async move {
            let err = downloader
                .download_model(&cx, &manifest, dest.path(), &mut lifecycle, |_| {})
                .await
                .unwrap_err();
            assert!(matches!(
                err,
                SearchError::InvalidConfig { ref field, .. }
                    if field == "download.max_response_bytes"
            ));
            assert!(err.to_string().contains("max_response_bytes"));
            assert!(matches!(
                lifecycle.state(),
                crate::model_manifest::ModelState::VerificationFailed { .. }
            ));
        });
    }

    const fn manifest_cap_test_bytes() -> usize {
        4 * 1024
    }

    #[test]
    fn download_file_with_retry_succeeds_after_transient_http_error() {
        let body = b"retry-success".to_vec();
        let file = ModelFile {
            name: "model.onnx".to_owned(),
            sha256: sha256_hex(&body),
            size: u64::try_from(body.len()).unwrap(),
            url: None,
        };

        let (base_url, served, handle) = spawn_test_http_server(vec![
            TestHttpResponse {
                status: 500,
                reason: "Internal Server Error",
                body: b"server error".to_vec(),
            },
            TestHttpResponse {
                status: 200,
                reason: "OK",
                body: body.clone(),
            },
        ]);
        let url = format!("{base_url}/model.onnx");
        let dest_dir = tempfile::tempdir().unwrap();
        let dest = dest_dir.path().join("model.onnx");
        let dest_for_task = dest.clone();
        let downloader = ModelDownloader::new(DownloadConfig {
            max_retries: 1,
            retry_base_delay: Duration::from_millis(1),
            user_agent: "frankensearch-test".to_owned(),
            max_redirects: 0,
            max_response_bytes: DEFAULT_MAX_MODEL_ARTIFACT_BYTES,
            proxy: ProxySetting::Disabled,
            ..DownloadConfig::default()
        });

        run_test_with_cx(|cx| async move {
            downloader
                .download_file_with_retry(&cx, &url, &dest_for_task, &file, 0, 1, &|_| {})
                .await
                .unwrap();
        });

        handle.join().unwrap();
        assert_eq!(served.load(Ordering::SeqCst), 2);
        assert_eq!(std::fs::read(dest).unwrap(), body);
    }

    #[test]
    fn download_file_with_retry_returns_error_after_max_attempts() {
        let file = ModelFile {
            name: "model.onnx".to_owned(),
            sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            size: 4,
            url: None,
        };

        let (base_url, served, handle) = spawn_test_http_server(vec![
            TestHttpResponse {
                status: 500,
                reason: "Internal Server Error",
                body: b"error".to_vec(),
            },
            TestHttpResponse {
                status: 500,
                reason: "Internal Server Error",
                body: b"error".to_vec(),
            },
        ]);
        let url = format!("{base_url}/model.onnx");
        let dest_dir = tempfile::tempdir().unwrap();
        let dest = dest_dir.path().join("model.onnx");
        let dest_for_task = dest.clone();
        let downloader = ModelDownloader::new(DownloadConfig {
            max_retries: 1,
            retry_base_delay: Duration::from_millis(1),
            user_agent: "frankensearch-test".to_owned(),
            max_redirects: 0,
            max_response_bytes: DEFAULT_MAX_MODEL_ARTIFACT_BYTES,
            proxy: ProxySetting::Disabled,
            ..DownloadConfig::default()
        });

        run_test_with_cx(|cx| async move {
            let err = downloader
                .download_file_with_retry(&cx, &url, &dest_for_task, &file, 0, 1, &|_| {})
                .await
                .unwrap_err();
            assert!(matches!(err, SearchError::ModelLoadFailed { .. }));
            assert!(err.to_string().contains("HTTP 500"));
        });

        handle.join().unwrap();
        assert_eq!(served.load(Ordering::SeqCst), 2);
        assert!(!dest.exists());
    }

    #[test]
    fn download_single_file_hash_mismatch_does_not_write_destination() {
        let expected = b"expected-content".to_vec();
        let file = ModelFile {
            name: "model.onnx".to_owned(),
            sha256: sha256_hex(&expected),
            size: u64::try_from(expected.len()).unwrap(),
            url: None,
        };

        let (base_url, served, handle) = spawn_test_http_server(vec![TestHttpResponse {
            status: 200,
            reason: "OK",
            body: b"different-content".to_vec(),
        }]);
        let url = format!("{base_url}/model.onnx");
        let dest_dir = tempfile::tempdir().unwrap();
        let dest = dest_dir.path().join("model.onnx");
        let dest_for_task = dest.clone();
        let downloader = ModelDownloader::new(DownloadConfig {
            max_retries: 0,
            retry_base_delay: Duration::from_millis(1),
            user_agent: "frankensearch-test".to_owned(),
            max_redirects: 0,
            max_response_bytes: DEFAULT_MAX_MODEL_ARTIFACT_BYTES,
            proxy: ProxySetting::Disabled,
            ..DownloadConfig::default()
        });

        run_test_with_cx(|cx| async move {
            let err = downloader
                .download_single_file(&cx, &url, &dest_for_task, &file, 0, 1, &|_| {})
                .await
                .unwrap_err();
            assert!(matches!(err, SearchError::HashMismatch { .. }));
        });

        handle.join().unwrap();
        assert_eq!(served.load(Ordering::SeqCst), 1);
        assert!(!dest.exists());
        assert!(!dest.with_extension("tmp").exists());
    }

    #[test]
    fn download_model_failure_transitions_lifecycle_to_verification_failed() {
        let manifest = ModelManifest {
            id: "test-model".to_owned(),
            version: "test-v1".to_owned(),
            display_name: None,
            description: None,
            repo: "owner/repo".to_owned(),
            revision: "d".repeat(40),
            files: vec![ModelFile {
                name: "bad-file.bin".to_owned(),
                sha256: "0".repeat(64),
                size: 1,
                // The manifest remains valid and production-ready, while the
                // missing authority forces an immediate client error.
                url: Some("https://".to_owned()),
            }],
            license: "Apache-2.0".to_owned(),
            dimension: None,
            tier: None,
            download_size_bytes: 0,
        };
        assert!(manifest.validate().is_ok());
        assert!(manifest.is_production_ready());
        let consent = crate::model_manifest::DownloadConsent::granted(
            crate::model_manifest::ConsentSource::Environment,
        );
        let mut lifecycle = ModelLifecycle::new(manifest.clone(), consent);
        let dest = tempfile::tempdir().unwrap();
        let downloader = ModelDownloader::new(DownloadConfig {
            max_retries: 0,
            retry_base_delay: Duration::from_millis(1),
            user_agent: "frankensearch-test".to_owned(),
            max_redirects: 0,
            max_response_bytes: DEFAULT_MAX_MODEL_ARTIFACT_BYTES,
            proxy: ProxySetting::Disabled,
            ..DownloadConfig::default()
        });

        run_test_with_cx(|cx| async move {
            let err = downloader
                .download_model(&cx, &manifest, dest.path(), &mut lifecycle, |_| {})
                .await
                .unwrap_err();
            assert!(matches!(err, SearchError::ModelLoadFailed { .. }));
            assert!(matches!(
                lifecycle.state(),
                crate::model_manifest::ModelState::VerificationFailed { .. }
            ));
            assert!(lifecycle.begin_download(1).is_ok());
        });
    }

    #[test]
    fn create_unique_staging_dir_returns_distinct_paths() {
        let temp = tempfile::tempdir().unwrap();
        let first = create_unique_staging_dir(temp.path()).expect("first staging dir");
        let second = create_unique_staging_dir(temp.path()).expect("second staging dir");

        assert_ne!(first, second);
        assert!(first.is_dir());
        assert!(second.is_dir());
    }

    // ─── bd-r476 tests begin ───

    #[test]
    fn response_content_length_found() {
        let headers = vec![("Content-Length".to_owned(), "42".to_owned())];
        assert_eq!(response_content_length(&headers), Some(42));
    }

    #[test]
    fn response_content_length_missing() {
        let headers = vec![("X-Custom".to_owned(), "value".to_owned())];
        assert_eq!(response_content_length(&headers), None);
    }

    #[test]
    fn response_content_length_invalid_value() {
        let headers = vec![("Content-Length".to_owned(), "not-a-number".to_owned())];
        assert_eq!(response_content_length(&headers), None);
    }

    #[test]
    fn response_content_length_case_insensitive() {
        let headers = vec![("content-length".to_owned(), "100".to_owned())];
        assert_eq!(response_content_length(&headers), Some(100));

        let headers_upper = vec![("CONTENT-LENGTH".to_owned(), "200".to_owned())];
        assert_eq!(response_content_length(&headers_upper), Some(200));
    }

    #[test]
    fn response_content_length_trims_whitespace() {
        let headers = vec![("Content-Length".to_owned(), "  300  ".to_owned())];
        assert_eq!(response_content_length(&headers), Some(300));
    }

    #[test]
    fn sha256_digest_hex_known_empty() {
        // SHA-256 of empty data
        let hash = sha256_hex(b"");
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_digest_hex_always_64_chars() {
        let hash = sha256_hex(b"test");
        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn sha256_digest_hex_lowercase() {
        let hash = sha256_hex(b"ABC");
        // Hex should be lowercase
        assert_eq!(hash, hash.to_lowercase());
    }

    #[test]
    fn format_bytes_boundary_values() {
        // Just below KB
        assert_eq!(format_bytes(1023), "1023 B");
        // Exactly KB
        assert_eq!(format_bytes(1024), "1.0 KB");
        // Just below MB
        assert!(format_bytes(1024 * 1024 - 1).contains("KB"));
        // Exactly MB
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        // Just below GB
        assert!(format_bytes(1024 * 1024 * 1024 - 1).contains("MB"));
        // Exactly GB
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn format_bytes_large_values() {
        let ten_gb = 10 * 1024 * 1024 * 1024_u64;
        let result = format_bytes(ten_gb);
        assert!(result.contains("GB"));
        assert!(result.contains("10.0"));
    }

    #[test]
    fn temp_file_guard_armed_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("temp.bin");
        std::fs::write(&path, b"data").unwrap();
        assert!(path.exists());
        {
            let _guard = TempFileGuard::new(path.clone());
            // guard drops here, armed=true
        }
        assert!(!path.exists(), "armed guard should remove file on drop");
    }

    #[test]
    fn temp_file_guard_disarmed_does_not_clean_up() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keep.bin");
        std::fs::write(&path, b"data").unwrap();
        assert!(path.exists());
        {
            let mut guard = TempFileGuard::new(path.clone());
            guard.disarm();
            // guard drops here, armed=false
        }
        assert!(path.exists(), "disarmed guard should leave file intact");
    }

    #[test]
    fn temp_file_guard_armed_nonexistent_file_does_not_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonexistent.bin");
        {
            let _guard = TempFileGuard::new(path);
            // guard drops on nonexistent file — should not panic
        }
    }

    #[test]
    fn download_progress_debug() {
        let progress = DownloadProgress {
            file_name: "model.onnx".to_owned(),
            bytes_downloaded: 0,
            total_bytes: Some(100),
            files_completed: 0,
            files_total: 1,
            speed_bytes_per_sec: 0.0,
            eta_seconds: None,
        };
        let debug = format!("{progress:?}");
        assert!(debug.contains("DownloadProgress"));
        assert!(debug.contains("model.onnx"));
    }

    #[test]
    fn download_progress_clone() {
        let progress = DownloadProgress {
            file_name: "file.bin".to_owned(),
            bytes_downloaded: 42,
            total_bytes: Some(100),
            files_completed: 1,
            files_total: 2,
            speed_bytes_per_sec: 1000.0,
            eta_seconds: Some(0.058),
        };
        #[allow(clippy::redundant_clone)]
        let cloned = progress.clone();
        assert_eq!(cloned.file_name, "file.bin");
        assert_eq!(cloned.bytes_downloaded, 42);
        assert_eq!(cloned.total_bytes, Some(100));
        assert_eq!(cloned.files_completed, 1);
        assert_eq!(cloned.files_total, 2);
    }

    #[test]
    fn download_config_clone_and_debug() {
        let config = DownloadConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.max_retries, config.max_retries);
        assert_eq!(cloned.user_agent, config.user_agent);
        let debug = format!("{config:?}");
        assert!(debug.contains("DownloadConfig"));
        assert!(debug.contains("max_retries"));
    }

    #[test]
    fn model_downloader_with_defaults_creates_valid_instance() {
        let _downloader = ModelDownloader::with_defaults();
    }

    #[test]
    fn download_progress_display_zero_total_bytes_no_percent() {
        let progress = DownloadProgress {
            file_name: "test.bin".to_owned(),
            bytes_downloaded: 500,
            total_bytes: Some(0),
            files_completed: 0,
            files_total: 1,
            speed_bytes_per_sec: 0.0,
            eta_seconds: None,
        };
        let display = progress.to_string();
        // total_bytes=0 is filtered out, so no percentage
        assert!(!display.contains('%'));
    }

    #[test]
    fn create_unique_staging_dir_creates_parent_if_needed() {
        let temp = tempfile::tempdir().unwrap();
        let nested = temp.path().join("deeply").join("nested").join("dir");
        let result = create_unique_staging_dir(&nested).expect("should create nested dir");
        assert!(result.is_dir());
        // The function creates the parent of dest_dir (for sibling staging), not dest_dir itself.
        assert!(nested.parent().unwrap().is_dir());
    }

    // ─── bd-r476 tests end ───

    #[test]
    fn download_model_rejects_non_production_ready_manifest() {
        let mut manifest = ModelManifest::minilm_v2();
        manifest.revision = PLACEHOLDER_PINNED_REVISION.to_owned();
        manifest.files[0].sha256 = PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned();
        manifest.files[0].size = 0;
        manifest.files[0].url = None;
        manifest.download_size_bytes = 0;
        let consent = crate::model_manifest::DownloadConsent::granted(
            crate::model_manifest::ConsentSource::Environment,
        );
        let mut lifecycle = ModelLifecycle::new(manifest.clone(), consent);
        let dest = tempfile::tempdir().unwrap();
        let downloader = ModelDownloader::new(DownloadConfig {
            max_retries: 0,
            retry_base_delay: Duration::from_millis(1),
            user_agent: "frankensearch-test".to_owned(),
            max_redirects: 0,
            max_response_bytes: DEFAULT_MAX_MODEL_ARTIFACT_BYTES,
            proxy: ProxySetting::Disabled,
            ..DownloadConfig::default()
        });

        run_test_with_cx(|cx| async move {
            let err = downloader
                .download_model(&cx, &manifest, dest.path(), &mut lifecycle, |_| {})
                .await
                .unwrap_err();
            assert!(matches!(err, SearchError::InvalidConfig { .. }));
            assert!(err.to_string().contains("production-ready"));
            assert!(matches!(
                lifecycle.state(),
                crate::model_manifest::ModelState::VerificationFailed { .. }
            ));
        });
    }

    #[test]
    fn download_model_rejects_manifest_with_path_traversal_filename() {
        let mut manifest = ModelManifest::minilm_v2();
        manifest.files[0].name = "../escape.bin".to_owned();
        let consent = crate::model_manifest::DownloadConsent::granted(
            crate::model_manifest::ConsentSource::Environment,
        );
        let mut lifecycle = ModelLifecycle::new(manifest.clone(), consent);
        let dest = tempfile::tempdir().unwrap();
        let downloader = ModelDownloader::new(DownloadConfig {
            max_retries: 0,
            retry_base_delay: Duration::from_millis(1),
            user_agent: "frankensearch-test".to_owned(),
            max_redirects: 0,
            max_response_bytes: DEFAULT_MAX_MODEL_ARTIFACT_BYTES,
            proxy: ProxySetting::Disabled,
            ..DownloadConfig::default()
        });

        run_test_with_cx(|cx| async move {
            let err = downloader
                .download_model(&cx, &manifest, dest.path(), &mut lifecycle, |_| {})
                .await
                .unwrap_err();
            assert!(matches!(
                err,
                SearchError::InvalidConfig { ref field, .. } if field == "files[].name"
            ));
            assert!(matches!(
                lifecycle.state(),
                crate::model_manifest::ModelState::VerificationFailed { .. }
            ));
        });
    }
}

#[cfg(test)]
mod download_wait_tests {
    use super::*;
    use asupersync::test_utils::run_test_with_cx;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Wake, Waker};

    const TEST_URL: &str = "https://model.invalid/model?secret=never-log";

    struct CountWake(AtomicUsize);

    impl Wake for CountWake {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct PendingWork(Arc<AtomicBool>);

    impl Future for PendingWork {
        type Output = SearchResult<()>;

        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }

    impl Drop for PendingWork {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn pending_download_wait_cancellation_wakes_and_releases_owned_work() {
        for cancel in [false, true] {
            run_test_with_cx(|cx| async move {
                let dropped = Arc::new(AtomicBool::new(false));
                let wake = Arc::new(CountWake(AtomicUsize::new(0)));
                let waker = Waker::from(Arc::clone(&wake));
                let progress = AtomicU64::new(0);
                let mut operation = Box::pin(run_download_io_with_elapsed(
                    &cx,
                    TEST_URL,
                    Duration::from_secs(3_600),
                    &progress,
                    PendingWork(Arc::clone(&dropped)),
                    || Duration::ZERO,
                ));
                assert!(
                    operation
                        .as_mut()
                        .poll(&mut Context::from_waker(&waker))
                        .is_pending()
                );
                assert!(!dropped.load(Ordering::SeqCst));
                if cancel {
                    let before = wake.0.load(Ordering::SeqCst);
                    cx.cancel_with(asupersync::CancelKind::User, Some("stop silent proxy"));
                    assert!(wake.0.load(Ordering::SeqCst) > before);
                    assert!(matches!(
                        operation.as_mut().poll(&mut Context::from_waker(&waker)),
                        Poll::Ready(Err(SearchError::Cancelled { .. }))
                    ));
                    assert!(dropped.load(Ordering::SeqCst));
                }
                drop(operation);
                assert!(dropped.load(Ordering::SeqCst));
            });
        }
    }

    #[test]
    fn retry_backoff_cancellation_does_not_wait_for_the_delay() {
        run_test_with_cx(|cx| async move {
            let wake = Arc::new(CountWake(AtomicUsize::new(0)));
            let waker = Waker::from(Arc::clone(&wake));
            let mut operation = Box::pin(wait_download_retry(&cx, TEST_URL, Duration::MAX));
            assert!(
                operation
                    .as_mut()
                    .poll(&mut Context::from_waker(&waker))
                    .is_pending()
            );
            let before = wake.0.load(Ordering::SeqCst);
            cx.cancel_with(asupersync::CancelKind::User, Some("stop backoff"));
            assert!(wake.0.load(Ordering::SeqCst) > before);
            assert!(matches!(
                operation.as_mut().poll(&mut Context::from_waker(&waker)),
                Poll::Ready(Err(SearchError::Cancelled { .. }))
            ));
        });
    }

    #[test]
    fn ambient_cancellation_terminates_a_wait_with_a_separate_caller_context() {
        run_test_with_cx(|cx| async move {
            let ambient = Cx::for_testing();
            let _ambient = Cx::set_current(Some(ambient.clone()));
            let progress = AtomicU64::new(0);
            let dropped = Arc::new(AtomicBool::new(false));
            let mut operation = Box::pin(run_download_io_with_elapsed(
                &cx,
                TEST_URL,
                Duration::from_secs(60),
                &progress,
                PendingWork(Arc::clone(&dropped)),
                || Duration::ZERO,
            ));
            let mut context = Context::from_waker(Waker::noop());
            assert!(operation.as_mut().poll(&mut context).is_pending());
            ambient.cancel_with(asupersync::CancelKind::User, Some("cancel ambient owner"));
            assert!(!cx.is_cancel_requested());
            // Sleep completes early under ambient cancellation. The guard must
            // report cancellation immediately and never repoll that completed
            // timer, even though the explicitly supplied caller is not cancelled.
            assert!(matches!(
                operation.as_mut().poll(&mut context),
                Poll::Ready(Err(SearchError::Cancelled { .. }))
            ));
            assert!(dropped.load(Ordering::SeqCst));
        });
    }

    struct QueuedBody {
        frames: Arc<Mutex<VecDeque<Frame<&'static [u8]>>>>,
        dropped: Arc<AtomicBool>,
    }

    impl Body for QueuedBody {
        type Data = &'static [u8];
        type Error = std::convert::Infallible;

        fn poll_frame(
            self: Pin<&mut Self>,
            _: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
            self.frames
                .lock()
                .unwrap()
                .pop_front()
                .map_or(Poll::Pending, |frame| Poll::Ready(Some(Ok(frame))))
        }
    }

    impl Drop for QueuedBody {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn model_body_idle_deadline_extends_only_when_nonempty_bytes_arrive() {
        run_test_with_cx(|cx| async move {
            let frames = Arc::new(Mutex::new(VecDeque::new()));
            let dropped = Arc::new(AtomicBool::new(false));
            let body = QueuedBody {
                frames: Arc::clone(&frames),
                dropped: Arc::clone(&dropped),
            };
            let milliseconds = AtomicU64::new(0);
            let elapsed = || Duration::from_millis(milliseconds.load(Ordering::SeqCst));
            let progress = AtomicU64::new(0);
            let copied = Arc::new(Mutex::new(Vec::new()));
            let copied_for_work = Arc::clone(&copied);
            let work = copy_download_body(&cx, TEST_URL, body, &progress, &elapsed, move |bytes| {
                copied_for_work.lock().unwrap().extend_from_slice(bytes);
                Ok(())
            });
            let mut operation = Box::pin(run_download_io_with_elapsed(
                &cx,
                TEST_URL,
                Duration::from_millis(50),
                &progress,
                work,
                &elapsed,
            ));
            let mut context = Context::from_waker(Waker::noop());
            assert!(operation.as_mut().poll(&mut context).is_pending());
            milliseconds.store(40, Ordering::SeqCst);
            frames.lock().unwrap().push_back(Frame::Data(b"a"));
            assert!(operation.as_mut().poll(&mut context).is_pending());
            milliseconds.store(80, Ordering::SeqCst);
            frames.lock().unwrap().push_back(Frame::Data(b"b"));
            assert!(operation.as_mut().poll(&mut context).is_pending());
            // Total time already exceeds the idle budget, yet real progress
            // keeps the download valid. Empty frames/trailers do not extend it.
            milliseconds.store(120, Ordering::SeqCst);
            frames.lock().unwrap().extend([
                Frame::Data(&b""[..]),
                Frame::Trailers(asupersync::http::body::HeaderMap::default()),
            ]);
            assert!(operation.as_mut().poll(&mut context).is_pending());
            assert_eq!(progress.load(Ordering::SeqCst), 80_000_000);
            milliseconds.store(130, Ordering::SeqCst);
            let result = operation.as_mut().poll(&mut context);
            let Poll::Ready(Err(error)) = result else {
                panic!("idle deadline did not terminate the transfer");
            };
            assert!(error.to_string().contains("request-deadline-exceeded"));
            assert!(!error.to_string().contains("never-log"));
            assert_eq!(*copied.lock().unwrap(), b"ab");
            assert!(dropped.load(Ordering::SeqCst));
        });
    }

    #[test]
    fn ready_empty_frames_yield_and_cannot_keep_a_download_alive() {
        struct EmptyFrames(Arc<AtomicUsize>);
        impl Body for EmptyFrames {
            type Data = &'static [u8];
            type Error = std::convert::Infallible;
            fn poll_frame(
                self: Pin<&mut Self>,
                _: &mut Context<'_>,
            ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
                let before = self.0.fetch_add(1, Ordering::SeqCst);
                assert!(
                    before < DOWNLOAD_BODY_STEPS_PER_POLL,
                    "body failed to yield"
                );
                Poll::Ready(Some(Ok(Frame::Data(&[]))))
            }
        }
        run_test_with_cx(|cx| async move {
            let polls = Arc::new(AtomicUsize::new(0));
            let milliseconds = AtomicU64::new(0);
            let elapsed = || Duration::from_millis(milliseconds.load(Ordering::SeqCst));
            let progress = AtomicU64::new(0);
            let work = copy_download_body(
                &cx,
                TEST_URL,
                EmptyFrames(Arc::clone(&polls)),
                &progress,
                &elapsed,
                |_| panic!("empty bytes must not count as progress"),
            );
            let mut operation = Box::pin(run_download_io_with_elapsed(
                &cx,
                TEST_URL,
                Duration::from_millis(50),
                &progress,
                work,
                &elapsed,
            ));
            let wake = Arc::new(CountWake(AtomicUsize::new(0)));
            let waker = Waker::from(Arc::clone(&wake));
            let mut context = Context::from_waker(&waker);
            assert!(operation.as_mut().poll(&mut context).is_pending());
            assert!(
                wake.0.load(Ordering::SeqCst) > 0,
                "body must schedule continuation"
            );
            let before = polls.load(Ordering::SeqCst);
            assert!((1..=DOWNLOAD_BODY_STEPS_PER_POLL).contains(&before));
            milliseconds.store(50, Ordering::SeqCst);
            assert!(matches!(
                operation.as_mut().poll(&mut context),
                Poll::Ready(Err(SearchError::ModelLoadFailed { .. }))
            ));
            assert_eq!(polls.load(Ordering::SeqCst), before);
            assert_eq!(progress.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn model_progress_cannot_extend_the_callers_deadline() {
        run_test_with_cx(|_| async move {
            let budget = asupersync::Budget::INFINITE
                .tightened_by_timeout(asupersync::time::wall_now(), Duration::from_secs(3_600));
            let cx = Cx::for_testing_with_budget(budget);
            let elapsed_seconds = AtomicU64::new(0);
            let progress = AtomicU64::new(0);
            let dropped = Arc::new(AtomicBool::new(false));
            let mut operation = Box::pin(run_download_io_with_elapsed(
                &cx,
                TEST_URL,
                Duration::from_secs(7_200),
                &progress,
                PendingWork(Arc::clone(&dropped)),
                || Duration::from_secs(elapsed_seconds.load(Ordering::SeqCst)),
            ));
            let mut context = Context::from_waker(Waker::noop());
            assert!(operation.as_mut().poll(&mut context).is_pending());
            elapsed_seconds.store(1, Ordering::SeqCst);
            progress.store(1_000_000_000, Ordering::SeqCst);
            assert!(operation.as_mut().poll(&mut context).is_pending());
            elapsed_seconds.store(3_601, Ordering::SeqCst);
            progress.store(3_601_000_000_000, Ordering::SeqCst);
            assert!(matches!(
                operation.as_mut().poll(&mut context),
                Poll::Ready(Err(SearchError::ModelLoadFailed { .. }))
            ));
            assert!(dropped.load(Ordering::SeqCst));
        });
    }

    #[test]
    fn model_progress_cannot_extend_a_shorter_ambient_deadline() {
        run_test_with_cx(|cx| async move {
            let ambient = Cx::for_testing_with_budget(
                asupersync::Budget::INFINITE
                    .tightened_by_timeout(asupersync::time::wall_now(), Duration::from_secs(3_600)),
            );
            let _ambient = Cx::set_current(Some(ambient));
            let elapsed_seconds = AtomicU64::new(0);
            let progress = AtomicU64::new(0);
            let dropped = Arc::new(AtomicBool::new(false));
            let mut operation = Box::pin(run_download_io_with_elapsed(
                &cx,
                TEST_URL,
                Duration::from_secs(7_200),
                &progress,
                PendingWork(Arc::clone(&dropped)),
                || Duration::from_secs(elapsed_seconds.load(Ordering::SeqCst)),
            ));
            let mut context = Context::from_waker(Waker::noop());
            assert!(operation.as_mut().poll(&mut context).is_pending());
            elapsed_seconds.store(3_601, Ordering::SeqCst);
            progress.store(3_601_000_000_000, Ordering::SeqCst);
            assert!(matches!(
                operation.as_mut().poll(&mut context),
                Poll::Ready(Err(SearchError::ModelLoadFailed { .. }))
            ));
            assert!(dropped.load(Ordering::SeqCst));
            assert!(!cx.is_cancel_requested());
        });
    }
}

#[cfg(test)]
mod proxy_transport_tests {
    use super::*;
    use asupersync::test_utils::{assert_completes_within, run_test_with_cx};
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpListener, TcpStream};
    use std::thread;

    const PEER_TIMEOUT: Duration = Duration::from_secs(10);
    const MODEL_BYTES: &[u8] = b"verified model bytes from a loopback proxy";
    const PROXY_USERNAME: &str = "proxy-canary-user";
    const PROXY_PASSWORD: &str = "proxy-canary-password";
    const PROXY_BASIC_AUTH: &str = "Basic cHJveHktY2FuYXJ5LXVzZXI6cHJveHktY2FuYXJ5LXBhc3N3b3Jk";

    /// Every accept/read/write has a bound so a routing regression cannot
    /// leave a fixture thread waiting forever for a connection or handshake.
    fn spawn_proxy<T: Send + 'static>(
        serve: impl FnOnce(TcpStream) -> T + Send + 'static,
    ) -> (SocketAddr, thread::JoinHandle<T>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let peer = thread::spawn(move || {
            let started = Instant::now();
            let stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(
                            started.elapsed() < PEER_TIMEOUT,
                            "proxy was never contacted"
                        );
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("proxy accept failed: {error}"),
                }
            };
            stream.set_nonblocking(false).unwrap();
            stream.set_read_timeout(Some(PEER_TIMEOUT)).unwrap();
            stream.set_write_timeout(Some(PEER_TIMEOUT)).unwrap();
            serve(stream)
        });
        (address, peer)
    }

    fn read_request(stream: &mut TcpStream) -> String {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        while !request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
            let count = stream.read(&mut buffer).unwrap();
            assert!(count > 0, "proxy saw EOF before complete request headers");
            request.extend_from_slice(&buffer[..count]);
            assert!(
                request.len() <= 16 * 1024,
                "unbounded proxy request headers"
            );
        }
        String::from_utf8(request).unwrap()
    }

    fn request_header<'a>(request: &'a str, name: &str) -> Option<&'a str> {
        request.lines().find_map(|line| {
            let (header, value) = line.split_once(':')?;
            header.eq_ignore_ascii_case(name).then(|| value.trim())
        })
    }

    fn download_once(url: &str, proxy_url: String, destination: &Path) -> SearchResult<()> {
        download_configured(
            url,
            DownloadConfig {
                max_retries: 0,
                proxy: ProxySetting::Url(proxy_url),
                ..DownloadConfig::default()
            },
            destination,
            |_| {},
            &|_| {},
        )
    }

    fn download_configured(
        url: &str,
        config: DownloadConfig,
        destination: &Path,
        on_started: impl FnOnce(&Cx),
        on_progress: &(impl Fn(&DownloadProgress) + Send + Sync),
    ) -> SearchResult<()> {
        let file = ModelFile {
            name: "model.onnx".to_owned(),
            sha256: sha256_hex(MODEL_BYTES),
            size: u64::try_from(MODEL_BYTES.len()).unwrap(),
            url: None,
        };
        let downloader = ModelDownloader::new(config);
        let mut result = None;
        let result_slot = &mut result;
        run_test_with_cx(|cx| async move {
            on_started(&cx);
            *result_slot = Some(
                assert_completes_within(PEER_TIMEOUT, "proxied model download", || {
                    Box::pin(downloader.download_single_file(
                        &cx,
                        url,
                        destination,
                        &file,
                        0,
                        1,
                        on_progress,
                    ))
                })
                .await,
            );
        });
        result.unwrap()
    }

    fn read_socks_string(stream: &mut TcpStream) -> Vec<u8> {
        let mut length = [0_u8; 1];
        stream.read_exact(&mut length).unwrap();
        let mut value = vec![0_u8; usize::from(length[0])];
        stream.read_exact(&mut value).unwrap();
        value
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum StallPoint {
        Connect,
        SocksGreeting,
        Headers,
        Body,
        DirectBody,
    }

    fn assert_stalled_download_terminates(point: StallPoint, cancel: bool) {
        let body_stalls = matches!(point, StallPoint::Body | StallPoint::DirectBody);
        let (context_sender, context_receiver) = std::sync::mpsc::channel::<Cx>();
        let (progress_sender, progress_receiver) = std::sync::mpsc::channel::<()>();
        let (address, peer) = spawn_proxy(move |mut stream| {
            if point == StallPoint::SocksGreeting {
                let mut greeting = [0_u8; 2];
                stream.read_exact(&mut greeting).unwrap();
                assert_eq!(greeting[0], 5);
                let mut methods = vec![0_u8; usize::from(greeting[1])];
                stream.read_exact(&mut methods).unwrap();
            } else {
                let request = read_request(&mut stream);
                assert!(request.starts_with(if point == StallPoint::Connect {
                    "CONNECT model-host.invalid:443 "
                } else if point == StallPoint::DirectBody {
                    "GET /model.onnx"
                } else {
                    "GET http://model-host.invalid/"
                }));
                if body_stalls {
                    write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                        MODEL_BYTES.len()
                    )
                    .unwrap();
                    stream.write_all(&MODEL_BYTES[..1]).unwrap();
                    stream.flush().unwrap();
                    // Ensure acquisition has consumed and staged a byte before
                    // cancellation. The peer stays open and never releases it.
                    progress_receiver.recv_timeout(PEER_TIMEOUT).unwrap();
                }
            }
            if cancel {
                context_receiver
                    .recv_timeout(PEER_TIMEOUT)
                    .unwrap()
                    .cancel_with(
                        asupersync::CancelKind::User,
                        Some("cancel stalled transfer"),
                    );
            }
            let mut byte = [0_u8; 1];
            match stream.read(&mut byte) {
                Ok(0) => {}
                Err(error) if error.kind() == ErrorKind::ConnectionReset => {}
                other => panic!("download did not release its stalled connection: {other:?}"),
            }
        });
        let (target_scheme, proxy_scheme) = match point {
            StallPoint::Connect => ("https", "http"),
            StallPoint::SocksGreeting => ("http", "socks5h"),
            StallPoint::Headers | StallPoint::Body | StallPoint::DirectBody => ("http", "http"),
        };
        let budget = if cancel {
            PEER_TIMEOUT
        } else {
            Duration::from_millis(200)
        };
        let config = DownloadConfig {
            max_retries: 0,
            request_timeout: if body_stalls {
                Duration::from_secs(5)
            } else {
                budget
            },
            idle_timeout: budget,
            proxy: if point == StallPoint::DirectBody {
                ProxySetting::Disabled
            } else {
                ProxySetting::Url(format!(
                    "{proxy_scheme}://{PROXY_USERNAME}:{PROXY_PASSWORD}@{address}"
                ))
            },
            ..DownloadConfig::default()
        };
        let destination_dir = tempfile::tempdir().unwrap();
        let destination = destination_dir.path().join("model.onnx");
        let previous = b"previous verified generation";
        std::fs::write(&destination, previous).unwrap();
        let url = if point == StallPoint::DirectBody {
            format!("http://{address}/model.onnx?token=origin-query-canary")
        } else {
            format!("{target_scheme}://model-host.invalid/model.onnx?token=origin-query-canary")
        };
        let error = download_configured(
            &url,
            config,
            &destination,
            |cx| context_sender.send(cx.clone()).unwrap(),
            &|progress| {
                if progress.bytes_downloaded > 0 {
                    let _ = progress_sender.send(());
                }
            },
        )
        .unwrap_err();
        peer.join().unwrap();
        if cancel {
            assert!(matches!(error, SearchError::Cancelled { .. }));
        } else {
            assert!(error.to_string().contains("request-deadline-exceeded"));
        }
        let rendered = format!("{error}\n{error:?}");
        assert!(!rendered.contains(PROXY_USERNAME));
        assert!(!rendered.contains(PROXY_PASSWORD));
        assert!(!rendered.contains("origin-query-canary"));
        assert_eq!(std::fs::read(&destination).unwrap(), previous);
        assert!(!destination.with_extension("tmp").exists());
    }

    #[test]
    fn stalled_connect_socks_headers_and_body_have_finite_deadlines() {
        for point in [
            StallPoint::Connect,
            StallPoint::SocksGreeting,
            StallPoint::Headers,
            StallPoint::Body,
            StallPoint::DirectBody,
        ] {
            assert_stalled_download_terminates(point, false);
        }
    }

    #[test]
    fn cancelled_connect_socks_headers_and_body_close_without_peer_activity() {
        for point in [
            StallPoint::Connect,
            StallPoint::SocksGreeting,
            StallPoint::Headers,
            StallPoint::Body,
            StallPoint::DirectBody,
        ] {
            assert_stalled_download_terminates(point, true);
        }
    }

    #[test]
    fn progressing_model_download_outlives_request_and_idle_timeout() {
        let (address, peer) = spawn_proxy(|mut stream| {
            read_request(&mut stream);
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                MODEL_BYTES.len()
            )
            .unwrap();
            for (index, chunk) in MODEL_BYTES.chunks(8).enumerate() {
                if index > 0 {
                    thread::sleep(Duration::from_millis(300));
                }
                stream.write_all(chunk).unwrap();
                stream.flush().unwrap();
            }
        });
        let destination_dir = tempfile::tempdir().unwrap();
        let destination = destination_dir.path().join("model.onnx");
        let started = Instant::now();
        download_configured(
            "http://model-host.invalid/model.onnx",
            DownloadConfig {
                max_retries: 0,
                request_timeout: Duration::from_secs(1),
                idle_timeout: Duration::from_secs(1),
                proxy: ProxySetting::Url(format!("http://{address}")),
                ..DownloadConfig::default()
            },
            &destination,
            |_| {},
            &|_| {},
        )
        .unwrap();
        peer.join().unwrap();
        assert!(started.elapsed() > Duration::from_secs(1));
        assert_eq!(std::fs::read(&destination).unwrap(), MODEL_BYTES);
        assert!(!destination.with_extension("tmp").exists());
    }

    fn assert_socks_download(scheme: &str, authenticated: bool) {
        let (address, peer) = spawn_proxy(move |mut stream| {
            let mut greeting = [0_u8; 2];
            stream.read_exact(&mut greeting).unwrap();
            assert_eq!(greeting[0], 5);
            let mut methods = vec![0_u8; usize::from(greeting[1])];
            stream.read_exact(&mut methods).unwrap();
            let selected_method = if authenticated { 2 } else { 0 };
            assert!(methods.contains(&selected_method));
            stream.write_all(&[5, selected_method]).unwrap();

            if authenticated {
                let mut auth_version = [0_u8; 1];
                stream.read_exact(&mut auth_version).unwrap();
                assert_eq!(auth_version, [1]);
                assert_eq!(read_socks_string(&mut stream), PROXY_USERNAME.as_bytes());
                assert_eq!(read_socks_string(&mut stream), PROXY_PASSWORD.as_bytes());
                stream.write_all(&[1, 0]).unwrap();
            }

            let mut connect = [0_u8; 4];
            stream.read_exact(&mut connect).unwrap();
            assert_eq!(
                connect,
                [5, 1, 0, 3],
                "hostname must be resolved by the proxy"
            );
            assert_eq!(read_socks_string(&mut stream), b"model-host.invalid");
            let mut port = [0_u8; 2];
            stream.read_exact(&mut port).unwrap();
            assert_eq!(u16::from_be_bytes(port), 8123);
            stream.write_all(&[5, 0, 0, 1, 127, 0, 0, 1, 0, 0]).unwrap();

            let request = read_request(&mut stream);
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                MODEL_BYTES.len()
            )
            .unwrap();
            stream.write_all(MODEL_BYTES).unwrap();
            stream.flush().unwrap();
            request
        });
        let credentials = if authenticated {
            format!("{PROXY_USERNAME}:{PROXY_PASSWORD}@")
        } else {
            String::new()
        };
        let destination_dir = tempfile::tempdir().unwrap();
        let destination = destination_dir.path().join("model.onnx");
        let result = download_once(
            "http://model-host.invalid:8123/model.onnx?revision=frozen",
            format!("{scheme}://{credentials}{address}"),
            &destination,
        );
        let request = peer.join().unwrap();
        result.unwrap();
        assert!(request.starts_with("GET /model.onnx?revision=frozen HTTP/1.1\r\n"));
        assert_eq!(
            request_header(&request, "Host"),
            Some("model-host.invalid:8123")
        );
        assert_eq!(request_header(&request, "Proxy-Authorization"), None);
        assert!(!request.contains(PROXY_USERNAME));
        assert!(!request.contains(PROXY_PASSWORD));
        let downloaded = std::fs::read(&destination).unwrap();
        assert_eq!(downloaded, MODEL_BYTES);
        assert_eq!(sha256_hex(&downloaded), sha256_hex(MODEL_BYTES));
        assert!(!destination.with_extension("tmp").exists());
    }

    #[test]
    fn socks5_and_socks5h_resolve_destination_hostnames_at_the_proxy() {
        for scheme in ["socks5", "socks5h"] {
            assert_socks_download(scheme, false);
        }
    }

    #[test]
    fn socks5_authentication_is_not_forwarded_as_origin_headers() {
        for scheme in ["socks5", "socks5h"] {
            assert_socks_download(scheme, true);
        }
    }

    #[test]
    fn https_download_connect_authentication_and_refusal_are_redacted() {
        let (address, peer) = spawn_proxy(|mut stream| {
            let request = read_request(&mut stream);
            write!(
                stream,
                "HTTP/1.1 407 proxy-reason-canary-{PROXY_USERNAME}-{PROXY_PASSWORD}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            .unwrap();
            stream.flush().unwrap();
            request
        });
        let destination_dir = tempfile::tempdir().unwrap();
        let destination = destination_dir.path().join("model.onnx");
        let result = download_once(
            "https://model-host.invalid/model.onnx?token=origin-query-canary",
            format!("http://{PROXY_USERNAME}:{PROXY_PASSWORD}@{address}"),
            &destination,
        );
        let request = peer.join().unwrap();
        assert!(request.starts_with("CONNECT model-host.invalid:443 HTTP/1.1\r\n"));
        assert_eq!(
            request_header(&request, "Proxy-Authorization"),
            Some(PROXY_BASIC_AUTH)
        );
        assert!(!request.contains("origin-query-canary"));
        let error = result.unwrap_err();
        let rendered = format!("{error}\n{error:?}");
        assert!(rendered.contains("proxy-tunnel-http-407"), "{rendered}");
        for secret in [
            PROXY_USERNAME,
            PROXY_PASSWORD,
            PROXY_BASIC_AUTH,
            "proxy-reason-canary",
            "origin-query-canary",
        ] {
            assert!(!rendered.contains(secret), "{rendered}");
        }
        assert!(!destination.exists());
        assert!(!destination.with_extension("tmp").exists());
    }

    #[test]
    fn https_proxy_requires_tls_before_sending_proxy_credentials() {
        let (address, peer) = spawn_proxy(|mut stream| {
            let mut record_header = [0_u8; 5];
            stream.read_exact(&mut record_header).unwrap();
            assert_eq!(record_header[0], 0x16, "expected TLS handshake record");
            assert_eq!(record_header[1], 0x03, "expected TLS record version");
            let record_length = u16::from_be_bytes([record_header[3], record_header[4]]);
            let mut hello = vec![0_u8; usize::from(record_length)];
            stream.read_exact(&mut hello).unwrap();
            assert!(
                !hello
                    .windows(PROXY_PASSWORD.len())
                    .any(|w| w == PROXY_PASSWORD.as_bytes())
            );
            // A plaintext peer cannot impersonate an HTTPS proxy. This is a
            // handshake-failure test; it does not install a test trust root.
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
        });
        let destination_dir = tempfile::tempdir().unwrap();
        let destination = destination_dir.path().join("model.onnx");
        let result = download_once(
            "http://model-host.invalid/model.onnx",
            format!("https://{PROXY_USERNAME}:{PROXY_PASSWORD}@{address}"),
            &destination,
        );
        peer.join().unwrap();
        let error = result.unwrap_err();
        let rendered = format!("{error}\n{error:?}");
        assert!(rendered.contains("tls-handshake-failure"), "{rendered}");
        assert!(!rendered.contains(PROXY_PASSWORD), "{rendered}");
        assert!(!destination.exists());
        assert!(!destination.with_extension("tmp").exists());
    }
}
