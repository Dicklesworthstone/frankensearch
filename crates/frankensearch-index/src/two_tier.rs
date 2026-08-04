//! Two-tier index wrapper for fast and quality vector indices.
//!
//! `TwoTierIndex` provides a single object that coordinates:
//! - fast-tier retrieval from `vector.fast.idx` (or `vector.idx` fallback)
//! - optional quality-tier rescoring from `vector.quality.idx`
//! - doc-id alignment between both tiers

use std::collections::HashMap;
#[cfg(feature = "ann")]
use std::collections::HashSet;
use std::fs;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::{Path, PathBuf};
#[cfg(feature = "ann")]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicU8, Ordering as AtomicOrdering};

use frankensearch_core::config::ZeroSignalReason;
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::{SearchError, SearchResult, TwoTierConfig, VectorHit};
use tracing::{debug, info, warn};

#[cfg(all(feature = "ann", test))]
use crate::hnsw::HNSW_META_FORMAT_CURRENT;
#[cfg(feature = "ann")]
use crate::hnsw::HnswLoadDisposition;
use crate::{
    ClassifiedHits, FsviAdmissionError, FsviUpgradeRequired, FsviV2IdentityBinding, Quantization,
    SearchParams, ValidatedFsviBytes, VectorIndex, VectorMetadata, dot_product_f32_f32,
};
#[cfg(feature = "ann")]
use crate::{HNSW_DEFAULT_MAX_LAYER, HnswConfig, HnswIndex};

/// Preferred fast-tier index filename.
pub const VECTOR_INDEX_FAST_FILENAME: &str = "vector.fast.idx";
/// Optional quality-tier index filename.
pub const VECTOR_INDEX_QUALITY_FILENAME: &str = "vector.quality.idx";
/// Fallback single-tier index filename used as the fast tier when no dedicated fast file exists.
pub const VECTOR_INDEX_FALLBACK_FILENAME: &str = "vector.idx";
/// Serialized fast-tier ANN sidecar.
#[cfg(feature = "ann")]
pub const VECTOR_ANN_FAST_FILENAME: &str = "vector.fast.hnsw";
/// Serialized quality-tier ANN sidecar.
#[cfg(feature = "ann")]
pub const VECTOR_ANN_QUALITY_FILENAME: &str = "vector.quality.hnsw";

/// Explicit filesystem layout for opening a [`TwoTierIndex`].
///
/// Use this when a consumer owns the index naming convention instead of using
/// the default `vector.fast.idx` / `vector.quality.idx` layout. Every configured
/// role must resolve to a distinct artifact. When the `ann` feature is enabled,
/// omitting an ANN sidecar path explicitly disables ANN for that tier even if
/// the configured record-count threshold is met. ANN paths are writable:
/// opening an index may create or replace those sidecars. Callers must keep all
/// configured artifact directories trusted and stable for the duration of
/// [`TwoTierIndex::open_with_paths`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwoTierIndexPaths {
    fast_index: PathBuf,
    quality_index: Option<PathBuf>,
    #[cfg(feature = "ann")]
    fast_ann: Option<PathBuf>,
    #[cfg(feature = "ann")]
    quality_ann: Option<PathBuf>,
}

impl TwoTierIndexPaths {
    /// Create an explicit layout with a required fast index and no quality tier.
    #[must_use]
    pub fn new(fast_index: impl Into<PathBuf>) -> Self {
        Self {
            fast_index: fast_index.into(),
            quality_index: None,
            #[cfg(feature = "ann")]
            fast_ann: None,
            #[cfg(feature = "ann")]
            quality_ann: None,
        }
    }

    /// Add an explicit quality-tier index path.
    #[must_use]
    pub fn with_quality_index(mut self, quality_index: impl Into<PathBuf>) -> Self {
        self.quality_index = Some(quality_index.into());
        self
    }

    /// Add an explicit fast-tier ANN sidecar path.
    #[cfg(feature = "ann")]
    #[must_use]
    pub fn with_fast_ann(mut self, fast_ann: impl Into<PathBuf>) -> Self {
        self.fast_ann = Some(fast_ann.into());
        self
    }

    /// Add an explicit quality-tier ANN sidecar path.
    #[cfg(feature = "ann")]
    #[must_use]
    pub fn with_quality_ann(mut self, quality_ann: impl Into<PathBuf>) -> Self {
        self.quality_ann = Some(quality_ann.into());
        self
    }

    /// Required fast-tier FSVI path.
    #[must_use]
    pub fn fast_index(&self) -> &Path {
        &self.fast_index
    }

    /// Optional quality-tier FSVI path.
    #[must_use]
    pub fn quality_index(&self) -> Option<&Path> {
        self.quality_index.as_deref()
    }

    /// Optional fast-tier ANN sidecar path.
    #[cfg(feature = "ann")]
    #[must_use]
    pub fn fast_ann(&self) -> Option<&Path> {
        self.fast_ann.as_deref()
    }

    /// Optional quality-tier ANN sidecar path.
    #[cfg(feature = "ann")]
    #[must_use]
    pub fn quality_ann(&self) -> Option<&Path> {
        self.quality_ann.as_deref()
    }

    /// Freeze every relative artifact path against the process's current
    /// directory.
    ///
    /// This preserves the original path components (including symlinks and
    /// `..`) so the operating system, rather than lexical path rewriting,
    /// determines their meaning.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the current directory cannot be read.
    pub fn into_absolute(self) -> SearchResult<Self> {
        let current_dir = std::env::current_dir()?;
        self.into_absolute_from(&current_dir)
    }

    /// Freeze every relative artifact path against one captured absolute base.
    ///
    /// This is useful when a higher-level constructor must resolve this layout
    /// and additional paths against the exact same current-directory snapshot.
    /// As with [`Self::into_absolute`], path components are preserved for the
    /// operating system to resolve.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when `base` is not absolute.
    pub fn into_absolute_from(mut self, base: &Path) -> SearchResult<Self> {
        if !base.is_absolute() {
            return Err(SearchError::InvalidConfig {
                field: "index_paths_base".to_owned(),
                value: base.display().to_string(),
                reason: "the explicit-path resolution base must be absolute".to_owned(),
            });
        }
        self.fast_index = make_path_absolute(base, self.fast_index);
        self.quality_index = self
            .quality_index
            .map(|path| make_path_absolute(base, path));
        #[cfg(feature = "ann")]
        {
            self.fast_ann = self.fast_ann.map(|path| make_path_absolute(base, path));
            self.quality_ann = self.quality_ann.map(|path| make_path_absolute(base, path));
        }
        Ok(self)
    }
}

fn make_path_absolute(current_dir: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        current_dir.join(path)
    }
}

fn canonical_path_identity(path: &Path) -> SearchResult<PathBuf> {
    let absolute = TwoTierIndexPaths::new(path).into_absolute()?.fast_index;
    let mut cursor = absolute.as_path();
    let mut missing_components = Vec::new();

    loop {
        match fs::canonicalize(cursor) {
            Ok(mut canonical) => {
                for component in missing_components.iter().rev() {
                    canonical.push(component);
                }
                return Ok(canonical);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let Some(file_name) = cursor.file_name() else {
                    return Err(SearchError::Io(error));
                };
                missing_components.push(file_name.to_os_string());
                let Some(parent) = cursor.parent() else {
                    return Err(SearchError::Io(error));
                };
                cursor = parent;
            }
            Err(error) => return Err(SearchError::Io(error)),
        }
    }
}

fn paths_alias(left: &Path, right: &Path) -> SearchResult<bool> {
    let left_identity = canonical_path_identity(left)?;
    let right_identity = canonical_path_identity(right)?;
    if left_identity == right_identity {
        return Ok(true);
    }

    match crate::file_identity::is_same_file(left, right) {
        Ok(is_same) => Ok(is_same),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(SearchError::Io(error)),
    }
}

fn reject_final_symlink(role: &str, path: &Path) -> SearchResult<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(SearchError::InvalidConfig {
            field: "index_paths".to_owned(),
            value: format!("{role}={}", path.display()),
            reason: "configured artifact roles must not be final-component symlinks".to_owned(),
        }),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(SearchError::Io(error)),
    }
}

fn validate_index_paths(paths: &TwoTierIndexPaths) -> SearchResult<()> {
    #[cfg(feature = "ann")]
    if paths.quality_ann.is_some() && paths.quality_index.is_none() {
        return Err(SearchError::InvalidConfig {
            field: "quality_ann".to_owned(),
            value: paths
                .quality_ann
                .as_deref()
                .map_or_else(String::new, |path| path.display().to_string()),
            reason: "a quality ANN sidecar requires a quality index path".to_owned(),
        });
    }

    let mut roles = vec![("fast_index", paths.fast_index())];
    if let Some(path) = paths.quality_index() {
        roles.push(("quality_index", path));
    }
    #[cfg(feature = "ann")]
    if let Some(path) = paths.fast_ann() {
        roles.push(("fast_ann", path));
    }
    #[cfg(feature = "ann")]
    if let Some(path) = paths.quality_ann() {
        roles.push(("quality_ann", path));
    }

    for (role, path) in &roles {
        reject_final_symlink(role, path)?;
    }

    for (index, (left_role, left_path)) in roles.iter().enumerate() {
        for (right_role, right_path) in &roles[index + 1..] {
            if paths_alias(left_path, right_path)? {
                return Err(SearchError::InvalidConfig {
                    field: "index_paths".to_owned(),
                    value: format!(
                        "{left_role}={}, {right_role}={}",
                        left_path.display(),
                        right_path.display()
                    ),
                    reason: "each index and ANN role must reference a distinct artifact".to_owned(),
                });
            }
        }
    }

    Ok(())
}

#[cfg(feature = "ann")]
fn validate_ann_save_lock_identities(
    paths: &TwoTierIndexPaths,
    fast_ann_enabled: bool,
    quality_ann_enabled: bool,
) -> SearchResult<()> {
    let configured_roles = {
        let mut roles = vec![("fast_index", paths.fast_index())];
        if let Some(path) = paths.quality_index() {
            roles.push(("quality_index", path));
        }
        if let Some(path) = paths.fast_ann() {
            roles.push(("fast_ann", path));
        }
        if let Some(path) = paths.quality_ann() {
            roles.push(("quality_ann", path));
        }
        roles
    };

    let mut lock_roles = Vec::with_capacity(2);
    for (role, path, enabled) in [
        ("fast_ann_lock", paths.fast_ann(), fast_ann_enabled),
        ("quality_ann_lock", paths.quality_ann(), quality_ann_enabled),
    ] {
        let Some(path) = path.filter(|_| enabled) else {
            continue;
        };
        let lock_path = crate::hnsw::hnsw_save_lock_artifact_path(path).map_err(|error| {
            SearchError::InvalidConfig {
                field: role.to_owned(),
                value: path.display().to_string(),
                reason: error.to_string(),
            }
        })?;
        for (artifact_role, artifact_path) in &configured_roles {
            if paths_alias(&lock_path, artifact_path)? {
                return Err(SearchError::InvalidConfig {
                    field: "index_paths".to_owned(),
                    value: format!(
                        "{role}={}, {artifact_role}={}",
                        lock_path.display(),
                        artifact_path.display()
                    ),
                    reason: "an ANN save-lock artifact must not alias any configured index or ANN \
                             role"
                        .to_owned(),
                });
            }
        }
        lock_roles.push((role, lock_path));
    }

    for (index, (left_role, left_path)) in lock_roles.iter().enumerate() {
        for (right_role, right_path) in &lock_roles[index + 1..] {
            if paths_alias(left_path, right_path)? {
                return Err(SearchError::InvalidConfig {
                    field: "index_paths".to_owned(),
                    value: format!(
                        "{left_role}={}, {right_role}={}",
                        left_path.display(),
                        right_path.display()
                    ),
                    reason: "configured ANN roles resolve to the same filesystem artifact"
                        .to_owned(),
                });
            }
        }
    }

    // Materializing the persistent lock artifacts turns still-missing,
    // potentially case/normalization-equivalent ANN leaves into real files.
    // Comparing those files delegates collation to the mounted filesystem
    // instead of guessing from the operating system.
    let mut materialized = Vec::with_capacity(lock_roles.len());
    for (role, lock_path) in lock_roles {
        let identity = crate::hnsw::materialize_hnsw_save_lock_artifact(&lock_path)?;
        materialized.push((role, lock_path, identity));
    }
    for (index, (left_role, left_path, left_identity)) in materialized.iter().enumerate() {
        for (right_role, right_path, right_identity) in &materialized[index + 1..] {
            if left_identity == right_identity
                || crate::file_identity::is_same_file(left_path, right_path)
                    .map_err(SearchError::Io)?
            {
                return Err(SearchError::InvalidConfig {
                    field: "index_paths".to_owned(),
                    value: format!(
                        "{left_role}={}, {right_role}={}",
                        left_path.display(),
                        right_path.display()
                    ),
                    reason: "configured ANN roles alias under the mounted filesystem's path \
                             comparison rules"
                        .to_owned(),
                });
            }
        }
    }

    Ok(())
}

#[cfg(feature = "ann")]
fn validate_ann_persistence_paths(
    paths: &TwoTierIndexPaths,
    fast_ann_enabled: bool,
    quality_ann_enabled: bool,
) -> SearchResult<()> {
    validate_index_paths(paths)?;
    validate_ann_save_lock_identities(paths, fast_ann_enabled, quality_ann_enabled)
}

#[derive(Debug)]
enum QualityAlignment {
    None,
    Aligned,
    Mapping(Vec<Option<usize>>),
}

/// One opened tier: either a plain v1 path-opened index or a fully retained
/// sealed FSVI v2 admission owner (bd-9xuj C4-write r2).
///
/// The `AdmittedV2` variant retains the complete [`ValidatedFsviBytes`]
/// owner — the `Arc`'d byte image, the complete admission witness, and the
/// publication state — honoring the owner contract on [`ValidatedFsviBytes`]:
/// the owner is never converted into a mutable/path-opened [`VectorIndex`];
/// the tier's index is only ever borrowed from inside the retained owner.
// The owner is materially larger than a plain `VectorIndex` (it additionally
// carries the witness and the byte handle), but exactly one `TierSource`
// exists per tier per opened index, so the variant-size asymmetry buys
// capability retention for a few hundred one-time bytes.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum TierSource {
    /// Plain v1 [`VectorIndex::open`] tier (mutable mapping, WAL-bearing).
    PathOpened(VectorIndex),
    /// Sealed FSVI v2 admission owner, retained in full.
    AdmittedV2(ValidatedFsviBytes),
}

impl TierSource {
    /// Borrow the tier's index for read-only serving.
    ///
    /// For `AdmittedV2` this borrows the validated index INSIDE the retained
    /// owner (crate-internal field access); the owner itself stays sealed and
    /// is never moved out of, so its byte/witness/publication capabilities
    /// remain intact for the lifetime of the tier.
    const fn index(&self) -> &VectorIndex {
        match self {
            Self::PathOpened(index) => index,
            Self::AdmittedV2(owner) => &owner.index,
        }
    }

    /// The retained sealed admission owner, when this tier came from exact
    /// FSVI v2 admission.
    const fn admitted_owner(&self) -> Option<&ValidatedFsviBytes> {
        match self {
            Self::PathOpened(_) => None,
            Self::AdmittedV2(owner) => Some(owner),
        }
    }
}

/// Dual-index container used by progressive search orchestration.
#[derive(Debug)]
pub struct TwoTierIndex {
    fast_source: TierSource,
    quality_source: Option<TierSource>,
    #[cfg(feature = "ann")]
    fast_ann: Option<HnswIndex>,
    #[cfg(feature = "ann")]
    quality_ann: Option<HnswIndex>,
    #[cfg(feature = "ann")]
    ann_fallback_count: AtomicU64,
    /// Last state-scoped [`ZeroSignalReason`] observed on the fast tier,
    /// encoded via [`zero_signal_code`]; [`ZERO_SIGNAL_NONE`] when the last
    /// search produced hits. Availability transitions log once per state
    /// change, never per query (bd-tqhc no-warn-storm policy).
    last_zero_signal: AtomicU8,
    quality_alignment: QualityAlignment,
    config: TwoTierConfig,
    /// Lowercase hex SHA-256 fingerprint of the fast tier's embedding space,
    /// when known (bd-9xuj T2-C2). Filled from the artifact's validated FSVI
    /// v2 identity header at open, or from the builder-declared producing
    /// identity when [`TwoTierIndexBuilder::finish`] wrote this tier in the
    /// current process. `None` is the typed legacy-unidentified state (v1
    /// artifacts) — never fabricated from the header's id/revision strings.
    fast_space_fingerprint_hex: Option<String>,
    /// Quality-tier counterpart of [`Self::fast_space_fingerprint_hex`].
    quality_space_fingerprint_hex: Option<String>,
    /// Complete identity bundle of the embedder that produced the fast
    /// tier's vectors, when the builder that wrote this tier declared it
    /// (bd-9xuj T2-C2). Retained so bundle-holding seams can apply the full
    /// admission law
    /// ([`frankensearch_core::BoundQueryEmbedding::verify_producer_conformance`])
    /// rather than the fingerprint-only join. Its storage component
    /// describes the PRODUCER's output contract, not this index's persisted
    /// encoding. Process-local for v1 artifacts: a reopen from disk has no
    /// bundle to retain and stays `None`.
    fast_declared_identity: Option<EmbeddingIdentityBundleV1>,
    /// Quality-tier counterpart of [`Self::fast_declared_identity`].
    quality_declared_identity: Option<EmbeddingIdentityBundleV1>,
}

/// Sentinel for "the last fast-tier search produced hits".
const ZERO_SIGNAL_NONE: u8 = u8::MAX;

/// Stable per-variant code for the transition state machine.
const fn zero_signal_code(reason: ZeroSignalReason) -> u8 {
    match reason {
        ZeroSignalReason::CallerRequestedZeroK => 0,
        ZeroSignalReason::FilterEliminatedAll => 1,
        ZeroSignalReason::NonFiniteQuery => 2,
        ZeroSignalReason::ZeroNormQuery => 3,
        ZeroSignalReason::NewlyCreatedEmpty => 4,
        ZeroSignalReason::AllTombstoned => 5,
        ZeroSignalReason::WalOnlyNoLiveRecords => 6,
        ZeroSignalReason::NoUsableVectors => 7,
        ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors => 8,
    }
}

impl TwoTierIndex {
    /// Open a two-tier index from a directory.
    ///
    /// Fast index lookup order:
    /// 1. `{dir}/vector.fast.idx`
    /// 2. `{dir}/vector.idx` (fallback)
    ///
    /// Quality index (optional):
    /// - `{dir}/vector.quality.idx`
    ///
    /// # Errors
    ///
    /// Returns `SearchError::IndexCandidatesNotFound` if neither fast-tier
    /// candidate exists, and propagates fast-tier parse/corruption errors from
    /// `VectorIndex::open`. A discovered quality-tier file is optional: an
    /// unavailable or corrupt one degrades this constructor to fast-only.
    pub fn open(dir: &Path, config: TwoTierConfig) -> SearchResult<Self> {
        let fast_path = resolve_fast_path(dir)?;
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
        let mut paths = TwoTierIndexPaths::new(fast_path);
        if quality_path.exists() {
            paths = paths.with_quality_index(quality_path);
        }
        #[cfg(feature = "ann")]
        {
            paths = paths.with_fast_ann(dir.join(VECTOR_ANN_FAST_FILENAME));
            if paths.quality_index().is_some() {
                paths = paths.with_quality_ann(dir.join(VECTOR_ANN_QUALITY_FILENAME));
            }
        }
        Self::open_with_paths_inner(&paths, config, true)
    }

    /// Open a two-tier index from explicit consumer-owned paths.
    ///
    /// Unlike [`Self::open`], this constructor performs no filename discovery.
    /// A supplied quality path is required to exist; omit it to open a
    /// fast-only index. ANN sidecars are used only when explicitly configured
    /// on [`TwoTierIndexPaths`].
    ///
    /// # Errors
    ///
    /// Returns `SearchError::IndexNotFound` when an explicitly supplied index
    /// path is missing, `SearchError::InvalidConfig` when configured artifact
    /// roles alias, and propagates parse/corruption errors from
    /// `VectorIndex::open`. Relative paths are frozen against the current
    /// directory before validation and opening.
    pub fn open_with_paths(paths: &TwoTierIndexPaths, config: TwoTierConfig) -> SearchResult<Self> {
        Self::open_with_paths_inner(paths, config, false)
    }

    #[allow(clippy::too_many_lines)]
    fn open_with_paths_inner(
        paths: &TwoTierIndexPaths,
        config: TwoTierConfig,
        degrade_discovered_quality_errors: bool,
    ) -> SearchResult<Self> {
        let paths = paths.clone().into_absolute()?;
        validate_index_paths(&paths)?;
        let fast_index = VectorIndex::open(paths.fast_index())?;
        warn_if_wal_rows_replayed("fast", paths.fast_index(), &fast_index);
        let fast_source = TierSource::PathOpened(fast_index);
        let quality_source = match paths.quality_index() {
            Some(quality_path) => match VectorIndex::open(quality_path) {
                Ok(quality_index) => {
                    warn_if_wal_rows_replayed("quality", quality_path, &quality_index);
                    Some(TierSource::PathOpened(quality_index))
                }
                Err(error) if degrade_discovered_quality_errors => {
                    warn!(
                        path = %quality_path.display(),
                        ?error,
                        "discovered optional quality index is unavailable; degrading to fast-only"
                    );
                    None
                }
                Err(error) => return Err(error),
            },
            None => None,
        };
        Self::assemble_opened(fast_source, quality_source, &paths, config)
    }

    /// Open a two-tier generation whose tiers are identity-complete FSVI v2
    /// artifacts, through exact admission (bd-9xuj T2-C4-write).
    ///
    /// [`VectorIndex::open`] is strictly v1 — it rejects v2 bytes with
    /// `IndexVersionMismatch` — so v2 tiers can never be plain-opened. Each
    /// supplied tier is admitted via [`VectorIndex::open_admitted_v2`]
    /// against its caller-held [`FsviV2IdentityBinding`], and the sealed
    /// [`ValidatedFsviBytes`] owner is RETAINED IN FULL (r2 repair of the
    /// C4-write NO-GO): the `Arc`'d byte image, complete witness, and
    /// publication state stay reachable via
    /// [`Self::fast_admitted_owner`] / [`Self::quality_admitted_owner`], and
    /// the tier is served by reference from inside the owner. The resulting
    /// index reports header-ATTESTED identity:
    /// [`Self::fast_identity_is_attested`] (and the quality counterpart when
    /// a quality tier is supplied) return `true`, and the per-tier space
    /// fingerprints come from each artifact's own validated header bytes.
    ///
    /// A quality path and its binding must be supplied together: a v2 tier
    /// without a binding has no legitimate open path.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when a quality path/binding is
    /// supplied without its counterpart, typed admission failures (mapped
    /// from [`FsviAdmissionError`], naming the tier), and the same path
    /// validation errors as [`Self::open_with_paths`].
    pub fn open_admitted_v2_with_paths(
        paths: &TwoTierIndexPaths,
        config: TwoTierConfig,
        fast_binding: &FsviV2IdentityBinding,
        quality_binding: Option<&FsviV2IdentityBinding>,
    ) -> SearchResult<Self> {
        let paths = paths.clone().into_absolute()?;
        validate_index_paths(&paths)?;
        let fast_source =
            TierSource::AdmittedV2(admit_v2_tier(paths.fast_index(), fast_binding, "fast")?);
        let quality_source = match (paths.quality_index(), quality_binding) {
            (Some(path), Some(binding)) => Some(TierSource::AdmittedV2(admit_v2_tier(
                path, binding, "quality",
            )?)),
            (None, None) => None,
            (Some(path), None) => {
                return Err(SearchError::InvalidConfig {
                    field: "two_tier.quality_v2_admission".to_owned(),
                    value: path.display().to_string(),
                    reason: "a quality index path was supplied without its identity binding; \
                             v2 tiers are only opened through exact admission"
                        .to_owned(),
                });
            }
            (None, Some(_)) => {
                return Err(SearchError::InvalidConfig {
                    field: "two_tier.quality_v2_admission".to_owned(),
                    value: "<no-quality-path>".to_owned(),
                    reason: "a quality identity binding was supplied without a quality index path"
                        .to_owned(),
                });
            }
        };
        Self::assemble_opened(fast_source, quality_source, &paths, config)
    }

    /// Shared assembly over tiers that were already opened — by the plain v1
    /// [`VectorIndex::open`] path or by exact FSVI v2 admission
    /// (bd-9xuj T2-C4-write refactor). Computes quality alignment, plans ANN
    /// sidecars, and retains per-tier header identity. Behavior for the v1
    /// path is unchanged; the identity retention comment below applies to
    /// both sources. Admitted v2 sources arrive as sealed retained owners
    /// ([`TierSource::AdmittedV2`]) and are only ever borrowed here.
    #[allow(clippy::too_many_lines)]
    fn assemble_opened(
        fast_source: TierSource,
        quality_source: Option<TierSource>,
        paths: &TwoTierIndexPaths,
        config: TwoTierConfig,
    ) -> SearchResult<Self> {
        let fast_index = fast_source.index();
        let mut quality_alignment = QualityAlignment::None;

        // A fast/quality pair is only blendable when both tiers came from the
        // SAME publication: a crash between the two per-tier installs leaves
        // a mixed-generation pair whose nonces disagree, and blending it
        // silently launders vectors from different corpus states (bd-miio8).
        // Legacy pre-nonce pairs read 0 == 0 and remain accepted. The
        // degraded pair drops the quality SOURCE (not just its borrow), so a
        // mixed-generation quality tier can never serve.
        let quality_source = quality_source.filter(|source| {
            let quality_nonce = source.index().publication_nonce();
            if quality_nonce == fast_index.publication_nonce() {
                true
            } else {
                warn!(
                    fast_nonce = fast_index.publication_nonce(),
                    quality_nonce,
                    "fast/quality publication identities disagree (mixed-generation pair, \
                     likely a crash between tier installs); degrading to fast-only"
                );
                false
            }
        });

        let quality_index = if let Some(quality) = quality_source.as_ref().map(TierSource::index) {
            if quality.record_count() != fast_index.record_count() {
                warn!(
                    fast_records = fast_index.record_count(),
                    quality_records = quality.record_count(),
                    "fast and quality index record counts differ; using doc-id alignment"
                );
            }

            quality_alignment = QualityAlignment::Aligned;
            let mut f_idx = 0;
            let mut q_idx = 0;
            let f_count = fast_index.record_count();
            let q_count = quality.record_count();
            let mut unmatched_quality_docs = 0;

            // Switch `quality_alignment` from `Aligned` to `Mapping` if not already.
            let ensure_mapping = |quality_alignment: &mut QualityAlignment,
                                  current_f_idx: usize| {
                if matches!(quality_alignment, QualityAlignment::Aligned) {
                    let map = (0..current_f_idx).map(Some).collect();
                    *quality_alignment = QualityAlignment::Mapping(map);
                }
            };

            while f_idx < f_count && q_idx < q_count {
                let f_rec = fast_index.record_at(f_idx)?;
                let q_rec = quality.record_at(q_idx)?;

                if crate::is_tombstoned_flags(f_rec.flags) {
                    ensure_mapping(&mut quality_alignment, f_idx);
                    if let QualityAlignment::Mapping(vec) = &mut quality_alignment {
                        vec.push(None);
                    }
                    f_idx += 1;
                    continue;
                }
                if crate::is_tombstoned_flags(q_rec.flags) {
                    q_idx += 1;
                    continue;
                }

                // If indices diverged, we must be in mapping mode
                if matches!(quality_alignment, QualityAlignment::Aligned) && f_idx != q_idx {
                    ensure_mapping(&mut quality_alignment, f_idx);
                }

                match f_rec.doc_id_hash.cmp(&q_rec.doc_id_hash) {
                    std::cmp::Ordering::Less => {
                        // Fast has doc, Quality missing
                        ensure_mapping(&mut quality_alignment, f_idx);
                        if let QualityAlignment::Mapping(vec) = &mut quality_alignment {
                            vec.push(None);
                        }
                        f_idx += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        unmatched_quality_docs += 1;
                        q_idx += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        let f_id = fast_index.doc_id_at(f_idx)?;
                        let q_id = quality.doc_id_at(q_idx)?;

                        match f_id.cmp(q_id) {
                            std::cmp::Ordering::Equal => {
                                if let QualityAlignment::Mapping(vec) = &mut quality_alignment {
                                    vec.push(Some(q_idx));
                                }
                                f_idx += 1;
                                q_idx += 1;
                            }
                            std::cmp::Ordering::Less => {
                                ensure_mapping(&mut quality_alignment, f_idx);
                                if let QualityAlignment::Mapping(vec) = &mut quality_alignment {
                                    vec.push(None);
                                }
                                f_idx += 1;
                            }
                            std::cmp::Ordering::Greater => {
                                unmatched_quality_docs += 1;
                                q_idx += 1;
                            }
                        }
                    }
                }
            }

            // Handle trailing fast docs
            if f_idx < f_count {
                ensure_mapping(&mut quality_alignment, f_idx);
                if let QualityAlignment::Mapping(vec) = &mut quality_alignment {
                    while vec.len() < f_count {
                        vec.push(None);
                    }
                }
            }

            while q_idx < q_count {
                let q_rec = quality.record_at(q_idx)?;
                if !crate::is_tombstoned_flags(q_rec.flags) {
                    unmatched_quality_docs += 1;
                }
                q_idx += 1;
            }

            if unmatched_quality_docs > 0 {
                warn!(
                    unmatched_quality_docs,
                    "quality index contains doc_ids that are not present in fast index"
                );
            }

            Some(quality)
        } else {
            None
        };

        #[cfg(feature = "ann")]
        let fast_ann_plan = paths.fast_ann.as_deref().and_then(|fast_ann_path| {
            plan_load_or_build_ann(
                fast_index,
                fast_ann_path,
                config.hnsw_threshold,
                &config,
                "fast",
            )
        });

        #[cfg(feature = "ann")]
        let quality_ann_plan = quality_index.and_then(|quality_index| {
            paths.quality_ann.as_deref().and_then(|quality_ann_path| {
                plan_load_or_build_ann(
                    quality_index,
                    quality_ann_path,
                    config.hnsw_threshold,
                    &config,
                    "quality",
                )
            })
        });

        #[cfg(feature = "ann")]
        {
            let fast_needs_persistence = fast_ann_plan
                .as_ref()
                .is_some_and(AnnOpenPlan::needs_persistence);
            let quality_needs_persistence = quality_ann_plan
                .as_ref()
                .is_some_and(AnnOpenPlan::needs_persistence);
            let persistence_prepared = if fast_needs_persistence || quality_needs_persistence {
                match validate_ann_persistence_paths(
                    paths,
                    fast_needs_persistence,
                    quality_needs_persistence,
                ) {
                    Ok(()) => true,
                    Err(error @ SearchError::InvalidConfig { .. }) => return Err(error),
                    Err(error) => {
                        warn!(
                            ?error,
                            fast_needs_persistence,
                            quality_needs_persistence,
                            "failed to prepare ANN persistence identities; all rebuilt ANN tiers \
                             stay in-memory for this process and the next startup may rebuild them"
                        );
                        false
                    }
                }
            } else {
                true
            };
            if persistence_prepared {
                if let (Some(plan), Some(path)) = (fast_ann_plan.as_ref(), paths.fast_ann()) {
                    persist_ann_plan(
                        plan,
                        path,
                        "fast",
                        paths,
                        fast_needs_persistence,
                        quality_needs_persistence,
                    );
                }
                if let (Some(plan), Some(path)) = (quality_ann_plan.as_ref(), paths.quality_ann()) {
                    persist_ann_plan(
                        plan,
                        path,
                        "quality",
                        paths,
                        fast_needs_persistence,
                        quality_needs_persistence,
                    );
                }
            }
        }

        #[cfg(feature = "ann")]
        let fast_ann = fast_ann_plan.map(|plan| plan.index);

        #[cfg(feature = "ann")]
        let quality_ann = quality_ann_plan.map(|plan| plan.index);

        #[cfg(feature = "ann")]
        debug!(
            fast_path = %paths.fast_index().display(),
            quality_path = ?paths.quality_index(),
            quality_available = quality_index.is_some(),
            fast_ann = fast_ann.is_some(),
            quality_ann = quality_ann.is_some(),
            doc_count = fast_index.record_count(),
            "opened two-tier index"
        );

        #[cfg(not(feature = "ann"))]
        debug!(
            fast_path = %paths.fast_index().display(),
            quality_path = ?paths.quality_index(),
            quality_available = quality_index.is_some(),
            doc_count = fast_index.record_count(),
            "opened two-tier index"
        );

        // Retain each tier's embedding-space identity from its artifact
        // header when the artifact carries one (bd-9xuj T2-C2). Legacy v1
        // artifacts have no identity header — `identity_v2()` is
        // structurally `None` there — and that absence is kept as the typed
        // legacy-unidentified state, never synthesized from id strings.
        let fast_space_fingerprint_hex = fast_index
            .identity_v2()
            .map(|identity| crate::fingerprint_hex(&identity.space_fingerprint));
        let quality_space_fingerprint_hex = quality_index
            .and_then(VectorIndex::identity_v2)
            .map(|identity| crate::fingerprint_hex(&identity.space_fingerprint));

        Ok(Self {
            fast_source,
            quality_source,
            #[cfg(feature = "ann")]
            fast_ann,
            #[cfg(feature = "ann")]
            quality_ann,
            #[cfg(feature = "ann")]
            ann_fallback_count: AtomicU64::new(0),
            last_zero_signal: AtomicU8::new(ZERO_SIGNAL_NONE),
            quality_alignment,
            config,
            fast_space_fingerprint_hex,
            quality_space_fingerprint_hex,
            fast_declared_identity: None,
            quality_declared_identity: None,
        })
    }

    /// Create a builder for a new two-tier index directory.
    ///
    /// The builder buffers added vectors and writes FSVI files on `finish()`.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` if the directory cannot be created.
    pub fn create(dir: &Path, config: TwoTierConfig) -> SearchResult<TwoTierIndexBuilder> {
        fs::create_dir_all(dir)?;
        Ok(TwoTierIndexBuilder::new(dir.to_path_buf(), config))
    }

    /// Borrow the fast tier's index for read paths.
    const fn fast_tier(&self) -> &VectorIndex {
        self.fast_source.index()
    }

    /// Borrow the quality tier's index for read paths, when loaded.
    fn quality_tier(&self) -> Option<&VectorIndex> {
        self.quality_source.as_ref().map(TierSource::index)
    }

    /// Test-only mutable access to a PATH-OPENED fast tier (WAL injection).
    ///
    /// `None` for an admitted v2 tier: sealed owners are never mutably
    /// exposed, in tests or otherwise.
    #[cfg(test)]
    fn fast_tier_mut_for_test(&mut self) -> Option<&mut VectorIndex> {
        match &mut self.fast_source {
            TierSource::PathOpened(index) => Some(index),
            TierSource::AdmittedV2(_) => None,
        }
    }

    /// The retained sealed admission owner of the fast tier, when this index
    /// was opened through exact FSVI v2 admission
    /// ([`Self::open_admitted_v2_with_paths`]).
    ///
    /// The owner carries the `Arc`'d validated byte image, the complete
    /// [`crate::FsviV2Witness`], and the publication state
    /// ([`ValidatedFsviBytes::published_wal_absent`]). Replacing, renaming,
    /// or mutating the source pathname after admission cannot alter what the
    /// retained owner serves. `None` for plain v1 path-opened tiers.
    #[must_use]
    pub const fn fast_admitted_owner(&self) -> Option<&ValidatedFsviBytes> {
        self.fast_source.admitted_owner()
    }

    /// Quality-tier counterpart of [`Self::fast_admitted_owner`].
    ///
    /// `None` both when no quality tier is loaded and when the loaded one is
    /// a plain v1 path-opened tier.
    #[must_use]
    pub fn quality_admitted_owner(&self) -> Option<&ValidatedFsviBytes> {
        self.quality_source
            .as_ref()
            .and_then(TierSource::admitted_owner)
    }

    /// Search the fast tier only.
    ///
    /// # Errors
    ///
    /// Propagates errors from source-aware HNSW candidate retrieval (when ANN
    /// is selected) or `VectorIndex::search_top_k` (brute-force fallback).
    pub fn search_fast(&self, query_vec: &[f32], k: usize) -> SearchResult<Vec<VectorHit>> {
        self.search_fast_with_params(query_vec, k, None)
    }

    /// Search the fast tier with optional brute-force parallelism overrides.
    ///
    /// When ANN is active for the fast tier, ANN continues to own candidate
    /// retrieval and `params` is ignored. When brute-force search is used,
    /// `params` controls the Rayon threshold/chunking path.
    ///
    /// # Errors
    ///
    /// Propagates errors from source-aware HNSW candidate retrieval (when ANN
    /// is selected) or `VectorIndex::search_top_k_with_params` / `search_top_k`.
    pub fn search_fast_with_params(
        &self,
        query_vec: &[f32],
        k: usize,
        params: Option<SearchParams>,
    ) -> SearchResult<Vec<VectorHit>> {
        #[cfg(feature = "ann")]
        if let Some(ann) = &self.fast_ann {
            // Source-aware HNSW filters tombstones and exact-repairs native
            // underfill. Request exactly k physical main candidates so
            // duplicate document IDs consume the same pre-resolution slots as
            // canonical VectorIndex search; overfetching would incorrectly
            // backfill lower-ranked public IDs.
            let (hits, stats) = ann.knn_search_raw_with_stats_against(
                self.fast_tier(),
                query_vec,
                k,
                self.config.hnsw_ef_search,
            )?;
            if let Some(reason) = stats.fallback_reason {
                let fallback_count = self
                    .ann_fallback_count
                    .fetch_add(1, AtomicOrdering::Relaxed)
                    .saturating_add(1);
                warn!(
                    ?reason,
                    fallback_count,
                    index_size = stats.index_size,
                    k_requested = stats.k_requested,
                    ef_search = stats.ef_search,
                    search_time_us = stats.search_time_us,
                    "fast-tier ANN degraded to exact search"
                );
            }

            return self.resolve_fast_ann_and_wal(hits, query_vec, k);
        }
        let mrl_config = crate::mrl::MrlConfig {
            search_dims: self.config.mrl_search_dims,
            rescore_dims: 0,
            rescore_top_k: self.config.mrl_rescore_top_k,
        };

        if mrl_config.search_dims > 0 && mrl_config.search_dims < self.fast_tier().dimension() {
            return self.fast_tier().mrl_search(query_vec, k, &mrl_config, None);
        }

        // Default (no explicit exact-scan params): the fast tier is a reranked candidate generator
        // (its hits feed RRF + graph + phase-1 corrections downstream), so use the lossless int8
        // two-pass rather than the exact f16 scan — matching the sync searcher's fast tier
        // (`sync_searcher::search_fast_hits`) and strictly MORE accurate than the ANN path this same
        // method uses when `ann` is enabled. Proven candidate-lossless (`int8_vs_f16_fast_ab`:
        // set-recall@10 = 1.0000, exact-order-match 32/32) and ~1.3-1.43x faster on the fast-tier
        // scan. int8 two-pass itself falls back to the exact scan for F32 slabs, WAL-dirty indexes,
        // or k == 0, so those paths stay bit-identical. Explicit `params` still honour the exact scan
        // + parallelism configuration.
        const FAST_TIER_MULT: usize = 3;
        params.map_or_else(
            || {
                self.fast_tier()
                    .search_top_k_int8_two_pass(query_vec, k, FAST_TIER_MULT)
            },
            |params| {
                self.fast_tier()
                    .search_top_k_with_params(query_vec, k, None, params)
            },
        )
    }

    /// Search the fast tier with typed zero-signal classification.
    ///
    /// Behaves like [`Self::search_fast`] with the fail-closed differences
    /// of the classified exact lane
    /// ([`VectorIndex::search_top_k_classified`]): non-finite queries are
    /// rejected instead of silently scoring garbage, and an empty result
    /// always carries a typed [`ZeroSignalReason`]. Availability transitions
    /// are logged once per state change, never per query.
    ///
    /// # Errors
    ///
    /// Everything [`Self::search_fast`] returns, plus
    /// [`SearchError::InvalidConfig`] for non-finite query vectors.
    pub fn search_fast_classified(
        &self,
        query_vec: &[f32],
        k: usize,
    ) -> SearchResult<ClassifiedHits> {
        if query_vec.len() != self.fast_tier().dimension() {
            return Err(SearchError::DimensionMismatch {
                expected: self.fast_tier().dimension(),
                found: query_vec.len(),
            });
        }
        if k == 0 {
            let classified = ClassifiedHits::empty(ZeroSignalReason::CallerRequestedZeroK);
            self.note_zero_signal(classified.zero_signal);
            return Ok(classified);
        }
        if query_vec.iter().any(|value| !value.is_finite()) {
            return Err(SearchError::InvalidConfig {
                field: "query".to_owned(),
                value: "<contains non-finite values>".to_owned(),
                reason: "query vector must be finite".to_owned(),
            });
        }
        if query_vec.iter().all(|&value| value == 0.0) {
            let classified = ClassifiedHits::empty(ZeroSignalReason::ZeroNormQuery);
            self.note_zero_signal(classified.zero_signal);
            return Ok(classified);
        }
        let hits = self.search_fast(query_vec, k)?;
        let zero_signal = hits.is_empty().then(|| self.classify_fast_empty());
        self.note_zero_signal(zero_signal);
        Ok(ClassifiedHits { hits, zero_signal })
    }

    /// Classify why a well-formed fast-tier search returned nothing.
    ///
    /// Mirrors [`VectorIndex::classify_empty_result`], with one refinement:
    /// when ANN owns candidate retrieval and the census still shows usable
    /// live vectors, the empty result is the ANN-availability anomaly rather
    /// than a data problem.
    fn classify_fast_empty(&self) -> ZeroSignalReason {
        let state = self.fast_tier().zero_signal_state();
        if let Some(reason) = state.state_reason() {
            return reason;
        }
        if state.is_wal_only() {
            return ZeroSignalReason::WalOnlyNoLiveRecords;
        }
        #[cfg(feature = "ann")]
        if self.fast_ann.is_some() {
            return ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors;
        }
        ZeroSignalReason::NoUsableVectors
    }

    /// Record a zero-signal observation and log state transitions exactly
    /// once.
    ///
    /// Request-scoped reasons (k = 0, filters, query vector defects) are
    /// per-request events: they log at debug and never touch the state
    /// machine, so an interleaved k = 0 query cannot fabricate a recovery
    /// or a re-degradation. State-scoped reasons participate in the
    /// transition machine: availability failures warn once per transition,
    /// benign states log at info once per transition, and the first
    /// hit-producing search after any state-scoped emptiness logs recovery.
    fn note_zero_signal(&self, reason: Option<ZeroSignalReason>) {
        match reason {
            Some(request_scoped) if request_scoped.is_request_scoped() => {
                debug!(
                    reason_code = request_scoped.reason_code(),
                    "fast-tier search returned empty: request-scoped zero-signal"
                );
            }
            Some(state_scoped) => {
                let code = zero_signal_code(state_scoped);
                let previous = self.last_zero_signal.swap(code, AtomicOrdering::Relaxed);
                if previous == code {
                    return;
                }
                if state_scoped.is_availability_failure() {
                    warn!(
                        reason_code = state_scoped.reason_code(),
                        reason = %state_scoped,
                        "fast-tier semantic lane is unusable"
                    );
                } else {
                    info!(
                        reason_code = state_scoped.reason_code(),
                        reason = %state_scoped,
                        "fast-tier semantic lane has no signal"
                    );
                }
            }
            None => {
                let previous = self
                    .last_zero_signal
                    .swap(ZERO_SIGNAL_NONE, AtomicOrdering::Relaxed);
                if previous != ZERO_SIGNAL_NONE {
                    info!("fast-tier semantic lane recovered: search produced hits");
                }
            }
        }
    }

    #[cfg(feature = "ann")]
    fn resolve_fast_ann_and_wal(
        &self,
        ann_hits: Vec<VectorHit>,
        query_vec: &[f32],
        k: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        // HNSW maps compact graph ids to canonical physical VectorIndex rows.
        // Keep that identity: resolving through doc_id would collapse distinct
        // physical rows before the canonical result resolver gets to rank them.
        let base_index = self.fast_tier().record_count();
        let mut hits = Vec::with_capacity(
            ann_hits
                .len()
                .saturating_add(self.fast_tier().wal_entries.len()),
        );
        for hit in ann_hits {
            if let Ok(position) = usize::try_from(hit.index)
                && position < base_index
                && !self.fast_tier().is_deleted(position)
            {
                hits.push(hit);
            }
        }

        // Resident WAL entries are not in the native graph. Preserve canonical
        // `VectorIndex::resolve_sorted_entries` semantics exactly:
        //
        // 1. rank the physical main+WAL candidate pool;
        // 2. retain only the physical top-k;
        // 3. suppress a main hit when any resident WAL version of its doc_id
        //    exists (including a non-finite/corrupt version);
        // 4. deduplicate public doc_ids best-first.
        //
        // Suppression must happen after physical top-k selection. Doing it
        // earlier backfills a lower-ranked hit that canonical exact search
        // deliberately does not return.
        for (wal_index, entry) in self.fast_tier().wal_entries.iter().enumerate() {
            let score = dot_product_f32_f32(&entry.embedding, query_vec)?;
            if !score.is_finite() {
                continue;
            }
            let virtual_index =
                base_index
                    .checked_add(wal_index)
                    .ok_or_else(|| SearchError::InvalidConfig {
                        field: "index".to_owned(),
                        value: wal_index.to_string(),
                        reason: "WAL virtual index overflow".to_owned(),
                    })?;
            let index = u32::try_from(virtual_index).map_err(|_| SearchError::InvalidConfig {
                field: "index".to_owned(),
                value: virtual_index.to_string(),
                reason: "WAL entry index exceeds u32 range".to_owned(),
            })?;
            hits.push(VectorHit {
                index,
                score,
                doc_id: entry.doc_id.as_str().into(),
            });
        }

        let by_score_index = |left: &VectorHit, right: &VectorHit| {
            left.cmp_by_score(right)
                .then_with(|| left.index.cmp(&right.index))
        };
        const SELECT_NTH_MIN: usize = 256;
        if k < hits.len() && hits.len() >= SELECT_NTH_MIN {
            hits.select_nth_unstable_by(k, by_score_index);
            hits.truncate(k);
            hits.sort_unstable_by(by_score_index);
        } else {
            hits.sort_by(by_score_index);
            hits.truncate(k);
        }

        let wal_doc_ids: HashSet<&str> = self
            .fast_index
            .wal_entries
            .iter()
            .map(|entry| entry.doc_id.as_str())
            .collect();
        let mut seen_doc_ids = HashSet::with_capacity(hits.len());
        hits.retain(|hit| {
            let is_main = usize::try_from(hit.index).is_ok_and(|index| index < base_index);
            if is_main && wal_doc_ids.contains(hit.doc_id.as_str()) {
                return false;
            }
            seen_doc_ids.insert(hit.doc_id.clone())
        });
        Ok(hits)
    }

    /// Compute quality-tier scores for fast-index document positions.
    ///
    /// Missing quality entries produce `None`, allowing downstream blending
    /// to use fast-only scores without penalizing documents that lack quality
    /// vectors.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if `query_vec` does not match
    /// the quality index dimensionality (when a quality index is present), and
    /// propagates decode/corruption errors from the quality index.
    pub fn quality_scores_for_hits(
        &self,
        query_vec: &[f32],
        hits: &[VectorHit],
    ) -> SearchResult<Vec<Option<f32>>> {
        let Some(quality_index) = self.quality_tier() else {
            return Ok(vec![None; hits.len()]);
        };

        if query_vec.len() != quality_index.dimension() {
            return Err(SearchError::DimensionMismatch {
                expected: quality_index.dimension(),
                found: query_vec.len(),
            });
        }

        // Precompute `doc_id → latest WAL entry index` once, instead of an O(W)
        // reverse linear scan of the quality WAL **per hit** (was O(hits·W); the
        // exact scan in `search.rs` already precomputes a WAL map). `&str` keys
        // borrow the WAL doc_ids (no clone); forward insert keeps the latest
        // (highest-index) entry, matching the prior rev-scan-first-match. Empty
        // WAL → non-allocating empty map, so the common (compacted) path is neutral.
        let wal_latest: HashMap<&str, usize> = {
            let mut m = HashMap::with_capacity(quality_index.wal_entries.len());
            for (i, entry) in quality_index.wal_entries.iter().enumerate() {
                m.insert(entry.doc_id.as_str(), i);
            }
            m
        };

        let mut scores = Vec::with_capacity(hits.len());
        for hit in hits {
            let mut found_score = if let Some(&i) = wal_latest.get(hit.doc_id.as_str()) {
                Some(dot_product_f32_f32(
                    &quality_index.wal_entries[i].embedding,
                    query_vec,
                )?)
            } else {
                None
            };

            if found_score.is_none() {
                let fast_idx = if hit.index == u32::MAX {
                    self.fast_tier().find_index_by_doc_id(&hit.doc_id)?
                } else if (hit.index as usize) < self.fast_tier().record_count() {
                    Some(hit.index as usize)
                } else {
                    None
                };

                if let Some(idx) = fast_idx {
                    found_score =
                        self.score_quality_for_fast_index(quality_index, query_vec, idx)?;
                }
            }

            if found_score.is_none() {
                if let Some(qual_idx) = quality_index.find_index_by_doc_id(&hit.doc_id)? {
                    found_score = Some(quality_index.dot_query_at(qual_idx, query_vec)?);
                }
            }

            scores.push(found_score);
        }
        Ok(scores)
    }

    /// Returns true when a quality index was loaded.
    #[must_use]
    pub const fn has_quality_index(&self) -> bool {
        self.quality_source.is_some()
    }

    /// Returns true when fast-tier ANN is loaded/enabled.
    #[cfg(feature = "ann")]
    #[must_use]
    pub const fn has_fast_ann(&self) -> bool {
        self.fast_ann.is_some()
    }

    /// Returns true when quality-tier ANN is loaded/enabled.
    #[cfg(feature = "ann")]
    #[must_use]
    pub const fn has_quality_ann(&self) -> bool {
        self.quality_ann.is_some()
    }

    /// Number of fast-tier ANN queries that degraded to an exact scan.
    ///
    /// This monotonic counter makes persistent graph underfill observable even
    /// through search APIs that return only hits.
    #[cfg(feature = "ann")]
    #[must_use]
    pub fn ann_fallback_count(&self) -> u64 {
        self.ann_fallback_count.load(AtomicOrdering::Relaxed)
    }

    /// Number of documents in the fast tier (canonical document count).
    #[must_use]
    pub const fn doc_count(&self) -> usize {
        self.fast_tier().record_count()
    }

    /// Embedder identity recorded in the fast-tier index header.
    #[must_use]
    pub fn fast_embedder_id(&self) -> &str {
        self.fast_tier().embedder_id()
    }

    /// Embedder revision recorded in the fast-tier index header.
    #[must_use]
    pub fn fast_embedder_revision(&self) -> &str {
        self.fast_tier().embedder_revision()
    }

    /// Embedder identity recorded in the quality-tier index header, when loaded.
    #[must_use]
    pub fn quality_embedder_id(&self) -> Option<&str> {
        self.quality_tier().map(VectorIndex::embedder_id)
    }

    /// Embedder revision recorded in the quality-tier index header, when loaded.
    #[must_use]
    pub fn quality_embedder_revision(&self) -> Option<&str> {
        self.quality_tier().map(VectorIndex::embedder_revision)
    }

    /// Lowercase hex SHA-256 fingerprint of the fast tier's embedding space,
    /// when known (bd-9xuj T2-C2).
    ///
    /// This is the index-side join key for
    /// [`frankensearch_core::BoundQueryEmbedding::verify_space_identity`]:
    /// a bound query embedding is admissible against the fast tier exactly
    /// when its space fingerprint equals this value (necessary, never
    /// sufficient for a foreign producer — bundle-holding seams must apply
    /// [`frankensearch_core::BoundQueryEmbedding::verify_producer_conformance`]).
    ///
    /// `Some` when the tier's artifact carries a validated FSVI v2 identity
    /// header, or when this instance was returned by a
    /// [`TwoTierIndexBuilder`] whose caller declared the producing identity
    /// ([`TwoTierIndexBuilder::set_fast_identity`]). `None` is the typed
    /// legacy-unidentified state: v1 artifacts persist no space identity, so
    /// a reopen from disk of a v1-built index returns `None` and must be
    /// routed as `LegacyUnidentified` reindex — never admitted on the
    /// id/revision strings or dimension equality.
    #[must_use]
    pub fn fast_space_fingerprint_hex(&self) -> Option<&str> {
        self.fast_space_fingerprint_hex.as_deref()
    }

    /// Quality-tier counterpart of [`Self::fast_space_fingerprint_hex`].
    ///
    /// `None` both when no quality index is loaded and when the loaded one
    /// carries no identity; either way there is no quality-tier space to
    /// verify a query embedding against.
    #[must_use]
    pub fn quality_space_fingerprint_hex(&self) -> Option<&str> {
        self.quality_space_fingerprint_hex.as_deref()
    }

    /// Complete identity bundle of the embedder that produced the fast
    /// tier's vectors, when this instance was returned by a builder that
    /// declared it (bd-9xuj T2-C2; see
    /// [`TwoTierIndexBuilder::set_fast_identity`]).
    ///
    /// This is the `expected` side for
    /// [`frankensearch_core::BoundQueryEmbedding::verify_producer_conformance`].
    /// Its storage component describes the producing embedder's output
    /// contract, not this index's persisted encoding. Process-local for v1
    /// artifacts: a reopen from disk has no bundle and returns `None`.
    #[must_use]
    pub const fn fast_declared_identity(&self) -> Option<&EmbeddingIdentityBundleV1> {
        self.fast_declared_identity.as_ref()
    }

    /// Quality-tier counterpart of [`Self::fast_declared_identity`].
    #[must_use]
    pub const fn quality_declared_identity(&self) -> Option<&EmbeddingIdentityBundleV1> {
        self.quality_declared_identity.as_ref()
    }

    /// Whether the fast tier's identity is FSVI-v2-HEADER-attested rather
    /// than builder-time declared (bd-9xuj T2-C4-write, admission guards
    /// 2+8).
    ///
    /// The attested bit derives from WHERE the identity came from, not from
    /// stored state that could drift: a tier's `VectorIndex` carries
    /// `identity_v2()` metadata exactly when its bytes were parsed as a
    /// validated FSVI v2 header inside exact admission
    /// ([`Self::open_admitted_v2_with_paths`] →
    /// [`VectorIndex::open_admitted_v2`]; plain [`VectorIndex::open`] is
    /// strictly v1 and can never produce it). A builder-declared identity
    /// ([`TwoTierIndexBuilder::set_fast_identity`]) populates
    /// [`Self::fast_declared_identity`] and the space fingerprint, but never
    /// this bit: a declaration is a claim by the constructing process, an
    /// attestation is read out of the artifact's own bytes. Attested-only
    /// admission seams (the refresh identity-bound merge) must join against
    /// attested identity exclusively and route declared-only or v1 tiers to
    /// the typed legacy refusal.
    #[must_use]
    pub fn fast_identity_is_attested(&self) -> bool {
        self.fast_tier().identity_v2().is_some()
    }

    /// Quality-tier counterpart of [`Self::fast_identity_is_attested`].
    ///
    /// `false` both when no quality index is loaded and when the loaded one
    /// carries no validated v2 identity header.
    #[must_use]
    pub fn quality_identity_is_attested(&self) -> bool {
        self.quality_tier()
            .is_some_and(|index| index.identity_v2().is_some())
    }

    /// Space fingerprint of the tier a semantic vector was served from, when
    /// known (bd-9xuj T2-C2). The typed join between
    /// [`Self::semantic_vector_with_tier_for_doc_id`]'s provenance and the
    /// per-tier space accessors.
    #[must_use]
    pub fn space_fingerprint_hex_for_tier(&self, tier: SemanticVectorTier) -> Option<&str> {
        match tier {
            SemanticVectorTier::Fast => self.fast_space_fingerprint_hex(),
            SemanticVectorTier::Quality => self.quality_space_fingerprint_hex(),
        }
    }

    /// Filesystem path of the loaded fast-tier index artifact.
    #[must_use]
    pub fn fast_index_path(&self) -> &Path {
        &self.fast_tier().path
    }

    /// Filesystem path of the loaded quality-tier index artifact, when loaded.
    #[must_use]
    pub fn quality_index_path(&self) -> Option<&Path> {
        self.quality_tier().map(|index| index.path.as_path())
    }

    /// Iterate over all document IDs in fast-tier order.
    pub fn iter_doc_ids(&self) -> impl Iterator<Item = SearchResult<String>> + '_ {
        (0..self.doc_count()).map(|i| self.fast_tier().doc_id_at(i).map(ToOwned::to_owned))
    }

    /// Document ID at a given fast-tier index position.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the index is out of bounds or reading fails.
    pub fn doc_id_at(&self, index: usize) -> SearchResult<&str> {
        self.fast_tier().doc_id_at(index)
    }

    /// Fast-tier index position for a given document id.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if index reading fails.
    pub fn fast_index_for_doc_id(&self, doc_id: &str) -> SearchResult<Option<usize>> {
        self.fast_tier().find_index_by_doc_id(doc_id)
    }

    /// Fast-tier vector for the given document id.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if index access fails.
    pub fn fast_vector_for_doc_id(&self, doc_id: &str) -> SearchResult<Option<Vec<f32>>> {
        let hash = crate::fnv1a_hash(doc_id.as_bytes());
        for entry in self.fast_tier().wal_entries.iter().rev() {
            if entry.doc_id_hash == hash && entry.doc_id == doc_id {
                return Ok(Some(entry.embedding.clone()));
            }
        }

        if let Some(index) = self.fast_tier().find_index_by_doc_id(doc_id)? {
            return self.fast_tier().vector_at_f32(index).map(Some);
        }

        Ok(None)
    }

    /// Quality-tier vector for the given document id when available.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if index access fails.
    pub fn quality_vector_for_doc_id(&self, doc_id: &str) -> SearchResult<Option<Vec<f32>>> {
        let Some(quality_index) = self.quality_tier() else {
            return Ok(None);
        };

        let hash = crate::fnv1a_hash(doc_id.as_bytes());
        for entry in quality_index.wal_entries.iter().rev() {
            if entry.doc_id_hash == hash && entry.doc_id == doc_id {
                return Ok(Some(entry.embedding.clone()));
            }
        }

        if let Some(fast_index) = self.fast_tier().find_index_by_doc_id(doc_id)? {
            if let Some(quality_index_pos) = self.quality_index_for_fast_index(fast_index) {
                return quality_index.vector_at_f32(quality_index_pos).map(Some);
            }
        }

        if let Some(qual_idx) = quality_index.find_index_by_doc_id(doc_id)? {
            return quality_index.vector_at_f32(qual_idx).map(Some);
        }

        Ok(None)
    }

    /// Semantic vector for the given document id, preferring quality tier.
    ///
    /// Falls back to the fast-tier vector when the quality tier is unavailable
    /// or missing for this document.
    ///
    /// The fallback is SILENT here: the returned vector may live in the fast
    /// tier's embedding space rather than the quality tier's, and this
    /// signature cannot say which. Callers that must know — any caller that
    /// compares the vector against other vectors — should use
    /// [`Self::semantic_vector_with_tier_for_doc_id`], which returns the
    /// serving tier alongside the vector (bd-9xuj T2-C2).
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if index access fails.
    pub fn semantic_vector_for_doc_id(&self, doc_id: &str) -> SearchResult<Option<Vec<f32>>> {
        Ok(self
            .semantic_vector_with_tier_for_doc_id(doc_id)?
            .map(|(_, vector)| vector))
    }

    /// Semantic vector for the given document id with typed tier provenance
    /// (bd-9xuj T2-C2).
    ///
    /// Identical lookup order and results as
    /// [`Self::semantic_vector_for_doc_id`] — quality tier first, silent
    /// fast-tier fallback — but the returned [`SemanticVectorTier`] names the
    /// tier that served the vector, so a caller can resolve which embedding
    /// space it lives in via [`Self::space_fingerprint_hex_for_tier`] instead
    /// of comparing vectors across spaces unknowingly. Changing the fallback
    /// itself is out of scope here (C4); this accessor only makes it
    /// observable.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if index access fails.
    pub fn semantic_vector_with_tier_for_doc_id(
        &self,
        doc_id: &str,
    ) -> SearchResult<Option<(SemanticVectorTier, Vec<f32>)>> {
        if let Some(quality) = self.quality_vector_for_doc_id(doc_id)? {
            return Ok(Some((SemanticVectorTier::Quality, quality)));
        }
        Ok(self
            .fast_vector_for_doc_id(doc_id)?
            .map(|vector| (SemanticVectorTier::Fast, vector)))
    }

    /// Whether the fast-tier document at `index` has a quality-tier vector.
    #[must_use]
    pub fn has_quality_for_index(&self, index: usize) -> bool {
        if index >= self.doc_count() {
            return false;
        }
        match &self.quality_alignment {
            QualityAlignment::None => false,

            QualityAlignment::Aligned => true,

            QualityAlignment::Mapping(map) => map.get(index).copied().flatten().is_some(),
        }
    }

    /// Accessor for the configuration used to open this index.
    #[must_use]
    pub const fn config(&self) -> &TwoTierConfig {
        &self.config
    }

    fn score_quality_for_fast_index(
        &self,

        quality_index: &VectorIndex,

        query_vec: &[f32],

        fast_idx: usize,
    ) -> SearchResult<Option<f32>> {
        if fast_idx >= self.doc_count() {
            return Ok(None);
        }
        let quality_idx = match &self.quality_alignment {
            QualityAlignment::None => return Ok(None),

            QualityAlignment::Aligned => fast_idx,

            QualityAlignment::Mapping(map) => match map.get(fast_idx).copied().flatten() {
                Some(idx) => idx,

                None => return Ok(None),
            },
        };

        // Fused byte-based dot (no per-hit `Vec<f32>` decode), matching the
        // brute-force scan's scorer; bit-identical for `dim % 32 == 0`.
        quality_index.dot_query_at(quality_idx, query_vec).map(Some)
    }

    fn quality_index_for_fast_index(&self, fast_idx: usize) -> Option<usize> {
        match &self.quality_alignment {
            QualityAlignment::None => None,
            QualityAlignment::Aligned => Some(fast_idx),
            QualityAlignment::Mapping(map) => map.get(fast_idx).copied().flatten(),
        }
    }
}

/// Tier that served a semantic vector
/// ([`TwoTierIndex::semantic_vector_with_tier_for_doc_id`]; bd-9xuj T2-C2).
///
/// The tier is the typed handle to the embedding space the vector lives in:
/// resolve it via [`TwoTierIndex::space_fingerprint_hex_for_tier`]. Vectors
/// from different tiers are NOT comparable — the silent quality→fast fallback
/// is exactly the seam this type makes observable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticVectorTier {
    /// Served from the quality tier (its embedding space is the quality
    /// tier's).
    Quality,
    /// Served from the fast tier: either no quality index is loaded or the
    /// quality tier has no vector for this document.
    Fast,
}

/// Builder for writing fast and optional quality FSVI indices.
#[derive(Debug)]
pub struct TwoTierIndexBuilder {
    dir: PathBuf,
    config: TwoTierConfig,
    fast_embedder_id: String,
    quality_embedder_id: String,
    fast_embedder_revision: String,
    quality_embedder_revision: String,
    fast_identity: Option<EmbeddingIdentityBundleV1>,
    quality_identity: Option<EmbeddingIdentityBundleV1>,
    fast_dimension: Option<usize>,
    quality_dimension: Option<usize>,
    fast_records: Vec<(String, Vec<f32>)>,
    quality_records: Vec<(String, Vec<f32>)>,
    fast_ids: std::collections::HashSet<String>,
    quality_ids: std::collections::HashSet<String>,
}

impl TwoTierIndexBuilder {
    fn new(dir: PathBuf, config: TwoTierConfig) -> Self {
        Self {
            dir,
            config,
            fast_embedder_id: "fast-tier".to_owned(),
            quality_embedder_id: "quality-tier".to_owned(),
            fast_embedder_revision: String::new(),
            quality_embedder_revision: String::new(),
            fast_identity: None,
            quality_identity: None,
            fast_dimension: None,
            quality_dimension: None,
            fast_records: Vec::new(),
            quality_records: Vec::new(),
            fast_ids: std::collections::HashSet::new(),
            quality_ids: std::collections::HashSet::new(),
        }
    }

    /// Override the embedder id written to the fast-tier index header.
    pub fn set_fast_embedder_id(&mut self, embedder_id: impl Into<String>) -> &mut Self {
        self.fast_embedder_id = embedder_id.into();
        self
    }

    /// Override the embedder id written to the quality-tier index header.
    pub fn set_quality_embedder_id(&mut self, embedder_id: impl Into<String>) -> &mut Self {
        self.quality_embedder_id = embedder_id.into();
        self
    }

    /// Declare the complete identity of the embedder producing the fast
    /// tier's vectors (bd-9xuj T2-C2).
    ///
    /// The bundle is validated before it is accepted — a bound identity is a
    /// claim every downstream verifier trusts — and [`Self::finish`]
    /// additionally rejects it when its space dimension does not describe the
    /// vectors actually written. On success:
    ///
    /// - the fast-tier header's revision string is set to the space's
    ///   `immutable_revision` (the same rule `VectorIndex::create_v2` uses),
    ///   so the persisted v1 header finally carries a real revision;
    /// - the finished [`TwoTierIndex`] retains the bundle
    ///   ([`TwoTierIndex::fast_declared_identity`]) and exposes the space
    ///   join key ([`TwoTierIndex::fast_space_fingerprint_hex`]).
    ///
    /// The operational id string ([`Self::set_fast_embedder_id`]) is left
    /// untouched: id strings are diagnostics, and this typed bundle — never
    /// a string — is the compatibility authority. Retention is process-local
    /// for the v1 artifacts this builder writes: a later reopen from disk is
    /// legacy-unidentified until an identity-persisting writer exists.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when the bundle fails its own
    /// validation.
    pub fn set_fast_identity(
        &mut self,
        identity: &EmbeddingIdentityBundleV1,
    ) -> SearchResult<&mut Self> {
        identity.validate()?;
        self.fast_embedder_revision
            .clone_from(&identity.space.immutable_revision);
        self.fast_identity = Some(identity.clone());
        Ok(self)
    }

    /// Quality-tier counterpart of [`Self::set_fast_identity`].
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when the bundle fails its own
    /// validation.
    pub fn set_quality_identity(
        &mut self,
        identity: &EmbeddingIdentityBundleV1,
    ) -> SearchResult<&mut Self> {
        identity.validate()?;
        self.quality_embedder_revision
            .clone_from(&identity.space.immutable_revision);
        self.quality_identity = Some(identity.clone());
        Ok(self)
    }

    /// Add a fast-tier vector record.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if this record dimension differs
    /// from previously added fast-tier vectors.
    pub fn add_fast_record(
        &mut self,
        doc_id: impl Into<String>,
        embedding: &[f32],
    ) -> SearchResult<()> {
        let dimension = embedding.len();
        let expected = self.fast_dimension.get_or_insert(dimension);
        if *expected != dimension {
            return Err(SearchError::DimensionMismatch {
                expected: *expected,
                found: dimension,
            });
        }
        let doc_id = doc_id.into();
        if !self.fast_ids.insert(doc_id.clone()) {
            return Err(SearchError::InvalidConfig {
                field: "doc_id".to_owned(),
                value: doc_id,
                reason: "duplicate doc_id in fast tier; each document must have a unique id"
                    .to_owned(),
            });
        }
        self.fast_records.push((doc_id, embedding.to_vec()));
        Ok(())
    }

    /// Bench-only comparator that adds an already-owned fast-tier vector record.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if this record dimension differs
    /// from previously added fast-tier vectors.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn add_fast_record_owned_for_benchmark(
        &mut self,
        doc_id: impl Into<String>,
        embedding: Vec<f32>,
    ) -> SearchResult<()> {
        let dimension = embedding.len();
        let expected = self.fast_dimension.get_or_insert(dimension);
        if *expected != dimension {
            return Err(SearchError::DimensionMismatch {
                expected: *expected,
                found: dimension,
            });
        }
        let doc_id = doc_id.into();
        if !self.fast_ids.insert(doc_id.clone()) {
            return Err(SearchError::InvalidConfig {
                field: "doc_id".to_owned(),
                value: doc_id,
                reason: "duplicate doc_id in fast tier; each document must have a unique id"
                    .to_owned(),
            });
        }
        self.fast_records.push((doc_id, embedding));
        Ok(())
    }

    /// Add a quality-tier vector record.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if this record dimension differs
    /// from previously added quality-tier vectors.
    pub fn add_quality_record(
        &mut self,
        doc_id: impl Into<String>,
        embedding: &[f32],
    ) -> SearchResult<()> {
        let dimension = embedding.len();
        let expected = self.quality_dimension.get_or_insert(dimension);
        if *expected != dimension {
            return Err(SearchError::DimensionMismatch {
                expected: *expected,
                found: dimension,
            });
        }
        let doc_id = doc_id.into();
        if !self.quality_ids.insert(doc_id.clone()) {
            return Err(SearchError::InvalidConfig {
                field: "doc_id".to_owned(),
                value: doc_id,
                reason: "duplicate doc_id in quality tier; each document must have a unique id"
                    .to_owned(),
            });
        }
        self.quality_records.push((doc_id, embedding.to_vec()));
        Ok(())
    }

    /// Add a fast record and optionally a matching quality record.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if either tier dimension is inconsistent.
    pub fn add_record(
        &mut self,
        doc_id: impl Into<String>,
        fast_embedding: &[f32],
        quality_embedding: Option<&[f32]>,
    ) -> SearchResult<()> {
        let doc_id = doc_id.into();
        self.add_fast_record(doc_id.clone(), fast_embedding)?;
        if let Some(quality_embedding) = quality_embedding {
            self.add_quality_record(doc_id, quality_embedding)?;
        }
        Ok(())
    }

    /// Write all buffered records and open the resulting `TwoTierIndex`.
    ///
    /// When a producing identity was declared
    /// ([`Self::set_fast_identity`] / [`Self::set_quality_identity`]), it is
    /// checked against the vectors actually written (its space dimension
    /// must equal the tier dimension — an index must never carry an identity
    /// that does not describe its vectors), written into the tier header's
    /// id/revision strings as far as the v1 format can carry it, and
    /// retained on the returned [`TwoTierIndex`] so consumers can verify
    /// per-tier space identity (bd-9xuj T2-C2). A declared quality identity
    /// is attached only when this build wrote a quality tier: it must never
    /// describe a stale quality artifact discovered on disk.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if no fast-tier records were
    /// added or a declared identity does not describe the written vectors,
    /// and propagates writer/open errors from `VectorIndex`.
    pub fn finish(self) -> SearchResult<TwoTierIndex> {
        let fast_dimension = self
            .fast_dimension
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "fast_records".to_owned(),
                value: "0".to_owned(),
                reason: "at least one fast-tier record is required".to_owned(),
            })?;
        if let Some(identity) = &self.fast_identity {
            ensure_identity_describes_tier(identity, fast_dimension, "fast_identity")?;
        }
        if let (Some(identity), Some(quality_dimension)) =
            (&self.quality_identity, self.quality_dimension)
        {
            ensure_identity_describes_tier(identity, quality_dimension, "quality_identity")?;
        }

        // Build each tier at a staging path and route through
        // install_replacement with a carried-forward generation: publishing
        // over a live directory with a plain in-place build left the
        // destination's WAL sidecar adoptable by the fresh generation
        // (bd-zhjv8 finding 1 — acknowledged appends from the superseded
        // generation resurrected inside the rebuild). Both tiers carry ONE
        // publication nonce, the successor of the destination pair's, so a
        // crash between the two installs is detectable at open time
        // (bd-miio8).
        let fast_path = self.dir.join(VECTOR_INDEX_FAST_FILENAME);
        // `create(path, id, dim)` is `create_with_revision(path, id, "", dim,
        // F16)`, so the written bytes are identical to the former `create`
        // call for builders that declared no identity (empty default
        // revision). Each tier is written at a sibling staging path and
        // installed over the canonical name through `publish_tier`, which
        // carries the successor compaction generation and the shared
        // publication nonce so `install_replacement`'s generation authority
        // atomically invalidates any surviving destination sidecar
        // (bd-zhjv8, bd-miio8; supersedes the r3 retire-before-publish
        // ordering — see `publish_tier`).
        let publication_nonce = crate::next_publication_nonce(if fast_path.exists() {
            VectorIndex::peek_publication_nonce(&fast_path)?
        } else {
            0
        });
        publish_tier(
            &fast_path,
            &self.fast_embedder_id,
            &self.fast_embedder_revision,
            fast_dimension,
            publication_nonce,
            &self.fast_records,
        )?;

        if let Some(quality_dimension) = self.quality_dimension {
            let quality_path = self.dir.join(VECTOR_INDEX_QUALITY_FILENAME);
            publish_tier(
                &quality_path,
                &self.quality_embedder_id,
                &self.quality_embedder_revision,
                quality_dimension,
                publication_nonce,
                &self.quality_records,
            )?;
        }

        let mut index = TwoTierIndex::open(&self.dir, self.config)?;

        // Retain the declared producing identities on the opened index
        // (bd-9xuj T2-C2). This is retention of a validated declaration by
        // the process that produced every vector it wrote — the same class
        // as `InMemoryVectorIndex::from_vectors_with_identity` — not
        // fabrication: an artifact-header identity, were one ever present,
        // wins, and a reopen from disk of these v1 artifacts stays typed
        // legacy-unidentified. The quality identity is dropped unless THIS
        // build wrote the quality tier, so it can never describe a stale
        // quality artifact `open` discovered on disk.
        if let Some(identity) = self.fast_identity {
            if index.fast_space_fingerprint_hex.is_none() {
                index.fast_space_fingerprint_hex = Some(identity.space.fingerprint());
            }
            index.fast_declared_identity = Some(identity);
        }
        if let Some(identity) = self.quality_identity
            && self.quality_dimension.is_some()
            && index.quality_source.is_some()
        {
            if index.quality_space_fingerprint_hex.is_none() {
                index.quality_space_fingerprint_hex = Some(identity.space.fingerprint());
            }
            index.quality_declared_identity = Some(identity);
        }
        Ok(index)
    }
}

/// Stage one tier's records and durably install them over `canonical_path`
/// with the generation discipline [`VectorIndex::install_replacement`]
/// requires (bd-zhjv8) and the shared cross-tier publication nonce
/// (bd-miio8).
///
/// Protocol:
/// 1. The replacement is FULLY written at a sibling staging path
///    (`temporary_output_path`), never at the canonical name, carrying
///    `next_generation(destination)` and the caller's publication nonce.
/// 2. [`VectorIndex::install_replacement`] validates the staged artifact
///    and atomically renames it over the canonical main. The generation
///    byte is the authority: the rename installs the new data AND
///    invalidates any surviving destination WAL (whose records bind to the
///    superseded generation), so sidecar retirement happens safely AFTER
///    publication.
///
/// This deliberately supersedes the bd-9xuj C4-write r3
/// retire-before-publish ordering (`write_tier_with_durable_wal_retirement`,
/// audits #8366/#8367): durably retiring the sidecar BEFORE the rename
/// opens a crash window in which the SURVIVING old generation loses its
/// acknowledged appends — the two-resource-commit finding that closed
/// bd-zhjv8 (authority, not ordering). A failed publication cleans up its
/// own staged artifact and leaves the old generation, main AND sidecar,
/// intact; a crash may leave a stray staging file, which lives under a
/// `.tmp.<pid>.<nanos>` name that no open or discovery path resolves.
fn publish_tier(
    canonical_path: &Path,
    embedder_id: &str,
    embedder_revision: &str,
    dimension: usize,
    publication_nonce: u16,
    records: &[(String, Vec<f32>)],
) -> SearchResult<()> {
    let generation = if canonical_path.exists() {
        crate::next_generation(VectorIndex::peek_compaction_gen(canonical_path)?)
    } else {
        1
    };
    let staging_path = crate::temporary_output_path(canonical_path);
    let result = (|| {
        let mut writer = VectorIndex::create_with_revision(
            &staging_path,
            embedder_id,
            embedder_revision,
            dimension,
            Quantization::F16,
        )?
        .with_generation(generation)
        .with_publication_nonce(publication_nonce);
        for (doc_id, embedding) in records {
            writer.write_record(doc_id, embedding)?;
        }
        writer.finish()?;
        VectorIndex::install_replacement(canonical_path, &staging_path).map(drop)
    })();
    if result.is_err()
        && staging_path.exists()
        && let Err(cleanup_err) = fs::remove_file(&staging_path)
    {
        // Best-effort cleanup of the staged artifact only — the staging file
        // is this function's own product under a name no open path resolves,
        // never the canonical main or its WAL (mirrors the error path of
        // `VectorIndexWriter::finish`).
        warn!(
            staging_path = %staging_path.display(),
            ?cleanup_err,
            "failed to clean up staged tier replacement after error"
        );
    }
    result
}

/// Reject a declared producing identity whose space dimension does not
/// describe the tier's written vectors (bd-9xuj T2-C2): the identity claim is
/// checked before it is retained, never trusted on assertion alone.
fn ensure_identity_describes_tier(
    identity: &EmbeddingIdentityBundleV1,
    tier_dimension: usize,
    tier_field: &str,
) -> SearchResult<()> {
    if usize::try_from(identity.space.dimension).ok() == Some(tier_dimension) {
        return Ok(());
    }
    Err(SearchError::InvalidConfig {
        field: format!("{tier_field}.space.dimension"),
        value: identity.space.dimension.to_string(),
        reason: format!(
            "declared embedding-space dimension must equal the written tier dimension \
             ({tier_dimension}); refusing to retain an identity that does not describe \
             this tier's vectors"
        ),
    })
}

/// Admit one identity-complete FSVI v2 tier through the only legitimate v2
/// open path and hand back the SEALED OWNER, whole (bd-9xuj T2-C4-write r2).
///
/// [`VectorIndex::open_admitted_v2`] copies the artifact once into a sealed
/// byte owner, verifies the complete identity/content bindings against
/// `binding`, and rejects any WAL directory entry. The r1 revision of this
/// helper moved `validated.index` out and DROPPED the owner — discarding the
/// `ValidatedFsviBytes` capability, the complete witness, and the
/// publication state, contradicting the owner contract on
/// [`ValidatedFsviBytes`] ("no conversion into a mutable/path-opened
/// `VectorIndex`"). The r2 repair retains the owner in full; callers wrap it
/// in [`TierSource::AdmittedV2`] and borrow the validated index from inside
/// it, so the parsed v2 identity metadata (`identity_v2()` stays `Some`)
/// still marks the tier ATTESTED on the assembled [`TwoTierIndex`].
fn admit_v2_tier(
    path: &Path,
    binding: &FsviV2IdentityBinding,
    tier: &str,
) -> SearchResult<ValidatedFsviBytes> {
    VectorIndex::open_admitted_v2(path, binding)
        .map_err(|error| admission_error_to_search_error(error, tier, path))
}

/// Map a typed [`FsviAdmissionError`] into the [`SearchError`] surface,
/// naming the tier so refusals stay attributable. I/O and corruption pass
/// through unchanged; reindex/upgrade/snapshot outcomes become typed
/// `InvalidConfig` refusals rather than being flattened into strings at the
/// caller.
fn admission_error_to_search_error(
    error: FsviAdmissionError,
    tier: &str,
    path: &Path,
) -> SearchError {
    match error {
        FsviAdmissionError::Index(error) => error,
        other => SearchError::InvalidConfig {
            field: format!("two_tier.{tier}_v2_admission"),
            value: path.display().to_string(),
            reason: other.to_string(),
        },
    }
}

/// Loudly surface v1 WAL replay at two-tier reopen (bd-9xuj C4-write r3,
/// fault test iv — a generation-collision sidecar must never be treated as
/// active rows SILENTLY).
///
/// In the v1 format a WAL whose generation equals `next_generation(main)`
/// is byte-indistinguishable from legitimate incremental appends — and that
/// includes a FOREIGN sidecar left beside a freshly republished main by a
/// pre-r3 crash: fresh legacy mains start at compaction generation 1, so a
/// leftover generation-2 sidecar collides exactly.
/// `TwoTierIndexBuilder::finish` now publishes each tier as an
/// authority-carrying successor generation (`publish_tier`, bd-zhjv8), so
/// the rename itself invalidates the destination's sidecar and finish can
/// no longer manufacture that state; this warning keeps
/// any residual legacy state (pre-r3 crash artifacts, external
/// manipulation) from being replayed silently. Read-side classification
/// cannot refuse here without breaking the legitimate v1
/// incremental-append reopen contract, which external `VectorIndex`
/// consumers rely on.
fn warn_if_wal_rows_replayed(tier: &str, path: &Path, index: &VectorIndex) {
    let replayed = index.wal_record_count();
    if replayed > 0 {
        warn!(
            tier,
            path = %path.display(),
            replayed_wal_records = replayed,
            "two-tier v1 open replayed WAL sidecar rows; if this main was \
             just republished, the rows may belong to a replaced foreign \
             generation (generation counters wrap mod 255)"
        );
    }
}

fn resolve_fast_path(dir: &Path) -> SearchResult<PathBuf> {
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    if fast_path.exists() {
        return Ok(fast_path);
    }

    let fallback_path = dir.join(VECTOR_INDEX_FALLBACK_FILENAME);
    if fallback_path.exists() {
        return Ok(fallback_path);
    }

    Err(SearchError::IndexCandidatesNotFound {
        paths: vec![fast_path, fallback_path],
    })
}

// ---------------------------------------------------------------------------
// Read-only tier observation (bd-9xuj C4-write r2)
// ---------------------------------------------------------------------------

/// What [`observe_tier`] proved about one on-disk FSVI tier artifact.
///
/// Every variant is established WITHOUT any mutable open: no WAL sidecar is
/// ever deleted or truncated, no file is memory-mapped writable, and no byte
/// of the artifact or its directory changes. Access metadata is the one
/// exception — see [`observe_tier`] for the exact atime/symlink caveat of
/// the ordinary read-only fallback. This is the classification carrier for
/// pre-drain refresh admission (the r2 repair of the C4-write NO-GO's
/// mutable-`VectorIndex::open`-during-classification hazard).
#[derive(Debug)]
pub enum FsviTierObservation {
    /// Recognized legacy FSVI v1 bytes, header-parsed only.
    V1(FsviV1Observation),
    /// Identity-complete FSVI v2 header. Content admission (digest
    /// recomputation, sealed-owner retention) still happens through
    /// [`VectorIndex::open_admitted_v2`]; this variant only proves the
    /// header parses as identity-complete v2.
    V2IdentityComplete(Box<VectorMetadata>),
    /// A newer FSVI schema than this reader supports.
    UpgradeRequired(FsviUpgradeRequired),
}

/// Header-level, read-only observation of a legacy FSVI v1 tier.
///
/// Record FLAGS are deliberately not inspected (that would require reading
/// the record table): `record_count` counts live AND tombstoned rows, so
/// [`Self::retains_content`] is conservative — an all-tombstoned v1 tier
/// reads as retaining content and fails closed at admission seams. Restoring
/// flag-level precision needs a read-only record-table inspector in the
/// crate root (deferred to the observational-open train; see
/// `VectorIndex::inspect`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsviV1Observation {
    /// Total record slots in the main slab (live + tombstoned).
    pub record_count: usize,
    /// Decodable WAL entries whose compaction generation matches the main
    /// slab. `0` when the sidecar is absent, empty, STALE (generation
    /// mismatch), or when only a corrupt trailer follows zero valid batches.
    /// Unlike `VectorIndex::open`, observing a stale or trailer-corrupt WAL
    /// never deletes or truncates it.
    pub active_wal_records: usize,
    /// Whether a WAL sidecar file exists at all (any content state).
    pub wal_sidecar_present: bool,
}

impl FsviV1Observation {
    /// Conservative content-retention signal for admission seams.
    ///
    /// True when the main slab has any record slot (live or tombstoned) or
    /// the WAL retains generation-matched appends. Fail-closed by
    /// construction: it can claim retention for content-free artifacts
    /// (all-tombstoned slabs) but never the reverse.
    #[must_use]
    pub const fn retains_content(&self) -> bool {
        self.record_count > 0 || self.active_wal_records > 0
    }
}

/// Open one tier artifact strictly read-only, preserving atime when the
/// platform allows.
///
/// Prefers the crate's `O_NOATIME | O_NOFOLLOW | O_CLOEXEC` opener (the same
/// one exact v2 admission uses). Where that is denied or unsupported (non-
/// owner files, non-Linux targets, symlinked finals) this falls back to an
/// ordinary read-only [`fs::File::open`], whose only observable effects are
/// a possible atime update and symlink traversal — never weaker than
/// `VectorIndex::inspect`'s unconditional ordinary open, and never
/// write-capable.
fn open_tier_readonly(path: &Path) -> SearchResult<fs::File> {
    crate::open_readonly_noatime_nofollow(path)
        .or_else(|_| fs::File::open(path).map_err(SearchError::Io))
}

/// Classify one on-disk FSVI tier artifact without writes.
///
/// Exact guarantee: nothing is written, truncated, or deleted, and no
/// writable open or mapping occurs — bytes, directory entries, sizes, and
/// mtimes are invariant across the call.
///
/// The one deliberate non-guarantee is access metadata (r3 claim precision,
/// audits #8366/#8367): the preferred opener is the no-atime/no-follow fast
/// path (`O_NOATIME | O_NOFOLLOW | O_CLOEXEC`), but where that open is
/// denied or unsupported the documented fallback is an ordinary read-only
/// `File::open` — and the v1 WAL sidecar is read via `wal::read_wal`'s
/// ordinary open — either of which may update atime and follow symlinks.
///
/// Contrast with the two open paths this deliberately is not:
/// - [`VectorIndex::open`] (v1) opens WRITE-capable, deletes a stale WAL
///   sidecar, and truncates a corrupt WAL trailer — it must never run during
///   classification;
/// - [`VectorIndex::inspect`] is header-only but discards the parsed v1
///   metadata, so callers cannot distinguish an empty v1 seed from a
///   content-retaining one without a mutable open.
///
/// This function parses the SAME headers through the SAME crate parsers
/// (`parse_header` / `parse_v2_header`), reads the v1 WAL sidecar through
/// the same read-only `wal::read_wal`, and applies the same staleness
/// predicate `VectorIndex::open` uses — but performs no deletion, no
/// truncation, and no writable mapping. v2 recognition here is HEADER-ONLY:
/// content admission still requires [`VectorIndex::open_admitted_v2`].
///
/// # Errors
///
/// Returns [`SearchError::IndexNotFound`] for a missing path, I/O errors,
/// and [`SearchError::IndexCorrupted`] for bad magic, malformed or
/// CRC-drifted headers, WAL header corruption, or unsupported historical
/// versions — the same typed outcomes the corresponding open paths produce.
pub fn observe_tier(path: &Path) -> SearchResult<FsviTierObservation> {
    if !path.exists() {
        return Err(SearchError::IndexNotFound {
            path: path.to_path_buf(),
        });
    }
    let mut file = open_tier_readonly(path)?;
    let file_len = file.metadata().map_err(SearchError::Io)?.len();
    let mut prefix = [0_u8; 6];
    crate::read_exact_index_bytes(path, &mut file, &mut prefix, "magic and version")?;
    if prefix[..4] != crate::FSVI_MAGIC {
        return Err(crate::index_corrupted(
            path,
            format!(
                "bad magic bytes: expected {:?}, found {:?}",
                crate::FSVI_MAGIC,
                &prefix[..4]
            ),
        ));
    }
    let version = u16::from_le_bytes([prefix[4], prefix[5]]);
    match version {
        crate::FSVI_VERSION => {
            let bounded_len = usize::try_from(file_len)
                .unwrap_or(usize::MAX)
                .min(crate::FSVI_V1_MAX_HEADER_BYTES);
            file.seek(SeekFrom::Start(0)).map_err(SearchError::Io)?;
            let mut header = Vec::with_capacity(bounded_len);
            file.take(u64::try_from(bounded_len).unwrap_or(u64::MAX))
                .read_to_end(&mut header)
                .map_err(SearchError::Io)?;
            let (metadata, _header_len) = crate::parse_header(path, &header)?;
            Ok(FsviTierObservation::V1(observe_v1_wal(path, &metadata)?))
        }
        crate::FSVI_V2_VERSION => {
            let mut encoded_size = [0_u8; 4];
            crate::read_exact_index_bytes(path, &mut file, &mut encoded_size, "v2 header_size")?;
            let header_size = usize::try_from(u32::from_le_bytes(encoded_size)).map_err(|_| {
                crate::index_corrupted(path, "v2 header_size does not fit in usize")
            })?;
            crate::validate_v2_header_size(path, header_size)?;
            if u64::try_from(header_size).is_ok_and(|size| size > file_len) {
                return Err(crate::index_corrupted(
                    path,
                    format!(
                        "v2 header is truncated: declared {header_size} bytes, file has {file_len}"
                    ),
                ));
            }
            file.seek(SeekFrom::Start(0)).map_err(SearchError::Io)?;
            let mut header = vec![0_u8; header_size];
            crate::read_exact_index_bytes(path, &mut file, &mut header, "v2 header")?;
            let (metadata, _header_len) = crate::parse_v2_header(path, &header)?;
            Ok(FsviTierObservation::V2IdentityComplete(Box::new(metadata)))
        }
        found if found > crate::FSVI_V2_VERSION => {
            Ok(FsviTierObservation::UpgradeRequired(FsviUpgradeRequired {
                found_version: found,
                supported_version: crate::FSVI_V2_VERSION,
            }))
        }
        found => Err(crate::index_corrupted(
            path,
            format!("unsupported historical FSVI schema version {found}"),
        )),
    }
}

/// Read-only WAL observation for a v1 tier: same parser and same staleness
/// predicate as `VectorIndex::open`, minus the deletion/truncation side
/// effects.
fn observe_v1_wal(path: &Path, metadata: &VectorMetadata) -> SearchResult<FsviV1Observation> {
    let wal_path = crate::wal::wal_path_for(path);
    let wal_sidecar_present = wal_path.exists();
    let (entries, wal_compaction_gen, valid_len) =
        crate::wal::read_wal(&wal_path, metadata.dimension, metadata.quantization)?;
    // Staleness predicate mirrored from `VectorIndex::open` (the single
    // other consumer of `read_wal`'s generation output). A stale sidecar's
    // entries belong to a dead generation: they are not counted as retained
    // content, but the FILE is left exactly as found.
    let is_stale = if valid_len > 0 {
        if wal_compaction_gen == 0 {
            metadata.compaction_gen > 0
        } else {
            wal_compaction_gen != crate::next_generation(metadata.compaction_gen)
        }
    } else {
        false
    };
    let active_wal_records = if is_stale { 0 } else { entries.len() };
    Ok(FsviV1Observation {
        record_count: metadata.record_count,
        active_wal_records,
        wal_sidecar_present,
    })
}

#[cfg(feature = "ann")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AnnPersistenceReason {
    FreshBuild,
    RebuiltFallback,
}

#[cfg(feature = "ann")]
#[derive(Debug)]
struct AnnOpenPlan {
    index: HnswIndex,
    persistence: Option<AnnPersistenceReason>,
}

#[cfg(feature = "ann")]
impl AnnOpenPlan {
    const fn needs_persistence(&self) -> bool {
        self.persistence.is_some()
    }
}

#[cfg(feature = "ann")]
fn plan_load_or_build_ann(
    vector_index: &VectorIndex,
    ann_path: &Path,
    threshold: usize,
    config: &TwoTierConfig,
    tier: &str,
) -> Option<AnnOpenPlan> {
    if vector_index.record_count() < threshold {
        return None;
    }

    let ann_config = HnswConfig {
        m: config.hnsw_m,
        ef_construction: config.hnsw_ef_construction,
        ef_search: config.hnsw_ef_search,
        max_layer: HNSW_DEFAULT_MAX_LAYER,
    };

    if ann_path.exists() {
        match HnswIndex::load_with_disposition(ann_path, vector_index) {
            Ok((ann, load_disposition)) => match ann.matches_vector_index(vector_index) {
                Ok(true) => {
                    let loaded_config = ann.config();
                    if loaded_config == ann_config {
                        return Some(AnnOpenPlan {
                            index: ann,
                            persistence: (load_disposition == HnswLoadDisposition::Rebuilt)
                                .then_some(AnnPersistenceReason::RebuiltFallback),
                        });
                    }
                    warn!(
                        tier,
                        ann_path = %ann_path.display(),
                        ?loaded_config,
                        ?ann_config,
                        "ANN sidecar config differs from requested config; rebuilding"
                    );
                }
                Ok(false) => {
                    warn!(
                        tier,
                        ann_path = %ann_path.display(),
                        "ANN sidecar exists but does not match vector index; rebuilding"
                    );
                }
                Err(error) => {
                    warn!(
                        tier,
                        ann_path = %ann_path.display(),
                        ?error,
                        "failed to validate ANN sidecar; rebuilding"
                    );
                }
            },
            Err(error) => {
                warn!(
                    tier,
                    ann_path = %ann_path.display(),
                    ?error,
                    "failed to load ANN sidecar; rebuilding"
                );
            }
        }
    }

    let ann = match HnswIndex::build_from_vector_index(vector_index, ann_config) {
        Ok(ann) => ann,
        Err(error) => {
            warn!(
                tier,
                ?error,
                "failed to build ANN index; using brute-force fallback"
            );
            return None;
        }
    };

    Some(AnnOpenPlan {
        index: ann,
        persistence: Some(AnnPersistenceReason::FreshBuild),
    })
}

#[cfg(feature = "ann")]
fn persist_ann_plan(
    plan: &AnnOpenPlan,
    ann_path: &Path,
    tier: &str,
    paths: &TwoTierIndexPaths,
    fast_ann_enabled: bool,
    quality_ann_enabled: bool,
) {
    let Some(reason) = plan.persistence else {
        return;
    };
    if let Err(error) = validate_ann_persistence_paths(paths, fast_ann_enabled, quality_ann_enabled)
        .and_then(|()| HnswIndex::save(&plan.index, ann_path))
    {
        warn!(
            tier,
            ann_path = %ann_path.display(),
            ?reason,
            ?error,
            "failed to persist ANN sidecar; ANN stays in-memory for this process, persistence \
             durability was not confirmed, and the next startup may rebuild it again; check path \
             permissions and free space"
        );
    } else {
        debug!(
            tier,
            ann_path = %ann_path.display(),
            ?reason,
            "persisted ANN sidecar"
        );
    }
}

#[cfg(all(feature = "ann", test))]
fn maybe_load_or_build_ann_with_save<Validate, Save>(
    vector_index: &VectorIndex,
    ann_path: &Path,
    threshold: usize,
    config: &TwoTierConfig,
    tier: &str,
    validate_before_save: Validate,
    save_ann: Save,
) -> Option<HnswIndex>
where
    Validate: Fn() -> SearchResult<()>,
    Save: Fn(&HnswIndex, &Path) -> SearchResult<()>,
{
    let plan = plan_load_or_build_ann(vector_index, ann_path, threshold, config, tier)?;
    if let Some(reason) = plan.persistence {
        if let Err(error) = validate_before_save().and_then(|()| save_ann(&plan.index, ann_path)) {
            warn!(
                tier,
                ann_path = %ann_path.display(),
                ?reason,
                ?error,
                "failed to persist ANN sidecar; ANN stays in-memory for this process"
            );
        }
    }
    Some(plan.index)
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temp_index_dir(label: &str) -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "frankensearch-two-tier-{label}-{}-{timestamp}",
            std::process::id()
        ))
    }

    fn write_index_file(path: &Path, rows: &[(&str, &[f32])]) -> SearchResult<()> {
        let dimension = rows
            .first()
            .map(|(_, vector)| vector.len())
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "rows".to_owned(),
                value: "[]".to_owned(),
                reason: "rows must not be empty".to_owned(),
            })?;
        let mut writer = VectorIndex::create(path, "test", dimension)?;
        for (doc_id, vector) in rows {
            writer.write_record(doc_id, vector)?;
        }
        writer.finish()
    }

    #[cfg(feature = "ann")]
    fn load_native_ann_sidecar(path: &Path, source_index: &VectorIndex) -> HnswIndex {
        let (ann, disposition) = HnswIndex::load_with_disposition(path, source_index)
            .expect("load persisted ANN sidecar");
        assert_eq!(
            disposition,
            HnswLoadDisposition::Native,
            "the repaired sidecar must use the native graph path on its next load"
        );

        let metadata: serde_json::Value =
            serde_json::from_slice(&fs::read(path).expect("read ANN metadata"))
                .expect("parse ANN metadata");
        assert_eq!(
            metadata["format_version"].as_u64(),
            Some(u64::from(HNSW_META_FORMAT_CURRENT)),
            "repaired ANN metadata must use the current format"
        );
        let generation = metadata["sidecar_generation"]
            .as_str()
            .expect("current metadata generation");
        let basename = metadata["sidecar_basename"]
            .as_str()
            .expect("current metadata basename");
        let sidecar_parent = path.parent().expect("ANN metadata parent").join(generation);
        assert!(
            sidecar_parent
                .join(format!("{basename}.hnsw.graph"))
                .is_file(),
            "repaired native graph sibling must exist"
        );
        assert!(
            sidecar_parent
                .join(format!("{basename}.hnsw.data"))
                .is_file(),
            "repaired native data sibling must exist"
        );
        ann
    }

    #[test]
    fn opens_with_fallback_fast_index() {
        let dir = temp_index_dir("fallback");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fallback = dir.join(VECTOR_INDEX_FALLBACK_FILENAME);

        write_index_file(
            &fallback,
            &[
                ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
                ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
            ],
        )
        .expect("write fallback index");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open two-tier");
        assert_eq!(index.doc_count(), 2);
        assert!(!index.has_quality_index());
        let ids: Vec<String> = index
            .iter_doc_ids()
            .collect::<SearchResult<_>>()
            .expect("ids");
        assert_eq!(ids, vec!["doc-a".to_owned(), "doc-b".to_owned()]);
        assert_eq!(index.fast_index_for_doc_id("doc-a").unwrap(), Some(0));
        assert_eq!(index.fast_index_for_doc_id("doc-b").unwrap(), Some(1));
        assert_eq!(index.fast_index_for_doc_id("missing").unwrap(), None);

        let hits = index
            .search_fast(&[1.0, 0.0, 0.0, 0.0], 1)
            .expect("fast search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-a");
        #[cfg(feature = "ann")]
        assert_eq!(index.ann_fallback_count(), 0);
    }

    #[test]
    fn search_fast_with_params_matches_default_path() {
        let dir = temp_index_dir("search-params-default-match");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[
                ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
                ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
                ("doc-c", &[0.0, 0.0, 1.0, 0.0]),
            ],
        )
        .expect("write fast index");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open two-tier");
        let baseline = index
            .search_fast(&[1.0, 0.0, 0.0, 0.0], 2)
            .expect("baseline");
        let overridden = index
            .search_fast_with_params(&[1.0, 0.0, 0.0, 0.0], 2, Some(SearchParams::default()))
            .expect("search with params");
        assert_eq!(baseline, overridden);
    }

    #[test]
    fn search_fast_with_params_accepts_explicit_sequential_override() {
        let dir = temp_index_dir("search-params-seq-override");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[
                ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
                ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
                ("doc-c", &[0.0, 0.0, 1.0, 0.0]),
                ("doc-d", &[0.0, 0.0, 0.0, 1.0]),
            ],
        )
        .expect("write fast index");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open two-tier");
        let params = SearchParams {
            parallel_enabled: false,
            parallel_threshold: usize::MAX,
            parallel_chunk_size: 2,
        };
        let hits = index
            .search_fast_with_params(&[1.0, 0.0, 0.0, 0.0], 3, Some(params))
            .expect("sequential override search");
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].doc_id, "doc-a");
    }

    #[test]
    fn quality_alignment_handles_partial_coverage() {
        let dir = temp_index_dir("alignment");
        fs::create_dir_all(&dir).expect("create temp dir");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);

        write_index_file(
            &fast_path,
            &[
                ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
                ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
                ("doc-c", &[0.0, 0.0, 1.0, 0.0]),
            ],
        )
        .expect("write fast index");

        // Quality tier intentionally omits doc-b and uses different order.
        write_index_file(
            &quality_path,
            &[
                ("doc-c", &[0.0, 1.0, 0.0, 0.0]),
                ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
            ],
        )
        .expect("write quality index");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open two-tier");
        assert!(index.has_quality_index());
        assert!(index.has_quality_for_index(0));
        assert!(!index.has_quality_for_index(1));
        assert!(index.has_quality_for_index(2));

        let hits = vec![
            VectorHit {
                index: 0,
                score: 0.0,
                doc_id: "doc-a".into(),
            },
            VectorHit {
                index: 1,
                score: 0.0,
                doc_id: "doc-b".into(),
            },
            VectorHit {
                index: 2,
                score: 0.0,
                doc_id: "doc-c".into(),
            },
        ];
        let scores = index
            .quality_scores_for_hits(&[1.0, 0.0, 0.0, 0.0], &hits)
            .expect("quality scores");
        assert_eq!(scores.len(), 3);
        // doc-a has quality vector [1,0,0,0], query=[1,0,0,0] → dot=1.0
        assert!((scores[0].unwrap() - 1.0).abs() < 1e-6);
        // doc-b has NO quality vector → None
        assert!(scores[1].is_none());
        // doc-c has quality vector [0,1,0,0], query=[1,0,0,0] → dot=0.0
        assert!(scores[2].unwrap().abs() < 1e-6);
    }

    #[test]
    fn quality_scores_are_none_without_quality_index() {
        let dir = temp_index_dir("no-quality");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[("doc-a", &[1.0, 0.0]), ("doc-b", &[0.0, 1.0])],
        )
        .expect("write fast index");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let hits = vec![
            VectorHit {
                index: 0,
                score: 0.0,
                doc_id: "doc-a".into(),
            },
            VectorHit {
                index: 1,
                score: 0.0,
                doc_id: "doc-b".into(),
            },
            VectorHit {
                index: 99,
                score: 0.0,
                doc_id: "doc-missing".into(),
            },
        ];
        let scores = index
            .quality_scores_for_hits(&[1.0, 0.0], &hits)
            .expect("scores");
        assert_eq!(scores, vec![None, None, None]);
    }

    /// A crash between the two per-tier installs leaves a mixed-generation
    /// pair (new fast + old quality). The pair's publication nonces disagree,
    /// and open must degrade to fast-only instead of silently blending tiers
    /// from different publications (bd-miio8).
    #[test]
    fn open_degrades_to_fast_only_when_tier_publication_identities_disagree() {
        let dir = temp_index_dir("mixed-pair-degrade");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .add_record("doc-a", &[1.0, 0.0, 0.0, 0.0], Some(&[1.0, 0.0]))
            .expect("add doc-a");
        drop(builder.finish().expect("finish pair"));

        // Simulate the crash window: only the fast tier of the NEXT
        // publication lands.
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let fast_gen =
            crate::next_generation(VectorIndex::peek_compaction_gen(&fast_path).expect("peek gen"));
        let fast_nonce = crate::next_publication_nonce(
            VectorIndex::peek_publication_nonce(&fast_path).expect("peek nonce"),
        );
        let staging = crate::temporary_output_path(&fast_path);
        let mut writer = VectorIndex::create(&staging, "fast-tier", 4)
            .expect("staging writer")
            .with_generation(fast_gen)
            .with_publication_nonce(fast_nonce);
        writer
            .write_record("doc-b", &[0.0, 1.0, 0.0, 0.0])
            .expect("write doc-b");
        writer.finish().expect("finish staging");
        VectorIndex::install_replacement(&fast_path, &staging).expect("install fast tier only");

        let reopened = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open mixed pair");
        assert!(
            !reopened.has_quality_index(),
            "a mixed-publication pair must degrade to fast-only, not blend tiers"
        );
    }

    #[test]
    fn open_degrades_to_fast_only_when_discovered_quality_index_is_corrupt() {
        let dir = temp_index_dir("corrupt-discovered-quality-degrade");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[("doc-a", &[1.0, 0.0]), ("doc-b", &[0.0, 1.0])],
        )
        .expect("write fast index");
        fs::write(dir.join(VECTOR_INDEX_QUALITY_FILENAME), b"not an FSVI file")
            .expect("write corrupt optional quality index");

        let opened = TwoTierIndex::open(&dir, TwoTierConfig::default())
            .expect("a corrupt discovered optional quality tier must not block fast-only open");

        assert_eq!(opened.doc_count(), 2);
        assert!(
            !opened.has_quality_index(),
            "a corrupt discovered quality tier must degrade to fast-only"
        );
        assert_eq!(
            opened
                .search_fast(&[1.0, 0.0], 1)
                .expect("fast search after quality degradation")[0]
                .doc_id,
            "doc-a"
        );
    }

    #[test]
    fn open_with_paths_rejects_a_corrupt_explicit_quality_index() {
        let dir = temp_index_dir("corrupt-explicit-quality-reject");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join("fast.fsvi");
        let quality_path = dir.join("quality.fsvi");
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast index");
        fs::write(&quality_path, b"not an FSVI file").expect("write corrupt quality index");

        let error = TwoTierIndex::open_with_paths(
            &TwoTierIndexPaths::new(&fast_path).with_quality_index(&quality_path),
            TwoTierConfig::default(),
        )
        .expect_err("an explicit corrupt quality path must remain an error");

        assert!(
            !matches!(error, SearchError::IndexNotFound { .. }),
            "the corrupt explicit path must not be reported as absent: {error:?}"
        );
    }

    #[test]
    fn finish_stamps_both_tiers_with_one_nonzero_publication_nonce() {
        let dir = temp_index_dir("pair-nonce");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .add_record("doc-a", &[1.0, 0.0], Some(&[0.0, 1.0]))
            .expect("add doc-a");
        drop(builder.finish().expect("finish pair"));

        let fast_nonce = VectorIndex::peek_publication_nonce(&dir.join(VECTOR_INDEX_FAST_FILENAME))
            .expect("fast nonce");
        let quality_nonce =
            VectorIndex::peek_publication_nonce(&dir.join(VECTOR_INDEX_QUALITY_FILENAME))
                .expect("quality nonce");
        assert_ne!(fast_nonce, 0, "a published pair must carry a real identity");
        assert_eq!(
            fast_nonce, quality_nonce,
            "both tiers share one publication"
        );

        // Republishing advances the shared identity.
        let mut second = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        second
            .add_record("doc-b", &[0.0, 1.0], Some(&[1.0, 0.0]))
            .expect("add doc-b");
        drop(second.finish().expect("finish second pair"));
        let second_nonce =
            VectorIndex::peek_publication_nonce(&dir.join(VECTOR_INDEX_FAST_FILENAME))
                .expect("second fast nonce");
        assert_ne!(second_nonce, fast_nonce);
    }

    /// Rebuilding a live directory must SUPERSEDE the prior generation's WAL,
    /// never adopt it: an acknowledged append from the old generation may not
    /// resurrect inside the freshly built one (bd-zhjv8 finding 1).
    #[test]
    fn finish_over_live_dir_never_adopts_the_prior_generations_wal() {
        let dir = temp_index_dir("rebuild-wal-supersede");
        let mut first = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        first
            .add_record("doc-a", &[1.0, 0.0, 0.0, 0.0], None)
            .expect("add doc-a");
        drop(first.finish().expect("finish first generation"));

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let mut live = VectorIndex::open(&fast_path).expect("open first generation");
        live.append("ghost", &[0.0, 1.0, 0.0, 0.0])
            .expect("append acknowledged WAL record");
        drop(live);

        let mut second = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        second
            .add_record("doc-b", &[0.0, 0.0, 1.0, 0.0], None)
            .expect("add doc-b");
        drop(second.finish().expect("finish second generation"));

        let rebuilt = VectorIndex::open(&fast_path).expect("open second generation");
        let live_ids = rebuilt.live_doc_ids().expect("live ids");
        assert!(
            !live_ids.contains("ghost"),
            "prior generation's WAL record resurrected into the rebuilt index: {live_ids:?}"
        );
        assert!(live_ids.contains("doc-b"));
        assert_eq!(
            rebuilt.wal_record_count(),
            0,
            "rebuilt generation must start with no adopted WAL"
        );
    }

    #[test]
    fn builder_round_trips_fast_and_quality_records() {
        let dir = temp_index_dir("builder");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .set_fast_embedder_id("fast-test")
            .set_quality_embedder_id("quality-test");
        builder
            .add_record("doc-a", &[1.0, 0.0, 0.0], Some(&[1.0, 0.0, 0.0]))
            .expect("add doc-a");
        builder
            .add_record("doc-b", &[0.0, 1.0, 0.0], None)
            .expect("add doc-b");

        let index = builder.finish().expect("finish builder");
        assert_eq!(index.doc_count(), 2);
        assert!(index.has_quality_index());
        assert!(index.has_quality_for_index(0));
        assert!(!index.has_quality_for_index(1));
    }

    #[test]
    fn builder_rejects_inconsistent_fast_dimension() {
        let dir = temp_index_dir("bad-dim");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .add_fast_record("doc-a", &[1.0, 0.0, 0.0])
            .expect("first record");

        let err = builder
            .add_fast_record("doc-b", &[1.0, 0.0])
            .expect_err("must reject dimension mismatch");
        assert!(matches!(
            err,
            SearchError::DimensionMismatch {
                expected: 3,
                found: 2
            }
        ));
    }

    #[cfg(feature = "ann")]
    #[test]
    fn ann_sidecar_is_created_when_threshold_is_met() {
        let dir = temp_index_dir("ann-enabled");
        fs::create_dir_all(&dir).expect("create temp dir");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[
                ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
                ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
                ("doc-c", &[0.0, 0.0, 1.0, 0.0]),
            ],
        )
        .expect("write fast index");

        let config = TwoTierConfig {
            hnsw_threshold: 1,
            hnsw_ef_search: 32,
            ..TwoTierConfig::default()
        };
        let index = TwoTierIndex::open(&dir, config).expect("open with ann");
        assert!(index.has_fast_ann());
        assert!(dir.join(VECTOR_ANN_FAST_FILENAME).exists());

        let hits = index
            .search_fast(&[1.0, 0.0, 0.0, 0.0], 1)
            .expect("ann search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-a");
    }

    #[cfg(feature = "ann")]
    #[test]
    fn ann_wal_merge_matches_canonical_resolution_before_and_after_exact_fallback() {
        use crate::wal::WalEntry;

        let dir = temp_index_dir("ann-wal-canonical-resolution");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[
                ("doc-a", &[1.0, 0.0]),
                ("doc-b", &[0.8, 0.6]),
                ("doc-c", &[0.0, 1.0]),
                ("doc-delete", &[-1.0, 0.0]),
            ],
        )
        .expect("write fast index");

        let config = TwoTierConfig {
            hnsw_threshold: 1,
            hnsw_ef_search: 32,
            ..TwoTierConfig::default()
        };
        let mut index = TwoTierIndex::open(&dir, config).expect("open ANN index");
        assert!(index.has_fast_ann());

        // Model the crash-recovery window in which a durable WAL update exists
        // but its best-effort main-slab tombstone did not land. Include a
        // repeated WAL identity to exercise the canonical post-rank dedup rule.
        index.fast_index.wal_entries.extend([
            WalEntry {
                doc_id: "doc-a".into(),
                doc_id_hash: crate::fnv1a_hash(b"doc-a"),
                embedding: vec![-1.0, 0.0],
            },
            WalEntry {
                doc_id: "doc-a".into(),
                doc_id_hash: crate::fnv1a_hash(b"doc-a"),
                embedding: vec![0.2, 0.979_795_9],
            },
            WalEntry {
                doc_id: "doc-new".into(),
                doc_id_hash: crate::fnv1a_hash(b"doc-new"),
                embedding: vec![0.95, 0.312_249_9],
            },
        ]);

        let query = [1.0_f32, 0.0];
        let assert_canonical = |index: &TwoTierIndex, label: &str| {
            let expected = index
                .fast_index
                .search_top_k(&query, 10, None)
                .expect("canonical main plus WAL search");
            let actual = index.search_fast(&query, 10).expect("ANN plus WAL search");
            let expected_identity: Vec<_> = expected
                .iter()
                .map(|hit| (hit.doc_id.clone(), hit.index))
                .collect();
            let actual_identity: Vec<_> = actual
                .iter()
                .map(|hit| (hit.doc_id.clone(), hit.index))
                .collect();
            assert_eq!(
                actual_identity, expected_identity,
                "{label}: ANN and exact search must resolve the same public identities"
            );

            let unique_ids: HashSet<_> = actual.iter().map(|hit| hit.doc_id.as_str()).collect();
            assert_eq!(
                unique_ids.len(),
                actual.len(),
                "{label}: public results must not contain duplicate document IDs"
            );
            let doc_a = actual
                .iter()
                .find(|hit| hit.doc_id == "doc-a")
                .expect("WAL doc-a remains searchable");
            assert!(
                usize::try_from(doc_a.index).expect("u32 fits usize")
                    >= index.fast_index.record_count(),
                "{label}: WAL doc-a must supersede the stale main-slab version"
            );
        };

        assert_canonical(&index, "normal ANN");
        assert_eq!(index.ann_fallback_count(), 0);

        // A post-build tombstone removes one native candidate. Fetching the
        // entire four-point graph must therefore exact-repair the main tier,
        // then merge the resident WAL exactly once through the same resolver.
        assert!(
            index
                .fast_index
                .soft_delete("doc-delete")
                .expect("post-build tombstone")
        );
        assert_canonical(&index, "exact-underfill fallback");
        assert_eq!(
            index.ann_fallback_count(),
            1,
            "one underfilled ANN request must increment the public counter once"
        );
    }

    #[cfg(feature = "ann")]
    #[test]
    fn ann_wal_shadowed_top_hit_stays_suppressed_through_raw_exact_fallback() {
        use crate::wal::WalEntry;

        let dir = temp_index_dir("ann-wal-shadowed-top");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[
                ("doc-a", &[0.9, 0.435_889_9]),
                ("doc-b", &[0.8, -0.6]),
                ("doc-tombstone", &[0.0, 1.0]),
            ],
        )
        .expect("write fast index");
        let config = TwoTierConfig {
            hnsw_threshold: 1,
            hnsw_ef_search: 32,
            ..TwoTierConfig::default()
        };
        let mut index = TwoTierIndex::open(&dir, config).expect("open ANN index");
        index.fast_index.wal_entries.push(WalEntry {
            doc_id: "doc-a".into(),
            doc_id_hash: crate::fnv1a_hash(b"doc-a"),
            embedding: vec![-1.0, 0.0],
        });

        let normal_query = [1.0_f32, 0.0];
        assert!(
            index
                .fast_index
                .search_top_k(&normal_query, 1, None)
                .expect("canonical normal search")
                .is_empty(),
            "the physical main winner is shadowed after top-k selection"
        );
        assert!(
            index
                .search_fast(&normal_query, 1)
                .expect("normal ANN search")
                .is_empty(),
            "normal ANN must not backfill after suppressing a WAL-shadowed winner"
        );
        assert_eq!(index.ann_fallback_count(), 0);

        assert!(
            index
                .fast_index
                .soft_delete("doc-tombstone")
                .expect("post-build tombstone")
        );
        let fallback_query = [0.0_f32, 1.0];
        assert!(
            index
                .fast_index
                .search_top_k(&fallback_query, 1, None)
                .expect("canonical fallback search")
                .is_empty(),
            "the exact main winner is still shadowed by the WAL"
        );
        assert!(
            index
                .search_fast(&fallback_query, 1)
                .expect("underfilled ANN search")
                .is_empty(),
            "raw exact repair must defer WAL suppression until after shared top-k selection"
        );
        assert_eq!(index.ann_fallback_count(), 1);
    }

    #[cfg(feature = "ann")]
    #[test]
    fn ann_two_tier_duplicate_main_ids_consume_physical_top_k_slots() {
        let dir = temp_index_dir("ann-duplicate-main-top-k");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[
                ("duplicate", &[1.0, 0.0]),
                ("duplicate", &[0.95, 0.312_249_9]),
                ("lower-unique", &[0.8, 0.6]),
            ],
        )
        .expect("write duplicate-ID fast index");
        let config = TwoTierConfig {
            hnsw_threshold: 1,
            hnsw_ef_search: 32,
            ..TwoTierConfig::default()
        };
        let index = TwoTierIndex::open(&dir, config).expect("open ANN index");
        let query = [1.0_f32, 0.0];
        let expected = index
            .fast_index
            .search_top_k(&query, 2, None)
            .expect("canonical duplicate-ID search");
        let actual = index
            .search_fast(&query, 2)
            .expect("ANN duplicate-ID search");

        assert_eq!(expected.len(), 1);
        assert_eq!(actual, expected);
        assert_eq!(actual[0].doc_id, "duplicate");
        assert!(
            actual.iter().all(|hit| hit.doc_id != "lower-unique"),
            "requesting k=2 must not overfetch and backfill a third physical row"
        );
    }

    #[cfg(feature = "ann")]
    #[test]
    fn ann_is_skipped_below_threshold() {
        let dir = temp_index_dir("ann-disabled");
        fs::create_dir_all(&dir).expect("create temp dir");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[("doc-a", &[1.0, 0.0]), ("doc-b", &[0.0, 1.0])],
        )
        .expect("write fast index");

        let config = TwoTierConfig {
            hnsw_threshold: 10_000,
            ..TwoTierConfig::default()
        };
        let index = TwoTierIndex::open(&dir, config).expect("open");
        assert!(!index.has_fast_ann());
        assert!(!dir.join(VECTOR_ANN_FAST_FILENAME).exists());
    }

    // ── Error paths ──────────────────────────────────────────────────

    #[test]
    fn open_returns_index_not_found_when_no_fast_or_fallback() {
        let dir = temp_index_dir("missing");
        fs::create_dir_all(&dir).expect("create temp dir");
        let error = TwoTierIndex::open(&dir, TwoTierConfig::default()).unwrap_err();
        let paths = match &error {
            SearchError::IndexCandidatesNotFound { paths } => paths.as_slice(),
            _ => &[],
        };
        let expected_paths = [
            dir.join(VECTOR_INDEX_FAST_FILENAME),
            dir.join(VECTOR_INDEX_FALLBACK_FILENAME),
        ];
        assert_eq!(
            paths,
            expected_paths.as_slice(),
            "unexpected error variant: {error:?}"
        );
        let message = error.to_string();
        assert!(message.contains(VECTOR_INDEX_FAST_FILENAME));
        assert!(message.contains(VECTOR_INDEX_FALLBACK_FILENAME));
    }

    #[test]
    fn open_with_paths_supports_consumer_owned_filenames() {
        let dir = temp_index_dir("explicit-paths");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join("index-fnv1a-384.fsvi");
        let quality_path = dir.join("index-minilm-384.fsvi");
        write_index_file(
            &fast_path,
            &[("doc-a", &[1.0, 0.0]), ("doc-b", &[0.0, 1.0])],
        )
        .expect("write custom fast index");
        write_index_file(
            &quality_path,
            &[("doc-a", &[0.5, 0.5]), ("doc-b", &[1.0, 0.0])],
        )
        .expect("write custom quality index");

        let paths = TwoTierIndexPaths::new(&fast_path).with_quality_index(&quality_path);
        let index = TwoTierIndex::open_with_paths(&paths, TwoTierConfig::default())
            .expect("open custom two-tier paths");

        assert_eq!(paths.fast_index(), fast_path);
        assert_eq!(paths.quality_index(), Some(quality_path.as_path()));
        assert!(index.has_quality_index());
        assert_eq!(index.doc_count(), 2);
        assert_eq!(index.fast_embedder_id(), "test");
        assert_eq!(index.fast_embedder_revision(), "");
        assert_eq!(index.quality_embedder_id(), Some("test"));
        assert_eq!(index.quality_embedder_revision(), Some(""));
        assert_eq!(index.fast_index_path(), fast_path);
        assert_eq!(index.quality_index_path(), Some(quality_path.as_path()));
        assert!(!dir.join(VECTOR_INDEX_FAST_FILENAME).exists());
        assert!(!dir.join(VECTOR_INDEX_QUALITY_FILENAME).exists());

        let hits = index.search_fast(&[1.0, 0.0], 2).expect("search fast");
        let quality_scores = index
            .quality_scores_for_hits(&[1.0, 0.0], &hits)
            .expect("score quality tier");
        assert_eq!(quality_scores.len(), hits.len());
        assert!(quality_scores.iter().all(Option::is_some));
    }

    #[test]
    fn open_with_paths_supports_custom_fast_only_without_copying() {
        let dir = temp_index_dir("explicit-fast-only");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join("index-fnv1a-384.fsvi");
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write custom fast index");

        let paths = TwoTierIndexPaths::new(&fast_path);
        let index = TwoTierIndex::open_with_paths(&paths, TwoTierConfig::default())
            .expect("open custom fast-only path");

        assert_eq!(index.fast_index_path(), fast_path);
        assert_eq!(index.quality_index_path(), None);
        assert!(!index.has_quality_index());
        assert!(!dir.join(VECTOR_INDEX_FAST_FILENAME).exists());
        assert!(!dir.join(VECTOR_INDEX_FALLBACK_FILENAME).exists());
        assert!(!dir.join(VECTOR_INDEX_QUALITY_FILENAME).exists());
    }

    #[test]
    fn open_with_paths_reports_the_explicit_missing_path() {
        let dir = temp_index_dir("explicit-missing");
        fs::create_dir_all(&dir).expect("create temp dir");
        let missing_path = dir.join("index-custom.fsvi");
        let paths = TwoTierIndexPaths::new(&missing_path);

        let error = TwoTierIndex::open_with_paths(&paths, TwoTierConfig::default()).unwrap_err();
        assert!(
            matches!(
                error,
                SearchError::IndexNotFound { ref path } if path == &missing_path
            ),
            "expected exact explicit path, got {error:?}"
        );
    }

    #[test]
    fn open_with_paths_reports_an_explicit_missing_quality_path() {
        let dir = temp_index_dir("explicit-missing-quality");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join("index-fnv1a-384.fsvi");
        let missing_quality_path = dir.join("index-minilm-384.fsvi");
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write custom fast index");
        let paths = TwoTierIndexPaths::new(&fast_path).with_quality_index(&missing_quality_path);

        let error = TwoTierIndex::open_with_paths(&paths, TwoTierConfig::default()).unwrap_err();
        assert!(
            matches!(
                error,
                SearchError::IndexNotFound { ref path } if path == &missing_quality_path
            ),
            "expected exact missing quality path, got {error:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn explicit_fast_final_symlink_to_distinct_target_is_rejected() {
        use std::os::unix::fs::symlink;

        let dir = temp_index_dir("explicit-fast-distinct-symlink");
        fs::create_dir_all(&dir).expect("create temp dir");
        let target_path = dir.join("target.fsvi");
        let symlink_path = dir.join("index-fast.fsvi");
        write_index_file(&target_path, &[("doc-a", &[1.0, 0.0])])
            .expect("write distinct fast target");
        symlink(&target_path, &symlink_path).expect("create fast final symlink");

        let error = TwoTierIndex::open_with_paths(
            &TwoTierIndexPaths::new(&symlink_path),
            TwoTierConfig::default(),
        )
        .expect_err("a distinct-target final symlink must be rejected");

        assert!(
            matches!(
                error,
                SearchError::InvalidConfig { ref field, .. } if field == "index_paths"
            ),
            "unexpected final-symlink error: {error:?}"
        );
        assert_eq!(
            fs::read(&target_path).expect("read preserved target"),
            fs::read(&symlink_path).expect("read through preserved symlink")
        );
    }

    #[cfg(unix)]
    #[test]
    fn explicit_dangling_fast_final_symlink_is_rejected_as_configuration() {
        use std::os::unix::fs::symlink;

        let dir = temp_index_dir("explicit-fast-dangling-symlink");
        fs::create_dir_all(&dir).expect("create temp dir");
        let missing_target = dir.join("missing-target.fsvi");
        let symlink_path = dir.join("index-fast.fsvi");
        symlink(&missing_target, &symlink_path).expect("create dangling fast symlink");

        let error = TwoTierIndex::open_with_paths(
            &TwoTierIndexPaths::new(&symlink_path),
            TwoTierConfig::default(),
        )
        .expect_err("a dangling final symlink must be rejected");

        assert!(
            matches!(
                error,
                SearchError::InvalidConfig { ref field, .. } if field == "index_paths"
            ),
            "unexpected dangling-symlink error: {error:?}"
        );
        assert!(!missing_target.exists());
        assert!(fs::symlink_metadata(&symlink_path).is_ok());
    }

    #[cfg(feature = "ann")]
    #[test]
    fn explicit_ann_paths_are_inspectable() {
        let fast_index = PathBuf::from("index-fast.fsvi");
        let quality_index = PathBuf::from("index-quality.fsvi");
        let fast_ann = PathBuf::from("index-fast.hnsw");
        let quality_ann = PathBuf::from("index-quality.hnsw");
        let paths = TwoTierIndexPaths::new(&fast_index)
            .with_quality_index(&quality_index)
            .with_fast_ann(&fast_ann)
            .with_quality_ann(&quality_ann);

        assert_eq!(paths.fast_index(), fast_index);
        assert_eq!(paths.quality_index(), Some(quality_index.as_path()));
        assert_eq!(paths.fast_ann(), Some(fast_ann.as_path()));
        assert_eq!(paths.quality_ann(), Some(quality_ann.as_path()));
    }

    #[cfg(feature = "ann")]
    #[test]
    fn explicit_path_roles_must_not_alias() {
        let cases = [
            TwoTierIndexPaths::new("fast.fsvi").with_quality_index("fast.fsvi"),
            TwoTierIndexPaths::new("fast.fsvi").with_fast_ann("fast.fsvi"),
            TwoTierIndexPaths::new("fast.fsvi")
                .with_quality_index("quality.fsvi")
                .with_quality_ann("fast.fsvi"),
            TwoTierIndexPaths::new("fast.fsvi")
                .with_quality_index("quality.fsvi")
                .with_fast_ann("quality.fsvi"),
            TwoTierIndexPaths::new("fast.fsvi")
                .with_quality_index("quality.fsvi")
                .with_quality_ann("quality.fsvi"),
            TwoTierIndexPaths::new("fast.fsvi")
                .with_quality_index("quality.fsvi")
                .with_fast_ann("shared.hnsw")
                .with_quality_ann("shared.hnsw"),
        ];

        for paths in cases {
            let error = validate_index_paths(&paths).expect_err("aliased roles must be rejected");
            assert!(
                matches!(
                    error,
                    SearchError::InvalidConfig { ref field, .. } if field == "index_paths"
                ),
                "unexpected alias error: {error:?}"
            );
        }
    }

    #[cfg(feature = "ann")]
    #[test]
    fn quality_ann_requires_a_quality_index() {
        let paths = TwoTierIndexPaths::new("fast.fsvi").with_quality_ann("orphan-quality.hnsw");
        let error = validate_index_paths(&paths).expect_err("orphan quality ANN must be rejected");

        assert!(
            matches!(
                error,
                SearchError::InvalidConfig { ref field, .. } if field == "quality_ann"
            ),
            "unexpected quality ANN error: {error:?}"
        );
    }

    #[cfg(feature = "ann")]
    #[test]
    fn custom_ann_path_builds_without_creating_a_conventional_sidecar() {
        let dir = temp_index_dir("explicit-custom-ann");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join("index-fnv1a-4.fsvi");
        let fast_ann_path = dir.join("index-fnv1a-4.hnsw");
        write_index_file(
            &fast_path,
            &[
                ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
                ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
            ],
        )
        .expect("write custom fast index");
        let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&fast_ann_path);
        let config = TwoTierConfig {
            hnsw_threshold: 1,
            ..TwoTierConfig::default()
        };

        let index =
            TwoTierIndex::open_with_paths(&paths, config).expect("open and build custom ANN");

        assert!(index.has_fast_ann());
        assert!(fast_ann_path.exists());
        assert!(!dir.join(VECTOR_ANN_FAST_FILENAME).exists());
    }

    #[cfg(feature = "ann")]
    #[test]
    fn native_custom_ann_reopen_does_not_materialize_a_missing_save_lock() {
        let dir = temp_index_dir("explicit-native-ann-read-only-reopen");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join("index-fnv1a-4.fsvi");
        let fast_ann_path = dir.join("index-fnv1a-4.hnsw");
        write_index_file(
            &fast_path,
            &[
                ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
                ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
            ],
        )
        .expect("write custom fast index");
        let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&fast_ann_path);
        let config = TwoTierConfig {
            hnsw_threshold: 1,
            ..TwoTierConfig::default()
        };
        let first = TwoTierIndex::open_with_paths(&paths, config.clone())
            .expect("build custom ANN sidecar");
        assert!(first.has_fast_ann());
        load_native_ann_sidecar(&fast_ann_path, &first.fast_index);
        drop(first);

        let lock_path =
            crate::hnsw::hnsw_save_lock_artifact_path(&fast_ann_path).expect("lock path");
        let retained_lock = lock_path.with_extension("lock.retained-for-native-reopen");
        fs::rename(&lock_path, &retained_lock).expect("retain lock under a non-active name");
        assert!(!lock_path.exists());
        assert!(retained_lock.exists());

        let reopened =
            TwoTierIndex::open_with_paths(&paths, config).expect("native read-only-style reopen");
        assert!(reopened.has_fast_ann());
        assert!(
            !lock_path.exists(),
            "a native-valid reopen must not require or recreate a writable save lock"
        );
        assert!(retained_lock.exists());
    }

    #[cfg(feature = "ann")]
    #[test]
    fn custom_ann_save_creates_a_missing_parent_and_reopens_natively() {
        let dir = temp_index_dir("explicit-custom-ann-missing-parent");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join("index-fnv1a-4.fsvi");
        let fast_ann_path = dir
            .join("consumer-owned")
            .join("ann")
            .join("index-fnv1a-4.hnsw");
        write_index_file(
            &fast_path,
            &[
                ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
                ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
            ],
        )
        .expect("write custom fast index");
        assert!(!fast_ann_path.parent().expect("ANN parent").exists());
        let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&fast_ann_path);
        let config = TwoTierConfig {
            hnsw_threshold: 1,
            ..TwoTierConfig::default()
        };

        let first = TwoTierIndex::open_with_paths(&paths, config.clone())
            .expect("build ANN below a newly created custom parent");
        assert!(first.has_fast_ann());
        load_native_ann_sidecar(&fast_ann_path, &first.fast_index);
        drop(first);

        let reopened =
            TwoTierIndex::open_with_paths(&paths, config).expect("native custom ANN reopen");
        assert!(reopened.has_fast_ann());
        load_native_ann_sidecar(&fast_ann_path, &reopened.fast_index);
        assert!(!dir.join(VECTOR_ANN_FAST_FILENAME).exists());
    }

    #[cfg(feature = "ann")]
    #[test]
    fn custom_fast_and_quality_ann_paths_reopen_through_native_sidecars() {
        let dir = temp_index_dir("explicit-custom-two-tier-ann");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join("index-fnv1a-4.fsvi");
        let quality_path = dir.join("index-minilm-4.fsvi");
        let fast_ann_path = dir.join("index-fnv1a-4.hnsw");
        let quality_ann_path = dir.join("index-minilm-4.hnsw");
        let rows = [
            ("doc-a", &[1.0, 0.0, 0.0, 0.0][..]),
            ("doc-b", &[0.0, 1.0, 0.0, 0.0][..]),
            ("doc-c", &[0.0, 0.0, 1.0, 0.0][..]),
        ];
        write_index_file(&fast_path, &rows).expect("write custom fast index");
        write_index_file(&quality_path, &rows).expect("write custom quality index");
        let paths = TwoTierIndexPaths::new(&fast_path)
            .with_quality_index(&quality_path)
            .with_fast_ann(&fast_ann_path)
            .with_quality_ann(&quality_ann_path);
        let config = TwoTierConfig {
            hnsw_threshold: 1,
            ..TwoTierConfig::default()
        };

        let first = TwoTierIndex::open_with_paths(&paths, config.clone())
            .expect("build both custom ANN tiers");
        assert!(first.has_fast_ann());
        assert!(first.has_quality_ann());
        load_native_ann_sidecar(&fast_ann_path, &first.fast_index);
        load_native_ann_sidecar(
            &quality_ann_path,
            first.quality_index.as_ref().expect("quality index"),
        );
        let fast_metadata = fs::read(&fast_ann_path).expect("read fast ANN metadata");
        let quality_metadata = fs::read(&quality_ann_path).expect("read quality ANN metadata");
        drop(first);

        let reopened =
            TwoTierIndex::open_with_paths(&paths, config).expect("reopen both custom ANN tiers");
        assert!(reopened.has_fast_ann());
        assert!(reopened.has_quality_ann());
        load_native_ann_sidecar(&fast_ann_path, &reopened.fast_index);
        load_native_ann_sidecar(
            &quality_ann_path,
            reopened.quality_index.as_ref().expect("quality index"),
        );
        assert_eq!(
            fs::read(&fast_ann_path).expect("reread fast ANN metadata"),
            fast_metadata,
            "native reopen must not republish fast ANN metadata"
        );
        assert_eq!(
            fs::read(&quality_ann_path).expect("reread quality ANN metadata"),
            quality_metadata,
            "native reopen must not republish quality ANN metadata"
        );
        assert!(!dir.join(VECTOR_ANN_FAST_FILENAME).exists());
        assert!(!dir.join(VECTOR_ANN_QUALITY_FILENAME).exists());
    }

    #[cfg(feature = "ann")]
    #[test]
    fn custom_quality_only_ann_path_does_not_enable_or_create_fast_ann() {
        let dir = temp_index_dir("explicit-custom-quality-only-ann");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join("index-fnv1a-4.fsvi");
        let quality_path = dir.join("index-minilm-4.fsvi");
        let quality_ann_path = dir.join("index-minilm-4.hnsw");
        let rows = [
            ("doc-a", &[1.0, 0.0, 0.0, 0.0][..]),
            ("doc-b", &[0.0, 1.0, 0.0, 0.0][..]),
        ];
        write_index_file(&fast_path, &rows).expect("write custom fast index");
        write_index_file(&quality_path, &rows).expect("write custom quality index");
        let paths = TwoTierIndexPaths::new(&fast_path)
            .with_quality_index(&quality_path)
            .with_quality_ann(&quality_ann_path);
        let config = TwoTierConfig {
            hnsw_threshold: 1,
            ..TwoTierConfig::default()
        };

        let first = TwoTierIndex::open_with_paths(&paths, config.clone())
            .expect("build only the custom quality ANN tier");
        assert!(!first.has_fast_ann());
        assert!(first.has_quality_ann());
        load_native_ann_sidecar(
            &quality_ann_path,
            first.quality_index.as_ref().expect("quality index"),
        );
        drop(first);

        let reopened =
            TwoTierIndex::open_with_paths(&paths, config).expect("reopen quality-only custom ANN");
        assert!(!reopened.has_fast_ann());
        assert!(reopened.has_quality_ann());
        load_native_ann_sidecar(
            &quality_ann_path,
            reopened.quality_index.as_ref().expect("quality index"),
        );
        assert!(!dir.join("index-fnv1a-4.hnsw").exists());
        assert!(!dir.join(VECTOR_ANN_FAST_FILENAME).exists());
        assert!(!dir.join(VECTOR_ANN_QUALITY_FILENAME).exists());
    }

    #[cfg(feature = "ann")]
    #[test]
    fn missing_ann_role_aliases_follow_the_mounted_filesystem_not_the_os_name() {
        let dir = temp_index_dir("ann-volume-collation");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join("fast.fsvi");
        let quality_path = dir.join("quality.fsvi");
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast index");
        write_index_file(&quality_path, &[("doc-a", &[1.0, 0.0])]).expect("write quality index");

        for (label, probe_name, alternate_probe_name, left_name, right_name) in [
            (
                "ascii",
                "ascii-Probe",
                "ASCII-pROBE",
                "shared-ascii.hnsw",
                "SHARED-ASCII.HNSW",
            ),
            (
                "unicode-full-case-fold",
                "unicode-Straße",
                "unicode-STRASSE",
                "shared-Straße.hnsw",
                "shared-STRASSE.hnsw",
            ),
            (
                "unicode-one-to-one-case",
                "unicode-É-probe",
                "unicode-é-probe",
                "shared-É.hnsw",
                "shared-é.hnsw",
            ),
            (
                "unicode-normalization",
                "unicode-é-normalization-probe",
                "unicode-e\u{301}-normalization-probe",
                "shared-é-normalization.hnsw",
                "shared-e\u{301}-normalization.hnsw",
            ),
        ] {
            let probe = dir.join(probe_name);
            fs::write(&probe, b"filesystem collation probe").expect("write collation probe");
            let alternate_probe = dir.join(alternate_probe_name);
            let volume_aliases = crate::file_identity::is_same_file(&probe, &alternate_probe)
                .unwrap_or_else(|error| {
                    assert_eq!(
                        error.kind(),
                        std::io::ErrorKind::NotFound,
                        "unexpected collation probe error"
                    );
                    false
                });
            eprintln!(
                "bd-07os filesystem-alias probe: label={label} primary={} alternate={} \
                 aliases={volume_aliases}",
                probe.display(),
                alternate_probe.display()
            );

            let paths = TwoTierIndexPaths::new(&fast_path)
                .with_quality_index(&quality_path)
                .with_fast_ann(dir.join(left_name))
                .with_quality_ann(dir.join(right_name));
            validate_index_paths(&paths).expect("missing leaves are initially distinct");
            let validation = validate_ann_save_lock_identities(&paths, true, true);
            if volume_aliases {
                assert!(
                    matches!(
                        validation,
                        Err(SearchError::InvalidConfig { ref field, .. })
                            if field == "index_paths"
                    ),
                    "mounted filesystem must reject {label} ANN aliases: {validation:?}"
                );
            } else {
                validation.unwrap_or_else(|error| {
                    panic!(
                        "mounted filesystem must preserve distinct {label} ANN roles: \
                         {error:?}"
                    )
                });
            }
        }
    }

    #[cfg(feature = "ann")]
    #[test]
    fn ann_persistence_revalidation_rejects_new_save_lock_alias() {
        let dir = temp_index_dir("ann-save-lock-alias-race");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join("index-fast.fsvi");
        let missing_quality_path = dir.join("quality-will-alias-lock.fsvi");
        let ann_path = dir.join("index-fast.hnsw");
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast index");
        let paths = TwoTierIndexPaths::new(&fast_path)
            .with_quality_index(&missing_quality_path)
            .with_fast_ann(&ann_path);

        validate_ann_persistence_paths(&paths, true, false)
            .expect("initial roles and materialized save lock are distinct");
        let lock_path =
            crate::hnsw::hnsw_save_lock_artifact_path(&ann_path).expect("derive save-lock path");
        fs::hard_link(&lock_path, &missing_quality_path)
            .expect("inject a configured-role alias to the materialized save lock");

        let error = validate_ann_persistence_paths(&paths, true, false)
            .expect_err("complete immediate pre-save validation must reject the new lock alias");
        assert!(
            matches!(
                error,
                SearchError::InvalidConfig { ref field, .. } if field == "index_paths"
            ),
            "unexpected save-lock alias error: {error:?}"
        );
        assert!(
            crate::file_identity::is_same_file(&lock_path, &missing_quality_path)
                .expect("compare injected save-lock alias")
        );
    }

    #[cfg(all(feature = "ann", unix))]
    #[test]
    fn fresh_ann_save_revalidates_paths_after_an_alias_race() {
        use std::cell::Cell;

        let dir = temp_index_dir("ann-fresh-save-alias-race");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join("index-fnv1a-4.fsvi");
        let ann_path = dir.join("index-fnv1a-4.hnsw");
        write_index_file(
            &fast_path,
            &[
                ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
                ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
            ],
        )
        .expect("write custom fast index");
        let original = fs::read(&fast_path).expect("read original FSVI");
        let vector_index = VectorIndex::open(&fast_path).expect("open custom fast index");
        let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&ann_path);
        validate_index_paths(&paths).expect("paths start distinct");
        let save_called = Cell::new(false);
        let config = TwoTierConfig {
            hnsw_threshold: 1,
            ..TwoTierConfig::default()
        };

        let ann = maybe_load_or_build_ann_with_save(
            &vector_index,
            &ann_path,
            1,
            &config,
            "fast",
            || {
                fs::hard_link(&fast_path, &ann_path).expect("inject hardlink alias before save");
                validate_ann_persistence_paths(&paths, true, false)
            },
            |_, _| {
                save_called.set(true);
                Ok(())
            },
        )
        .expect("keep freshly built ANN in memory after rejected persistence");

        assert!(
            !save_called.get(),
            "save must not run after revalidation fails"
        );
        assert_eq!(ann.len(), 2);
        assert!(
            crate::file_identity::is_same_file(&fast_path, &ann_path).expect("compare aliases")
        );
        assert_eq!(fs::read(&fast_path).expect("read preserved FSVI"), original);
    }

    #[cfg(all(feature = "ann", unix))]
    #[test]
    fn rebuilt_ann_resave_revalidates_every_role_after_an_alias_race() {
        use std::cell::Cell;

        let dir = temp_index_dir("ann-rebuilt-save-alias-race");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join("index-fnv1a-4.fsvi");
        let quality_path = dir.join("index-minilm-4.fsvi");
        let fast_ann_path = dir.join("index-fnv1a-4.hnsw");
        let quality_ann_path = dir.join("index-minilm-4.hnsw");
        let rows = [
            ("doc-a", &[1.0, 0.0, 0.0, 0.0][..]),
            ("doc-b", &[0.0, 1.0, 0.0, 0.0][..]),
        ];
        write_index_file(&fast_path, &rows).expect("write custom fast index");
        write_index_file(&quality_path, &rows).expect("write custom quality index");
        let paths = TwoTierIndexPaths::new(&fast_path)
            .with_quality_index(&quality_path)
            .with_fast_ann(&fast_ann_path)
            .with_quality_ann(&quality_ann_path);
        let config = TwoTierConfig {
            hnsw_threshold: 1,
            ..TwoTierConfig::default()
        };
        let seeded = TwoTierIndex::open_with_paths(
            &TwoTierIndexPaths::new(&fast_path).with_fast_ann(&fast_ann_path),
            config.clone(),
        )
        .expect("seed fast ANN sidecar");
        assert!(seeded.has_fast_ann());
        drop(seeded);

        let mut metadata: serde_json::Value =
            serde_json::from_slice(&fs::read(&fast_ann_path).expect("read ANN metadata"))
                .expect("parse ANN metadata");
        let object = metadata.as_object_mut().expect("ANN metadata object");
        object.remove("format_version");
        object.remove("sidecar_generation");
        object.remove("sidecar_basename");
        fs::write(
            &fast_ann_path,
            serde_json::to_vec(&metadata).expect("serialize legacy metadata"),
        )
        .expect("write legacy ANN metadata");

        let vector_index = VectorIndex::open(&fast_path).expect("reopen custom fast index");
        validate_index_paths(&paths).expect("paths start distinct");
        let save_called = Cell::new(false);
        let quality_original = fs::read(&quality_path).expect("read original quality FSVI");
        let ann = maybe_load_or_build_ann_with_save(
            &vector_index,
            &fast_ann_path,
            1,
            &config,
            "fast",
            || {
                fs::hard_link(&quality_path, &quality_ann_path)
                    .expect("inject quality-role hardlink alias before rebuilt resave");
                validate_ann_persistence_paths(&paths, true, true)
            },
            |_, _| {
                save_called.set(true);
                Ok(())
            },
        )
        .expect("keep rebuilt ANN in memory after rejected resave");

        assert!(
            !save_called.get(),
            "resave must not run after revalidation fails"
        );
        assert_eq!(ann.len(), 2);
        assert!(
            crate::file_identity::is_same_file(&quality_path, &quality_ann_path)
                .expect("compare aliases")
        );
        assert_eq!(
            fs::read(&quality_path).expect("read preserved quality FSVI"),
            quality_original
        );
    }

    #[cfg(feature = "ann")]
    #[test]
    fn ann_path_equal_to_fsvi_is_rejected_before_any_overwrite() {
        let dir = temp_index_dir("ann-direct-alias");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join("index-fnv1a-4.fsvi");
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])])
            .expect("write custom fast index");
        let original = fs::read(&fast_path).expect("read original FSVI");
        let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&fast_path);
        let config = TwoTierConfig {
            hnsw_threshold: 1,
            ..TwoTierConfig::default()
        };

        let error =
            TwoTierIndex::open_with_paths(&paths, config).expect_err("alias must be rejected");

        assert!(matches!(error, SearchError::InvalidConfig { .. }));
        assert_eq!(fs::read(&fast_path).expect("read preserved FSVI"), original);
    }

    #[cfg(all(feature = "ann", unix))]
    #[test]
    fn ann_hardlink_alias_is_rejected_before_any_overwrite() {
        let dir = temp_index_dir("ann-hardlink-alias");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join("index-fnv1a-4.fsvi");
        let ann_path = dir.join("index-fnv1a-4.hnsw");
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])])
            .expect("write custom fast index");
        fs::hard_link(&fast_path, &ann_path).expect("create hardlink alias");
        let original = fs::read(&fast_path).expect("read original FSVI");
        let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&ann_path);
        let config = TwoTierConfig {
            hnsw_threshold: 1,
            ..TwoTierConfig::default()
        };

        let error =
            TwoTierIndex::open_with_paths(&paths, config).expect_err("alias must be rejected");

        assert!(matches!(error, SearchError::InvalidConfig { .. }));
        assert_eq!(fs::read(&fast_path).expect("read preserved FSVI"), original);
    }

    #[cfg(all(feature = "ann", unix))]
    #[test]
    fn ann_symlink_alias_is_rejected_before_any_overwrite() {
        use std::os::unix::fs::symlink;

        let dir = temp_index_dir("ann-symlink-alias");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join("index-fnv1a-4.fsvi");
        let ann_path = dir.join("index-fnv1a-4.hnsw");
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])])
            .expect("write custom fast index");
        symlink(&fast_path, &ann_path).expect("create symlink alias");
        let original = fs::read(&fast_path).expect("read original FSVI");
        let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&ann_path);
        let config = TwoTierConfig {
            hnsw_threshold: 1,
            ..TwoTierConfig::default()
        };

        let error =
            TwoTierIndex::open_with_paths(&paths, config).expect_err("alias must be rejected");

        assert!(matches!(error, SearchError::InvalidConfig { .. }));
        assert_eq!(fs::read(&fast_path).expect("read preserved FSVI"), original);
    }

    #[cfg(all(feature = "ann", unix))]
    #[test]
    fn ann_final_symlink_to_distinct_target_is_rejected_before_load_or_save() {
        use std::os::unix::fs::symlink;

        let dir = temp_index_dir("ann-distinct-final-symlink");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join("index-fnv1a-4.fsvi");
        let ann_target = dir.join("distinct-target.hnsw");
        let ann_path = dir.join("index-fnv1a-4.hnsw");
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])])
            .expect("write custom fast index");
        fs::write(&ann_target, b"preserve distinct ANN target").expect("write distinct ANN target");
        let original_target = fs::read(&ann_target).expect("read original ANN target");
        symlink(&ann_target, &ann_path).expect("create distinct-target ANN symlink");
        let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&ann_path);
        let config = TwoTierConfig {
            hnsw_threshold: 1,
            ..TwoTierConfig::default()
        };

        let error = TwoTierIndex::open_with_paths(&paths, config)
            .expect_err("a distinct-target ANN final symlink must be rejected");

        assert!(
            matches!(
                error,
                SearchError::InvalidConfig { ref field, .. } if field == "index_paths"
            ),
            "unexpected final-symlink error: {error:?}"
        );
        assert_eq!(
            fs::read(&ann_target).expect("read preserved ANN target"),
            original_target
        );
    }

    #[cfg(all(feature = "ann", unix))]
    #[test]
    fn dangling_ann_final_symlink_is_rejected_before_parent_or_lock_creation() {
        use std::os::unix::fs::symlink;

        let dir = temp_index_dir("ann-dangling-final-symlink");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join("index-fnv1a-4.fsvi");
        let missing_target = dir.join("missing-target.hnsw");
        let ann_path = dir.join("index-fnv1a-4.hnsw");
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])])
            .expect("write custom fast index");
        symlink(&missing_target, &ann_path).expect("create dangling ANN symlink");
        let lock_path =
            crate::hnsw::hnsw_save_lock_artifact_path(&ann_path).expect("derive save-lock path");
        let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&ann_path);
        let config = TwoTierConfig {
            hnsw_threshold: 1,
            ..TwoTierConfig::default()
        };

        let error = TwoTierIndex::open_with_paths(&paths, config)
            .expect_err("a dangling ANN final symlink must be rejected");

        assert!(
            matches!(
                error,
                SearchError::InvalidConfig { ref field, .. } if field == "index_paths"
            ),
            "unexpected dangling-symlink error: {error:?}"
        );
        assert!(!missing_target.exists());
        assert!(!lock_path.exists());
        assert!(fs::symlink_metadata(&ann_path).is_ok());
    }

    #[cfg(all(feature = "ann", unix))]
    #[test]
    fn ann_symlinked_ancestor_with_parent_component_cannot_alias_fsvi() {
        use std::os::unix::fs::symlink;

        let dir = temp_index_dir("ann-symlink-ancestor-parent");
        let artifact_dir = dir.join("artifacts");
        let nested_dir = artifact_dir.join("nested");
        let indirection_dir = dir.join("indirection");
        fs::create_dir_all(&nested_dir).expect("create artifact directories");
        fs::create_dir_all(&indirection_dir).expect("create indirection directory");

        let fast_path = artifact_dir.join("index-fnv1a-4.fsvi");
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])])
            .expect("write custom fast index");
        let original = fs::read(&fast_path).expect("read original FSVI");

        let ancestor_link = indirection_dir.join("link");
        symlink(&nested_dir, &ancestor_link).expect("create symlinked ancestor");
        let ann_path = ancestor_link.join("..").join("index-fnv1a-4.fsvi");
        assert_eq!(
            fs::canonicalize(&ann_path).expect("resolve OS path semantics"),
            fs::canonicalize(&fast_path).expect("resolve fast path")
        );

        let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&ann_path);
        let config = TwoTierConfig {
            hnsw_threshold: 1,
            ..TwoTierConfig::default()
        };
        let error =
            TwoTierIndex::open_with_paths(&paths, config).expect_err("alias must be rejected");

        assert!(matches!(error, SearchError::InvalidConfig { .. }));
        assert_eq!(fs::read(&fast_path).expect("read preserved FSVI"), original);
    }

    #[cfg(all(feature = "ann", unix))]
    #[test]
    fn missing_ann_leaves_cannot_alias_through_a_symlinked_ancestor_and_parent_component() {
        use std::os::unix::fs::symlink;

        let dir = temp_index_dir("ann-missing-symlink-ancestor-parent");
        let artifact_dir = dir.join("artifacts");
        let nested_dir = artifact_dir.join("nested");
        let indirection_dir = dir.join("indirection");
        fs::create_dir_all(&nested_dir).expect("create artifact directories");
        fs::create_dir_all(&indirection_dir).expect("create indirection directory");

        let fast_path = artifact_dir.join("fast.fsvi");
        let quality_path = artifact_dir.join("quality.fsvi");
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast index");
        write_index_file(&quality_path, &[("doc-a", &[1.0, 0.0])]).expect("write quality index");

        let ancestor_link = indirection_dir.join("link");
        symlink(&nested_dir, &ancestor_link).expect("create symlinked ancestor");
        let fast_ann_path = artifact_dir.join("shared.hnsw");
        let quality_ann_path = ancestor_link.join("..").join("shared.hnsw");
        assert!(!fast_ann_path.exists());
        assert!(!quality_ann_path.exists());

        let paths = TwoTierIndexPaths::new(&fast_path)
            .with_quality_index(&quality_path)
            .with_fast_ann(&fast_ann_path)
            .with_quality_ann(&quality_ann_path);
        let error = validate_index_paths(&paths)
            .expect_err("missing leaves with the same OS-resolved identity must be rejected");

        assert!(
            matches!(
                error,
                SearchError::InvalidConfig { ref field, .. } if field == "index_paths"
            ),
            "unexpected missing-leaf alias error: {error:?}"
        );
        assert!(!fast_ann_path.exists());
        assert!(!quality_ann_path.exists());
    }

    #[test]
    fn quality_scores_dimension_mismatch() {
        let dir = temp_index_dir("quality-dim");
        fs::create_dir_all(&dir).expect("create temp dir");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);

        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])])
            .expect("write fast index");
        write_index_file(&quality_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])])
            .expect("write quality index");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        assert!(index.has_quality_index());

        // Query dimension (4) doesn't match quality dimension (6)
        let hits = vec![VectorHit {
            index: 0,
            score: 0.0,
            doc_id: "doc-a".into(),
        }];
        let error = index
            .quality_scores_for_hits(&[1.0, 0.0, 0.0, 0.0], &hits)
            .unwrap_err();
        assert!(
            matches!(
                error,
                SearchError::DimensionMismatch {
                    expected: 6,
                    found: 4
                }
            ),
            "expected DimensionMismatch, got {error:?}"
        );
    }

    #[test]
    fn has_quality_for_index_out_of_bounds_returns_false() {
        let dir = temp_index_dir("oob");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast index");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        assert!(!index.has_quality_for_index(999));
    }

    // ── Builder error paths ─────────────────────────────────────────

    #[test]
    fn builder_finish_rejects_empty_fast_records() {
        let dir = temp_index_dir("empty-builder");
        let builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        let error = builder.finish().unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "fast_records"),
            "expected InvalidConfig for fast_records, got {error:?}"
        );
    }

    #[test]
    fn builder_rejects_inconsistent_quality_dimension() {
        let dir = temp_index_dir("bad-quality-dim");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .add_fast_record("doc-a", &[1.0, 0.0, 0.0])
            .expect("fast record");
        builder
            .add_quality_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("first quality");
        let error = builder
            .add_quality_record("doc-b", &[1.0, 0.0])
            .unwrap_err();
        assert!(
            matches!(
                error,
                SearchError::DimensionMismatch {
                    expected: 4,
                    found: 2
                }
            ),
            "expected DimensionMismatch, got {error:?}"
        );
    }

    // ── Fast-tier with explicit fast.idx (not fallback) ─────────────

    #[test]
    fn opens_with_explicit_fast_index() {
        let dir = temp_index_dir("explicit-fast");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[("doc-x", &[0.0, 1.0, 0.0]), ("doc-y", &[0.0, 0.0, 1.0])],
        )
        .expect("write fast index");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        assert_eq!(index.doc_count(), 2);
        let ids: Vec<String> = index
            .iter_doc_ids()
            .collect::<SearchResult<_>>()
            .expect("ids");
        assert_eq!(ids, vec!["doc-x".to_owned(), "doc-y".to_owned()]);

        let hits = index.search_fast(&[0.0, 0.0, 1.0], 1).expect("fast search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-y");
    }

    // ── Accessors ───────────────────────────────────────────────────

    #[test]
    fn config_accessor_returns_construction_config() {
        let dir = temp_index_dir("config-acc");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast index");

        let config = TwoTierConfig {
            hnsw_threshold: 42,
            ..TwoTierConfig::default()
        };
        let index = TwoTierIndex::open(&dir, config).expect("open");
        assert_eq!(index.config().hnsw_threshold, 42);
    }

    // ── Quality alignment: unmatched doc_ids ────────────────────────

    #[test]
    fn quality_index_with_extra_doc_ids_still_opens() {
        let dir = temp_index_dir("quality-extra");
        fs::create_dir_all(&dir).expect("create temp dir");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);

        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast");
        // Quality has a doc_id not in fast — should trigger warning but still open
        write_index_file(
            &quality_path,
            &[
                ("doc-a", &[1.0, 0.0]),
                ("doc-z", &[0.0, 1.0]), // extra
            ],
        )
        .expect("write quality");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        assert!(index.has_quality_index());
        assert!(index.has_quality_for_index(0)); // doc-a matched
        assert_eq!(index.doc_count(), 1); // only fast-tier docs counted
    }

    // ── Builder: fast+quality via add_record convenience ────────────

    #[test]
    fn builder_add_record_with_none_quality_skips_quality_tier() {
        let dir = temp_index_dir("add-record-no-quality");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .add_record("doc-a", &[1.0, 0.0], None)
            .expect("add doc-a");
        builder
            .add_record("doc-b", &[0.0, 1.0], None)
            .expect("add doc-b");

        let index = builder.finish().expect("finish");
        assert_eq!(index.doc_count(), 2);
        assert!(!index.has_quality_index());
    }

    #[cfg(feature = "ann")]
    #[test]
    fn ann_sidecar_rebuilds_when_config_changes() {
        let dir = temp_index_dir("ann-rebuild-config");
        fs::create_dir_all(&dir).expect("create temp dir");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[
                ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
                ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
                ("doc-c", &[0.0, 0.0, 1.0, 0.0]),
            ],
        )
        .expect("write fast index");

        let initial = TwoTierConfig {
            hnsw_threshold: 1,
            hnsw_m: 8,
            hnsw_ef_construction: 64,
            hnsw_ef_search: 16,
            ..TwoTierConfig::default()
        };
        let first_open = TwoTierIndex::open(&dir, initial).expect("open with initial ann config");
        assert!(first_open.has_fast_ann());

        let ann_path = dir.join(VECTOR_ANN_FAST_FILENAME);
        let before = load_native_ann_sidecar(&ann_path, &first_open.fast_index);
        let before_config = before.config();
        assert_eq!(before_config.m, 8);
        assert_eq!(before_config.ef_construction, 64);
        assert_eq!(before_config.ef_search, 16);

        let updated = TwoTierConfig {
            hnsw_threshold: 1,
            hnsw_m: 24,
            hnsw_ef_construction: 96,
            hnsw_ef_search: 48,
            ..TwoTierConfig::default()
        };
        let second_open = TwoTierIndex::open(&dir, updated).expect("open with updated ann config");
        assert!(second_open.has_fast_ann());

        let after = load_native_ann_sidecar(&ann_path, &second_open.fast_index);
        let after_config = after.config();
        assert_eq!(after_config.m, 24);
        assert_eq!(after_config.ef_construction, 96);
        assert_eq!(after_config.ef_search, 48);
    }

    #[cfg(feature = "ann")]
    #[test]
    fn ann_sidecar_rebuilds_when_vectors_change_with_same_doc_ids() {
        let dir = temp_index_dir("ann-rebuild-vectors");
        fs::create_dir_all(&dir).expect("create temp dir");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[("doc-a", &[1.0, 0.0, 0.0]), ("doc-b", &[0.0, 1.0, 0.0])],
        )
        .expect("write initial fast index");

        let config = TwoTierConfig {
            hnsw_threshold: 1,
            hnsw_ef_search: 64,
            ..TwoTierConfig::default()
        };
        let first = TwoTierIndex::open(&dir, config.clone()).expect("open initial");
        assert!(first.has_fast_ann());
        let before = first
            .search_fast(&[1.0, 0.0, 0.0], 1)
            .expect("search before");
        assert_eq!(before[0].doc_id, "doc-a");

        // Same doc IDs/order, but vectors are swapped. Sidecar must rebuild.
        write_index_file(
            &fast_path,
            &[("doc-a", &[0.0, 1.0, 0.0]), ("doc-b", &[1.0, 0.0, 0.0])],
        )
        .expect("rewrite fast index");

        let reopened = TwoTierIndex::open(&dir, config).expect("reopen");
        assert!(reopened.has_fast_ann());
        let ann_path = dir.join(VECTOR_ANN_FAST_FILENAME);
        load_native_ann_sidecar(&ann_path, &reopened.fast_index);
        let after = reopened
            .search_fast(&[1.0, 0.0, 0.0], 1)
            .expect("search after");
        assert_eq!(
            after[0].doc_id, "doc-b",
            "ANN sidecar should rebuild when vector content changes"
        );
    }

    #[cfg(feature = "ann")]
    #[test]
    fn legacy_ann_fallback_is_persisted_for_native_second_load() {
        let dir = temp_index_dir("ann-self-heal-legacy");
        fs::create_dir_all(&dir).expect("create temp dir");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[("doc-a", &[1.0, 0.0, 0.0]), ("doc-b", &[0.0, 1.0, 0.0])],
        )
        .expect("write fast index");

        let config = TwoTierConfig {
            hnsw_threshold: 1,
            ..TwoTierConfig::default()
        };
        let initial = TwoTierIndex::open(&dir, config.clone()).expect("seed ANN sidecar");
        assert!(initial.has_fast_ann());

        let ann_path = dir.join(VECTOR_ANN_FAST_FILENAME);
        let mut metadata: serde_json::Value =
            serde_json::from_slice(&fs::read(&ann_path).expect("read ANN metadata"))
                .expect("parse ANN metadata");
        let object = metadata.as_object_mut().expect("ANN metadata object");
        object.remove("format_version");
        object.remove("sidecar_generation");
        object.remove("sidecar_basename");
        fs::write(
            &ann_path,
            serde_json::to_vec(&metadata).expect("serialize legacy metadata"),
        )
        .expect("write legacy metadata");

        let (_, disposition) = HnswIndex::load_with_disposition(&ann_path, &initial.fast_index)
            .expect("legacy fallback rebuild");
        assert_eq!(disposition, HnswLoadDisposition::Rebuilt);

        let reopened = TwoTierIndex::open(&dir, config).expect("self-heal legacy ANN sidecar");
        assert!(reopened.has_fast_ann());
        load_native_ann_sidecar(&ann_path, &reopened.fast_index);
    }

    #[cfg(feature = "ann")]
    #[test]
    fn degraded_ann_fallback_is_persisted_for_native_second_load() {
        let dir = temp_index_dir("ann-self-heal-degraded");
        fs::create_dir_all(&dir).expect("create temp dir");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[("doc-a", &[1.0, 0.0, 0.0]), ("doc-b", &[0.0, 1.0, 0.0])],
        )
        .expect("write fast index");

        let config = TwoTierConfig {
            hnsw_threshold: 1,
            ..TwoTierConfig::default()
        };
        let initial = TwoTierIndex::open(&dir, config.clone()).expect("seed ANN sidecar");
        assert!(initial.has_fast_ann());

        let ann_path = dir.join(VECTOR_ANN_FAST_FILENAME);
        let missing_generation = ".missing-hnsw-generation";
        let mut metadata: serde_json::Value =
            serde_json::from_slice(&fs::read(&ann_path).expect("read ANN metadata"))
                .expect("parse ANN metadata");
        metadata["sidecar_generation"] = missing_generation.into();
        fs::write(
            &ann_path,
            serde_json::to_vec(&metadata).expect("serialize degraded metadata"),
        )
        .expect("write degraded metadata");

        let (_, disposition) = HnswIndex::load_with_disposition(&ann_path, &initial.fast_index)
            .expect("degraded fallback rebuild");
        assert_eq!(disposition, HnswLoadDisposition::Rebuilt);

        let reopened = TwoTierIndex::open(&dir, config).expect("self-heal degraded ANN sidecar");
        assert!(reopened.has_fast_ann());
        load_native_ann_sidecar(&ann_path, &reopened.fast_index);
        let repaired: serde_json::Value =
            serde_json::from_slice(&fs::read(&ann_path).expect("read repaired metadata"))
                .expect("parse repaired metadata");
        assert_ne!(
            repaired["sidecar_generation"].as_str(),
            Some(missing_generation),
            "self-heal must publish metadata naming the new native generation"
        );
    }

    #[cfg(feature = "ann")]
    #[test]
    fn ann_persistence_failure_keeps_rebuilt_index_in_memory() {
        fn reject_persistence(_: &HnswIndex, _: &Path) -> SearchResult<()> {
            Err(SearchError::Io(std::io::Error::other(
                "injected ANN persistence failure",
            )))
        }

        let dir = temp_index_dir("ann-persist-failure");
        fs::create_dir_all(&dir).expect("create temp dir");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[("doc-a", &[1.0, 0.0, 0.0]), ("doc-b", &[0.0, 1.0, 0.0])],
        )
        .expect("write fast index");

        let ann_path = dir.join(VECTOR_ANN_FAST_FILENAME);
        let config = TwoTierConfig {
            hnsw_threshold: 1,
            hnsw_ef_search: 64,
            ..TwoTierConfig::default()
        };
        let initial = TwoTierIndex::open(&dir, config.clone()).expect("seed ANN sidecar");
        assert!(initial.has_fast_ann());

        // Force the successful-load fallback path, then inject a persistence
        // failure at the exact save seam. Startup must retain the rebuilt ANN
        // rather than turn a repair failure into a brute-force fallback.
        let mut metadata: serde_json::Value =
            serde_json::from_slice(&fs::read(&ann_path).expect("read ANN metadata"))
                .expect("parse ANN metadata");
        let object = metadata.as_object_mut().expect("ANN metadata object");
        object.remove("format_version");
        object.remove("sidecar_generation");
        object.remove("sidecar_basename");
        fs::write(
            &ann_path,
            serde_json::to_vec(&metadata).expect("serialize legacy metadata"),
        )
        .expect("write legacy metadata");

        let rebuilt = maybe_load_or_build_ann_with_save(
            &initial.fast_index,
            &ann_path,
            1,
            &config,
            "fast",
            || Ok(()),
            reject_persistence,
        )
        .expect("retain rebuilt ANN after persistence failure");

        // The injected failure left legacy metadata installed, proving the
        // error occurred after a successful rebuild rather than after a native
        // reload. That means the next startup will retry the repair.
        let (_, disposition) = HnswIndex::load_with_disposition(&ann_path, &initial.fast_index)
            .expect("legacy sidecar still rebuilds after failed persistence");
        assert_eq!(disposition, HnswLoadDisposition::Rebuilt);

        let hits = rebuilt
            .knn_search(&[1.0, 0.0, 0.0], 2, config.hnsw_ef_search)
            .expect("search in-memory rebuilt ANN");
        assert_eq!(
            hits.len(),
            2,
            "a retained rebuilt ANN must keep every live point reachable"
        );
        let mut hit_ids: Vec<&str> = hits.iter().map(|hit| hit.doc_id.as_str()).collect();
        hit_ids.sort_unstable();
        assert_eq!(hit_ids, ["doc-a", "doc-b"]);
    }

    #[cfg(feature = "ann")]
    #[test]
    fn ann_search_excludes_tombstoned_docs() {
        let dir = temp_index_dir("ann-tombstones");
        fs::create_dir_all(&dir).expect("create temp dir");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[
                ("doc-a", &[1.0, 0.0, 0.0]),
                ("doc-b", &[0.0, 1.0, 0.0]),
                ("doc-c", &[0.0, 0.0, 1.0]),
            ],
        )
        .expect("write fast index");

        let mut fast_index = VectorIndex::open(&fast_path).expect("open fast index");
        let deleted = fast_index
            .soft_delete("doc-b")
            .expect("soft delete should succeed");
        assert!(deleted);

        let config = TwoTierConfig {
            hnsw_threshold: 1,
            hnsw_ef_search: 64,
            ..TwoTierConfig::default()
        };
        let index = TwoTierIndex::open(&dir, config).expect("open with ann");
        assert!(index.has_fast_ann());

        let hits = index.search_fast(&[0.0, 1.0, 0.0], 10).expect("search");
        assert!(
            !hits.iter().any(|hit| hit.doc_id == "doc-b"),
            "tombstoned document should not be returned by ANN search"
        );
    }

    /// Regression test for bd-2grj: HNSW `d_id` diverges from `VectorIndex` position
    /// after tombstone-aware rebuild. Verifies that live docs survive the tombstone
    /// filter even when their HNSW `d_id` differs from their `VectorIndex` position.
    #[cfg(feature = "ann")]
    #[test]
    fn ann_tombstone_filter_uses_doc_id_not_hnsw_position() {
        let dir = temp_index_dir("ann-tombstone-docid");
        fs::create_dir_all(&dir).expect("create temp dir");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        // A@0, B@1, C@2, D@3
        write_index_file(
            &fast_path,
            &[
                ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
                ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
                ("doc-c", &[0.0, 0.0, 1.0, 0.0]),
                ("doc-d", &[0.0, 0.0, 0.0, 1.0]),
            ],
        )
        .expect("write fast index");

        // Soft-delete doc-b (position 1) — creates gap between HNSW d_ids and positions.
        // After rebuild: HNSW d_ids = {0:doc-a, 1:doc-c, 2:doc-d}
        // VectorIndex positions = {0:doc-a, 1:doc-b(deleted), 2:doc-c, 3:doc-d}
        let mut fast_index = VectorIndex::open(&fast_path).expect("open for delete");
        assert!(fast_index.soft_delete("doc-b").expect("soft_delete"));

        let config = TwoTierConfig {
            hnsw_threshold: 1,
            hnsw_ef_search: 64,
            ..TwoTierConfig::default()
        };
        let index = TwoTierIndex::open(&dir, config).expect("open with ann");
        assert!(index.has_fast_ann());

        // Search for all docs — should return doc-a, doc-c, doc-d (NOT doc-b)
        let hits = index
            .search_fast(&[0.25, 0.25, 0.25, 0.25], 10)
            .expect("search");

        let hit_ids: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();

        // Critical: doc-c and doc-d must survive even though their HNSW d_ids (1, 2)
        // differ from their VectorIndex positions (2, 3). Before bd-2grj fix,
        // doc-c was incorrectly filtered because is_deleted(1) checked position 1
        // (doc-b, which IS deleted) instead of position 2 (doc-c, which is live).
        assert!(
            hit_ids.contains(&"doc-a"),
            "doc-a should be returned, got: {hit_ids:?}"
        );
        assert!(
            hit_ids.contains(&"doc-c"),
            "doc-c should be returned (bd-2grj regression), got: {hit_ids:?}"
        );
        assert!(
            hit_ids.contains(&"doc-d"),
            "doc-d should be returned (bd-2grj regression), got: {hit_ids:?}"
        );
        assert!(
            !hit_ids.contains(&"doc-b"),
            "tombstoned doc-b should NOT be returned, got: {hit_ids:?}"
        );
        assert_eq!(
            hits.len(),
            3,
            "expected exactly 3 live docs, got: {hit_ids:?}"
        );
    }

    // ─── bd-3nsq: NaN score in WAL merge ───

    #[test]
    fn search_fast_skips_nan_score_wal_entries() {
        use crate::wal::WalEntry;

        let dir = temp_index_dir("nan-wal-score");
        fs::create_dir_all(&dir).expect("create temp dir");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])])
            .expect("write fast index");

        let mut index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");

        // Inject a WAL entry with NaN in the embedding. dot_product with NaN
        // produces NaN, which should be filtered out by the is_finite() guard.
        index
            .fast_tier_mut_for_test()
            .expect("path-opened tier")
            .wal_entries
            .push(WalEntry {
                doc_id: "doc-nan".into(),
                doc_id_hash: crate::fnv1a_hash(b"doc-nan"),
                embedding: vec![f32::NAN, 0.0, 0.0, 0.0],
            });

        // Also inject a valid WAL entry to confirm it's still returned.
        index
            .fast_tier_mut_for_test()
            .expect("path-opened tier")
            .wal_entries
            .push(WalEntry {
                doc_id: "doc-wal-ok".into(),
                doc_id_hash: crate::fnv1a_hash(b"doc-wal-ok"),
                embedding: vec![0.0, 1.0, 0.0, 0.0],
            });

        let hits = index
            .search_fast(&[1.0, 0.0, 0.0, 0.0], 10)
            .expect("search");
        let ids: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();

        assert!(
            !ids.contains(&"doc-nan"),
            "NaN-scored WAL entry must be excluded, got: {ids:?}"
        );
        assert!(
            ids.contains(&"doc-a"),
            "base doc-a should be returned, got: {ids:?}"
        );
        assert!(
            ids.contains(&"doc-wal-ok"),
            "valid WAL entry should be returned, got: {ids:?}"
        );

        // Verify all returned scores are finite.
        for hit in &hits {
            assert!(
                hit.score.is_finite(),
                "hit {} has non-finite score {}",
                hit.doc_id,
                hit.score
            );
        }
    }

    // ─── bd-3szp tests begin ───

    #[test]
    fn filename_constants_are_correct() {
        assert_eq!(VECTOR_INDEX_FAST_FILENAME, "vector.fast.idx");
        assert_eq!(VECTOR_INDEX_QUALITY_FILENAME, "vector.quality.idx");
        assert_eq!(VECTOR_INDEX_FALLBACK_FILENAME, "vector.idx");
    }

    #[test]
    fn two_tier_index_implements_debug() {
        let dir = temp_index_dir("debug-index");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let debug_str = format!("{index:?}");
        assert!(debug_str.contains("TwoTierIndex"));
    }

    #[test]
    fn two_tier_index_builder_implements_debug() {
        let dir = temp_index_dir("debug-builder");
        let builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        let debug_str = format!("{builder:?}");
        assert!(debug_str.contains("TwoTierIndexBuilder"));
    }

    #[test]
    fn fast_index_preferred_over_fallback_when_both_exist() {
        let dir = temp_index_dir("prefer-fast");
        fs::create_dir_all(&dir).expect("create temp dir");

        // Write fallback with doc-fallback
        let fallback_path = dir.join(VECTOR_INDEX_FALLBACK_FILENAME);
        write_index_file(&fallback_path, &[("doc-fallback", &[0.0, 1.0])]).expect("write fallback");

        // Write explicit fast with doc-fast
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(&fast_path, &[("doc-fast", &[1.0, 0.0])]).expect("write fast");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        assert_eq!(index.doc_count(), 1);
        let ids: Vec<String> = index
            .iter_doc_ids()
            .collect::<SearchResult<_>>()
            .expect("ids");
        assert_eq!(ids, vec!["doc-fast".to_owned()]);
    }

    #[test]
    fn search_fast_returns_sorted_by_score_descending() {
        let dir = temp_index_dir("sort-order");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[
                ("doc-low", &[0.1, 0.0, 0.0]),
                ("doc-mid", &[0.5, 0.5, 0.0]),
                ("doc-high", &[1.0, 0.0, 0.0]),
            ],
        )
        .expect("write fast");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let hits = index.search_fast(&[1.0, 0.0, 0.0], 3).expect("search");
        assert_eq!(hits.len(), 3);
        // Scores should be in descending order
        assert!(hits[0].score >= hits[1].score);
        assert!(hits[1].score >= hits[2].score);
        assert_eq!(hits[0].doc_id, "doc-high");
    }

    #[test]
    fn search_fast_k_zero_returns_empty() {
        let dir = temp_index_dir("k-zero");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let hits = index.search_fast(&[1.0, 0.0], 0).expect("search k=0");
        assert!(hits.is_empty());
    }

    #[test]
    fn search_fast_k_larger_than_doc_count_returns_all() {
        let dir = temp_index_dir("k-large");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[("doc-a", &[1.0, 0.0]), ("doc-b", &[0.0, 1.0])],
        )
        .expect("write fast");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let hits = index.search_fast(&[1.0, 0.0], 100).expect("search k=100");
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn quality_scores_empty_indices_returns_empty() {
        let dir = temp_index_dir("empty-indices");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast");
        write_index_file(&quality_path, &[("doc-a", &[1.0, 0.0])]).expect("write quality");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let scores = index
            .quality_scores_for_hits(&[1.0, 0.0], &[])
            .expect("empty indices");
        assert!(scores.is_empty());
    }

    #[test]
    fn quality_scores_full_coverage() {
        let dir = temp_index_dir("full-quality");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);

        write_index_file(
            &fast_path,
            &[("doc-a", &[1.0, 0.0, 0.0]), ("doc-b", &[0.0, 1.0, 0.0])],
        )
        .expect("write fast");

        write_index_file(
            &quality_path,
            &[("doc-a", &[0.0, 0.0, 1.0]), ("doc-b", &[0.0, 1.0, 0.0])],
        )
        .expect("write quality");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        assert!(index.has_quality_for_index(0));
        assert!(index.has_quality_for_index(1));

        let hits = vec![
            VectorHit {
                index: 0,
                score: 0.0,
                doc_id: "doc-a".into(),
            },
            VectorHit {
                index: 1,
                score: 0.0,
                doc_id: "doc-b".into(),
            },
        ];
        let scores = index
            .quality_scores_for_hits(&[0.0, 1.0, 0.0], &hits)
            .expect("quality scores");
        assert_eq!(scores.len(), 2);
        // doc-a quality = [0,0,1] dot [0,1,0] = 0.0
        assert!(scores[0].unwrap().abs() < 1e-6);
        // doc-b quality = [0,1,0] dot [0,1,0] = 1.0
        assert!((scores[1].unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn builder_embedder_id_chaining() {
        let dir = temp_index_dir("embedder-chain");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");

        // Verify chaining returns &mut Self
        let _same_ref = builder
            .set_fast_embedder_id("custom-fast")
            .set_quality_embedder_id("custom-quality");

        builder
            .add_record("doc-a", &[1.0, 0.0], Some(&[0.0, 1.0]))
            .expect("add record");
        let index = builder.finish().expect("finish");
        assert_eq!(index.doc_count(), 1);
    }

    // ─── Embedding-space identity retention (bd-9xuj T2-C2) ─────────────────

    #[test]
    fn builder_finish_retains_declared_typed_identity() {
        let dir = temp_index_dir("typed-identity-retention");
        let fast_bundle = EmbeddingIdentityBundleV1::explicit_test_model("builder-fast-model", 4);
        let quality_bundle =
            EmbeddingIdentityBundleV1::explicit_test_model("builder-quality-model", 4);

        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder.set_fast_embedder_id("op-fast-id");
        builder.set_quality_embedder_id("op-quality-id");
        builder
            .set_fast_identity(&fast_bundle)
            .expect("declare fast identity");
        builder
            .set_quality_identity(&quality_bundle)
            .expect("declare quality identity");
        builder
            .add_record("doc-a", &[1.0, 0.0, 0.0, 0.0], Some(&[0.0, 1.0, 0.0, 0.0]))
            .expect("add record");
        let index = builder.finish().expect("finish");

        // The finished index carries the REAL typed identity, per tier.
        assert_eq!(
            index.fast_space_fingerprint_hex(),
            Some(fast_bundle.space.fingerprint().as_str())
        );
        assert_eq!(
            index.quality_space_fingerprint_hex(),
            Some(quality_bundle.space.fingerprint().as_str())
        );
        assert_ne!(
            index.fast_space_fingerprint_hex(),
            index.quality_space_fingerprint_hex(),
            "distinct producing models must stay distinct per tier"
        );
        assert_eq!(
            index
                .fast_declared_identity()
                .map(EmbeddingIdentityBundleV1::fingerprint),
            Some(fast_bundle.fingerprint()),
            "the full producing bundle survives finish for the admission law"
        );
        assert_eq!(
            index
                .quality_declared_identity()
                .map(EmbeddingIdentityBundleV1::fingerprint),
            Some(quality_bundle.fingerprint())
        );

        // The operational id string is untouched by the typed declaration;
        // the header revision now carries the space's immutable revision.
        assert_eq!(index.fast_embedder_id(), "op-fast-id");
        assert_eq!(index.quality_embedder_id(), Some("op-quality-id"));
        assert_eq!(
            index.fast_embedder_revision(),
            fast_bundle.space.immutable_revision.as_str()
        );
        assert_eq!(
            index.quality_embedder_revision(),
            Some(quality_bundle.space.immutable_revision.as_str())
        );
        drop(index);

        // Reopen from disk: the v1 artifacts persist id/revision strings but
        // NO space identity — absence stays typed (LegacyUnidentified), and
        // is never re-fabricated from the surviving strings.
        let reopened = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("reopen");
        assert_eq!(reopened.fast_embedder_id(), "op-fast-id");
        assert_eq!(
            reopened.fast_embedder_revision(),
            fast_bundle.space.immutable_revision.as_str(),
            "the declared revision must survive persistence in the v1 header"
        );
        assert_eq!(reopened.fast_space_fingerprint_hex(), None);
        assert_eq!(reopened.quality_space_fingerprint_hex(), None);
        assert!(reopened.fast_declared_identity().is_none());
        assert!(reopened.quality_declared_identity().is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn builder_finish_rejects_identity_not_describing_vectors() -> Result<(), String> {
        // A 16-dim space cannot describe 4-dim written vectors: the claim is
        // checked against what was actually written, never trusted alone.
        let dir = temp_index_dir("typed-identity-dim-mismatch");
        let wrong_bundle = EmbeddingIdentityBundleV1::explicit_test_model("wrong-dim-model", 16);

        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .set_fast_identity(&wrong_bundle)
            .expect("bundle itself is coherent; the mismatch is against the records");
        builder
            .add_record("doc-a", &[1.0, 0.0, 0.0, 0.0], None)
            .expect("add record");
        let error = builder
            .finish()
            .expect_err("a non-describing identity must be rejected at finish");
        let rendered = format!("{error:?}");
        let SearchError::InvalidConfig { field, value, .. } = error else {
            return Err(format!("expected InvalidConfig, got {rendered}"));
        };
        assert_eq!(field, "fast_identity.space.dimension");
        assert_eq!(value, "16");

        // Quality arm: same law, its own field name.
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .set_quality_identity(&wrong_bundle)
            .expect("bundle itself is coherent");
        builder
            .add_record("doc-a", &[1.0, 0.0, 0.0, 0.0], Some(&[0.0, 1.0, 0.0, 0.0]))
            .expect("add record");
        let error = builder
            .finish()
            .expect_err("a non-describing quality identity must be rejected at finish");
        let rendered = format!("{error:?}");
        let SearchError::InvalidConfig { field, value, .. } = error else {
            return Err(format!("expected InvalidConfig, got {rendered}"));
        };
        assert_eq!(field, "quality_identity.space.dimension");
        assert_eq!(value, "16");

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn builder_quality_identity_without_quality_tier_is_dropped() {
        // A declared quality identity with no quality records written must
        // NOT be attached: this build produced no quality tier, and the
        // declaration must never end up describing a stale or absent one.
        let dir = temp_index_dir("typed-identity-no-quality");
        let quality_bundle = EmbeddingIdentityBundleV1::explicit_test_model("unused-quality", 4);

        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .set_quality_identity(&quality_bundle)
            .expect("declare quality identity");
        builder
            .add_record("doc-a", &[1.0, 0.0, 0.0, 0.0], None)
            .expect("add fast-only record");
        let index = builder.finish().expect("finish fast-only build");
        assert!(!index.has_quality_index());
        assert_eq!(index.quality_space_fingerprint_hex(), None);
        assert!(index.quality_declared_identity().is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn builder_without_identity_stays_legacy_unidentified() {
        // The pre-C2 builder path, byte-compatible: no declaration, empty
        // header revision, and the typed-absent identity state throughout.
        let dir = temp_index_dir("typed-identity-legacy");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .add_record("doc-a", &[1.0, 0.0], Some(&[0.0, 1.0]))
            .expect("add record");
        let index = builder.finish().expect("finish");
        assert_eq!(index.fast_space_fingerprint_hex(), None);
        assert_eq!(index.quality_space_fingerprint_hex(), None);
        assert!(index.fast_declared_identity().is_none());
        assert!(index.quality_declared_identity().is_none());
        assert_eq!(index.fast_embedder_revision(), "");
        assert_eq!(index.quality_embedder_revision(), Some(""));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn semantic_vector_with_tier_reports_serving_tier() {
        // doc-a exists in both tiers; doc-b is fast-only. The provenance twin
        // must (1) return exactly what semantic_vector_for_doc_id returns,
        // (2) name the tier that served it, and (3) join to that tier's
        // space through space_fingerprint_hex_for_tier.
        let dir = temp_index_dir("semantic-vector-tier");
        let fast_bundle = EmbeddingIdentityBundleV1::explicit_test_model("svt-fast-model", 4);
        let quality_bundle = EmbeddingIdentityBundleV1::explicit_test_model("svt-quality-model", 4);

        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .set_fast_identity(&fast_bundle)
            .expect("declare fast identity");
        builder
            .set_quality_identity(&quality_bundle)
            .expect("declare quality identity");
        builder
            .add_record("doc-a", &[1.0, 0.0, 0.0, 0.0], Some(&[0.0, 1.0, 0.0, 0.0]))
            .expect("add doc-a");
        builder
            .add_record("doc-b", &[0.0, 0.0, 1.0, 0.0], None)
            .expect("add doc-b");
        let index = builder.finish().expect("finish");

        // Quality-covered doc: served from the quality tier.
        let (tier, vector) = index
            .semantic_vector_with_tier_for_doc_id("doc-a")
            .expect("lookup doc-a")
            .expect("doc-a exists");
        assert_eq!(tier, SemanticVectorTier::Quality);
        assert_eq!(
            Some(vector.clone()),
            index
                .semantic_vector_for_doc_id("doc-a")
                .expect("legacy lookup doc-a"),
            "the provenance twin must return exactly the legacy accessor's vector"
        );
        assert_eq!(
            index.space_fingerprint_hex_for_tier(tier),
            Some(quality_bundle.space.fingerprint().as_str()),
            "Quality provenance joins to the quality tier's space"
        );

        // Quality-missing doc: the silent fast fallback, now observable.
        let (tier, vector) = index
            .semantic_vector_with_tier_for_doc_id("doc-b")
            .expect("lookup doc-b")
            .expect("doc-b exists");
        assert_eq!(tier, SemanticVectorTier::Fast);
        assert_eq!(
            Some(vector),
            index
                .semantic_vector_for_doc_id("doc-b")
                .expect("legacy lookup doc-b")
        );
        assert_eq!(
            index.space_fingerprint_hex_for_tier(tier),
            Some(fast_bundle.space.fingerprint().as_str()),
            "Fast provenance joins to the fast tier's space"
        );

        // Missing doc: both accessors agree on absence.
        assert!(
            index
                .semantic_vector_with_tier_for_doc_id("doc-missing")
                .expect("lookup missing")
                .is_none()
        );
        assert!(
            index
                .semantic_vector_for_doc_id("doc-missing")
                .expect("legacy lookup missing")
                .is_none()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn builder_fast_only_no_quality_index_created() {
        let dir = temp_index_dir("fast-only-builder");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .add_fast_record("doc-a", &[1.0, 0.0, 0.0])
            .expect("fast a");
        builder
            .add_fast_record("doc-b", &[0.0, 1.0, 0.0])
            .expect("fast b");

        let index = builder.finish().expect("finish");
        assert_eq!(index.doc_count(), 2);
        assert!(!index.has_quality_index());
        assert!(!dir.join(VECTOR_INDEX_QUALITY_FILENAME).exists());
    }

    #[test]
    fn builder_add_record_with_quality_creates_both_tiers() {
        let dir = temp_index_dir("both-tiers-builder");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .add_record("doc-a", &[1.0, 0.0], Some(&[0.5, 0.5, 0.0]))
            .expect("add doc-a");
        builder
            .add_record("doc-b", &[0.0, 1.0], Some(&[0.0, 0.5, 0.5]))
            .expect("add doc-b");

        let index = builder.finish().expect("finish");
        assert_eq!(index.doc_count(), 2);
        assert!(index.has_quality_index());
        assert!(index.has_quality_for_index(0));
        assert!(index.has_quality_for_index(1));
    }

    #[test]
    fn builder_preserves_all_doc_ids() {
        let dir = temp_index_dir("all-docids");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        let names = ["zebra", "apple", "mango", "banana"];
        for name in &names {
            builder
                .add_fast_record(*name, &[1.0, 0.0])
                .expect("add record");
        }
        let index = builder.finish().expect("finish");
        assert_eq!(index.doc_count(), 4);
        let ids: Vec<String> = index
            .iter_doc_ids()
            .collect::<SearchResult<_>>()
            .expect("ids");
        let mut actual: Vec<&str> = ids.iter().map(String::as_str).collect();
        actual.sort_unstable();
        let mut expected = names.to_vec();
        expected.sort_unstable();
        assert_eq!(actual, expected);
    }

    #[test]
    fn fast_index_for_doc_id_empty_string_returns_none() {
        let dir = temp_index_dir("empty-docid-lookup");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        assert_eq!(index.fast_index_for_doc_id("").unwrap(), None);
    }

    #[test]
    fn has_quality_for_index_boundary_last_valid() {
        let dir = temp_index_dir("boundary-last");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);

        write_index_file(
            &fast_path,
            &[
                ("doc-a", &[1.0, 0.0]),
                ("doc-b", &[0.0, 1.0]),
                ("doc-c", &[0.5, 0.5]),
            ],
        )
        .expect("write fast");

        // Quality only has last doc
        write_index_file(&quality_path, &[("doc-c", &[0.5, 0.5])]).expect("write quality");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        assert!(!index.has_quality_for_index(0));
        assert!(!index.has_quality_for_index(1));
        assert!(index.has_quality_for_index(2)); // last valid index
        assert!(!index.has_quality_for_index(3)); // out of bounds
    }

    #[test]
    fn quality_scores_no_quality_index_ignores_query_dimension() {
        // When there's no quality index, any query dimension is accepted (returns all 0.0)
        let dir = temp_index_dir("no-quality-any-dim");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        assert!(!index.has_quality_index());

        // Use a completely different dimension query — should still return 0s
        let hits = vec![VectorHit {
            index: 0,
            score: 0.0,
            doc_id: "doc-a".into(),
        }];
        let scores = index
            .quality_scores_for_hits(&[1.0, 2.0, 3.0, 4.0, 5.0], &hits)
            .expect("any dim accepted");
        assert_eq!(scores, vec![None]);
    }

    #[test]
    fn search_fast_returns_correct_doc_ids() {
        let dir = temp_index_dir("correct-docids");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[
                ("alpha", &[1.0, 0.0, 0.0]),
                ("beta", &[0.0, 1.0, 0.0]),
                ("gamma", &[0.0, 0.0, 1.0]),
            ],
        )
        .expect("write fast");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");

        // Query aligned with beta
        let hits = index
            .search_fast(&[0.0, 1.0, 0.0], 1)
            .expect("search for beta");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "beta");

        // Query aligned with gamma
        let hits = index
            .search_fast(&[0.0, 0.0, 1.0], 1)
            .expect("search for gamma");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "gamma");
    }

    #[test]
    fn builder_add_record_dimension_mismatch_in_quality() {
        let dir = temp_index_dir("record-quality-dim");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .add_record("doc-a", &[1.0, 0.0], Some(&[1.0, 0.0, 0.0]))
            .expect("first record ok");

        // Second record has different quality dimension
        let err = builder
            .add_record("doc-b", &[0.0, 1.0], Some(&[1.0, 0.0]))
            .expect_err("quality dim mismatch");
        assert!(matches!(
            err,
            SearchError::DimensionMismatch {
                expected: 3,
                found: 2
            }
        ));
    }

    #[test]
    fn open_nonexistent_directory_returns_error() {
        let dir = temp_index_dir("nonexistent-subdir");
        // Don't create the directory
        let result = TwoTierIndex::open(&dir, TwoTierConfig::default());
        assert!(result.is_err());
    }

    // ─── bd-3szp tests end ───

    // ─── Attested vs declared identity discriminator (bd-9xuj T2-C4-write) ──

    use frankensearch_core::generation::{ArtifactGenerationIdentityV1, QuantizationFormat};

    /// Identity binding for `model_id` with canonical FSVI v2 storage, plus
    /// the in-memory-storage sibling bundle a producing embedder would hold.
    fn fsvi_v2_binding(
        model_id: &str,
        dimension: u32,
        sequence: u64,
    ) -> (FsviV2IdentityBinding, EmbeddingIdentityBundleV1) {
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(model_id, dimension);
        "fsvi-v2".clone_into(&mut identity.storage.format);
        identity.storage.quantization = QuantizationFormat::F16;
        "little-endian".clone_into(&mut identity.storage.endianness);
        let generation =
            ArtifactGenerationIdentityV1::new(sequence, [0x4d; 16]).expect("valid test generation");
        let binding = FsviV2IdentityBinding::new(
            generation,
            identity.freeze().expect("freeze artifact identity"),
        )
        .expect("valid FSVI v2 identity binding");
        (binding, identity)
    }

    fn write_v2_tier(path: &Path, binding: &FsviV2IdentityBinding, rows: &[(&str, &[f32])]) {
        let mut writer =
            VectorIndex::create_v2(path, binding.clone()).expect("create_v2 writer for fixture");
        for (doc_id, vector) in rows {
            writer.write_record(doc_id, vector).expect("write v2 row");
        }
        writer.finish().expect("finish v2 fixture");
    }

    #[test]
    fn admitted_v2_open_reports_header_attested_identity() {
        // Isolated dir: exact admission snapshots the containing directory
        // and fails closed when sibling files churn it, so the shared test
        // temp dir is not a legal admission site (see 58726e26).
        let dir = temp_index_dir("admitted-v2-attested");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
        let (fast_binding, fast_identity) = fsvi_v2_binding("attested-fast-model", 4, 3);
        let (quality_binding, quality_identity) = fsvi_v2_binding("attested-quality-model", 4, 3);
        let rows: [(&str, &[f32]); 2] = [
            ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
            ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
        ];
        write_v2_tier(&fast_path, &fast_binding, &rows);
        write_v2_tier(&quality_path, &quality_binding, &rows);

        // Plain discovery/open can never reach a v2 tier: VectorIndex::open
        // is strictly v1, so the attested state is unreachable except through
        // exact admission.
        let plain = TwoTierIndex::open(&dir, TwoTierConfig::default());
        assert!(
            matches!(plain, Err(SearchError::IndexVersionMismatch { .. })),
            "plain open must reject v2 bytes, got {plain:?}"
        );

        let paths = TwoTierIndexPaths::new(&fast_path).with_quality_index(&quality_path);
        let index = TwoTierIndex::open_admitted_v2_with_paths(
            &paths,
            TwoTierConfig::default(),
            &fast_binding,
            Some(&quality_binding),
        )
        .expect("admit both v2 tiers");

        assert!(
            index.fast_identity_is_attested(),
            "identity parsed from the artifact's validated v2 header is ATTESTED"
        );
        assert!(index.quality_identity_is_attested());
        assert_eq!(
            index.fast_space_fingerprint_hex(),
            Some(fast_identity.space.fingerprint().as_str()),
            "the join key must be the header's space fingerprint, bit-for-bit"
        );
        assert_eq!(
            index.quality_space_fingerprint_hex(),
            Some(quality_identity.space.fingerprint().as_str())
        );
        assert!(
            index.fast_declared_identity().is_none(),
            "attested identity is not a builder declaration"
        );
        assert!(index.quality_declared_identity().is_none());
        assert_eq!(index.doc_count(), 2);
        let hits = index
            .search_fast(&[0.0, 1.0, 0.0, 0.0], 1)
            .expect("search admitted v2 fast tier");
        assert_eq!(hits[0].doc_id, "doc-b");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn builder_declared_identity_is_never_attested() {
        // The C4-write discriminator half of red proof (c): a builder-time
        // declaration retains the identity (C2) but must NOT read as
        // attested — the persisted artifacts are v1 and their headers attest
        // nothing.
        let dir = temp_index_dir("declared-not-attested");
        let fast_bundle = EmbeddingIdentityBundleV1::explicit_test_model("declared-fast", 4);
        let quality_bundle = EmbeddingIdentityBundleV1::explicit_test_model("declared-quality", 4);
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .set_fast_identity(&fast_bundle)
            .expect("declare fast identity");
        builder
            .set_quality_identity(&quality_bundle)
            .expect("declare quality identity");
        builder
            .add_record("doc-a", &[1.0, 0.0, 0.0, 0.0], Some(&[0.0, 1.0, 0.0, 0.0]))
            .expect("add record");
        let index = builder.finish().expect("finish");

        assert!(
            index.fast_space_fingerprint_hex().is_some()
                && index.fast_declared_identity().is_some(),
            "the declaration is retained (C2)…"
        );
        assert!(
            !index.fast_identity_is_attested(),
            "…but a declaration is a process-local claim, never a header attestation"
        );
        assert!(!index.quality_identity_is_attested());

        drop(index);
        let reopened = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("reopen");
        assert!(!reopened.fast_identity_is_attested());
        assert!(!reopened.quality_identity_is_attested());
        assert_eq!(reopened.fast_space_fingerprint_hex(), None);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn admitted_v2_open_requires_binding_for_quality_path() {
        let dir = temp_index_dir("admitted-v2-missing-binding");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
        let (fast_binding, _) = fsvi_v2_binding("binding-gap-fast", 4, 1);
        let (quality_binding, _) = fsvi_v2_binding("binding-gap-quality", 4, 1);
        let rows: [(&str, &[f32]); 1] = [("doc-a", &[1.0, 0.0, 0.0, 0.0])];
        write_v2_tier(&fast_path, &fast_binding, &rows);
        write_v2_tier(&quality_path, &quality_binding, &rows);

        let paths = TwoTierIndexPaths::new(&fast_path).with_quality_index(&quality_path);
        let error = TwoTierIndex::open_admitted_v2_with_paths(
            &paths,
            TwoTierConfig::default(),
            &fast_binding,
            None,
        )
        .expect_err("a quality path without its binding must be refused");
        assert!(
            matches!(
                error,
                SearchError::InvalidConfig { ref field, .. }
                    if field == "two_tier.quality_v2_admission"
            ),
            "got {error:?}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn admitted_v2_open_rejects_foreign_binding() {
        let dir = temp_index_dir("admitted-v2-foreign-binding");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let (fast_binding, _) = fsvi_v2_binding("real-artifact-model", 4, 1);
        let rows: [(&str, &[f32]); 1] = [("doc-a", &[1.0, 0.0, 0.0, 0.0])];
        write_v2_tier(&fast_path, &fast_binding, &rows);

        let (foreign_binding, _) = fsvi_v2_binding("foreign-expectation-model", 4, 1);
        let paths = TwoTierIndexPaths::new(&fast_path);
        let error = TwoTierIndex::open_admitted_v2_with_paths(
            &paths,
            TwoTierConfig::default(),
            &foreign_binding,
            None,
        )
        .expect_err("admission against a foreign identity binding must be refused");
        assert!(
            matches!(
                error,
                SearchError::InvalidConfig { ref field, .. }
                    if field == "two_tier.fast_v2_admission"
            ),
            "got {error:?}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    // ─── C4-write r2: retained sealed owners + read-only observation ─────

    /// NO-GO item 3 repair: exact v2 admission must retain the complete
    /// `ValidatedFsviBytes` owner per tier — byte capability, witness, and
    /// publication state — instead of peeling `validated.index` and dropping
    /// the rest. Red on 868c0801: `fast_admitted_owner` does not exist there
    /// (the owner was destructured away), so this test cannot compile.
    #[test]
    fn admitted_v2_open_retains_sealed_owners_in_full() {
        let dir = temp_index_dir("admitted-v2-retained-owners");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
        let (fast_binding, _) = fsvi_v2_binding("retained-fast-model", 4, 5);
        let (quality_binding, _) = fsvi_v2_binding("retained-quality-model", 4, 5);
        let rows: [(&str, &[f32]); 2] = [
            ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
            ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
        ];
        write_v2_tier(&fast_path, &fast_binding, &rows);
        write_v2_tier(&quality_path, &quality_binding, &rows);

        let paths = TwoTierIndexPaths::new(&fast_path).with_quality_index(&quality_path);
        let index = TwoTierIndex::open_admitted_v2_with_paths(
            &paths,
            TwoTierConfig::default(),
            &fast_binding,
            Some(&quality_binding),
        )
        .expect("admit both v2 tiers");

        let fast_owner = index
            .fast_admitted_owner()
            .expect("fast admission owner must be retained");
        let quality_owner = index
            .quality_admitted_owner()
            .expect("quality admission owner must be retained");
        assert_eq!(fast_owner.witness().record_count, 2);
        assert_eq!(quality_owner.witness().record_count, 2);
        assert!(
            fast_owner.published_wal_absent(),
            "canonical pathname admission proves WAL absence; the publication \
             state must survive into the retained owner"
        );
        assert!(quality_owner.published_wal_absent());
        assert_eq!(
            fast_owner.owned_byte_len(),
            usize::try_from(fs::metadata(&fast_path).expect("stat fast").len())
                .expect("length fits usize"),
            "the retained owner holds the complete admitted byte image"
        );
        // The retained owner and the served tier are the same admission, not
        // a re-open: identity metadata must agree bit-for-bit.
        assert!(index.fast_identity_is_attested());
        assert_eq!(
            index.fast_space_fingerprint_hex(),
            Some(crate::fingerprint_hex(&fast_owner.identity_v2().space_fingerprint).as_str())
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// Required test (iv): after admission, replacing the underlying file
    /// does not affect the retained owner's reads — the `Arc`'d bytes are
    /// the authority (owner contract, lib.rs).
    #[test]
    fn admitted_owner_reads_survive_path_replacement() {
        let dir = temp_index_dir("admitted-v2-path-replacement");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let (fast_binding, _) = fsvi_v2_binding("replacement-proof-model", 4, 9);
        let rows: [(&str, &[f32]); 2] = [
            ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
            ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
        ];
        write_v2_tier(&fast_path, &fast_binding, &rows);

        let paths = TwoTierIndexPaths::new(&fast_path);
        let index = TwoTierIndex::open_admitted_v2_with_paths(
            &paths,
            TwoTierConfig::default(),
            &fast_binding,
            None,
        )
        .expect("admit fast v2 tier");
        let witness_before = index
            .fast_admitted_owner()
            .expect("retained owner")
            .witness()
            .clone();
        let hits_before = index
            .search_fast(&[0.0, 1.0, 0.0, 0.0], 1)
            .expect("search before replacement");
        assert_eq!(hits_before[0].doc_id, "doc-b");

        // Rename the source away, then plant garbage at the original path.
        let renamed = dir.join("vector.fast.idx.renamed-away");
        fs::rename(&fast_path, &renamed).expect("rename admitted source");
        fs::write(&fast_path, b"garbage-not-an-index").expect("plant garbage");

        let owner = index.fast_admitted_owner().expect("retained owner");
        assert_eq!(
            owner.witness(),
            &witness_before,
            "witness is part of the sealed owner, not re-derived from the path"
        );
        let hits_after = index
            .search_fast(&[0.0, 1.0, 0.0, 0.0], 1)
            .expect("search after replacement");
        assert_eq!(
            hits_after[0].doc_id, "doc-b",
            "reads must serve the admitted Arc'd bytes, never the pathname"
        );
        assert_eq!(owner.row(0).expect("row 0").doc_id(), "doc-a");
        assert_eq!(index.doc_count(), 2);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn plain_v1_open_has_no_admitted_owners() {
        let dir = temp_index_dir("plain-v1-no-owners");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])])
            .expect("write v1 fixture");
        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open v1");
        assert!(index.fast_admitted_owner().is_none());
        assert!(index.quality_admitted_owner().is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    /// Byte-for-byte fixture snapshot of one artifact and (optionally) its
    /// WAL sidecar, for observation-invariance assertions.
    fn tier_snapshot(path: &Path) -> (Vec<u8>, Option<Vec<u8>>) {
        let main = fs::read(path).expect("read main artifact");
        let wal_path = crate::wal::wal_path_for(path);
        let wal = wal_path
            .exists()
            .then(|| fs::read(&wal_path).expect("read wal"));
        (main, wal)
    }

    #[test]
    fn observe_tier_classifies_v1_seed_live_and_v2_without_mutation() {
        let dir = temp_index_dir("observe-basic");
        fs::create_dir_all(&dir).expect("create temp dir");

        // Empty v1 seed: no content retained.
        let seed_path = dir.join("seed.idx");
        VectorIndex::create(&seed_path, "seed-model", 4)
            .expect("create seed")
            .finish()
            .expect("finish seed");
        let seed_snapshot = tier_snapshot(&seed_path);
        let seed_observation = observe_tier(&seed_path).expect("observe seed");
        assert!(
            matches!(seed_observation, FsviTierObservation::V1(_)),
            "v1 seed must observe as V1, got {seed_observation:?}"
        );
        let FsviTierObservation::V1(seed) = seed_observation else {
            return;
        };
        assert_eq!(seed.record_count, 0);
        assert_eq!(seed.active_wal_records, 0);
        assert!(!seed.wal_sidecar_present);
        assert!(!seed.retains_content());
        assert_eq!(tier_snapshot(&seed_path), seed_snapshot);

        // Live v1 with a fresh WAL append: content retained via both signals.
        let live_path = dir.join("live.idx");
        write_index_file(&live_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])]).expect("write live v1");
        {
            let mut live = VectorIndex::open(&live_path).expect("open live v1");
            live.append("doc-wal", &[0.0, 1.0, 0.0, 0.0])
                .expect("append WAL resident");
        }
        let live_snapshot = tier_snapshot(&live_path);
        assert!(live_snapshot.1.is_some(), "fixture must have a WAL sidecar");
        let live_observation = observe_tier(&live_path).expect("observe live");
        assert!(
            matches!(live_observation, FsviTierObservation::V1(_)),
            "live v1 must observe as V1, got {live_observation:?}"
        );
        let FsviTierObservation::V1(live) = live_observation else {
            return;
        };
        assert_eq!(live.record_count, 1);
        assert_eq!(live.active_wal_records, 1);
        assert!(live.wal_sidecar_present);
        assert!(live.retains_content());
        assert_eq!(
            tier_snapshot(&live_path),
            live_snapshot,
            "observation must not rewrite the artifact or its WAL"
        );

        // v2: header-only recognition.
        let v2_path = dir.join("observed.v2.idx");
        let (binding, _) = fsvi_v2_binding("observe-v2-model", 4, 2);
        write_v2_tier(&v2_path, &binding, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])]);
        let v2_snapshot = tier_snapshot(&v2_path);
        let v2_observation = observe_tier(&v2_path).expect("observe v2");
        assert!(
            matches!(v2_observation, FsviTierObservation::V2IdentityComplete(_)),
            "v2 tier must observe as V2IdentityComplete, got {v2_observation:?}"
        );
        let FsviTierObservation::V2IdentityComplete(metadata) = v2_observation else {
            return;
        };
        assert_eq!(metadata.record_count, 1);
        assert!(metadata.identity_v2.is_some());
        assert_eq!(tier_snapshot(&v2_path), v2_snapshot);

        let _ = fs::remove_dir_all(&dir);
    }

    /// NO-GO item 2 repair, stale-WAL half: `VectorIndex::open` DELETES a
    /// stale WAL sidecar; read-only observation must classify it as inactive
    /// while leaving the file byte-identical.
    #[test]
    fn observe_tier_never_deletes_a_stale_wal_sidecar() {
        let dir = temp_index_dir("observe-stale-wal");
        fs::create_dir_all(&dir).expect("create temp dir");

        // Donor: advance the compaction generation once, then append, so the
        // donor WAL header carries generation 2 (= next(1)).
        let donor_path = dir.join("donor.idx");
        write_index_file(&donor_path, &[("donor-a", &[1.0, 0.0, 0.0, 0.0])]).expect("write donor");
        {
            let mut donor = VectorIndex::open(&donor_path).expect("open donor");
            donor
                .append("donor-wal-1", &[0.0, 1.0, 0.0, 0.0])
                .expect("append pre-compaction");
            donor.compact().expect("compact to bump generation");
            donor
                .append("donor-wal-2", &[0.0, 0.0, 1.0, 0.0])
                .expect("append post-compaction");
        }
        let donor_wal = crate::wal::wal_path_for(&donor_path);
        assert!(donor_wal.exists(), "donor WAL must exist");

        // Target: generation-0 main slab with a live row; the donor WAL's
        // generation cannot match next(0), so it reads as STALE here.
        let target_path = dir.join("target.idx");
        write_index_file(&target_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])]).expect("write target");
        let target_wal = crate::wal::wal_path_for(&target_path);
        fs::copy(&donor_wal, &target_wal).expect("transplant stale WAL");

        // Precondition: the transplanted WAL really is stale for the target.
        let (entries, wal_gen, valid_len) =
            crate::wal::read_wal(&target_wal, 4, Quantization::F16).expect("read fixture WAL");
        assert!(
            !entries.is_empty() && valid_len > 0,
            "fixture WAL has entries"
        );
        assert_ne!(
            wal_gen,
            crate::next_generation(0),
            "fixture WAL generation must mismatch the target main slab"
        );

        let snapshot = tier_snapshot(&target_path);
        let target_observation = observe_tier(&target_path).expect("observe target");
        assert!(
            matches!(target_observation, FsviTierObservation::V1(_)),
            "target must observe as V1, got {target_observation:?}"
        );
        let FsviTierObservation::V1(observation) = target_observation else {
            return;
        };
        assert_eq!(
            observation.active_wal_records, 0,
            "stale WAL entries belong to a dead generation"
        );
        assert!(observation.wal_sidecar_present);
        assert!(
            observation.retains_content(),
            "the live main row retains content regardless of WAL staleness"
        );
        assert!(
            target_wal.exists(),
            "read-only observation must NEVER delete a stale WAL (VectorIndex::open does)"
        );
        assert_eq!(
            tier_snapshot(&target_path),
            snapshot,
            "artifact and WAL must be byte-identical after observation"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// NO-GO item 2 repair, corrupt-trailer half: `VectorIndex::open`
    /// TRUNCATES a corrupt WAL trailer; read-only observation must count the
    /// valid prefix and leave the trailer bytes in place.
    #[test]
    fn observe_tier_never_truncates_a_corrupt_wal_trailer() {
        let dir = temp_index_dir("observe-corrupt-trailer");
        fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("trailer.idx");
        write_index_file(&path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])]).expect("write v1");
        {
            let mut index = VectorIndex::open(&path).expect("open v1");
            index
                .append("doc-wal", &[0.0, 1.0, 0.0, 0.0])
                .expect("append valid WAL entry");
        }
        let wal_path = crate::wal::wal_path_for(&path);
        let clean_len = fs::metadata(&wal_path).expect("stat wal").len();
        {
            use std::io::Write as _;
            let mut wal_file = fs::OpenOptions::new()
                .append(true)
                .open(&wal_path)
                .expect("open wal for corruption");
            wal_file
                .write_all(&[0xAB; 32])
                .expect("append corrupt trailer");
        }
        let corrupted_len = fs::metadata(&wal_path).expect("stat wal").len();
        assert_eq!(corrupted_len, clean_len + 32);

        let snapshot = tier_snapshot(&path);
        let trailer_observation = observe_tier(&path).expect("observe");
        assert!(
            matches!(trailer_observation, FsviTierObservation::V1(_)),
            "must observe as V1, got {trailer_observation:?}"
        );
        let FsviTierObservation::V1(observation) = trailer_observation else {
            return;
        };
        assert_eq!(
            observation.active_wal_records, 1,
            "the valid prefix still counts"
        );
        assert_eq!(
            fs::metadata(&wal_path).expect("re-stat wal").len(),
            corrupted_len,
            "read-only observation must NEVER truncate a corrupt trailer \
             (VectorIndex::open does)"
        );
        assert_eq!(tier_snapshot(&path), snapshot);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn observe_tier_reports_upgrade_required_for_future_schema() {
        let dir = temp_index_dir("observe-upgrade");
        fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("future.idx");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&crate::FSVI_MAGIC);
        bytes.extend_from_slice(&3_u16.to_le_bytes());
        bytes.extend_from_slice(&[0_u8; 16]);
        fs::write(&path, &bytes).expect("write future-schema fixture");
        let observation = observe_tier(&path).expect("observe future schema");
        assert!(
            matches!(
                observation,
                FsviTierObservation::UpgradeRequired(FsviUpgradeRequired {
                    found_version: 3,
                    supported_version: crate::FSVI_V2_VERSION,
                })
            ),
            "got {observation:?}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// Write-side WAL hygiene: a builder rewrite of a tier removes the
    /// now-dead adjacent WAL sidecar so it can never resurrect foreign rows
    /// into the new generation (read-only classification deliberately leaves
    /// stale sidecars in place; the WRITE path owns their cleanup).
    #[test]
    fn builder_finish_removes_adjacent_wal_sidecars() {
        let dir = temp_index_dir("builder-wal-hygiene");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(&fast_path, &[("old-doc", &[1.0, 0.0, 0.0, 0.0])])
            .expect("write prior generation");
        {
            let mut prior = VectorIndex::open(&fast_path).expect("open prior");
            prior
                .append("wal-resident", &[0.0, 1.0, 0.0, 0.0])
                .expect("append WAL resident");
        }
        let wal_path = crate::wal::wal_path_for(&fast_path);
        assert!(wal_path.exists(), "fixture WAL must exist before rebuild");

        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .add_fast_record("new-doc", &[0.0, 0.0, 1.0, 0.0])
            .expect("add rebuild row");
        let rebuilt = builder.finish().expect("finish rebuild");

        assert!(
            !wal_path.exists(),
            "the rewritten tier's dead WAL sidecar must be removed by the write path"
        );
        assert_eq!(rebuilt.doc_count(), 1);
        let ids: Vec<String> = rebuilt.iter_doc_ids().filter_map(Result::ok).collect();
        assert_eq!(ids, vec!["new-doc".to_owned()]);
        assert!(
            !ids.iter().any(|id| id == "wal-resident" || id == "old-doc"),
            "no row from the replaced generation may leak into the rebuild: {ids:?}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// Directory entries under `dir` whose names carry the sibling staging
    /// marker (`.tmp.<pid>.<nanos>`), for asserting staged replacements
    /// never leak.
    fn staging_leftovers(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .expect("read dir")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp."))
            .collect();
        names.sort();
        names
    }

    /// r3 fault tests (ii) + (iv) — the crash-after-main-rename state under
    /// the OLD r2 ordering, and the generation-2 collision reopen
    /// (bd-9xuj C4-write r3; audits #8366/#8367).
    ///
    /// Constructs on disk exactly what a crash between r2's
    /// `writer.finish()` rename and its best-effort WAL removal left:
    /// a fresh generation-1 main REPLACED over the canonical name with the
    /// old generation's generation-2 WAL still adjacent. Pins, exactly:
    /// - the observer reports the sidecar PRESENT and its rows ACTIVE
    ///   (`next_generation(1) == 2` — byte-indistinguishable from
    ///   legitimate incremental appends), so `retains_content()` holds and
    ///   observation-driven admission seams fail closed on this state
    ///   instead of consuming it (the refresh-side refusal is pinned in
    ///   `frankensearch-fusion/src/refresh.rs`);
    /// - the plain v1 reopen REPLAYS the foreign row (the frozen
    ///   `VectorIndex::open` v1 contract; `lib.rs` is not editable in this
    ///   train) — this resurrection is precisely why the r3 protocol makes
    ///   the state unconstructable, and the two-tier open path now surfaces
    ///   the replay with a warning rather than silence;
    /// - `finish()` over ANY WAL-bearing directory exits with
    ///   {new main, no WAL}: `publish_tier` installs an authority-carrying
    ///   successor generation (bd-zhjv8) whose rename atomically invalidates
    ///   the destination's sidecar, so the hazard state {new main + ADOPTABLE
    ///   old WAL} is unreachable through `finish()` at every interruption
    ///   point.
    #[test]
    fn crash_after_rename_state_is_pinned_and_unconstructable_through_finish() {
        let dir = temp_index_dir("r3-crash-after-rename");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);

        // Old generation: fresh legacy main (compaction generation 1) plus
        // one incremental append, which creates a generation-2 WAL sidecar.
        write_index_file(&fast_path, &[("old-doc", &[1.0, 0.0, 0.0, 0.0])])
            .expect("write old generation");
        {
            let mut old = VectorIndex::open(&fast_path).expect("open old generation");
            old.append("foreign-doc", &[0.0, 1.0, 0.0, 0.0])
                .expect("append WAL resident");
        }
        let wal_path = crate::wal::wal_path_for(&fast_path);
        let (entries, wal_gen, valid_len) =
            crate::wal::read_wal(&wal_path, 4, Quantization::F16).expect("read fixture WAL");
        assert_eq!(entries.len(), 1, "fixture WAL carries the foreign row");
        assert!(valid_len > 0);
        assert_eq!(
            wal_gen,
            crate::next_generation(1),
            "fixture WAL generation must be 2 = next(1): the exact collision"
        );

        // Simulate r2's crash point: a replacement generation-1 main is
        // renamed over the canonical name; the WAL removal never ran.
        let scratch = dir.join("replacement.scratch");
        write_index_file(&scratch, &[("new-doc", &[0.0, 0.0, 1.0, 0.0])])
            .expect("write replacement main");
        fs::rename(&scratch, &fast_path).expect("simulate crash after rename");
        assert!(wal_path.exists(), "crash state: gen-2 WAL still adjacent");

        // Pin (iv), observer half: the sidecar is REPORTED, its rows count
        // as ACTIVE against the fresh generation-1 main, and the state
        // therefore reads as content-retaining — the conservative signal
        // admission seams refuse on. Nothing is silently dropped and
        // nothing on disk is touched.
        let observation = observe_tier(&fast_path).expect("observe crash state");
        assert!(
            matches!(observation, FsviTierObservation::V1(_)),
            "crash state must observe as V1, got {observation:?}"
        );
        let FsviTierObservation::V1(observed) = observation else {
            return;
        };
        assert_eq!(observed.record_count, 1);
        assert_eq!(
            observed.active_wal_records, 1,
            "a generation-2 WAL is byte-indistinguishable from legitimate \
             incremental appends against a generation-1 main"
        );
        assert!(observed.wal_sidecar_present);
        assert!(observed.retains_content());
        assert!(wal_path.exists(), "observation never deletes the sidecar");

        // Pin (iv), reopen half: the plain v1 open REPLAYS the foreign row
        // (frozen `VectorIndex::open` contract). This is the resurrection
        // the r3 ordering exists to make unmanufacturable; the two-tier
        // reopen surfaces it via `warn_if_wal_rows_replayed` rather than
        // silence.
        {
            let reopened = VectorIndex::open(&fast_path).expect("v1 reopen of crash state");
            assert_eq!(reopened.record_count(), 1, "main slab rows");
            assert_eq!(
                reopened.wal_record_count(),
                1,
                "the foreign WAL row IS replayed by the frozen v1 open path"
            );
            let wal_ids: Vec<&str> = reopened.wal_records().map(|(id, _)| id).collect();
            assert_eq!(
                wal_ids,
                vec!["foreign-doc"],
                "the replayed resident row is exactly the foreign append"
            );
        }
        let two_tier = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("two-tier reopen");
        assert_eq!(
            two_tier.doc_count(),
            1,
            "the doc-id surface counts the main slab only"
        );
        let hits = two_tier
            .search_fast(&[0.0, 0.8, 0.6, 0.0], 4)
            .expect("search crash state");
        let hit_ids: Vec<&str> = hits.iter().map(|hit| hit.doc_id.as_str()).collect();
        assert_eq!(
            hit_ids,
            vec!["foreign-doc", "new-doc"],
            "pinned: the legacy crash state resurrects the foreign row \
             through the SEARCH surface on plain reopen — unreachable \
             through finish() after r3"
        );
        drop(two_tier);

        // Pin (ii): the NEW protocol cannot leave this state. A full
        // rebuild over the WAL-bearing directory publishes a successor
        // generation whose rename invalidates the sidecar, retires it after
        // the authority switch, and exits with {new main, no WAL}.
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .add_fast_record("rebuilt-doc", &[0.5, 0.5, 0.0, 0.0])
            .expect("add rebuild row");
        let rebuilt = builder.finish().expect("rebuild over crash state");
        assert!(
            !wal_path.exists(),
            "finish() must never exit with a published main and a live WAL"
        );
        let ids: Vec<String> = rebuilt.iter_doc_ids().filter_map(Result::ok).collect();
        assert_eq!(ids, vec!["rebuilt-doc".to_owned()]);
        assert_eq!(
            staging_leftovers(&dir),
            Vec::<String>::new(),
            "no staged replacement survives a successful finish()"
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
