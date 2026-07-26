//! Two-tier index wrapper for fast and quality vector indices.
//!
//! `TwoTierIndex` provides a single object that coordinates:
//! - fast-tier retrieval from `vector.fast.idx` (or `vector.idx` fallback)
//! - optional quality-tier rescoring from `vector.quality.idx`
//! - doc-id alignment between both tiers

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use frankensearch_core::{SearchError, SearchResult, TwoTierConfig, VectorHit};
use tracing::{debug, warn};

#[cfg(all(feature = "ann", test))]
use crate::hnsw::HNSW_META_FORMAT_CURRENT;
#[cfg(feature = "ann")]
use crate::hnsw::HnswLoadDisposition;
#[cfg(feature = "ann")]
use crate::{HNSW_DEFAULT_MAX_LAYER, HnswConfig, HnswIndex};
use crate::{SearchParams, VectorIndex, dot_product_f32_f32};

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

    match same_file::is_same_file(left, right) {
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
                || same_file::is_same_file(left_path, right_path).map_err(SearchError::Io)?
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

/// Dual-index container used by progressive search orchestration.
#[derive(Debug)]
pub struct TwoTierIndex {
    fast_index: VectorIndex,
    quality_index: Option<VectorIndex>,
    #[cfg(feature = "ann")]
    fast_ann: Option<HnswIndex>,
    #[cfg(feature = "ann")]
    quality_ann: Option<HnswIndex>,
    quality_alignment: QualityAlignment,
    config: TwoTierConfig,
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
    /// candidate exists, and propagates index parse/corruption errors from
    /// `VectorIndex::open`.
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
        Self::open_with_paths(&paths, config)
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
    #[allow(clippy::too_many_lines)]
    pub fn open_with_paths(paths: &TwoTierIndexPaths, config: TwoTierConfig) -> SearchResult<Self> {
        let paths = paths.clone().into_absolute()?;
        validate_index_paths(&paths)?;
        let fast_index = VectorIndex::open(paths.fast_index())?;
        let mut quality_alignment = QualityAlignment::None;

        let quality_index = if let Some(quality_path) = paths.quality_index() {
            let quality = VectorIndex::open(quality_path)?;

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
                &fast_index,
                fast_ann_path,
                config.hnsw_threshold,
                &config,
                "fast",
            )
        });

        #[cfg(feature = "ann")]
        let quality_ann_plan = quality_index.as_ref().and_then(|quality_index| {
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
                    &paths,
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
                        &paths,
                        fast_needs_persistence,
                        quality_needs_persistence,
                    );
                }
                if let (Some(plan), Some(path)) = (quality_ann_plan.as_ref(), paths.quality_ann()) {
                    persist_ann_plan(
                        plan,
                        path,
                        "quality",
                        &paths,
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

        Ok(Self {
            fast_index,
            quality_index,
            #[cfg(feature = "ann")]
            fast_ann,
            #[cfg(feature = "ann")]
            quality_ann,
            quality_alignment,
            config,
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

    /// Search the fast tier only.
    ///
    /// # Errors
    ///
    /// Propagates errors from `HnswIndex::knn_search` (when ANN is selected)
    /// or `VectorIndex::search_top_k` (brute-force fallback).
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
    /// Propagates errors from `HnswIndex::knn_search` (when ANN is selected)
    /// or `VectorIndex::search_top_k_with_params` / `search_top_k`.
    pub fn search_fast_with_params(
        &self,
        query_vec: &[f32],
        k: usize,
        params: Option<SearchParams>,
    ) -> SearchResult<Vec<VectorHit>> {
        #[cfg(feature = "ann")]
        if let Some(ann) = &self.fast_ann {
            // Fetch a few extra candidates to buffer against soft-deleted records.
            // This isn't perfect but helps maintain recall when tombstones exist.
            let fetch_k = k.saturating_add(10);
            let hits = ann.knn_search(query_vec, fetch_k, self.config.hnsw_ef_search)?;

            // Filter soft-deleted records from ANN results and resolve real VectorIndex positions.
            // NOTE: hit.index from ANN is the compact HNSW d_id, NOT the VectorIndex position.
            // We map it back to the canonical position so downstream consumers (like quality scoring)
            // get valid indices.
            let mut resolved_hits = Vec::with_capacity(hits.len());
            for mut hit in hits {
                // doc_id missing or decode error → treat as deleted
                if let Ok(Some(pos)) = self.fast_index.find_index_by_doc_id(&hit.doc_id)
                    && !self.fast_index.is_deleted(pos)
                {
                    hit.index = u32::try_from(pos).unwrap_or(u32::MAX);
                    resolved_hits.push(hit);
                }
            }
            let mut hits = resolved_hits;

            // Merge WAL entries (not yet in ANN).
            if !self.fast_index.wal_entries.is_empty() {
                let base_index = self.fast_index.record_count();
                for (i, entry) in self.fast_index.wal_entries.iter().enumerate() {
                    let score = dot_product_f32_f32(&entry.embedding, query_vec)?;
                    // Guard: corrupt WAL embeddings (e.g. from crash recovery) can
                    // produce NaN/Inf scores that poison the top-k sort. Skip them.
                    if !score.is_finite() {
                        continue;
                    }
                    // Virtual index logic must match VectorIndex::resolve_wal_hit
                    let index = u32::try_from(base_index + i).unwrap_or(u32::MAX);
                    hits.push(VectorHit {
                        index,
                        score,
                        doc_id: entry.doc_id.as_str().into(),
                    });
                }
                // Re-sort and truncate after merging. The comparator is a strict
                // total order (score `total_cmp` + unique `index` tiebreak), so
                // `select_nth`/`sort_unstable` are bit-identical to the stable sort.
                // For a large merged pool (ANN `ef` + resident WAL), partition to the
                // top-`k` in O(n) then sort only those — the MRL rescored top-k lever
                // (`mrl_topk_select_ab`, ~2×). Gated by `SELECT_NTH_MIN` so small pools
                // keep the stable sort (n-dependent constant factors; see a0fd2090).
                let by_score_index = |a: &VectorHit, b: &VectorHit| {
                    b.score
                        .total_cmp(&a.score)
                        .then_with(|| a.index.cmp(&b.index))
                };
                const SELECT_NTH_MIN: usize = 256;
                if k < hits.len() && hits.len() >= SELECT_NTH_MIN {
                    hits.select_nth_unstable_by(k, by_score_index);
                    hits.truncate(k);
                    hits.sort_unstable_by(by_score_index);
                } else {
                    hits.sort_by(by_score_index);
                    if hits.len() > k {
                        hits.truncate(k);
                    }
                }
            } else if hits.len() > k {
                // If no WAL but we fetched extra for filtering, truncate back to k
                hits.truncate(k);
            }
            return Ok(hits);
        }
        let mrl_config = crate::mrl::MrlConfig {
            search_dims: self.config.mrl_search_dims,
            rescore_dims: 0,
            rescore_top_k: self.config.mrl_rescore_top_k,
        };

        if mrl_config.search_dims > 0 && mrl_config.search_dims < self.fast_index.dimension() {
            return self.fast_index.mrl_search(query_vec, k, &mrl_config, None);
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
                self.fast_index
                    .search_top_k_int8_two_pass(query_vec, k, FAST_TIER_MULT)
            },
            |params| {
                self.fast_index
                    .search_top_k_with_params(query_vec, k, None, params)
            },
        )
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
        let Some(quality_index) = &self.quality_index else {
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
                    self.fast_index.find_index_by_doc_id(&hit.doc_id)?
                } else if (hit.index as usize) < self.fast_index.record_count() {
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
        self.quality_index.is_some()
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

    /// Number of documents in the fast tier (canonical document count).
    #[must_use]
    pub const fn doc_count(&self) -> usize {
        self.fast_index.record_count()
    }

    /// Embedder identity recorded in the fast-tier index header.
    #[must_use]
    pub fn fast_embedder_id(&self) -> &str {
        self.fast_index.embedder_id()
    }

    /// Embedder revision recorded in the fast-tier index header.
    #[must_use]
    pub fn fast_embedder_revision(&self) -> &str {
        self.fast_index.embedder_revision()
    }

    /// Embedder identity recorded in the quality-tier index header, when loaded.
    #[must_use]
    pub fn quality_embedder_id(&self) -> Option<&str> {
        self.quality_index.as_ref().map(VectorIndex::embedder_id)
    }

    /// Embedder revision recorded in the quality-tier index header, when loaded.
    #[must_use]
    pub fn quality_embedder_revision(&self) -> Option<&str> {
        self.quality_index
            .as_ref()
            .map(VectorIndex::embedder_revision)
    }

    /// Filesystem path of the loaded fast-tier index artifact.
    #[must_use]
    pub fn fast_index_path(&self) -> &Path {
        &self.fast_index.path
    }

    /// Filesystem path of the loaded quality-tier index artifact, when loaded.
    #[must_use]
    pub fn quality_index_path(&self) -> Option<&Path> {
        self.quality_index
            .as_ref()
            .map(|index| index.path.as_path())
    }

    /// Iterate over all document IDs in fast-tier order.
    pub fn iter_doc_ids(&self) -> impl Iterator<Item = SearchResult<String>> + '_ {
        (0..self.doc_count()).map(|i| self.fast_index.doc_id_at(i).map(ToOwned::to_owned))
    }

    /// Document ID at a given fast-tier index position.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the index is out of bounds or reading fails.
    pub fn doc_id_at(&self, index: usize) -> SearchResult<&str> {
        self.fast_index.doc_id_at(index)
    }

    /// Fast-tier index position for a given document id.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if index reading fails.
    pub fn fast_index_for_doc_id(&self, doc_id: &str) -> SearchResult<Option<usize>> {
        self.fast_index.find_index_by_doc_id(doc_id)
    }

    /// Fast-tier vector for the given document id.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if index access fails.
    pub fn fast_vector_for_doc_id(&self, doc_id: &str) -> SearchResult<Option<Vec<f32>>> {
        let hash = crate::fnv1a_hash(doc_id.as_bytes());
        for entry in self.fast_index.wal_entries.iter().rev() {
            if entry.doc_id_hash == hash && entry.doc_id == doc_id {
                return Ok(Some(entry.embedding.clone()));
            }
        }

        if let Some(index) = self.fast_index.find_index_by_doc_id(doc_id)? {
            return self.fast_index.vector_at_f32(index).map(Some);
        }

        Ok(None)
    }

    /// Quality-tier vector for the given document id when available.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if index access fails.
    pub fn quality_vector_for_doc_id(&self, doc_id: &str) -> SearchResult<Option<Vec<f32>>> {
        let Some(quality_index) = self.quality_index.as_ref() else {
            return Ok(None);
        };

        let hash = crate::fnv1a_hash(doc_id.as_bytes());
        for entry in quality_index.wal_entries.iter().rev() {
            if entry.doc_id_hash == hash && entry.doc_id == doc_id {
                return Ok(Some(entry.embedding.clone()));
            }
        }

        if let Some(fast_index) = self.fast_index.find_index_by_doc_id(doc_id)? {
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
    /// # Errors
    ///
    /// Returns `SearchError` if index access fails.
    pub fn semantic_vector_for_doc_id(&self, doc_id: &str) -> SearchResult<Option<Vec<f32>>> {
        if let Some(quality) = self.quality_vector_for_doc_id(doc_id)? {
            return Ok(Some(quality));
        }
        self.fast_vector_for_doc_id(doc_id)
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

/// Builder for writing fast and optional quality FSVI indices.
#[derive(Debug)]
pub struct TwoTierIndexBuilder {
    dir: PathBuf,
    config: TwoTierConfig,
    fast_embedder_id: String,
    quality_embedder_id: String,
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
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if no fast-tier records were added,
    /// and propagates writer/open errors from `VectorIndex`.
    pub fn finish(self) -> SearchResult<TwoTierIndex> {
        let fast_dimension = self
            .fast_dimension
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "fast_records".to_owned(),
                value: "0".to_owned(),
                reason: "at least one fast-tier record is required".to_owned(),
            })?;

        let fast_path = self.dir.join(VECTOR_INDEX_FAST_FILENAME);
        let mut fast_writer =
            VectorIndex::create(&fast_path, &self.fast_embedder_id, fast_dimension)?;
        for (doc_id, embedding) in &self.fast_records {
            fast_writer.write_record(doc_id, embedding)?;
        }
        fast_writer.finish()?;

        if let Some(quality_dimension) = self.quality_dimension {
            let quality_path = self.dir.join(VECTOR_INDEX_QUALITY_FILENAME);
            let mut quality_writer =
                VectorIndex::create(&quality_path, &self.quality_embedder_id, quality_dimension)?;
            for (doc_id, embedding) in &self.quality_records {
                quality_writer.write_record(doc_id, embedding)?;
            }
            quality_writer.finish()?;
        }

        TwoTierIndex::open(&self.dir, self.config)
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
            let volume_aliases =
                same_file::is_same_file(&probe, &alternate_probe).unwrap_or_else(|error| {
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
            same_file::is_same_file(&lock_path, &missing_quality_path)
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
        assert!(same_file::is_same_file(&fast_path, &ann_path).expect("compare aliases"));
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
            same_file::is_same_file(&quality_path, &quality_ann_path).expect("compare aliases")
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
        index.fast_index.wal_entries.push(WalEntry {
            doc_id: "doc-nan".into(),
            doc_id_hash: crate::fnv1a_hash(b"doc-nan"),
            embedding: vec![f32::NAN, 0.0, 0.0, 0.0],
        });

        // Also inject a valid WAL entry to confirm it's still returned.
        index.fast_index.wal_entries.push(WalEntry {
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
}
