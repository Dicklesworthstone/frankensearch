//! Persisted native ANN on the existing cache's retained v2 owners.
//!
//! A cache's graph paths and construction policy are immutable. Reload opens
//! all requested graphs before the existing pointer-fenced swap; it never
//! rebuilds them or silently installs an exact-only substitute.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use asupersync::Cx;
use frankensearch_core::{SearchResult, TwoTierConfig};
use frankensearch_index::native_hnsw::{HnswParams, native_hnsw_generation_receipt_path};
use frankensearch_index::{FsviV2IdentityBinding, TwoTierIndex, TwoTierIndexPaths};

use super::super::{IndexCache, IndexOpenSpec, StalenessDetector, absolute_from};
use super::{checkpoint, open_exact, rejected, validate_retained_replacement};

/// One persisted native graph and the construction policy it must retain.
///
/// The associated receipt is the existing native-HNSW receipt beside this path.
/// Loading checks it against the complete retained FSVI owner, then validates
/// the graph and these parameters/seed. No file is created, repaired or rebuilt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeCacheTier {
    /// Native graph file, not a legacy `hnsw_rs` sidecar.
    pub graph_path: PathBuf,
    /// Exact parameters required of the stored graph.
    pub params: HnswParams,
    /// Exact seed required of the stored graph.
    pub seed: u64,
}

/// Immutable per-tier native retrieval policy for an identity-bound cache.
///
/// A missing entry explicitly keeps that tier exact. At least one graph must
/// be supplied; a quality graph requires a selected quality vector tier. Paths
/// are frozen with the vector paths at construction and never rebound by reload.
/// Use a new cache to change the graph paths, parameters, seed or tier policy.
///
/// This is read-only graph/source admission in trusted stable directories, not
/// a composite publication authority or an externally pinned graph digest. The
/// existing native loader authenticates each graph against its local receipt
/// and exact vector owner. Publishers own coherent replacement of the vector,
/// graph and receipt files. Intermediate or corrupt pairs fail without changing
/// the installed snapshot. Same-generation vector content/coverage cannot change.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NativeCachePolicy {
    /// Fast-tier graph, or exact fast retrieval.
    pub fast: Option<NativeCacheTier>,
    /// Quality-tier graph, or exact quality retrieval.
    pub quality: Option<NativeCacheTier>,
}

fn checked(cx: Option<&Cx>) -> SearchResult<()> {
    cx.map_or(Ok(()), checkpoint)
}

impl NativeCachePolicy {
    fn into_absolute(mut self, base: &Path) -> SearchResult<Self> {
        if self.fast.is_none() && self.quality.is_none() {
            return Err(rejected(
                "native_policy",
                "at least one native graph is required",
            ));
        }
        for tier in [&mut self.fast, &mut self.quality].into_iter().flatten() {
            tier.params.validate()?;
            tier.graph_path = absolute_from(&tier.graph_path, base);
        }
        Ok(self)
    }

    /// All vector, graph and receipt roles must be existing distinct files.
    /// These checks precede vector decoding or native graph admission. Paths
    /// remain a trusted-directory protocol, not hostile-filesystem capabilities.
    fn validate_paths(&self, paths: &TwoTierIndexPaths) -> SearchResult<()> {
        if self.quality.is_some() && paths.quality_index().is_none() {
            return Err(rejected(
                "native_policy",
                "a quality graph requires a quality vector tier",
            ));
        }
        let mut roles = vec![paths.fast_index().to_path_buf()];
        roles.extend(paths.quality_index().map(Path::to_path_buf));
        for tier in [&self.fast, &self.quality].into_iter().flatten() {
            roles.push(tier.graph_path.clone());
            roles.push(native_hnsw_generation_receipt_path(&tier.graph_path)?);
        }
        let mut canonical = Vec::with_capacity(roles.len());
        for path in roles {
            let metadata = fs::symlink_metadata(&path)?;
            if !metadata.is_file() {
                return Err(rejected(
                    "native_paths",
                    "native cache roles must be regular non-symlink files",
                ));
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                if metadata.nlink() != 1 {
                    return Err(rejected(
                        "native_paths",
                        "hard-linked native cache artifacts are not supported",
                    ));
                }
            }
            let path = fs::canonicalize(path)?;
            if canonical.contains(&path) {
                return Err(rejected(
                    "native_paths",
                    "vector, graph and receipt roles must not alias",
                ));
            }
            canonical.push(path);
        }
        Ok(())
    }

    pub(super) fn load(
        &self,
        cx: Option<&Cx>,
        paths: &TwoTierIndexPaths,
        index: &mut TwoTierIndex,
    ) -> SearchResult<()> {
        checked(cx)?;
        self.validate_paths(paths)?;
        checked(cx)?;
        if let Some(tier) = &self.fast {
            let result = index.load_native_fast_hnsw(&tier.graph_path, tier.params, tier.seed);
            checked(cx)?;
            result?;
        }
        if let Some(tier) = &self.quality {
            let result = index.load_native_quality_hnsw(&tier.graph_path, tier.params, tier.seed);
            checked(cx)?;
            result?;
        }
        self.validate_presence(index)
    }

    fn validate_presence(&self, index: &TwoTierIndex) -> SearchResult<()> {
        if index.has_native_fast_hnsw() != self.fast.is_some()
            || index.has_native_quality_hnsw() != self.quality.is_some()
        {
            return Err(rejected(
                "native_policy",
                "loaded retrieval differs from the retained native policy",
            ));
        }
        Ok(())
    }
}

impl IndexCache {
    /// Open persisted native ANN on exact v2 owners in this existing cache.
    ///
    /// Both configured vector tiers and every requested graph are required.
    /// The optional `ann` feature and writable legacy sidecar paths are not
    /// used. Graph/source/receipt or policy mismatch fails; no build, repair,
    /// model loading, sentinel write or exact fallback occurs.
    ///
    /// `current()` returns the same `Arc<TwoTierIndex>` consumed by the ordinary
    /// progressive searcher, now with native fast and/or independent quality
    /// retrieval. Ordinary `reload()` preserves this native policy and the
    /// installed v2 bindings. To advance a generation, use the existing
    /// `reload_admitted_v2_if_current` with trusted successor bindings. It loads
    /// both graphs before the pointer-fenced swap. Generic prebuilt replacement
    /// APIs refuse native caches because they do not carry this load evidence.
    ///
    /// All paths freeze against one CWD snapshot. The caller owns durable
    /// multi-file publication and keeps directories trusted during admission.
    /// Queries retain their owned bytes/graphs after replacement or path changes.
    /// Synchronous decoding and filesystem work run on the caller's lane;
    /// cancellation is checked around, not inside, blocking operations.
    ///
    /// # Errors
    /// Returns cancellation, path, topology, identity, policy or load failures.
    #[allow(clippy::too_many_arguments)]
    pub fn open_admitted_v2_with_native(
        cx: &Cx,
        paths: TwoTierIndexPaths,
        state_dir: &Path,
        config: TwoTierConfig,
        fast_binding: &FsviV2IdentityBinding,
        quality_binding: Option<&FsviV2IdentityBinding>,
        native: NativeCachePolicy,
        detector: Box<dyn StalenessDetector>,
    ) -> SearchResult<Self> {
        checkpoint(cx)?;
        super::validate_bindings(&paths, fast_binding, quality_binding)?;
        let base = std::env::current_dir()?;
        let paths = paths.into_absolute_from(&base)?;
        let state_dir = absolute_from(state_dir, &base);
        let native = native.into_absolute(&base)?;
        native.validate_paths(&paths)?;
        checkpoint(cx)?;
        let result = open_exact(&paths, config.clone(), fast_binding, quality_binding);
        checkpoint(cx)?;
        let mut index = result?;
        native.load(Some(cx), &paths, &mut index)?;
        checkpoint(cx)?;
        Ok(Self {
            inner: RwLock::new(Arc::new(index)),
            detector,
            open_spec: IndexOpenSpec::Explicit(paths),
            state_dir,
            config,
            native_reload: Some(native),
        })
    }

    /// Native graph paths and policy retained for every reload, if opted in.
    #[must_use]
    pub const fn native_cache_policy(&self) -> Option<&NativeCachePolicy> {
        self.native_reload.as_ref()
    }

    // Only cache-owned loaders may enter here: public prebuilt replacement
    // has no witness that the candidate used these graph paths/parameters.
    pub(in crate::cache) fn install_loaded_native(
        &self,
        cx: Option<&Cx>,
        expected: &Arc<TwoTierIndex>,
        candidate: TwoTierIndex,
    ) -> SearchResult<bool> {
        checked(cx)?;
        let policy = self.native_reload.as_ref().ok_or_else(|| {
            rejected(
                "native_policy",
                "native installation requires an opted-in cache",
            )
        })?;
        let paths = self.index_paths().ok_or_else(|| {
            rejected(
                "native_paths",
                "native installation requires explicit vector paths",
            )
        })?;
        let candidate = Arc::new(candidate);
        let retired = {
            let mut current = self.write_index();
            checked(cx)?;
            if !Arc::ptr_eq(&current, expected) {
                return Ok(false);
            }
            if current.fast_index_path() != paths.fast_index()
                || candidate.fast_index_path() != paths.fast_index()
                || current.quality_index_path() != paths.quality_index()
                || candidate.quality_index_path() != paths.quality_index()
            {
                return Err(rejected(
                    "native_paths",
                    "replacement differs from retained vector paths",
                ));
            }
            policy.validate_presence(&current)?;
            policy.validate_presence(&candidate)?;
            validate_retained_replacement(&current, &candidate)?;
            checked(cx)?;
            std::mem::replace(&mut *current, candidate)
        };
        // No cancellation point follows visible installation. Drop large old
        // graphs after unlocking; a slow destructor cannot block snapshot reads.
        tracing::debug!(target: "frankensearch.cache", "installed native v2 cache snapshot");
        drop(retired);
        Ok(true)
    }
}

#[cfg(test)]
mod tests;
