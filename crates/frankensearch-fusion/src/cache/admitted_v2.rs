//! Exact v2 admission for the existing index cache. No generation discovery,
//! writer, persistent cache, or second selection authority is introduced.

use std::path::Path;
use std::sync::{Arc, RwLock};

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult, TwoTierConfig};
use frankensearch_index::{FsviV2IdentityBinding, TwoTierIndex, TwoTierIndexPaths};

use super::{IndexCache, IndexOpenSpec, StalenessDetector, absolute_from};

fn rejected(field: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("index_cache.{field}"),
        value: "admitted-v2".to_owned(),
        reason: reason.to_owned(),
    }
}

fn checkpoint(cx: &Cx) -> SearchResult<()> {
    if cx.checkpoint().is_err() || cx.is_cancel_requested() {
        return Err(SearchError::Cancelled {
            phase: "index_cache.admitted_v2".to_owned(),
            reason: "index-cache admission or installation cancelled".to_owned(),
        });
    }
    Ok(())
}

fn validate_bindings(
    paths: &TwoTierIndexPaths,
    fast: &FsviV2IdentityBinding,
    quality: Option<&FsviV2IdentityBinding>,
) -> SearchResult<()> {
    // Exact caches perform read-only admission. Constructing this role set
    // also detects configured legacy ANN paths without a feature-dependent
    // accessor: dependency feature unification can enable those paths even
    // when this crate defines no ANN feature itself.
    let mut exact = TwoTierIndexPaths::new(paths.fast_index());
    if let Some(path) = paths.quality_index() {
        exact = exact.with_quality_index(path);
    }
    if &exact != paths {
        return Err(rejected(
            "open",
            "v2 cache admission is exact and read-only; writable ANN sidecar paths are not supported",
        ));
    }
    if paths.quality_index().is_some() != quality.is_some() {
        return Err(rejected(
            "open",
            "a quality path and its exact identity binding must be supplied together",
        ));
    }
    if let Some(quality) = quality {
        let fast_identity = &fast.frozen_identity().identity;
        let quality_identity = &quality.frozen_identity().identity;
        if fast.generation() != quality.generation()
            || fast_identity.input.doc_id_semantics != quality_identity.input.doc_id_semantics
            || fast_identity.space.kind != quality_identity.space.kind
        {
            return Err(rejected(
                "open",
                "all tiers must name one complete generation, document-ID contract and semantic/control kind",
            ));
        }
    }
    Ok(())
}

fn open_exact(
    paths: &TwoTierIndexPaths,
    config: TwoTierConfig,
    fast: &FsviV2IdentityBinding,
    quality: Option<&FsviV2IdentityBinding>,
) -> SearchResult<TwoTierIndex> {
    validate_bindings(paths, fast, quality)?;
    let index = TwoTierIndex::open_admitted_v2_with_paths(paths, config, fast, quality)?;
    validate_admitted(&index)?;
    // The common index assembler can degrade a mixed publication to fast-only.
    // A caller-selected required tier must never disappear through that path.
    if index.fast_admitted_binding() != Some(fast)
        || index.quality_admitted_binding() != quality
    {
        return Err(rejected(
            "open",
            "opened resources do not retain every explicitly selected tier and binding",
        ));
    }
    Ok(index)
}

fn validate_admitted(index: &TwoTierIndex) -> SearchResult<()> {
    let fast = index
        .fast_admitted_binding()
        .ok_or_else(|| rejected("replace", "the fast tier must retain exact v2 admission"))?;
    let paths = match index.quality_index_path() {
        Some(quality) => TwoTierIndexPaths::new(index.fast_index_path()).with_quality_index(quality),
        None => TwoTierIndexPaths::new(index.fast_index_path()),
    };
    validate_bindings(&paths, fast, index.quality_admitted_binding())?;
    for owner in [index.fast_admitted_owner(), index.quality_admitted_owner()]
        .into_iter()
        .flatten()
    {
        if !owner.published_wal_absent() {
            return Err(rejected(
                "replace",
                "cache admission requires a published WAL-absent retained owner",
            ));
        }
    }
    if index.fast_admitted_owner().is_none()
        || index.quality_index_path().is_some() != index.quality_admitted_owner().is_some()
    {
        return Err(rejected(
            "replace",
            "every selected tier must retain its validated bytes",
        ));
    }
    if index.has_native_fast_hnsw() || index.has_native_quality_hnsw() {
        return Err(rejected(
            "replace",
            "this exact v2 cache cannot preserve native ANN policy across reload; use the native retained-owner lifecycle",
        ));
    }
    Ok(())
}

fn validate_successor_bindings(
    current: &TwoTierIndex,
    fast: &FsviV2IdentityBinding,
    quality: Option<&FsviV2IdentityBinding>,
) -> SearchResult<()> {
    validate_admitted(current)?;
    if current.quality_admitted_binding().is_some() != quality.is_some() {
        return Err(rejected(
            "replace",
            "replacement cannot change required tier topology",
        ));
    }
    for (old, next) in [
        (current.fast_admitted_binding(), Some(fast)),
        (current.quality_admitted_binding(), quality),
    ] {
        if let (Some(old), Some(next)) = (old, next) {
            if old.frozen_identity().canonical_bytes != next.frozen_identity().canonical_bytes {
                return Err(rejected(
                    "replace",
                    "replacement changes the complete space, producer, input or storage identity",
                ));
            }
            let before = old.generation();
            let after = next.generation();
            if after.sequence < before.sequence
                || (after.sequence == before.sequence && after != before)
            {
                return Err(rejected(
                    "replace",
                    "replacement regresses the generation or reuses its sequence with another identity",
                ));
            }
        }
    }
    Ok(())
}

/// Also called by legacy public replace/reload entry points, so constructing a
/// candidate elsewhere cannot bypass the stricter contract after v2 admission.
pub(super) fn validate_replacement(
    current: &TwoTierIndex,
    candidate: &TwoTierIndex,
) -> SearchResult<()> {
    if current.fast_admitted_binding().is_none() {
        // Existing explicit legacy identity-enrichment behavior remains intact.
        // When the candidate is v2, its whole topology still has to be admitted.
        return if candidate.fast_admitted_binding().is_some() {
            validate_admitted(candidate)
        } else {
            Ok(())
        };
    }
    validate_admitted(candidate)?;
    let fast = candidate
        .fast_admitted_binding()
        .ok_or_else(|| rejected("replace", "replacement cannot discard v2 admission"))?;
    validate_successor_bindings(current, fast, candidate.quality_admitted_binding())?;
    for (old, new) in [
        (current.fast_admitted_owner(), candidate.fast_admitted_owner()),
        (current.quality_admitted_owner(), candidate.quality_admitted_owner()),
    ] {
        if let (Some(old), Some(new)) = (old, new)
            && old.witness().generation == new.witness().generation
            && old.witness() != new.witness()
        {
            return Err(rejected(
                "replace",
                "the same immutable generation cannot be rebound to different content or coverage",
            ));
        }
    }
    Ok(())
}

impl IndexCache {
    /// Open exact, identity-bound v2 tiers in the existing shared index cache.
    ///
    /// Bindings must come from the caller's trusted publication, not from
    /// inspecting an unknown file and relabeling it. Paths are frozen against
    /// one current-directory snapshot; no conventional filenames are discovered.
    /// Both configured tiers are required and retain their admitted byte owners.
    /// Different tier dimensions and explicitly partial document coverage remain
    /// supported. The supplied staleness detector is advisory, not authority.
    ///
    /// This route is exact and read-only: writable ANN paths are refused, not
    /// silently ignored. No model, source file, sidecar or sentinel is written.
    /// Filesystem work is synchronous on the caller's blocking lane. Cancellation
    /// brackets admission; it does not preempt a filesystem syscall or lock.
    ///
    /// # Errors
    /// Returns cancellation, path/topology, identity, v2 admission or I/O errors.
    pub fn open_admitted_v2_with_paths(
        cx: &Cx,
        paths: TwoTierIndexPaths,
        state_dir: &Path,
        config: TwoTierConfig,
        fast_binding: &FsviV2IdentityBinding,
        quality_binding: Option<&FsviV2IdentityBinding>,
        detector: Box<dyn StalenessDetector>,
    ) -> SearchResult<Self> {
        checkpoint(cx)?;
        validate_bindings(&paths, fast_binding, quality_binding)?;
        let base = std::env::current_dir()?;
        let paths = paths.into_absolute_from(&base)?;
        let state_dir = absolute_from(state_dir, &base);
        let result = open_exact(&paths, config.clone(), fast_binding, quality_binding);
        checkpoint(cx)?;
        let index = result?;
        Ok(Self {
            inner: RwLock::new(Arc::new(index)),
            detector,
            open_spec: IndexOpenSpec::Explicit(paths),
            state_dir,
            config,
        })
    }

    pub(super) fn open_for_reload(&self, current: &TwoTierIndex) -> SearchResult<TwoTierIndex> {
        match current.fast_admitted_binding() {
            Some(fast) => {
                let paths = self.index_paths().ok_or_else(|| {
                    rejected(
                        "reload",
                        "v2 reload requires a retained explicit path specification",
                    )
                })?;
                open_exact(
                    paths,
                    self.config.clone(),
                    fast,
                    current.quality_admitted_binding(),
                )
            }
            None => self.open_spec.open(self.config.clone()),
        }
    }

    /// Load a caller-selected v2 successor and install only against `expected`.
    ///
    /// Obtain `expected` with `current()` BEFORE preparing or selecting the
    /// successor. A stale/foreign token returns `Ok(false)` before file access,
    /// and is checked again at installation. Required paths/topology and complete
    /// per-tier identities cannot change. Generation sequences may advance; a
    /// same-generation reload must retain byte/content/coverage witnesses exactly.
    /// Subsequent ordinary `reload()` uses the INSTALLED bindings, never the
    /// constructor's obsolete generation or fresh metadata discovered from disk.
    ///
    /// No persistent authority is changed and no candidate/predecessor is deleted.
    /// The caller owns durable publication and stable artifact path replacement.
    /// Losing an in-process race never authorizes rolling back that publication.
    ///
    /// # Errors
    /// Returns cancellation, non-v2 cache, path/topology, contract, admission or
    /// I/O errors. Every refusal preserves the installed snapshot and old readers.
    pub fn reload_admitted_v2_if_current(
        &self,
        cx: &Cx,
        expected: &Arc<TwoTierIndex>,
        fast_binding: &FsviV2IdentityBinding,
        quality_binding: Option<&FsviV2IdentityBinding>,
    ) -> SearchResult<bool> {
        self.reload_admitted_v2_with(cx, expected, fast_binding, quality_binding, |paths| {
            open_exact(paths, self.config.clone(), fast_binding, quality_binding)
        })
    }

    fn reload_admitted_v2_with(
        &self,
        cx: &Cx,
        expected: &Arc<TwoTierIndex>,
        fast_binding: &FsviV2IdentityBinding,
        quality_binding: Option<&FsviV2IdentityBinding>,
        load: impl FnOnce(&TwoTierIndexPaths) -> SearchResult<TwoTierIndex>,
    ) -> SearchResult<bool> {
        checkpoint(cx)?;
        if !Arc::ptr_eq(&self.current(), expected) {
            return Ok(false);
        }
        let paths = self.index_paths().ok_or_else(|| {
            rejected("reload", "selected v2 reload requires explicit cache paths")
        })?;
        validate_bindings(paths, fast_binding, quality_binding)?;
        validate_successor_bindings(expected, fast_binding, quality_binding)?;
        let result = load(paths);
        checkpoint(cx)?;
        self.replace_admitted_v2_if_current(cx, expected, result?)
    }

    /// Install an already-admitted exact v2 candidate without reopening it.
    ///
    /// The cache keeps this candidate's validated owners. The original explicit
    /// artifact paths, complete tier contracts and generation/content checks are
    /// enforced under the same lock as the predecessor comparison and swap.
    /// Cancellation is checked after acquiring the lock and immediately before
    /// swapping. No cancellation checkpoint or fallible reopen follows success.
    /// Destruction of retired owners occurs after releasing the cache lock.
    ///
    /// # Errors
    /// Returns cancellation, path or v2 contract errors. A stale predecessor is
    /// `Ok(false)` and cannot alter selection. A successful install returns true.
    pub fn replace_admitted_v2_if_current(
        &self,
        cx: &Cx,
        expected: &Arc<TwoTierIndex>,
        candidate: TwoTierIndex,
    ) -> SearchResult<bool> {
        checkpoint(cx)?;
        if !Arc::ptr_eq(&self.current(), expected) {
            return Ok(false);
        }
        // Resolve filesystem context outside the lock, matching legacy replace.
        let base = std::env::current_dir()?;
        let candidate = Arc::new(candidate);
        let retired = {
            let mut current = self.write_index();
            checkpoint(cx)?;
            if !Arc::ptr_eq(&current, expected) {
                return Ok(false);
            }
            validate_admitted(&current)?;
            Self::validate_replacement(&current, &candidate, &base)?;
            checkpoint(cx)?;
            std::mem::replace(&mut *current, candidate)
        };
        tracing::debug!(
            target: "frankensearch.cache",
            "installed caller-admitted v2 cache successor"
        );
        drop(retired);
        Ok(true)
    }
}

#[cfg(test)]
mod tests;
