//! Explicit read-only admission; ordinary readers and all writers stay strict.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use frankensearch_index::mapped_file::ReadOnlyMappedFile;
use rayon::prelude::*;

use super::{
    AuthenticatedFileWitness, KeeperError, KeeperSnapshot, ManifestSegment, RecoveredSegment,
    authenticate_segment_witness, canonical_segment_name, load_manifest_pair, recovery_retryable,
    validate_loaded_schema, validate_recovery_claims, validate_segment_witnesses,
};
use crate::error::QuillError;
use crate::read_open_receipts::{Binding, Identity, Policy, Receipt, ReceiptBook, supported};
use crate::schema::SchemaDescriptor;
use crate::segment::{SegmentLimits, SegmentReader};

struct Guard {
    file: File,
    identity: Identity,
    path: std::path::PathBuf,
}

impl Guard {
    fn check(&self) -> Result<(), KeeperError> {
        if Identity::of_file(&self.file) != Some(self.identity) {
            return Err(changed_file(&self.path));
        }
        Ok(())
    }
}

struct AdmittedSegment {
    segment: RecoveredSegment,
    receipt: Option<Receipt>,
    guard: Option<Guard>,
    reused: bool,
}

struct CheckedWitness {
    witness: AuthenticatedFileWitness,
    receipt: Option<Receipt>,
    guard: Option<Guard>,
    reused: bool,
}

// Stop starting expensive opens beyond a known failure, without making the
// reported error depend on Rayon completion order. Earlier MANIFEST entries
// must still run: one of them may contain the error a strict open would report.
// The ordinary indexed collection joins already-running work and retains its
// results for normal RAII cleanup; this does not interrupt an executing open.
struct AdmissionFrontier {
    first_error: AtomicUsize,
}

impl AdmissionFrontier {
    fn new() -> Self {
        Self {
            first_error: AtomicUsize::new(usize::MAX),
        }
    }

    fn run<T, E>(
        &self,
        position: usize,
        open: impl FnOnce() -> Result<T, E>,
    ) -> Option<Result<T, E>> {
        if position > self.first_error.load(Ordering::Relaxed) {
            return None;
        }
        let result = open();
        if result.is_err() {
            // This atomic carries only a pruning bound, not borrowed error or
            // segment data. Rayon joins publish the actual owned results.
            self.first_error.fetch_min(position, Ordering::Relaxed);
        }
        Some(result)
    }
}

// Retaining every opened descriptor makes the optional fast path exhaust
// RLIMIT_NOFILE on large manifests, then repeat the entire open strictly.
// A bounded selection keeps the most expensive file-prefix checks reusable.
// Equal sizes prefer earlier MANIFEST entries, not worker completion order.
type ReceiptCandidate = Reverse<(u64, Reverse<usize>)>;

enum ReceiptCandidates {
    All,
    Selected(Vec<ReceiptCandidate>),
}

impl ReceiptCandidates {
    fn select(records: &[ManifestSegment], limit: usize) -> Self {
        if limit >= records.len() {
            return Self::All;
        }
        let mut best: BinaryHeap<ReceiptCandidate> = BinaryHeap::new();
        if limit == 0 || best.try_reserve_exact(limit).is_err() {
            return Self::Selected(Vec::new());
        }
        for (position, record) in records.iter().enumerate() {
            let candidate = Reverse((record.file_len, Reverse(position)));
            if best.len() < limit {
                best.push(candidate);
            } else if let Some(mut least_cost) = best.peek_mut()
                && candidate < *least_cost
            {
                *least_cost = candidate;
            }
        }
        // Reuse the heap allocation. Membership checks preserve manifest
        // order without allocating a bitmap for every segment in the index.
        let mut selected = best.into_vec();
        selected.sort_unstable_by_key(|Reverse((_, Reverse(position)))| *position);
        Self::Selected(selected)
    }

    fn allows(&self, position: usize) -> bool {
        match self {
            Self::All => true,
            Self::Selected(selected) => selected
                .binary_search_by_key(&position, |Reverse((_, Reverse(index)))| *index)
                .is_ok(),
        }
    }

    fn len(&self, record_count: usize) -> usize {
        match self {
            Self::All => record_count,
            Self::Selected(selected) => selected.len(),
        }
    }
}

#[cfg(any(target_os = "linux", test))]
fn guard_budget(soft_limit: Option<u64>, open_fds: u64, workers: usize, records: usize) -> usize {
    // Leave descriptors for in-flight opens/clones, manifest/cache operations
    // and unrelated application work. This is a conservative observation, not
    // an OS reservation: the existing EMFILE/ENFILE fallback remains necessary.
    let workers = u64::try_from(workers).unwrap_or(u64::MAX);
    let reserve = workers.saturating_mul(2).saturating_add(16);
    let available = soft_limit
        .unwrap_or(u64::MAX)
        .saturating_sub(open_fds)
        .saturating_sub(reserve);
    usize::try_from(available / 2)
        .unwrap_or(usize::MAX)
        .min(records)
}

#[cfg(target_os = "linux")]
fn receipt_guard_budget(records: usize, workers: usize) -> usize {
    // Count actual descriptors, not just segment files. Failure to inspect
    // process headroom disables this optimization rather than guessing.
    let open_fds = std::fs::read_dir("/proc/self/fd").and_then(|mut entries| {
        entries.try_fold(0_u64, |count, entry| entry.map(|_| count.saturating_add(1)))
    });
    let Ok(open_fds) = open_fds else {
        return 0;
    };
    let limit = rustix::process::getrlimit(rustix::process::Resource::Nofile);
    guard_budget(limit.current, open_fds, workers, records)
}

#[cfg(not(target_os = "linux"))]
fn receipt_guard_budget(_records: usize, _workers: usize) -> usize {
    0
}

impl KeeperSnapshot {
    /// Open immutable published segments with optional Linux-local receipts.
    ///
    /// Only a previously successful file-prefix check can be reused. Container,
    /// schema, MANIFEST and lazy section checks remain active. The cache must be
    /// an existing owner-private directory outside the index. Missing, damaged,
    /// foreign, expired or unusable proofs fall back to a full prefix check.
    /// Unsupported filesystems and platforms also remain strict.
    /// Descriptor headroom bounds receipt use to the largest candidate files;
    /// all remaining segments receive full prefix verification. Unavailable
    /// headroom information or selection storage falls back to a strict open.
    ///
    /// This is explicitly weaker than freshly hashing every byte: metadata-
    /// preserving storage faults are not detected by receipt reuse. The owner
    /// of the MANIFEST and cache is trusted. Use [`Self::open`] for strict
    /// admission. Neither entry point permits concurrent writes to published
    /// memory-mapped segments. Ordinary writers never call this entry point.
    ///
    /// # Errors
    /// Returns the ordinary snapshot errors, including corruption when a mapped
    /// descriptor's identity changes during admission.
    pub fn open_with_local_receipts(
        directory: impl AsRef<Path>,
        schema: SchemaDescriptor,
        cache_directory: impl AsRef<Path>,
    ) -> Result<Self, KeeperError> {
        let open = || {
            Self::open_local_receipts_once(
                directory.as_ref(),
                schema,
                cache_directory.as_ref(),
                Policy::default(),
            )
        };
        let result = match open() {
            Err(error) if recovery_retryable(&error) => open(),
            result => result,
        };
        match result {
            // Receipt guards temporarily retain extra descriptors. That cache
            // policy must not make an otherwise readable index unavailable.
            Err(error) if descriptor_pressure(&error) => Self::open(directory.as_ref(), schema),
            result => result,
        }
    }

    pub(super) fn open_local_receipts_once(
        directory: &Path,
        schema: SchemaDescriptor,
        cache_directory: &Path,
        policy: Policy,
    ) -> Result<Self, KeeperError> {
        // Reader-owned advisory data must not be written into the index. Resolve
        // symlinks first; failures are cache misses, not index-open failures.
        let (Ok(index_path), Ok(cache_path)) =
            (directory.canonicalize(), cache_directory.canonicalize())
        else {
            return Self::open_once(directory, schema);
        };
        if cache_path.starts_with(&index_path) || same_directory(&index_path, &cache_path) {
            return Self::open_once(directory, schema);
        }
        let mut book = ReceiptBook::load(&cache_path, policy);
        if !book.enabled() {
            return Self::open_once(directory, schema);
        }
        let schema_id = schema
            .schema_id()
            .map_err(|source| KeeperError::InvalidSchema { source })?;
        let loaded = load_manifest_pair(directory)?;
        validate_loaded_schema(directory, schema_id, &loaded)?;
        validate_recovery_claims(directory, &loaded)?;
        let records = &loaded.manifest.segments;
        let pool = receipt_pool();
        let candidates = ReceiptCandidates::select(
            records,
            receipt_guard_budget(
                records.len(),
                pool.map_or(1, rayon::ThreadPool::current_num_threads),
            ),
        );
        let candidate_count = candidates.len(records.len());
        if candidate_count == 0 {
            // Release even the cache-directory descriptor before strict open.
            drop(book);
            return Self::open_once(directory, schema);
        }
        let frontier = AdmissionFrontier::new();
        let open = |(position, manifest): (usize, &ManifestSegment)| {
            frontier.run(position, || {
                open_segment(
                    directory,
                    schema,
                    schema_id,
                    manifest,
                    &book,
                    candidates.allows(position),
                )
            })
        };
        // Keep indexed MANIFEST order. A skipped entry is strictly after an
        // actual retained error; flattening cannot turn failure into success.
        // Do not use parallel Result collection, whose chosen error can race.
        let opened: Vec<Option<Result<AdmittedSegment, KeeperError>>> = pool.map_or_else(
            || records.iter().enumerate().map(&open).collect(),
            |pool| pool.install(|| records.par_iter().enumerate().map(&open).collect()),
        );
        let mut segments: Vec<RecoveredSegment> = Vec::new();
        let mut guards = Vec::new();
        segments
            .try_reserve_exact(opened.len())
            .map_err(|error| KeeperError::Io {
                operation: "allocate receipted snapshot segments",
                path: directory.to_path_buf(),
                source: std::io::Error::other(error.to_string()),
            })?;
        guards
            .try_reserve_exact(candidate_count)
            .map_err(|error| KeeperError::Io {
                operation: "allocate receipted snapshot descriptor guards",
                path: directory.to_path_buf(),
                source: std::io::Error::other(error.to_string()),
            })?;
        let mut receipt_hits = 0_usize;
        for result in opened.into_iter().flatten() {
            let admitted = result?;
            receipt_hits += usize::from(admitted.reused);
            if let Some(receipt) = admitted.receipt {
                book.keep(receipt);
            }
            if let Some(guard) = admitted.guard {
                guards.push(guard);
            }
            segments.push(admitted.segment);
        }
        let full_verifications = segments.len() - receipt_hits;
        let snapshot = Self::from_parts(Some(directory.to_path_buf()), schema, loaded, segments)?;
        // Keep ALL receipted-inode descriptors through cross-segment validation.
        // A fast segment cannot mint a proof while another segment is binding.
        for guard in &guards {
            guard.check()?;
        }
        book.persist();
        if !snapshot.quarantined_segments.is_empty() {
            tracing::warn!(
                target: crate::tracing_conventions::TARGET,
                event = "quill.keeper.quarantine_persisted",
                directory = %directory.display(),
                quarantined_segments = snapshot.quarantined_segments.len(),
                estimated_missing_docs = snapshot.estimated_missing_docs(),
                unknown_missing_doc_segments = snapshot.quarantined_segments.iter()
                    .filter(|segment| segment.estimated_missing_docs.is_none()).count(),
                "Quill opened with retained quarantined segments; results must be surfaced as degraded"
            );
        }
        tracing::debug!(
            receipt_hits,
            full_verifications,
            candidate_count,
            "Quill read-only local-receipt admission completed"
        );
        Ok(snapshot)
    }
}

#[cfg(target_os = "linux")]
fn same_directory(left: &Path, right: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (std::fs::metadata(left), std::fs::metadata(right)) {
        (Ok(left), Ok(right)) => left.dev() == right.dev() && left.ino() == right.ino(),
        _ => true, // Unknown identity must fall back to strict admission.
    }
}

#[cfg(not(target_os = "linux"))]
fn same_directory(_left: &Path, _right: &Path) -> bool {
    true
}

fn descriptor_pressure(error: &KeeperError) -> bool {
    #[cfg(target_os = "linux")]
    {
        let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(error);
        while let Some(error) = cause {
            if error.downcast_ref::<std::io::Error>().is_some_and(|error| {
                // Linux ENFILE / EMFILE. Receipt reuse is Linux-only.
                matches!(error.raw_os_error(), Some(23 | 24))
            }) {
                return true;
            }
            cause = error.source();
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = error;
    false
}

fn receipt_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(
                std::thread::available_parallelism()
                    .map_or(1, usize::from)
                    .min(8),
            )
            .thread_name(|index| format!("quill-receipt-open-{index}"))
            .build()
            .ok()
    })
    .as_ref()
}

fn changed_file(path: &Path) -> KeeperError {
    KeeperError::SegmentOpen {
        path: path.to_path_buf(),
        source: QuillError::IndexCorrupted {
            path: path.to_path_buf(),
            detail: "mapped segment identity changed during read-only admission".to_owned(),
        },
    }
}

fn open_segment(
    directory: &Path,
    schema: SchemaDescriptor,
    schema_id: u64,
    manifest: &ManifestSegment,
    book: &ReceiptBook,
    allow_receipt: bool,
) -> Result<AdmittedSegment, KeeperError> {
    let path = directory.join(canonical_segment_name(manifest.segment_id));
    let (reader, checked) = SegmentReader::open_published_checked(
        &path,
        schema,
        SegmentLimits::default(),
        |reader, file| {
            Ok(check_witness(
                &path,
                schema_id,
                manifest,
                reader,
                file,
                book,
                allow_receipt,
            ))
        },
    )
    .map_err(|source| KeeperError::SegmentOpen {
        path: path.clone(),
        source,
    })?;
    let checked = checked?;
    let segment = RecoveredSegment::bind(path, manifest.clone(), reader, schema, checked.witness)?;
    if let Some(guard) = &checked.guard {
        guard.check()?;
    }
    Ok(AdmittedSegment {
        segment,
        receipt: checked.receipt,
        guard: checked.guard,
        reused: checked.reused,
    })
}

fn check_witness(
    path: &Path,
    schema_id: u64,
    manifest: &ManifestSegment,
    reader: &SegmentReader<ReadOnlyMappedFile>,
    file: &mut File,
    book: &ReceiptBook,
    allow_receipt: bool,
) -> Result<CheckedWitness, KeeperError> {
    let before = (allow_receipt && supported(file))
        .then(|| Identity::of_file(file))
        .flatten();
    let Some(before) = before else {
        return authenticate_segment_witness(path, manifest, reader, file).map(|witness| {
            CheckedWitness {
                witness,
                receipt: None,
                guard: None,
                reused: false,
            }
        });
    };
    let binding = Binding {
        schema_id,
        segment_id: manifest.segment_id,
        file_len: manifest.file_len,
        file_xxh3: manifest.file_xxh3,
    };
    let existing = book.admitted(binding, before);
    let reused = existing.is_some();
    let witness = if reused {
        let file_xxh3 = reader.file_xxh3();
        validate_segment_witnesses(path, manifest, reader, || Ok(file_xxh3))?;
        // A hit covers ONLY the prefix. Never seed the lazy section-check cache.
        AuthenticatedFileWitness {
            file_xxh3,
            #[cfg(test)]
            full_prefix_hash_count: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    } else {
        authenticate_segment_witness(path, manifest, reader, file)?
    };
    let after = Identity::of_file(file);
    if after != Some(before) {
        return Err(changed_file(path));
    }
    let file = file
        .try_clone()
        .map_err(|source| KeeperError::SegmentOpen {
            path: path.to_path_buf(),
            source: QuillError::Io(source),
        })?;
    let receipt = existing.or_else(|| book.verified(binding, before, after));
    Ok(CheckedWitness {
        witness,
        receipt,
        reused,
        guard: Some(Guard {
            file,
            identity: before,
            path: path.to_path_buf(),
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_admission_stops_later_sequential_work_without_poisoning_reopen() {
        let frontier = AdmissionFrontier::new();
        let calls = AtomicUsize::new(0);
        let results: Vec<_> = (0..10_000)
            .map(|position| {
                frontier.run(position, || {
                    calls.fetch_add(1, Ordering::Relaxed);
                    if position == 7 {
                        Err(position)
                    } else {
                        Ok(position)
                    }
                })
            })
            .collect();
        assert_eq!(calls.load(Ordering::Relaxed), 8);
        assert_eq!(
            results.into_iter().flatten().collect::<Result<Vec<_>, _>>(),
            Err(7)
        );

        let reopened = AdmissionFrontier::new();
        let values: Result<Vec<_>, usize> = (0..12)
            .filter_map(|position| reopened.run(position, || Ok(position)))
            .collect();
        assert_eq!(values.unwrap(), (0..12).collect::<Vec<_>>());
    }

    #[test]
    fn admission_error_is_manifest_first_for_every_small_schedule() {
        fn check(schedule: &[usize; 5]) {
            for failed in 0_u32..32 {
                let frontier = AdmissionFrontier::new();
                let mut results = vec![None; 5];
                let mut called = [false; 5];
                for &position in schedule {
                    results[position] = frontier.run(position, || {
                        called[position] = true;
                        if failed & (1 << position) == 0 {
                            Ok(position)
                        } else {
                            Err(position)
                        }
                    });
                }
                let first = (0..5).find(|&position| failed & (1 << position) != 0);
                let required = first.map_or(5, |position| position + 1);
                assert!(called[..required].iter().all(|&called| called));
                let actual = results.into_iter().flatten().collect::<Result<Vec<_>, _>>();
                match first {
                    Some(position) => assert_eq!(actual, Err(position)),
                    None => assert_eq!(actual, Ok((0..5).collect())),
                }
            }
        }

        fn permute(schedule: &mut [usize; 5], start: usize) {
            if start == schedule.len() {
                check(schedule);
                return;
            }
            for position in start..schedule.len() {
                schedule.swap(start, position);
                permute(schedule, start + 1);
                schedule.swap(start, position);
            }
        }

        permute(&mut [0, 1, 2, 3, 4], 0);
    }

    #[test]
    fn parallel_admission_preserves_earliest_error_and_drops_partial_results() {
        use std::sync::Arc;

        struct Tracked(Arc<AtomicUsize>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for failed in [None, Some(0), Some(3), Some(31)] {
            let frontier = AdmissionFrontier::new();
            let constructed = AtomicUsize::new(0);
            let dropped = Arc::new(AtomicUsize::new(0));
            let results: Vec<_> = pool.install(|| {
                (0..64)
                    .into_par_iter()
                    .map(|position| {
                        frontier.run(position, || {
                            if failed
                                .is_some_and(|first| position == first || position == first + 1)
                            {
                                Err(position)
                            } else {
                                constructed.fetch_add(1, Ordering::Relaxed);
                                Ok(Tracked(Arc::clone(&dropped)))
                            }
                        })
                    })
                    .collect()
            });
            let result = results.into_iter().flatten().collect::<Result<Vec<_>, _>>();
            match failed {
                Some(position) => assert!(matches!(result, Err(error) if error == position)),
                None => assert_eq!(result.as_ref().unwrap().len(), 64),
            }
            drop(result);
            assert_eq!(
                dropped.load(Ordering::Relaxed),
                constructed.load(Ordering::Relaxed)
            );
        }
    }

    #[test]
    fn already_running_admission_finishes_but_new_later_work_is_skipped() {
        use std::sync::Barrier;

        let frontier = AdmissionFrontier::new();
        let entered = Barrier::new(2);
        let release = Barrier::new(2);
        let later_calls = AtomicUsize::new(0);
        let (failure, skipped, running) = std::thread::scope(|scope| {
            let pending = scope.spawn(|| {
                frontier.run(9, || {
                    entered.wait();
                    release.wait();
                    Ok::<_, usize>(9)
                })
            });
            entered.wait();
            let failure = frontier.run(2, || Err::<usize, _>(2));
            let skipped = frontier.run(10, || {
                later_calls.fetch_add(1, Ordering::Relaxed);
                Ok::<_, usize>(10)
            });
            // Release and join before assertions so even a failing assertion
            // cannot leave a test thread blocked at the rendezvous.
            release.wait();
            (failure, skipped, pending.join().unwrap())
        });
        assert_eq!(failure, Some(Err(2)));
        assert_eq!(skipped, None);
        assert_eq!(running, Some(Ok(9)));
        assert_eq!(later_calls.load(Ordering::Relaxed), 0);
    }

    fn records(sizes: &[u64]) -> Vec<ManifestSegment> {
        sizes
            .iter()
            .enumerate()
            .map(|(position, &file_len)| {
                let position = u64::try_from(position).expect("small fixture");
                ManifestSegment {
                    segment_id: position,
                    seal_seq: 1,
                    file_len,
                    file_xxh3: 0,
                    docid_lo: position,
                    docid_hi: position + 1,
                    doc_count: 1,
                    tombstones: crate::TombstoneSet::new(),
                }
            })
            .collect()
    }

    #[test]
    fn descriptor_budget_reserves_worker_and_application_headroom() {
        assert_eq!(guard_budget(Some(64), 5, 8, 128), 13);
        assert_eq!(guard_budget(Some(64), 5, 1, 128), 20);
        assert_eq!(guard_budget(Some(64), 5, 8, 2), 2);
        assert_eq!(guard_budget(Some(64), 5, 8, 0), 0);
        assert_eq!(guard_budget(None, 5, 8, 128), 128);
    }

    #[test]
    fn descriptor_budget_saturates_instead_of_wrapping() {
        assert_eq!(guard_budget(Some(32), 16, 8, 128), 0);
        assert_eq!(guard_budget(Some(4), 5, 1, 128), 0);
        assert_eq!(guard_budget(None, u64::MAX, 1, usize::MAX), 0);
        assert_eq!(guard_budget(Some(64), 5, usize::MAX, 128), 0);
    }

    #[test]
    fn largest_candidates_win_with_manifest_order_ties() {
        let records = records(&[9, 11, 11, 3, 11]);
        let selected = ReceiptCandidates::select(&records, 2);
        assert_eq!(selected.len(records.len()), 2);
        let positions: Vec<_> = (0..records.len())
            .filter(|&position| selected.allows(position))
            .collect();
        assert_eq!(positions, [1, 2]);
        let none = ReceiptCandidates::select(&records, 0);
        assert_eq!(none.len(records.len()), 0);
        assert!((0..records.len()).all(|position| !none.allows(position)));
        let all = ReceiptCandidates::select(&records, usize::MAX);
        assert_eq!(all.len(records.len()), records.len());
        assert!((0..records.len()).all(|position| all.allows(position)));
    }

    #[test]
    fn bounded_selection_matches_an_independent_full_sort() {
        for len in 0..=6_u32 {
            let len_usize = usize::try_from(len).expect("small fixture");
            for mut code in 0..3_usize.pow(len) {
                let sizes: Vec<_> = (0..len_usize)
                    .map(|_| {
                        let size = [0, 4096, u64::MAX][code % 3];
                        code /= 3;
                        size
                    })
                    .collect();
                let records = records(&sizes);
                for limit in 0..=len_usize + 1 {
                    let selected = ReceiptCandidates::select(&records, limit);
                    let mut oracle: Vec<_> = (0..len_usize).collect();
                    oracle.sort_unstable_by_key(|&position| (Reverse(sizes[position]), position));
                    oracle.truncate(limit);
                    assert_eq!(selected.len(records.len()), oracle.len());
                    for position in 0..len_usize {
                        assert_eq!(selected.allows(position), oracle.contains(&position));
                    }
                }
            }
        }
    }
}
