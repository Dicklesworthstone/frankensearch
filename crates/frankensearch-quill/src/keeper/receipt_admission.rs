//! Explicit read-only admission; ordinary readers and all writers stay strict.

use std::fs::File;
use std::path::Path;

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

impl KeeperSnapshot {
    /// Open immutable published segments with optional Linux-local receipts.
    ///
    /// Only a previously successful file-prefix check can be reused. Container,
    /// schema, MANIFEST and lazy section checks remain active. The cache must be
    /// an existing owner-private directory outside the index. Missing, damaged,
    /// foreign, expired or unusable proofs fall back to a full prefix check.
    /// Unsupported filesystems and platforms also remain strict.
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
        let open = |manifest: &ManifestSegment| {
            open_segment(directory, schema, schema_id, manifest, &book)
        };
        let records = &loaded.manifest.segments;
        // Collect in MANIFEST order so parallel work does not randomize errors.
        let opened: Vec<Result<AdmittedSegment, KeeperError>> = receipt_pool().map_or_else(
            || records.iter().map(&open).collect(),
            |pool| pool.install(|| records.par_iter().map(&open).collect()),
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
            .try_reserve_exact(opened.len())
            .map_err(|error| KeeperError::Io {
                operation: "allocate receipted snapshot descriptor guards",
                path: directory.to_path_buf(),
                source: std::io::Error::other(error.to_string()),
            })?;
        let mut receipt_hits = 0_usize;
        for result in opened {
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
        // Keep ALL mapped-inode descriptors through cross-segment validation.
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
) -> Result<AdmittedSegment, KeeperError> {
    let path = directory.join(canonical_segment_name(manifest.segment_id));
    let (reader, checked) = SegmentReader::open_published_checked(
        &path,
        schema,
        SegmentLimits::default(),
        |reader, file| {
            Ok(check_witness(
                &path, schema_id, manifest, reader, file, book,
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
) -> Result<CheckedWitness, KeeperError> {
    let before = supported(file).then(|| Identity::of_file(file)).flatten();
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
