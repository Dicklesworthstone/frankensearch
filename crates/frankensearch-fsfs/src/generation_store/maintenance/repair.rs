//! Reconstruct damaged vector artifacts into an independent, unselected bundle.
//!
//! The trusted old manifest, not a sidecar's checksum, defines the required
//! bytes. Only the two fsfs vector roles can be repaired. Every other artifact,
//! including the selected FEC sidecars, must remain intact. No original file
//! is opened for writing, no source document is reread and no model is invoked.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use frankensearch_durability::{
    DefaultSymbolCodec, DurabilityConfig, FileProtector, FileRecoveryOutcome,
};
use sha2::{Digest, Sha256};

use super::super::{Artifact, MAX_ARTIFACTS, MAX_TREE_DEPTH};
use super::{
    BundleManifest, COMPLETE_GENERATION_MANIFEST, CompleteGenerationStore, GENERATIONS,
    MAX_MANIFEST_BYTES, NEXT_GENERATION, POINTER_MAGIC, PreparedGenerationRestore,
    PublicationLease, checkpoint, decode_pointer, digest, ensure_selection, hex, invalid,
    open_regular, pin_generation, read_bounded_regular, require_directory, selection_evidence,
    sync_directory, sync_tree, write_new_synced,
};

// These are the fsfs on-disk vector roles, not an extension-based permission
// to reconstruct catalogs, lexical segments, manifests or arbitrary .fsvi files.
const VECTOR_PATHS: [&str; 2] = ["vector/index.fsvi", "vector/quality.fsvi"];
const MAX_REPAIR_SOURCE_BYTES: u64 = 128 * 1024 * 1024;
const MAX_REPAIR_SIDECAR_BYTES: u64 = 256 * 1024 * 1024;

type FileInventory = BTreeMap<String, (u64, String)>;

impl CompleteGenerationStore {
    /// Reconstruct receipted vector damage in a new, unselected complete bundle.
    ///
    /// The generation ID and manifest digest must come from a trusted original
    /// publication receipt. Both canonical fsfs vector roles can recover bitrot,
    /// truncation or absence using their intact, receipted `.fec` sidecars. A
    /// damaged catalog, lexical artifact, inventory or sidecar is refused. No
    /// directory scan chooses another generation and no model is substituted.
    ///
    /// Every original artifact is read-only. All surviving artifacts are copied
    /// to new inodes; recovery runs against those private copies and its complete
    /// output must match the ORIGINAL SHA-256 and length. A new manifest names
    /// the copied cohort. The returned ordinary restore plan still requires
    /// consumer admission and compares the exact selection captured BEFORE
    /// copying. Preparing, failing or dropping this plan never changes CURRENT.
    /// An all-intact target is refused; use `prepare_restore` without a copy.
    ///
    /// `max_copy_bytes` bounds both the original sealed inventory and observed
    /// input inventory. Each damaged vector is additionally limited to 128 MiB
    /// and its sidecar to 256 MiB before decoding. These are payload/I/O bounds,
    /// not an allocator/RSS ceiling; codec working memory and a temporary repaired
    /// output add overhead. Filesystem and codec work are synchronous on the
    /// caller's lane. Cancellation brackets decoding and each copy chunk, but
    /// cannot preempt an executing codec call or OS operation.
    ///
    /// Failed partial copies are retained, never automatically deleted or used
    /// as recovery authority. This uses the existing cooperative publication
    /// lease and reader pins, not hostile-directory safety or antirollback.
    /// Available on Unix, like this repair path's directory durability barrier.
    ///
    /// # Errors
    /// Returns invalid receipts/limits, unsupported damage, insufficient repair
    /// symbols, content mismatch, publisher contention, I/O or cancellation.
    pub fn prepare_vector_repair(
        &self,
        cx: &Cx,
        id: &str,
        manifest_sha256: &str,
        max_copy_bytes: u64,
    ) -> SearchResult<PreparedGenerationRestore> {
        checkpoint(cx)?;
        if max_copy_bytes == 0 || id.len() != 60 || !hex(manifest_sha256, 64) {
            return Err(invalid(&self.root, "invalid repair receipt or zero copy budget"));
        }
        let pointer = format!("{POINTER_MAGIC}\n{id}\n{manifest_sha256}\n");
        decode_pointer(pointer.as_bytes(), &self.root)?;
        let lease = PublicationLease::acquire(&self.root)?;
        lease.fence("complete-generation vector repair entry")?;
        let expected_selection = selection_evidence(&self.root)?;
        let parent = self.root.join(GENERATIONS);
        require_directory(&parent)?;
        let source = parent.join(id);
        require_directory(&source)?;
        let source_pin = pin_generation(&self.root, id)?;
        let bytes = read_bounded_regular(
            &source.join(COMPLETE_GENERATION_MANIFEST), MAX_MANIFEST_BYTES,
        )?;
        if digest(&bytes) != manifest_sha256 {
            return Err(invalid(&source, "repair inventory differs from the trusted receipt"));
        }
        let mut manifest: BundleManifest = serde_json::from_slice(&bytes)
            .map_err(|error| invalid(&source, &format!("invalid repair inventory: {error}")))?;
        validate_manifest(&source, &manifest, id, max_copy_bytes)?;
        // Repair cannot use the normal same-process digest memo: media damage
        // may be precisely why the caller asked to read the bytes again.
        let observed = fresh_inventory(cx, &source, max_copy_bytes, true)?;
        let damaged = damaged_vectors(&source, &manifest, &observed)?;
        ensure_selection(&self.root, expected_selection.as_deref())?;
        checkpoint(cx)?;
        lease.fence("complete-generation vector repair copy")?;
        let (candidate, candidate_id) = new_directory(cx, &parent)?;

        // Copy EVERY extant input by its freshly observed checksum, including
        // damaged vectors. The codec sees only exclusively-created local files,
        // never a source pathname that could change while decoding is pending.
        for artifact in &manifest.files {
            checkpoint(cx)?;
            let destination = candidate.join(&artifact.path);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            if let Some((size, sha256)) = observed.get(&artifact.path) {
                copy_checked(cx, &source.join(&artifact.path), &destination, *size, sha256)?;
            }
        }
        let protector = FileProtector::new(Arc::new(DefaultSymbolCodec), DurabilityConfig::default())?;
        for artifact in manifest.files.iter().filter(|file| damaged.contains(&file.path)) {
            checkpoint(cx)?;
            lease.fence("complete-generation vector repair decode")?;
            recover_vector(cx, &candidate, artifact, &protector)?;
        }
        // A successful RaptorQ decode or CRC is not enough. All files must have
        // exactly the previously published content, including non-vector arms.
        if fresh_inventory(cx, &candidate, max_copy_bytes, false)? != expected_files(&manifest) {
            return Err(invalid(&candidate, "repaired bundle differs from the original inventory"));
        }
        manifest.generation = candidate_id.clone();
        let encoded = serde_json::to_vec(&manifest)
            .map_err(|error| invalid(&candidate, &error.to_string()))?;
        if u64::try_from(encoded.len()).unwrap_or(u64::MAX) > MAX_MANIFEST_BYTES {
            return Err(invalid(&candidate, "repaired manifest exceeds the descriptor limit"));
        }
        write_new_synced(&candidate.join(COMPLETE_GENERATION_MANIFEST), &encoded)?;
        sync_tree(cx, &candidate, 0)?;
        sync_directory(&parent)?;
        let target = self.open_retained(cx, &candidate_id, &digest(&encoded))?;
        ensure_selection(&self.root, expected_selection.as_deref())?;
        checkpoint(cx)?;
        lease.fence("complete-generation vector repair prepared")?;
        drop(source_pin);
        drop(lease);
        Ok(PreparedGenerationRestore {
            store: self.clone(),
            target,
            expected_selection,
        })
    }
}

fn expected_files(manifest: &BundleManifest) -> FileInventory {
    manifest.files.iter().map(|file| {
        (file.path.clone(), (file.bytes, file.sha256.clone()))
    }).collect()
}

fn validate_manifest(
    root: &Path,
    manifest: &BundleManifest,
    id: &str,
    budget: u64,
) -> SearchResult<()> {
    if manifest.schema_version != 1 || manifest.generation != id
        || manifest.files.is_empty() || manifest.files.len() > MAX_ARTIFACTS
    {
        return Err(invalid(root, "invalid repair inventory identity, version or file count"));
    }
    let mut previous: Option<&str> = None;
    let mut remaining = budget;
    let mut files = BTreeSet::new();
    for file in &manifest.files {
        let path = Path::new(&file.path);
        if file.path.is_empty() || file.path.contains(['\\', '\0'])
            || file.path.len() > usize::from(u16::MAX)
            || file.path.split('/').any(|part| part.is_empty() || part == "." || part == "..")
            || path.components().any(|part| !matches!(part, Component::Normal(_)))
            || path.components().count() > MAX_TREE_DEPTH + 1
            || file.path == COMPLETE_GENERATION_MANIFEST
            || previous.is_some_and(|old| old >= file.path.as_str())
            || !hex(&file.sha256, 64) || file.bytes > remaining
        {
            return Err(invalid(root, "noncanonical or oversized repair inventory"));
        }
        // A file cannot also be an ancestor directory of another artifact.
        for parent in path.ancestors().skip(1).filter_map(Path::to_str) {
            if files.contains(parent) {
                return Err(invalid(root, "repair inventory aliases file and directory roles"));
            }
        }
        remaining -= file.bytes;
        previous = Some(&file.path);
        files.insert(file.path.as_str());
    }
    Ok(())
}

fn damaged_vectors(
    root: &Path,
    manifest: &BundleManifest,
    observed: &FileInventory,
) -> SearchResult<BTreeSet<String>> {
    let expected = expected_files(manifest);
    if observed.keys().any(|path| !expected.contains_key(path)) {
        return Err(invalid(root, "unreceipted files in the damaged bundle"));
    }
    let mut damaged = BTreeSet::new();
    for file in &manifest.files {
        if observed.get(&file.path) == expected.get(&file.path) {
            continue;
        }
        if !VECTOR_PATHS.contains(&file.path.as_str()) {
            return Err(invalid(root, "repair requires intact catalog, lexical, metadata and sidecar files"));
        }
        let sidecar = format!("{}.fec", file.path);
        let protected = expected.get(&sidecar).ok_or_else(|| {
            invalid(root, "damaged vector has no receipted repair sidecar")
        })?;
        if observed.get(&sidecar) != Some(protected)
            || file.bytes > MAX_REPAIR_SOURCE_BYTES
            || observed.get(&file.path).is_some_and(|(size, _)| *size > MAX_REPAIR_SOURCE_BYTES)
            || protected.0 > MAX_REPAIR_SIDECAR_BYTES
        {
            return Err(invalid(root, "repair sidecar changed or vector recovery exceeds its payload bound"));
        }
        damaged.insert(file.path.clone());
    }
    if damaged.is_empty() {
        return Err(invalid(root, "no vector damage; use ordinary receipt-directed restore without a copy"));
    }
    Ok(damaged)
}

/// Bound the entire filesystem inventory before hashing payloads or allocating
/// a successor. Empty directories also consume the traversal-entry budget.
fn fresh_inventory(cx: &Cx, root: &Path, budget: u64, sealed: bool) -> SearchResult<FileInventory> {
    let mut files = BTreeMap::new();
    let mut remaining = budget;
    let mut entries = 0_usize;
    let mut pending = vec![(root.to_path_buf(), 0_usize)];
    while let Some((directory, depth)) = pending.pop() {
        checkpoint(cx)?;
        require_directory(&directory)?;
        if depth > MAX_TREE_DEPTH {
            return Err(invalid(root, "repair inventory exceeds depth bound"));
        }
        for entry in fs::read_dir(&directory)? {
            checkpoint(cx)?;
            let path = entry?.path();
            if sealed && path == root.join(COMPLETE_GENERATION_MANIFEST) {
                continue;
            }
            entries += 1;
            if entries > MAX_ARTIFACTS {
                return Err(invalid(root, "repair inventory exceeds entry bound"));
            }
            let metadata = fs::symlink_metadata(&path)?;
            if metadata.is_dir() {
                pending.push((path, depth + 1));
            } else if metadata.is_file() {
                if metadata.len() > remaining {
                    return Err(invalid(root, "observed repair input exceeds copy budget"));
                }
                remaining -= metadata.len();
                let relative = path.strip_prefix(root).ok().and_then(Path::to_str)
                    .ok_or_else(|| invalid(root, "repair artifact has no relative UTF-8 path"))?;
                files.insert(relative.to_owned(), metadata.len());
            } else {
                return Err(invalid(&path, "repair refuses symlinks and special files"));
            }
        }
    }
    let mut inventory = BTreeMap::new();
    for (relative, size) in files {
        checkpoint(cx)?;
        let hash = transfer_checked(cx, &root.join(&relative), &mut std::io::sink(), size)?;
        inventory.insert(relative, (size, hash));
    }
    Ok(inventory)
}

/// One bounded, no-follow regular-file read shared by verification and copying.
/// Never consult digest_memo, including when ctime/mtime did not change.
fn transfer_checked(
    cx: &Cx,
    source: &Path,
    destination: &mut impl Write,
    expected_size: u64,
) -> SearchResult<String> {
    checkpoint(cx)?;
    let mut input = open_regular(source)?;
    if input.metadata()?.len() != expected_size {
        return Err(invalid(source, "repair input length changed"));
    }
    let mut hasher = Sha256::new();
    let mut remaining = expected_size;
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        checkpoint(cx)?;
        let read_limit = usize::try_from(remaining.min(buffer.len() as u64)).unwrap_or(buffer.len());
        // The extra byte detects growth without consuming unbounded input.
        let limit = if read_limit == 0 { 1 } else { read_limit };
        let count = match input.read(&mut buffer[..limit]) {
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            result => result?,
        };
        checkpoint(cx)?;
        if count == 0 {
            break;
        }
        if count as u64 > remaining {
            return Err(invalid(source, "repair input grew during reading"));
        }
        remaining -= count as u64;
        hasher.update(&buffer[..count]);
        destination.write_all(&buffer[..count])?;
    }
    if remaining != 0 || input.metadata()?.len() != expected_size {
        return Err(invalid(source, "repair input truncated during reading"));
    }
    checkpoint(cx)?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn copy_checked(cx: &Cx, source: &Path, target: &Path, size: u64, hash: &str) -> SearchResult<()> {
    checkpoint(cx)?;
    let mut output = OpenOptions::new().write(true).create_new(true).open(target)?;
    if transfer_checked(cx, source, &mut output, size)? != hash {
        return Err(invalid(source, "repair input changed between observation and copying"));
    }
    output.sync_all()?;
    checkpoint(cx)
}

fn new_directory(cx: &Cx, parent: &Path) -> SearchResult<(PathBuf, String)> {
    for _ in 0..64 {
        checkpoint(cx)?;
        let serial = NEXT_GENERATION.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| invalid(parent, "generation serial exhausted"))?;
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let id = format!("g-{now:032x}-{:08x}-{serial:016x}", std::process::id());
        let path = parent.join(&id);
        match fs::create_dir(&path) {
            Ok(()) => return Ok((path, id)),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
    }
    Err(invalid(parent, "cannot allocate an independent repair generation"))
}

fn recover_vector(cx: &Cx, root: &Path, artifact: &Artifact, protector: &FileProtector) -> SearchResult<()> {
    let path = root.join(&artifact.path);
    let sidecar = FileProtector::sidecar_path(&path);
    {
        let bytes = read_bounded_regular(&sidecar, MAX_REPAIR_SIDECAR_BYTES)?;
        let (header, _) = frankensearch_durability::deserialize_repair_trailer(&bytes)?;
        if header.source_len != artifact.bytes {
            return Err(invalid(&sidecar, "repair sidecar does not describe the receipted vector length"));
        }
    }
    checkpoint(cx)?;
    let outcome = protector.recover_file_bytes(&path, &sidecar);
    checkpoint(cx)?;
    let bytes = match outcome? {
        FileRecoveryOutcome::Recovered { bytes, .. } => bytes,
        FileRecoveryOutcome::NotNeeded => {
            // A weak sidecar checksum cannot overrule the trusted SHA-256.
            return Err(invalid(&path, "sidecar claims intact data that differs from the trusted inventory"));
        }
        FileRecoveryOutcome::Unrecoverable { reason, .. } => {
            return Err(SearchError::IndexCorrupted {
                path,
                detail: format!("receipted vector could not be reconstructed: {reason:?}"),
            });
        }
    };
    if bytes.len() as u64 != artifact.bytes || digest(&bytes) != artifact.sha256 {
        return Err(invalid(&path, "reconstructed bytes do not match the original SHA-256 and length"));
    }
    checkpoint(cx)?;
    // The source evidence remains in the ORIGINAL bundle. Only our private
    // unselected damaged copy is replaced; readers can never map this candidate
    // before its full seal is admitted. Failed temporary outputs are retained.
    let temporary = root.join(format!(".FSFS-REPAIRED-{}", artifact.path.replace('/', "-")));
    write_new_synced(&temporary, &bytes)?;
    checkpoint(cx)?;
    fs::rename(&temporary, &path)?;
    checkpoint(cx)
}

#[cfg(test)]
mod tests;