//! Self-healing Tantivy segment wrapper with `RaptorQ` durability.
//!
//! Wraps a Tantivy [`Index`] to add per-segment `.seg.fec` sidecar files
//! containing `RaptorQ` repair symbols. The approach is entirely external
//! to Tantivy: we enumerate segments via [`Index::searchable_segment_metas`],
//! protect their component files through the generic [`FileProtector`],
//! and verify/repair on open.
//!
//! # Lifecycle
//!
//! 1. **On commit** — call [`DurableTantivyIndex::protect_segments`] to
//!    generate `.seg.fec` sidecars for any new or changed segments.
//! 2. **On open** — call [`DurableTantivyIndex::verify_and_repair`] to
//!    check all segments and auto-repair any corruption.
//! 3. **On merge/GC** — orphaned `.seg.fec` files whose segments no longer
//!    exist are lazily cleaned up during [`DurableTantivyIndex::protect_segments`].

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use frankensearch_core::{SearchError, SearchResult};
use fsqlite_core::raptorq_integration::SymbolCodec;
use tantivy::Index;
use tracing::{debug, info, warn};

use crate::config::DurabilityConfig;
use crate::file_protector::{
    FileHealth, FileProtectionResult, FileProtector, FileRepairOutcome, FileVerifyResult,
};
use crate::metrics::{DurabilityMetrics, DurabilityMetricsSnapshot};

/// Report produced after protecting segments.
#[derive(Debug, Clone)]
pub struct SegmentProtectionReport {
    /// Number of segments that were newly protected.
    pub segments_protected: usize,
    /// Number of segments already protected (skipped).
    pub segments_already_protected: usize,
    /// Number of orphaned sidecar files cleaned up.
    pub orphans_cleaned: usize,
    /// Total source bytes across all newly protected segments.
    pub total_source_bytes: u64,
    /// Total repair sidecar bytes generated.
    pub total_repair_bytes: u64,
    /// Wall-clock time for the entire protection pass.
    pub encode_time: Duration,
}

/// Report produced after verifying (and optionally repairing) segments.
#[derive(Debug, Clone)]
pub struct SegmentHealthReport {
    /// Total segment component files checked.
    pub files_checked: usize,
    /// Files confirmed intact.
    pub files_intact: usize,
    /// Files that were repaired successfully.
    pub files_repaired: usize,
    /// Files that could not be repaired.
    pub files_unrecoverable: usize,
    /// Files without a sidecar (unprotected).
    pub files_unprotected: usize,
    /// Wall-clock time for verification.
    pub verify_time: Duration,
    /// Wall-clock time for repair attempts.
    pub repair_time: Duration,
}

/// Thin helper for applying durability sidecars to individual Tantivy segment files.
///
/// This is the low-level building block. For full lifecycle management,
/// use [`DurableTantivyIndex`].
#[derive(Debug, Clone)]
pub struct TantivySegmentProtector {
    protector: FileProtector,
}

impl TantivySegmentProtector {
    pub fn new(codec: Arc<dyn SymbolCodec>, config: DurabilityConfig) -> SearchResult<Self> {
        Ok(Self {
            protector: FileProtector::new(codec, config)?,
        })
    }

    pub fn protect_segment(&self, segment_path: &Path) -> SearchResult<FileProtectionResult> {
        self.protector.protect_file(segment_path)
    }

    pub fn verify_segment(&self, segment_path: &Path) -> SearchResult<FileVerifyResult> {
        let sidecar = FileProtector::sidecar_path(segment_path);
        self.protector.verify_file(segment_path, &sidecar)
    }

    pub fn repair_segment(&self, segment_path: &Path) -> SearchResult<FileRepairOutcome> {
        let sidecar = FileProtector::sidecar_path(segment_path);
        self.protector.repair_file(segment_path, &sidecar)
    }

    pub fn protect_segments<'a, I>(&self, segments: I) -> SearchResult<Vec<FileProtectionResult>>
    where
        I: IntoIterator<Item = &'a Path>,
    {
        segments
            .into_iter()
            .map(|segment| self.protect_segment(segment))
            .collect()
    }
}

/// Tantivy index wrapped with per-segment `RaptorQ` durability.
///
/// Provides segment-level protect/verify/repair operations that integrate
/// with Tantivy's segment lifecycle. Component files within each segment
/// are protected individually, with each producing its own `.fec` sidecar.
///
/// # Usage
///
/// ```ignore
/// let durable = DurableTantivyIndex::open(data_dir, codec, config)?;
///
/// // After committing new documents:
/// let protection = durable.protect_segments()?;
///
/// // On startup, verify and auto-repair:
/// let health = durable.verify_and_repair()?;
/// ```
#[derive(Debug)]
pub struct DurableTantivyIndex {
    index: Index,
    protector: FileProtector,
    data_dir: PathBuf,
    metrics: Arc<DurabilityMetrics>,
}

impl DurableTantivyIndex {
    /// Open a Tantivy index with durability protection.
    ///
    /// The caller is responsible for creating or opening the Tantivy `Index`
    /// (so they control the schema and configuration). This wrapper adds
    /// the durability layer on top.
    pub fn new(
        index: Index,
        data_dir: PathBuf,
        codec: Arc<dyn SymbolCodec>,
        config: DurabilityConfig,
    ) -> SearchResult<Self> {
        let metrics = Arc::new(DurabilityMetrics::default());
        let protector = FileProtector::new_with_metrics(codec, config, Arc::clone(&metrics))?;
        Ok(Self {
            index,
            protector,
            data_dir,
            metrics,
        })
    }

    /// Access the inner Tantivy index for search/write operations.
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Access the data directory path.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Get a durability metrics snapshot.
    pub fn metrics_snapshot(&self) -> DurabilityMetricsSnapshot {
        self.metrics.snapshot()
    }

    /// Protect all unprotected segment component files.
    ///
    /// Enumerates the current searchable segments, generates `.fec` sidecars
    /// for any component files that don't already have one, and cleans up
    /// orphaned sidecars from segments that no longer exist (post-merge GC).
    pub fn protect_segments(&self) -> SearchResult<SegmentProtectionReport> {
        let start = Instant::now();
        let mut protected = 0_usize;
        let mut already_protected = 0_usize;
        let mut total_source = 0_u64;
        let mut total_repair = 0_u64;

        let segment_metas =
            self.index
                .searchable_segment_metas()
                .map_err(|e| SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(e),
                })?;

        // Collect all known segment files for orphan detection.
        let mut known_files = HashSet::new();

        for meta in &segment_metas {
            let files = meta.list_files();
            for relative_path in &files {
                // Validate that the relative path doesn't escape data_dir
                // (corrupted metadata could contain "../" or absolute paths).
                if !is_safe_relative_path(relative_path) {
                    warn!(
                        path = %relative_path.display(),
                        "segment file path contains unsafe components, skipping"
                    );
                    continue;
                }
                let abs_path = self.data_dir.join(relative_path);
                known_files.insert(abs_path.clone());

                if !abs_path.exists() {
                    debug!(
                        path = %abs_path.display(),
                        "segment component file does not exist, skipping"
                    );
                    continue;
                }

                let sidecar = FileProtector::sidecar_path(&abs_path);
                if sidecar.exists() {
                    already_protected += 1;
                    continue;
                }

                match self.protector.protect_file(&abs_path) {
                    Ok(result) => {
                        total_source += result.source_len;
                        let repair_size = fs::metadata(&result.sidecar_path).map_or(0, |m| m.len());
                        total_repair += repair_size;
                        protected += 1;

                        debug!(
                            segment_id = %meta.id().short_uuid_string(),
                            file = %relative_path.display(),
                            source_bytes = result.source_len,
                            repair_bytes = repair_size,
                            "segment component protected"
                        );
                    }
                    Err(e) => {
                        warn!(
                            segment_id = %meta.id().short_uuid_string(),
                            file = %relative_path.display(),
                            error = %e,
                            "failed to protect segment component"
                        );
                    }
                }
            }
        }

        // Clean up orphaned sidecars from merged/GC'd segments.
        let orphans_cleaned = self.cleanup_orphaned_sidecars(&known_files);

        let encode_time = start.elapsed();

        info!(
            segments_protected = protected,
            segments_already_protected = already_protected,
            orphans_cleaned,
            total_source_bytes = total_source,
            total_repair_bytes = total_repair,
            encode_time_ms = encode_time.as_millis(),
            "segment protection pass complete"
        );

        Ok(SegmentProtectionReport {
            segments_protected: protected,
            segments_already_protected: already_protected,
            orphans_cleaned,
            total_source_bytes: total_source,
            total_repair_bytes: total_repair,
            encode_time,
        })
    }

    /// Verify all segment component files and auto-repair any corruption.
    ///
    /// Iterates over all searchable segments in creation order (oldest first
    /// for detection, newest first for repair priority per the bead spec).
    /// Returns a health report summarizing the results.
    #[allow(clippy::too_many_lines)]
    pub fn verify_and_repair(&self) -> SearchResult<SegmentHealthReport> {
        let verify_start = Instant::now();
        let mut files_checked = 0_usize;
        let mut files_intact = 0_usize;
        let mut files_repaired = 0_usize;
        let mut files_unrecoverable = 0_usize;
        let mut files_unprotected = 0_usize;
        let mut repair_time = Duration::ZERO;

        let segment_metas =
            self.index
                .searchable_segment_metas()
                .map_err(|e| SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(e),
                })?;

        for meta in &segment_metas {
            let files = meta.list_files();
            for relative_path in &files {
                if !is_safe_relative_path(relative_path) {
                    warn!(
                        path = %relative_path.display(),
                        "segment file path contains unsafe components, skipping"
                    );
                    continue;
                }
                let abs_path = self.data_dir.join(relative_path);
                if !abs_path.exists() {
                    continue;
                }

                files_checked += 1;
                let check_start = Instant::now();
                match self.protector.verify_and_repair_file(&abs_path) {
                    Ok(health) => match health.status {
                        FileHealth::Intact => {
                            files_intact += 1;
                        }
                        FileHealth::Repaired {
                            bytes_written,
                            repair_time: file_repair_time,
                        } => {
                            files_repaired += 1;
                            repair_time += file_repair_time;
                            info!(
                                segment_id = %meta.id().short_uuid_string(),
                                file = %relative_path.display(),
                                bytes_written,
                                repair_time_ms = file_repair_time.as_millis(),
                                "segment component repaired"
                            );
                        }
                        FileHealth::Unrecoverable { reason } => {
                            files_unrecoverable += 1;
                            repair_time += check_start.elapsed();
                            warn!(
                                segment_id = %meta.id().short_uuid_string(),
                                file = %relative_path.display(),
                                reason,
                                "segment component unrecoverable"
                            );
                        }
                        FileHealth::Unprotected => {
                            files_unprotected += 1;
                            debug!(
                                segment_id = %meta.id().short_uuid_string(),
                                file = %relative_path.display(),
                                "no sidecar, skipping verification"
                            );
                        }
                    },
                    Err(e) => {
                        files_unrecoverable += 1;
                        repair_time += check_start.elapsed();
                        warn!(
                            segment_id = %meta.id().short_uuid_string(),
                            file = %relative_path.display(),
                            error = %e,
                            "verify-and-repair failed"
                        );
                    }
                }
            }
        }

        let verify_time = verify_start.elapsed().saturating_sub(repair_time);

        info!(
            files_checked,
            files_intact,
            files_repaired,
            files_unrecoverable,
            files_unprotected,
            verify_time_ms = verify_time.as_millis(),
            repair_time_ms = repair_time.as_millis(),
            "segment health check complete"
        );

        Ok(SegmentHealthReport {
            files_checked,
            files_intact,
            files_repaired,
            files_unrecoverable,
            files_unprotected,
            verify_time,
            repair_time,
        })
    }

    /// Remove `.fec` sidecar files whose corresponding segment files
    /// no longer exist in the index (post-merge/GC cleanup).
    fn cleanup_orphaned_sidecars(&self, known_files: &HashSet<PathBuf>) -> usize {
        let mut cleaned = 0_usize;

        let entries = match fs::read_dir(&self.data_dir) {
            Ok(entries) => entries,
            Err(e) => {
                warn!(
                    dir = %self.data_dir.display(),
                    error = %e,
                    "cannot read data dir for orphan cleanup"
                );
                return 0;
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();
            // Only look at .fec files.
            if !Self::has_fec_extension(&path) {
                continue;
            }

            // Derive the source file path by stripping the extension.
            let source_path = path.with_extension("");

            if !known_files.contains(&source_path) && !source_path.exists() {
                match fs::remove_file(&path) {
                    Ok(()) => {
                        debug!(
                            sidecar = %path.display(),
                            "removed orphaned sidecar"
                        );
                        cleaned += 1;
                    }
                    Err(e) => {
                        warn!(
                            sidecar = %path.display(),
                            error = %e,
                            "failed to remove orphaned sidecar"
                        );
                    }
                }
            }
        }

        cleaned
    }

    fn has_fec_extension(path: &Path) -> bool {
        path.extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("fec"))
    }
}

/// Validate that a relative path from Tantivy metadata is safe to join.
///
/// Rejects absolute paths and paths containing `..` components, which could
/// escape `data_dir` if Tantivy metadata is corrupted or maliciously crafted.
fn is_safe_relative_path(path: &Path) -> bool {
    use std::path::Component;
    if path.is_absolute() {
        return false;
    }
    for component in path.components() {
        match component {
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return false,
            Component::Normal(_) | Component::CurDir => {}
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use fsqlite_core::raptorq_integration::{CodecDecodeResult, CodecEncodeResult, SymbolCodec};
    use fsqlite_types::cx::Cx;
    use tantivy::Index;
    use tantivy::schema::{STORED, STRING, Schema, TEXT};

    use super::{DurableTantivyIndex, TantivySegmentProtector};
    use crate::config::DurabilityConfig;
    use crate::file_protector::FileProtector;

    #[derive(Debug)]
    struct MockCodec;

    impl SymbolCodec for MockCodec {
        fn encode(
            &self,
            _cx: &Cx,
            source_data: &[u8],
            symbol_size: u32,
            _repair_overhead: f64,
        ) -> fsqlite_error::Result<CodecEncodeResult> {
            let symbol_size_usize = usize::try_from(symbol_size).unwrap_or(1);
            let mut source_symbols = Vec::new();
            let mut repair_symbols = Vec::new();

            let mut esi: u32 = 0;
            for chunk in source_data.chunks(symbol_size_usize) {
                let mut data = chunk.to_vec();
                if data.len() < symbol_size_usize {
                    data.resize(symbol_size_usize, 0);
                }
                source_symbols.push((esi, data.clone()));
                repair_symbols.push((esi + 1_000_000, data));
                esi = esi.saturating_add(1);
            }

            Ok(CodecEncodeResult {
                source_symbols,
                repair_symbols,
                k_source: esi,
            })
        }

        fn decode(
            &self,
            _cx: &Cx,
            symbols: &[(u32, Vec<u8>)],
            k_source: u32,
            _symbol_size: u32,
        ) -> fsqlite_error::Result<CodecDecodeResult> {
            let mut reconstructed = Vec::new();
            for source_esi in 0..k_source {
                let primary = symbols
                    .iter()
                    .find(|(esi, _)| *esi == source_esi)
                    .map(|(_, data)| data.clone());
                let fallback = symbols
                    .iter()
                    .find(|(esi, _)| *esi == source_esi + 1_000_000)
                    .map(|(_, data)| data.clone());

                match primary.or(fallback) {
                    Some(data) => reconstructed.extend_from_slice(&data),
                    None => {
                        return Ok(CodecDecodeResult::Failure {
                            reason:
                                fsqlite_core::raptorq_integration::DecodeFailureReason::InsufficientSymbols,
                            symbols_received: u32::try_from(symbols.len()).unwrap_or(u32::MAX),
                            k_required: k_source,
                        });
                    }
                }
            }

            Ok(CodecDecodeResult::Success {
                data: reconstructed,
                symbols_used: k_source,
                peeled_count: k_source,
                inactivated_count: 0,
            })
        }
    }

    fn test_config() -> DurabilityConfig {
        DurabilityConfig {
            symbol_size: 256,
            repair_overhead: 2.0,
            ..DurabilityConfig::default()
        }
    }

    fn temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "frankensearch-tantivy-{prefix}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn test_schema() -> Schema {
        let mut builder = Schema::builder();
        builder.add_text_field("id", STRING | STORED);
        builder.add_text_field("content", TEXT | STORED);
        builder.build()
    }

    // --- TantivySegmentProtector tests ---

    #[test]
    fn segment_protector_protect_and_verify() {
        let protector =
            TantivySegmentProtector::new(Arc::new(MockCodec), test_config()).expect("protector");

        let dir = temp_dir("seg-protector");
        let file = dir.join("test-segment.idx");
        std::fs::write(&file, vec![42_u8; 500]).expect("write");

        let result = protector.protect_segment(&file).expect("protect");
        assert!(result.sidecar_path.exists());

        let verify = protector.verify_segment(&file).expect("verify");
        assert!(verify.healthy);
    }

    // --- DurableTantivyIndex tests ---

    #[test]
    fn durable_index_protect_and_verify_roundtrip() {
        let dir = temp_dir("durable-roundtrip");
        let schema = test_schema();
        let index = Index::create_in_dir(&dir, schema.clone()).expect("create index");

        let id_field = schema.get_field("id").unwrap();
        let content_field = schema.get_field("content").unwrap();

        // Index some documents.
        let mut writer = index.writer(15_000_000).expect("writer");
        for i in 0..5 {
            let mut doc = tantivy::TantivyDocument::new();
            doc.add_text(id_field, format!("doc-{i}"));
            doc.add_text(content_field, format!("test content number {i}"));
            writer.add_document(doc).expect("add doc");
        }
        writer.commit().expect("commit");

        // Create durable wrapper and protect.
        let durable = DurableTantivyIndex::new(index, dir, Arc::new(MockCodec), test_config())
            .expect("durable index");

        let protection = durable.protect_segments().expect("protect");
        assert!(protection.segments_protected > 0);
        assert_eq!(protection.segments_already_protected, 0);
        assert!(protection.total_source_bytes > 0);
        assert!(protection.total_repair_bytes > 0);

        // Second protect should find everything already protected.
        let protection2 = durable.protect_segments().expect("protect again");
        assert_eq!(protection2.segments_protected, 0);
        assert!(protection2.segments_already_protected > 0);

        // Verify should report all intact.
        let health = durable.verify_and_repair().expect("verify");
        assert!(health.files_checked > 0);
        assert_eq!(
            health.files_intact,
            health.files_checked - health.files_unprotected
        );
        assert_eq!(health.files_repaired, 0);
        assert_eq!(health.files_unrecoverable, 0);
    }

    #[test]
    fn durable_index_detects_corruption() {
        let dir = temp_dir("durable-corrupt");
        let schema = test_schema();
        let index = Index::create_in_dir(&dir, schema.clone()).expect("create index");

        let id_field = schema.get_field("id").unwrap();
        let content_field = schema.get_field("content").unwrap();

        let mut writer = index.writer(15_000_000).expect("writer");
        let mut doc = tantivy::TantivyDocument::new();
        doc.add_text(id_field, "doc-1");
        doc.add_text(content_field, "some content for corruption test");
        writer.add_document(doc).expect("add doc");
        writer.commit().expect("commit");

        let durable =
            DurableTantivyIndex::new(index, dir.clone(), Arc::new(MockCodec), test_config())
                .expect("durable index");

        durable.protect_segments().expect("protect");

        // Corrupt a segment file.
        let segment_metas = durable.index().searchable_segment_metas().expect("metas");
        assert!(!segment_metas.is_empty());

        let files = segment_metas[0].list_files();
        let target = files
            .iter()
            .find(|p| {
                let abs = dir.join(p);
                abs.exists() && FileProtector::sidecar_path(&abs).exists()
            })
            .expect("find protected file");
        let abs_target = dir.join(target);
        let original = std::fs::read(&abs_target).expect("read original");
        let mut corrupted = original;
        if !corrupted.is_empty() {
            corrupted[0] ^= 0xFF;
        }
        std::fs::write(&abs_target, &corrupted).expect("write corrupted");

        // Verify should detect and repair the corruption.
        let health = durable.verify_and_repair().expect("verify and repair");
        assert!(
            health.files_repaired > 0 || health.files_unrecoverable > 0,
            "corruption should be detected"
        );
    }

    #[test]
    fn durable_index_sidecar_decode_failures_are_reported_as_unrecoverable() {
        let dir = temp_dir("durable-sidecar-corrupt");
        let schema = test_schema();
        let index = Index::create_in_dir(&dir, schema.clone()).expect("create index");

        let id_field = schema.get_field("id").expect("id field");
        let content_field = schema.get_field("content").expect("content field");

        let mut writer = index.writer(15_000_000).expect("writer");
        let mut doc = tantivy::TantivyDocument::new();
        doc.add_text(id_field, "doc-1");
        doc.add_text(content_field, "content for sidecar corruption test");
        writer.add_document(doc).expect("add doc");
        writer.commit().expect("commit");

        let durable =
            DurableTantivyIndex::new(index, dir.clone(), Arc::new(MockCodec), test_config())
                .expect("durable index");
        durable.protect_segments().expect("protect");

        let segment_metas = durable.index().searchable_segment_metas().expect("metas");
        let target = segment_metas[0]
            .list_files()
            .into_iter()
            .find(|relative| {
                let abs = dir.join(relative);
                abs.exists() && FileProtector::sidecar_path(&abs).exists()
            })
            .expect("find protected file");
        let abs_target = dir.join(target);
        let sidecar = FileProtector::sidecar_path(&abs_target);
        std::fs::write(&sidecar, b"invalid sidecar payload").expect("corrupt sidecar");

        let health = durable.verify_and_repair().expect("verify");
        assert!(
            health.files_unrecoverable > 0,
            "verify errors should increment files_unrecoverable"
        );
    }

    #[test]
    fn durable_index_orphan_cleanup() {
        let dir = temp_dir("durable-orphan");
        let schema = test_schema();
        let index = Index::create_in_dir(&dir, schema).expect("create index");

        let durable =
            DurableTantivyIndex::new(index, dir.clone(), Arc::new(MockCodec), test_config())
                .expect("durable index");

        // Create a fake orphaned sidecar.
        let orphan = dir.join("nonexistent-segment.idx.fec");
        std::fs::write(&orphan, b"fake sidecar").expect("write orphan");
        assert!(orphan.exists());

        // Protect should clean up the orphan.
        let report = durable.protect_segments().expect("protect");
        assert!(report.orphans_cleaned >= 1);
        assert!(!orphan.exists());
    }

    #[test]
    fn durable_index_metrics_tracking() {
        let dir = temp_dir("durable-metrics");
        let schema = test_schema();
        let index = Index::create_in_dir(&dir, schema.clone()).expect("create index");

        let id_field = schema.get_field("id").unwrap();
        let content_field = schema.get_field("content").unwrap();

        let mut writer = index.writer(15_000_000).expect("writer");
        let mut doc = tantivy::TantivyDocument::new();
        doc.add_text(id_field, "doc-1");
        doc.add_text(content_field, "metrics test content");
        writer.add_document(doc).expect("add doc");
        writer.commit().expect("commit");

        let durable = DurableTantivyIndex::new(index, dir, Arc::new(MockCodec), test_config())
            .expect("durable index");

        durable.protect_segments().expect("protect");

        let snap = durable.metrics_snapshot();
        assert!(snap.encode_ops >= 1);
    }

    #[test]
    fn durable_index_empty_index_no_panic() {
        let dir = temp_dir("durable-empty");
        let schema = test_schema();
        let index = Index::create_in_dir(&dir, schema).expect("create index");

        let durable = DurableTantivyIndex::new(index, dir, Arc::new(MockCodec), test_config())
            .expect("durable index");

        // Should work fine with no segments.
        let protection = durable.protect_segments().expect("protect empty");
        assert_eq!(protection.segments_protected, 0);
        assert_eq!(protection.segments_already_protected, 0);

        let health = durable.verify_and_repair().expect("verify empty");
        assert_eq!(health.files_checked, 0);
    }

    // ── has_fec_extension edge cases ────────────────────────────────────

    #[test]
    fn has_fec_extension_lowercase() {
        assert!(DurableTantivyIndex::has_fec_extension(Path::new(
            "/tmp/file.fec"
        )));
    }

    #[test]
    fn has_fec_extension_uppercase() {
        assert!(DurableTantivyIndex::has_fec_extension(Path::new(
            "/tmp/file.FEC"
        )));
    }

    #[test]
    fn has_fec_extension_mixed_case() {
        assert!(DurableTantivyIndex::has_fec_extension(Path::new(
            "/tmp/file.Fec"
        )));
    }

    #[test]
    fn has_fec_extension_no_extension() {
        assert!(!DurableTantivyIndex::has_fec_extension(Path::new(
            "/tmp/noext"
        )));
    }

    #[test]
    fn has_fec_extension_wrong_extension() {
        assert!(!DurableTantivyIndex::has_fec_extension(Path::new(
            "/tmp/file.idx"
        )));
    }

    #[test]
    fn has_fec_extension_empty_path() {
        assert!(!DurableTantivyIndex::has_fec_extension(Path::new("")));
    }

    #[test]
    fn has_fec_extension_dot_only() {
        assert!(!DurableTantivyIndex::has_fec_extension(Path::new(
            "/tmp/file."
        )));
    }

    // ── Report struct clone ─────────────────────────────────────────────

    #[test]
    fn segment_protection_report_clone() {
        let report = super::SegmentProtectionReport {
            segments_protected: 3,
            segments_already_protected: 2,
            orphans_cleaned: 1,
            total_source_bytes: 1024,
            total_repair_bytes: 256,
            encode_time: Duration::from_millis(50),
        };
        #[allow(clippy::redundant_clone)]
        let cloned = report.clone();
        assert_eq!(cloned.segments_protected, 3);
        assert_eq!(cloned.segments_already_protected, 2);
        assert_eq!(cloned.orphans_cleaned, 1);
        assert_eq!(cloned.total_source_bytes, 1024);
        assert_eq!(cloned.total_repair_bytes, 256);
    }

    #[test]
    fn segment_health_report_clone() {
        let report = super::SegmentHealthReport {
            files_checked: 10,
            files_intact: 8,
            files_repaired: 1,
            files_unrecoverable: 0,
            files_unprotected: 1,
            verify_time: Duration::from_millis(100),
            repair_time: Duration::from_millis(20),
        };
        #[allow(clippy::redundant_clone)]
        let cloned = report.clone();
        assert_eq!(cloned.files_checked, 10);
        assert_eq!(cloned.files_intact, 8);
        assert_eq!(cloned.files_repaired, 1);
        assert_eq!(cloned.files_unrecoverable, 0);
        assert_eq!(cloned.files_unprotected, 1);
    }

    // ── DurableTantivyIndex accessors ───────────────────────────────────

    #[test]
    fn durable_index_data_dir_accessor() {
        let dir = temp_dir("durable-accessor");
        let schema = test_schema();
        let index = Index::create_in_dir(&dir, schema).expect("create index");

        let durable =
            DurableTantivyIndex::new(index, dir.clone(), Arc::new(MockCodec), test_config())
                .expect("durable index");

        assert_eq!(durable.data_dir(), dir.as_path());
    }

    #[test]
    fn durable_index_index_accessor_returns_valid_index() {
        let dir = temp_dir("durable-index-acc");
        let schema = test_schema();
        let index = Index::create_in_dir(&dir, schema).expect("create index");

        let durable = DurableTantivyIndex::new(index, dir, Arc::new(MockCodec), test_config())
            .expect("durable index");

        // Verify the accessor returns an index that can list segments (even if empty).
        let metas = durable.index().searchable_segment_metas().expect("metas");
        assert!(metas.is_empty());
    }

    // ── TantivySegmentProtector batch protect ───────────────────────────

    #[test]
    fn segment_protector_batch_protect() {
        let protector =
            TantivySegmentProtector::new(Arc::new(MockCodec), test_config()).expect("protector");

        let dir = temp_dir("seg-batch");
        let files: Vec<PathBuf> = (0..3)
            .map(|i| {
                let file = dir.join(format!("segment-{i}.idx"));
                std::fs::write(&file, vec![42_u8; 200 + i * 100]).expect("write");
                file
            })
            .collect();

        let paths: Vec<&Path> = files.iter().map(std::path::PathBuf::as_path).collect();
        let results = protector.protect_segments(paths).expect("batch protect");
        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.sidecar_path.exists());
        }
    }

    // ── TantivySegmentProtector repair ──────────────────────────────────

    #[test]
    fn segment_protector_repair_after_corruption() {
        let protector =
            TantivySegmentProtector::new(Arc::new(MockCodec), test_config()).expect("protector");

        let dir = temp_dir("seg-repair");
        let file = dir.join("repair-test.idx");
        let original = vec![42_u8; 500];
        std::fs::write(&file, &original).expect("write");
        protector.protect_segment(&file).expect("protect");

        // Corrupt the file
        let mut corrupted = original;
        corrupted[0] ^= 0xFF;
        std::fs::write(&file, &corrupted).expect("write corrupted");

        let outcome = protector.repair_segment(&file).expect("repair");
        // Either repaired or error reported, but should not panic
        let _ = outcome;
    }

    // ── Verify with no sidecars ─────────────────────────────────────────

    #[test]
    fn verify_without_sidecars_reports_unprotected() {
        let dir = temp_dir("durable-nosidecar");
        let schema = test_schema();
        let index = Index::create_in_dir(&dir, schema.clone()).expect("create index");

        let id_field = schema.get_field("id").unwrap();
        let content_field = schema.get_field("content").unwrap();

        let mut writer = index.writer(15_000_000).expect("writer");
        let mut doc = tantivy::TantivyDocument::new();
        doc.add_text(id_field, "doc-1");
        doc.add_text(content_field, "unprotected content");
        writer.add_document(doc).expect("add doc");
        writer.commit().expect("commit");

        // Do NOT call protect_segments — go straight to verify
        let durable = DurableTantivyIndex::new(index, dir, Arc::new(MockCodec), test_config())
            .expect("durable index");

        let health = durable.verify_and_repair().expect("verify");
        assert!(health.files_checked > 0);
        assert_eq!(health.files_repaired, 0);
        // All files should be unprotected since we never ran protect
        assert!(health.files_unprotected > 0);
    }

    // ── Double verify is idempotent ─────────────────────────────────────

    #[test]
    fn double_verify_is_idempotent() {
        let dir = temp_dir("durable-double-verify");
        let schema = test_schema();
        let index = Index::create_in_dir(&dir, schema.clone()).expect("create index");

        let id_field = schema.get_field("id").unwrap();
        let content_field = schema.get_field("content").unwrap();

        let mut writer = index.writer(15_000_000).expect("writer");
        let mut doc = tantivy::TantivyDocument::new();
        doc.add_text(id_field, "doc-1");
        doc.add_text(content_field, "double verify content");
        writer.add_document(doc).expect("add doc");
        writer.commit().expect("commit");

        let durable = DurableTantivyIndex::new(index, dir, Arc::new(MockCodec), test_config())
            .expect("durable index");

        durable.protect_segments().expect("protect");

        let health1 = durable.verify_and_repair().expect("first verify");
        let health2 = durable.verify_and_repair().expect("second verify");

        assert_eq!(health1.files_checked, health2.files_checked);
        assert_eq!(health1.files_intact, health2.files_intact);
        assert_eq!(health1.files_repaired, health2.files_repaired);
    }

    // ── Metrics snapshot initial state ───────────────────────────────────

    #[test]
    fn metrics_snapshot_initial_is_zero() {
        let dir = temp_dir("durable-metrics-init");
        let schema = test_schema();
        let index = Index::create_in_dir(&dir, schema).expect("create index");

        let durable = DurableTantivyIndex::new(index, dir, Arc::new(MockCodec), test_config())
            .expect("durable index");

        let snap = durable.metrics_snapshot();
        assert_eq!(snap.encode_ops, 0);
        assert_eq!(snap.decode_ops, 0);
    }

    // ── Non-.fec files ignored during orphan cleanup ─────────────────────

    #[test]
    fn orphan_cleanup_ignores_non_fec_files() {
        let dir = temp_dir("durable-orphan-nonfec");
        let schema = test_schema();
        let index = Index::create_in_dir(&dir, schema).expect("create index");

        let durable =
            DurableTantivyIndex::new(index, dir.clone(), Arc::new(MockCodec), test_config())
                .expect("durable index");

        // Create a non-.fec file that should NOT be cleaned up
        let non_fec = dir.join("some-random-file.txt");
        std::fs::write(&non_fec, b"keep me").expect("write");

        let report = durable.protect_segments().expect("protect");
        assert_eq!(report.orphans_cleaned, 0);
        assert!(non_fec.exists(), "non-.fec file should not be removed");
    }
}
