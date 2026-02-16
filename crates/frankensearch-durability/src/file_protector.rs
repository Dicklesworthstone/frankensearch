use std::ffi::OsString;
use std::fs;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use frankensearch_core::{SearchError, SearchResult};
use fsqlite_core::raptorq_integration::{DecodeFailureReason, SymbolCodec};
use memmap2::Mmap;
use serde::Serialize;
use tracing::{debug, info, warn};
use xxhash_rust::xxh3::xxh3_64;

use crate::codec::{CodecFacade, DecodedPayload};
use crate::config::DurabilityConfig;
use crate::metrics::{DurabilityMetrics, DurabilityMetricsSnapshot};
use crate::repair_trailer::{
    RepairSymbol, RepairTrailerHeader, deserialize_repair_trailer, serialize_repair_trailer,
};

/// Result produced after writing a durability sidecar.
#[derive(Debug, Clone)]
pub struct FileProtectionResult {
    pub sidecar_path: PathBuf,
    pub source_len: u64,
    pub source_crc32: u32,
    pub source_xxh3: u64,
    /// Number of source symbols the file was split into.
    pub k_source: u32,
    pub repair_symbol_count: u32,
}

/// Verification status for a payload+sidecar pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileVerifyResult {
    pub healthy: bool,
    pub expected_crc32: u32,
    pub actual_crc32: u32,
    pub expected_xxh3: u64,
    pub expected_len: u64,
    pub actual_len: u64,
}

/// Repair outcome for a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileRepairOutcome {
    NotNeeded,
    Repaired {
        bytes_written: usize,
        symbols_used: u32,
    },
    Unrecoverable {
        reason: DecodeFailureReason,
        symbols_received: u32,
        k_required: u32,
    },
}

/// Health status for a single file after verify-and-repair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileHealth {
    /// File integrity confirmed; no action needed.
    Intact,
    /// Corruption was detected and successfully repaired.
    Repaired {
        /// Number of bytes written during repair.
        bytes_written: usize,
        /// Wall-clock time for the repair operation.
        repair_time: Duration,
    },
    /// Corruption was detected but repair failed.
    Unrecoverable {
        /// Explanation of why repair failed.
        reason: String,
    },
    /// No `.fec` sidecar exists for this file.
    Unprotected,
}

/// Result of a single-file verify-and-repair pipeline.
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Path to the checked file.
    pub path: PathBuf,
    /// Health status after check (and optional repair).
    pub status: FileHealth,
}

/// Report produced after protecting all files in a directory.
#[derive(Debug, Clone)]
pub struct DirectoryProtectionReport {
    /// Number of files newly protected.
    pub files_protected: usize,
    /// Number of files already protected (skipped).
    pub files_already_protected: usize,
    /// Total source bytes across newly protected files.
    pub total_source_bytes: u64,
    /// Total repair sidecar bytes generated.
    pub total_repair_bytes: u64,
    /// Wall-clock time for the protection pass.
    pub elapsed: Duration,
}

/// Report produced after verifying all files in a directory.
#[derive(Debug, Clone)]
pub struct DirectoryHealthReport {
    /// Per-file health check results.
    pub results: Vec<HealthCheckResult>,
    /// Number of intact files.
    pub intact_count: usize,
    /// Number of repaired files.
    pub repaired_count: usize,
    /// Number of unrecoverable files.
    pub unrecoverable_count: usize,
    /// Number of unprotected files (no sidecar).
    pub unprotected_count: usize,
    /// Wall-clock time for the full check.
    pub elapsed: Duration,
}

/// JSONL repair event record, appended to the repair log file.
#[derive(Debug, Serialize)]
struct RepairEvent {
    timestamp: String,
    path: String,
    corrupted: bool,
    repair_succeeded: bool,
    bytes_written: usize,
    source_crc32_expected: u32,
    source_crc32_after: u32,
    repair_time_ms: u64,
}

/// Abstract durability provider with no-op defaults.
///
/// When the `durability` feature is disabled at compile time, consumers can
/// use [`NoopDurability`] which satisfies this trait with zero overhead.
pub trait DurabilityProvider: Send + Sync {
    /// Protect a file by generating a `.fec` sidecar.
    fn protect(&self, path: &Path) -> SearchResult<FileProtectionResult> {
        let _ = path;
        Ok(FileProtectionResult {
            sidecar_path: PathBuf::new(),
            source_len: 0,
            source_crc32: 0,
            source_xxh3: 0,
            k_source: 0,
            repair_symbol_count: 0,
        })
    }

    /// Verify a file's integrity using its sidecar.
    fn verify(&self, path: &Path) -> SearchResult<FileVerifyResult> {
        let _ = path;
        Ok(FileVerifyResult {
            healthy: true,
            expected_crc32: 0,
            actual_crc32: 0,
            expected_xxh3: 0,
            expected_len: 0,
            actual_len: 0,
        })
    }

    /// Attempt to repair a corrupted file.
    fn repair(&self, path: &Path) -> SearchResult<FileRepairOutcome> {
        let _ = path;
        Err(SearchError::DurabilityDisabled)
    }

    /// Verify and optionally repair a single file.
    fn check_health(&self, path: &Path) -> SearchResult<HealthCheckResult> {
        let _ = path;
        Ok(HealthCheckResult {
            path: PathBuf::new(),
            status: FileHealth::Unprotected,
        })
    }

    /// Protect all protectable files in a directory.
    fn protect_directory(&self, dir: &Path) -> SearchResult<DirectoryProtectionReport> {
        let _ = dir;
        Ok(DirectoryProtectionReport {
            files_protected: 0,
            files_already_protected: 0,
            total_source_bytes: 0,
            total_repair_bytes: 0,
            elapsed: Duration::ZERO,
        })
    }

    /// Verify (and auto-repair) all protected files in a directory.
    fn verify_directory(&self, dir: &Path) -> SearchResult<DirectoryHealthReport> {
        let _ = dir;
        Ok(DirectoryHealthReport {
            results: Vec::new(),
            intact_count: 0,
            repaired_count: 0,
            unrecoverable_count: 0,
            unprotected_count: 0,
            elapsed: Duration::ZERO,
        })
    }

    /// Get a metrics snapshot.
    fn metrics_snapshot(&self) -> DurabilityMetricsSnapshot;
}

/// No-op durability provider for when the feature is disabled.
#[derive(Debug, Default)]
pub struct NoopDurability;

impl DurabilityProvider for NoopDurability {
    fn metrics_snapshot(&self) -> DurabilityMetricsSnapshot {
        DurabilityMetricsSnapshot {
            encoded_bytes_total: 0,
            source_symbols_total: 0,
            repair_symbols_total: 0,
            decoded_bytes_total: 0,
            decode_symbols_used_total: 0,
            decode_symbols_received_total: 0,
            decode_k_required_total: 0,
            encode_ops: 0,
            decode_ops: 0,
            decode_failures: 0,
            decode_failures_recoverable: 0,
            decode_failures_unrecoverable: 0,
            encode_latency_us_total: 0,
            decode_latency_us_total: 0,
            repair_attempts: 0,
            repair_successes: 0,
            repair_failures: 0,
        }
    }
}

/// Configuration for the repair pipeline.
#[derive(Debug, Clone)]
pub struct RepairPipelineConfig {
    /// Whether to verify indices on load.
    pub verify_on_open: bool,
    /// Whether to generate `.fec` after index write.
    pub protect_on_write: bool,
    /// Whether to attempt repair when corruption detected.
    pub auto_repair: bool,
    /// Optional directory for JSONL repair event logs.
    pub repair_log_dir: Option<PathBuf>,
    /// Maximum repair log entries before rotation.
    pub max_repair_log_entries: usize,
}

impl Default for RepairPipelineConfig {
    fn default() -> Self {
        Self {
            verify_on_open: true,
            protect_on_write: true,
            auto_repair: true,
            repair_log_dir: None,
            max_repair_log_entries: 1000,
        }
    }
}

/// File-level protect/verify/repair orchestrator.
#[derive(Debug, Clone)]
pub struct FileProtector {
    codec: CodecFacade,
    metrics: Arc<DurabilityMetrics>,
    pipeline_config: RepairPipelineConfig,
}

impl FileProtector {
    pub fn new(codec: Arc<dyn SymbolCodec>, config: DurabilityConfig) -> SearchResult<Self> {
        let metrics = Arc::new(DurabilityMetrics::default());
        Self::new_with_metrics(codec, config, metrics)
    }

    /// Create a `FileProtector` sharing an externally-owned metrics instance.
    pub fn new_with_metrics(
        codec: Arc<dyn SymbolCodec>,
        config: DurabilityConfig,
        metrics: Arc<DurabilityMetrics>,
    ) -> SearchResult<Self> {
        let verify_on_open = config.verify_on_open;
        let codec = CodecFacade::new(codec, config, Arc::clone(&metrics))?;
        let pipeline_config = RepairPipelineConfig {
            verify_on_open,
            ..RepairPipelineConfig::default()
        };
        Ok(Self {
            codec,
            metrics,
            pipeline_config,
        })
    }

    /// Create a `FileProtector` with full pipeline configuration.
    pub fn new_with_pipeline_config(
        codec: Arc<dyn SymbolCodec>,
        config: DurabilityConfig,
        metrics: Arc<DurabilityMetrics>,
        mut pipeline_config: RepairPipelineConfig,
    ) -> SearchResult<Self> {
        pipeline_config.verify_on_open = config.verify_on_open;
        let codec = CodecFacade::new(codec, config, Arc::clone(&metrics))?;
        Ok(Self {
            codec,
            metrics,
            pipeline_config,
        })
    }

    /// Access the pipeline configuration.
    pub fn pipeline_config(&self) -> &RepairPipelineConfig {
        &self.pipeline_config
    }

    pub fn metrics_snapshot(&self) -> DurabilityMetricsSnapshot {
        self.metrics.snapshot()
    }

    pub fn sidecar_path(path: &Path) -> PathBuf {
        let mut sidecar = path.as_os_str().to_os_string();
        sidecar.push(".fec");
        PathBuf::from(sidecar)
    }

    fn backup_path(path: &Path, timestamp: u64) -> PathBuf {
        let mut backup: OsString = path.as_os_str().to_os_string();
        backup.push(".corrupt.");
        backup.push(timestamp.to_string());
        PathBuf::from(backup)
    }

    fn restore_backup(backup_path: &Path, destination: &Path) -> SearchResult<()> {
        if destination.exists()
            && let Err(error) = fs::remove_file(destination)
        {
            warn!(
                destination = %destination.display(),
                error = %error,
                "failed to remove repaired destination before backup restore"
            );
            return Err(error.into());
        }
        if let Err(error) = fs::rename(backup_path, destination) {
            warn!(
                backup = %backup_path.display(),
                destination = %destination.display(),
                error = %error,
                "failed to restore backup"
            );
            return Err(error.into());
        }
        Ok(())
    }

    fn has_fec_extension(path: &Path) -> bool {
        path.extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("fec"))
    }

    fn should_skip_directory_entry(path: &Path) -> bool {
        if Self::has_fec_extension(path) {
            return true;
        }
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with('.') || name.contains(".corrupt."))
    }

    #[allow(unsafe_code)] // Mmap::map requires unsafe for memory-mapped I/O.
    pub fn protect_file(&self, path: &Path) -> SearchResult<FileProtectionResult> {
        let file = fs::File::open(path)?;
        let len = file.metadata()?.len();
        let (encoded, source_xxh3) = if len == 0 {
            (self.codec.encode(&[])?, xxh3_64(&[]))
        } else {
            // SAFETY: We assume the file is not modified concurrently (advisory).
            // This is a standard assumption for CLI tools operating on files.
            let mmap = unsafe { Mmap::map(&file).map_err(SearchError::Io)? };
            (self.codec.encode(&mmap)?, xxh3_64(&mmap))
        };

        let repair_symbol_count = u32::try_from(encoded.repair_symbols.len()).map_err(|_| {
            SearchError::InvalidConfig {
                field: "repair_symbol_count".to_owned(),
                value: encoded.repair_symbols.len().to_string(),
                reason: "repair symbol count exceeds u32".to_owned(),
            }
        })?;
        let header = RepairTrailerHeader {
            symbol_size: encoded.symbol_size,
            k_source: encoded.k_source,
            source_len: encoded.source_len,
            source_crc32: encoded.source_crc32,
            source_xxh3,
            repair_symbol_count,
        };

        let repair_symbols: Vec<RepairSymbol> = encoded
            .repair_symbols
            .into_iter()
            .map(|(esi, data)| RepairSymbol { esi, data })
            .collect();
        let trailer = serialize_repair_trailer(&header, &repair_symbols)?;

        let sidecar_path = Self::sidecar_path(path);
        // Write through a File handle so we can fsync before returning.
        // A bare fs::write() is not durable: the kernel may buffer the data
        // and a power loss before writeback would silently lose the sidecar.
        {
            let mut file = fs::File::create(&sidecar_path)?;
            file.write_all(&trailer)?;
            file.sync_all()?;
        }

        info!(
            path = %path.display(),
            sidecar = %sidecar_path.display(),
            repair_symbols = repair_symbol_count,
            "durability sidecar written"
        );

        Ok(FileProtectionResult {
            sidecar_path,
            source_len: header.source_len,
            source_crc32: header.source_crc32,
            source_xxh3,
            k_source: header.k_source,
            repair_symbol_count: header.repair_symbol_count,
        })
    }

    #[allow(unsafe_code)] // Mmap::map requires unsafe for memory-mapped I/O.
    pub fn verify_file(&self, path: &Path, sidecar_path: &Path) -> SearchResult<FileVerifyResult> {
        let file = fs::File::open(path)?;
        let len = file.metadata()?.len();
        let (actual_crc32, actual_len) = if len == 0 {
            (crc32fast::hash(&[]), 0)
        } else {
            // SAFETY: mmap is read-only.
            let mmap = unsafe { Mmap::map(&file).map_err(SearchError::Io)? };
            (crc32fast::hash(&mmap), saturating_u64(mmap.len()))
        };

        let trailer_bytes = fs::read(sidecar_path)?;
        let (header, _) = deserialize_repair_trailer(&trailer_bytes)?;

        let healthy = actual_crc32 == header.source_crc32 && actual_len == header.source_len;

        Ok(FileVerifyResult {
            healthy,
            expected_crc32: header.source_crc32,
            actual_crc32,
            expected_xxh3: header.source_xxh3,
            expected_len: header.source_len,
            actual_len,
        })
    }

    #[allow(clippy::too_many_lines)]
    #[allow(unsafe_code)] // Mmap::map requires unsafe for memory-mapped I/O.
    pub fn repair_file(&self, path: &Path, sidecar_path: &Path) -> SearchResult<FileRepairOutcome> {
        self.metrics.record_repair_attempt();

        let source_file = match fs::File::open(path) {
            Ok(f) => Some(f),
            Err(e) if e.kind() == ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        // Determine if we have a source file to verify/read
        if let Some(ref file) = source_file {
            // Verify first - using mmap
            let len = file.metadata()?.len();
            let healthy = if len == 0 {
                let trailer_bytes = fs::read(sidecar_path)?;
                let (header, _) = deserialize_repair_trailer(&trailer_bytes)?;
                header.source_crc32 == crc32fast::hash(&[]) && header.source_len == 0
            } else {
                let mmap = unsafe { Mmap::map(file).map_err(SearchError::Io)? };
                let trailer_bytes = fs::read(sidecar_path)?;
                let (header, _) = deserialize_repair_trailer(&trailer_bytes)?;
                let actual_crc32 = crc32fast::hash(&mmap);
                actual_crc32 == header.source_crc32 && len == header.source_len
            };

            if healthy {
                return Ok(FileRepairOutcome::NotNeeded);
            }
        }

        let trailer_bytes = fs::read(sidecar_path)?;
        let (header, trailer_symbols) = deserialize_repair_trailer(&trailer_bytes)?;

        if header.source_len == 0 {
            let empty_crc32 = crc32fast::hash(&[]);
            if header.source_crc32 != empty_crc32 {
                self.metrics.record_repair_failure();
                return Err(SearchError::IndexCorrupted {
                    path: sidecar_path.to_path_buf(),
                    detail: "sidecar metadata is inconsistent for empty source payload".to_owned(),
                });
            }

            // Ensure no source handle is kept while rewriting the destination file.
            drop(source_file);
            write_durable(path, &[])?;
            self.metrics.record_repair_success();
            info!(
                path = %path.display(),
                "durability repair completed (empty payload sidecar)"
            );
            return Ok(FileRepairOutcome::Repaired {
                bytes_written: 0,
                symbols_used: 0,
            });
        }

        let repair_symbols: Vec<(u32, Vec<u8>)> = trailer_symbols
            .into_iter()
            .map(|symbol| (symbol.esi, symbol.data))
            .collect();

        // Load source symbols from file (via mmap) if available
        let mut symbols = if let Some(ref file) = source_file {
            let len = file.metadata()?.len();
            if len > 0 {
                if len == header.source_len {
                    // Bit-rot case: length matches but verification failed.
                    // Feeding corrupted source symbols to an erasure codec (which expects
                    // valid symbols or erasures) usually prevents recovery or produces
                    // garbage. We skip loading source symbols to avoid OOM on large files
                    // and rely entirely on repair symbols.
                    warn!(
                        path = %path.display(),
                        len,
                        "source length matches header but CRC failed; skipping source symbols"
                    );
                    Vec::new()
                } else {
                    // Truncation case: we need the valid prefix.
                    let mmap = unsafe { Mmap::map(file).map_err(SearchError::Io)? };
                    source_symbols_from_bytes(&mmap, header.symbol_size, header.k_source)?
                }
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        symbols.extend(repair_symbols.iter().cloned());
        // Avoid holding a read handle on the source path while writing repaired content.
        drop(source_file);

        match self
            .codec
            .decode_for_symbol_size(&symbols, header.k_source, header.symbol_size)?
        {
            DecodedPayload::Success {
                data, symbols_used, ..
            } => {
                let data = normalize_recovered_data(data, &header)?;
                let recovered_crc32 = crc32fast::hash(&data);

                if recovered_crc32 != header.source_crc32 {
                    self.metrics.record_repair_failure();
                    warn!(
                        path = %path.display(),
                        expected_crc32 = header.source_crc32,
                        recovered_crc32,
                        "decoded payload failed crc verification"
                    );
                    return Ok(FileRepairOutcome::Unrecoverable {
                        reason: DecodeFailureReason::SymbolSizeMismatch,
                        symbols_received: u32::try_from(symbols.len()).unwrap_or(u32::MAX),
                        k_required: header.k_source,
                    });
                }

                write_durable(path, &data)?;
                self.metrics.record_repair_success();
                info!(
                    path = %path.display(),
                    bytes_written = data.len(),
                    symbols_used,
                    "durability repair completed"
                );
                Ok(FileRepairOutcome::Repaired {
                    bytes_written: data.len(),
                    symbols_used,
                })
            }
            DecodedPayload::Failure {
                reason,
                symbols_received,
                k_required,
                ..
            } => {
                self.metrics.record_repair_failure();
                warn!(
                    path = %path.display(),
                    ?reason,
                    symbols_received,
                    k_required,
                    "durability repair could not recover file"
                );
                Ok(FileRepairOutcome::Unrecoverable {
                    reason,
                    symbols_received,
                    k_required,
                })
            }
        }
    }

    /// Single-file verify-and-repair pipeline with backup-before-repair.
    ///
    /// Steps:
    /// 1. Check for `.fec` sidecar — if missing, return `Unprotected`.
    /// 2. Verify file integrity via CRC.
    /// 3. If corrupted and `auto_repair` is enabled, back up the corrupted file,
    ///    attempt repair, verify the result, and clean up or restore the backup.
    /// 4. Log the repair event if `repair_log_dir` is configured.
    pub fn verify_and_repair_file(&self, path: &Path) -> SearchResult<HealthCheckResult> {
        let sidecar = Self::sidecar_path(path);
        if !sidecar.exists() {
            return Ok(HealthCheckResult {
                path: path.to_path_buf(),
                status: FileHealth::Unprotected,
            });
        }

        match self.verify_file(path, &sidecar) {
            Ok(verify) if verify.healthy => {
                return Ok(HealthCheckResult {
                    path: path.to_path_buf(),
                    status: FileHealth::Intact,
                });
            }
            Ok(verify) => {
                debug!(
                    path = %path.display(),
                    expected_crc32 = verify.expected_crc32,
                    actual_crc32 = verify.actual_crc32,
                    "corruption detected"
                );
            }
            Err(SearchError::Io(ref e)) if e.kind() == ErrorKind::NotFound => {
                // Source file missing entirely — still try repair if auto_repair enabled.
                debug!(
                    path = %path.display(),
                    "source file missing, will attempt repair from sidecar"
                );
            }
            Err(e) => return Err(e),
        }

        if !self.pipeline_config.auto_repair {
            return Ok(HealthCheckResult {
                path: path.to_path_buf(),
                status: FileHealth::Unrecoverable {
                    reason: "auto_repair is disabled".to_owned(),
                },
            });
        }

        // Backup-before-repair: rename corrupted file to .corrupt.{timestamp}
        let timestamp = unix_timestamp_secs();
        let backup_path = Self::backup_path(path, timestamp);
        let had_source = path.exists();
        if had_source {
            fs::rename(path, &backup_path).map_err(|e| {
                warn!(
                    path = %path.display(),
                    backup = %backup_path.display(),
                    error = %e,
                    "failed to create backup before repair"
                );
                e
            })?;
        }

        let repair_start = Instant::now();
        let outcome = self.repair_file(path, &sidecar);
        let repair_time = repair_start.elapsed();

        self.finalize_repair(
            path,
            &backup_path,
            &sidecar,
            had_source,
            outcome,
            repair_time,
        )
    }

    /// Process the repair outcome, verify the result, restore backups on
    /// failure, and log the event.
    fn finalize_repair(
        &self,
        path: &Path,
        backup_path: &Path,
        sidecar: &Path,
        had_source: bool,
        outcome: SearchResult<FileRepairOutcome>,
        repair_time: Duration,
    ) -> SearchResult<HealthCheckResult> {
        match outcome {
            Ok(FileRepairOutcome::Repaired { bytes_written, .. }) => {
                // Verify the repaired file passes integrity check.
                let post_verify = self.verify_file(path, sidecar);
                match post_verify {
                    Ok(v) if v.healthy => {
                        // Success — clean up backup.
                        if had_source && let Err(error) = fs::remove_file(backup_path) {
                            warn!(
                                backup = %backup_path.display(),
                                error = %error,
                                "failed to remove backup after successful repair"
                            );
                        }
                        self.log_repair_event(
                            path,
                            true,
                            bytes_written,
                            v.expected_crc32,
                            v.actual_crc32,
                            repair_time,
                        );
                        Ok(HealthCheckResult {
                            path: path.to_path_buf(),
                            status: FileHealth::Repaired {
                                bytes_written,
                                repair_time,
                            },
                        })
                    }
                    _ => {
                        // Repaired file failed verification — restore backup.
                        warn!(
                            path = %path.display(),
                            "repaired file failed post-repair verification, restoring backup"
                        );
                        if had_source {
                            Self::restore_backup(backup_path, path)?;
                        }
                        self.log_repair_event(path, false, 0, 0, 0, repair_time);
                        Ok(HealthCheckResult {
                            path: path.to_path_buf(),
                            status: FileHealth::Unrecoverable {
                                reason: "repaired file failed post-repair verification".to_owned(),
                            },
                        })
                    }
                }
            }
            Ok(FileRepairOutcome::NotNeeded) => {
                // Race condition: file was fine when repair ran.
                if had_source {
                    Self::restore_backup(backup_path, path)?;
                }
                Ok(HealthCheckResult {
                    path: path.to_path_buf(),
                    status: FileHealth::Intact,
                })
            }
            Ok(FileRepairOutcome::Unrecoverable { reason, .. }) => {
                // Restore backup — repair failed.
                if had_source {
                    Self::restore_backup(backup_path, path)?;
                }
                self.log_repair_event(path, false, 0, 0, 0, repair_time);
                Ok(HealthCheckResult {
                    path: path.to_path_buf(),
                    status: FileHealth::Unrecoverable {
                        reason: format!("{reason:?}"),
                    },
                })
            }
            Err(e) => {
                // Restore backup on error.
                if had_source {
                    Self::restore_backup(backup_path, path)?;
                }
                Err(e)
            }
        }
    }

    /// Protect all protectable files in a directory.
    ///
    /// Scans for files without a corresponding `.fec` sidecar and generates
    /// protection for them. Skips `.fec` files themselves and hidden files.
    pub fn protect_directory(&self, dir: &Path) -> SearchResult<DirectoryProtectionReport> {
        let start = Instant::now();
        let mut files_protected = 0_usize;
        let mut files_already_protected = 0_usize;
        let mut total_source_bytes = 0_u64;
        let mut total_repair_bytes = 0_u64;

        let entries = fs::read_dir(dir)?;
        for entry in entries.flatten() {
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(error) => {
                    warn!(
                        dir = %dir.display(),
                        error = %error,
                        "failed to read directory entry type during protection pass"
                    );
                    continue;
                }
            };
            if !file_type.is_file() {
                continue;
            }
            let path = entry.path();
            if Self::should_skip_directory_entry(&path) {
                continue;
            }

            let sidecar = Self::sidecar_path(&path);
            if sidecar.exists() {
                files_already_protected += 1;
                continue;
            }

            match self.protect_file(&path) {
                Ok(result) => {
                    total_source_bytes += result.source_len;
                    let repair_size = fs::metadata(&result.sidecar_path).map_or(0, |m| m.len());
                    total_repair_bytes += repair_size;
                    files_protected += 1;
                }
                Err(e) => {
                    warn!(
                        path = %path.display(),
                        error = %e,
                        "failed to protect file in directory scan"
                    );
                }
            }
        }

        let elapsed = start.elapsed();
        info!(
            dir = %dir.display(),
            files_protected,
            files_already_protected,
            total_source_bytes,
            total_repair_bytes,
            elapsed_ms = elapsed.as_millis(),
            "directory protection pass complete"
        );

        Ok(DirectoryProtectionReport {
            files_protected,
            files_already_protected,
            total_source_bytes,
            total_repair_bytes,
            elapsed,
        })
    }

    /// Verify (and auto-repair) all protected files in a directory.
    pub fn verify_directory(&self, dir: &Path) -> SearchResult<DirectoryHealthReport> {
        let start = Instant::now();
        let mut results = Vec::new();
        let mut intact_count = 0_usize;
        let mut repaired_count = 0_usize;
        let mut unrecoverable_count = 0_usize;
        let mut unprotected_count = 0_usize;

        let entries = fs::read_dir(dir)?;
        for entry in entries.flatten() {
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(error) => {
                    warn!(
                        dir = %dir.display(),
                        error = %error,
                        "failed to read directory entry type during verification pass"
                    );
                    continue;
                }
            };
            if !file_type.is_file() {
                continue;
            }
            let path = entry.path();
            if Self::should_skip_directory_entry(&path) {
                continue;
            }

            let result = self.verify_and_repair_file(&path)?;
            match &result.status {
                FileHealth::Intact => intact_count += 1,
                FileHealth::Repaired { .. } => repaired_count += 1,
                FileHealth::Unrecoverable { .. } => unrecoverable_count += 1,
                FileHealth::Unprotected => unprotected_count += 1,
            }
            results.push(result);
        }

        let elapsed = start.elapsed();
        info!(
            dir = %dir.display(),
            intact = intact_count,
            repaired = repaired_count,
            unrecoverable = unrecoverable_count,
            unprotected = unprotected_count,
            elapsed_ms = elapsed.as_millis(),
            "directory health check complete"
        );

        Ok(DirectoryHealthReport {
            results,
            intact_count,
            repaired_count,
            unrecoverable_count,
            unprotected_count,
            elapsed,
        })
    }

    /// Protect all existing unprotected files in a directory.
    ///
    /// This handles the migration case where durability is enabled on a
    /// system with pre-existing unprotected indices. Without this, all
    /// existing indices emit warnings on every open.
    pub fn protect_all_existing(&self, dir: &Path) -> SearchResult<DirectoryProtectionReport> {
        self.protect_directory(dir)
    }

    /// Log a repair event to the configured JSONL log directory.
    fn log_repair_event(
        &self,
        path: &Path,
        repair_succeeded: bool,
        bytes_written: usize,
        expected_crc32: u32,
        actual_crc32: u32,
        repair_time: Duration,
    ) {
        let Some(log_dir) = &self.pipeline_config.repair_log_dir else {
            return;
        };
        if let Err(e) = fs::create_dir_all(log_dir) {
            warn!(
                log_dir = %log_dir.display(),
                error = %e,
                "failed to create repair log directory"
            );
            return;
        }

        let event = RepairEvent {
            timestamp: iso8601_now(),
            path: path.display().to_string(),
            corrupted: true,
            repair_succeeded,
            bytes_written,
            source_crc32_expected: expected_crc32,
            source_crc32_after: actual_crc32,
            repair_time_ms: u64::try_from(repair_time.as_millis()).unwrap_or(u64::MAX),
        };

        let log_path = log_dir.join("repair-events.jsonl");

        let json = match serde_json::to_string(&event) {
            Ok(json) => json,
            Err(e) => {
                warn!(
                    path = %path.display(),
                    error = %e,
                    "failed to serialize repair event to JSON"
                );
                return;
            }
        };
        {
            // Rotate if needed.
            if matches!(
                should_rotate(&log_path, self.pipeline_config.max_repair_log_entries),
                Ok(true)
            ) {
                let rotated = log_dir.join("repair-events.1.jsonl");
                let _ = fs::rename(&log_path, &rotated);
            }

            let line = format!("{json}\n");
            if let Err(e) = append_to_file(&log_path, line.as_bytes()) {
                warn!(
                    log_path = %log_path.display(),
                    error = %e,
                    "failed to write repair event log"
                );
            }
        }
    }
}

impl DurabilityProvider for FileProtector {
    fn protect(&self, path: &Path) -> SearchResult<FileProtectionResult> {
        self.protect_file(path)
    }

    fn verify(&self, path: &Path) -> SearchResult<FileVerifyResult> {
        let sidecar = Self::sidecar_path(path);
        self.verify_file(path, &sidecar)
    }

    fn repair(&self, path: &Path) -> SearchResult<FileRepairOutcome> {
        let sidecar = Self::sidecar_path(path);
        self.repair_file(path, &sidecar)
    }

    fn check_health(&self, path: &Path) -> SearchResult<HealthCheckResult> {
        self.verify_and_repair_file(path)
    }

    fn protect_directory(&self, dir: &Path) -> SearchResult<DirectoryProtectionReport> {
        Self::protect_directory(self, dir)
    }

    fn verify_directory(&self, dir: &Path) -> SearchResult<DirectoryHealthReport> {
        Self::verify_directory(self, dir)
    }

    fn metrics_snapshot(&self) -> DurabilityMetricsSnapshot {
        self.metrics.snapshot()
    }
}

fn normalize_recovered_data(
    mut data: Vec<u8>,
    header: &RepairTrailerHeader,
) -> SearchResult<Vec<u8>> {
    let expected_len =
        usize::try_from(header.source_len).map_err(|_| SearchError::InvalidConfig {
            field: "source_len".to_owned(),
            value: header.source_len.to_string(),
            reason: "cannot convert source_len to usize".to_owned(),
        })?;
    if data.len() > expected_len {
        data.truncate(expected_len);
    }
    Ok(data)
}

fn source_symbols_from_bytes(
    bytes: &[u8],
    symbol_size: u32,
    k_source: u32,
) -> SearchResult<Vec<(u32, Vec<u8>)>> {
    if symbol_size == 0 {
        return Err(SearchError::InvalidConfig {
            field: "symbol_size".to_owned(),
            value: "0".to_owned(),
            reason: "must be greater than zero".to_owned(),
        });
    }

    let symbol_size_usize =
        usize::try_from(symbol_size).map_err(|_| SearchError::InvalidConfig {
            field: "symbol_size".to_owned(),
            value: symbol_size.to_string(),
            reason: "cannot convert symbol_size to usize".to_owned(),
        })?;

    let mut out = Vec::new();
    let max_symbols = bytes.len().div_ceil(symbol_size_usize);
    let max_symbols_u32 = u32::try_from(max_symbols).unwrap_or(u32::MAX);
    for esi in 0..k_source.min(max_symbols_u32) {
        let esi_usize = usize::try_from(esi).map_err(|_| SearchError::InvalidConfig {
            field: "esi".to_owned(),
            value: esi.to_string(),
            reason: "cannot convert symbol index to usize".to_owned(),
        })?;
        let start =
            esi_usize
                .checked_mul(symbol_size_usize)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "start_offset".to_owned(),
                    value: format!("{esi_usize}*{symbol_size_usize}"),
                    reason: "source symbol offset overflow".to_owned(),
                })?;
        if start >= bytes.len() {
            continue;
        }

        let end = start.saturating_add(symbol_size_usize).min(bytes.len());
        let mut symbol = bytes[start..end].to_vec();
        if symbol.len() < symbol_size_usize {
            symbol.resize(symbol_size_usize, 0);
        }
        out.push((esi, symbol));
    }

    Ok(out)
}

fn saturating_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn iso8601_now() -> String {
    format_iso8601_from_unix(unix_timestamp_secs())
}

fn format_iso8601_from_unix(secs: u64) -> String {
    let days = secs / 86_400;
    let remaining = secs % 86_400;
    let hours = remaining / 3_600;
    let minutes = (remaining % 3_600) / 60;
    let seconds = remaining % 60;
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    let era_days = days + 719_468;
    let era = era_days / 146_097;
    let day_of_era = era_days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let normalized_year = if month <= 2 { year + 1 } else { year };
    (normalized_year, month, day)
}

fn should_rotate(log_path: &Path, max_entries: usize) -> std::io::Result<bool> {
    if !log_path.exists() {
        return Ok(false);
    }
    let contents = fs::read_to_string(log_path)?;
    let line_count = contents.lines().count();
    Ok(line_count >= max_entries)
}

fn append_to_file(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(data)?;
    Ok(())
}

/// Write `data` to `path` and fsync before returning so the content is
/// durable even under sudden power loss.  Used for repair outputs and
/// other writes where silent data loss would be unacceptable.
fn write_durable(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let mut file = fs::File::create(path)?;
    file.write_all(data)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use fsqlite_core::raptorq_integration::{CodecDecodeResult, CodecEncodeResult, SymbolCodec};

    use super::{
        DurabilityProvider, FileHealth, FileProtector, FileRepairOutcome, NoopDurability,
        RepairPipelineConfig,
    };
    use crate::config::DurabilityConfig;

    #[derive(Debug)]
    struct MockRepairCodec;

    impl SymbolCodec for MockRepairCodec {
        fn encode(
            &self,
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
                            reason: fsqlite_core::raptorq_integration::DecodeFailureReason::InsufficientSymbols,
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

    #[test]
    fn protect_verify_and_repair_file_roundtrip() {
        let config = DurabilityConfig {
            symbol_size: 256,
            // Overhead must be >= 100% so repair symbols cover all source symbols
            // when the entire source file is lost.
            repair_overhead: 2.0,
            ..DurabilityConfig::default()
        };
        let protector = FileProtector::new(Arc::new(MockRepairCodec), config).expect("protector");

        let path = temp_path("durability-roundtrip");
        let payload = vec![42_u8; 700];
        std::fs::write(&path, &payload).expect("write payload");

        let protected = protector.protect_file(&path).expect("protect");
        let verify = protector
            .verify_file(&path, &protected.sidecar_path)
            .expect("verify");
        assert!(verify.healthy);

        // Simulate catastrophic data loss; repair should restore from sidecar symbols.
        std::fs::write(&path, []).expect("wipe file");
        let repaired = protector
            .repair_file(&path, &protected.sidecar_path)
            .expect("repair");
        assert!(matches!(repaired, FileRepairOutcome::Repaired { .. }));

        let restored = std::fs::read(&path).expect("read restored");
        assert_eq!(restored, payload);

        let snapshot = protector.metrics_snapshot();
        assert_eq!(snapshot.repair_attempts, 1);
        assert_eq!(snapshot.repair_successes, 1);
    }

    #[test]
    fn repair_restores_deleted_file_from_sidecar() {
        let config = DurabilityConfig {
            symbol_size: 256,
            repair_overhead: 2.0,
            ..DurabilityConfig::default()
        };
        let protector = FileProtector::new(Arc::new(MockRepairCodec), config).expect("protector");

        let path = temp_path("durability-missing-file");
        let payload = b"recover-me-from-sidecar".to_vec();
        std::fs::write(&path, &payload).expect("write payload");
        let protected = protector.protect_file(&path).expect("protect");

        std::fs::remove_file(&path).expect("remove payload file");
        let repaired = protector
            .repair_file(&path, &protected.sidecar_path)
            .expect("repair missing file");
        assert!(matches!(repaired, FileRepairOutcome::Repaired { .. }));

        let restored = std::fs::read(&path).expect("read restored file");
        assert_eq!(restored, payload);
    }

    fn temp_path(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "frankensearch-durability-{prefix}-{}-{nanos}.bin",
            std::process::id()
        ))
    }

    fn temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "frankensearch-durability-dir-{prefix}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn test_config() -> DurabilityConfig {
        DurabilityConfig {
            symbol_size: 256,
            repair_overhead: 2.0,
            ..DurabilityConfig::default()
        }
    }

    // --- DurabilityProvider trait tests ---

    #[test]
    fn noop_durability_returns_defaults() {
        let noop = NoopDurability;
        let result = noop.protect(std::path::Path::new("/nonexistent")).unwrap();
        assert_eq!(result.source_len, 0);

        let verify = noop.verify(std::path::Path::new("/nonexistent")).unwrap();
        assert!(verify.healthy);

        let repair = noop.repair(std::path::Path::new("/nonexistent"));
        assert!(repair.is_err());

        let snap = noop.metrics_snapshot();
        assert_eq!(snap.encode_ops, 0);
    }

    // --- HealthCheckResult / verify_and_repair_file tests ---

    #[test]
    fn verify_and_repair_intact_file() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("health-intact");
        std::fs::write(&path, vec![42_u8; 500]).expect("write");
        protector.protect_file(&path).expect("protect");

        let result = protector.verify_and_repair_file(&path).expect("check");
        assert!(matches!(result.status, FileHealth::Intact));
    }

    #[test]
    fn verify_and_repair_unprotected_file() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("health-unprotected");
        std::fs::write(&path, vec![42_u8; 500]).expect("write");

        let result = protector.verify_and_repair_file(&path).expect("check");
        assert!(matches!(result.status, FileHealth::Unprotected));
    }

    #[test]
    fn verify_and_repair_corrupted_file_with_backup() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("health-corrupt");
        let payload = vec![42_u8; 500];
        std::fs::write(&path, &payload).expect("write");
        protector.protect_file(&path).expect("protect");

        // Corrupt the file.
        let mut corrupted = payload.clone();
        corrupted[0] ^= 0xFF;
        std::fs::write(&path, &corrupted).expect("corrupt");

        let result = protector.verify_and_repair_file(&path).expect("check");
        assert!(
            matches!(result.status, FileHealth::Repaired { .. }),
            "expected Repaired, got {:?}",
            result.status
        );

        // Verify the file was restored.
        let restored = std::fs::read(&path).expect("read");
        assert_eq!(restored, payload);

        // Verify no backup file remains (successful repair cleans up).
        let dir = path.parent().unwrap();
        let backup_exists = std::fs::read_dir(dir).unwrap().flatten().any(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.contains(".corrupt."))
                && e.path()
                    .to_str()
                    .is_some_and(|p| p.contains("health-corrupt"))
        });
        assert!(
            !backup_exists,
            "backup should be cleaned up after successful repair"
        );
    }

    #[test]
    fn verify_and_repair_auto_repair_disabled() {
        let metrics = Arc::new(crate::metrics::DurabilityMetrics::default());
        let pipeline_config = RepairPipelineConfig {
            auto_repair: false,
            ..RepairPipelineConfig::default()
        };
        let protector = FileProtector::new_with_pipeline_config(
            Arc::new(MockRepairCodec),
            test_config(),
            metrics,
            pipeline_config,
        )
        .expect("protector");

        let path = temp_path("health-no-repair");
        let payload = vec![42_u8; 500];
        std::fs::write(&path, &payload).expect("write");
        protector.protect_file(&path).expect("protect");

        // Corrupt the file.
        let mut corrupted = payload;
        corrupted[0] ^= 0xFF;
        std::fs::write(&path, &corrupted).expect("corrupt");

        let result = protector.verify_and_repair_file(&path).expect("check");
        assert!(
            matches!(result.status, FileHealth::Unrecoverable { .. }),
            "expected Unrecoverable when auto_repair disabled, got {:?}",
            result.status
        );
    }

    #[test]
    fn verify_on_open_is_propagated_from_durability_config() {
        let metrics = Arc::new(crate::metrics::DurabilityMetrics::default());
        let config = DurabilityConfig {
            verify_on_open: false,
            ..test_config()
        };
        let protector = FileProtector::new_with_metrics(Arc::new(MockRepairCodec), config, metrics)
            .expect("protector");
        assert!(!protector.pipeline_config().verify_on_open);
    }

    #[test]
    fn restore_backup_replaces_existing_destination_file() {
        let path = temp_path("restore-backup-destination");
        let backup_path = FileProtector::backup_path(&path, super::unix_timestamp_secs());
        let original = vec![1_u8, 2, 3, 4];
        let replacement = vec![9_u8, 9, 9, 9];

        std::fs::write(&backup_path, &original).expect("write backup");
        std::fs::write(&path, &replacement).expect("write destination");
        FileProtector::restore_backup(&backup_path, &path).expect("restore backup");

        let restored = std::fs::read(&path).expect("read restored");
        assert_eq!(restored, original);
        assert!(
            !backup_path.exists(),
            "backup should be moved back into place"
        );
    }

    // --- Directory-level operation tests ---

    #[test]
    fn protect_directory_generates_sidecars() {
        let dir = temp_dir("protect-dir");
        std::fs::write(dir.join("file1.dat"), vec![1_u8; 300]).expect("write");
        std::fs::write(dir.join("file2.dat"), vec![2_u8; 400]).expect("write");

        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");
        let report = protector.protect_directory(&dir).expect("protect dir");

        assert_eq!(report.files_protected, 2);
        assert_eq!(report.files_already_protected, 0);
        assert!(report.total_source_bytes > 0);
        assert!(report.total_repair_bytes > 0);

        // Second pass should skip.
        let report2 = protector
            .protect_directory(&dir)
            .expect("protect dir again");
        assert_eq!(report2.files_protected, 0);
        assert_eq!(report2.files_already_protected, 2);
    }

    #[test]
    fn verify_directory_detects_corruption() {
        let dir = temp_dir("verify-dir");
        let payload = vec![42_u8; 500];
        std::fs::write(dir.join("good.dat"), &payload).expect("write");
        std::fs::write(dir.join("bad.dat"), &payload).expect("write");

        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");
        protector.protect_directory(&dir).expect("protect dir");

        // Corrupt one file.
        let mut corrupted = payload;
        corrupted[0] ^= 0xFF;
        std::fs::write(dir.join("bad.dat"), &corrupted).expect("corrupt");

        let report = protector.verify_directory(&dir).expect("verify dir");
        assert_eq!(report.intact_count, 1);
        assert!(
            report.repaired_count >= 1,
            "expected at least 1 repaired, got {}",
            report.repaired_count
        );
        assert_eq!(report.unrecoverable_count, 0);
    }

    // --- Repair event logging tests ---

    #[test]
    fn repair_event_is_logged_to_jsonl() {
        let log_dir = temp_dir("repair-log");
        let metrics = Arc::new(crate::metrics::DurabilityMetrics::default());
        let pipeline_config = RepairPipelineConfig {
            repair_log_dir: Some(log_dir.clone()),
            ..RepairPipelineConfig::default()
        };
        let protector = FileProtector::new_with_pipeline_config(
            Arc::new(MockRepairCodec),
            test_config(),
            metrics,
            pipeline_config,
        )
        .expect("protector");

        let path = temp_path("repair-logged");
        let payload = vec![42_u8; 500];
        std::fs::write(&path, &payload).expect("write");
        protector.protect_file(&path).expect("protect");

        // Corrupt and repair.
        let mut corrupted = payload;
        corrupted[0] ^= 0xFF;
        std::fs::write(&path, &corrupted).expect("corrupt");
        let result = protector.verify_and_repair_file(&path).expect("repair");
        assert!(matches!(result.status, FileHealth::Repaired { .. }));

        // Check log file.
        let log_path = log_dir.join("repair-events.jsonl");
        assert!(log_path.exists(), "repair event log should exist");
        let contents = std::fs::read_to_string(&log_path).expect("read log");
        assert!(!contents.is_empty(), "log should not be empty");
        assert!(
            contents.contains("repair_succeeded"),
            "log should contain event data"
        );
    }

    #[test]
    fn repair_event_logging_creates_missing_directory() {
        let parent = temp_dir("repair-log-create");
        let log_dir = parent.join("nested").join("logs");
        assert!(
            !log_dir.exists(),
            "test precondition requires missing log directory"
        );
        let metrics = Arc::new(crate::metrics::DurabilityMetrics::default());
        let pipeline_config = RepairPipelineConfig {
            repair_log_dir: Some(log_dir.clone()),
            ..RepairPipelineConfig::default()
        };
        let protector = FileProtector::new_with_pipeline_config(
            Arc::new(MockRepairCodec),
            test_config(),
            metrics,
            pipeline_config,
        )
        .expect("protector");

        let path = temp_path("repair-log-create-file");
        let payload = vec![42_u8; 500];
        std::fs::write(&path, &payload).expect("write");
        protector.protect_file(&path).expect("protect");
        let mut corrupted = payload;
        corrupted[0] ^= 0xFF;
        std::fs::write(&path, &corrupted).expect("corrupt");
        let result = protector.verify_and_repair_file(&path).expect("repair");
        assert!(matches!(result.status, FileHealth::Repaired { .. }));

        let log_path = log_dir.join("repair-events.jsonl");
        assert!(log_path.exists(), "repair event log should be created");
    }

    #[test]
    fn iso8601_now_uses_utc_timestamp_shape() {
        let ts = super::iso8601_now();
        assert_eq!(ts.len(), 20);
        assert!(ts.ends_with('Z'));
        assert_eq!(&ts[4..5], "-");
        assert_eq!(&ts[7..8], "-");
        assert_eq!(&ts[10..11], "T");
        assert_eq!(&ts[13..14], ":");
        assert_eq!(&ts[16..17], ":");
    }

    // --- DurabilityProvider trait impl tests ---

    #[test]
    fn file_protector_implements_durability_provider() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("provider-impl");
        std::fs::write(&path, vec![42_u8; 500]).expect("write");

        // Use through trait.
        let provider: &dyn DurabilityProvider = &protector;
        let _protection = provider.protect(&path).expect("protect via trait");

        let verify = provider.verify(&path).expect("verify via trait");
        assert!(verify.healthy);

        let health = provider.check_health(&path).expect("health via trait");
        assert!(matches!(health.status, FileHealth::Intact));
    }

    // --- protect_all_existing migration test ---

    #[test]
    fn protect_all_existing_migration() {
        let dir = temp_dir("migrate");
        std::fs::write(dir.join("old_index.fsvi"), vec![1_u8; 300]).expect("write");
        std::fs::write(dir.join("old_index.tantivy"), vec![2_u8; 400]).expect("write");

        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");
        let report = protector.protect_all_existing(&dir).expect("migrate");
        assert_eq!(report.files_protected, 2);

        // Verify both have sidecars now.
        assert!(dir.join("old_index.fsvi.fec").exists());
        assert!(dir.join("old_index.tantivy.fec").exists());
    }

    // --- Corruption simulation tests ---

    #[test]
    fn detect_single_bit_flip() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("single-bit-flip");
        let payload = vec![42_u8; 500];
        std::fs::write(&path, &payload).expect("write");
        let result = protector.protect_file(&path).expect("protect");

        // Flip a single bit.
        let mut corrupted = payload;
        corrupted[100] ^= 0x01;
        std::fs::write(&path, &corrupted).expect("corrupt");

        let verify = protector
            .verify_file(&path, &result.sidecar_path)
            .expect("verify");
        assert!(!verify.healthy, "single bit flip should be detected");
    }

    #[test]
    fn repair_single_bit_flip() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("repair-bit-flip");
        let payload = vec![42_u8; 500];
        std::fs::write(&path, &payload).expect("write");
        protector.protect_file(&path).expect("protect");

        // Flip a single bit and repair via pipeline.
        let mut corrupted = payload.clone();
        corrupted[100] ^= 0x01;
        std::fs::write(&path, &corrupted).expect("corrupt");

        let result = protector.verify_and_repair_file(&path).expect("repair");
        assert!(
            matches!(result.status, FileHealth::Repaired { .. }),
            "single bit flip should be repaired, got {:?}",
            result.status
        );
        let restored = std::fs::read(&path).expect("read");
        assert_eq!(restored, payload);
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn detect_zeroed_block() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("zeroed-block");
        // Use data with non-zero content so zeroing is detectable.
        let payload: Vec<u8> = (0u32..1024).map(|i| (i % 256) as u8).collect();
        std::fs::write(&path, &payload).expect("write");
        let result = protector.protect_file(&path).expect("protect");

        // Zero out a 256-byte block (one symbol).
        let mut corrupted = payload;
        corrupted[256..512].fill(0);
        std::fs::write(&path, &corrupted).expect("corrupt");

        let verify = protector
            .verify_file(&path, &result.sidecar_path)
            .expect("verify");
        assert!(!verify.healthy, "zeroed block should be detected");
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn repair_zeroed_block() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("repair-zeroed");
        let payload: Vec<u8> = (0u32..1024).map(|i| (i % 256) as u8).collect();
        std::fs::write(&path, &payload).expect("write");
        protector.protect_file(&path).expect("protect");

        // Zero out a 256-byte block.
        let mut corrupted = payload.clone();
        corrupted[256..512].fill(0);
        std::fs::write(&path, &corrupted).expect("corrupt");

        let result = protector.verify_and_repair_file(&path).expect("repair");
        assert!(
            matches!(result.status, FileHealth::Repaired { .. }),
            "zeroed block should be repaired, got {:?}",
            result.status
        );
        let restored = std::fs::read(&path).expect("read");
        assert_eq!(restored, payload);
    }

    #[test]
    fn detect_appended_data() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("appended-data");
        let payload = vec![42_u8; 500];
        std::fs::write(&path, &payload).expect("write");
        let result = protector.protect_file(&path).expect("protect");

        // Append extra bytes.
        let mut extended = payload;
        extended.extend_from_slice(&[0xFF; 100]);
        std::fs::write(&path, &extended).expect("extend");

        let verify = protector
            .verify_file(&path, &result.sidecar_path)
            .expect("verify");
        assert!(!verify.healthy, "appended data should change CRC");
        assert_ne!(verify.expected_len, verify.actual_len);
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn repair_multiple_non_adjacent_corruptions() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("multi-corrupt");
        let payload: Vec<u8> = (0u32..2048).map(|i| ((i * 7) % 256) as u8).collect();
        std::fs::write(&path, &payload).expect("write");
        protector.protect_file(&path).expect("protect");

        // Corrupt 3 non-adjacent 32-byte regions.
        let mut corrupted = payload.clone();
        for byte in &mut corrupted[0..32] {
            *byte ^= 0xFF;
        }
        for byte in &mut corrupted[512..544] {
            *byte ^= 0xFF;
        }
        for byte in &mut corrupted[1024..1056] {
            *byte ^= 0xFF;
        }
        std::fs::write(&path, &corrupted).expect("corrupt");

        let result = protector.verify_and_repair_file(&path).expect("repair");
        assert!(
            matches!(result.status, FileHealth::Repaired { .. }),
            "multiple non-adjacent corruptions should be repaired, got {:?}",
            result.status
        );
        let restored = std::fs::read(&path).expect("read");
        assert_eq!(restored, payload);
    }

    #[test]
    fn small_file_protect_and_repair() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        // File smaller than one symbol (256 bytes).
        let path = temp_path("tiny-file");
        let payload = vec![7_u8; 50];
        std::fs::write(&path, &payload).expect("write");
        protector.protect_file(&path).expect("protect");

        // Delete and repair.
        std::fs::remove_file(&path).expect("delete");
        let result = protector.verify_and_repair_file(&path).expect("repair");
        assert!(
            matches!(result.status, FileHealth::Repaired { .. }),
            "small file should be repaired, got {:?}",
            result.status
        );
        let restored = std::fs::read(&path).expect("read");
        assert_eq!(restored, payload);
    }

    #[test]
    fn directory_skips_hidden_and_backup_files() {
        let dir = temp_dir("skip-hidden");
        std::fs::write(dir.join("normal.dat"), vec![1_u8; 300]).expect("write normal");
        std::fs::write(dir.join(".hidden"), vec![2_u8; 300]).expect("write hidden");
        std::fs::write(dir.join("old.dat.corrupt.12345"), vec![3_u8; 300]).expect("write backup");

        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");
        let report = protector.protect_directory(&dir).expect("protect");
        assert_eq!(
            report.files_protected, 1,
            "only normal.dat should be protected"
        );
    }

    #[cfg(unix)]
    #[test]
    fn directory_scans_skip_symlink_entries() {
        use std::os::unix::fs::symlink;

        let dir = temp_dir("skip-symlink");
        let external_target = temp_path("symlink-target");
        std::fs::write(&external_target, vec![9_u8; 256]).expect("write symlink target");
        let link_path = dir.join("external-link.dat");
        symlink(&external_target, &link_path).expect("create symlink");

        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");
        let protect_report = protector
            .protect_directory(&dir)
            .expect("protect directory");
        assert_eq!(
            protect_report.files_protected, 0,
            "symlinks must be skipped during protection scans"
        );
        assert!(
            !FileProtector::sidecar_path(&link_path).exists(),
            "sidecar should not be created for symlink entries"
        );

        let verify_report = protector.verify_directory(&dir).expect("verify directory");
        assert!(
            verify_report.results.is_empty(),
            "symlinks must be skipped during verification scans"
        );

        let _ = std::fs::remove_file(link_path);
        let _ = std::fs::remove_file(external_target);
    }

    #[test]
    fn empty_file_protect_and_verify() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("empty-file");
        std::fs::write(&path, []).expect("write empty");

        // Empty file should still be protectable (0 source symbols).
        let result = protector.protect_file(&path).expect("protect");
        assert_eq!(result.source_len, 0);

        let verify = protector
            .verify_file(&path, &result.sidecar_path)
            .expect("verify");
        assert!(verify.healthy);
    }

    #[test]
    fn empty_file_restore_from_sidecar() {
        let protector =
            FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector");

        let path = temp_path("empty-file-restore");
        std::fs::write(&path, []).expect("write empty");
        protector.protect_file(&path).expect("protect");

        std::fs::remove_file(&path).expect("delete");
        let result = protector.verify_and_repair_file(&path).expect("repair");
        assert!(
            matches!(result.status, FileHealth::Repaired { .. }),
            "empty file should be repairable from sidecar, got {:?}",
            result.status
        );

        let restored = std::fs::read(&path).expect("read");
        assert!(restored.is_empty());
    }
}

// ─── E2E corruption-and-recovery integration tests (bd-3w1.18) ──────────────
//
// These tests verify end-to-end corruption detection, repair, and recovery
// scenarios that exercise the full durability pipeline across components.
#[cfg(test)]
mod e2e_tests {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use fsqlite_core::raptorq_integration::{CodecDecodeResult, CodecEncodeResult, SymbolCodec};

    use super::{FileHealth, FileProtector, RepairPipelineConfig};
    use crate::config::DurabilityConfig;
    use crate::fsvi_protector::{FsviProtector, FsviVerifyResult};
    use crate::metrics::DurabilityMetrics;
    use frankensearch_core::SearchError;

    /// Mock codec that creates 1:1 repair symbols (each source symbol has a
    /// matching repair symbol at ESI + `1_000_000`). This allows repair of any
    /// individual corrupted symbol.
    #[derive(Debug)]
    struct MockRepairCodec;

    impl SymbolCodec for MockRepairCodec {
        fn encode(
            &self,
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
                            reason: fsqlite_core::raptorq_integration::DecodeFailureReason::InsufficientSymbols,
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

    fn temp_path(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "frankensearch-e2e-{prefix}-{}-{nanos}.bin",
            std::process::id()
        ))
    }

    fn temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "frankensearch-e2e-dir-{prefix}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn test_config() -> DurabilityConfig {
        DurabilityConfig {
            symbol_size: 256,
            repair_overhead: 2.0,
            ..DurabilityConfig::default()
        }
    }

    fn make_protector() -> FileProtector {
        FileProtector::new(Arc::new(MockRepairCodec), test_config()).expect("protector")
    }

    fn make_fsvi_protector() -> FsviProtector {
        FsviProtector::new(Arc::new(MockRepairCodec), test_config()).expect("fsvi protector")
    }

    #[allow(clippy::cast_possible_truncation)]
    fn synthetic_data(size: usize) -> Vec<u8> {
        (0..size).map(|i| ((i * 7 + 13) % 256) as u8).collect()
    }

    // ── Scenario 1: Power loss during index write (truncated file) ──

    #[test]
    fn power_loss_truncated_file_repaired_from_sidecar() {
        let protector = make_protector();
        let path = temp_path("power-loss");
        let payload = synthetic_data(2048);

        // Write and protect original file.
        std::fs::write(&path, &payload).expect("write");
        let protected = protector.protect_file(&path).expect("protect");

        // Simulate power loss: truncate file at random offset mid-write.
        let truncated = &payload[..payload.len() / 3];
        std::fs::write(&path, truncated).expect("truncate");

        // Verify detects corruption (length mismatch + CRC mismatch).
        let verify = protector
            .verify_file(&path, &protected.sidecar_path)
            .expect("verify");
        assert!(!verify.healthy);
        assert_ne!(verify.expected_len, verify.actual_len);

        // Repair restores the original.
        let result = protector.verify_and_repair_file(&path).expect("repair");
        assert!(
            matches!(result.status, FileHealth::Repaired { .. }),
            "truncated file should be repaired, got {:?}",
            result.status
        );
        let restored = std::fs::read(&path).expect("read");
        assert_eq!(restored, payload);
    }

    // ── Scenario 2: Gradual bit rot (cumulative bit flips) ──────────

    #[test]
    fn gradual_bit_rot_survives_repeated_single_bit_flips() {
        let protector = make_protector();
        let path = temp_path("bit-rot");
        let payload = synthetic_data(4096);

        std::fs::write(&path, &payload).expect("write");
        protector.protect_file(&path).expect("protect");

        // Simulate 10 "days" of single-bit rot: each day, flip one bit,
        // detect corruption, repair, and re-protect.
        let mut surviving_days = 0;
        for day in 0..10_usize {
            let mut data = std::fs::read(&path).expect("read current");
            let byte_idx = (day * 137 + 41) % data.len();
            let bit_idx = (day * 3 + 1) % 8;
            data[byte_idx] ^= 1 << bit_idx;
            std::fs::write(&path, &data).expect("inject bit rot");

            let result = protector.verify_and_repair_file(&path).expect("repair");
            match result.status {
                FileHealth::Repaired { .. } => {
                    surviving_days += 1;
                    // Re-protect with fresh sidecar after repair.
                    protector.protect_file(&path).expect("re-protect");
                }
                FileHealth::Intact => {
                    // If somehow the bit flip didn't change CRC (unlikely).
                    surviving_days += 1;
                }
                _ => break,
            }
        }

        assert_eq!(
            surviving_days, 10,
            "index should survive 10 days of gradual bit rot"
        );

        // Final data should match original.
        let final_data = std::fs::read(&path).expect("read final");
        assert_eq!(final_data, payload);
    }

    // ── Scenario 3: Storage medium failure (zeroed blocks) ──────────

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn zeroed_block_bad_sector_is_repaired() {
        let protector = make_protector();
        let path = temp_path("bad-sector");
        let payload = synthetic_data(4096);

        std::fs::write(&path, &payload).expect("write");
        protector.protect_file(&path).expect("protect");

        // Simulate a bad sector: zero out a 256-byte block.
        let mut corrupted = payload.clone();
        corrupted[512..768].fill(0);
        std::fs::write(&path, &corrupted).expect("corrupt");

        let result = protector.verify_and_repair_file(&path).expect("repair");
        assert!(
            matches!(result.status, FileHealth::Repaired { .. }),
            "bad sector should be repaired, got {:?}",
            result.status
        );
        let restored = std::fs::read(&path).expect("read");
        assert_eq!(restored, payload);
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn multiple_zeroed_blocks_repaired() {
        let protector = make_protector();
        let path = temp_path("multi-bad-sector");
        let payload = synthetic_data(4096);

        std::fs::write(&path, &payload).expect("write");
        protector.protect_file(&path).expect("protect");

        // Zero out 3 non-adjacent 256-byte blocks (simulating 3 bad sectors).
        let mut corrupted = payload.clone();
        corrupted[0..256].fill(0);
        corrupted[1024..1280].fill(0);
        corrupted[2560..2816].fill(0);
        std::fs::write(&path, &corrupted).expect("corrupt");

        let result = protector.verify_and_repair_file(&path).expect("repair");
        assert!(
            matches!(result.status, FileHealth::Repaired { .. }),
            "multiple bad sectors should be repaired, got {:?}",
            result.status
        );
        let restored = std::fs::read(&path).expect("read");
        assert_eq!(restored, payload);
    }

    // ── Scenario 4: Cascading corruption (index + sidecar) ──────────

    #[test]
    fn corrupted_sidecar_makes_repair_impossible() {
        let protector = make_protector();
        let path = temp_path("cascade");
        let payload = synthetic_data(1024);

        std::fs::write(&path, &payload).expect("write");
        let protected = protector.protect_file(&path).expect("protect");

        // Corrupt both the index and the sidecar.
        let mut corrupted_data = payload;
        corrupted_data[0] ^= 0xFF;
        std::fs::write(&path, &corrupted_data).expect("corrupt index");

        // Corrupt sidecar trailer (overwrite magic bytes).
        let mut sidecar = std::fs::read(&protected.sidecar_path).expect("read sidecar");
        if sidecar.len() >= 4 {
            sidecar[0..4].fill(0x00);
        }
        std::fs::write(&protected.sidecar_path, &sidecar).expect("corrupt sidecar");

        // Repair should fail gracefully (Unrecoverable, not panic).
        let result = protector.verify_and_repair_file(&path);
        if let Ok(check) = result {
            assert!(
                matches!(
                    check.status,
                    FileHealth::Unrecoverable { .. } | FileHealth::Unprotected
                ),
                "cascading corruption should be unrecoverable, got {:?}",
                check.status
            );
        }
        // An Err is also acceptable (corrupted sidecar can't be parsed).
    }

    // ── Scenario 5: Full index deletion and detection ───────────────

    #[test]
    fn deleted_index_detected_and_rebuilt_from_sidecar() {
        let protector = make_protector();
        let path = temp_path("full-delete");
        let payload = synthetic_data(2048);

        std::fs::write(&path, &payload).expect("write");
        protector.protect_file(&path).expect("protect");

        // Delete the index file entirely (simulating catastrophic loss).
        std::fs::remove_file(&path).expect("delete");
        assert!(!path.exists());

        // Repair restores from sidecar.
        let result = protector.verify_and_repair_file(&path).expect("repair");
        assert!(
            matches!(result.status, FileHealth::Repaired { .. }),
            "deleted file should be rebuilt from sidecar, got {:?}",
            result.status
        );
        let restored = std::fs::read(&path).expect("read");
        assert_eq!(restored, payload);
    }

    // ── Scenario 6: FEC sidecar corruption detection ────────────────

    #[test]
    fn corrupted_fec_sidecar_detected_by_verification() {
        let protector = make_protector();
        let path = temp_path("fec-corrupt-detect");
        let payload = synthetic_data(1024);

        std::fs::write(&path, &payload).expect("write");
        let protected = protector.protect_file(&path).expect("protect");

        // Corrupt the FEC sidecar (flip bytes in trailer).
        let mut sidecar = std::fs::read(&protected.sidecar_path).expect("read sidecar");
        if sidecar.len() >= 10 {
            // Corrupt data in the middle of the sidecar.
            let mid = sidecar.len() / 2;
            sidecar[mid] ^= 0xFF;
            sidecar[mid + 1] ^= 0xFF;
        }
        std::fs::write(&protected.sidecar_path, &sidecar).expect("corrupt sidecar");

        // Corrupted sidecar means the repair trailer CRC won't match,
        // so verify_file returns an IndexCorrupted error during deserialization.
        let verify_err = protector
            .verify_file(&path, &protected.sidecar_path)
            .expect_err("corrupted sidecar should fail verification");
        assert!(
            matches!(verify_err, SearchError::IndexCorrupted { .. }),
            "expected IndexCorrupted, got: {verify_err:?}"
        );

        // Regenerate sidecar from intact source → verification succeeds again.
        let re_protected = protector.protect_file(&path).expect("re-protect");
        let new_verify = protector
            .verify_file(&path, &re_protected.sidecar_path)
            .expect("verify new");
        assert!(new_verify.healthy, "regenerated sidecar should verify");
    }

    // ── Scenario 7: FSVI-specific protect-corrupt-repair cycle ──────

    #[test]
    fn fsvi_protect_corrupt_repair_preserves_data() {
        let protector = make_fsvi_protector();
        let path = temp_path("fsvi-e2e");
        // Fake FSVI file content.
        let payload = synthetic_data(3000);

        std::fs::write(&path, &payload).expect("write");
        let protected = protector.protect_atomic(&path).expect("protect");
        assert!(protected.sidecar_path.exists());

        // Verify original is intact.
        let verify = protector.verify(&path).expect("verify");
        assert!(
            matches!(verify, FsviVerifyResult::Intact),
            "expected Intact, got {:?}",
            verify
        );

        // Corrupt the FSVI file (byte flips in data region).
        let mut corrupted = payload.clone();
        for i in (0..corrupted.len()).step_by(300) {
            corrupted[i] ^= 0xFF;
        }
        std::fs::write(&path, &corrupted).expect("corrupt");

        // Verify detects corruption.
        let verify = protector.verify(&path).expect("verify corrupted");
        assert!(
            matches!(verify, FsviVerifyResult::Corrupted { repairable: true }),
            "expected Corrupted+repairable, got {:?}",
            verify
        );

        // Repair restores original data.
        let repaired = protector.repair(&path).expect("repair");
        assert!(repaired.bytes_written > 0);

        let restored = std::fs::read(&path).expect("read");
        assert_eq!(restored, payload);
    }

    // ── Scenario 8: Directory-level corruption and recovery ─────────

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn directory_level_mixed_corruption_recovery() {
        let dir = temp_dir("dir-e2e");
        let protector = make_protector();

        // Create 5 data files with unique content.
        let mut original_data = Vec::new();
        for i in 0..5 {
            let data = synthetic_data(512 + i * 100);
            let name = format!("index-{i}.dat");
            std::fs::write(dir.join(&name), &data).expect("write");
            original_data.push((name, data));
        }

        // Protect all files.
        let protect_report = protector.protect_directory(&dir).expect("protect");
        assert_eq!(protect_report.files_protected, 5);

        // Corrupt files 0 and 2 (byte flip), delete file 4.
        let mut corrupted = original_data[0].1.clone();
        corrupted[0] ^= 0xFF;
        std::fs::write(dir.join(&original_data[0].0), &corrupted).expect("corrupt 0");

        let mut corrupted2 = original_data[2].1.clone();
        corrupted2[100] ^= 0xFF;
        std::fs::write(dir.join(&original_data[2].0), &corrupted2).expect("corrupt 2");

        std::fs::remove_file(dir.join(&original_data[4].0)).expect("delete 4");

        // Verify directory: should detect 3 issues (2 corrupted + 1 missing).
        let health = protector.verify_directory(&dir).expect("verify");
        assert_eq!(health.intact_count, 2, "files 1 and 3 should be intact");
        assert!(
            health.repaired_count >= 2,
            "at least files 0 and 2 should be repaired"
        );

        // Verify all files are restored.
        for (name, data) in &original_data {
            let path = dir.join(name);
            if path.exists() {
                let restored = std::fs::read(&path).expect("read restored");
                assert_eq!(
                    &restored, data,
                    "{name} should be restored to original content"
                );
            }
        }
    }

    // ── Scenario 9: Metrics accumulation across repair pipeline ─────

    #[test]
    fn metrics_track_all_repair_operations() {
        let metrics = Arc::new(DurabilityMetrics::default());
        let protector = FileProtector::new_with_metrics(
            Arc::new(MockRepairCodec),
            test_config(),
            Arc::clone(&metrics),
        )
        .expect("protector");

        // Protect 3 files.
        let paths: Vec<_> = (0..3)
            .map(|i| {
                let path = temp_path(&format!("metrics-{i}"));
                let data = synthetic_data(512 + i * 100);
                std::fs::write(&path, &data).expect("write");
                (path, data)
            })
            .collect();

        for (path, _) in &paths {
            protector.protect_file(path).expect("protect");
        }

        let snap = metrics.snapshot();
        assert_eq!(snap.encode_ops, 3, "3 encode operations expected");
        assert!(snap.encoded_bytes_total > 0);
        assert!(snap.source_symbols_total > 0);
        assert!(snap.repair_symbols_total > 0);

        // Corrupt and repair 2 files.
        for (path, _) in &paths[0..2] {
            let mut data = std::fs::read(path).expect("read");
            data[0] ^= 0xFF;
            std::fs::write(path, &data).expect("corrupt");
            protector.verify_and_repair_file(path).expect("repair");
        }

        let snap = metrics.snapshot();
        assert_eq!(snap.repair_attempts, 2);
        assert_eq!(snap.repair_successes, 2);
        assert_eq!(snap.repair_failures, 0);
        assert!(snap.decode_ops >= 2);
    }

    // ── Scenario 10: Repair logging with event trail ────────────────

    #[test]
    fn repair_events_logged_to_jsonl_across_multiple_repairs() {
        let log_dir = temp_dir("e2e-repair-log");
        let metrics = Arc::new(DurabilityMetrics::default());
        let pipeline_config = RepairPipelineConfig {
            repair_log_dir: Some(log_dir.clone()),
            ..RepairPipelineConfig::default()
        };
        let protector = FileProtector::new_with_pipeline_config(
            Arc::new(MockRepairCodec),
            test_config(),
            metrics,
            pipeline_config,
        )
        .expect("protector");

        // Create, protect, corrupt, and repair 3 different files.
        for i in 0..3 {
            let path = temp_path(&format!("repair-log-{i}"));
            let payload = synthetic_data(512);
            std::fs::write(&path, &payload).expect("write");
            protector.protect_file(&path).expect("protect");

            let mut corrupted = payload;
            corrupted[i * 50] ^= 0xFF;
            std::fs::write(&path, &corrupted).expect("corrupt");
            let result = protector.verify_and_repair_file(&path).expect("repair");
            assert!(
                matches!(result.status, FileHealth::Repaired { .. }),
                "file {i} should be repaired"
            );
        }

        // Verify log file contains all 3 repair events.
        let log_path = log_dir.join("repair-events.jsonl");
        assert!(log_path.exists(), "repair event log should exist");
        let contents = std::fs::read_to_string(&log_path).expect("read log");
        let lines: Vec<_> = contents.lines().collect();
        assert_eq!(
            lines.len(),
            3,
            "expected 3 repair event lines, got {}",
            lines.len()
        );
        for line in &lines {
            assert!(
                line.contains("repair_succeeded"),
                "each line should contain repair_succeeded"
            );
        }
    }

    #[test]
    fn write_durable_creates_file_with_expected_content() {
        let dir = temp_dir("write-durable");
        let path = dir.join("durable.bin");
        let payload = b"durable content here";

        super::write_durable(&path, payload).expect("write_durable");

        let read_back = std::fs::read(&path).expect("read back");
        assert_eq!(read_back, payload);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_durable_overwrites_existing_file() {
        let dir = temp_dir("write-durable-overwrite");
        let path = dir.join("overwrite.bin");

        super::write_durable(&path, b"first").expect("write first");
        super::write_durable(&path, b"second").expect("write second");

        let read_back = std::fs::read(&path).expect("read back");
        assert_eq!(read_back, b"second");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_durable_empty_data_creates_zero_length_file() {
        let dir = temp_dir("write-durable-empty");
        let path = dir.join("empty.bin");

        super::write_durable(&path, &[]).expect("write empty");

        let meta = std::fs::metadata(&path).expect("metadata");
        assert_eq!(meta.len(), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_durable_fails_on_nonexistent_parent() {
        let path = std::path::Path::new("/nonexistent/dir/file.bin");
        let result = super::write_durable(path, b"data");
        assert!(result.is_err());
    }
}
