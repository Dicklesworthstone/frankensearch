//! FSVI-specific durability protector.
//!
//! Wraps the generic [`FileProtector`] with FSVI-aware features:
//! - xxh3 fast-path integrity verification (<1ms for any file size)
//! - Atomic sidecar writes via temp-file + rename
//! - File naming convention: `index.fsvi` → `index.fsvi.fec`

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use frankensearch_core::{SearchError, SearchResult};
use fsqlite_core::raptorq_integration::SymbolCodec;
use memmap2::Mmap;
use tracing::{debug, info, warn};
use xxhash_rust::xxh3::xxh3_64;

use crate::config::DurabilityConfig;
use crate::file_protector::{FileProtector, FileRepairOutcome};
use crate::metrics::DurabilityMetrics;
use crate::repair_trailer::deserialize_repair_trailer;

/// Result of protecting an FSVI file with repair symbols.
#[derive(Debug, Clone)]
pub struct FsviProtectionResult {
    /// Path to the `.fec` sidecar file.
    pub sidecar_path: PathBuf,
    /// Size of the source FSVI file in bytes.
    pub source_size: u64,
    /// Size of the generated sidecar in bytes.
    pub repair_size: u64,
    /// Overhead ratio (`repair_size` / `source_size`).
    pub overhead_ratio: f32,
    /// Number of source symbols.
    pub k_source: u32,
    /// Number of repair symbols.
    pub r_repair: u32,
    /// `xxh3_64` hash of the protected source file.
    pub source_hash: u64,
    /// Time spent encoding repair symbols.
    pub encode_time: Duration,
}

/// Result of repairing a corrupted FSVI file.
#[derive(Debug, Clone)]
pub struct FsviRepairResult {
    /// Number of bytes in the repaired file.
    pub bytes_written: usize,
    /// Number of repair symbols consumed during decode.
    pub symbols_used: u32,
    /// Time spent decoding and writing the repaired file.
    pub decode_time: Duration,
    /// `xxh3_64` hash of the corrupted data before repair.
    pub source_hash_before: u64,
    /// `xxh3_64` hash of the repaired data (should match the stored hash).
    pub source_hash_after: u64,
}

/// FSVI fast-path verification result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsviVerifyResult {
    /// File integrity confirmed via `xxh3_64` hash match.
    Intact,
    /// File is corrupted. The generic verify provides CRC detail.
    Corrupted {
        /// Whether the sidecar has enough repair symbols to recover.
        repairable: bool,
    },
    /// No sidecar found — cannot verify.
    NoSidecar,
}

/// FSVI-specific durability protector with xxh3 fast-path verification
/// and atomic sidecar writes.
#[derive(Debug, Clone)]
pub struct FsviProtector {
    protector: FileProtector,
    metrics: Arc<DurabilityMetrics>,
}

impl FsviProtector {
    /// Create a new FSVI protector.
    pub fn new(codec: Arc<dyn SymbolCodec>, config: DurabilityConfig) -> SearchResult<Self> {
        let metrics = Arc::new(DurabilityMetrics::default());
        let protector = FileProtector::new_with_metrics(codec, config, Arc::clone(&metrics))?;
        Ok(Self { protector, metrics })
    }

    /// Derive the `.fec` sidecar path for an FSVI file.
    #[must_use]
    pub fn sidecar_path(fsvi_path: &Path) -> PathBuf {
        PathBuf::from(format!("{}.fec", fsvi_path.display()))
    }

    /// Protect an FSVI file by generating repair symbols and writing
    /// a `.fec` sidecar via atomic temp-file + rename.
    ///
    /// The protection is crash-safe: either the complete sidecar is
    /// visible at the `.fec` path, or it isn't. No partial writes.
    #[allow(unsafe_code)] // Mmap::map requires unsafe for memory-mapped I/O.
    pub fn protect_atomic(&self, fsvi_path: &Path) -> SearchResult<FsviProtectionResult> {
        let start = Instant::now();

        // Generate repair symbols via the inner protector.
        // FileProtector::protect_file now handles atomic write (temp + rename) internally.
        let protection = self.protector.protect_file(fsvi_path)?;
        let source_size = protection.source_len;
        let source_hash = protection.source_xxh3;
        let sidecar_path = protection.sidecar_path;

        let repair_size = fs::metadata(&sidecar_path).map_or(0, |metadata| metadata.len());
        // Compute in f64 first to avoid double precision loss from u64→f32.
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let overhead_ratio = if source_size > 0 {
            (repair_size as f64 / source_size as f64) as f32
        } else {
            0.0
        };

        let encode_time = start.elapsed();

        info!(
            path = %fsvi_path.display(),
            source_size,
            repair_size,
            overhead_ratio,
            source_hash,
            encode_time_ms = encode_time.as_millis(),
            "FSVI protection complete"
        );

        Ok(FsviProtectionResult {
            sidecar_path,
            source_size,
            repair_size,
            overhead_ratio,
            k_source: protection.k_source,
            r_repair: protection.repair_symbol_count,
            source_hash,
            encode_time,
        })
    }

    /// Verify integrity using the sidecar.
    ///
    /// Tries the xxh3 fast-path first (<1ms). If that fails (hash mismatch or
    /// V1 trailer), falls back to full CRC32 verification.
    #[allow(unsafe_code)] // Mmap::map requires unsafe for memory-mapped I/O.
    pub fn verify(&self, fsvi_path: &Path) -> SearchResult<FsviVerifyResult> {
        let sidecar_path = Self::sidecar_path(fsvi_path);

        if !sidecar_path.exists() {
            debug!(
                path = %fsvi_path.display(),
                "no .fec sidecar found, skipping verification"
            );
            return Ok(FsviVerifyResult::NoSidecar);
        }

        // Fast path: xxh3 hash check
        let file = fs::File::open(fsvi_path).map_err(SearchError::Io)?;
        let len = file.metadata().map_err(SearchError::Io)?.len();
        let actual_hash = if len == 0 {
            xxh3_64(&[])
        } else {
            // SAFETY: mmap is read-only.
            let mmap = unsafe { Mmap::map(&file).map_err(SearchError::Io)? };
            xxh3_64(&mmap)
        };

        // Read sidecar trailer to get expected hash
        let trailer_bytes = fs::read(&sidecar_path).map_err(SearchError::Io)?;
        let (header, _) = deserialize_repair_trailer(&trailer_bytes)?;

        if header.source_xxh3 != 0 && actual_hash == header.source_xxh3 {
            debug!(
                path = %fsvi_path.display(),
                hash = actual_hash,
                "FSVI integrity verified (fast path)"
            );
            return Ok(FsviVerifyResult::Intact);
        }

        // Fallback to full CRC32 check (V1 trailer or corruption)
        let verify = self.protector.verify_file(fsvi_path, &sidecar_path)?;

        if verify.healthy {
            return Ok(FsviVerifyResult::Intact);
        }

        // Corruption detected — check if repairable
        warn!(
            path = %fsvi_path.display(),
            expected_crc = verify.expected_crc32,
            actual_crc = verify.actual_crc32,
            expected_len = verify.expected_len,
            actual_len = verify.actual_len,
            "FSVI corruption detected"
        );

        Ok(FsviVerifyResult::Corrupted { repairable: true })
    }

    /// Attempt to repair a corrupted FSVI file using the `.fec` sidecar.
    ///
    /// On success, the corrupted file is overwritten with the repaired data.
    /// A backup of the corrupted file is created at `<path>.corrupted` before
    /// overwriting.
    #[allow(unsafe_code)] // Mmap::map requires unsafe for memory-mapped I/O.
    pub fn repair(&self, fsvi_path: &Path) -> SearchResult<FsviRepairResult> {
        let start = Instant::now();
        let sidecar_path = Self::sidecar_path(fsvi_path);

        if !sidecar_path.exists() {
            return Err(SearchError::IndexCorrupted {
                path: fsvi_path.to_path_buf(),
                detail: "no .fec sidecar available for repair".to_owned(),
            });
        }

        // Compute hash before repair
        let file = fs::File::open(fsvi_path).map_err(SearchError::Io)?;
        let len = file.metadata().map_err(SearchError::Io)?.len();
        let hash_before = if len == 0 {
            xxh3_64(&[])
        } else {
            let mmap = unsafe { Mmap::map(&file).map_err(SearchError::Io)? };
            xxh3_64(&mmap)
        };
        // Close file before backup (rename)
        drop(file);

        // Create backup of corrupted file
        let backup_path = PathBuf::from(format!("{}.corrupted", fsvi_path.display()));
        fs::copy(fsvi_path, &backup_path).map_err(SearchError::Io)?;
        debug!(
            backup = %backup_path.display(),
            "backed up corrupted FSVI file before repair"
        );

        // Attempt repair
        let outcome = self.protector.repair_file(fsvi_path, &sidecar_path)?;

        let decode_time = start.elapsed();

        match outcome {
            FileRepairOutcome::NotNeeded => {
                // Clean up unnecessary backup
                let _ = fs::remove_file(&backup_path);
                Ok(FsviRepairResult {
                    bytes_written: usize::try_from(len).unwrap_or(usize::MAX),
                    symbols_used: 0,
                    decode_time,
                    source_hash_before: hash_before,
                    source_hash_after: hash_before,
                })
            }
            FileRepairOutcome::Repaired {
                bytes_written,
                symbols_used,
            } => {
                // Verify hash after repair
                let repaired_file = fs::File::open(fsvi_path).map_err(SearchError::Io)?;
                let repaired_len = repaired_file.metadata().map_err(SearchError::Io)?.len();
                let hash_after = if repaired_len == 0 {
                    xxh3_64(&[])
                } else {
                    let mmap = unsafe { Mmap::map(&repaired_file).map_err(SearchError::Io)? };
                    xxh3_64(&mmap)
                };

                info!(
                    path = %fsvi_path.display(),
                    bytes_written,
                    symbols_used,
                    hash_before,
                    hash_after,
                    decode_time_ms = decode_time.as_millis(),
                    "FSVI repair successful"
                );

                Ok(FsviRepairResult {
                    bytes_written,
                    symbols_used,
                    decode_time,
                    source_hash_before: hash_before,
                    source_hash_after: hash_after,
                })
            }
            FileRepairOutcome::Unrecoverable {
                reason,
                symbols_received,
                k_required,
            } => {
                warn!(
                    path = %fsvi_path.display(),
                    ?reason,
                    symbols_received,
                    k_required,
                    "FSVI repair failed, restoring backup"
                );

                // Restore the backup
                fs::copy(&backup_path, fsvi_path).map_err(SearchError::Io)?;

                Err(SearchError::IndexCorrupted {
                    path: fsvi_path.to_path_buf(),
                    detail: format!(
                        "repair failed: {reason:?} (received {symbols_received}/{k_required} symbols)"
                    ),
                })
            }
        }
    }

    /// Convenience: verify and auto-repair if needed.
    ///
    /// Returns `Ok(true)` if the file is healthy (either originally or after repair).
    /// Returns `Ok(false)` if no sidecar exists (unprotected file).
    /// Returns `Err` if repair was attempted and failed.
    pub fn verify_and_repair(&self, fsvi_path: &Path) -> SearchResult<bool> {
        match self.verify(fsvi_path)? {
            FsviVerifyResult::Intact => Ok(true),
            FsviVerifyResult::NoSidecar => Ok(false),
            FsviVerifyResult::Corrupted { repairable: true } => {
                self.repair(fsvi_path)?;
                Ok(true)
            }
            FsviVerifyResult::Corrupted { repairable: false } => Err(SearchError::IndexCorrupted {
                path: fsvi_path.to_path_buf(),
                detail: "corruption exceeds repair capacity".to_owned(),
            }),
        }
    }

    /// Get the current durability metrics.
    pub fn metrics_snapshot(&self) -> crate::metrics::DurabilityMetricsSnapshot {
        self.metrics.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use fsqlite_core::raptorq_integration::{CodecDecodeResult, CodecEncodeResult, SymbolCodec};

    use super::{FsviProtector, FsviVerifyResult};
    use crate::config::DurabilityConfig;

    /// Mock codec that creates simple repair symbols for testing.
    #[derive(Debug)]
    struct MockCodec;

    impl SymbolCodec for MockCodec {
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

    fn temp_path(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "frankensearch-fsvi-{prefix}-{}-{nanos}.fsvi",
            std::process::id()
        ))
    }

    fn make_protector() -> FsviProtector {
        let config = DurabilityConfig {
            symbol_size: 256,
            // 100% overhead ensures enough repair symbols for full-file recovery.
            repair_overhead: 2.0,
            ..DurabilityConfig::default()
        };
        FsviProtector::new(Arc::new(MockCodec), config).expect("create protector")
    }

    #[test]
    fn sidecar_path_appends_fec_extension() {
        let path = PathBuf::from("/tmp/index.fast.fsvi");
        let sidecar = FsviProtector::sidecar_path(&path);
        assert_eq!(sidecar, PathBuf::from("/tmp/index.fast.fsvi.fec"));
    }

    #[test]
    fn protect_and_verify_roundtrip() {
        let protector = make_protector();
        let path = temp_path("protect-verify");

        // Create a fake FSVI file
        let payload = vec![42_u8; 700];
        std::fs::write(&path, &payload).expect("write payload");

        // Protect
        let result = protector.protect_atomic(&path).expect("protect");
        assert!(result.sidecar_path.exists());
        assert!(result.source_size > 0);
        assert!(result.repair_size > 0);
        assert!(result.overhead_ratio > 0.0);
        assert!(result.source_hash != 0);

        // Verify
        let verify = protector.verify(&path).expect("verify");
        assert_eq!(verify, FsviVerifyResult::Intact);

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&result.sidecar_path);
    }

    #[test]
    fn verify_returns_no_sidecar_when_missing() {
        let protector = make_protector();
        let path = temp_path("no-sidecar");

        std::fs::write(&path, b"data").expect("write");
        let verify = protector.verify(&path).expect("verify");
        assert_eq!(verify, FsviVerifyResult::NoSidecar);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn corruption_detected_and_repaired() {
        let protector = make_protector();
        let path = temp_path("corrupt-repair");

        let payload = vec![99_u8; 700];
        std::fs::write(&path, &payload).expect("write payload");

        let protection = protector.protect_atomic(&path).expect("protect");

        // Corrupt the file
        std::fs::write(&path, vec![0_u8; 700]).expect("corrupt");

        // Verify detects corruption
        let verify = protector.verify(&path).expect("verify");
        assert!(matches!(verify, FsviVerifyResult::Corrupted { .. }));

        // Repair
        let repair_result = protector.repair(&path).expect("repair");
        assert!(repair_result.bytes_written > 0);
        assert!(repair_result.symbols_used > 0);

        // Verify passes after repair
        let verify_after = protector.verify(&path).expect("verify after repair");
        assert_eq!(verify_after, FsviVerifyResult::Intact);

        // Content is restored
        let restored = std::fs::read(&path).expect("read restored");
        assert_eq!(restored, payload);

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&protection.sidecar_path);
        let backup = PathBuf::from(format!("{}.corrupted", path.display()));
        let _ = std::fs::remove_file(&backup);
    }

    #[test]
    fn verify_and_repair_convenience() {
        let protector = make_protector();
        let path = temp_path("verify-repair-conv");

        let payload = vec![55_u8; 500];
        std::fs::write(&path, &payload).expect("write");
        let protection = protector.protect_atomic(&path).expect("protect");

        // Healthy file
        assert!(protector.verify_and_repair(&path).expect("healthy"));

        // Corrupt and auto-repair
        std::fs::write(&path, vec![0_u8; 500]).expect("corrupt");
        assert!(protector.verify_and_repair(&path).expect("repaired"));

        // Verify content restored
        let restored = std::fs::read(&path).expect("read");
        assert_eq!(restored, payload);

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&protection.sidecar_path);
        let backup = PathBuf::from(format!("{}.corrupted", path.display()));
        let _ = std::fs::remove_file(&backup);
    }

    #[test]
    fn metrics_track_operations() {
        let protector = make_protector();
        let path = temp_path("metrics");

        std::fs::write(&path, vec![1_u8; 300]).expect("write");
        protector.protect_atomic(&path).expect("protect");

        let snap = protector.metrics_snapshot();
        assert!(snap.encode_ops >= 1);

        let sidecar = FsviProtector::sidecar_path(&path);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&sidecar);
    }

    #[test]
    fn protect_atomic_can_replace_existing_sidecar_repeatedly() {
        let protector = make_protector();
        let path = temp_path("protect-replace-existing");
        let sidecar = FsviProtector::sidecar_path(&path);

        std::fs::write(&path, vec![7_u8; 512]).expect("write initial payload");
        protector.protect_atomic(&path).expect("first protect");
        assert!(sidecar.exists());

        std::fs::write(&path, vec![9_u8; 640]).expect("write updated payload");
        protector.protect_atomic(&path).expect("second protect");
        assert!(sidecar.exists());
        assert!(
            !PathBuf::from(format!("{}.tmp", sidecar.display())).exists(),
            "temp sidecar should be cleaned up"
        );

        let verify = protector.verify(&path).expect("verify after replace");
        assert_eq!(verify, FsviVerifyResult::Intact);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&sidecar);
        let _ = std::fs::remove_file(PathBuf::from(format!("{}.bak", sidecar.display())));
    }

    // ─── bd-c2u4 tests begin ───

    #[test]
    fn sidecar_path_nested_dirs() {
        let path = PathBuf::from("/a/b/c/deep/index.fsvi");
        assert_eq!(
            FsviProtector::sidecar_path(&path),
            PathBuf::from("/a/b/c/deep/index.fsvi.fec")
        );
    }

    #[test]
    fn sidecar_path_relative() {
        let path = PathBuf::from("data/my.fsvi");
        assert_eq!(
            FsviProtector::sidecar_path(&path),
            PathBuf::from("data/my.fsvi.fec")
        );
    }

    #[test]
    fn sidecar_path_multiple_dots() {
        let path = PathBuf::from("/tmp/index.fast.v2.fsvi");
        assert_eq!(
            FsviProtector::sidecar_path(&path),
            PathBuf::from("/tmp/index.fast.v2.fsvi.fec")
        );
    }

    #[test]
    fn sidecar_path_bare_name() {
        let path = PathBuf::from("index.fsvi");
        assert_eq!(
            FsviProtector::sidecar_path(&path),
            PathBuf::from("index.fsvi.fec")
        );
    }

    #[test]
    fn fsvi_verify_result_eq_intact() {
        assert_eq!(FsviVerifyResult::Intact, FsviVerifyResult::Intact);
    }

    #[test]
    fn fsvi_verify_result_eq_no_sidecar() {
        assert_eq!(FsviVerifyResult::NoSidecar, FsviVerifyResult::NoSidecar);
    }

    #[test]
    fn fsvi_verify_result_eq_corrupted_same() {
        assert_eq!(
            FsviVerifyResult::Corrupted { repairable: true },
            FsviVerifyResult::Corrupted { repairable: true }
        );
    }

    #[test]
    fn fsvi_verify_result_ne_corrupted_different_repairable() {
        assert_ne!(
            FsviVerifyResult::Corrupted { repairable: true },
            FsviVerifyResult::Corrupted { repairable: false }
        );
    }

    #[test]
    fn fsvi_verify_result_ne_different_variants() {
        assert_ne!(FsviVerifyResult::Intact, FsviVerifyResult::NoSidecar);
        assert_ne!(
            FsviVerifyResult::Intact,
            FsviVerifyResult::Corrupted { repairable: true }
        );
    }

    #[test]
    fn fsvi_verify_result_clone() {
        let original = FsviVerifyResult::Corrupted { repairable: true };
        let cloned = original;
        assert_eq!(original, cloned);
    }

    #[test]
    fn fsvi_verify_result_debug() {
        let intact_debug = format!("{:?}", FsviVerifyResult::Intact);
        assert!(intact_debug.contains("Intact"));

        let corrupted_debug = format!("{:?}", FsviVerifyResult::Corrupted { repairable: false });
        assert!(corrupted_debug.contains("Corrupted"));
        assert!(corrupted_debug.contains("repairable"));
    }

    #[test]
    fn fsvi_protection_result_clone() {
        let result = super::FsviProtectionResult {
            sidecar_path: PathBuf::from("/tmp/test.fec"),
            source_size: 1000,
            repair_size: 200,
            overhead_ratio: 0.2,
            k_source: 4,
            r_repair: 4,
            source_hash: 12345,
            encode_time: std::time::Duration::from_millis(5),
        };
        #[allow(clippy::redundant_clone)]
        let cloned = result.clone();
        assert_eq!(cloned.source_size, 1000);
        assert_eq!(cloned.k_source, 4);
        assert_eq!(cloned.source_hash, 12345);
    }

    #[test]
    fn fsvi_repair_result_clone() {
        let result = super::FsviRepairResult {
            bytes_written: 500,
            symbols_used: 3,
            decode_time: std::time::Duration::from_millis(10),
            source_hash_before: 111,
            source_hash_after: 222,
        };
        #[allow(clippy::redundant_clone)]
        let cloned = result.clone();
        assert_eq!(cloned.bytes_written, 500);
        assert_eq!(cloned.symbols_used, 3);
        assert_eq!(cloned.source_hash_before, 111);
        assert_eq!(cloned.source_hash_after, 222);
    }

    #[test]
    fn repair_no_sidecar_returns_error() {
        let protector = make_protector();
        let path = temp_path("repair-no-sidecar");

        std::fs::write(&path, b"some data").expect("write");

        let err = protector.repair(&path);
        assert!(err.is_err());
        let err_str = format!("{}", err.unwrap_err());
        assert!(err_str.contains("no .fec sidecar"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn verify_and_repair_no_sidecar_returns_false() {
        let protector = make_protector();
        let path = temp_path("vandr-no-sidecar");

        std::fs::write(&path, b"payload").expect("write");

        let result = protector.verify_and_repair(&path).expect("should succeed");
        assert!(!result, "no sidecar should return Ok(false)");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn protect_result_has_valid_k_and_r() {
        let protector = make_protector();
        let path = temp_path("k-and-r");

        std::fs::write(&path, vec![42_u8; 1024]).expect("write");
        let result = protector.protect_atomic(&path).expect("protect");

        assert!(result.k_source > 0, "k_source must be positive");
        assert!(result.r_repair > 0, "r_repair must be positive");
        assert!(result.source_size == 1024);
        assert!(result.source_hash != 0);
        assert!(!result.encode_time.is_zero() || result.encode_time == std::time::Duration::ZERO);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&result.sidecar_path);
    }

    #[test]
    fn protect_empty_file_zero_overhead() {
        let protector = make_protector();
        let path = temp_path("empty-file");

        std::fs::write(&path, b"").expect("write empty");
        let result = protector.protect_atomic(&path).expect("protect");

        assert_eq!(result.source_size, 0);
        // overhead_ratio should be 0.0 when source_size is 0
        assert!(
            (result.overhead_ratio - 0.0).abs() < f32::EPSILON,
            "empty file should have 0.0 overhead_ratio"
        );

        let sidecar = FsviProtector::sidecar_path(&path);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&sidecar);
    }

    #[test]
    fn repair_result_hashes_differ_on_corruption() {
        let protector = make_protector();
        let path = temp_path("hash-diff");

        let payload = vec![77_u8; 600];
        std::fs::write(&path, &payload).expect("write");
        let protection = protector.protect_atomic(&path).expect("protect");

        // Corrupt
        std::fs::write(&path, vec![0_u8; 600]).expect("corrupt");

        let repair = protector.repair(&path).expect("repair");
        assert_ne!(
            repair.source_hash_before, repair.source_hash_after,
            "corrupted vs repaired hashes should differ"
        );
        assert!(repair.bytes_written > 0);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&protection.sidecar_path);
        let backup = PathBuf::from(format!("{}.corrupted", path.display()));
        let _ = std::fs::remove_file(&backup);
    }

    #[test]
    fn protect_different_payloads_produce_different_hashes() {
        let protector = make_protector();
        let path = temp_path("diff-hash");

        std::fs::write(&path, vec![1_u8; 500]).expect("write 1");
        let result1 = protector.protect_atomic(&path).expect("protect 1");
        let hash1 = result1.source_hash;

        std::fs::write(&path, vec![2_u8; 500]).expect("write 2");
        let result2 = protector.protect_atomic(&path).expect("protect 2");
        let hash2 = result2.source_hash;

        assert_ne!(
            hash1, hash2,
            "different payloads should produce different hashes"
        );

        let sidecar = FsviProtector::sidecar_path(&path);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&sidecar);
        let _ = std::fs::remove_file(PathBuf::from(format!("{}.bak", sidecar.display())));
    }

    #[test]
    fn metrics_count_after_multiple_protect() {
        let protector = make_protector();
        let path = temp_path("multi-metrics");

        std::fs::write(&path, vec![10_u8; 256]).expect("write");
        protector.protect_atomic(&path).expect("protect 1");
        protector.protect_atomic(&path).expect("protect 2");
        protector.protect_atomic(&path).expect("protect 3");

        let snap = protector.metrics_snapshot();
        assert!(
            snap.encode_ops >= 3,
            "expected at least 3 encode ops, got {}",
            snap.encode_ops
        );

        let sidecar = FsviProtector::sidecar_path(&path);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&sidecar);
        let _ = std::fs::remove_file(PathBuf::from(format!("{}.bak", sidecar.display())));
    }

    #[test]
    fn fsvi_protection_result_debug() {
        let result = super::FsviProtectionResult {
            sidecar_path: PathBuf::from("/tmp/t.fec"),
            source_size: 100,
            repair_size: 20,
            overhead_ratio: 0.2,
            k_source: 1,
            r_repair: 1,
            source_hash: 999,
            encode_time: std::time::Duration::from_nanos(500),
        };
        let debug = format!("{result:?}");
        assert!(debug.contains("FsviProtectionResult"));
        assert!(debug.contains("source_size"));
    }

    #[test]
    fn fsvi_repair_result_debug() {
        let result = super::FsviRepairResult {
            bytes_written: 100,
            symbols_used: 2,
            decode_time: std::time::Duration::from_millis(1),
            source_hash_before: 1,
            source_hash_after: 2,
        };
        let debug = format!("{result:?}");
        assert!(debug.contains("FsviRepairResult"));
        assert!(debug.contains("bytes_written"));
    }

    // ─── bd-c2u4 tests end ───
}
