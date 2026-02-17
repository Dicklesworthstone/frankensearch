// Mmap (memmap2) requires unsafe for memory-mapped I/O in file_protector and fsvi_protector.
#![allow(unsafe_code)]
//! Durability primitives for frankensearch.
//!
//! This crate provides a thin, library-focused integration layer around
//! `FrankenSQLite`'s `RaptorQ` abstractions (`SymbolCodec`) plus:
//! - binary repair-trailer I/O,
//! - file protection / verification / repair orchestration,
//! - Tantivy segment helper wrappers,
//! - durability telemetry counters.
#![allow(
    clippy::missing_const_for_fn,
    clippy::missing_errors_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::uninlined_format_args
)]

pub mod codec;
pub mod config;
pub mod file_protector;
pub mod fsvi_protector;
pub mod metrics;
pub mod repair_trailer;
pub mod tantivy_wrapper;

pub use codec::{
    CodecFacade, DecodeFailureClass, DecodedPayload, DefaultSymbolCodec, EncodedData,
    EncodedPayload, RepairCodec, RepairCodecConfig, RepairData, VerifyResult,
    classify_decode_failure,
};
pub use config::DurabilityConfig;
pub use file_protector::{
    DirectoryHealthReport, DirectoryProtectionReport, DurabilityProvider, FileHealth,
    FileProtectionResult, FileProtector, FileRepairOutcome, FileVerifyResult, HealthCheckResult,
    NoopDurability, RepairPipelineConfig,
};
pub use fsvi_protector::{FsviProtectionResult, FsviProtector, FsviRepairResult, FsviVerifyResult};
pub use metrics::{DecodeOutcomeClass, DurabilityMetrics, DurabilityMetricsSnapshot};
pub use repair_trailer::{
    REPAIR_TRAILER_MAGIC, REPAIR_TRAILER_VERSION, RepairSymbol, RepairTrailerHeader,
    deserialize_repair_trailer, serialize_repair_trailer,
};
pub use tantivy_wrapper::{
    DurableTantivyIndex, SegmentHealthReport, SegmentProtectionReport, TantivySegmentProtector,
};
