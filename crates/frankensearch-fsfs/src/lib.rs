//! fsfs scaffold crate.
//!
//! This crate establishes the standalone fsfs binary surface with explicit
//! separation between reusable runtime/config logic and UX adapters.

#![forbid(unsafe_code)]

pub mod adapters;
pub mod concurrency;
pub mod config;
pub mod evidence;
pub mod lifecycle;
pub mod mount_info;
pub mod redaction;
pub mod repro;
pub mod runtime;
pub mod shutdown;

pub use adapters::cli::{
    CliCommand, CliInput, CommandSource, ConfigAction, OutputFormat, detect_auto_mode, exit_code,
    parse_cli_args,
};
pub use adapters::tui::TuiAdapterSettings;
pub use concurrency::{
    AccessMode, ContentionMetrics, ContentionPolicy, ContentionSnapshot, LockLevel, LockOrderGuard,
    LockSentinel, PipelineStageAccess, ResourceId, ResourceToken, pipeline_access_matrix,
    read_sentinel, remove_sentinel, try_acquire_sentinel, write_sentinel,
};
pub use config::{
    CliOverrides, ConfigLoadResult, ConfigLoadedEvent, ConfigSource, ConfigWarning, Density,
    DiscoveryConfig, FsfsConfig, IndexingConfig, MountPolicyEntry, PathExpansion, PressureConfig,
    PressureProfile, PrivacyConfig, SearchConfig, StorageConfig, TextSelectionMode, TuiConfig,
    TuiTheme, default_config_file_path, emit_config_loaded, load_from_sources, load_from_str,
};
pub use evidence::{
    ALL_FSFS_REASON_CODES, FsfsEventFamily, FsfsEvidenceEvent, FsfsReasonCode, ScopeDecision,
    ScopeDecisionKind, TraceLink, ValidationResult, ValidationViolation, is_valid_fsfs_reason_code,
    validate_event,
};
pub use lifecycle::{
    DaemonPhase, DaemonStatus, HealthStatus, LifecycleTracker, LimitViolation, PidFile,
    PidFileContents, ResourceLimits, ResourceUsage, SubsystemHealth, SubsystemId, WatchdogConfig,
};
pub use mount_info::{
    ChangeDetectionStrategy, ErrorClass, FsCategory, MountEntry, MountOverride, MountPolicy,
    MountTable, ProbeResult, classify_fstype, classify_io_error, parse_proc_mounts, probe_mount,
    read_system_mounts,
};
pub use redaction::{
    ArtifactRetention, ArtifactType, DataClass, HARD_DENY_PATH_PATTERNS, MaskSeed, OutputSurface,
    REDACTION_POLICY_VERSION, RedactionPolicy, RedactionResult, RedactionTransform, TransformRule,
    classify_path, default_artifact_retention, default_rule_matrix, deterministic_hash,
    deterministic_mask, deterministic_truncate, is_hard_deny_path,
};
pub use repro::{
    ArtifactEntry, CaptureReason, EnvEntry, EnvSnapshot, FrameSeqRange, IndexChecksum,
    IndexChecksums, ModelManifest, ModelSnapshot, PACK_FILES, REPRO_SCHEMA_VERSION, ReplayMeta,
    ReplayMode, ReproInstance, ReproManifest, RetentionPolicy, RetentionTier, files_for_tier,
    should_capture_env, should_redact_env,
};
pub use runtime::{FsfsRuntime, InterfaceMode};
pub use shutdown::{FORCE_EXIT_WINDOW, ShutdownCoordinator, ShutdownReason, ShutdownState};
