use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fs;
use std::hash::BuildHasher;
use std::path::{Component, Path, PathBuf};
use std::str::FromStr;

use frankensearch_core::{SearchError, SearchResult};
use serde::{Deserialize, Serialize};
use tracing::info;

const PRECEDENCE: [ConfigSource; 4] = [
    ConfigSource::Cli,
    ConfigSource::Env,
    ConfigSource::File,
    ConfigSource::Defaults,
];

/// Versioned profile contract revision for pressure-profile policy resolution.
pub const PRESSURE_PROFILE_VERSION: u16 = 1;

/// Deterministic precedence chain for profile-managed fields.
pub const PROFILE_PRECEDENCE_CHAIN: [&str; 5] = [
    "hard_safety_guards",
    "cli_override",
    "env_override",
    "config_override",
    "profile_default",
];

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TextSelectionMode {
    #[default]
    Blocklist,
    Allowlist,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PressureProfile {
    Strict,
    #[default]
    Performance,
    Degraded,
}

impl FromStr for PressureProfile {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "strict" => Ok(Self::Strict),
            "performance" => Ok(Self::Performance),
            "degraded" => Ok(Self::Degraded),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProfileSchedulerMode {
    FairShare,
    LatencySensitive,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PressureProfileField {
    SchedulerMode,
    MaxEmbedConcurrency,
    MaxIndexConcurrency,
    QualityEnabled,
    AllowBackgroundIndexing,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProfileOverrideSource {
    Cli,
    Env,
    Config,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PressureProfileOverridePolicy {
    pub overridable_fields: Vec<PressureProfileField>,
    pub locked_fields: Vec<PressureProfileField>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PressureProfileEffectiveSettings {
    pub scheduler_mode: ProfileSchedulerMode,
    pub max_embed_concurrency: u8,
    pub max_index_concurrency: u8,
    pub quality_enabled: bool,
    pub allow_background_indexing: bool,
    pub pressure_enter_threshold_per_mille: u16,
    pub pressure_exit_threshold_per_mille: u16,
    pub override_policy: PressureProfileOverridePolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PressureProfileOverrideDecision {
    pub field: PressureProfileField,
    pub source: ProfileOverrideSource,
    pub requested_value: String,
    pub applied: bool,
    pub reason_code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PressureProfileSafetyClamp {
    pub field: PressureProfileField,
    pub clamped_to: String,
    pub reason_code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PressureProfileResolutionDiagnostics {
    pub event: String,
    pub precedence_chain: Vec<String>,
    pub reason_code: String,
    pub effective_profile_version: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PressureProfileResolution {
    pub selected_profile: PressureProfile,
    pub overrides: Vec<PressureProfileOverrideDecision>,
    pub effective: PressureProfileEffectiveSettings,
    pub safety_clamps: Vec<PressureProfileSafetyClamp>,
    pub conflict_detected: bool,
    pub diagnostics: PressureProfileResolutionDiagnostics,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DegradationOverrideMode {
    #[default]
    Auto,
    ForceFull,
    ForceEmbedDeferred,
    ForceLexicalOnly,
    ForceMetadataOnly,
    ForcePaused,
}

impl FromStr for DegradationOverrideMode {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "full" | "force_full" => Ok(Self::ForceFull),
            "embed_deferred" | "force_embed_deferred" => Ok(Self::ForceEmbedDeferred),
            "lexical_only" | "force_lexical_only" => Ok(Self::ForceLexicalOnly),
            "metadata_only" | "force_metadata_only" => Ok(Self::ForceMetadataOnly),
            "paused" | "force_paused" => Ok(Self::ForcePaused),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TuiTheme {
    Auto,
    Light,
    #[default]
    Dark,
}

impl FromStr for TuiTheme {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "light" => Ok(Self::Light),
            "dark" => Ok(Self::Dark),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Density {
    Compact,
    #[default]
    Normal,
    Expanded,
}

const HIGH_UTILITY_EXTENSIONS: &[&str] = &[
    "rs", "py", "ts", "tsx", "js", "jsx", "go", "java", "kt", "swift", "c", "cpp", "h", "hpp",
    "toml", "yaml", "yml", "json", "md", "markdown", "txt", "rst", "sql", "proto", "ini", "cfg",
    "conf", "sh", "bash", "zsh", "fish",
];

const TEXT_ALLOWLIST_EXTENSIONS: &[&str] = &[
    "rs", "py", "ts", "tsx", "js", "jsx", "go", "java", "kt", "swift", "c", "cpp", "h", "hpp",
    "toml", "yaml", "yml", "json", "md", "markdown", "txt", "rst", "sql", "proto", "ini", "cfg",
    "conf", "sh", "bash", "zsh", "fish", "xml", "html", "css", "scss", "csv", "log",
];

const LOW_UTILITY_PATH_COMPONENTS: &[&str] = &[
    "node_modules",
    "target",
    "vendor",
    "__pycache__",
    ".venv",
    "dist",
    "build",
    ".next",
    ".cache",
];

const HIGH_SIGNAL_FILENAMES: &[&str] = &[
    "readme.md",
    "cargo.toml",
    "package.json",
    "pyproject.toml",
    "makefile",
    "justfile",
];

const LOW_UTILITY_FILENAMES: &[&str] = &[
    "cargo.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
];

const REASON_DISCOVERY_ROOT_ACCEPTED: &str = "discovery.root.accepted";
const REASON_DISCOVERY_ROOT_REJECTED: &str = "discovery.root.rejected";
const REASON_DISCOVERY_FILE_INCLUDED: &str = "discovery.file.included";
const REASON_DISCOVERY_FILE_EXCLUDED: &str = "discovery.file.excluded_pattern";
const REASON_DISCOVERY_FILE_TOO_LARGE: &str = "discovery.file.too_large";
const REASON_DISCOVERY_FILE_BINARY_BLOCKED: &str = "discovery.file.binary_blocked";

const STRICT_OVERRIDABLE_FIELDS: &[PressureProfileField] = &[
    PressureProfileField::SchedulerMode,
    PressureProfileField::MaxIndexConcurrency,
];
const STRICT_LOCKED_FIELDS: &[PressureProfileField] = &[
    PressureProfileField::QualityEnabled,
    PressureProfileField::AllowBackgroundIndexing,
    PressureProfileField::MaxEmbedConcurrency,
];
const PERFORMANCE_OVERRIDABLE_FIELDS: &[PressureProfileField] = &[
    PressureProfileField::SchedulerMode,
    PressureProfileField::MaxEmbedConcurrency,
    PressureProfileField::MaxIndexConcurrency,
    PressureProfileField::AllowBackgroundIndexing,
];
const PERFORMANCE_LOCKED_FIELDS: &[PressureProfileField] = &[PressureProfileField::QualityEnabled];
const DEGRADED_LOCKED_FIELDS: &[PressureProfileField] = &[
    PressureProfileField::SchedulerMode,
    PressureProfileField::MaxEmbedConcurrency,
    PressureProfileField::MaxIndexConcurrency,
    PressureProfileField::QualityEnabled,
    PressureProfileField::AllowBackgroundIndexing,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PressureProfileContract {
    scheduler_mode: ProfileSchedulerMode,
    max_embed_concurrency: u8,
    max_index_concurrency: u8,
    quality_enabled: bool,
    allow_background_indexing: bool,
    pressure_enter_threshold_per_mille: u16,
    pressure_exit_threshold_per_mille: u16,
    overridable_fields: &'static [PressureProfileField],
    locked_fields: &'static [PressureProfileField],
}

impl PressureProfileContract {
    #[must_use]
    fn to_effective_settings(self) -> PressureProfileEffectiveSettings {
        PressureProfileEffectiveSettings {
            scheduler_mode: self.scheduler_mode,
            max_embed_concurrency: self.max_embed_concurrency,
            max_index_concurrency: self.max_index_concurrency,
            quality_enabled: self.quality_enabled,
            allow_background_indexing: self.allow_background_indexing,
            pressure_enter_threshold_per_mille: self.pressure_enter_threshold_per_mille,
            pressure_exit_threshold_per_mille: self.pressure_exit_threshold_per_mille,
            override_policy: PressureProfileOverridePolicy {
                overridable_fields: self.overridable_fields.to_vec(),
                locked_fields: self.locked_fields.to_vec(),
            },
        }
    }

    #[must_use]
    fn is_locked_field(self, field: PressureProfileField) -> bool {
        let mut idx = 0;
        while idx < self.locked_fields.len() {
            if self.locked_fields[idx] == field {
                return true;
            }
            idx += 1;
        }
        false
    }
}

impl PressureProfile {
    #[must_use]
    const fn contract(self) -> PressureProfileContract {
        match self {
            Self::Strict => PressureProfileContract {
                scheduler_mode: ProfileSchedulerMode::FairShare,
                max_embed_concurrency: 2,
                max_index_concurrency: 2,
                quality_enabled: false,
                allow_background_indexing: false,
                pressure_enter_threshold_per_mille: 350,
                pressure_exit_threshold_per_mille: 200,
                overridable_fields: STRICT_OVERRIDABLE_FIELDS,
                locked_fields: STRICT_LOCKED_FIELDS,
            },
            Self::Performance => PressureProfileContract {
                scheduler_mode: ProfileSchedulerMode::LatencySensitive,
                max_embed_concurrency: 6,
                max_index_concurrency: 8,
                quality_enabled: true,
                allow_background_indexing: true,
                pressure_enter_threshold_per_mille: 650,
                pressure_exit_threshold_per_mille: 450,
                overridable_fields: PERFORMANCE_OVERRIDABLE_FIELDS,
                locked_fields: PERFORMANCE_LOCKED_FIELDS,
            },
            Self::Degraded => PressureProfileContract {
                scheduler_mode: ProfileSchedulerMode::FairShare,
                max_embed_concurrency: 1,
                max_index_concurrency: 1,
                quality_enabled: false,
                allow_background_indexing: false,
                pressure_enter_threshold_per_mille: 150,
                pressure_exit_threshold_per_mille: 100,
                overridable_fields: &[],
                locked_fields: DEGRADED_LOCKED_FIELDS,
            },
        }
    }
}

/// Ingestion class assigned by discovery policy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IngestionClass {
    FullSemanticLexical,
    LexicalOnly,
    MetadataOnly,
    Skip,
}

impl IngestionClass {
    #[must_use]
    pub const fn is_indexed(self) -> bool {
        !matches!(self, Self::Skip)
    }
}

/// Discovery scope decision produced by root/candidate policy evaluation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryScopeDecision {
    Include,
    Exclude,
}

/// Policy input for file-level discovery decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveryCandidate<'a> {
    pub path: &'a Path,
    pub byte_len: u64,
    pub is_symlink: bool,
    pub mount_category: Option<crate::mount_info::FsCategory>,
}

impl<'a> DiscoveryCandidate<'a> {
    #[must_use]
    pub const fn new(path: &'a Path, byte_len: u64) -> Self {
        Self {
            path,
            byte_len,
            is_symlink: false,
            mount_category: None,
        }
    }

    #[must_use]
    pub const fn with_symlink(mut self, is_symlink: bool) -> Self {
        self.is_symlink = is_symlink;
        self
    }

    #[must_use]
    pub const fn with_mount_category(
        mut self,
        mount_category: crate::mount_info::FsCategory,
    ) -> Self {
        self.mount_category = Some(mount_category);
        self
    }
}

/// Policy output for root-level discovery.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RootDiscoveryDecision {
    pub scope: DiscoveryScopeDecision,
    pub reason_codes: Vec<String>,
}

impl RootDiscoveryDecision {
    #[must_use]
    pub const fn include(&self) -> bool {
        matches!(self.scope, DiscoveryScopeDecision::Include)
    }
}

/// Policy output for file-level discovery and ingestion class assignment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiscoveryDecision {
    pub scope: DiscoveryScopeDecision,
    pub ingestion_class: IngestionClass,
    pub utility_score: i32,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiscoveryConfig {
    pub roots: Vec<String>,
    pub exclude_patterns: Vec<String>,
    pub text_selection_mode: TextSelectionMode,
    pub binary_blocklist_extensions: Vec<String>,
    pub max_file_size_mb: usize,
    pub follow_symlinks: bool,
    /// Per-mount-point policy overrides. Each entry maps a mount path
    /// (e.g., "/mnt/nfs") to its behavioral override.
    #[serde(default)]
    pub mount_overrides: Vec<MountPolicyEntry>,
    /// Whether to skip network mounts entirely during discovery.
    #[serde(default)]
    pub skip_network_mounts: bool,
}

/// A named mount-point policy override for the config file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MountPolicyEntry {
    /// Mount point path (e.g., "/mnt/nfs").
    pub mount_point: String,
    /// Optional category override for deterministic policy simulation.
    pub category: Option<crate::mount_info::FsCategory>,
    /// Whether to enable or disable this mount.
    pub enabled: Option<bool>,
    /// Override change detection strategy.
    pub change_detection: Option<crate::mount_info::ChangeDetectionStrategy>,
    /// Override stat timeout in milliseconds.
    pub stat_timeout_ms: Option<u64>,
    /// Override max concurrent I/O.
    pub max_concurrent_io: Option<usize>,
    /// Override poll interval in seconds.
    pub poll_interval_secs: Option<u64>,
}

impl MountPolicyEntry {
    /// Convert to a `MountOverride` for use with `MountTable`.
    #[must_use]
    pub const fn to_mount_override(&self) -> crate::mount_info::MountOverride {
        crate::mount_info::MountOverride {
            category: self.category,
            change_detection: self.change_detection,
            stat_timeout_ms: self.stat_timeout_ms,
            max_concurrent_io: self.max_concurrent_io,
            poll_interval_secs: self.poll_interval_secs,
            enabled: self.enabled,
        }
    }
}

impl DiscoveryConfig {
    /// Build a `HashMap` of mount overrides suitable for `MountTable::new`.
    #[must_use]
    pub fn mount_override_map(
        &self,
    ) -> std::collections::HashMap<String, crate::mount_info::MountOverride> {
        self.mount_overrides
            .iter()
            .map(|entry| (entry.mount_point.clone(), entry.to_mount_override()))
            .collect()
    }

    /// Evaluate whether a discovery root should be included before any walk.
    #[must_use]
    pub fn evaluate_root(
        &self,
        root: &Path,
        mount_category: Option<crate::mount_info::FsCategory>,
    ) -> RootDiscoveryDecision {
        let mut reason_codes = Vec::new();
        let normalized = normalize_path(root);

        if root.as_os_str().is_empty() {
            reason_codes.push(REASON_DISCOVERY_ROOT_REJECTED.to_string());
            return RootDiscoveryDecision {
                scope: DiscoveryScopeDecision::Exclude,
                reason_codes,
            };
        }

        if mount_category.is_some_and(crate::mount_info::FsCategory::is_virtual) {
            reason_codes.push(REASON_DISCOVERY_ROOT_REJECTED.to_string());
            return RootDiscoveryDecision {
                scope: DiscoveryScopeDecision::Exclude,
                reason_codes,
            };
        }

        if self.skip_network_mounts
            && mount_category.is_some_and(crate::mount_info::FsCategory::is_network)
        {
            reason_codes.push(REASON_DISCOVERY_ROOT_REJECTED.to_string());
            return RootDiscoveryDecision {
                scope: DiscoveryScopeDecision::Exclude,
                reason_codes,
            };
        }

        if self.matches_exclude_patterns(root, &normalized) {
            reason_codes.push(REASON_DISCOVERY_ROOT_REJECTED.to_string());
            return RootDiscoveryDecision {
                scope: DiscoveryScopeDecision::Exclude,
                reason_codes,
            };
        }

        reason_codes.push(REASON_DISCOVERY_ROOT_ACCEPTED.to_string());
        RootDiscoveryDecision {
            scope: DiscoveryScopeDecision::Include,
            reason_codes,
        }
    }

    /// Evaluate a file candidate for inclusion and ingestion class assignment.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn evaluate_candidate(&self, candidate: &DiscoveryCandidate<'_>) -> DiscoveryDecision {
        let mut reason_codes = Vec::new();

        if candidate.path.as_os_str().is_empty() {
            reason_codes.push(REASON_DISCOVERY_ROOT_REJECTED.to_string());
            reason_codes.push(REASON_DISCOVERY_FILE_EXCLUDED.to_string());
            normalize_reason_codes(&mut reason_codes);
            return DiscoveryDecision {
                scope: DiscoveryScopeDecision::Exclude,
                ingestion_class: IngestionClass::Skip,
                utility_score: i32::MIN,
                reason_codes,
            };
        }

        if candidate
            .mount_category
            .is_some_and(crate::mount_info::FsCategory::is_virtual)
        {
            reason_codes.push(REASON_DISCOVERY_ROOT_REJECTED.to_string());
            reason_codes.push(REASON_DISCOVERY_FILE_EXCLUDED.to_string());
            normalize_reason_codes(&mut reason_codes);
            return DiscoveryDecision {
                scope: DiscoveryScopeDecision::Exclude,
                ingestion_class: IngestionClass::Skip,
                utility_score: i32::MIN,
                reason_codes,
            };
        }

        if self.skip_network_mounts
            && candidate
                .mount_category
                .is_some_and(crate::mount_info::FsCategory::is_network)
        {
            reason_codes.push(REASON_DISCOVERY_ROOT_REJECTED.to_string());
            reason_codes.push(REASON_DISCOVERY_FILE_EXCLUDED.to_string());
            normalize_reason_codes(&mut reason_codes);
            return DiscoveryDecision {
                scope: DiscoveryScopeDecision::Exclude,
                ingestion_class: IngestionClass::Skip,
                utility_score: i32::MIN,
                reason_codes,
            };
        }

        reason_codes.push(REASON_DISCOVERY_ROOT_ACCEPTED.to_string());
        let normalized = normalize_path(candidate.path);

        if self.matches_exclude_patterns(candidate.path, &normalized) {
            reason_codes.push(REASON_DISCOVERY_FILE_EXCLUDED.to_string());
            normalize_reason_codes(&mut reason_codes);
            return DiscoveryDecision {
                scope: DiscoveryScopeDecision::Exclude,
                ingestion_class: IngestionClass::Skip,
                utility_score: i32::MIN,
                reason_codes,
            };
        }

        if candidate.is_symlink && !self.follow_symlinks {
            reason_codes.push(REASON_DISCOVERY_FILE_EXCLUDED.to_string());
            normalize_reason_codes(&mut reason_codes);
            return DiscoveryDecision {
                scope: DiscoveryScopeDecision::Exclude,
                ingestion_class: IngestionClass::Skip,
                utility_score: i32::MIN,
                reason_codes,
            };
        }

        let extension = lower_extension(candidate.path);
        if extension
            .as_deref()
            .is_some_and(|ext| self.binary_blocklist_contains(ext))
        {
            reason_codes.push(REASON_DISCOVERY_FILE_BINARY_BLOCKED.to_string());
            normalize_reason_codes(&mut reason_codes);
            return DiscoveryDecision {
                scope: DiscoveryScopeDecision::Exclude,
                ingestion_class: IngestionClass::Skip,
                utility_score: i32::MIN,
                reason_codes,
            };
        }

        let filename = lower_filename(candidate.path);
        let max_bytes = self.max_file_size_mb.saturating_mul(1024 * 1024) as u64;
        let mut utility_score = 50_i32;

        if candidate.byte_len > max_bytes {
            utility_score -= 20;
            reason_codes.push(REASON_DISCOVERY_FILE_TOO_LARGE.to_string());
        }

        if candidate.byte_len > max_bytes.saturating_mul(4) {
            reason_codes.push(REASON_DISCOVERY_FILE_TOO_LARGE.to_string());
            reason_codes.push(REASON_DISCOVERY_FILE_EXCLUDED.to_string());
            normalize_reason_codes(&mut reason_codes);
            return DiscoveryDecision {
                scope: DiscoveryScopeDecision::Exclude,
                ingestion_class: IngestionClass::Skip,
                utility_score,
                reason_codes,
            };
        }

        if candidate
            .mount_category
            .is_some_and(crate::mount_info::FsCategory::is_network)
        {
            utility_score -= 10;
        }

        if has_low_utility_component(candidate.path) {
            utility_score -= 30;
        }

        if filename
            .as_deref()
            .is_some_and(|value| HIGH_SIGNAL_FILENAMES.contains(&value))
        {
            utility_score += 20;
        }

        if filename
            .as_deref()
            .is_some_and(|value| LOW_UTILITY_FILENAMES.contains(&value))
        {
            utility_score -= 20;
        }

        if filename
            .as_deref()
            .is_some_and(is_generated_or_minified_filename)
        {
            utility_score -= 25;
        }

        match extension.as_deref() {
            Some(ext) if HIGH_UTILITY_EXTENSIONS.contains(&ext) => {
                utility_score += 30;
            }
            Some(ext) if is_low_value_extension(ext) => {
                utility_score -= 15;
            }
            None => {
                utility_score -= 5;
            }
            _ => {}
        }

        if self.text_selection_mode == TextSelectionMode::Allowlist
            && extension
                .as_deref()
                .is_none_or(|ext| !TEXT_ALLOWLIST_EXTENSIONS.contains(&ext))
        {
            utility_score -= 35;
        }

        let mut ingestion_class = if utility_score >= 70 {
            IngestionClass::FullSemanticLexical
        } else if utility_score >= 45 {
            IngestionClass::LexicalOnly
        } else if utility_score >= 20 {
            IngestionClass::MetadataOnly
        } else {
            IngestionClass::Skip
        };

        if candidate.byte_len > max_bytes && ingestion_class == IngestionClass::FullSemanticLexical
        {
            ingestion_class = IngestionClass::LexicalOnly;
        }

        if candidate.byte_len > max_bytes.saturating_mul(2)
            && ingestion_class == IngestionClass::LexicalOnly
        {
            ingestion_class = IngestionClass::MetadataOnly;
        }

        let scope = if ingestion_class.is_indexed() {
            reason_codes.push(REASON_DISCOVERY_FILE_INCLUDED.to_string());
            DiscoveryScopeDecision::Include
        } else {
            reason_codes.push(REASON_DISCOVERY_FILE_EXCLUDED.to_string());
            DiscoveryScopeDecision::Exclude
        };

        normalize_reason_codes(&mut reason_codes);
        DiscoveryDecision {
            scope,
            ingestion_class,
            utility_score,
            reason_codes,
        }
    }

    fn matches_exclude_patterns(&self, path: &Path, normalized_path: &str) -> bool {
        let components = normalized_components(path);
        self.exclude_patterns.iter().any(|pattern| {
            path_matches_pattern(
                &pattern.replace('\\', "/").to_ascii_lowercase(),
                normalized_path,
                &components,
            )
        })
    }

    fn binary_blocklist_contains(&self, extension: &str) -> bool {
        self.binary_blocklist_extensions.iter().any(|blocked| {
            blocked
                .trim_start_matches('.')
                .eq_ignore_ascii_case(extension)
        })
    }
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            roots: vec![".".into()],
            exclude_patterns: vec![
                ".git".into(),
                "node_modules".into(),
                "target".into(),
                "__pycache__".into(),
                ".venv".into(),
                "vendor".into(),
                "dist".into(),
                "build".into(),
                ".next".into(),
            ],
            text_selection_mode: TextSelectionMode::Blocklist,
            binary_blocklist_extensions: vec![
                ".exe".into(),
                ".dll".into(),
                ".so".into(),
                ".o".into(),
                ".class".into(),
                ".jar".into(),
                ".zip".into(),
                ".tar".into(),
                ".gz".into(),
                ".png".into(),
                ".jpg".into(),
                ".jpeg".into(),
                ".mp3".into(),
                ".mp4".into(),
                ".wasm".into(),
                ".pyc".into(),
                ".pdb".into(),
            ],
            max_file_size_mb: 10,
            follow_symlinks: false,
            mount_overrides: Vec::new(),
            skip_network_mounts: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexingConfig {
    pub fast_model: String,
    pub quality_model: String,
    pub model_dir: String,
    pub embedding_batch_size: usize,
    pub reindex_on_change: bool,
    pub watch_mode: bool,
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            fast_model: "potion-multilingual-128M".into(),
            quality_model: "all-MiniLM-L6-v2".into(),
            model_dir: "~/.local/share/frankensearch/models".into(),
            embedding_batch_size: 64,
            reindex_on_change: true,
            watch_mode: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchConfig {
    pub default_limit: usize,
    pub quality_weight: f64,
    pub rrf_k: f64,
    pub quality_timeout_ms: u64,
    pub fast_only: bool,
    pub explain: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_limit: 10,
            quality_weight: 0.7,
            rrf_k: 60.0,
            quality_timeout_ms: 500,
            fast_only: false,
            explain: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PressureConfig {
    pub profile: PressureProfile,
    pub cpu_ceiling_pct: u8,
    pub memory_ceiling_mb: usize,
    pub sample_interval_ms: u64,
    pub ewma_alpha_per_mille: u16,
    pub anti_flap_readings: u8,
    pub io_ceiling_bytes_per_sec: u64,
    pub load_ceiling_per_mille: u16,
    pub degradation_override: DegradationOverrideMode,
    pub hard_pause_requested: bool,
    pub quality_circuit_open: bool,
}

impl Default for PressureConfig {
    fn default() -> Self {
        Self {
            profile: PressureProfile::Performance,
            cpu_ceiling_pct: 80,
            memory_ceiling_mb: 2048,
            sample_interval_ms: 2_000,
            ewma_alpha_per_mille: 300,
            anti_flap_readings: 3,
            io_ceiling_bytes_per_sec: 100 * 1024 * 1024,
            load_ceiling_per_mille: 3_000,
            degradation_override: DegradationOverrideMode::Auto,
            hard_pause_requested: false,
            quality_circuit_open: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TuiConfig {
    pub theme: TuiTheme,
    pub frame_budget_ms: u16,
    pub show_explanations: bool,
    pub density: Density,
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            theme: TuiTheme::Dark,
            frame_budget_ms: 16,
            show_explanations: true,
            density: Density::Normal,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StorageConfig {
    pub index_dir: String,
    pub db_path: String,
    pub evidence_retention_days: u16,
    pub summary_retention_days: u16,
    pub disk_budget_bytes: Option<u64>,
    pub storage_pressure_emergency: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            index_dir: ".frankensearch".into(),
            db_path: "~/.local/share/fsfs/fsfs.db".into(),
            evidence_retention_days: 7,
            summary_retention_days: 90,
            disk_budget_bytes: None,
            storage_pressure_emergency: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrivacyConfig {
    pub redact_file_contents_in_logs: bool,
    pub redact_paths_in_telemetry: bool,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            redact_file_contents_in_logs: true,
            redact_paths_in_telemetry: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct FsfsConfig {
    pub discovery: DiscoveryConfig,
    pub indexing: IndexingConfig,
    pub search: SearchConfig,
    pub pressure: PressureConfig,
    pub tui: TuiConfig,
    pub storage: StorageConfig,
    pub privacy: PrivacyConfig,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
struct DiscoveryConfigPatch {
    roots: Option<Vec<String>>,
    exclude_patterns: Option<Vec<String>>,
    text_selection_mode: Option<TextSelectionMode>,
    binary_blocklist_extensions: Option<Vec<String>>,
    max_file_size_mb: Option<usize>,
    follow_symlinks: Option<bool>,
    mount_overrides: Option<Vec<MountPolicyEntry>>,
    skip_network_mounts: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
struct IndexingConfigPatch {
    fast_model: Option<String>,
    quality_model: Option<String>,
    model_dir: Option<String>,
    embedding_batch_size: Option<usize>,
    reindex_on_change: Option<bool>,
    watch_mode: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
struct SearchConfigPatch {
    default_limit: Option<usize>,
    quality_weight: Option<f64>,
    rrf_k: Option<f64>,
    quality_timeout_ms: Option<u64>,
    fast_only: Option<bool>,
    explain: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
struct PressureConfigPatch {
    profile: Option<PressureProfile>,
    cpu_ceiling_pct: Option<u8>,
    memory_ceiling_mb: Option<usize>,
    sample_interval_ms: Option<u64>,
    ewma_alpha_per_mille: Option<u16>,
    anti_flap_readings: Option<u8>,
    io_ceiling_bytes_per_sec: Option<u64>,
    load_ceiling_per_mille: Option<u16>,
    degradation_override: Option<DegradationOverrideMode>,
    hard_pause_requested: Option<bool>,
    quality_circuit_open: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
struct TuiConfigPatch {
    theme: Option<TuiTheme>,
    frame_budget_ms: Option<u16>,
    show_explanations: Option<bool>,
    density: Option<Density>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
struct StorageConfigPatch {
    index_dir: Option<String>,
    db_path: Option<String>,
    evidence_retention_days: Option<u16>,
    summary_retention_days: Option<u16>,
    disk_budget_bytes: Option<u64>,
    storage_pressure_emergency: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
struct PrivacyConfigPatch {
    redact_file_contents_in_logs: Option<bool>,
    redact_paths_in_telemetry: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
struct FsfsConfigPatch {
    discovery: Option<DiscoveryConfigPatch>,
    indexing: Option<IndexingConfigPatch>,
    search: Option<SearchConfigPatch>,
    pressure: Option<PressureConfigPatch>,
    tui: Option<TuiConfigPatch>,
    storage: Option<StorageConfigPatch>,
    privacy: Option<PrivacyConfigPatch>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigSource {
    Cli,
    Env,
    File,
    Defaults,
    Runtime,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConfigWarning {
    pub reason_code: String,
    pub field: String,
    pub source: ConfigSource,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PathExpansion {
    pub field: String,
    pub raw: String,
    pub expanded: String,
    pub source: ConfigSource,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConfigLoadResult {
    pub config: FsfsConfig,
    pub source_precedence: [ConfigSource; 4],
    pub config_file_used: Option<PathBuf>,
    pub cli_flags_used: Vec<String>,
    pub env_keys_used: Vec<String>,
    pub pressure_profile_resolution: PressureProfileResolution,
    pub warnings: Vec<ConfigWarning>,
    pub path_expansions: Vec<PathExpansion>,
}

impl ConfigLoadResult {
    #[must_use]
    pub fn to_loaded_event(&self) -> ConfigLoadedEvent {
        let mut reason_codes: Vec<String> = self
            .warnings
            .iter()
            .map(|warning| warning.reason_code.clone())
            .collect();
        reason_codes.push(
            self.pressure_profile_resolution
                .diagnostics
                .reason_code
                .clone(),
        );
        reason_codes.extend(
            self.pressure_profile_resolution
                .overrides
                .iter()
                .filter(|decision| !decision.applied)
                .map(|decision| decision.reason_code.clone()),
        );
        reason_codes.extend(
            self.pressure_profile_resolution
                .safety_clamps
                .iter()
                .map(|clamp| clamp.reason_code.clone()),
        );
        reason_codes.sort_unstable();
        reason_codes.dedup();

        ConfigLoadedEvent {
            event: "config_loaded".into(),
            source_precedence_applied: PRECEDENCE,
            config_file_used: self.config_file_used.clone(),
            cli_flags_used: self.cli_flags_used.clone(),
            env_keys_used: self.env_keys_used.clone(),
            pressure_profile_resolution: self.pressure_profile_resolution.clone(),
            resolved_values: self.config.clone(),
            warnings: self.warnings.clone(),
            reason_codes,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfigLoadedEvent {
    pub event: String,
    pub source_precedence_applied: [ConfigSource; 4],
    pub config_file_used: Option<PathBuf>,
    pub cli_flags_used: Vec<String>,
    pub env_keys_used: Vec<String>,
    pub pressure_profile_resolution: PressureProfileResolution,
    pub resolved_values: FsfsConfig,
    pub warnings: Vec<ConfigWarning>,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CliOverrides {
    pub roots: Option<Vec<String>>,
    pub exclude_patterns: Option<Vec<String>>,
    pub limit: Option<usize>,
    pub fast_only: Option<bool>,
    pub allow_background_indexing: Option<bool>,
    pub explain: Option<bool>,
    pub profile: Option<PressureProfile>,
    pub degradation_override: Option<DegradationOverrideMode>,
    pub hard_pause_requested: Option<bool>,
    pub quality_circuit_open: Option<bool>,
    pub theme: Option<TuiTheme>,
    pub config_path: Option<PathBuf>,
}

impl CliOverrides {
    #[must_use]
    pub fn used_flags(&self) -> Vec<String> {
        let mut flags = Vec::new();
        if self.roots.is_some() {
            flags.push("--roots".into());
        }
        if self.exclude_patterns.is_some() {
            flags.push("--exclude".into());
        }
        if self.limit.is_some() {
            flags.push("--limit".into());
        }
        if self.fast_only.is_some() {
            flags.push("--fast-only".into());
        }
        if self.allow_background_indexing.is_some() {
            flags.push("--watch-mode".into());
        }
        if self.explain.is_some() {
            flags.push("--explain".into());
        }
        if self.profile.is_some() {
            flags.push("--profile".into());
        }
        if self.degradation_override.is_some() {
            flags.push("--degradation-override".into());
        }
        if self.hard_pause_requested.is_some() {
            flags.push("--hard-pause".into());
        }
        if self.quality_circuit_open.is_some() {
            flags.push("--quality-circuit-open".into());
        }
        if self.theme.is_some() {
            flags.push("--theme".into());
        }
        if self.config_path.is_some() {
            flags.push("--config".into());
        }
        flags
    }
}

#[must_use]
pub fn default_user_config_file_path(home_dir: &Path) -> PathBuf {
    if let Some(xdg_config_home) = std::env::var_os("XDG_CONFIG_HOME") {
        return PathBuf::from(xdg_config_home)
            .join("frankensearch")
            .join("config.toml");
    }

    home_dir
        .join(".config")
        .join("frankensearch")
        .join("config.toml")
}

#[must_use]
pub fn default_project_config_file_path(cwd: &Path) -> PathBuf {
    project_root_from_cwd(cwd)
        .join(".frankensearch")
        .join("config.toml")
}

#[must_use]
pub fn default_config_file_path(home_dir: &Path) -> PathBuf {
    default_user_config_file_path(home_dir)
}

#[must_use]
fn expand_home_prefix(path: &Path, home_dir: &Path) -> PathBuf {
    let mut components = path.components();
    let Some(Component::Normal(first_segment)) = components.next() else {
        return path.to_path_buf();
    };

    if first_segment != OsStr::new("~") {
        return path.to_path_buf();
    }

    let mut expanded = home_dir.to_path_buf();
    for segment in components {
        match segment {
            Component::Normal(part) => expanded.push(part),
            _ => return path.to_path_buf(),
        }
    }

    expanded
}

#[must_use]
fn project_root_from_cwd(cwd: &Path) -> PathBuf {
    cwd.ancestors()
        .find(|candidate| candidate.join(".git").exists())
        .map_or_else(|| cwd.to_path_buf(), Path::to_path_buf)
}

/// Load config from file/env/CLI overlays using the fsfs precedence contract.
///
/// # Errors
///
/// Returns `SearchError::InvalidConfig` for parse/validation failures and
/// `SearchError::Io` if reading a present file fails.
pub fn load_from_sources<S>(
    config_file: Option<&Path>,
    env: &HashMap<String, String, S>,
    cli: &CliOverrides,
    home_dir: &Path,
) -> SearchResult<ConfigLoadResult>
where
    S: BuildHasher,
{
    let expanded_config_file = config_file.map(|path| expand_home_prefix(path, home_dir));
    let (toml_contents, config_file_used) = match expanded_config_file {
        Some(path) if path.exists() => (Some(fs::read_to_string(&path)?), Some(path)),
        Some(_) | None => (None, None),
    };

    load_from_str(
        toml_contents.as_deref(),
        config_file_used.as_deref(),
        env,
        cli,
        home_dir,
    )
}

/// Load config with layered file precedence (`project > user`), then env and
/// CLI overlays.
///
/// # Errors
///
/// Returns `SearchError::InvalidConfig` for parse/validation failures and
/// `SearchError::Io` if reading a present file fails.
pub fn load_from_layered_sources<S>(
    project_config_file: Option<&Path>,
    user_config_file: Option<&Path>,
    env: &HashMap<String, String, S>,
    cli: &CliOverrides,
    home_dir: &Path,
) -> SearchResult<ConfigLoadResult>
where
    S: BuildHasher,
{
    let expanded_user_config = user_config_file.map(|path| expand_home_prefix(path, home_dir));
    let expanded_project_config =
        project_config_file.map(|path| expand_home_prefix(path, home_dir));

    let (user_toml, user_config_used) = match expanded_user_config {
        Some(path) if path.exists() => (Some(fs::read_to_string(&path)?), Some(path)),
        Some(_) | None => (None, None),
    };
    let (project_toml, project_config_used) = match expanded_project_config {
        Some(path) if path.exists() => (Some(fs::read_to_string(&path)?), Some(path)),
        Some(_) | None => (None, None),
    };

    load_from_str_layers(
        user_toml.as_deref(),
        user_config_used.as_deref(),
        project_toml.as_deref(),
        project_config_used.as_deref(),
        env,
        cli,
        home_dir,
    )
}

/// Load config from raw TOML/env/CLI overlays using the fsfs precedence
/// contract (`CLI > env > file > defaults`).
///
/// # Errors
///
/// Returns `SearchError::InvalidConfig` when parsing/validation fails.
pub fn load_from_str<S>(
    config_toml: Option<&str>,
    config_file_path: Option<&Path>,
    env: &HashMap<String, String, S>,
    cli: &CliOverrides,
    home_dir: &Path,
) -> SearchResult<ConfigLoadResult>
where
    S: BuildHasher,
{
    load_from_str_layers(
        config_toml,
        config_file_path,
        None,
        None,
        env,
        cli,
        home_dir,
    )
}

const fn merge_profile_overrides(
    base: &mut ProfileSourceOverrides,
    overlay: ProfileSourceOverrides,
) {
    if overlay.quality_enabled.is_some() {
        base.quality_enabled = overlay.quality_enabled;
    }
    if overlay.allow_background_indexing.is_some() {
        base.allow_background_indexing = overlay.allow_background_indexing;
    }
}

fn apply_file_patch(
    config: &mut FsfsConfig,
    warnings: &mut Vec<ConfigWarning>,
    file_profile_overrides: &mut ProfileSourceOverrides,
    config_toml: &str,
) -> SearchResult<()> {
    warnings.extend(collect_unknown_key_warnings(config_toml)?);
    let patch: FsfsConfigPatch =
        toml::from_str(config_toml).map_err(|error| SearchError::InvalidConfig {
            field: "config_file".into(),
            value: "<toml>".into(),
            reason: error.to_string(),
        })?;
    merge_profile_overrides(
        file_profile_overrides,
        profile_overrides_from_file_patch(&patch),
    );
    apply_patch(config, patch);
    Ok(())
}

fn load_from_str_layers<S>(
    user_config_toml: Option<&str>,
    user_config_path: Option<&Path>,
    project_config_toml: Option<&str>,
    project_config_path: Option<&Path>,
    env: &HashMap<String, String, S>,
    cli: &CliOverrides,
    home_dir: &Path,
) -> SearchResult<ConfigLoadResult>
where
    S: BuildHasher,
{
    let mut config = FsfsConfig::default();
    let mut warnings = Vec::new();
    let mut file_profile_overrides = ProfileSourceOverrides::default();

    if let Some(config_toml) = user_config_toml {
        apply_file_patch(
            &mut config,
            &mut warnings,
            &mut file_profile_overrides,
            config_toml,
        )?;
    }

    if let Some(config_toml) = project_config_toml {
        apply_file_patch(
            &mut config,
            &mut warnings,
            &mut file_profile_overrides,
            config_toml,
        )?;
    }

    let env_report = apply_env_overrides(&mut config, env)?;
    let cli_profile_overrides = apply_cli_overrides(&mut config, cli);
    let path_expansions = expand_tilde_paths(&mut config, home_dir);
    validate_config(&config, &mut warnings)?;

    let pressure_profile_resolution = resolve_pressure_profile(
        &mut config,
        file_profile_overrides,
        env_report.profile_overrides,
        cli_profile_overrides,
        &mut warnings,
    );

    Ok(ConfigLoadResult {
        config,
        source_precedence: PRECEDENCE,
        config_file_used: project_config_path
            .or(user_config_path)
            .map(Path::to_path_buf),
        cli_flags_used: cli.used_flags(),
        env_keys_used: env_report.keys_used,
        pressure_profile_resolution,
        warnings,
        path_expansions,
    })
}

pub fn emit_config_loaded(event: &ConfigLoadedEvent) {
    info!(
        event = %event.event,
        precedence = ?event.source_precedence_applied,
        config_file_used = ?event.config_file_used,
        cli_flags_used = ?event.cli_flags_used,
        env_keys_used = ?event.env_keys_used,
        pressure_profile = ?event.pressure_profile_resolution.selected_profile,
        profile_conflict = event.pressure_profile_resolution.conflict_detected,
        profile_reason_code = %event.pressure_profile_resolution.diagnostics.reason_code,
        profile_version = event.pressure_profile_resolution.diagnostics.effective_profile_version,
        reason_codes = ?event.reason_codes,
        "fsfs configuration loaded"
    );
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ProfileSourceOverrides {
    quality_enabled: Option<bool>,
    allow_background_indexing: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct EnvOverrideReport {
    keys_used: Vec<String>,
    profile_overrides: ProfileSourceOverrides,
}

#[must_use]
fn profile_overrides_from_file_patch(patch: &FsfsConfigPatch) -> ProfileSourceOverrides {
    ProfileSourceOverrides {
        quality_enabled: patch
            .search
            .as_ref()
            .and_then(|search| search.fast_only.map(|fast_only| !fast_only)),
        allow_background_indexing: patch
            .indexing
            .as_ref()
            .and_then(|indexing| indexing.watch_mode),
    }
}

#[allow(clippy::too_many_lines)]
fn resolve_pressure_profile(
    config: &mut FsfsConfig,
    file_overrides: ProfileSourceOverrides,
    env_overrides: ProfileSourceOverrides,
    cli_overrides: ProfileSourceOverrides,
    warnings: &mut Vec<ConfigWarning>,
) -> PressureProfileResolution {
    let contract = config.pressure.profile.contract();
    let mut effective = contract.to_effective_settings();
    let mut overrides = Vec::new();
    let mut safety_clamps = Vec::new();
    let mut conflict_detected = false;

    apply_profile_bool_override(
        config.pressure.profile,
        PressureProfileField::QualityEnabled,
        ProfileOverrideSource::Config,
        file_overrides.quality_enabled,
        &mut effective.quality_enabled,
        contract,
        &mut overrides,
        &mut conflict_detected,
        warnings,
    );
    apply_profile_bool_override(
        config.pressure.profile,
        PressureProfileField::QualityEnabled,
        ProfileOverrideSource::Env,
        env_overrides.quality_enabled,
        &mut effective.quality_enabled,
        contract,
        &mut overrides,
        &mut conflict_detected,
        warnings,
    );
    apply_profile_bool_override(
        config.pressure.profile,
        PressureProfileField::QualityEnabled,
        ProfileOverrideSource::Cli,
        cli_overrides.quality_enabled,
        &mut effective.quality_enabled,
        contract,
        &mut overrides,
        &mut conflict_detected,
        warnings,
    );

    apply_profile_bool_override(
        config.pressure.profile,
        PressureProfileField::AllowBackgroundIndexing,
        ProfileOverrideSource::Config,
        file_overrides.allow_background_indexing,
        &mut effective.allow_background_indexing,
        contract,
        &mut overrides,
        &mut conflict_detected,
        warnings,
    );
    apply_profile_bool_override(
        config.pressure.profile,
        PressureProfileField::AllowBackgroundIndexing,
        ProfileOverrideSource::Env,
        env_overrides.allow_background_indexing,
        &mut effective.allow_background_indexing,
        contract,
        &mut overrides,
        &mut conflict_detected,
        warnings,
    );
    apply_profile_bool_override(
        config.pressure.profile,
        PressureProfileField::AllowBackgroundIndexing,
        ProfileOverrideSource::Cli,
        cli_overrides.allow_background_indexing,
        &mut effective.allow_background_indexing,
        contract,
        &mut overrides,
        &mut conflict_detected,
        warnings,
    );

    if config.pressure.hard_pause_requested {
        if effective.quality_enabled {
            effective.quality_enabled = false;
            safety_clamps.push(PressureProfileSafetyClamp {
                field: PressureProfileField::QualityEnabled,
                clamped_to: "false".into(),
                reason_code: "safety.clamp.hard_pause.quality_enabled".into(),
            });
        }
        if effective.allow_background_indexing {
            effective.allow_background_indexing = false;
            safety_clamps.push(PressureProfileSafetyClamp {
                field: PressureProfileField::AllowBackgroundIndexing,
                clamped_to: "false".into(),
                reason_code: "safety.clamp.hard_pause.allow_background_indexing".into(),
            });
        }
    }

    config.search.fast_only = !effective.quality_enabled;
    config.indexing.watch_mode = effective.allow_background_indexing;

    let diagnostics_reason_code = if conflict_detected {
        "profile.resolution.conflict"
    } else {
        "profile.resolution.ok"
    };

    PressureProfileResolution {
        selected_profile: config.pressure.profile,
        overrides,
        effective,
        safety_clamps,
        conflict_detected,
        diagnostics: PressureProfileResolutionDiagnostics {
            event: "profile_resolution_completed".into(),
            precedence_chain: PROFILE_PRECEDENCE_CHAIN
                .iter()
                .map(|value| (*value).to_owned())
                .collect(),
            reason_code: diagnostics_reason_code.into(),
            effective_profile_version: PRESSURE_PROFILE_VERSION,
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_profile_bool_override(
    profile: PressureProfile,
    field: PressureProfileField,
    source: ProfileOverrideSource,
    requested: Option<bool>,
    effective_value: &mut bool,
    contract: PressureProfileContract,
    overrides: &mut Vec<PressureProfileOverrideDecision>,
    conflict_detected: &mut bool,
    warnings: &mut Vec<ConfigWarning>,
) {
    let Some(requested) = requested else {
        return;
    };

    let locked = contract.is_locked_field(field);
    if locked {
        overrides.push(PressureProfileOverrideDecision {
            field,
            source,
            requested_value: requested.to_string(),
            applied: false,
            reason_code: "override.rejected.locked_field".into(),
        });
        if requested != *effective_value {
            *conflict_detected = true;
            warnings.push(ConfigWarning {
                reason_code: "override.rejected.locked_field".into(),
                field: profile_field_path(field).into(),
                source: config_source_for_override(source),
                message: format!(
                    "profile {:?} locks {}; requested {} from {:?} rejected",
                    profile,
                    profile_field_path(field),
                    requested,
                    source
                ),
            });
        }
        return;
    }

    *effective_value = requested;
    overrides.push(PressureProfileOverrideDecision {
        field,
        source,
        requested_value: requested.to_string(),
        applied: true,
        reason_code: format!("override.applied.{}_field", override_source_label(source)),
    });
}

#[must_use]
const fn override_source_label(source: ProfileOverrideSource) -> &'static str {
    match source {
        ProfileOverrideSource::Cli => "cli",
        ProfileOverrideSource::Env => "env",
        ProfileOverrideSource::Config => "config",
    }
}

#[must_use]
const fn config_source_for_override(source: ProfileOverrideSource) -> ConfigSource {
    match source {
        ProfileOverrideSource::Cli => ConfigSource::Cli,
        ProfileOverrideSource::Env => ConfigSource::Env,
        ProfileOverrideSource::Config => ConfigSource::File,
    }
}

#[must_use]
const fn profile_field_path(field: PressureProfileField) -> &'static str {
    match field {
        PressureProfileField::SchedulerMode => "pressure.profile.scheduler_mode",
        PressureProfileField::MaxEmbedConcurrency => "pressure.profile.max_embed_concurrency",
        PressureProfileField::MaxIndexConcurrency => "pressure.profile.max_index_concurrency",
        PressureProfileField::QualityEnabled => "pressure.profile.quality_enabled",
        PressureProfileField::AllowBackgroundIndexing => {
            "pressure.profile.allow_background_indexing"
        }
    }
}

#[allow(clippy::too_many_lines)]
fn apply_patch(config: &mut FsfsConfig, patch: FsfsConfigPatch) {
    if let Some(discovery) = patch.discovery {
        if let Some(roots) = discovery.roots {
            config.discovery.roots = roots;
        }
        if let Some(exclude_patterns) = discovery.exclude_patterns {
            config.discovery.exclude_patterns = exclude_patterns;
        }
        if let Some(text_selection_mode) = discovery.text_selection_mode {
            config.discovery.text_selection_mode = text_selection_mode;
        }
        if let Some(binary_blocklist_extensions) = discovery.binary_blocklist_extensions {
            config.discovery.binary_blocklist_extensions = binary_blocklist_extensions;
        }
        if let Some(max_file_size_mb) = discovery.max_file_size_mb {
            config.discovery.max_file_size_mb = max_file_size_mb;
        }
        if let Some(follow_symlinks) = discovery.follow_symlinks {
            config.discovery.follow_symlinks = follow_symlinks;
        }
        if let Some(mount_overrides) = discovery.mount_overrides {
            config.discovery.mount_overrides = mount_overrides;
        }
        if let Some(skip_network_mounts) = discovery.skip_network_mounts {
            config.discovery.skip_network_mounts = skip_network_mounts;
        }
    }

    if let Some(indexing) = patch.indexing {
        if let Some(fast_model) = indexing.fast_model {
            config.indexing.fast_model = fast_model;
        }
        if let Some(quality_model) = indexing.quality_model {
            config.indexing.quality_model = quality_model;
        }
        if let Some(model_dir) = indexing.model_dir {
            config.indexing.model_dir = model_dir;
        }
        if let Some(embedding_batch_size) = indexing.embedding_batch_size {
            config.indexing.embedding_batch_size = embedding_batch_size;
        }
        if let Some(reindex_on_change) = indexing.reindex_on_change {
            config.indexing.reindex_on_change = reindex_on_change;
        }
        if let Some(watch_mode) = indexing.watch_mode {
            config.indexing.watch_mode = watch_mode;
        }
    }

    if let Some(search) = patch.search {
        if let Some(default_limit) = search.default_limit {
            config.search.default_limit = default_limit;
        }
        if let Some(quality_weight) = search.quality_weight {
            config.search.quality_weight = quality_weight;
        }
        if let Some(rrf_k) = search.rrf_k {
            config.search.rrf_k = rrf_k;
        }
        if let Some(quality_timeout_ms) = search.quality_timeout_ms {
            config.search.quality_timeout_ms = quality_timeout_ms;
        }
        if let Some(fast_only) = search.fast_only {
            config.search.fast_only = fast_only;
        }
        if let Some(explain) = search.explain {
            config.search.explain = explain;
        }
    }

    if let Some(pressure) = patch.pressure {
        if let Some(profile) = pressure.profile {
            config.pressure.profile = profile;
        }
        if let Some(cpu_ceiling_pct) = pressure.cpu_ceiling_pct {
            config.pressure.cpu_ceiling_pct = cpu_ceiling_pct;
        }
        if let Some(memory_ceiling_mb) = pressure.memory_ceiling_mb {
            config.pressure.memory_ceiling_mb = memory_ceiling_mb;
        }
        if let Some(sample_interval_ms) = pressure.sample_interval_ms {
            config.pressure.sample_interval_ms = sample_interval_ms;
        }
        if let Some(ewma_alpha_per_mille) = pressure.ewma_alpha_per_mille {
            config.pressure.ewma_alpha_per_mille = ewma_alpha_per_mille;
        }
        if let Some(anti_flap_readings) = pressure.anti_flap_readings {
            config.pressure.anti_flap_readings = anti_flap_readings;
        }
        if let Some(io_ceiling_bytes_per_sec) = pressure.io_ceiling_bytes_per_sec {
            config.pressure.io_ceiling_bytes_per_sec = io_ceiling_bytes_per_sec;
        }
        if let Some(load_ceiling_per_mille) = pressure.load_ceiling_per_mille {
            config.pressure.load_ceiling_per_mille = load_ceiling_per_mille;
        }
        if let Some(degradation_override) = pressure.degradation_override {
            config.pressure.degradation_override = degradation_override;
        }
        if let Some(hard_pause_requested) = pressure.hard_pause_requested {
            config.pressure.hard_pause_requested = hard_pause_requested;
        }
        if let Some(quality_circuit_open) = pressure.quality_circuit_open {
            config.pressure.quality_circuit_open = quality_circuit_open;
        }
    }

    if let Some(tui) = patch.tui {
        if let Some(theme) = tui.theme {
            config.tui.theme = theme;
        }
        if let Some(frame_budget_ms) = tui.frame_budget_ms {
            config.tui.frame_budget_ms = frame_budget_ms;
        }
        if let Some(show_explanations) = tui.show_explanations {
            config.tui.show_explanations = show_explanations;
        }
        if let Some(density) = tui.density {
            config.tui.density = density;
        }
    }

    if let Some(storage) = patch.storage {
        if let Some(index_dir) = storage.index_dir {
            config.storage.index_dir = index_dir;
        }
        if let Some(db_path) = storage.db_path {
            config.storage.db_path = db_path;
        }
        if let Some(evidence_retention_days) = storage.evidence_retention_days {
            config.storage.evidence_retention_days = evidence_retention_days;
        }
        if let Some(summary_retention_days) = storage.summary_retention_days {
            config.storage.summary_retention_days = summary_retention_days;
        }
        if let Some(disk_budget_bytes) = storage.disk_budget_bytes {
            config.storage.disk_budget_bytes = Some(disk_budget_bytes);
        }
        if let Some(storage_pressure_emergency) = storage.storage_pressure_emergency {
            config.storage.storage_pressure_emergency = storage_pressure_emergency;
        }
    }

    if let Some(privacy) = patch.privacy {
        if let Some(redact_file_contents_in_logs) = privacy.redact_file_contents_in_logs {
            config.privacy.redact_file_contents_in_logs = redact_file_contents_in_logs;
        }
        if let Some(redact_paths_in_telemetry) = privacy.redact_paths_in_telemetry {
            config.privacy.redact_paths_in_telemetry = redact_paths_in_telemetry;
        }
    }
}

#[allow(clippy::too_many_lines)]
fn apply_env_overrides(
    config: &mut FsfsConfig,
    env: &HashMap<String, String, impl BuildHasher>,
) -> SearchResult<EnvOverrideReport> {
    fn env_override<'a>(
        env: &'a HashMap<String, String, impl BuildHasher>,
        canonical: &'static str,
        legacy: &'static str,
    ) -> Option<(&'static str, &'a String)> {
        env.get(canonical)
            .map(|value| (canonical, value))
            .or_else(|| env.get(legacy).map(|value| (legacy, value)))
    }

    let mut keys_used = Vec::new();
    let mut profile_overrides = ProfileSourceOverrides::default();

    if let Some((key, value)) =
        env_override(env, "FRANKENSEARCH_DISCOVERY_ROOTS", "FSFS_DISCOVERY_ROOTS")
    {
        config.discovery.roots = parse_csv(value, "discovery.roots")?;
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(
        env,
        "FRANKENSEARCH_DISCOVERY_EXCLUDE_PATTERNS",
        "FSFS_DISCOVERY_EXCLUDE_PATTERNS",
    ) {
        config.discovery.exclude_patterns = parse_csv(value, "discovery.exclude_patterns")?;
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(
        env,
        "FRANKENSEARCH_SEARCH_DEFAULT_LIMIT",
        "FSFS_SEARCH_DEFAULT_LIMIT",
    ) {
        config.search.default_limit = parse_usize(value, "search.default_limit")?;
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(
        env,
        "FRANKENSEARCH_FAST_ONLY",
        "FRANKENSEARCH_SEARCH_FAST_ONLY",
    )
    .or_else(|| env_override(env, "FSFS_FAST_ONLY", "FSFS_SEARCH_FAST_ONLY"))
    {
        let fast_only = parse_bool(value, "search.fast_only")?;
        config.search.fast_only = fast_only;
        profile_overrides.quality_enabled = Some(!fast_only);
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(
        env,
        "FRANKENSEARCH_QUALITY_WEIGHT",
        "FRANKENSEARCH_SEARCH_QUALITY_WEIGHT",
    )
    .or_else(|| env_override(env, "FSFS_QUALITY_WEIGHT", "FSFS_SEARCH_QUALITY_WEIGHT"))
    {
        config.search.quality_weight = parse_f64(value, "search.quality_weight")?;
        keys_used.push(key.into());
    }

    if let Some((key, value)) =
        env_override(env, "FRANKENSEARCH_RRF_K", "FRANKENSEARCH_SEARCH_RRF_K")
            .or_else(|| env_override(env, "FSFS_RRF_K", "FSFS_SEARCH_RRF_K"))
    {
        config.search.rrf_k = parse_f64(value, "search.rrf_k")?;
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(
        env,
        "FRANKENSEARCH_INDEXING_WATCH_MODE",
        "FSFS_INDEXING_WATCH_MODE",
    ) {
        let watch_mode = parse_bool(value, "indexing.watch_mode")?;
        config.indexing.watch_mode = watch_mode;
        profile_overrides.allow_background_indexing = Some(watch_mode);
        keys_used.push(key.into());
    }

    if let Some((key, value)) =
        env_override(env, "FRANKENSEARCH_SEARCH_EXPLAIN", "FSFS_SEARCH_EXPLAIN")
    {
        config.search.explain = parse_bool(value, "search.explain")?;
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(
        env,
        "FRANKENSEARCH_PRESSURE_PROFILE",
        "FSFS_PRESSURE_PROFILE",
    ) {
        config.pressure.profile =
            PressureProfile::from_str(value).map_err(|()| SearchError::InvalidConfig {
                field: "pressure.profile".into(),
                value: value.clone(),
                reason: "expected strict|performance|degraded".into(),
            })?;
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(
        env,
        "FRANKENSEARCH_PRESSURE_SAMPLE_INTERVAL_MS",
        "FSFS_PRESSURE_SAMPLE_INTERVAL_MS",
    ) {
        config.pressure.sample_interval_ms = parse_u64(value, "pressure.sample_interval_ms")?;
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(
        env,
        "FRANKENSEARCH_PRESSURE_EWMA_ALPHA_PER_MILLE",
        "FSFS_PRESSURE_EWMA_ALPHA_PER_MILLE",
    ) {
        config.pressure.ewma_alpha_per_mille = parse_u16(value, "pressure.ewma_alpha_per_mille")?;
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(
        env,
        "FRANKENSEARCH_PRESSURE_ANTI_FLAP_READINGS",
        "FSFS_PRESSURE_ANTI_FLAP_READINGS",
    ) {
        config.pressure.anti_flap_readings = parse_u8(value, "pressure.anti_flap_readings")?;
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(
        env,
        "FRANKENSEARCH_PRESSURE_IO_CEILING_BYTES_PER_SEC",
        "FSFS_PRESSURE_IO_CEILING_BYTES_PER_SEC",
    ) {
        config.pressure.io_ceiling_bytes_per_sec =
            parse_u64(value, "pressure.io_ceiling_bytes_per_sec")?;
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(
        env,
        "FRANKENSEARCH_PRESSURE_LOAD_CEILING_PER_MILLE",
        "FSFS_PRESSURE_LOAD_CEILING_PER_MILLE",
    ) {
        config.pressure.load_ceiling_per_mille =
            parse_u16(value, "pressure.load_ceiling_per_mille")?;
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(
        env,
        "FRANKENSEARCH_PRESSURE_DEGRADATION_OVERRIDE",
        "FSFS_PRESSURE_DEGRADATION_OVERRIDE",
    ) {
        config.pressure.degradation_override =
            DegradationOverrideMode::from_str(value).map_err(|()| SearchError::InvalidConfig {
                field: "pressure.degradation_override".into(),
                value: value.clone(),
                reason: "expected auto|full|embed_deferred|lexical_only|metadata_only|paused"
                    .into(),
            })?;
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(
        env,
        "FRANKENSEARCH_PRESSURE_HARD_PAUSE_REQUESTED",
        "FSFS_PRESSURE_HARD_PAUSE_REQUESTED",
    ) {
        config.pressure.hard_pause_requested = parse_bool(value, "pressure.hard_pause_requested")?;
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(
        env,
        "FRANKENSEARCH_PRESSURE_QUALITY_CIRCUIT_OPEN",
        "FSFS_PRESSURE_QUALITY_CIRCUIT_OPEN",
    ) {
        config.pressure.quality_circuit_open = parse_bool(value, "pressure.quality_circuit_open")?;
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(env, "FRANKENSEARCH_TUI_THEME", "FSFS_TUI_THEME") {
        config.tui.theme = TuiTheme::from_str(value).map_err(|()| SearchError::InvalidConfig {
            field: "tui.theme".into(),
            value: value.clone(),
            reason: "expected auto|light|dark".into(),
        })?;
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(
        env,
        "FRANKENSEARCH_PRIVACY_REDACT_PATHS_IN_TELEMETRY",
        "FSFS_PRIVACY_REDACT_PATHS_IN_TELEMETRY",
    ) {
        config.privacy.redact_paths_in_telemetry =
            parse_bool(value, "privacy.redact_paths_in_telemetry")?;
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(env, "FRANKENSEARCH_INDEX_DIR", "FSFS_INDEX_DIR")
        .or_else(|| {
            env_override(
                env,
                "FRANKENSEARCH_STORAGE_INDEX_DIR",
                "FSFS_STORAGE_INDEX_DIR",
            )
        })
    {
        config.storage.index_dir.clone_from(value);
        keys_used.push(key.into());
    }

    if let Some((key, value)) =
        env_override(env, "FRANKENSEARCH_STORAGE_DB_PATH", "FSFS_STORAGE_DB_PATH")
    {
        config.storage.db_path.clone_from(value);
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(
        env,
        "FRANKENSEARCH_STORAGE_DISK_BUDGET_BYTES",
        "FSFS_STORAGE_DISK_BUDGET_BYTES",
    ) {
        config.storage.disk_budget_bytes = Some(parse_u64(value, "storage.disk_budget_bytes")?);
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(
        env,
        "FRANKENSEARCH_STORAGE_PRESSURE_EMERGENCY",
        "FSFS_STORAGE_PRESSURE_EMERGENCY",
    ) {
        config.storage.storage_pressure_emergency =
            parse_bool(value, "storage.storage_pressure_emergency")?;
        keys_used.push(key.into());
    }

    if let Some((key, value)) = env_override(env, "FRANKENSEARCH_MODEL_DIR", "FSFS_MODEL_DIR") {
        config.indexing.model_dir.clone_from(value);
        keys_used.push(key.into());
    }

    Ok(EnvOverrideReport {
        keys_used,
        profile_overrides,
    })
}

fn apply_cli_overrides(config: &mut FsfsConfig, cli: &CliOverrides) -> ProfileSourceOverrides {
    if let Some(roots) = &cli.roots {
        config.discovery.roots.clone_from(roots);
    }

    if let Some(exclude_patterns) = &cli.exclude_patterns {
        config
            .discovery
            .exclude_patterns
            .clone_from(exclude_patterns);
    }

    if let Some(limit) = cli.limit {
        config.search.default_limit = limit;
    }

    if let Some(fast_only) = cli.fast_only {
        config.search.fast_only = fast_only;
    }

    if let Some(allow_background_indexing) = cli.allow_background_indexing {
        config.indexing.watch_mode = allow_background_indexing;
    }

    if let Some(explain) = cli.explain {
        config.search.explain = explain;
    }

    if let Some(profile) = cli.profile {
        config.pressure.profile = profile;
    }

    if let Some(degradation_override) = cli.degradation_override {
        config.pressure.degradation_override = degradation_override;
    }

    if let Some(hard_pause_requested) = cli.hard_pause_requested {
        config.pressure.hard_pause_requested = hard_pause_requested;
    }

    if let Some(quality_circuit_open) = cli.quality_circuit_open {
        config.pressure.quality_circuit_open = quality_circuit_open;
    }

    if let Some(theme) = cli.theme {
        config.tui.theme = theme;
    }

    ProfileSourceOverrides {
        quality_enabled: cli.fast_only.map(|fast_only| !fast_only),
        allow_background_indexing: cli.allow_background_indexing,
    }
}

#[allow(clippy::too_many_lines)]
fn collect_unknown_key_warnings(config_toml: &str) -> SearchResult<Vec<ConfigWarning>> {
    let value: toml::Value =
        toml::from_str(config_toml).map_err(|error| SearchError::InvalidConfig {
            field: "config_file".into(),
            value: "<toml>".into(),
            reason: error.to_string(),
        })?;

    let root = value.as_table().ok_or_else(|| SearchError::InvalidConfig {
        field: "config_file".into(),
        value: "<toml>".into(),
        reason: "expected table at root".into(),
    })?;

    let known_top_level: HashSet<&str> = [
        "discovery",
        "indexing",
        "search",
        "pressure",
        "tui",
        "storage",
        "privacy",
    ]
    .into_iter()
    .collect();

    let mut warnings = Vec::new();

    for (section, section_value) in root {
        if !known_top_level.contains(section.as_str()) {
            warnings.push(ConfigWarning {
                reason_code: "config.unknown_key.warning".into(),
                field: format!("config.{section}"),
                source: ConfigSource::File,
                message: format!("Unknown section {section} ignored"),
            });
            continue;
        }

        let Some(section_table) = section_value.as_table() else {
            continue;
        };

        let known_section_keys: HashSet<&str> = match section.as_str() {
            "discovery" => [
                "roots",
                "exclude_patterns",
                "text_selection_mode",
                "binary_blocklist_extensions",
                "max_file_size_mb",
                "follow_symlinks",
                "mount_overrides",
                "skip_network_mounts",
            ]
            .into_iter()
            .collect(),
            "indexing" => [
                "fast_model",
                "quality_model",
                "model_dir",
                "embedding_batch_size",
                "reindex_on_change",
                "watch_mode",
            ]
            .into_iter()
            .collect(),
            "search" => [
                "default_limit",
                "quality_weight",
                "rrf_k",
                "quality_timeout_ms",
                "fast_only",
                "explain",
            ]
            .into_iter()
            .collect(),
            "pressure" => [
                "profile",
                "cpu_ceiling_pct",
                "memory_ceiling_mb",
                "sample_interval_ms",
                "ewma_alpha_per_mille",
                "anti_flap_readings",
                "io_ceiling_bytes_per_sec",
                "load_ceiling_per_mille",
                "degradation_override",
                "hard_pause_requested",
                "quality_circuit_open",
            ]
            .into_iter()
            .collect(),
            "tui" => ["theme", "frame_budget_ms", "show_explanations", "density"]
                .into_iter()
                .collect(),
            "storage" => [
                "index_dir",
                "db_path",
                "evidence_retention_days",
                "summary_retention_days",
                "disk_budget_bytes",
                "storage_pressure_emergency",
            ]
            .into_iter()
            .collect(),
            "privacy" => ["redact_file_contents_in_logs", "redact_paths_in_telemetry"]
                .into_iter()
                .collect(),
            _ => HashSet::new(),
        };

        for key in section_table.keys() {
            if !known_section_keys.contains(key.as_str()) {
                warnings.push(ConfigWarning {
                    reason_code: "config.unknown_key.warning".into(),
                    field: format!("{section}.{key}"),
                    source: ConfigSource::File,
                    message: format!("Unknown key {section}.{key} ignored"),
                });
            }
        }
    }

    Ok(warnings)
}

fn expand_tilde_paths(config: &mut FsfsConfig, home_dir: &Path) -> Vec<PathExpansion> {
    let mut expansions = Vec::new();

    for root in &mut config.discovery.roots {
        if let Some(expanded) = expand_tilde(root, home_dir) {
            expansions.push(PathExpansion {
                field: "discovery.roots".into(),
                raw: root.clone(),
                expanded: expanded.clone(),
                source: ConfigSource::Runtime,
            });
            *root = expanded;
        }
    }

    if let Some(expanded) = expand_tilde(&config.indexing.model_dir, home_dir) {
        expansions.push(PathExpansion {
            field: "indexing.model_dir".into(),
            raw: config.indexing.model_dir.clone(),
            expanded: expanded.clone(),
            source: ConfigSource::Runtime,
        });
        config.indexing.model_dir = expanded;
    }

    if let Some(expanded) = expand_tilde(&config.storage.db_path, home_dir) {
        expansions.push(PathExpansion {
            field: "storage.db_path".into(),
            raw: config.storage.db_path.clone(),
            expanded: expanded.clone(),
            source: ConfigSource::Runtime,
        });
        config.storage.db_path = expanded;
    }

    if let Some(expanded) = expand_tilde(&config.storage.index_dir, home_dir) {
        expansions.push(PathExpansion {
            field: "storage.index_dir".into(),
            raw: config.storage.index_dir.clone(),
            expanded: expanded.clone(),
            source: ConfigSource::Runtime,
        });
        config.storage.index_dir = expanded;
    }

    expansions
}

fn expand_tilde(value: &str, home_dir: &Path) -> Option<String> {
    if value == "~" {
        return Some(home_dir.to_string_lossy().into_owned());
    }

    value
        .strip_prefix("~/")
        .map(|rest| home_dir.join(rest).to_string_lossy().into_owned())
}

#[allow(clippy::too_many_lines)]
fn validate_config(config: &FsfsConfig, warnings: &mut Vec<ConfigWarning>) -> SearchResult<()> {
    if !(1_usize..=1024_usize).contains(&config.discovery.max_file_size_mb) {
        return Err(SearchError::InvalidConfig {
            field: "discovery.max_file_size_mb".into(),
            value: config.discovery.max_file_size_mb.to_string(),
            reason: "must be between 1 and 1024".into(),
        });
    }

    if !(1_usize..=4096_usize).contains(&config.indexing.embedding_batch_size) {
        return Err(SearchError::InvalidConfig {
            field: "indexing.embedding_batch_size".into(),
            value: config.indexing.embedding_batch_size.to_string(),
            reason: "must be between 1 and 4096".into(),
        });
    }

    if !(1_usize..=200_usize).contains(&config.search.default_limit) {
        return Err(SearchError::InvalidConfig {
            field: "search.default_limit".into(),
            value: config.search.default_limit.to_string(),
            reason: "must be between 1 and 200".into(),
        });
    }

    if config.storage.summary_retention_days < config.storage.evidence_retention_days {
        return Err(SearchError::InvalidConfig {
            field: "storage.summary_retention_days".into(),
            value: config.storage.summary_retention_days.to_string(),
            reason: "must be >= storage.evidence_retention_days".into(),
        });
    }

    if !(0.0..=1.0).contains(&config.search.quality_weight) {
        return Err(SearchError::InvalidConfig {
            field: "search.quality_weight".into(),
            value: config.search.quality_weight.to_string(),
            reason: "must be between 0.0 and 1.0".into(),
        });
    }

    if !config.search.rrf_k.is_finite() || config.search.rrf_k < 1.0 {
        return Err(SearchError::InvalidConfig {
            field: "search.rrf_k".into(),
            value: config.search.rrf_k.to_string(),
            reason: "must be >= 1.0".into(),
        });
    }

    if config.search.quality_timeout_ms < 50 {
        return Err(SearchError::InvalidConfig {
            field: "search.quality_timeout_ms".into(),
            value: config.search.quality_timeout_ms.to_string(),
            reason: "must be >= 50".into(),
        });
    }

    if !(1_u8..=100_u8).contains(&config.pressure.cpu_ceiling_pct) {
        return Err(SearchError::InvalidConfig {
            field: "pressure.cpu_ceiling_pct".into(),
            value: config.pressure.cpu_ceiling_pct.to_string(),
            reason: "must be between 1 and 100".into(),
        });
    }

    if config.pressure.memory_ceiling_mb < 128 {
        return Err(SearchError::InvalidConfig {
            field: "pressure.memory_ceiling_mb".into(),
            value: config.pressure.memory_ceiling_mb.to_string(),
            reason: "must be >= 128".into(),
        });
    }

    if config.pressure.sample_interval_ms < 100 {
        return Err(SearchError::InvalidConfig {
            field: "pressure.sample_interval_ms".into(),
            value: config.pressure.sample_interval_ms.to_string(),
            reason: "must be >= 100".into(),
        });
    }

    if !(1_u16..=1000_u16).contains(&config.pressure.ewma_alpha_per_mille) {
        return Err(SearchError::InvalidConfig {
            field: "pressure.ewma_alpha_per_mille".into(),
            value: config.pressure.ewma_alpha_per_mille.to_string(),
            reason: "must be between 1 and 1000".into(),
        });
    }

    if !(1_u8..=32_u8).contains(&config.pressure.anti_flap_readings) {
        return Err(SearchError::InvalidConfig {
            field: "pressure.anti_flap_readings".into(),
            value: config.pressure.anti_flap_readings.to_string(),
            reason: "must be between 1 and 32".into(),
        });
    }

    if config.pressure.io_ceiling_bytes_per_sec == 0 {
        return Err(SearchError::InvalidConfig {
            field: "pressure.io_ceiling_bytes_per_sec".into(),
            value: config.pressure.io_ceiling_bytes_per_sec.to_string(),
            reason: "must be > 0".into(),
        });
    }

    if config.pressure.load_ceiling_per_mille < 100 {
        return Err(SearchError::InvalidConfig {
            field: "pressure.load_ceiling_per_mille".into(),
            value: config.pressure.load_ceiling_per_mille.to_string(),
            reason: "must be >= 100".into(),
        });
    }

    if !(8_u16..=200_u16).contains(&config.tui.frame_budget_ms) {
        return Err(SearchError::InvalidConfig {
            field: "tui.frame_budget_ms".into(),
            value: config.tui.frame_budget_ms.to_string(),
            reason: "must be between 8 and 200".into(),
        });
    }

    if !(1_u16..=3650_u16).contains(&config.storage.evidence_retention_days) {
        return Err(SearchError::InvalidConfig {
            field: "storage.evidence_retention_days".into(),
            value: config.storage.evidence_retention_days.to_string(),
            reason: "must be between 1 and 3650".into(),
        });
    }

    if !(1_u16..=3650_u16).contains(&config.storage.summary_retention_days) {
        return Err(SearchError::InvalidConfig {
            field: "storage.summary_retention_days".into(),
            value: config.storage.summary_retention_days.to_string(),
            reason: "must be between 1 and 3650".into(),
        });
    }

    if config.storage.index_dir.trim().is_empty() {
        return Err(SearchError::InvalidConfig {
            field: "storage.index_dir".into(),
            value: config.storage.index_dir.clone(),
            reason: "must not be empty".into(),
        });
    }

    if config.search.fast_only && !config.indexing.quality_model.trim().is_empty() {
        warnings.push(ConfigWarning {
            reason_code: "config.search.fast_only_with_quality_model".into(),
            field: "search.fast_only".into(),
            source: ConfigSource::Runtime,
            message: "fast_only=true while quality_model is configured".into(),
        });
    }

    Ok(())
}

fn parse_csv(value: &str, field: &str) -> SearchResult<Vec<String>> {
    let parts: Vec<String> = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect();

    if parts.is_empty() {
        return Err(SearchError::InvalidConfig {
            field: field.into(),
            value: value.into(),
            reason: "expected at least one comma-separated value".into(),
        });
    }

    Ok(parts)
}

fn parse_bool(value: &str, field: &str) -> SearchResult<bool> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(SearchError::InvalidConfig {
            field: field.into(),
            value: value.into(),
            reason: "expected boolean (true/false/1/0/yes/no/on/off)".into(),
        }),
    }
}

fn parse_usize(value: &str, field: &str) -> SearchResult<usize> {
    value
        .parse::<usize>()
        .map_err(|_| SearchError::InvalidConfig {
            field: field.into(),
            value: value.into(),
            reason: "expected unsigned integer".into(),
        })
}

fn parse_u64(value: &str, field: &str) -> SearchResult<u64> {
    value
        .parse::<u64>()
        .map_err(|_| SearchError::InvalidConfig {
            field: field.into(),
            value: value.into(),
            reason: "expected unsigned integer".into(),
        })
}

fn parse_u16(value: &str, field: &str) -> SearchResult<u16> {
    value
        .parse::<u16>()
        .map_err(|_| SearchError::InvalidConfig {
            field: field.into(),
            value: value.into(),
            reason: "expected unsigned integer".into(),
        })
}

fn parse_u8(value: &str, field: &str) -> SearchResult<u8> {
    value.parse::<u8>().map_err(|_| SearchError::InvalidConfig {
        field: field.into(),
        value: value.into(),
        reason: "expected unsigned integer".into(),
    })
}

fn parse_f64(value: &str, field: &str) -> SearchResult<f64> {
    value
        .parse::<f64>()
        .map_err(|_| SearchError::InvalidConfig {
            field: field.into(),
            value: value.into(),
            reason: "expected floating-point number".into(),
        })
}

fn normalize_path(path: &Path) -> String {
    path.to_string_lossy()
        .replace('\\', "/")
        .to_ascii_lowercase()
}

fn normalized_components(path: &Path) -> Vec<String> {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(value) => Some(value.to_string_lossy().to_ascii_lowercase()),
            _ => None,
        })
        .collect()
}

fn path_matches_pattern(pattern: &str, normalized_path: &str, components: &[String]) -> bool {
    let trimmed = pattern.trim_matches('/');
    if trimmed.is_empty() {
        return false;
    }

    if trimmed.contains('*') {
        return wildcard_match(normalized_path, trimmed);
    }

    if trimmed.contains('/') {
        return normalized_path.contains(trimmed);
    }

    components.iter().any(|component| component == trimmed)
}

fn wildcard_match(haystack: &str, pattern: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.is_empty() {
        return haystack.is_empty();
    }

    let starts_with_wildcard = pattern.starts_with('*');
    let ends_with_wildcard = pattern.ends_with('*');
    let mut search_from = 0_usize;

    for (idx, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }

        if idx == 0 && !starts_with_wildcard {
            if !haystack.starts_with(part) {
                return false;
            }
            search_from = part.len();
            continue;
        }

        if let Some(offset) = haystack[search_from..].find(part) {
            search_from += offset + part.len();
        } else {
            return false;
        }
    }

    if ends_with_wildcard {
        return true;
    }

    parts
        .iter()
        .rev()
        .find(|part| !part.is_empty())
        .is_none_or(|last_non_empty| haystack.ends_with(last_non_empty))
}

fn lower_extension(path: &Path) -> Option<String> {
    path.extension()
        .map(|value| value.to_string_lossy().to_ascii_lowercase())
}

fn lower_filename(path: &Path) -> Option<String> {
    path.file_name()
        .map(|value| value.to_string_lossy().to_ascii_lowercase())
}

fn has_low_utility_component(path: &Path) -> bool {
    normalized_components(path)
        .iter()
        .any(|component| LOW_UTILITY_PATH_COMPONENTS.contains(&component.as_str()))
}

fn is_low_value_extension(extension: &str) -> bool {
    matches!(extension, "log" | "tmp" | "bak" | "old" | "map")
}

fn is_generated_or_minified_filename(filename: &str) -> bool {
    filename.ends_with(".min.js")
        || filename.ends_with(".bundle.js")
        || filename.ends_with(".generated.rs")
        || filename.ends_with(".generated.ts")
}

fn normalize_reason_codes(reason_codes: &mut Vec<String>) {
    reason_codes.sort_unstable();
    reason_codes.dedup();
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    use frankensearch_core::SearchError;
    use proptest::prelude::*;

    use super::{
        CliOverrides, DiscoveryCandidate, DiscoveryScopeDecision, IngestionClass,
        PRESSURE_PROFILE_VERSION, PressureProfileField, ProfileOverrideSource,
        default_config_file_path, default_project_config_file_path, default_user_config_file_path,
        load_from_layered_sources, load_from_sources, load_from_str, parse_bool, parse_csv,
    };

    fn home() -> &'static Path {
        Path::new("/home/tester")
    }

    #[test]
    fn default_config_has_safe_privacy_defaults() {
        let result = load_from_str(
            None,
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect("load defaults");
        assert!(result.config.privacy.redact_file_contents_in_logs);
        assert!(result.config.privacy.redact_paths_in_telemetry);
    }

    #[test]
    fn precedence_is_cli_then_env_then_file_then_defaults() {
        let file = "\
[search]\ndefault_limit = 11\n\
[tui]\ntheme = \"light\"\n";
        let env = HashMap::from([
            ("FSFS_SEARCH_DEFAULT_LIMIT".into(), "17".into()),
            ("FSFS_TUI_THEME".into(), "auto".into()),
        ]);

        let cli = CliOverrides {
            limit: Some(29),
            theme: Some(super::TuiTheme::Dark),
            ..CliOverrides::default()
        };

        let result = load_from_str(Some(file), None, &env, &cli, home()).expect("load config");
        assert_eq!(result.config.search.default_limit, 29);
        assert_eq!(result.config.tui.theme, super::TuiTheme::Dark);
        assert!(
            result
                .env_keys_used
                .contains(&"FSFS_SEARCH_DEFAULT_LIMIT".to_string())
        );
        assert!(result.cli_flags_used.contains(&"--limit".to_string()));
    }

    #[test]
    fn profile_resolution_rejects_locked_quality_override() {
        let file = "\
[pressure]\nprofile = \"performance\"\n\
[search]\nfast_only = true\n";

        let result = load_from_str(
            Some(file),
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect("load config");

        assert!(!result.config.search.fast_only);
        assert!(result.pressure_profile_resolution.conflict_detected);
        assert_eq!(
            result
                .pressure_profile_resolution
                .diagnostics
                .effective_profile_version,
            PRESSURE_PROFILE_VERSION
        );
        assert!(
            result
                .pressure_profile_resolution
                .overrides
                .iter()
                .any(|decision| {
                    decision.field == PressureProfileField::QualityEnabled
                        && decision.source == ProfileOverrideSource::Config
                        && !decision.applied
                        && decision.reason_code == "override.rejected.locked_field"
                })
        );
    }

    #[test]
    fn profile_resolution_uses_cli_env_file_precedence_for_overridable_fields() {
        let file = "\
[pressure]\nprofile = \"performance\"\n\
[indexing]\nwatch_mode = false\n";
        let env = HashMap::from([("FSFS_INDEXING_WATCH_MODE".into(), "true".into())]);
        let cli = CliOverrides {
            allow_background_indexing: Some(false),
            ..CliOverrides::default()
        };

        let result = load_from_str(Some(file), None, &env, &cli, home()).expect("load config");

        assert!(!result.config.indexing.watch_mode);
        assert_eq!(
            result
                .pressure_profile_resolution
                .overrides
                .iter()
                .filter(|decision| {
                    decision.field == PressureProfileField::AllowBackgroundIndexing
                        && decision.applied
                })
                .count(),
            3
        );
    }

    #[test]
    fn hard_pause_clamps_profile_managed_capabilities() {
        let cli = CliOverrides {
            profile: Some(super::PressureProfile::Performance),
            hard_pause_requested: Some(true),
            allow_background_indexing: Some(true),
            ..CliOverrides::default()
        };

        let result = load_from_str(None, None, &HashMap::new(), &cli, home()).expect("load");
        assert!(result.config.search.fast_only);
        assert!(!result.config.indexing.watch_mode);
        assert_eq!(result.pressure_profile_resolution.safety_clamps.len(), 2);
    }

    #[test]
    fn unknown_keys_are_reported_as_warnings() {
        let file = "\
[search]\nshadow_mode = true\n";
        let result = load_from_str(
            Some(file),
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect("load config");

        assert!(result.warnings.iter().any(|warning| warning.reason_code
            == "config.unknown_key.warning"
            && warning.field == "search.shadow_mode"));
    }

    #[test]
    fn tilde_paths_are_expanded() {
        let file = "\
[indexing]\nmodel_dir = \"~/.cache/fsfs/models\"\n\
[storage]\ndb_path = \"~/.local/share/fsfs/data.db\"\n";
        let result = load_from_str(
            Some(file),
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect("load config");

        assert_eq!(
            result.config.indexing.model_dir,
            "/home/tester/.cache/fsfs/models"
        );
        assert_eq!(
            result.config.storage.db_path,
            "/home/tester/.local/share/fsfs/data.db"
        );
        assert_eq!(result.path_expansions.len(), 2);
    }

    #[test]
    fn fast_only_emits_warning_when_quality_model_exists() {
        let cli = CliOverrides {
            fast_only: Some(true),
            ..CliOverrides::default()
        };

        let result = load_from_str(None, None, &HashMap::new(), &cli, home()).expect("load");
        assert!(result.warnings.iter().any(|warning| {
            warning.reason_code == "config.search.fast_only_with_quality_model"
        }));
    }

    #[test]
    fn invalid_retention_is_rejected() {
        let file = "\
[storage]\nevidence_retention_days = 30\nsummary_retention_days = 10\n";
        let err = load_from_str(
            Some(file),
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect_err("must reject invalid retention");

        assert!(matches!(
            err,
            SearchError::InvalidConfig { field, .. } if field == "storage.summary_retention_days"
        ));
    }

    #[test]
    fn invalid_env_boolean_is_rejected() {
        let env = HashMap::from([("FSFS_SEARCH_FAST_ONLY".into(), "not-a-bool".into())]);
        let err = load_from_str(None, None, &env, &CliOverrides::default(), home())
            .expect_err("must reject invalid bool");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    fn with_case_mask(input: &str, mask: u32) -> String {
        let mut bit = 0_u32;
        input
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphabetic() {
                    let upper = (mask >> bit) & 1 == 1;
                    bit = bit.saturating_add(1);
                    if upper {
                        ch.to_ascii_uppercase()
                    } else {
                        ch.to_ascii_lowercase()
                    }
                } else {
                    ch
                }
            })
            .collect()
    }

    proptest! {
        #[test]
        fn parse_bool_accepts_case_variants(
            is_true in any::<bool>(),
            token_idx in 0usize..4,
            mask in any::<u32>(),
        ) {
            const TRUE_TOKENS: [&str; 4] = ["true", "yes", "on", "1"];
            const FALSE_TOKENS: [&str; 4] = ["false", "no", "off", "0"];
            let base = if is_true {
                TRUE_TOKENS[token_idx]
            } else {
                FALSE_TOKENS[token_idx]
            };
            let candidate = with_case_mask(base, mask);
            let parsed = parse_bool(&candidate, "test.bool").expect("recognized boolean token");
            prop_assert_eq!(parsed, is_true);
        }

        #[test]
        fn parse_bool_rejects_non_boolean_tokens(token in "[A-Za-z_][A-Za-z0-9_\\-]{0,15}") {
            let normalized = token.to_ascii_lowercase();
            prop_assume!(!matches!(
                normalized.as_str(),
                "1" | "0" | "true" | "false" | "yes" | "no" | "on" | "off"
            ));
            let parsed = parse_bool(&token, "test.bool");
            assert!(matches!(parsed, Err(SearchError::InvalidConfig { .. })));
        }

        #[test]
        fn parse_csv_roundtrips_trimmed_tokens(tokens in prop::collection::vec("[A-Za-z0-9_./\\-]{1,12}", 1..8)) {
            let csv = tokens
                .iter()
                .map(|token| format!("  {token}  "))
                .collect::<Vec<_>>()
                .join(" , ");
            let parsed = parse_csv(&csv, "test.csv").expect("csv parse must succeed");
            assert!(parsed.iter().all(|token| !token.trim().is_empty()));
            prop_assert_eq!(parsed, tokens);
        }
    }

    fn assert_invalid_field(file: &str, field: &str) {
        let err = load_from_str(
            Some(file),
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect_err("must reject invalid config");
        assert!(
            matches!(err, SearchError::InvalidConfig { field: err_field, .. } if err_field == field)
        );
    }

    #[test]
    fn enforces_numeric_range_constraints() {
        assert_invalid_field(
            "[discovery]\nmax_file_size_mb = 0\n",
            "discovery.max_file_size_mb",
        );
        assert_invalid_field(
            "[indexing]\nembedding_batch_size = 0\n",
            "indexing.embedding_batch_size",
        );
        assert_invalid_field("[search]\ndefault_limit = 0\n", "search.default_limit");
        assert_invalid_field("[search]\nrrf_k = 0.5\n", "search.rrf_k");
        assert_invalid_field(
            "[search]\nquality_timeout_ms = 49\n",
            "search.quality_timeout_ms",
        );
        assert_invalid_field(
            "[pressure]\ncpu_ceiling_pct = 0\n",
            "pressure.cpu_ceiling_pct",
        );
        assert_invalid_field(
            "[pressure]\nmemory_ceiling_mb = 64\n",
            "pressure.memory_ceiling_mb",
        );
        assert_invalid_field(
            "[pressure]\nsample_interval_ms = 99\n",
            "pressure.sample_interval_ms",
        );
        assert_invalid_field(
            "[pressure]\newma_alpha_per_mille = 0\n",
            "pressure.ewma_alpha_per_mille",
        );
        assert_invalid_field(
            "[pressure]\nanti_flap_readings = 0\n",
            "pressure.anti_flap_readings",
        );
        assert_invalid_field(
            "[pressure]\nio_ceiling_bytes_per_sec = 0\n",
            "pressure.io_ceiling_bytes_per_sec",
        );
        assert_invalid_field(
            "[pressure]\nload_ceiling_per_mille = 99\n",
            "pressure.load_ceiling_per_mille",
        );
        assert_invalid_field("[tui]\nframe_budget_ms = 7\n", "tui.frame_budget_ms");
        assert_invalid_field(
            "[storage]\nevidence_retention_days = 0\n",
            "storage.evidence_retention_days",
        );
        assert_invalid_field(
            "[storage]\nsummary_retention_days = 0\n",
            "storage.summary_retention_days",
        );
    }

    #[test]
    fn load_from_sources_expands_tilde_config_path() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let home_dir = std::env::temp_dir().join(format!("fsfs-config-home-{unique}"));
        let config_file = home_dir
            .join(".config")
            .join("frankensearch")
            .join("config.toml");
        fs::create_dir_all(config_file.parent().expect("parent")).expect("mkdir");
        fs::write(&config_file, "[search]\ndefault_limit = 42\n").expect("write");

        let result = load_from_sources(
            Some(Path::new("~/.config/frankensearch/config.toml")),
            &HashMap::new(),
            &CliOverrides::default(),
            &home_dir,
        )
        .expect("load with tilde path");

        assert_eq!(result.config.search.default_limit, 42);
        assert_eq!(result.config_file_used, Some(config_file));
    }

    #[test]
    fn discovery_policy_excludes_binary_extension() {
        let config = load_from_str(
            None,
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect("defaults")
        .config;
        let candidate = DiscoveryCandidate::new(Path::new("/home/tester/app/main.wasm"), 2_048);
        let decision = config.discovery.evaluate_candidate(&candidate);

        assert_eq!(decision.scope, DiscoveryScopeDecision::Exclude);
        assert_eq!(decision.ingestion_class, IngestionClass::Skip);
        assert!(
            decision
                .reason_codes
                .iter()
                .any(|code| code == "discovery.file.binary_blocked")
        );
    }

    #[test]
    fn discovery_policy_assigns_full_for_source_code() {
        let config = load_from_str(
            None,
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect("defaults")
        .config;
        let candidate = DiscoveryCandidate::new(Path::new("/home/tester/src/lib.rs"), 8_192);
        let decision = config.discovery.evaluate_candidate(&candidate);

        assert_eq!(decision.scope, DiscoveryScopeDecision::Include);
        assert_eq!(
            decision.ingestion_class,
            IngestionClass::FullSemanticLexical
        );
        assert!(decision.utility_score >= 70);
    }

    #[test]
    fn discovery_policy_downgrades_large_candidate() {
        let config = load_from_str(
            None,
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect("defaults")
        .config;
        let oversized = (config.discovery.max_file_size_mb as u64 * 1024 * 1024) + 1;
        let candidate =
            DiscoveryCandidate::new(Path::new("/home/tester/docs/reference.md"), oversized);
        let decision = config.discovery.evaluate_candidate(&candidate);

        assert_eq!(decision.scope, DiscoveryScopeDecision::Include);
        assert!(matches!(
            decision.ingestion_class,
            IngestionClass::LexicalOnly | IngestionClass::MetadataOnly
        ));
        assert!(
            decision
                .reason_codes
                .iter()
                .any(|code| code == "discovery.file.too_large")
        );
    }

    #[test]
    fn discovery_policy_metadata_fallback_at_threshold() {
        let config = load_from_str(
            None,
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect("defaults")
        .config;
        let candidate =
            DiscoveryCandidate::new(Path::new("/home/tester/.cache/snapshot.lockstate"), 2_048);
        let decision = config.discovery.evaluate_candidate(&candidate);

        assert_eq!(decision.utility_score, 20);
        assert_eq!(decision.scope, DiscoveryScopeDecision::Include);
        assert_eq!(decision.ingestion_class, IngestionClass::MetadataOnly);
        assert!(
            decision
                .reason_codes
                .iter()
                .any(|code| code == "discovery.file.included")
        );
    }

    #[test]
    fn discovery_policy_allowlist_unknown_extension_falls_back_to_skip() {
        let file = "\
[discovery]\ntext_selection_mode = \"allowlist\"\n";
        let config = load_from_str(
            Some(file),
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect("config")
        .config;
        let candidate =
            DiscoveryCandidate::new(Path::new("/home/tester/docs/snapshot.unknown"), 1_024);
        let decision = config.discovery.evaluate_candidate(&candidate);

        assert!(decision.utility_score < 20);
        assert_eq!(decision.scope, DiscoveryScopeDecision::Exclude);
        assert_eq!(decision.ingestion_class, IngestionClass::Skip);
        assert!(
            decision
                .reason_codes
                .iter()
                .any(|code| code == "discovery.file.excluded_pattern")
        );
    }

    #[test]
    fn discovery_policy_skips_network_mount_when_configured() {
        let file = "\
[discovery]\nskip_network_mounts = true\n";
        let config = load_from_str(
            Some(file),
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect("config")
        .config;
        let candidate = DiscoveryCandidate::new(Path::new("/mnt/nfs/project/main.rs"), 4_096)
            .with_mount_category(crate::mount_info::FsCategory::Nfs);
        let decision = config.discovery.evaluate_candidate(&candidate);

        assert_eq!(decision.scope, DiscoveryScopeDecision::Exclude);
        assert_eq!(decision.ingestion_class, IngestionClass::Skip);
        assert!(
            decision
                .reason_codes
                .iter()
                .any(|code| code == "discovery.root.rejected")
        );
    }

    #[test]
    fn discovery_policy_is_deterministic() {
        let config = load_from_str(
            None,
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect("defaults")
        .config;
        let candidate =
            DiscoveryCandidate::new(Path::new("/home/tester/node_modules/pkg/index.ts"), 6_144);
        let first = config.discovery.evaluate_candidate(&candidate);
        let second = config.discovery.evaluate_candidate(&candidate);

        assert_eq!(first, second);

        let mut sorted = first.reason_codes.clone();
        sorted.sort_unstable();
        assert_eq!(first.reason_codes, sorted);
    }

    #[test]
    fn discovery_policy_uses_file_excluded_reason_for_path_match() {
        let config = load_from_str(
            None,
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect("defaults")
        .config;
        let candidate =
            DiscoveryCandidate::new(Path::new("/home/tester/target/debug/fsfs.log"), 1_024);
        let decision = config.discovery.evaluate_candidate(&candidate);

        assert_eq!(decision.scope, DiscoveryScopeDecision::Exclude);
        assert!(
            decision
                .reason_codes
                .iter()
                .any(|code| code == "discovery.file.excluded_pattern")
        );
        assert!(
            decision
                .reason_codes
                .iter()
                .any(|code| code == "discovery.root.accepted")
        );
    }

    #[test]
    fn uses_xdg_config_home_when_available() {
        // Verify helper shape without mutating process environment in tests.
        let path = default_config_file_path(home());
        let rendered = path.to_string_lossy();
        assert!(rendered.contains("/frankensearch/config.toml"));
    }

    #[test]
    fn layered_sources_apply_project_over_user() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("fsfs-config-layer-{unique}"));
        let project = root.join("repo");
        let project_config = project.join(".frankensearch").join("config.toml");
        let user_config = root
            .join(".config")
            .join("frankensearch")
            .join("config.toml");

        fs::create_dir_all(project_config.parent().expect("project parent")).expect("mkdir");
        fs::create_dir_all(user_config.parent().expect("user parent")).expect("mkdir");

        fs::write(&user_config, "[search]\ndefault_limit = 15\n").expect("write user");
        fs::write(&project_config, "[search]\ndefault_limit = 7\n").expect("write project");

        let result = load_from_layered_sources(
            Some(project_config.as_path()),
            Some(user_config.as_path()),
            &HashMap::new(),
            &CliOverrides::default(),
            &root,
        )
        .expect("layered load");

        assert_eq!(result.config.search.default_limit, 7);
        assert_eq!(result.config_file_used, Some(project_config));
    }

    #[test]
    fn frankensearch_env_prefix_takes_precedence_over_legacy_fsfs_prefix() {
        let env = HashMap::from([
            ("FRANKENSEARCH_SEARCH_DEFAULT_LIMIT".into(), "13".into()),
            ("FSFS_SEARCH_DEFAULT_LIMIT".into(), "29".into()),
        ]);

        let result =
            load_from_str(None, None, &env, &CliOverrides::default(), home()).expect("load config");
        assert_eq!(result.config.search.default_limit, 13);
        assert!(
            result
                .env_keys_used
                .contains(&"FRANKENSEARCH_SEARCH_DEFAULT_LIMIT".to_string())
        );
    }

    #[test]
    fn frankensearch_index_dir_alias_takes_precedence() {
        let env = HashMap::from([
            ("FRANKENSEARCH_INDEX_DIR".into(), "/tmp/from-alias".into()),
            (
                "FRANKENSEARCH_STORAGE_INDEX_DIR".into(),
                "/tmp/from-storage-key".into(),
            ),
        ]);

        let result =
            load_from_str(None, None, &env, &CliOverrides::default(), home()).expect("load config");
        assert_eq!(result.config.storage.index_dir, "/tmp/from-alias");
        assert!(
            result
                .env_keys_used
                .contains(&"FRANKENSEARCH_INDEX_DIR".to_string())
        );
    }

    #[test]
    fn default_paths_and_zero_config_storage_defaults_are_project_friendly() {
        let user = default_user_config_file_path(home());
        assert!(
            user.to_string_lossy()
                .contains("/frankensearch/config.toml")
        );

        let project = default_project_config_file_path(Path::new("/tmp/workspace"));
        assert_eq!(
            project,
            Path::new("/tmp/workspace/.frankensearch/config.toml")
        );

        let result = load_from_str(
            None,
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect("load defaults");
        assert_eq!(result.config.discovery.roots, vec![".".to_string()]);
        assert!(!result.config.discovery.follow_symlinks);
        assert_eq!(result.config.storage.index_dir, ".frankensearch");
        assert_eq!(result.config.search.default_limit, 10);
    }

    #[test]
    fn project_config_path_walks_up_to_git_root() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("fsfs-git-root-{unique}"));
        let nested = root.join("deep").join("inside").join("repo");
        fs::create_dir_all(&nested).expect("mkdir nested");
        fs::create_dir_all(root.join(".git")).expect("mkdir .git");

        let path = default_project_config_file_path(&nested);
        assert_eq!(path, root.join(".frankensearch").join("config.toml"));
    }

    #[test]
    fn project_config_path_accepts_worktree_git_file() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("fsfs-worktree-root-{unique}"));
        let nested = root.join("subdir");
        fs::create_dir_all(&nested).expect("mkdir nested");
        fs::write(root.join(".git"), "gitdir: /tmp/virtual/gitdir\n").expect("write .git file");

        let path = default_project_config_file_path(&nested);
        assert_eq!(path, root.join(".frankensearch").join("config.toml"));
    }

    #[test]
    fn frankensearch_model_and_search_env_aliases_apply() {
        let env = HashMap::from([
            ("FRANKENSEARCH_MODEL_DIR".into(), "/tmp/env-models".into()),
            ("FRANKENSEARCH_QUALITY_WEIGHT".into(), "0.42".into()),
            ("FRANKENSEARCH_RRF_K".into(), "77.0".into()),
            ("FRANKENSEARCH_FAST_ONLY".into(), "true".into()),
        ]);

        let result =
            load_from_str(None, None, &env, &CliOverrides::default(), home()).expect("load config");
        assert_eq!(result.config.indexing.model_dir, "/tmp/env-models");
        assert!((result.config.search.quality_weight - 0.42).abs() < f64::EPSILON);
        assert!((result.config.search.rrf_k - 77.0).abs() < f64::EPSILON);
        assert!(
            result
                .env_keys_used
                .contains(&"FRANKENSEARCH_FAST_ONLY".to_string())
        );
    }

    // ─── bd-12af tests begin ───

    // ── Enum defaults ──

    #[test]
    fn text_selection_mode_default_is_blocklist() {
        assert_eq!(
            super::TextSelectionMode::default(),
            super::TextSelectionMode::Blocklist
        );
    }

    #[test]
    fn pressure_profile_default_is_performance() {
        assert_eq!(
            super::PressureProfile::default(),
            super::PressureProfile::Performance
        );
    }

    #[test]
    fn degradation_override_mode_default_is_auto() {
        assert_eq!(
            super::DegradationOverrideMode::default(),
            super::DegradationOverrideMode::Auto
        );
    }

    #[test]
    fn tui_theme_default_is_dark() {
        assert_eq!(super::TuiTheme::default(), super::TuiTheme::Dark);
    }

    #[test]
    fn density_default_is_normal() {
        assert_eq!(super::Density::default(), super::Density::Normal);
    }

    // ── FromStr implementations ──

    #[test]
    fn pressure_profile_from_str_all_variants() {
        use std::str::FromStr;
        assert_eq!(
            super::PressureProfile::from_str("strict").unwrap(),
            super::PressureProfile::Strict
        );
        assert_eq!(
            super::PressureProfile::from_str("performance").unwrap(),
            super::PressureProfile::Performance
        );
        assert_eq!(
            super::PressureProfile::from_str("degraded").unwrap(),
            super::PressureProfile::Degraded
        );
    }

    #[test]
    fn pressure_profile_from_str_invalid() {
        use std::str::FromStr;
        assert!(super::PressureProfile::from_str("STRICT").is_err());
        assert!(super::PressureProfile::from_str("unknown").is_err());
        assert!(super::PressureProfile::from_str("").is_err());
    }

    #[test]
    fn degradation_override_mode_from_str_all_aliases() {
        use std::str::FromStr;
        assert_eq!(
            super::DegradationOverrideMode::from_str("auto").unwrap(),
            super::DegradationOverrideMode::Auto
        );
        assert_eq!(
            super::DegradationOverrideMode::from_str("full").unwrap(),
            super::DegradationOverrideMode::ForceFull
        );
        assert_eq!(
            super::DegradationOverrideMode::from_str("force_full").unwrap(),
            super::DegradationOverrideMode::ForceFull
        );
        assert_eq!(
            super::DegradationOverrideMode::from_str("embed_deferred").unwrap(),
            super::DegradationOverrideMode::ForceEmbedDeferred
        );
        assert_eq!(
            super::DegradationOverrideMode::from_str("force_embed_deferred").unwrap(),
            super::DegradationOverrideMode::ForceEmbedDeferred
        );
        assert_eq!(
            super::DegradationOverrideMode::from_str("lexical_only").unwrap(),
            super::DegradationOverrideMode::ForceLexicalOnly
        );
        assert_eq!(
            super::DegradationOverrideMode::from_str("force_lexical_only").unwrap(),
            super::DegradationOverrideMode::ForceLexicalOnly
        );
        assert_eq!(
            super::DegradationOverrideMode::from_str("metadata_only").unwrap(),
            super::DegradationOverrideMode::ForceMetadataOnly
        );
        assert_eq!(
            super::DegradationOverrideMode::from_str("force_metadata_only").unwrap(),
            super::DegradationOverrideMode::ForceMetadataOnly
        );
        assert_eq!(
            super::DegradationOverrideMode::from_str("paused").unwrap(),
            super::DegradationOverrideMode::ForcePaused
        );
        assert_eq!(
            super::DegradationOverrideMode::from_str("force_paused").unwrap(),
            super::DegradationOverrideMode::ForcePaused
        );
    }

    #[test]
    fn degradation_override_mode_from_str_invalid() {
        use std::str::FromStr;
        assert!(super::DegradationOverrideMode::from_str("AUTO").is_err());
        assert!(super::DegradationOverrideMode::from_str("").is_err());
        assert!(super::DegradationOverrideMode::from_str("force").is_err());
    }

    #[test]
    fn tui_theme_from_str_all_variants() {
        use std::str::FromStr;
        assert_eq!(
            super::TuiTheme::from_str("auto").unwrap(),
            super::TuiTheme::Auto
        );
        assert_eq!(
            super::TuiTheme::from_str("light").unwrap(),
            super::TuiTheme::Light
        );
        assert_eq!(
            super::TuiTheme::from_str("dark").unwrap(),
            super::TuiTheme::Dark
        );
    }

    #[test]
    fn tui_theme_from_str_invalid() {
        use std::str::FromStr;
        assert!(super::TuiTheme::from_str("DARK").is_err());
        assert!(super::TuiTheme::from_str("").is_err());
    }

    // ── Serde roundtrips ──

    #[test]
    fn text_selection_mode_serde_roundtrip() {
        for mode in [
            super::TextSelectionMode::Blocklist,
            super::TextSelectionMode::Allowlist,
        ] {
            let json = serde_json::to_string(&mode).unwrap();
            let back: super::TextSelectionMode = serde_json::from_str(&json).unwrap();
            assert_eq!(mode, back);
        }
    }

    #[test]
    fn pressure_profile_serde_roundtrip() {
        for profile in [
            super::PressureProfile::Strict,
            super::PressureProfile::Performance,
            super::PressureProfile::Degraded,
        ] {
            let json = serde_json::to_string(&profile).unwrap();
            let back: super::PressureProfile = serde_json::from_str(&json).unwrap();
            assert_eq!(profile, back);
        }
    }

    #[test]
    fn degradation_override_mode_serde_roundtrip() {
        for mode in [
            super::DegradationOverrideMode::Auto,
            super::DegradationOverrideMode::ForceFull,
            super::DegradationOverrideMode::ForceEmbedDeferred,
            super::DegradationOverrideMode::ForceLexicalOnly,
            super::DegradationOverrideMode::ForceMetadataOnly,
            super::DegradationOverrideMode::ForcePaused,
        ] {
            let json = serde_json::to_string(&mode).unwrap();
            let back: super::DegradationOverrideMode = serde_json::from_str(&json).unwrap();
            assert_eq!(mode, back);
        }
    }

    #[test]
    fn tui_theme_serde_roundtrip() {
        for theme in [
            super::TuiTheme::Auto,
            super::TuiTheme::Light,
            super::TuiTheme::Dark,
        ] {
            let json = serde_json::to_string(&theme).unwrap();
            let back: super::TuiTheme = serde_json::from_str(&json).unwrap();
            assert_eq!(theme, back);
        }
    }

    #[test]
    fn density_serde_roundtrip() {
        for density in [
            super::Density::Compact,
            super::Density::Normal,
            super::Density::Expanded,
        ] {
            let json = serde_json::to_string(&density).unwrap();
            let back: super::Density = serde_json::from_str(&json).unwrap();
            assert_eq!(density, back);
        }
    }

    #[test]
    fn ingestion_class_serde_roundtrip() {
        for class in [
            IngestionClass::FullSemanticLexical,
            IngestionClass::LexicalOnly,
            IngestionClass::MetadataOnly,
            IngestionClass::Skip,
        ] {
            let json = serde_json::to_string(&class).unwrap();
            let back: IngestionClass = serde_json::from_str(&json).unwrap();
            assert_eq!(class, back);
        }
    }

    #[test]
    fn config_source_serde_roundtrip() {
        for source in [
            super::ConfigSource::Cli,
            super::ConfigSource::Env,
            super::ConfigSource::File,
            super::ConfigSource::Defaults,
            super::ConfigSource::Runtime,
        ] {
            let json = serde_json::to_string(&source).unwrap();
            let back: super::ConfigSource = serde_json::from_str(&json).unwrap();
            assert_eq!(source, back);
        }
    }

    #[test]
    fn profile_scheduler_mode_serde_roundtrip() {
        for mode in [
            super::ProfileSchedulerMode::FairShare,
            super::ProfileSchedulerMode::LatencySensitive,
        ] {
            let json = serde_json::to_string(&mode).unwrap();
            let back: super::ProfileSchedulerMode = serde_json::from_str(&json).unwrap();
            assert_eq!(mode, back);
        }
    }

    #[test]
    fn pressure_profile_field_serde_roundtrip() {
        for field in [
            PressureProfileField::SchedulerMode,
            PressureProfileField::MaxEmbedConcurrency,
            PressureProfileField::MaxIndexConcurrency,
            PressureProfileField::QualityEnabled,
            PressureProfileField::AllowBackgroundIndexing,
        ] {
            let json = serde_json::to_string(&field).unwrap();
            let back: PressureProfileField = serde_json::from_str(&json).unwrap();
            assert_eq!(field, back);
        }
    }

    #[test]
    fn profile_override_source_serde_roundtrip() {
        for source in [
            ProfileOverrideSource::Cli,
            ProfileOverrideSource::Env,
            ProfileOverrideSource::Config,
        ] {
            let json = serde_json::to_string(&source).unwrap();
            let back: ProfileOverrideSource = serde_json::from_str(&json).unwrap();
            assert_eq!(source, back);
        }
    }

    #[test]
    fn discovery_scope_decision_serde_roundtrip() {
        for scope in [
            DiscoveryScopeDecision::Include,
            DiscoveryScopeDecision::Exclude,
        ] {
            let json = serde_json::to_string(&scope).unwrap();
            let back: DiscoveryScopeDecision = serde_json::from_str(&json).unwrap();
            assert_eq!(scope, back);
        }
    }

    // ── IngestionClass::is_indexed ──

    #[test]
    fn ingestion_class_is_indexed_all_variants() {
        assert!(IngestionClass::FullSemanticLexical.is_indexed());
        assert!(IngestionClass::LexicalOnly.is_indexed());
        assert!(IngestionClass::MetadataOnly.is_indexed());
        assert!(!IngestionClass::Skip.is_indexed());
    }

    // ── PressureProfileContract values per profile ──

    #[test]
    fn strict_contract_values() {
        let contract = super::PressureProfile::Strict.contract();
        assert_eq!(
            contract.scheduler_mode,
            super::ProfileSchedulerMode::FairShare
        );
        assert_eq!(contract.max_embed_concurrency, 2);
        assert_eq!(contract.max_index_concurrency, 2);
        assert!(!contract.quality_enabled);
        assert!(!contract.allow_background_indexing);
        assert_eq!(contract.pressure_enter_threshold_per_mille, 350);
        assert_eq!(contract.pressure_exit_threshold_per_mille, 200);
    }

    #[test]
    fn performance_contract_values() {
        let contract = super::PressureProfile::Performance.contract();
        assert_eq!(
            contract.scheduler_mode,
            super::ProfileSchedulerMode::LatencySensitive
        );
        assert_eq!(contract.max_embed_concurrency, 6);
        assert_eq!(contract.max_index_concurrency, 8);
        assert!(contract.quality_enabled);
        assert!(contract.allow_background_indexing);
        assert_eq!(contract.pressure_enter_threshold_per_mille, 650);
        assert_eq!(contract.pressure_exit_threshold_per_mille, 450);
    }

    #[test]
    fn degraded_contract_values() {
        let contract = super::PressureProfile::Degraded.contract();
        assert_eq!(
            contract.scheduler_mode,
            super::ProfileSchedulerMode::FairShare
        );
        assert_eq!(contract.max_embed_concurrency, 1);
        assert_eq!(contract.max_index_concurrency, 1);
        assert!(!contract.quality_enabled);
        assert!(!contract.allow_background_indexing);
        assert_eq!(contract.pressure_enter_threshold_per_mille, 150);
        assert_eq!(contract.pressure_exit_threshold_per_mille, 100);
    }

    #[test]
    fn strict_contract_locked_fields() {
        let contract = super::PressureProfile::Strict.contract();
        assert!(contract.is_locked_field(PressureProfileField::QualityEnabled));
        assert!(contract.is_locked_field(PressureProfileField::AllowBackgroundIndexing));
        assert!(contract.is_locked_field(PressureProfileField::MaxEmbedConcurrency));
        assert!(!contract.is_locked_field(PressureProfileField::SchedulerMode));
        assert!(!contract.is_locked_field(PressureProfileField::MaxIndexConcurrency));
    }

    #[test]
    fn performance_contract_locked_fields() {
        let contract = super::PressureProfile::Performance.contract();
        assert!(contract.is_locked_field(PressureProfileField::QualityEnabled));
        assert!(!contract.is_locked_field(PressureProfileField::AllowBackgroundIndexing));
        assert!(!contract.is_locked_field(PressureProfileField::MaxEmbedConcurrency));
        assert!(!contract.is_locked_field(PressureProfileField::SchedulerMode));
        assert!(!contract.is_locked_field(PressureProfileField::MaxIndexConcurrency));
    }

    #[test]
    fn degraded_contract_locks_all_fields() {
        let contract = super::PressureProfile::Degraded.contract();
        assert!(contract.is_locked_field(PressureProfileField::QualityEnabled));
        assert!(contract.is_locked_field(PressureProfileField::AllowBackgroundIndexing));
        assert!(contract.is_locked_field(PressureProfileField::MaxEmbedConcurrency));
        assert!(contract.is_locked_field(PressureProfileField::SchedulerMode));
        assert!(contract.is_locked_field(PressureProfileField::MaxIndexConcurrency));
    }

    #[test]
    fn to_effective_settings_preserves_contract() {
        let contract = super::PressureProfile::Performance.contract();
        let effective = contract.to_effective_settings();
        assert_eq!(effective.scheduler_mode, contract.scheduler_mode);
        assert_eq!(
            effective.max_embed_concurrency,
            contract.max_embed_concurrency
        );
        assert_eq!(
            effective.max_index_concurrency,
            contract.max_index_concurrency
        );
        assert_eq!(effective.quality_enabled, contract.quality_enabled);
        assert_eq!(
            effective.allow_background_indexing,
            contract.allow_background_indexing
        );
        assert_eq!(
            effective.pressure_enter_threshold_per_mille,
            contract.pressure_enter_threshold_per_mille
        );
        assert_eq!(
            effective.pressure_exit_threshold_per_mille,
            contract.pressure_exit_threshold_per_mille
        );
    }

    // ── Helper functions ──

    #[test]
    fn wildcard_match_star_middle() {
        assert!(super::wildcard_match("src/main.rs", "src/*.rs"));
    }

    #[test]
    fn wildcard_match_star_prefix() {
        assert!(super::wildcard_match("foo/bar.js", "*.js"));
    }

    #[test]
    fn wildcard_match_star_suffix() {
        assert!(super::wildcard_match("readme.md", "readme*"));
    }

    #[test]
    fn wildcard_match_no_star() {
        assert!(super::wildcard_match("exact", "exact"));
        assert!(!super::wildcard_match("exact", "notexact"));
    }

    #[test]
    fn wildcard_match_empty_pattern() {
        assert!(super::wildcard_match("", ""));
        // Empty pattern splits to [""], all parts are empty and skipped,
        // so ends_with check passes vacuously.
        assert!(super::wildcard_match("notempty", ""));
    }

    #[test]
    fn wildcard_match_double_star() {
        assert!(super::wildcard_match("a/b/c/d.rs", "a/**/d.rs"));
    }

    #[test]
    fn wildcard_match_trailing_requires_match() {
        assert!(!super::wildcard_match("foo.js.bak", "*.js"));
    }

    #[test]
    fn path_matches_pattern_simple_component() {
        let components = super::normalized_components(Path::new("/src/node_modules/pkg/index.js"));
        assert!(super::path_matches_pattern(
            "node_modules",
            "src/node_modules/pkg/index.js",
            &components,
        ));
    }

    #[test]
    fn path_matches_pattern_slash_segment() {
        let components = super::normalized_components(Path::new("/vendor/lib/foo.go"));
        assert!(super::path_matches_pattern(
            "vendor/lib",
            "vendor/lib/foo.go",
            &components,
        ));
    }

    #[test]
    fn path_matches_pattern_wildcard() {
        let components = super::normalized_components(Path::new("/src/main.rs"));
        assert!(super::path_matches_pattern(
            "*.rs",
            "src/main.rs",
            &components,
        ));
    }

    #[test]
    fn path_matches_pattern_empty_trimmed() {
        let components = super::normalized_components(Path::new("/src/main.rs"));
        assert!(!super::path_matches_pattern(
            "//",
            "src/main.rs",
            &components
        ));
    }

    #[test]
    fn normalize_path_backslashes_and_case() {
        assert_eq!(
            super::normalize_path(Path::new("SRC\\Main.RS")),
            "src/main.rs"
        );
    }

    #[test]
    fn normalized_components_filters_non_normal() {
        let components = super::normalized_components(Path::new("/foo/bar/baz.txt"));
        assert!(components.contains(&"foo".to_string()));
        assert!(components.contains(&"bar".to_string()));
        assert!(components.contains(&"baz.txt".to_string()));
    }

    #[test]
    fn lower_extension_returns_lowercase() {
        assert_eq!(
            super::lower_extension(Path::new("foo.RS")),
            Some("rs".to_string())
        );
    }

    #[test]
    fn lower_extension_returns_none_for_no_ext() {
        assert_eq!(super::lower_extension(Path::new("Makefile")), None);
    }

    #[test]
    fn lower_filename_returns_lowercase() {
        assert_eq!(
            super::lower_filename(Path::new("/src/Main.rs")),
            Some("main.rs".to_string())
        );
    }

    #[test]
    fn has_low_utility_component_detects() {
        assert!(super::has_low_utility_component(Path::new(
            "/project/node_modules/pkg/index.js"
        )));
        assert!(super::has_low_utility_component(Path::new(
            "/project/.cache/data"
        )));
    }

    #[test]
    fn has_low_utility_component_clean_path() {
        assert!(!super::has_low_utility_component(Path::new(
            "/project/src/main.rs"
        )));
    }

    #[test]
    fn is_low_value_extension_all_variants() {
        assert!(super::is_low_value_extension("log"));
        assert!(super::is_low_value_extension("tmp"));
        assert!(super::is_low_value_extension("bak"));
        assert!(super::is_low_value_extension("old"));
        assert!(super::is_low_value_extension("map"));
        assert!(!super::is_low_value_extension("rs"));
        assert!(!super::is_low_value_extension("js"));
    }

    #[test]
    fn is_generated_or_minified_all_patterns() {
        assert!(super::is_generated_or_minified_filename("app.min.js"));
        assert!(super::is_generated_or_minified_filename("vendor.bundle.js"));
        assert!(super::is_generated_or_minified_filename(
            "schema.generated.rs"
        ));
        assert!(super::is_generated_or_minified_filename(
            "types.generated.ts"
        ));
        assert!(!super::is_generated_or_minified_filename("main.rs"));
        assert!(!super::is_generated_or_minified_filename("app.js"));
    }

    #[test]
    fn expand_tilde_bare() {
        assert_eq!(
            super::expand_tilde("~", Path::new("/home/user")),
            Some("/home/user".to_string())
        );
    }

    #[test]
    fn expand_tilde_with_subpath() {
        let result = super::expand_tilde("~/docs/readme.md", Path::new("/home/user"));
        assert_eq!(result, Some("/home/user/docs/readme.md".to_string()));
    }

    #[test]
    fn expand_tilde_no_tilde() {
        assert_eq!(
            super::expand_tilde("/absolute/path", Path::new("/home/user")),
            None
        );
        assert_eq!(
            super::expand_tilde("relative/path", Path::new("/home/user")),
            None
        );
    }

    #[test]
    fn normalize_reason_codes_dedup_and_sort() {
        let mut codes = vec![
            "b.code".to_string(),
            "a.code".to_string(),
            "b.code".to_string(),
        ];
        super::normalize_reason_codes(&mut codes);
        assert_eq!(codes, vec!["a.code", "b.code"]);
    }

    // ── Parse helper functions ──

    #[test]
    fn parse_usize_valid() {
        assert_eq!(super::parse_usize("42", "test").unwrap(), 42);
    }

    #[test]
    fn parse_usize_invalid() {
        assert!(super::parse_usize("abc", "test").is_err());
        assert!(super::parse_usize("-1", "test").is_err());
    }

    #[test]
    fn parse_u64_valid() {
        assert_eq!(
            super::parse_u64("18446744073709551615", "test").unwrap(),
            u64::MAX
        );
    }

    #[test]
    fn parse_u64_invalid() {
        assert!(super::parse_u64("xyz", "test").is_err());
    }

    #[test]
    fn parse_u16_valid() {
        assert_eq!(super::parse_u16("65535", "test").unwrap(), u16::MAX);
    }

    #[test]
    fn parse_u16_invalid() {
        assert!(super::parse_u16("70000", "test").is_err());
    }

    #[test]
    fn parse_u8_valid() {
        assert_eq!(super::parse_u8("255", "test").unwrap(), u8::MAX);
    }

    #[test]
    fn parse_u8_invalid() {
        assert!(super::parse_u8("300", "test").is_err());
    }

    #[test]
    fn parse_f64_valid() {
        assert!((super::parse_f64("2.5", "test").unwrap() - 2.5_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_f64_invalid() {
        assert!(super::parse_f64("not_a_number", "test").is_err());
    }

    #[test]
    fn parse_csv_empty_value_rejected() {
        assert!(parse_csv("", "test").is_err());
        assert!(parse_csv("   , ,  ", "test").is_err());
    }

    // ── Config sub-struct defaults ──

    #[test]
    fn fsfs_config_default_has_all_sub_defaults() {
        let config = super::FsfsConfig::default();
        assert_eq!(config.discovery.roots, vec![".".to_string()]);
        assert_eq!(config.indexing.fast_model, "potion-multilingual-128M");
        assert_eq!(config.search.default_limit, 10);
        assert_eq!(config.pressure.profile, super::PressureProfile::Performance);
        assert_eq!(config.tui.theme, super::TuiTheme::Dark);
        assert_eq!(config.storage.index_dir, ".frankensearch");
        assert!(config.privacy.redact_file_contents_in_logs);
    }

    #[test]
    fn indexing_config_default_values() {
        let cfg = super::IndexingConfig::default();
        assert_eq!(cfg.fast_model, "potion-multilingual-128M");
        assert_eq!(cfg.quality_model, "all-MiniLM-L6-v2");
        assert_eq!(cfg.embedding_batch_size, 64);
        assert!(cfg.reindex_on_change);
        assert!(!cfg.watch_mode);
    }

    #[test]
    fn search_config_default_values() {
        let cfg = super::SearchConfig::default();
        assert_eq!(cfg.default_limit, 10);
        assert!((cfg.quality_weight - 0.7).abs() < f64::EPSILON);
        assert!((cfg.rrf_k - 60.0).abs() < f64::EPSILON);
        assert_eq!(cfg.quality_timeout_ms, 500);
        assert!(!cfg.fast_only);
        assert!(!cfg.explain);
    }

    #[test]
    fn pressure_config_default_values() {
        let cfg = super::PressureConfig::default();
        assert_eq!(cfg.cpu_ceiling_pct, 80);
        assert_eq!(cfg.memory_ceiling_mb, 2048);
        assert_eq!(cfg.sample_interval_ms, 2_000);
        assert_eq!(cfg.ewma_alpha_per_mille, 300);
        assert_eq!(cfg.anti_flap_readings, 3);
        assert_eq!(cfg.io_ceiling_bytes_per_sec, 100 * 1024 * 1024);
        assert_eq!(cfg.load_ceiling_per_mille, 3_000);
        assert_eq!(
            cfg.degradation_override,
            super::DegradationOverrideMode::Auto
        );
        assert!(!cfg.hard_pause_requested);
        assert!(!cfg.quality_circuit_open);
    }

    #[test]
    fn tui_config_default_values() {
        let cfg = super::TuiConfig::default();
        assert_eq!(cfg.frame_budget_ms, 16);
        assert!(cfg.show_explanations);
        assert_eq!(cfg.density, super::Density::Normal);
    }

    #[test]
    fn storage_config_default_values() {
        let cfg = super::StorageConfig::default();
        assert_eq!(cfg.index_dir, ".frankensearch");
        assert_eq!(cfg.db_path, "~/.local/share/fsfs/fsfs.db");
        assert_eq!(cfg.evidence_retention_days, 7);
        assert_eq!(cfg.summary_retention_days, 90);
        assert_eq!(cfg.disk_budget_bytes, None);
        assert!(!cfg.storage_pressure_emergency);
    }

    #[test]
    fn privacy_config_default_values() {
        let cfg = super::PrivacyConfig::default();
        assert!(cfg.redact_file_contents_in_logs);
        assert!(cfg.redact_paths_in_telemetry);
    }

    #[test]
    fn discovery_config_default_values() {
        let cfg = super::DiscoveryConfig::default();
        assert_eq!(cfg.roots, vec![".".to_string()]);
        assert!(!cfg.exclude_patterns.is_empty());
        assert!(cfg.exclude_patterns.contains(&".git".to_string()));
        assert!(cfg.exclude_patterns.contains(&"node_modules".to_string()));
        assert!(cfg.exclude_patterns.contains(&"target".to_string()));
        assert_eq!(cfg.text_selection_mode, super::TextSelectionMode::Blocklist);
        assert!(!cfg.binary_blocklist_extensions.is_empty());
        assert!(
            cfg.binary_blocklist_extensions
                .contains(&".exe".to_string())
        );
        assert_eq!(cfg.max_file_size_mb, 10);
        assert!(!cfg.follow_symlinks);
        assert!(cfg.mount_overrides.is_empty());
        assert!(!cfg.skip_network_mounts);
    }

    // ── CliOverrides::used_flags ──

    #[test]
    fn cli_overrides_used_flags_empty() {
        let cli = CliOverrides::default();
        assert!(cli.used_flags().is_empty());
    }

    #[test]
    fn cli_overrides_used_flags_all_set() {
        let cli = CliOverrides {
            roots: Some(vec!["src".into()]),
            exclude_patterns: Some(vec!["target".into()]),
            limit: Some(20),
            fast_only: Some(true),
            allow_background_indexing: Some(false),
            explain: Some(true),
            profile: Some(super::PressureProfile::Strict),
            degradation_override: Some(super::DegradationOverrideMode::ForcePaused),
            hard_pause_requested: Some(true),
            quality_circuit_open: Some(false),
            theme: Some(super::TuiTheme::Light),
            config_path: Some(Path::new("/tmp/config.toml").to_path_buf()),
        };
        let flags = cli.used_flags();
        assert_eq!(flags.len(), 12);
        assert!(flags.contains(&"--roots".to_string()));
        assert!(flags.contains(&"--exclude".to_string()));
        assert!(flags.contains(&"--limit".to_string()));
        assert!(flags.contains(&"--fast-only".to_string()));
        assert!(flags.contains(&"--watch-mode".to_string()));
        assert!(flags.contains(&"--explain".to_string()));
        assert!(flags.contains(&"--profile".to_string()));
        assert!(flags.contains(&"--degradation-override".to_string()));
        assert!(flags.contains(&"--hard-pause".to_string()));
        assert!(flags.contains(&"--quality-circuit-open".to_string()));
        assert!(flags.contains(&"--theme".to_string()));
        assert!(flags.contains(&"--config".to_string()));
    }

    // ── DiscoveryCandidate builders ──

    #[test]
    fn discovery_candidate_new_defaults() {
        let candidate = DiscoveryCandidate::new(Path::new("/foo.rs"), 1024);
        assert_eq!(candidate.byte_len, 1024);
        assert!(!candidate.is_symlink);
        assert!(candidate.mount_category.is_none());
    }

    #[test]
    fn discovery_candidate_with_symlink() {
        let candidate = DiscoveryCandidate::new(Path::new("/foo.rs"), 100).with_symlink(true);
        assert!(candidate.is_symlink);
    }

    #[test]
    fn discovery_candidate_with_mount_category() {
        let candidate = DiscoveryCandidate::new(Path::new("/foo.rs"), 100)
            .with_mount_category(crate::mount_info::FsCategory::Nfs);
        assert_eq!(
            candidate.mount_category,
            Some(crate::mount_info::FsCategory::Nfs)
        );
    }

    // ── RootDiscoveryDecision ──

    #[test]
    fn root_discovery_decision_include_method() {
        let included = super::RootDiscoveryDecision {
            scope: DiscoveryScopeDecision::Include,
            reason_codes: vec![],
        };
        assert!(included.include());

        let excluded = super::RootDiscoveryDecision {
            scope: DiscoveryScopeDecision::Exclude,
            reason_codes: vec![],
        };
        assert!(!excluded.include());
    }

    // ── Discovery evaluate_root ──

    #[test]
    fn evaluate_root_empty_path_rejected() {
        let config = super::DiscoveryConfig::default();
        let decision = config.evaluate_root(Path::new(""), None);
        assert!(!decision.include());
        assert!(
            decision
                .reason_codes
                .contains(&"discovery.root.rejected".to_string())
        );
    }

    #[test]
    fn evaluate_root_virtual_mount_rejected() {
        let config = super::DiscoveryConfig::default();
        let decision = config.evaluate_root(
            Path::new("/proc"),
            Some(crate::mount_info::FsCategory::Virtual),
        );
        assert!(!decision.include());
    }

    #[test]
    fn evaluate_root_excluded_pattern_rejected() {
        let config = super::DiscoveryConfig::default();
        let decision = config.evaluate_root(Path::new("/project/node_modules"), None);
        assert!(!decision.include());
    }

    #[test]
    fn evaluate_root_normal_accepted() {
        let config = super::DiscoveryConfig::default();
        let decision = config.evaluate_root(Path::new("/home/user/src"), None);
        assert!(decision.include());
    }

    // ── Discovery evaluate_candidate edge cases ──

    #[test]
    fn evaluate_candidate_empty_path_excluded() {
        let config = super::DiscoveryConfig::default();
        let candidate = DiscoveryCandidate::new(Path::new(""), 100);
        let decision = config.evaluate_candidate(&candidate);
        assert_eq!(decision.scope, DiscoveryScopeDecision::Exclude);
        assert_eq!(decision.utility_score, i32::MIN);
    }

    #[test]
    fn evaluate_candidate_virtual_mount_excluded() {
        let config = super::DiscoveryConfig::default();
        let candidate = DiscoveryCandidate::new(Path::new("/proc/cpuinfo"), 100)
            .with_mount_category(crate::mount_info::FsCategory::Virtual);
        let decision = config.evaluate_candidate(&candidate);
        assert_eq!(decision.scope, DiscoveryScopeDecision::Exclude);
        assert_eq!(decision.ingestion_class, IngestionClass::Skip);
    }

    #[test]
    fn evaluate_candidate_symlink_excluded_when_not_following() {
        let config = super::DiscoveryConfig::default();
        let candidate =
            DiscoveryCandidate::new(Path::new("/home/user/link.rs"), 100).with_symlink(true);
        let decision = config.evaluate_candidate(&candidate);
        assert_eq!(decision.scope, DiscoveryScopeDecision::Exclude);
        assert_eq!(decision.ingestion_class, IngestionClass::Skip);
    }

    #[test]
    fn evaluate_candidate_extremely_large_file_excluded() {
        let config = super::DiscoveryConfig::default();
        let max = config.max_file_size_mb as u64 * 1024 * 1024;
        let candidate =
            DiscoveryCandidate::new(Path::new("/home/user/huge.rs"), max.saturating_mul(4) + 1);
        let decision = config.evaluate_candidate(&candidate);
        assert_eq!(decision.scope, DiscoveryScopeDecision::Exclude);
        assert_eq!(decision.ingestion_class, IngestionClass::Skip);
    }

    #[test]
    fn evaluate_candidate_high_signal_filename_boosts_score() {
        let config = super::DiscoveryConfig::default();
        let candidate = DiscoveryCandidate::new(Path::new("/home/user/project/Cargo.toml"), 512);
        let decision = config.evaluate_candidate(&candidate);
        assert!(decision.utility_score >= 70);
        assert_eq!(
            decision.ingestion_class,
            IngestionClass::FullSemanticLexical
        );
    }

    #[test]
    fn evaluate_candidate_low_utility_filename_reduces_score() {
        let config = super::DiscoveryConfig::default();
        let candidate =
            DiscoveryCandidate::new(Path::new("/home/user/project/package-lock.json"), 4096);
        let decision = config.evaluate_candidate(&candidate);
        // Base 50 + 30 (high ext json) - 20 (low utility filename) = 60 < 70
        assert!(decision.utility_score < 70);
    }

    #[test]
    fn evaluate_candidate_network_mount_reduces_score() {
        let config = super::DiscoveryConfig::default();
        let candidate = DiscoveryCandidate::new(Path::new("/mnt/nfs/src/main.rs"), 1024)
            .with_mount_category(crate::mount_info::FsCategory::Nfs);
        let decision = config.evaluate_candidate(&candidate);
        // 50 + 30 (high ext) - 10 (network) = 70
        assert!(decision.utility_score <= 70);
    }

    // ── MountPolicyEntry ──

    #[test]
    fn mount_policy_entry_to_mount_override() {
        let entry = super::MountPolicyEntry {
            mount_point: "/mnt/nfs".into(),
            category: Some(crate::mount_info::FsCategory::Nfs),
            enabled: Some(true),
            change_detection: None,
            stat_timeout_ms: Some(5000),
            max_concurrent_io: Some(4),
            poll_interval_secs: Some(30),
        };
        let mount_override = entry.to_mount_override();
        assert_eq!(
            mount_override.category,
            Some(crate::mount_info::FsCategory::Nfs)
        );
        assert_eq!(mount_override.enabled, Some(true));
        assert_eq!(mount_override.stat_timeout_ms, Some(5000));
        assert_eq!(mount_override.max_concurrent_io, Some(4));
        assert_eq!(mount_override.poll_interval_secs, Some(30));
    }

    #[test]
    fn mount_override_map_builds_correctly() {
        let config = super::DiscoveryConfig {
            mount_overrides: vec![
                super::MountPolicyEntry {
                    mount_point: "/mnt/a".into(),
                    category: None,
                    enabled: Some(false),
                    change_detection: None,
                    stat_timeout_ms: None,
                    max_concurrent_io: None,
                    poll_interval_secs: None,
                },
                super::MountPolicyEntry {
                    mount_point: "/mnt/b".into(),
                    category: None,
                    enabled: Some(true),
                    change_detection: None,
                    stat_timeout_ms: None,
                    max_concurrent_io: None,
                    poll_interval_secs: None,
                },
            ],
            ..super::DiscoveryConfig::default()
        };
        let map = config.mount_override_map();
        assert_eq!(map.len(), 2);
        assert_eq!(map["/mnt/a"].enabled, Some(false));
        assert_eq!(map["/mnt/b"].enabled, Some(true));
    }

    // ── Helper function labels ──

    #[test]
    fn override_source_label_all_variants() {
        assert_eq!(
            super::override_source_label(ProfileOverrideSource::Cli),
            "cli"
        );
        assert_eq!(
            super::override_source_label(ProfileOverrideSource::Env),
            "env"
        );
        assert_eq!(
            super::override_source_label(ProfileOverrideSource::Config),
            "config"
        );
    }

    #[test]
    fn config_source_for_override_all_variants() {
        assert_eq!(
            super::config_source_for_override(ProfileOverrideSource::Cli),
            super::ConfigSource::Cli
        );
        assert_eq!(
            super::config_source_for_override(ProfileOverrideSource::Env),
            super::ConfigSource::Env
        );
        assert_eq!(
            super::config_source_for_override(ProfileOverrideSource::Config),
            super::ConfigSource::File
        );
    }

    #[test]
    fn profile_field_path_all_variants() {
        assert_eq!(
            super::profile_field_path(PressureProfileField::SchedulerMode),
            "pressure.profile.scheduler_mode"
        );
        assert_eq!(
            super::profile_field_path(PressureProfileField::MaxEmbedConcurrency),
            "pressure.profile.max_embed_concurrency"
        );
        assert_eq!(
            super::profile_field_path(PressureProfileField::MaxIndexConcurrency),
            "pressure.profile.max_index_concurrency"
        );
        assert_eq!(
            super::profile_field_path(PressureProfileField::QualityEnabled),
            "pressure.profile.quality_enabled"
        );
        assert_eq!(
            super::profile_field_path(PressureProfileField::AllowBackgroundIndexing),
            "pressure.profile.allow_background_indexing"
        );
    }

    // ── Validation boundary tests ──

    #[test]
    fn validate_quality_weight_above_one_rejected() {
        assert_invalid_field("[search]\nquality_weight = 1.1\n", "search.quality_weight");
    }

    #[test]
    fn validate_quality_weight_negative_rejected() {
        assert_invalid_field("[search]\nquality_weight = -0.1\n", "search.quality_weight");
    }

    #[test]
    fn validate_rrf_k_nan_rejected() {
        assert_invalid_field("[search]\nrrf_k = nan\n", "search.rrf_k");
    }

    #[test]
    fn validate_empty_index_dir_rejected() {
        assert_invalid_field("[storage]\nindex_dir = \"  \"\n", "storage.index_dir");
    }

    #[test]
    fn validate_cpu_ceiling_pct_101_rejected() {
        assert_invalid_field(
            "[pressure]\ncpu_ceiling_pct = 101\n",
            "pressure.cpu_ceiling_pct",
        );
    }

    #[test]
    fn validate_anti_flap_readings_33_rejected() {
        assert_invalid_field(
            "[pressure]\nanti_flap_readings = 33\n",
            "pressure.anti_flap_readings",
        );
    }

    #[test]
    fn validate_ewma_alpha_1001_rejected() {
        assert_invalid_field(
            "[pressure]\newma_alpha_per_mille = 1001\n",
            "pressure.ewma_alpha_per_mille",
        );
    }

    #[test]
    fn validate_frame_budget_ms_201_rejected() {
        assert_invalid_field("[tui]\nframe_budget_ms = 201\n", "tui.frame_budget_ms");
    }

    #[test]
    fn validate_evidence_retention_3651_rejected() {
        // summary_retention_days defaults to 90, which is < 3651,
        // so the summary < evidence check fires first.
        assert_invalid_field(
            "[storage]\nevidence_retention_days = 3651\nsummary_retention_days = 3651\n",
            "storage.evidence_retention_days",
        );
    }

    #[test]
    fn validate_summary_retention_3651_rejected() {
        assert_invalid_field(
            "[storage]\nsummary_retention_days = 3651\n",
            "storage.summary_retention_days",
        );
    }

    #[test]
    fn validate_max_file_size_mb_1025_rejected() {
        assert_invalid_field(
            "[discovery]\nmax_file_size_mb = 1025\n",
            "discovery.max_file_size_mb",
        );
    }

    #[test]
    fn validate_embedding_batch_size_4097_rejected() {
        assert_invalid_field(
            "[indexing]\nembedding_batch_size = 4097\n",
            "indexing.embedding_batch_size",
        );
    }

    #[test]
    fn validate_default_limit_201_rejected() {
        assert_invalid_field("[search]\ndefault_limit = 201\n", "search.default_limit");
    }

    // ── ConfigLoadResult::to_loaded_event ──

    #[test]
    fn to_loaded_event_collects_reason_codes() {
        let result = load_from_str(
            None,
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect("load defaults");
        let event = result.to_loaded_event();
        assert_eq!(event.event, "config_loaded");
        assert!(!event.reason_codes.is_empty());
        // reason_codes should be sorted and deduped
        let mut sorted = event.reason_codes.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(event.reason_codes, sorted);
    }

    // ── Unknown section warning ──

    #[test]
    fn unknown_section_produces_warning() {
        let file = "[fantasy]\nfoo = 42\n";
        let result = load_from_str(
            Some(file),
            None,
            &HashMap::new(),
            &CliOverrides::default(),
            home(),
        )
        .expect("load config");
        assert!(result.warnings.iter().any(|w| {
            w.reason_code == "config.unknown_key.warning" && w.field == "config.fantasy"
        }));
    }

    // ── expand_home_prefix ──

    #[test]
    fn expand_home_prefix_tilde() {
        let expanded =
            super::expand_home_prefix(Path::new("~/docs/readme.md"), Path::new("/home/user"));
        assert_eq!(expanded, Path::new("/home/user/docs/readme.md"));
    }

    #[test]
    fn expand_home_prefix_no_tilde() {
        let expanded =
            super::expand_home_prefix(Path::new("/absolute/path"), Path::new("/home/user"));
        assert_eq!(expanded, Path::new("/absolute/path"));
    }

    #[test]
    fn expand_home_prefix_relative_no_tilde() {
        let expanded =
            super::expand_home_prefix(Path::new("relative/path"), Path::new("/home/user"));
        assert_eq!(expanded, Path::new("relative/path"));
    }

    // ── Constants sanity checks ──

    #[test]
    fn pressure_profile_version_is_nonzero() {
        assert_ne!(PRESSURE_PROFILE_VERSION, 0);
    }

    #[test]
    fn profile_precedence_chain_length() {
        assert_eq!(super::PROFILE_PRECEDENCE_CHAIN.len(), 5);
    }

    // ── Env override integration ──

    #[test]
    fn env_override_pressure_settings() {
        let env = HashMap::from([
            (
                "FRANKENSEARCH_PRESSURE_SAMPLE_INTERVAL_MS".into(),
                "500".into(),
            ),
            (
                "FRANKENSEARCH_PRESSURE_EWMA_ALPHA_PER_MILLE".into(),
                "200".into(),
            ),
            (
                "FRANKENSEARCH_PRESSURE_ANTI_FLAP_READINGS".into(),
                "5".into(),
            ),
            (
                "FRANKENSEARCH_PRESSURE_IO_CEILING_BYTES_PER_SEC".into(),
                "52428800".into(),
            ),
            (
                "FRANKENSEARCH_PRESSURE_LOAD_CEILING_PER_MILLE".into(),
                "2000".into(),
            ),
        ]);
        let result =
            load_from_str(None, None, &env, &CliOverrides::default(), home()).expect("load");
        assert_eq!(result.config.pressure.sample_interval_ms, 500);
        assert_eq!(result.config.pressure.ewma_alpha_per_mille, 200);
        assert_eq!(result.config.pressure.anti_flap_readings, 5);
        assert_eq!(result.config.pressure.io_ceiling_bytes_per_sec, 52_428_800);
        assert_eq!(result.config.pressure.load_ceiling_per_mille, 2000);
    }

    #[test]
    fn env_override_degradation_override() {
        let env = HashMap::from([(
            "FRANKENSEARCH_PRESSURE_DEGRADATION_OVERRIDE".into(),
            "force_paused".into(),
        )]);
        let result =
            load_from_str(None, None, &env, &CliOverrides::default(), home()).expect("load");
        assert_eq!(
            result.config.pressure.degradation_override,
            super::DegradationOverrideMode::ForcePaused
        );
    }

    #[test]
    fn env_override_tui_theme() {
        let env = HashMap::from([("FRANKENSEARCH_TUI_THEME".into(), "light".into())]);
        let result =
            load_from_str(None, None, &env, &CliOverrides::default(), home()).expect("load");
        assert_eq!(result.config.tui.theme, super::TuiTheme::Light);
    }

    #[test]
    fn env_override_privacy_redact() {
        let env = HashMap::from([(
            "FRANKENSEARCH_PRIVACY_REDACT_PATHS_IN_TELEMETRY".into(),
            "false".into(),
        )]);
        let result =
            load_from_str(None, None, &env, &CliOverrides::default(), home()).expect("load");
        assert!(!result.config.privacy.redact_paths_in_telemetry);
    }

    #[test]
    fn env_override_storage_db_path() {
        let env = HashMap::from([(
            "FRANKENSEARCH_STORAGE_DB_PATH".into(),
            "/tmp/custom.db".into(),
        )]);
        let result =
            load_from_str(None, None, &env, &CliOverrides::default(), home()).expect("load");
        assert_eq!(result.config.storage.db_path, "/tmp/custom.db");
    }

    #[test]
    fn env_override_storage_budget_and_emergency() {
        let env = HashMap::from([
            (
                "FRANKENSEARCH_STORAGE_DISK_BUDGET_BYTES".into(),
                "123456".into(),
            ),
            (
                "FRANKENSEARCH_STORAGE_PRESSURE_EMERGENCY".into(),
                "true".into(),
            ),
        ]);
        let result =
            load_from_str(None, None, &env, &CliOverrides::default(), home()).expect("load");
        assert_eq!(result.config.storage.disk_budget_bytes, Some(123_456));
        assert!(result.config.storage.storage_pressure_emergency);
    }

    #[test]
    fn env_override_search_explain() {
        let env = HashMap::from([("FRANKENSEARCH_SEARCH_EXPLAIN".into(), "true".into())]);
        let result =
            load_from_str(None, None, &env, &CliOverrides::default(), home()).expect("load");
        assert!(result.config.search.explain);
    }

    #[test]
    fn env_override_discovery_roots() {
        let env = HashMap::from([(
            "FRANKENSEARCH_DISCOVERY_ROOTS".into(),
            "src, lib, tests".into(),
        )]);
        let result =
            load_from_str(None, None, &env, &CliOverrides::default(), home()).expect("load");
        assert_eq!(result.config.discovery.roots, vec!["src", "lib", "tests"]);
    }

    #[test]
    fn env_override_invalid_degradation_override_rejected() {
        let env = HashMap::from([(
            "FRANKENSEARCH_PRESSURE_DEGRADATION_OVERRIDE".into(),
            "bogus".into(),
        )]);
        let err = load_from_str(None, None, &env, &CliOverrides::default(), home())
            .expect_err("should reject invalid degradation override");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn env_override_invalid_tui_theme_rejected() {
        let env = HashMap::from([("FRANKENSEARCH_TUI_THEME".into(), "neon".into())]);
        let err = load_from_str(None, None, &env, &CliOverrides::default(), home())
            .expect_err("should reject invalid tui theme");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn env_override_invalid_pressure_profile_rejected() {
        let env = HashMap::from([("FRANKENSEARCH_PRESSURE_PROFILE".into(), "turbo".into())]);
        let err = load_from_str(None, None, &env, &CliOverrides::default(), home())
            .expect_err("should reject invalid pressure profile");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    // ─── bd-12af tests end ───
}
