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
            category: None,
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
            roots: vec!["$HOME".into()],
            exclude_patterns: vec![
                ".git".into(),
                "node_modules".into(),
                "target".into(),
                "__pycache__".into(),
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
            model_dir: "~/.cache/frankensearch/models".into(),
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
            default_limit: 20,
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
    pub db_path: String,
    pub evidence_retention_days: u16,
    pub summary_retention_days: u16,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            db_path: "~/.local/share/fsfs/fsfs.db".into(),
            evidence_retention_days: 7,
            summary_retention_days: 90,
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
    db_path: Option<String>,
    evidence_retention_days: Option<u16>,
    summary_retention_days: Option<u16>,
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
pub fn default_config_file_path(home_dir: &Path) -> PathBuf {
    if let Some(xdg_config_home) = std::env::var_os("XDG_CONFIG_HOME") {
        return PathBuf::from(xdg_config_home)
            .join("fsfs")
            .join("config.toml");
    }

    home_dir.join(".config").join("fsfs").join("config.toml")
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
    let mut config = FsfsConfig::default();
    let mut warnings = Vec::new();
    let mut file_profile_overrides = ProfileSourceOverrides::default();

    if let Some(config_toml) = config_toml {
        warnings.extend(collect_unknown_key_warnings(config_toml)?);
        let patch: FsfsConfigPatch =
            toml::from_str(config_toml).map_err(|error| SearchError::InvalidConfig {
                field: "config_file".into(),
                value: "<toml>".into(),
                reason: error.to_string(),
            })?;
        file_profile_overrides = profile_overrides_from_file_patch(&patch);
        apply_patch(&mut config, patch);
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
        config_file_used: config_file_path.map(Path::to_path_buf),
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
        if let Some(db_path) = storage.db_path {
            config.storage.db_path = db_path;
        }
        if let Some(evidence_retention_days) = storage.evidence_retention_days {
            config.storage.evidence_retention_days = evidence_retention_days;
        }
        if let Some(summary_retention_days) = storage.summary_retention_days {
            config.storage.summary_retention_days = summary_retention_days;
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

fn apply_env_overrides(
    config: &mut FsfsConfig,
    env: &HashMap<String, String, impl BuildHasher>,
) -> SearchResult<EnvOverrideReport> {
    let mut keys_used = Vec::new();
    let mut profile_overrides = ProfileSourceOverrides::default();

    if let Some(value) = env.get("FSFS_DISCOVERY_ROOTS") {
        config.discovery.roots = parse_csv(value, "discovery.roots")?;
        keys_used.push("FSFS_DISCOVERY_ROOTS".into());
    }

    if let Some(value) = env.get("FSFS_DISCOVERY_EXCLUDE_PATTERNS") {
        config.discovery.exclude_patterns = parse_csv(value, "discovery.exclude_patterns")?;
        keys_used.push("FSFS_DISCOVERY_EXCLUDE_PATTERNS".into());
    }

    if let Some(value) = env.get("FSFS_SEARCH_DEFAULT_LIMIT") {
        config.search.default_limit = parse_usize(value, "search.default_limit")?;
        keys_used.push("FSFS_SEARCH_DEFAULT_LIMIT".into());
    }

    if let Some(value) = env.get("FSFS_SEARCH_FAST_ONLY") {
        let fast_only = parse_bool(value, "search.fast_only")?;
        config.search.fast_only = fast_only;
        profile_overrides.quality_enabled = Some(!fast_only);
        keys_used.push("FSFS_SEARCH_FAST_ONLY".into());
    }

    if let Some(value) = env.get("FSFS_INDEXING_WATCH_MODE") {
        let watch_mode = parse_bool(value, "indexing.watch_mode")?;
        config.indexing.watch_mode = watch_mode;
        profile_overrides.allow_background_indexing = Some(watch_mode);
        keys_used.push("FSFS_INDEXING_WATCH_MODE".into());
    }

    if let Some(value) = env.get("FSFS_SEARCH_EXPLAIN") {
        config.search.explain = parse_bool(value, "search.explain")?;
        keys_used.push("FSFS_SEARCH_EXPLAIN".into());
    }

    if let Some(value) = env.get("FSFS_PRESSURE_PROFILE") {
        config.pressure.profile =
            PressureProfile::from_str(value).map_err(|()| SearchError::InvalidConfig {
                field: "pressure.profile".into(),
                value: value.clone(),
                reason: "expected strict|performance|degraded".into(),
            })?;
        keys_used.push("FSFS_PRESSURE_PROFILE".into());
    }

    if let Some(value) = env.get("FSFS_PRESSURE_SAMPLE_INTERVAL_MS") {
        config.pressure.sample_interval_ms = parse_u64(value, "pressure.sample_interval_ms")?;
        keys_used.push("FSFS_PRESSURE_SAMPLE_INTERVAL_MS".into());
    }

    if let Some(value) = env.get("FSFS_PRESSURE_EWMA_ALPHA_PER_MILLE") {
        config.pressure.ewma_alpha_per_mille = parse_u16(value, "pressure.ewma_alpha_per_mille")?;
        keys_used.push("FSFS_PRESSURE_EWMA_ALPHA_PER_MILLE".into());
    }

    if let Some(value) = env.get("FSFS_PRESSURE_ANTI_FLAP_READINGS") {
        config.pressure.anti_flap_readings = parse_u8(value, "pressure.anti_flap_readings")?;
        keys_used.push("FSFS_PRESSURE_ANTI_FLAP_READINGS".into());
    }

    if let Some(value) = env.get("FSFS_PRESSURE_IO_CEILING_BYTES_PER_SEC") {
        config.pressure.io_ceiling_bytes_per_sec =
            parse_u64(value, "pressure.io_ceiling_bytes_per_sec")?;
        keys_used.push("FSFS_PRESSURE_IO_CEILING_BYTES_PER_SEC".into());
    }

    if let Some(value) = env.get("FSFS_PRESSURE_LOAD_CEILING_PER_MILLE") {
        config.pressure.load_ceiling_per_mille =
            parse_u16(value, "pressure.load_ceiling_per_mille")?;
        keys_used.push("FSFS_PRESSURE_LOAD_CEILING_PER_MILLE".into());
    }

    if let Some(value) = env.get("FSFS_PRESSURE_DEGRADATION_OVERRIDE") {
        config.pressure.degradation_override =
            DegradationOverrideMode::from_str(value).map_err(|()| SearchError::InvalidConfig {
                field: "pressure.degradation_override".into(),
                value: value.clone(),
                reason: "expected auto|full|embed_deferred|lexical_only|metadata_only|paused"
                    .into(),
            })?;
        keys_used.push("FSFS_PRESSURE_DEGRADATION_OVERRIDE".into());
    }

    if let Some(value) = env.get("FSFS_PRESSURE_HARD_PAUSE_REQUESTED") {
        config.pressure.hard_pause_requested = parse_bool(value, "pressure.hard_pause_requested")?;
        keys_used.push("FSFS_PRESSURE_HARD_PAUSE_REQUESTED".into());
    }

    if let Some(value) = env.get("FSFS_PRESSURE_QUALITY_CIRCUIT_OPEN") {
        config.pressure.quality_circuit_open = parse_bool(value, "pressure.quality_circuit_open")?;
        keys_used.push("FSFS_PRESSURE_QUALITY_CIRCUIT_OPEN".into());
    }

    if let Some(value) = env.get("FSFS_TUI_THEME") {
        config.tui.theme = TuiTheme::from_str(value).map_err(|()| SearchError::InvalidConfig {
            field: "tui.theme".into(),
            value: value.clone(),
            reason: "expected auto|light|dark".into(),
        })?;
        keys_used.push("FSFS_TUI_THEME".into());
    }

    if let Some(value) = env.get("FSFS_PRIVACY_REDACT_PATHS_IN_TELEMETRY") {
        config.privacy.redact_paths_in_telemetry =
            parse_bool(value, "privacy.redact_paths_in_telemetry")?;
        keys_used.push("FSFS_PRIVACY_REDACT_PATHS_IN_TELEMETRY".into());
    }

    if let Some(value) = env.get("FSFS_STORAGE_DB_PATH") {
        config.storage.db_path.clone_from(value);
        keys_used.push("FSFS_STORAGE_DB_PATH".into());
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
                "db_path",
                "evidence_retention_days",
                "summary_retention_days",
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

    use super::{
        CliOverrides, DiscoveryCandidate, DiscoveryScopeDecision, IngestionClass,
        PRESSURE_PROFILE_VERSION, PressureProfileField, ProfileOverrideSource,
        default_config_file_path, load_from_sources, load_from_str,
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
        let config_file = home_dir.join(".config").join("fsfs").join("config.toml");
        fs::create_dir_all(config_file.parent().expect("parent")).expect("mkdir");
        fs::write(&config_file, "[search]\ndefault_limit = 42\n").expect("write");

        let result = load_from_sources(
            Some(Path::new("~/.config/fsfs/config.toml")),
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
        assert!(rendered.contains("/fsfs/config.toml"));
    }
}
