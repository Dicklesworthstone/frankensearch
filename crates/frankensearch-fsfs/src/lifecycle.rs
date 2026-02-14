//! Daemon lifecycle, health checks, PID file management, and self-supervision.
//!
//! Provides types and logic for running fsfs as a long-lived background process.
//! The lifecycle model covers:
//!
//! 1. **PID file management** — Prevents duplicate daemon instances using
//!    file-based locking with stale PID detection.
//! 2. **Subsystem health tracking** — Each pipeline stage reports its health
//!    via [`SubsystemHealth`]; the daemon aggregates into [`DaemonStatus`].
//! 3. **Watchdog supervision** — Detects crashed subsystems and manages
//!    restart attempts with exponential backoff.
//! 4. **Resource limits** — Configurable caps on threads, memory, and open files.
//! 5. **Status reporting** — Machine-readable status for CLI `fsfs status` command.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

// ─── Daemon Phase ───────────────────────────────────────────────────────────

/// Lifecycle phase of the fsfs daemon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonPhase {
    /// Process starting, acquiring lock file, validating config.
    Initializing,
    /// Performing index integrity checks and loading existing indices.
    StartingUp,
    /// All subsystems running normally.
    Running,
    /// One or more subsystems degraded but service still available.
    Degraded,
    /// Graceful shutdown in progress.
    ShuttingDown,
    /// Process has exited.
    Stopped,
}

impl DaemonPhase {
    /// Whether the daemon is accepting work in this phase.
    #[must_use]
    pub const fn is_accepting_work(&self) -> bool {
        matches!(self, Self::Running | Self::Degraded)
    }

    /// Whether the daemon is in a terminal state.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Stopped)
    }
}

impl std::fmt::Display for DaemonPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Initializing => "initializing",
            Self::StartingUp => "starting_up",
            Self::Running => "running",
            Self::Degraded => "degraded",
            Self::ShuttingDown => "shutting_down",
            Self::Stopped => "stopped",
        };
        f.write_str(s)
    }
}

// ─── Subsystem Health ───────────────────────────────────────────────────────

/// Known subsystem identifiers within the fsfs daemon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubsystemId {
    /// File system crawler/scanner.
    Crawler,
    /// Fast-tier embedding worker.
    EmbedFast,
    /// Quality-tier embedding worker.
    EmbedQuality,
    /// Lexical (Tantivy) indexer.
    LexicalIndexer,
    /// Query server (search requests).
    QueryServer,
    /// Index refresh/cache update worker.
    RefreshWorker,
}

impl SubsystemId {
    /// Human-readable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Crawler => "crawler",
            Self::EmbedFast => "embed_fast",
            Self::EmbedQuality => "embed_quality",
            Self::LexicalIndexer => "lexical_indexer",
            Self::QueryServer => "query_server",
            Self::RefreshWorker => "refresh_worker",
        }
    }
}

impl std::fmt::Display for SubsystemId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Health status of a single subsystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    /// Not yet started.
    Pending,
    /// Running normally.
    Healthy,
    /// Running but with issues (e.g., high error rate).
    Degraded,
    /// Crashed or stopped unexpectedly.
    Failed,
    /// Stopped intentionally (during shutdown).
    Stopped,
}

impl HealthStatus {
    /// Whether this status counts as "alive" (not crashed/stopped).
    #[must_use]
    pub const fn is_alive(&self) -> bool {
        matches!(self, Self::Pending | Self::Healthy | Self::Degraded)
    }
}

/// Detailed health report for a subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsystemHealth {
    /// Which subsystem.
    pub id: SubsystemId,
    /// Current health status.
    pub status: HealthStatus,
    /// Number of times this subsystem has been restarted.
    pub restart_count: u32,
    /// Last error message, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    /// When the subsystem last reported healthy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_healthy_at: Option<String>,
    /// When the subsystem entered its current status.
    pub status_since: String,
}

// ─── Daemon Status ──────────────────────────────────────────────────────────

/// Complete daemon status snapshot for reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatus {
    /// Current lifecycle phase.
    pub phase: DaemonPhase,
    /// Process ID.
    pub pid: u32,
    /// When the daemon started.
    pub started_at: String,
    /// How long the daemon has been running.
    pub uptime_secs: u64,
    /// Per-subsystem health reports.
    pub subsystems: Vec<SubsystemHealth>,
    /// Total errors across all subsystems.
    pub total_errors: u64,
    /// Total panics caught and recovered.
    pub total_panics_recovered: u64,
    /// Resource usage snapshot.
    pub resources: ResourceUsage,
}

/// Resource usage snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Resident set size in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rss_bytes: Option<u64>,
    /// Number of active threads.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_count: Option<u32>,
    /// Number of open file descriptors.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub open_fds: Option<u32>,
}

// ─── PID File Management ────────────────────────────────────────────────────

/// Contents of a PID file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PidFileContents {
    /// Process ID.
    pub pid: u32,
    /// When the daemon started.
    pub started_at_ms: u64,
    /// Hostname.
    pub hostname: String,
    /// Version string.
    pub version: String,
}

impl PidFileContents {
    /// Create contents for the current process.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn current(version: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            pid: std::process::id(),
            started_at_ms: now,
            hostname: hostname(),
            version: version.into(),
        }
    }

    /// Whether the PID in this file is still alive on the local host.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        if self.hostname != hostname() {
            return true; // Can't verify cross-host.
        }
        Path::new(&format!("/proc/{}", self.pid)).exists()
    }
}

/// Manage a PID file at the given path.
pub struct PidFile {
    path: PathBuf,
}

impl PidFile {
    /// Create a PID file manager for the given path.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Default PID file path.
    #[must_use]
    pub fn default_path() -> PathBuf {
        let runtime_dir = dirs::runtime_dir()
            .or_else(|| dirs::cache_dir().map(|d| d.join("run")))
            .unwrap_or_else(|| PathBuf::from("/tmp"));
        runtime_dir.join("fsfs.pid")
    }

    /// Attempt to acquire the PID file. Returns `Ok(())` if acquired,
    /// `Err` if another live process holds it.
    ///
    /// # Errors
    ///
    /// Returns error if PID file is held by a live process or I/O fails.
    pub fn acquire(&self, version: &str) -> std::io::Result<()> {
        // Check for existing PID file.
        if let Ok(contents) = read_pid_file(&self.path) {
            if contents.is_alive() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AddrInUse,
                    format!(
                        "fsfs daemon already running (PID {}, started at {})",
                        contents.pid, contents.started_at_ms
                    ),
                ));
            }
            // Stale PID file — remove it.
            warn!(
                target: "frankensearch.fsfs.lifecycle",
                pid = contents.pid,
                "Removing stale PID file (process is dead)"
            );
            let _ = std::fs::remove_file(&self.path);
        }

        // Ensure parent directory exists.
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Write our PID file.
        let contents = PidFileContents::current(version);
        let json = serde_json::to_string_pretty(&contents)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(&self.path, json.as_bytes())?;

        info!(
            target: "frankensearch.fsfs.lifecycle",
            pid = contents.pid,
            path = %self.path.display(),
            "PID file acquired"
        );
        Ok(())
    }

    /// Release the PID file (remove it).
    pub fn release(&self) {
        match std::fs::remove_file(&self.path) {
            Ok(()) => {
                debug!(
                    target: "frankensearch.fsfs.lifecycle",
                    path = %self.path.display(),
                    "PID file released"
                );
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Already gone — fine.
            }
            Err(e) => {
                warn!(
                    target: "frankensearch.fsfs.lifecycle",
                    error = %e,
                    path = %self.path.display(),
                    "Failed to remove PID file"
                );
            }
        }
    }

    /// Read the current PID file contents.
    ///
    /// # Errors
    ///
    /// Returns error if the file doesn't exist or can't be parsed.
    pub fn read(&self) -> std::io::Result<PidFileContents> {
        read_pid_file(&self.path)
    }

    /// Path to the PID file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for PidFile {
    fn drop(&mut self) {
        self.release();
    }
}

fn read_pid_file(path: &Path) -> std::io::Result<PidFileContents> {
    let contents = std::fs::read_to_string(path)?;
    serde_json::from_str(&contents)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

// ─── Watchdog / Supervision ─────────────────────────────────────────────────

/// Configuration for subsystem watchdog/restart behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchdogConfig {
    /// Initial backoff before restarting a failed subsystem.
    #[serde(default = "default_initial_backoff_ms")]
    pub initial_backoff_ms: u64,
    /// Maximum backoff between restart attempts.
    #[serde(default = "default_max_backoff_ms")]
    pub max_backoff_ms: u64,
    /// Backoff multiplier.
    #[serde(default = "default_backoff_multiplier")]
    pub backoff_multiplier: f64,
    /// Maximum restart attempts before giving up on a subsystem.
    #[serde(default = "default_max_restarts")]
    pub max_restarts: u32,
    /// Health check interval.
    #[serde(default = "default_health_check_interval_ms")]
    pub health_check_interval_ms: u64,
}

const fn default_initial_backoff_ms() -> u64 {
    5_000
}
const fn default_max_backoff_ms() -> u64 {
    60_000
}
const fn default_backoff_multiplier() -> f64 {
    2.0
}
const fn default_max_restarts() -> u32 {
    5
}
const fn default_health_check_interval_ms() -> u64 {
    10_000
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            initial_backoff_ms: default_initial_backoff_ms(),
            max_backoff_ms: default_max_backoff_ms(),
            backoff_multiplier: default_backoff_multiplier(),
            max_restarts: default_max_restarts(),
            health_check_interval_ms: default_health_check_interval_ms(),
        }
    }
}

impl WatchdogConfig {
    /// Compute the backoff delay for the given restart attempt (0-indexed).
    #[must_use]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn backoff_delay(&self, attempt: u32) -> Duration {
        let multiplier = self.backoff_multiplier.powi(attempt as i32);
        let delay_ms = (self.initial_backoff_ms as f64) * multiplier;
        let capped_ms = delay_ms.min(self.max_backoff_ms as f64);
        Duration::from_millis(capped_ms as u64)
    }

    /// Whether the restart limit has been reached.
    #[must_use]
    pub const fn is_exhausted(&self, attempt: u32) -> bool {
        attempt >= self.max_restarts
    }
}

// ─── Resource Limits ────────────────────────────────────────────────────────

/// Configurable resource limits for the daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum RSS in bytes (0 = unlimited).
    #[serde(default)]
    pub max_rss_bytes: u64,
    /// Maximum number of threads (0 = unlimited).
    #[serde(default)]
    pub max_threads: u32,
    /// Maximum open file descriptors (0 = unlimited).
    #[serde(default)]
    pub max_open_fds: u32,
    /// Maximum disk usage for index data in bytes (0 = unlimited).
    #[serde(default)]
    pub max_index_bytes: u64,
}

impl Default for ResourceLimits {
    #[allow(clippy::derivable_impls)]
    fn default() -> Self {
        Self {
            max_rss_bytes: 0,
            max_threads: 0,
            max_open_fds: 0,
            max_index_bytes: 0,
        }
    }
}

impl ResourceLimits {
    /// Check whether the given usage exceeds any configured limit.
    #[must_use]
    pub fn check(&self, usage: &ResourceUsage) -> Vec<LimitViolation> {
        let mut violations = Vec::new();

        if self.max_rss_bytes > 0
            && let Some(rss) = usage.rss_bytes
            && rss > self.max_rss_bytes
        {
            violations.push(LimitViolation {
                resource: "rss_bytes",
                limit: self.max_rss_bytes,
                actual: rss,
            });
        }

        if self.max_threads > 0
            && let Some(threads) = usage.thread_count
            && threads > self.max_threads
        {
            violations.push(LimitViolation {
                resource: "threads",
                limit: u64::from(self.max_threads),
                actual: u64::from(threads),
            });
        }

        if self.max_open_fds > 0
            && let Some(fds) = usage.open_fds
            && fds > self.max_open_fds
        {
            violations.push(LimitViolation {
                resource: "open_fds",
                limit: u64::from(self.max_open_fds),
                actual: u64::from(fds),
            });
        }

        violations
    }
}

/// A resource limit violation.
#[derive(Debug, Clone)]
pub struct LimitViolation {
    /// Which resource is over limit.
    pub resource: &'static str,
    /// The configured limit.
    pub limit: u64,
    /// The actual value.
    pub actual: u64,
}

impl std::fmt::Display for LimitViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} exceeds limit {}",
            self.resource, self.actual, self.limit
        )
    }
}

// ─── Lifecycle Tracker ──────────────────────────────────────────────────────

/// Tracks the lifecycle state of the daemon and all subsystems.
pub struct LifecycleTracker {
    phase: std::sync::Mutex<DaemonPhase>,
    started_at: Instant,
    subsystems: std::sync::Mutex<HashMap<SubsystemId, SubsystemHealth>>,
    total_errors: AtomicU64,
    total_panics: AtomicU64,
    watchdog_config: WatchdogConfig,
    resource_limits: ResourceLimits,
}

impl LifecycleTracker {
    /// Create a new tracker in the Initializing phase.
    #[must_use]
    pub fn new(watchdog_config: WatchdogConfig, resource_limits: ResourceLimits) -> Self {
        Self {
            phase: std::sync::Mutex::new(DaemonPhase::Initializing),
            started_at: Instant::now(),
            subsystems: std::sync::Mutex::new(HashMap::new()),
            total_errors: AtomicU64::new(0),
            total_panics: AtomicU64::new(0),
            watchdog_config,
            resource_limits,
        }
    }

    /// Transition to a new lifecycle phase.
    ///
    /// # Panics
    ///
    /// Panics if the phase mutex is poisoned.
    pub fn transition_to(&self, phase: DaemonPhase) {
        let mut current = self.phase.lock().expect("phase lock");
        info!(
            target: "frankensearch.fsfs.lifecycle",
            from = %*current,
            to = %phase,
            "Daemon phase transition"
        );
        *current = phase;
    }

    /// Current lifecycle phase.
    ///
    /// # Panics
    ///
    /// Panics if the phase mutex is poisoned.
    #[must_use]
    pub fn current_phase(&self) -> DaemonPhase {
        *self.phase.lock().expect("phase lock")
    }

    /// Register a subsystem with initial Pending status.
    ///
    /// # Panics
    ///
    /// Panics if the subsystems mutex is poisoned.
    pub fn register_subsystem(&self, id: SubsystemId) {
        let now = iso_now();
        let mut subs = self.subsystems.lock().expect("subsystems lock");
        subs.insert(
            id,
            SubsystemHealth {
                id,
                status: HealthStatus::Pending,
                restart_count: 0,
                last_error: None,
                last_healthy_at: None,
                status_since: now,
            },
        );
    }

    /// Update a subsystem's health status.
    ///
    /// # Panics
    ///
    /// Panics if the subsystems mutex is poisoned.
    pub fn update_subsystem(&self, id: SubsystemId, status: HealthStatus, error: Option<String>) {
        let now = iso_now();
        let mut subs = self.subsystems.lock().expect("subsystems lock");
        if let Some(health) = subs.get_mut(&id) {
            if status == HealthStatus::Healthy {
                health.last_healthy_at = Some(now.clone());
            }
            if status == HealthStatus::Failed {
                self.total_errors.fetch_add(1, Ordering::Relaxed);
            }
            health.status = status;
            health.last_error = error;
            health.status_since = now;
        }
    }

    /// Record a recovered panic for a subsystem.
    ///
    /// # Panics
    ///
    /// Panics if the subsystems mutex is poisoned.
    pub fn record_panic_recovery(&self, id: SubsystemId, error_msg: &str) {
        self.total_panics.fetch_add(1, Ordering::Relaxed);
        let now = iso_now();
        let mut subs = self.subsystems.lock().expect("subsystems lock");
        if let Some(health) = subs.get_mut(&id) {
            health.restart_count += 1;
            health.last_error = Some(error_msg.to_owned());
            health.status = HealthStatus::Degraded;
            health.status_since = now;
        }
        drop(subs);

        warn!(
            target: "frankensearch.fsfs.lifecycle",
            subsystem = id.name(),
            error = error_msg,
            "Panic recovered in subsystem"
        );
    }

    /// Whether the watchdog should restart a given subsystem.
    ///
    /// # Panics
    ///
    /// Panics if the subsystems mutex is poisoned.
    #[must_use]
    pub fn should_restart(&self, id: SubsystemId) -> Option<Duration> {
        let subs = self.subsystems.lock().expect("subsystems lock");
        let health = subs.get(&id)?;

        if health.status != HealthStatus::Failed {
            return None;
        }

        let restart_count = health.restart_count;
        drop(subs);

        if self.watchdog_config.is_exhausted(restart_count) {
            warn!(
                target: "frankensearch.fsfs.lifecycle",
                subsystem = id.name(),
                restarts = restart_count,
                "Subsystem restart limit exhausted"
            );
            return None;
        }

        Some(self.watchdog_config.backoff_delay(restart_count))
    }

    /// Check resource limits against current usage.
    #[must_use]
    pub fn check_resource_limits(&self, usage: &ResourceUsage) -> Vec<LimitViolation> {
        self.resource_limits.check(usage)
    }

    /// Build a complete status snapshot.
    ///
    /// # Panics
    ///
    /// Panics if the subsystems or phase mutex is poisoned.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn status(&self) -> DaemonStatus {
        let subs = self.subsystems.lock().expect("subsystems lock");
        let subsystem_vec: Vec<SubsystemHealth> = subs.values().cloned().collect();
        drop(subs);

        let started_at_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
            - (self.started_at.elapsed().as_millis() as u64);

        // Compute aggregate phase.
        let phase = *self.phase.lock().expect("phase lock");

        DaemonStatus {
            phase,
            pid: std::process::id(),
            started_at: format_epoch_ms(started_at_ms),
            uptime_secs: self.started_at.elapsed().as_secs(),
            subsystems: subsystem_vec,
            total_errors: self.total_errors.load(Ordering::Relaxed),
            total_panics_recovered: self.total_panics.load(Ordering::Relaxed),
            resources: ResourceUsage::default(),
        }
    }

    /// Watchdog config reference.
    #[must_use]
    pub const fn watchdog_config(&self) -> &WatchdogConfig {
        &self.watchdog_config
    }

    /// Resource limits reference.
    #[must_use]
    pub const fn resource_limits(&self) -> &ResourceLimits {
        &self.resource_limits
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| "unknown".into())
}

fn iso_now() -> String {
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format_epoch_secs(secs)
}

fn format_epoch_ms(ms: u64) -> String {
    format_epoch_secs(ms / 1000)
}

fn format_epoch_secs(secs: u64) -> String {
    // Simple ISO 8601 formatting without chrono dependency.
    // Good enough for status reporting.
    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Approximate date calculation (not accounting for leap seconds).
    let (year, month, day) = epoch_days_to_ymd(days_since_epoch);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

const fn epoch_days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Civil calendar calculation from days since 1970-01-01.
    // Based on Howard Hinnant's algorithm.
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    // ── DaemonPhase ──

    #[test]
    fn daemon_phase_accepting_work() {
        assert!(!DaemonPhase::Initializing.is_accepting_work());
        assert!(!DaemonPhase::StartingUp.is_accepting_work());
        assert!(DaemonPhase::Running.is_accepting_work());
        assert!(DaemonPhase::Degraded.is_accepting_work());
        assert!(!DaemonPhase::ShuttingDown.is_accepting_work());
        assert!(!DaemonPhase::Stopped.is_accepting_work());
    }

    #[test]
    fn daemon_phase_terminal() {
        assert!(!DaemonPhase::Running.is_terminal());
        assert!(DaemonPhase::Stopped.is_terminal());
    }

    #[test]
    fn daemon_phase_display() {
        assert_eq!(format!("{}", DaemonPhase::Running), "running");
        assert_eq!(format!("{}", DaemonPhase::Degraded), "degraded");
    }

    #[test]
    fn daemon_phase_serde_roundtrip() {
        let phase = DaemonPhase::ShuttingDown;
        let json = serde_json::to_string(&phase).unwrap();
        assert_eq!(json, "\"shutting_down\"");
        let parsed: DaemonPhase = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, phase);
    }

    // ── SubsystemId ──

    #[test]
    fn subsystem_id_names() {
        assert_eq!(SubsystemId::Crawler.name(), "crawler");
        assert_eq!(SubsystemId::QueryServer.name(), "query_server");
    }

    // ── HealthStatus ──

    #[test]
    fn health_status_alive() {
        assert!(HealthStatus::Pending.is_alive());
        assert!(HealthStatus::Healthy.is_alive());
        assert!(HealthStatus::Degraded.is_alive());
        assert!(!HealthStatus::Failed.is_alive());
        assert!(!HealthStatus::Stopped.is_alive());
    }

    // ── PID File ──

    #[test]
    fn pid_file_acquire_and_release() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("test.pid");
        let pid_file = PidFile::new(&path);

        pid_file.acquire("0.1.0").expect("acquire");
        assert!(path.exists());

        // Read back.
        let contents = pid_file.read().expect("read");
        assert_eq!(contents.pid, std::process::id());
        assert_eq!(contents.version, "0.1.0");

        // Second acquire should fail (same PID is alive).
        let pid_file2 = PidFile::new(&path);
        assert!(pid_file2.acquire("0.1.0").is_err());

        // Release.
        pid_file.release();
        assert!(!path.exists());
    }

    #[test]
    fn pid_file_stale_recovery() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("test.pid");

        // Write a PID file with a dead PID.
        let mut contents = PidFileContents::current("0.1.0");
        contents.pid = 999_999_999;
        let json = serde_json::to_string_pretty(&contents).unwrap();
        std::fs::write(&path, json.as_bytes()).unwrap();

        // Should recover and acquire.
        let pid_file = PidFile::new(&path);
        pid_file.acquire("0.2.0").expect("recover stale");

        let new_contents = pid_file.read().expect("read");
        assert_eq!(new_contents.pid, std::process::id());
        assert_eq!(new_contents.version, "0.2.0");
    }

    #[test]
    fn pid_file_drop_releases() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("test.pid");

        {
            let pid_file = PidFile::new(&path);
            pid_file.acquire("0.1.0").expect("acquire");
            assert!(path.exists());
        } // Drop releases.

        assert!(!path.exists());
    }

    #[test]
    fn pid_file_default_path() {
        let path = PidFile::default_path();
        assert!(path.to_string_lossy().contains("fsfs.pid"));
    }

    // ── WatchdogConfig ──

    #[test]
    fn watchdog_backoff_exponential() {
        let config = WatchdogConfig::default();
        let d0 = config.backoff_delay(0);
        let d1 = config.backoff_delay(1);
        let d2 = config.backoff_delay(2);

        assert_eq!(d0, Duration::from_secs(5));
        assert_eq!(d1, Duration::from_secs(10));
        assert_eq!(d2, Duration::from_secs(20));
    }

    #[test]
    fn watchdog_backoff_caps_at_max() {
        let config = WatchdogConfig {
            initial_backoff_ms: 30_000,
            max_backoff_ms: 60_000,
            backoff_multiplier: 2.0,
            ..WatchdogConfig::default()
        };
        let d2 = config.backoff_delay(2);
        assert_eq!(d2, Duration::from_mins(1));
    }

    #[test]
    fn watchdog_exhaustion() {
        let config = WatchdogConfig {
            max_restarts: 3,
            ..WatchdogConfig::default()
        };
        assert!(!config.is_exhausted(2));
        assert!(config.is_exhausted(3));
    }

    // ── Resource Limits ──

    #[test]
    fn resource_limits_no_violations_when_unlimited() {
        let limits = ResourceLimits::default();
        let usage = ResourceUsage {
            rss_bytes: Some(1_000_000_000),
            thread_count: Some(100),
            open_fds: Some(5000),
        };
        assert!(limits.check(&usage).is_empty());
    }

    #[test]
    fn resource_limits_detects_violations() {
        let limits = ResourceLimits {
            max_rss_bytes: 100_000,
            max_threads: 10,
            max_open_fds: 100,
            max_index_bytes: 0,
        };
        let usage = ResourceUsage {
            rss_bytes: Some(200_000),
            thread_count: Some(20),
            open_fds: Some(50), // Under limit.
        };
        let violations = limits.check(&usage);
        assert_eq!(violations.len(), 2);
        assert_eq!(violations[0].resource, "rss_bytes");
        assert_eq!(violations[1].resource, "threads");
    }

    #[test]
    fn resource_limits_none_usage_no_violation() {
        let limits = ResourceLimits {
            max_rss_bytes: 100_000,
            ..ResourceLimits::default()
        };
        let usage = ResourceUsage::default();
        assert!(limits.check(&usage).is_empty());
    }

    // ── LifecycleTracker ──

    #[test]
    fn lifecycle_tracker_phase_transitions() {
        let tracker = LifecycleTracker::new(WatchdogConfig::default(), ResourceLimits::default());

        assert_eq!(tracker.current_phase(), DaemonPhase::Initializing);

        tracker.transition_to(DaemonPhase::StartingUp);
        assert_eq!(tracker.current_phase(), DaemonPhase::StartingUp);

        tracker.transition_to(DaemonPhase::Running);
        assert_eq!(tracker.current_phase(), DaemonPhase::Running);
    }

    #[test]
    fn lifecycle_tracker_subsystem_health() {
        let tracker = LifecycleTracker::new(WatchdogConfig::default(), ResourceLimits::default());

        tracker.register_subsystem(SubsystemId::Crawler);
        tracker.register_subsystem(SubsystemId::QueryServer);

        tracker.update_subsystem(SubsystemId::Crawler, HealthStatus::Healthy, None);
        tracker.update_subsystem(
            SubsystemId::QueryServer,
            HealthStatus::Failed,
            Some("connection refused".into()),
        );

        let status = tracker.status();
        assert_eq!(status.subsystems.len(), 2);
        assert_eq!(status.total_errors, 1);
    }

    #[test]
    fn lifecycle_tracker_panic_recovery() {
        let tracker = LifecycleTracker::new(WatchdogConfig::default(), ResourceLimits::default());

        tracker.register_subsystem(SubsystemId::EmbedFast);
        tracker.record_panic_recovery(SubsystemId::EmbedFast, "stack overflow");

        let status = tracker.status();
        assert_eq!(status.total_panics_recovered, 1);

        let sub = status
            .subsystems
            .iter()
            .find(|s| s.id == SubsystemId::EmbedFast)
            .unwrap();
        assert_eq!(sub.restart_count, 1);
        assert_eq!(sub.status, HealthStatus::Degraded);
        assert_eq!(sub.last_error.as_deref(), Some("stack overflow"));
    }

    #[test]
    fn lifecycle_tracker_should_restart() {
        let config = WatchdogConfig {
            max_restarts: 2,
            initial_backoff_ms: 1000,
            ..WatchdogConfig::default()
        };
        let tracker = LifecycleTracker::new(config, ResourceLimits::default());

        tracker.register_subsystem(SubsystemId::Crawler);
        tracker.update_subsystem(SubsystemId::Crawler, HealthStatus::Failed, None);

        // First failure: should restart with backoff.
        let delay = tracker.should_restart(SubsystemId::Crawler);
        assert!(delay.is_some());
        assert_eq!(delay.unwrap(), Duration::from_secs(1));

        // Simulate restart + second failure.
        tracker.record_panic_recovery(SubsystemId::Crawler, "crash");
        tracker.update_subsystem(SubsystemId::Crawler, HealthStatus::Failed, None);
        let delay = tracker.should_restart(SubsystemId::Crawler);
        assert!(delay.is_some());

        // Simulate another restart + third failure (exhausted).
        tracker.record_panic_recovery(SubsystemId::Crawler, "crash again");
        tracker.update_subsystem(SubsystemId::Crawler, HealthStatus::Failed, None);
        let delay = tracker.should_restart(SubsystemId::Crawler);
        assert!(delay.is_none(), "should be exhausted");
    }

    #[test]
    fn lifecycle_tracker_healthy_subsystem_not_restarted() {
        let tracker = LifecycleTracker::new(WatchdogConfig::default(), ResourceLimits::default());
        tracker.register_subsystem(SubsystemId::Crawler);
        tracker.update_subsystem(SubsystemId::Crawler, HealthStatus::Healthy, None);
        assert!(tracker.should_restart(SubsystemId::Crawler).is_none());
    }

    #[test]
    fn lifecycle_tracker_status_snapshot() {
        let tracker = LifecycleTracker::new(WatchdogConfig::default(), ResourceLimits::default());
        tracker.transition_to(DaemonPhase::Running);
        tracker.register_subsystem(SubsystemId::Crawler);

        let status = tracker.status();
        assert_eq!(status.phase, DaemonPhase::Running);
        assert_eq!(status.pid, std::process::id());
        assert!(status.uptime_secs < 5);
        assert_eq!(status.subsystems.len(), 1);
    }

    #[test]
    fn daemon_status_serializes_to_json() {
        let status = DaemonStatus {
            phase: DaemonPhase::Running,
            pid: 12345,
            started_at: "2026-02-14T06:00:00Z".into(),
            uptime_secs: 3600,
            subsystems: vec![SubsystemHealth {
                id: SubsystemId::Crawler,
                status: HealthStatus::Healthy,
                restart_count: 0,
                last_error: None,
                last_healthy_at: Some("2026-02-14T06:59:00Z".into()),
                status_since: "2026-02-14T06:00:00Z".into(),
            }],
            total_errors: 0,
            total_panics_recovered: 0,
            resources: ResourceUsage::default(),
        };

        let json = serde_json::to_string_pretty(&status).unwrap();
        assert!(json.contains("\"running\""));
        assert!(json.contains("12345"));
        assert!(json.contains("\"crawler\""));
    }

    // ── Date Formatting ──

    #[test]
    fn epoch_days_to_ymd_epoch() {
        let (y, m, d) = epoch_days_to_ymd(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn epoch_days_to_ymd_2026() {
        // 2026-02-14 is day 20498 since epoch.
        let (y, m, d) = epoch_days_to_ymd(20498);
        assert_eq!((y, m, d), (2026, 2, 14));
    }

    #[test]
    fn format_epoch_secs_basic() {
        let s = format_epoch_secs(0);
        assert_eq!(s, "1970-01-01T00:00:00Z");
    }

    // ── LimitViolation Display ──

    #[test]
    fn limit_violation_display() {
        let v = LimitViolation {
            resource: "rss_bytes",
            limit: 100,
            actual: 200,
        };
        assert_eq!(format!("{v}"), "rss_bytes: 200 exceeds limit 100");
    }
}
