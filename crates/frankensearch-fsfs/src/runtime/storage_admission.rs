//! Read-only disk admission for complete-generation and ordinary writers.
//!
//! This is a cooperative boundary, not a reservation or a filesystem quota.
//! An admitted write can grow until its next boundary. No generation, source,
//! or cache is evicted to satisfy the policy.

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};

use crate::generation_store::storage_footprint_bytes;
use crate::lifecycle::{DiskBudgetPolicy, DiskBudgetSnapshot, DiskBudgetStage};
use crate::pressure::PressureState;

use super::{
    DISK_BUDGET_FALLBACK_BYTES, FsfsRuntime, available_space_for_path,
    conservative_budget_from_available_bytes, retained_search_checkpoint,
};

const DISK_ADMISSION_SUBSYSTEM: &str = "fsfs.index.disk_pressure";

/// The automatic budget is fixed for one monitor, as for the legacy lifecycle
/// tracker. Explicit zero keeps that tracker's unbounded-budget convention.
pub(super) struct StorageAdmissionMonitor {
    root: PathBuf,
    budget_bytes: u64,
    emergency: Option<DiskBudgetSnapshot>,
    sample_interval: Duration,
    sampled: Option<(Instant, Option<DiskBudgetSnapshot>)>,
    last_stage: Option<DiskBudgetStage>,
}

impl StorageAdmissionMonitor {
    pub(super) fn new(runtime: &FsfsRuntime, root: &Path) -> SearchResult<Self> {
        let root = if root.is_absolute() {
            root.to_path_buf()
        } else {
            std::env::current_dir()?.join(root)
        };
        let budget_bytes = runtime.config.storage.disk_budget_bytes.unwrap_or_else(|| {
            available_space_for_path(&root).map_or(
                DISK_BUDGET_FALLBACK_BYTES,
                conservative_budget_from_available_bytes,
            )
        });
        Ok(Self {
            root,
            budget_bytes,
            emergency: runtime.apply_storage_emergency_override(None, None, budget_bytes),
            sample_interval: Duration::from_millis(runtime.config.pressure.sample_interval_ms.max(1)),
            sampled: None,
            last_stage: None,
        })
    }

    pub(super) fn paused(&mut self, cx: &Cx, now: Instant) -> SearchResult<bool> {
        Ok(self.snapshot(cx, now)?.is_some_and(pauses_writes))
    }

    /// A publication changes storage occupancy even when visibility is durable
    /// but the caller cannot prove the final directory sync completed.
    pub(super) fn invalidate(&mut self) {
        self.sampled = None;
    }

    /// A fresh writer guard can observe growth after the watch's cached sample.
    pub(super) fn record_denial(&mut self, snapshot: DiskBudgetSnapshot, now: Instant) {
        self.observe(snapshot);
        self.sampled = Some((now, Some(snapshot)));
    }

    fn snapshot(&mut self, cx: &Cx, now: Instant) -> SearchResult<Option<DiskBudgetSnapshot>> {
        retained_search_checkpoint(cx)?;
        // Emergency is configuration, not a sampled sensor: it must also win
        // with an unbounded budget or a nonexistent/inaccessible store.
        if let Some(snapshot) = self.emergency {
            self.observe(snapshot);
            return Ok(Some(snapshot));
        }
        if self.budget_bytes == 0 {
            return Ok(None);
        }
        if let Some((sampled_at, snapshot)) = self.sampled
            && now.saturating_duration_since(sampled_at) < self.sample_interval
        {
            return Ok(snapshot);
        }
        let used_bytes = storage_footprint_bytes(cx, &self.root)?;
        retained_search_checkpoint(cx)?;
        let snapshot = DiskBudgetPolicy::default().evaluate(used_bytes, self.budget_bytes);
        self.observe(snapshot);
        self.sampled = Some((now, Some(snapshot)));
        Ok(Some(snapshot))
    }

    fn observe(&mut self, snapshot: DiskBudgetSnapshot) {
        if self.last_stage != Some(snapshot.stage) {
            tracing::info!(
                reason_code = snapshot.reason_code,
                stage = ?snapshot.stage,
                action = ?snapshot.action,
                used_bytes = snapshot.used_bytes,
                budget_bytes = snapshot.budget_bytes,
                "fsfs storage admission changed"
            );
            self.last_stage = Some(snapshot.stage);
        }
    }
}

fn pauses_writes(snapshot: DiskBudgetSnapshot) -> bool {
    matches!(
        FsfsRuntime::disk_budget_control_plan(snapshot).watcher_pressure_state,
        PressureState::Degraded | PressureState::Emergency
    )
}

#[derive(Debug)]
struct StorageAdmissionDenied {
    snapshot: DiskBudgetSnapshot,
    phase: &'static str,
}

impl fmt::Display for StorageAdmissionDenied {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "write admission paused at {}: {} ({} bytes used, {} byte budget); ",
            self.phase,
            self.snapshot.reason_code,
            self.snapshot.used_bytes,
            self.snapshot.budget_bytes,
        )?;
        if self.snapshot.reason_code == super::DISK_BUDGET_REASON_EMERGENCY_OVERRIDE {
            formatter.write_str("clear storage.storage_pressure_emergency before retrying")
        } else {
            formatter.write_str("free storage or raise storage.disk_budget_bytes before retrying")
        }
    }
}

impl std::error::Error for StorageAdmissionDenied {}

/// Only this structured denial is retryable as a disk pause. Traversal,
/// integrity, and cancellation errors retain their original semantics.
pub(super) fn storage_admission_denial(error: &SearchError) -> Option<DiskBudgetSnapshot> {
    match error {
        SearchError::SubsystemError { subsystem, source }
            if *subsystem == DISK_ADMISSION_SUBSYSTEM =>
        {
            source
                .downcast_ref::<StorageAdmissionDenied>()
                .map(|denial| denial.snapshot)
        }
        _ => None,
    }
}

impl FsfsRuntime {
    /// Fresh admission immediately before a writer allocates or copies a
    /// successor. Callers retain their existing publication ownership/fences.
    pub(super) fn admit_storage_write(
        &self,
        cx: &Cx,
        root: &Path,
        phase: &'static str,
    ) -> SearchResult<()> {
        retained_search_checkpoint(cx)?;
        let snapshot = StorageAdmissionMonitor::new(self, root)?.snapshot(cx, Instant::now())?;
        if let Some(snapshot) = snapshot.filter(|snapshot| pauses_writes(*snapshot)) {
            return Err(SearchError::SubsystemError {
                subsystem: DISK_ADMISSION_SUBSYSTEM,
                source: Box::new(StorageAdmissionDenied { snapshot, phase }),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::config::FsfsConfig;

    use super::*;

    fn runtime(budget: u64, emergency: bool) -> FsfsRuntime {
        let mut config = FsfsConfig::default();
        config.storage.disk_budget_bytes = Some(budget);
        config.storage.storage_pressure_emergency = emergency;
        config.pressure.sample_interval_ms = 1_000;
        FsfsRuntime::new(config)
    }

    #[test]
    fn explicit_zero_is_unbounded_but_emergency_and_cancellation_still_win() {
        let cx = Cx::for_testing();
        let directory = tempfile::tempdir().unwrap();
        let absent = directory.path().join("not-created");
        runtime(0, false)
            .admit_storage_write(&cx, &absent, "test")
            .unwrap();
        let error = runtime(0, true)
            .admit_storage_write(&cx, &absent, "test")
            .unwrap_err();
        let denial = storage_admission_denial(&error).unwrap();
        assert_eq!(
            denial.reason_code,
            super::super::DISK_BUDGET_REASON_EMERGENCY_OVERRIDE
        );
        assert!(!absent.exists());
        cx.cancel_with(asupersync::CancelKind::User, Some("cancel disk admission"));
        let error = runtime(0, true)
            .admit_storage_write(&cx, &absent, "test")
            .unwrap_err();
        assert!(matches!(error, SearchError::Cancelled { .. }));
        assert!(storage_admission_denial(&error).is_none());
    }

    #[test]
    fn whole_store_accounting_is_read_only_and_retains_the_existing_stage_policy() {
        let cx = Cx::for_testing();
        let directory = tempfile::tempdir().unwrap();
        let retained = directory.path().join("retained");
        let inactive = directory.path().join("inactive");
        fs::create_dir(&retained).unwrap();
        fs::create_dir(&inactive).unwrap();
        fs::write(retained.join("old-vector"), [1_u8; 450]).unwrap();
        fs::write(inactive.join("unfinished-vector"), [2_u8; 450]).unwrap();
        let runtime = runtime(1_000, false);
        runtime
            .admit_storage_write(&cx, directory.path(), "test")
            .unwrap();
        fs::write(directory.path().join("catalog"), [3_u8; 101]).unwrap();
        let error = runtime
            .admit_storage_write(&cx, directory.path(), "test")
            .unwrap_err();
        let denial = storage_admission_denial(&error).unwrap();
        assert_eq!(denial.stage, DiskBudgetStage::OverLimit);
        assert_eq!(denial.used_bytes, 1_001);
        assert_eq!(fs::read(retained.join("old-vector")).unwrap(), [1_u8; 450]);
        assert_eq!(
            fs::read(inactive.join("unfinished-vector")).unwrap(),
            [2_u8; 450]
        );
        let io_error = SearchError::SubsystemError {
            subsystem: DISK_ADMISSION_SUBSYSTEM,
            source: Box::new(std::io::Error::other("foreign error")),
        };
        assert!(storage_admission_denial(&io_error).is_none());
    }

    #[test]
    fn sampling_cadence_and_publication_invalidation_observe_storage_changes() {
        let cx = Cx::for_testing();
        let directory = tempfile::tempdir().unwrap();
        let ballast = directory.path().join("ballast");
        fs::write(&ballast, [0_u8; 101]).unwrap();
        let mut monitor =
            StorageAdmissionMonitor::new(&runtime(100, false), directory.path()).unwrap();
        let now = Instant::now();
        assert!(monitor.paused(&cx, now).unwrap());
        fs::write(&ballast, []).unwrap();
        assert!(
            monitor
                .paused(&cx, now + Duration::from_millis(999))
                .unwrap()
        );
        assert!(!monitor.paused(&cx, now + Duration::from_secs(1)).unwrap());
        fs::write(&ballast, [0_u8; 110]).unwrap();
        assert!(
            !monitor
                .paused(&cx, now + Duration::from_millis(1_001))
                .unwrap()
        );
        monitor.invalidate();
        assert!(
            monitor
                .paused(&cx, now + Duration::from_millis(1_001))
                .unwrap()
        );
    }

    #[cfg(unix)]
    #[test]
    fn accounting_deduplicates_hardlinks_and_does_not_follow_symlinks() {
        let cx = Cx::for_testing();
        let directory = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("vector"), [0_u8; 100]).unwrap();
        fs::hard_link(
            directory.path().join("vector"),
            directory.path().join("retained-vector"),
        )
        .unwrap();
        fs::write(outside.path().join("large"), [0_u8; 2_000]).unwrap();
        std::os::unix::fs::symlink(outside.path(), directory.path().join("outside")).unwrap();
        assert_eq!(storage_footprint_bytes(&cx, directory.path()).unwrap(), 100);
        runtime(100, false)
            .admit_storage_write(&cx, directory.path(), "test")
            .unwrap();
        let error = runtime(100, false)
            .admit_storage_write(&cx, &directory.path().join("vector"), "test")
            .unwrap_err();
        assert!(storage_admission_denial(&error).is_none());
    }
}
