use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

#[derive(Debug, Default)]
pub struct StorageMetrics {
    opens: AtomicU64,
    schema_bootstraps: AtomicU64,
    tx_commits: AtomicU64,
    tx_rollbacks: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageMetricsSnapshot {
    pub opens: u64,
    pub schema_bootstraps: u64,
    pub tx_commits: u64,
    pub tx_rollbacks: u64,
}

impl StorageMetrics {
    pub fn record_open(&self) {
        self.opens.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_schema_bootstrap(&self) {
        self.schema_bootstraps.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_commit(&self) {
        self.tx_commits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_rollback(&self) {
        self.tx_rollbacks.fetch_add(1, Ordering::Relaxed);
    }

    #[must_use]
    pub fn snapshot(&self) -> StorageMetricsSnapshot {
        StorageMetricsSnapshot {
            opens: self.opens.load(Ordering::Relaxed),
            schema_bootstraps: self.schema_bootstraps.load(Ordering::Relaxed),
            tx_commits: self.tx_commits.load(Ordering::Relaxed),
            tx_rollbacks: self.tx_rollbacks.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StorageMetrics;

    #[test]
    fn default_metrics_are_zeroed() {
        let metrics = StorageMetrics::default();
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.opens, 0);
        assert_eq!(snapshot.schema_bootstraps, 0);
        assert_eq!(snapshot.tx_commits, 0);
        assert_eq!(snapshot.tx_rollbacks, 0);
    }

    #[test]
    fn counters_accumulate_independently() {
        let metrics = StorageMetrics::default();
        metrics.record_open();
        metrics.record_open();
        metrics.record_schema_bootstrap();
        metrics.record_commit();
        metrics.record_commit();
        metrics.record_commit();
        metrics.record_rollback();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.opens, 2);
        assert_eq!(snapshot.schema_bootstraps, 1);
        assert_eq!(snapshot.tx_commits, 3);
        assert_eq!(snapshot.tx_rollbacks, 1);
    }

    #[test]
    fn snapshot_is_point_in_time() {
        let metrics = StorageMetrics::default();
        metrics.record_open();
        let snap1 = metrics.snapshot();

        metrics.record_open();
        let snap2 = metrics.snapshot();

        assert_eq!(snap1.opens, 1);
        assert_eq!(snap2.opens, 2);
    }

    #[test]
    fn snapshot_serde_roundtrip() {
        let metrics = StorageMetrics::default();
        metrics.record_open();
        metrics.record_commit();
        let snapshot = metrics.snapshot();

        let json = serde_json::to_string(&snapshot).expect("serialize");
        let deserialized: super::StorageMetricsSnapshot =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(snapshot, deserialized);
    }
}
