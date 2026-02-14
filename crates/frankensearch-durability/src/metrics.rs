use std::sync::atomic::{AtomicU64, Ordering};

/// Decode failure classification used for durability telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeOutcomeClass {
    Recoverable,
    Unrecoverable,
}

/// Atomic counters for durability operations.
#[derive(Debug)]
pub struct DurabilityMetrics {
    /// Total bytes processed by encode operations.
    pub encoded_bytes_total: AtomicU64,
    /// Total source symbols produced.
    pub source_symbols_total: AtomicU64,
    /// Total repair symbols generated.
    pub repair_symbols_total: AtomicU64,
    /// Total bytes successfully decoded.
    pub decoded_bytes_total: AtomicU64,
    /// Total symbols consumed for successful decodes.
    pub decode_symbols_used_total: AtomicU64,
    /// Total symbols presented to decode calls.
    pub decode_symbols_received_total: AtomicU64,
    /// Total `k_required` values seen in decode calls.
    pub decode_k_required_total: AtomicU64,
    /// Number of encode operations.
    pub encode_ops: AtomicU64,
    /// Number of decode operations.
    pub decode_ops: AtomicU64,
    /// Number of decode failures.
    pub decode_failures: AtomicU64,
    /// Number of decode failures that can be retried with more/better symbols.
    pub decode_failures_recoverable: AtomicU64,
    /// Number of decode failures that are malformed/unrecoverable.
    pub decode_failures_unrecoverable: AtomicU64,
    /// Total encode latency in microseconds.
    pub encode_latency_us_total: AtomicU64,
    /// Total decode latency in microseconds.
    pub decode_latency_us_total: AtomicU64,
    /// Number of repair attempts.
    pub repair_attempts: AtomicU64,
    /// Number of successful repairs.
    pub repair_successes: AtomicU64,
    /// Number of failed repairs.
    pub repair_failures: AtomicU64,
}

impl DurabilityMetrics {
    /// Create a zeroed metrics container.
    pub const fn new() -> Self {
        Self {
            encoded_bytes_total: AtomicU64::new(0),
            source_symbols_total: AtomicU64::new(0),
            repair_symbols_total: AtomicU64::new(0),
            decoded_bytes_total: AtomicU64::new(0),
            decode_symbols_used_total: AtomicU64::new(0),
            decode_symbols_received_total: AtomicU64::new(0),
            decode_k_required_total: AtomicU64::new(0),
            encode_ops: AtomicU64::new(0),
            decode_ops: AtomicU64::new(0),
            decode_failures: AtomicU64::new(0),
            decode_failures_recoverable: AtomicU64::new(0),
            decode_failures_unrecoverable: AtomicU64::new(0),
            encode_latency_us_total: AtomicU64::new(0),
            decode_latency_us_total: AtomicU64::new(0),
            repair_attempts: AtomicU64::new(0),
            repair_successes: AtomicU64::new(0),
            repair_failures: AtomicU64::new(0),
        }
    }

    pub fn record_encode(
        &self,
        encoded_bytes: u64,
        source_symbols: u64,
        repair_symbols: u64,
        latency_us: u64,
    ) {
        self.encoded_bytes_total
            .fetch_add(encoded_bytes, Ordering::Relaxed);
        self.source_symbols_total
            .fetch_add(source_symbols, Ordering::Relaxed);
        self.repair_symbols_total
            .fetch_add(repair_symbols, Ordering::Relaxed);
        self.encode_latency_us_total
            .fetch_add(latency_us, Ordering::Relaxed);
        self.encode_ops.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_decode_success(
        &self,
        decoded_bytes: u64,
        symbols_used: u64,
        symbols_received: u64,
        k_required: u64,
        latency_us: u64,
    ) {
        self.decoded_bytes_total
            .fetch_add(decoded_bytes, Ordering::Relaxed);
        self.decode_symbols_used_total
            .fetch_add(symbols_used, Ordering::Relaxed);
        self.decode_symbols_received_total
            .fetch_add(symbols_received, Ordering::Relaxed);
        self.decode_k_required_total
            .fetch_add(k_required, Ordering::Relaxed);
        self.decode_latency_us_total
            .fetch_add(latency_us, Ordering::Relaxed);
        self.decode_ops.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_decode_failure(
        &self,
        class: DecodeOutcomeClass,
        symbols_received: u64,
        k_required: u64,
        latency_us: u64,
    ) {
        self.decode_symbols_received_total
            .fetch_add(symbols_received, Ordering::Relaxed);
        self.decode_k_required_total
            .fetch_add(k_required, Ordering::Relaxed);
        self.decode_latency_us_total
            .fetch_add(latency_us, Ordering::Relaxed);
        self.decode_ops.fetch_add(1, Ordering::Relaxed);
        self.decode_failures.fetch_add(1, Ordering::Relaxed);

        match class {
            DecodeOutcomeClass::Recoverable => {
                self.decode_failures_recoverable
                    .fetch_add(1, Ordering::Relaxed);
            }
            DecodeOutcomeClass::Unrecoverable => {
                self.decode_failures_unrecoverable
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn record_repair_attempt(&self) {
        self.repair_attempts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_repair_success(&self) {
        self.repair_successes.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_repair_failure(&self) {
        self.repair_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Capture a point-in-time snapshot of all counters.
    pub fn snapshot(&self) -> DurabilityMetricsSnapshot {
        DurabilityMetricsSnapshot {
            encoded_bytes_total: self.encoded_bytes_total.load(Ordering::Relaxed),
            source_symbols_total: self.source_symbols_total.load(Ordering::Relaxed),
            repair_symbols_total: self.repair_symbols_total.load(Ordering::Relaxed),
            decoded_bytes_total: self.decoded_bytes_total.load(Ordering::Relaxed),
            decode_symbols_used_total: self.decode_symbols_used_total.load(Ordering::Relaxed),
            decode_symbols_received_total: self
                .decode_symbols_received_total
                .load(Ordering::Relaxed),
            decode_k_required_total: self.decode_k_required_total.load(Ordering::Relaxed),
            encode_ops: self.encode_ops.load(Ordering::Relaxed),
            decode_ops: self.decode_ops.load(Ordering::Relaxed),
            decode_failures: self.decode_failures.load(Ordering::Relaxed),
            decode_failures_recoverable: self.decode_failures_recoverable.load(Ordering::Relaxed),
            decode_failures_unrecoverable: self
                .decode_failures_unrecoverable
                .load(Ordering::Relaxed),
            encode_latency_us_total: self.encode_latency_us_total.load(Ordering::Relaxed),
            decode_latency_us_total: self.decode_latency_us_total.load(Ordering::Relaxed),
            repair_attempts: self.repair_attempts.load(Ordering::Relaxed),
            repair_successes: self.repair_successes.load(Ordering::Relaxed),
            repair_failures: self.repair_failures.load(Ordering::Relaxed),
        }
    }
}

impl Default for DurabilityMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable snapshot of durability counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DurabilityMetricsSnapshot {
    pub encoded_bytes_total: u64,
    pub source_symbols_total: u64,
    pub repair_symbols_total: u64,
    pub decoded_bytes_total: u64,
    pub decode_symbols_used_total: u64,
    pub decode_symbols_received_total: u64,
    pub decode_k_required_total: u64,
    pub encode_ops: u64,
    pub decode_ops: u64,
    pub decode_failures: u64,
    pub decode_failures_recoverable: u64,
    pub decode_failures_unrecoverable: u64,
    pub encode_latency_us_total: u64,
    pub decode_latency_us_total: u64,
    pub repair_attempts: u64,
    pub repair_successes: u64,
    pub repair_failures: u64,
}

#[cfg(test)]
mod tests {
    use super::{DecodeOutcomeClass, DurabilityMetrics};

    #[test]
    fn new_metrics_are_zeroed() {
        let metrics = DurabilityMetrics::new();
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.encoded_bytes_total, 0);
        assert_eq!(snapshot.source_symbols_total, 0);
        assert_eq!(snapshot.repair_symbols_total, 0);
        assert_eq!(snapshot.decoded_bytes_total, 0);
        assert_eq!(snapshot.decode_symbols_used_total, 0);
        assert_eq!(snapshot.decode_symbols_received_total, 0);
        assert_eq!(snapshot.decode_k_required_total, 0);
        assert_eq!(snapshot.encode_ops, 0);
        assert_eq!(snapshot.decode_ops, 0);
        assert_eq!(snapshot.decode_failures, 0);
        assert_eq!(snapshot.decode_failures_recoverable, 0);
        assert_eq!(snapshot.decode_failures_unrecoverable, 0);
        assert_eq!(snapshot.encode_latency_us_total, 0);
        assert_eq!(snapshot.decode_latency_us_total, 0);
        assert_eq!(snapshot.repair_attempts, 0);
        assert_eq!(snapshot.repair_successes, 0);
        assert_eq!(snapshot.repair_failures, 0);
    }

    #[test]
    fn default_equals_new() {
        let from_new = DurabilityMetrics::new().snapshot();
        let from_default = DurabilityMetrics::default().snapshot();
        assert_eq!(from_new, from_default);
    }

    #[test]
    fn repair_counters_are_independent() {
        let metrics = DurabilityMetrics::default();
        metrics.record_repair_attempt();
        metrics.record_repair_attempt();
        metrics.record_repair_success();
        metrics.record_repair_failure();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.repair_attempts, 2);
        assert_eq!(snapshot.repair_successes, 1);
        assert_eq!(snapshot.repair_failures, 1);
    }

    #[test]
    fn multiple_encodes_accumulate() {
        let metrics = DurabilityMetrics::default();
        metrics.record_encode(100, 4, 2, 50);
        metrics.record_encode(200, 8, 4, 100);
        metrics.record_encode(300, 12, 6, 150);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.encode_ops, 3);
        assert_eq!(snapshot.encoded_bytes_total, 600);
        assert_eq!(snapshot.source_symbols_total, 24);
        assert_eq!(snapshot.repair_symbols_total, 12);
        assert_eq!(snapshot.encode_latency_us_total, 300);
    }

    #[test]
    fn decode_success_and_failure_ops_both_increment_decode_ops() {
        let metrics = DurabilityMetrics::default();
        metrics.record_decode_success(100, 5, 6, 5, 10);
        metrics.record_decode_failure(DecodeOutcomeClass::Recoverable, 3, 5, 20);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.decode_ops, 2);
        assert_eq!(snapshot.decode_latency_us_total, 30);
    }

    #[test]
    fn snapshot_reflects_recorded_events() {
        let metrics = DurabilityMetrics::default();
        metrics.record_encode(100, 4, 5, 11);
        metrics.record_decode_success(90, 3, 7, 5, 13);
        metrics.record_decode_failure(DecodeOutcomeClass::Recoverable, 2, 5, 17);
        metrics.record_decode_failure(DecodeOutcomeClass::Unrecoverable, 2, 5, 19);
        metrics.record_repair_attempt();
        metrics.record_repair_success();
        metrics.record_repair_failure();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.encoded_bytes_total, 100);
        assert_eq!(snapshot.source_symbols_total, 4);
        assert_eq!(snapshot.repair_symbols_total, 5);
        assert_eq!(snapshot.decoded_bytes_total, 90);
        assert_eq!(snapshot.decode_symbols_used_total, 3);
        assert_eq!(snapshot.decode_symbols_received_total, 11);
        assert_eq!(snapshot.decode_k_required_total, 15);
        assert_eq!(snapshot.encode_ops, 1);
        assert_eq!(snapshot.decode_ops, 3);
        assert_eq!(snapshot.decode_failures, 2);
        assert_eq!(snapshot.decode_failures_recoverable, 1);
        assert_eq!(snapshot.decode_failures_unrecoverable, 1);
        assert_eq!(snapshot.encode_latency_us_total, 11);
        assert_eq!(snapshot.decode_latency_us_total, 49);
        assert_eq!(snapshot.repair_attempts, 1);
        assert_eq!(snapshot.repair_successes, 1);
        assert_eq!(snapshot.repair_failures, 1);
    }
}
