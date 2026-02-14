//! Robust statistics primitives for search monitoring.
//!
//! Provides streaming, outlier-resistant metrics that are stable under
//! heavy-tailed latency distributions typical of search workloads.
//!
//! # Components
//!
//! - [`TDigest`]: Streaming quantile estimation (any percentile, O(δ) memory)
//! - [`MedianMAD`]: Robust center (median) + spread (MAD) via paired t-digests
//! - [`HuberEstimator`]: Outlier-resistant streaming mean (bounded influence)
//! - [`HyperLogLog`]: Probabilistic cardinality estimation (~16KB, <1% error)
//! - [`RobustMetrics`]: Composite struct combining all primitives
//!
//! # Performance
//!
//! Each `RobustMetrics::insert` performs 2 t-digest inserts + 1 Huber update,
//! targeting <500ns amortized per observation.
//!
//! # References
//!
//! - Dunning & Ertl (2019), "Computing Extremely Accurate Quantiles Using t-Digests"
//! - Huber (1981), "Robust Statistics"
//! - Flajolet et al. (2007), "HyperLogLog"

use std::hash::{DefaultHasher, Hash, Hasher};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// TDigest
// ---------------------------------------------------------------------------

/// Internal centroid representation.
#[derive(Debug, Clone, Copy)]
struct Centroid {
    mean: f64,
    weight: f64,
}

/// Streaming quantile estimation using the t-digest algorithm.
///
/// Provides accurate quantile estimates (p50, p90, p95, p99, p999) with
/// O(δ) memory and O(log δ) amortized update cost. Accuracy is highest
/// at the tails where it matters most for latency monitoring.
///
/// With default compression (δ=100), uses ~4KB of memory.
#[derive(Debug, Clone)]
pub struct TDigest {
    centroids: Vec<Centroid>,
    buffer: Vec<f64>,
    compression: f64,
    total_weight: f64,
    min: f64,
    max: f64,
    buffer_capacity: usize,
}

impl TDigest {
    /// Default compression parameter.
    pub const DEFAULT_COMPRESSION: f64 = 100.0;

    /// Creates a new t-digest with the given compression parameter.
    ///
    /// Higher compression = more centroids = better accuracy, more memory.
    /// δ=100 is a good default (~4KB, <1% error at tails).
    #[must_use]
    pub fn new(compression: f64) -> Self {
        let compression = compression.max(10.0);
        let buffer_capacity = (compression * 5.0) as usize;
        Self {
            centroids: Vec::with_capacity(compression as usize * 2),
            buffer: Vec::with_capacity(buffer_capacity),
            compression,
            total_weight: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            buffer_capacity,
        }
    }

    /// Creates a t-digest with default compression (δ=100).
    #[must_use]
    pub fn with_default_compression() -> Self {
        Self::new(Self::DEFAULT_COMPRESSION)
    }

    /// Records a single observation. NaN values are silently ignored.
    pub fn insert(&mut self, value: f64) {
        if value.is_nan() {
            return;
        }
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.buffer.push(value);
        if self.buffer.len() >= self.buffer_capacity {
            self.compress();
        }
    }

    /// Returns the estimated quantile value for `q` in \[0, 1\].
    ///
    /// Returns `None` if no observations have been recorded.
    /// Flushes any buffered observations before computing.
    pub fn quantile(&mut self, q: f64) -> Option<f64> {
        if !self.buffer.is_empty() {
            self.compress();
        }
        if self.centroids.is_empty() {
            return None;
        }

        let q = q.clamp(0.0, 1.0);

        if self.centroids.len() == 1 {
            return Some(self.centroids[0].mean);
        }
        if q <= 0.0 {
            return Some(self.min);
        }
        if q >= 1.0 {
            return Some(self.max);
        }

        let target = q * self.total_weight;
        let mut cumulative = 0.0;

        for i in 0..self.centroids.len() {
            let c = &self.centroids[i];
            let mid = cumulative + c.weight / 2.0;

            if target < mid {
                if i == 0 {
                    // Interpolate between min and first centroid
                    let first_mid = c.weight / 2.0;
                    if first_mid <= 0.0 {
                        return Some(self.min);
                    }
                    let ratio = target / first_mid;
                    return Some(self.min + ratio * (c.mean - self.min));
                }
                let prev = &self.centroids[i - 1];
                let prev_mid = cumulative - prev.weight / 2.0;
                let span = mid - prev_mid;
                if span <= 0.0 {
                    return Some(c.mean);
                }
                let ratio = (target - prev_mid) / span;
                return Some(prev.mean + ratio * (c.mean - prev.mean));
            }

            cumulative += c.weight;
        }

        // Interpolate between last centroid and max
        let last = &self.centroids[self.centroids.len() - 1];
        let last_mid = self.total_weight - last.weight / 2.0;
        let remaining = self.total_weight - last_mid;
        if remaining <= 0.0 {
            return Some(self.max);
        }
        let ratio = (target - last_mid) / remaining;
        Some(last.mean + ratio * (self.max - last.mean))
    }

    /// Convenience: p50 (median).
    pub fn p50(&mut self) -> Option<f64> {
        self.quantile(0.50)
    }

    /// Convenience: p90.
    pub fn p90(&mut self) -> Option<f64> {
        self.quantile(0.90)
    }

    /// Convenience: p95.
    pub fn p95(&mut self) -> Option<f64> {
        self.quantile(0.95)
    }

    /// Convenience: p99.
    pub fn p99(&mut self) -> Option<f64> {
        self.quantile(0.99)
    }

    /// Convenience: p999.
    pub fn p999(&mut self) -> Option<f64> {
        self.quantile(0.999)
    }

    /// Total number of observations recorded (including buffered).
    #[must_use]
    pub fn count(&self) -> u64 {
        (self.total_weight + self.buffer.len() as f64) as u64
    }

    /// Observed minimum, or `None` if empty.
    #[must_use]
    pub fn min(&self) -> Option<f64> {
        if self.total_weight == 0.0 && self.buffer.is_empty() {
            None
        } else {
            Some(self.min)
        }
    }

    /// Observed maximum, or `None` if empty.
    #[must_use]
    pub fn max(&self) -> Option<f64> {
        if self.total_weight == 0.0 && self.buffer.is_empty() {
            None
        } else {
            Some(self.max)
        }
    }

    /// Number of centroids currently stored (after compression).
    #[must_use]
    pub fn centroid_count(&self) -> usize {
        self.centroids.len()
    }

    /// Merges another t-digest into this one.
    ///
    /// Used for aggregating per-task metrics into a global view.
    pub fn merge(&mut self, other: &TDigest) {
        for &v in &other.buffer {
            self.buffer.push(v);
        }
        for c in &other.centroids {
            self.centroids.push(*c);
        }
        self.total_weight += other.total_weight;
        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }
        self.compress();
    }

    /// Discards all observations.
    pub fn reset(&mut self) {
        self.centroids.clear();
        self.buffer.clear();
        self.total_weight = 0.0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
    }

    // ---- internals ----

    fn compress(&mut self) {
        for &v in &self.buffer {
            self.centroids.push(Centroid {
                mean: v,
                weight: 1.0,
            });
        }
        let buffer_weight = self.buffer.len() as f64;
        self.buffer.clear();

        if self.centroids.is_empty() {
            return;
        }

        self.centroids.sort_by(|a, b| a.mean.total_cmp(&b.mean));

        let total_weight = self.total_weight + buffer_weight;
        self.total_weight = total_weight;

        if total_weight <= 0.0 {
            return;
        }

        let mut result = Vec::with_capacity(self.compression as usize * 2);
        let mut current = self.centroids[0];
        let mut weight_so_far = 0.0;

        for c in self.centroids.iter().skip(1) {
            let q = (weight_so_far + current.weight / 2.0) / total_weight;
            let limit = self.max_centroid_weight(q, total_weight);

            if current.weight + c.weight <= limit {
                let new_weight = current.weight + c.weight;
                current.mean =
                    (current.mean * current.weight + c.mean * c.weight) / new_weight;
                current.weight = new_weight;
            } else {
                weight_so_far += current.weight;
                result.push(current);
                current = *c;
            }
        }
        result.push(current);

        self.centroids = result;
    }

    /// k₁ scale function: allows larger centroids in the middle,
    /// smaller at the tails for higher tail accuracy.
    fn max_centroid_weight(&self, q: f64, total_weight: f64) -> f64 {
        4.0 * total_weight / self.compression * q * (1.0 - q)
    }
}

impl Default for TDigest {
    fn default() -> Self {
        Self::with_default_compression()
    }
}

// ---------------------------------------------------------------------------
// MedianMAD
// ---------------------------------------------------------------------------

/// Scale factor converting MAD to a consistent estimator of σ under normality.
const MAD_CONSISTENCY_FACTOR: f64 = 1.4826;

/// Robust center and spread estimation via median and Median Absolute Deviation.
///
/// Uses paired t-digests for O(log δ) amortized updates, meeting the <500ns
/// per-update budget. The MAD is scaled by 1.4826 to be a consistent estimator
/// of the standard deviation under normality.
///
/// Both median and MAD have a 50% breakdown point, meaning they remain valid
/// even when up to half the data is corrupted.
#[derive(Debug, Clone)]
pub struct MedianMAD {
    /// T-digest for estimating the median of raw values.
    /// Also serves as the primary quantile source for `RobustMetrics`.
    main: TDigest,
    /// T-digest for estimating the median of absolute deviations.
    deviation: TDigest,
    /// Running median estimate for deviation tracking.
    running_median: f64,
    /// Observations since last median refresh.
    since_refresh: u64,
    /// How often to refresh the running median.
    refresh_interval: u64,
    /// Whether we have seen at least one observation.
    initialized: bool,
}

impl MedianMAD {
    /// Creates a new MedianMAD estimator with default compression.
    #[must_use]
    pub fn new() -> Self {
        Self::with_compression(TDigest::DEFAULT_COMPRESSION)
    }

    /// Creates a new MedianMAD estimator with the given t-digest compression.
    #[must_use]
    pub fn with_compression(compression: f64) -> Self {
        Self {
            main: TDigest::new(compression),
            deviation: TDigest::new(compression),
            running_median: 0.0,
            since_refresh: 0,
            refresh_interval: 100,
            initialized: false,
        }
    }

    /// Records a single observation.
    pub fn insert(&mut self, value: f64) {
        if value.is_nan() {
            return;
        }

        self.main.insert(value);

        if !self.initialized {
            self.running_median = value;
            self.initialized = true;
        }

        self.since_refresh += 1;
        if self.since_refresh >= self.refresh_interval {
            if let Some(m) = self.main.p50() {
                self.running_median = m;
            }
            self.since_refresh = 0;
        }

        let deviation = (value - self.running_median).abs();
        self.deviation.insert(deviation);
    }

    /// Returns the estimated median.
    pub fn median(&mut self) -> Option<f64> {
        self.main.p50()
    }

    /// Returns the MAD scaled by 1.4826 (consistent with σ under normality).
    pub fn mad(&mut self) -> Option<f64> {
        self.deviation.p50().map(|d| d * MAD_CONSISTENCY_FACTOR)
    }

    /// Returns the raw (unscaled) MAD.
    pub fn mad_raw(&mut self) -> Option<f64> {
        self.deviation.p50()
    }

    /// Total number of observations.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.main.count()
    }

    /// Discards all observations.
    pub fn reset(&mut self) {
        self.main.reset();
        self.deviation.reset();
        self.running_median = 0.0;
        self.since_refresh = 0;
        self.initialized = false;
    }
}

impl Default for MedianMAD {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HuberEstimator
// ---------------------------------------------------------------------------

/// Streaming Huber M-estimator with bounded influence function.
///
/// Computes an outlier-resistant mean using an exponentially weighted update.
/// Observations beyond `k` units from the current estimate receive clipped
/// influence, bounding their effect on the result.
///
/// With default k=1.345, achieves 95% asymptotic efficiency under normality
/// while providing robust resistance to individual outliers.
///
/// The Huber M-estimator has breakdown point 0% (like all M-estimators of
/// location without scale). For true breakdown robustness, pair with
/// [`MedianMAD`] for scale estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuberEstimator {
    estimate: f64,
    k: f64,
    alpha: f64,
    count: u64,
    initialized: bool,
}

impl HuberEstimator {
    /// Default tuning constant (95% efficiency under normality).
    pub const DEFAULT_K: f64 = 1.345;

    /// Default exponential decay factor.
    pub const DEFAULT_ALPHA: f64 = 0.01;

    /// Creates a new Huber estimator with default parameters (k=1.345, α=0.01).
    #[must_use]
    pub fn new() -> Self {
        Self::with_params(Self::DEFAULT_K, Self::DEFAULT_ALPHA)
    }

    /// Creates a new Huber estimator with custom tuning constant and decay.
    ///
    /// - `k`: Outlier threshold. Smaller = more robust, less efficient.
    /// - `alpha`: Decay factor for streaming updates. Smaller = smoother.
    #[must_use]
    pub fn with_params(k: f64, alpha: f64) -> Self {
        Self {
            estimate: 0.0,
            k: k.abs().max(0.1),
            alpha: alpha.clamp(0.001, 1.0),
            count: 0,
            initialized: false,
        }
    }

    /// Records a single observation.
    pub fn insert(&mut self, value: f64) {
        if value.is_nan() {
            return;
        }

        self.count += 1;

        if !self.initialized {
            self.estimate = value;
            self.initialized = true;
            return;
        }

        let residual = value - self.estimate;
        let psi = self.huber_psi(residual);
        self.estimate += self.alpha * psi;
    }

    /// Returns the current robust mean estimate, or `None` if empty.
    #[must_use]
    pub fn estimate(&self) -> Option<f64> {
        if self.initialized {
            Some(self.estimate)
        } else {
            None
        }
    }

    /// Number of observations processed.
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Discards all state.
    pub fn reset(&mut self) {
        self.estimate = 0.0;
        self.count = 0;
        self.initialized = false;
    }

    /// Huber's ψ function: identity inside \[-k, k\], clipped outside.
    fn huber_psi(&self, residual: f64) -> f64 {
        if residual.abs() <= self.k {
            residual
        } else {
            self.k * residual.signum()
        }
    }
}

impl Default for HuberEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HyperLogLog
// ---------------------------------------------------------------------------

/// Probabilistic cardinality estimation using the HyperLogLog algorithm.
///
/// Estimates the number of distinct elements in a stream with ~0.81%
/// standard error using 16,384 registers (~16KB memory) at default
/// precision (p=14).
///
/// Useful for tracking unique queries and unique document IDs in search
/// results without storing them.
#[derive(Debug, Clone)]
pub struct HyperLogLog {
    registers: Vec<u8>,
    precision: u8,
    num_registers: usize,
}

impl HyperLogLog {
    /// Default precision (p=14, m=16384 registers, SE ≈ 0.81%).
    pub const DEFAULT_PRECISION: u8 = 14;

    /// Creates a new HyperLogLog with the given precision (4 ≤ p ≤ 18).
    #[must_use]
    pub fn new(precision: u8) -> Self {
        let precision = precision.clamp(4, 18);
        let num_registers = 1usize << precision;
        Self {
            registers: vec![0u8; num_registers],
            precision,
            num_registers,
        }
    }

    /// Creates a HyperLogLog with default precision (p=14).
    #[must_use]
    pub fn with_default_precision() -> Self {
        Self::new(Self::DEFAULT_PRECISION)
    }

    /// Records a hashable element.
    pub fn insert<T: Hash>(&mut self, value: &T) {
        let hash = self.hash_value(value);
        let index = (hash >> (64 - self.precision)) as usize;
        let w = hash << self.precision;
        let rho = w.leading_zeros() as u8 + 1;
        self.registers[index] = self.registers[index].max(rho);
    }

    /// Returns the estimated number of distinct elements.
    #[must_use]
    pub fn count(&self) -> u64 {
        let m = self.num_registers as f64;
        let alpha = self.alpha_m();

        let sum: f64 = self
            .registers
            .iter()
            .map(|&r| 2.0_f64.powi(-i32::from(r)))
            .sum();

        let raw_estimate = alpha * m * m / sum;

        // Small-range correction (linear counting)
        if raw_estimate <= 2.5 * m {
            let zeros = self.registers.iter().filter(|&&r| r == 0).count() as f64;
            if zeros > 0.0 {
                return (m * (m / zeros).ln()) as u64;
            }
        }

        // Large-range correction (hash collision)
        let two_32 = (1u64 << 32) as f64;
        if raw_estimate > two_32 / 30.0 {
            return (-two_32 * (1.0 - raw_estimate / two_32).ln()) as u64;
        }

        raw_estimate as u64
    }

    /// Merges another HyperLogLog into this one (element-wise max).
    ///
    /// Both must have the same precision; panics otherwise.
    pub fn merge(&mut self, other: &HyperLogLog) {
        assert_eq!(
            self.precision, other.precision,
            "HyperLogLog precision mismatch: {} vs {}",
            self.precision, other.precision
        );
        for (a, &b) in self.registers.iter_mut().zip(other.registers.iter()) {
            *a = (*a).max(b);
        }
    }

    /// Discards all state.
    pub fn reset(&mut self) {
        self.registers.fill(0);
    }

    /// Bytes used by registers.
    #[must_use]
    pub const fn memory_bytes(&self) -> usize {
        self.num_registers
    }

    fn hash_value<T: Hash>(&self, value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    /// Bias correction constant α_m.
    fn alpha_m(&self) -> f64 {
        match self.num_registers {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            m => 0.7213 / (1.0 + 1.079 / m as f64),
        }
    }
}

impl Default for HyperLogLog {
    fn default() -> Self {
        Self::with_default_precision()
    }
}

// ---------------------------------------------------------------------------
// RobustMetrics
// ---------------------------------------------------------------------------

/// Composite robust metrics collector for a single metric stream.
///
/// Combines streaming quantile estimation, robust center/spread (median + MAD),
/// and outlier-resistant mean (Huber) into a single struct. Each observation
/// is recorded in the MedianMAD (which handles quantiles via its main t-digest)
/// and the Huber estimator.
///
/// Memory: ~8KB per instance (two t-digests in MedianMAD + Huber state).
///
/// # Example
///
/// ```rust,ignore
/// let mut metrics = RobustMetrics::new();
/// for latency_ms in [1.2, 1.5, 1.3, 1.4, 50.0] {
///     metrics.insert(latency_ms);
/// }
/// // Median is ~1.4, unaffected by the 50.0 outlier
/// let median = metrics.median().unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RobustMetrics {
    median_mad: MedianMAD,
    huber: HuberEstimator,
    count: u64,
}

impl RobustMetrics {
    /// Creates a new collector with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            median_mad: MedianMAD::new(),
            huber: HuberEstimator::new(),
            count: 0,
        }
    }

    /// Records a single observation into all estimators.
    pub fn insert(&mut self, value: f64) {
        if value.is_nan() {
            return;
        }
        self.count += 1;
        self.median_mad.insert(value);
        self.huber.insert(value);
    }

    /// Estimated median (from t-digest).
    pub fn median(&mut self) -> Option<f64> {
        self.median_mad.median()
    }

    /// MAD-based spread estimate (scaled for normal consistency).
    pub fn mad(&mut self) -> Option<f64> {
        self.median_mad.mad()
    }

    /// Huber robust mean estimate.
    #[must_use]
    pub fn huber_mean(&self) -> Option<f64> {
        self.huber.estimate()
    }

    /// Estimated p90.
    pub fn p90(&mut self) -> Option<f64> {
        self.median_mad.main.p90()
    }

    /// Estimated p95.
    pub fn p95(&mut self) -> Option<f64> {
        self.median_mad.main.p95()
    }

    /// Estimated p99.
    pub fn p99(&mut self) -> Option<f64> {
        self.median_mad.main.p99()
    }

    /// Estimated p999.
    pub fn p999(&mut self) -> Option<f64> {
        self.median_mad.main.p999()
    }

    /// Arbitrary quantile from the t-digest.
    pub fn quantile(&mut self, q: f64) -> Option<f64> {
        self.median_mad.main.quantile(q)
    }

    /// Observed minimum.
    #[must_use]
    pub fn min(&self) -> Option<f64> {
        self.median_mad.main.min()
    }

    /// Observed maximum.
    #[must_use]
    pub fn max(&self) -> Option<f64> {
        self.median_mad.main.max()
    }

    /// Total observations recorded.
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Merges another `RobustMetrics` into this one.
    pub fn merge(&mut self, other: &RobustMetrics) {
        self.median_mad.main.merge(&other.median_mad.main);
        self.median_mad.deviation.merge(&other.median_mad.deviation);
        self.count += other.count;
        // Huber is streaming-only and does not support merge.
    }

    /// Discards all observations.
    pub fn reset(&mut self) {
        self.median_mad.reset();
        self.huber.reset();
        self.count = 0;
    }

    /// Emits a structured tracing report with all current metrics.
    pub fn report(&mut self, label: &str) {
        let p50 = self.median_mad.main.p50().unwrap_or(0.0);
        let p90 = self.median_mad.main.p90().unwrap_or(0.0);
        let p95 = self.median_mad.main.p95().unwrap_or(0.0);
        let p99 = self.median_mad.main.p99().unwrap_or(0.0);
        let p999 = self.median_mad.main.p999().unwrap_or(0.0);
        let mad = self.median_mad.mad().unwrap_or(0.0);
        let huber = self.huber.estimate().unwrap_or(0.0);

        tracing::info!(
            label,
            count = self.count,
            p50,
            p90,
            p95,
            p99,
            p999,
            mad,
            huber_mean = huber,
            "robust_metrics_report"
        );
    }

    /// Creates a serializable snapshot of current metrics.
    pub fn snapshot(&mut self, label: impl Into<String>) -> RobustMetricsSnapshot {
        RobustMetricsSnapshot {
            label: label.into(),
            count: self.count,
            p50: self.median_mad.main.p50().unwrap_or(0.0),
            p90: self.median_mad.main.p90().unwrap_or(0.0),
            p95: self.median_mad.main.p95().unwrap_or(0.0),
            p99: self.median_mad.main.p99().unwrap_or(0.0),
            p999: self.median_mad.main.p999().unwrap_or(0.0),
            mad: self.median_mad.mad().unwrap_or(0.0),
            huber_mean: self.huber.estimate().unwrap_or(0.0),
            min: self.median_mad.main.min().unwrap_or(0.0),
            max: self.median_mad.main.max().unwrap_or(0.0),
        }
    }
}

impl Default for RobustMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable snapshot of [`RobustMetrics`] at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustMetricsSnapshot {
    /// Label identifying what this metric tracks (e.g., "fast_latency_ms").
    pub label: String,
    /// Number of observations.
    pub count: u64,
    /// Estimated p50 (median).
    pub p50: f64,
    /// Estimated p90.
    pub p90: f64,
    /// Estimated p95.
    pub p95: f64,
    /// Estimated p99.
    pub p99: f64,
    /// Estimated p999.
    pub p999: f64,
    /// Median Absolute Deviation (scaled by 1.4826).
    pub mad: f64,
    /// Huber robust mean.
    pub huber_mean: f64,
    /// Observed minimum.
    pub min: f64,
    /// Observed maximum.
    pub max: f64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- TDigest tests ----

    #[test]
    fn tdigest_empty_returns_none() {
        let mut td = TDigest::default();
        assert!(td.quantile(0.5).is_none());
        assert!(td.min().is_none());
        assert!(td.max().is_none());
        assert_eq!(td.count(), 0);
    }

    #[test]
    fn tdigest_single_value() {
        let mut td = TDigest::default();
        td.insert(42.0);
        assert_eq!(td.p50(), Some(42.0));
        assert_eq!(td.min(), Some(42.0));
        assert_eq!(td.max(), Some(42.0));
        assert_eq!(td.count(), 1);
    }

    #[test]
    fn tdigest_nan_ignored() {
        let mut td = TDigest::default();
        td.insert(f64::NAN);
        assert_eq!(td.count(), 0);
        assert!(td.p50().is_none());
    }

    #[test]
    fn tdigest_uniform_quantiles() {
        let mut td = TDigest::default();
        // Insert values 1..=1000
        for i in 1..=1000 {
            td.insert(i as f64);
        }

        let p50 = td.p50().unwrap();
        assert!(
            (p50 - 500.0).abs() < 15.0,
            "p50 should be ~500, got {p50}"
        );

        let p90 = td.p90().unwrap();
        assert!(
            (p90 - 900.0).abs() < 30.0,
            "p90 should be ~900, got {p90}"
        );

        let p99 = td.p99().unwrap();
        assert!(
            (p99 - 990.0).abs() < 20.0,
            "p99 should be ~990, got {p99}"
        );

        assert_eq!(td.min(), Some(1.0));
        assert_eq!(td.max(), Some(1000.0));
    }

    #[test]
    fn tdigest_merge_preserves_quantiles() {
        let mut td1 = TDigest::default();
        let mut td2 = TDigest::default();

        for i in 1..=500 {
            td1.insert(i as f64);
        }
        for i in 501..=1000 {
            td2.insert(i as f64);
        }

        td1.merge(&td2);

        let p50 = td1.p50().unwrap();
        assert!(
            (p50 - 500.0).abs() < 20.0,
            "merged p50 should be ~500, got {p50}"
        );
        assert_eq!(td1.count(), 1000);
        assert_eq!(td1.min(), Some(1.0));
        assert_eq!(td1.max(), Some(1000.0));
    }

    #[test]
    fn tdigest_quantile_edges() {
        let mut td = TDigest::default();
        for i in 1..=100 {
            td.insert(i as f64);
        }
        assert_eq!(td.quantile(0.0), Some(1.0));
        assert_eq!(td.quantile(1.0), Some(100.0));
    }

    #[test]
    fn tdigest_compression_bounds_centroids() {
        let mut td = TDigest::new(50.0);
        for i in 0..10_000 {
            td.insert(i as f64);
        }
        // Force compression
        let _ = td.p50();
        // Centroid count should be bounded by compression parameter
        assert!(
            td.centroid_count() < 200,
            "expected < 200 centroids, got {}",
            td.centroid_count()
        );
    }

    #[test]
    fn tdigest_reset() {
        let mut td = TDigest::default();
        td.insert(1.0);
        td.insert(2.0);
        td.reset();
        assert_eq!(td.count(), 0);
        assert!(td.p50().is_none());
    }

    // ---- MedianMAD tests ----

    #[test]
    fn median_mad_empty() {
        let mut mm = MedianMAD::default();
        assert!(mm.median().is_none());
        assert!(mm.mad().is_none());
        assert_eq!(mm.count(), 0);
    }

    #[test]
    fn median_mad_known_dataset() {
        let mut mm = MedianMAD::new();
        // Dataset: [1, 2, 3, 4, 100] — median=3
        // Deviations from median: [2, 1, 0, 1, 97] — median deviation=1
        // MAD = 1 * 1.4826 = 1.4826
        for &v in &[1.0, 2.0, 3.0, 4.0, 100.0] {
            mm.insert(v);
        }
        let median = mm.median().unwrap();
        assert!(
            (median - 3.0).abs() < 1.0,
            "median should be ~3, got {median}"
        );
    }

    #[test]
    fn median_mad_symmetric_data() {
        let mut mm = MedianMAD::new();
        // Insert symmetric data around 50
        for i in 1..=100 {
            mm.insert(i as f64);
        }
        let median = mm.median().unwrap();
        assert!(
            (median - 50.0).abs() < 5.0,
            "median should be ~50, got {median}"
        );
    }

    #[test]
    fn median_mad_nan_ignored() {
        let mut mm = MedianMAD::new();
        mm.insert(f64::NAN);
        assert_eq!(mm.count(), 0);
    }

    #[test]
    fn median_mad_reset() {
        let mut mm = MedianMAD::new();
        mm.insert(42.0);
        mm.reset();
        assert_eq!(mm.count(), 0);
        assert!(mm.median().is_none());
    }

    // ---- HuberEstimator tests ----

    #[test]
    fn huber_empty() {
        let h = HuberEstimator::default();
        assert!(h.estimate().is_none());
        assert_eq!(h.count(), 0);
    }

    #[test]
    fn huber_single_value() {
        let mut h = HuberEstimator::new();
        h.insert(42.0);
        assert_eq!(h.estimate(), Some(42.0));
        assert_eq!(h.count(), 1);
    }

    #[test]
    fn huber_normal_data_converges_to_mean() {
        // With large alpha for faster convergence in testing
        let mut h = HuberEstimator::with_params(1.345, 0.1);
        // Insert values centered around 10
        let data = [9.5, 10.5, 10.0, 9.8, 10.2, 10.1, 9.9, 10.3, 9.7, 10.0];
        for &v in &data {
            h.insert(v);
        }
        let est = h.estimate().unwrap();
        let true_mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        assert!(
            (est - true_mean).abs() < 1.0,
            "Huber estimate {est} should be near mean {true_mean}"
        );
    }

    #[test]
    fn huber_resists_outliers() {
        let mut h = HuberEstimator::with_params(1.345, 0.1);
        // Normal data near 10, then one extreme outlier
        for &v in &[10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 10.0] {
            h.insert(v);
        }
        let before_outlier = h.estimate().unwrap();

        // Insert extreme outlier
        h.insert(1000.0);
        let after_outlier = h.estimate().unwrap();

        // The outlier should have bounded influence (shifted by at most k * alpha)
        let shift = (after_outlier - before_outlier).abs();
        assert!(
            shift < 1.0,
            "outlier shift {shift} should be bounded by Huber psi"
        );
    }

    #[test]
    fn huber_nan_ignored() {
        let mut h = HuberEstimator::new();
        h.insert(5.0);
        h.insert(f64::NAN);
        assert_eq!(h.count(), 1);
        assert_eq!(h.estimate(), Some(5.0));
    }

    #[test]
    fn huber_reset() {
        let mut h = HuberEstimator::new();
        h.insert(42.0);
        h.reset();
        assert_eq!(h.count(), 0);
        assert!(h.estimate().is_none());
    }

    #[test]
    fn huber_serde_roundtrip() {
        let mut h = HuberEstimator::with_params(1.5, 0.05);
        h.insert(10.0);
        h.insert(20.0);
        let json = serde_json::to_string(&h).unwrap();
        let h2: HuberEstimator = serde_json::from_str(&json).unwrap();
        assert_eq!(h.estimate(), h2.estimate());
        assert_eq!(h.count(), h2.count());
    }

    // ---- HyperLogLog tests ----

    #[test]
    fn hll_empty() {
        let hll = HyperLogLog::default();
        assert_eq!(hll.count(), 0);
    }

    #[test]
    fn hll_single_element() {
        let mut hll = HyperLogLog::default();
        hll.insert(&"hello");
        assert!(hll.count() >= 1);
    }

    #[test]
    fn hll_10k_distinct_elements() {
        let mut hll = HyperLogLog::with_default_precision();
        for i in 0..10_000u64 {
            hll.insert(&i);
        }
        let estimate = hll.count();
        let error = (estimate as f64 - 10_000.0).abs() / 10_000.0;
        assert!(
            error < 0.05,
            "HLL estimate {estimate} should be within 5% of 10000 (error: {:.1}%)",
            error * 100.0
        );
    }

    #[test]
    fn hll_duplicates_dont_increase_count() {
        let mut hll = HyperLogLog::with_default_precision();
        for _ in 0..1000 {
            hll.insert(&"same_value");
        }
        // Should report ~1 unique element
        assert!(
            hll.count() <= 5,
            "repeated element should not inflate count: got {}",
            hll.count()
        );
    }

    #[test]
    fn hll_merge() {
        let mut hll1 = HyperLogLog::with_default_precision();
        let mut hll2 = HyperLogLog::with_default_precision();

        for i in 0..5000u64 {
            hll1.insert(&i);
        }
        for i in 5000..10_000u64 {
            hll2.insert(&i);
        }

        hll1.merge(&hll2);
        let estimate = hll1.count();
        let error = (estimate as f64 - 10_000.0).abs() / 10_000.0;
        assert!(
            error < 0.05,
            "merged HLL estimate {estimate} should be within 5% of 10000"
        );
    }

    #[test]
    fn hll_reset() {
        let mut hll = HyperLogLog::default();
        hll.insert(&42u64);
        hll.reset();
        assert_eq!(hll.count(), 0);
    }

    #[test]
    fn hll_memory_size() {
        let hll = HyperLogLog::with_default_precision();
        assert_eq!(hll.memory_bytes(), 16384);
    }

    #[test]
    #[should_panic(expected = "precision mismatch")]
    fn hll_merge_different_precision_panics() {
        let mut hll1 = HyperLogLog::new(10);
        let hll2 = HyperLogLog::new(12);
        hll1.merge(&hll2);
    }

    // ---- RobustMetrics tests ----

    #[test]
    fn robust_metrics_empty() {
        let mut rm = RobustMetrics::new();
        assert!(rm.median().is_none());
        assert!(rm.mad().is_none());
        assert!(rm.huber_mean().is_none());
        assert!(rm.p90().is_none());
        assert!(rm.min().is_none());
        assert!(rm.max().is_none());
        assert_eq!(rm.count(), 0);
    }

    #[test]
    fn robust_metrics_basic_workflow() {
        let mut rm = RobustMetrics::new();
        for i in 1..=100 {
            rm.insert(i as f64);
        }

        assert_eq!(rm.count(), 100);

        let median = rm.median().unwrap();
        assert!(
            (median - 50.0).abs() < 5.0,
            "median should be ~50, got {median}"
        );

        let p90 = rm.p90().unwrap();
        assert!(
            (p90 - 90.0).abs() < 10.0,
            "p90 should be ~90, got {p90}"
        );

        assert_eq!(rm.min(), Some(1.0));
        assert_eq!(rm.max(), Some(100.0));

        let huber = rm.huber_mean().unwrap();
        assert!(
            (huber - 50.0).abs() < 20.0,
            "Huber mean should be near 50, got {huber}"
        );
    }

    #[test]
    fn robust_metrics_outlier_resistance() {
        let mut rm = RobustMetrics::new();
        // 99 values near 10, one extreme outlier
        for _ in 0..99 {
            rm.insert(10.0);
        }
        rm.insert(10_000.0);

        let median = rm.median().unwrap();
        assert!(
            (median - 10.0).abs() < 5.0,
            "median should be ~10, got {median}"
        );
    }

    #[test]
    fn robust_metrics_merge() {
        let mut rm1 = RobustMetrics::new();
        let mut rm2 = RobustMetrics::new();

        for i in 1..=50 {
            rm1.insert(i as f64);
        }
        for i in 51..=100 {
            rm2.insert(i as f64);
        }

        rm1.merge(&rm2);
        assert_eq!(rm1.count(), 100);

        let median = rm1.median().unwrap();
        assert!(
            (median - 50.0).abs() < 10.0,
            "merged median should be ~50, got {median}"
        );
    }

    #[test]
    fn robust_metrics_nan_ignored() {
        let mut rm = RobustMetrics::new();
        rm.insert(f64::NAN);
        assert_eq!(rm.count(), 0);
    }

    #[test]
    fn robust_metrics_reset() {
        let mut rm = RobustMetrics::new();
        rm.insert(42.0);
        rm.reset();
        assert_eq!(rm.count(), 0);
        assert!(rm.median().is_none());
    }

    #[test]
    fn robust_metrics_snapshot_serde() {
        let mut rm = RobustMetrics::new();
        for i in 1..=100 {
            rm.insert(i as f64);
        }

        let snap = rm.snapshot("test_latency_ms");
        let json = serde_json::to_string(&snap).unwrap();
        let decoded: RobustMetricsSnapshot = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.label, "test_latency_ms");
        assert_eq!(decoded.count, 100);
        assert!(decoded.p50 > 0.0);
        assert!(decoded.p90 > decoded.p50);
        assert!(decoded.p99 > decoded.p90);
        assert!((decoded.min - 1.0).abs() < f64::EPSILON);
        assert!((decoded.max - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn robust_metrics_quantile_arbitrary() {
        let mut rm = RobustMetrics::new();
        for i in 1..=1000 {
            rm.insert(i as f64);
        }
        let q75 = rm.quantile(0.75).unwrap();
        assert!(
            (q75 - 750.0).abs() < 30.0,
            "p75 should be ~750, got {q75}"
        );
    }

    // ---- Performance sanity ----

    #[test]
    fn tdigest_10k_inserts_performance() {
        let mut td = TDigest::default();
        let start = std::time::Instant::now();
        for i in 0..10_000 {
            td.insert(i as f64);
        }
        let elapsed = start.elapsed();
        // Should complete in < 50ms (amortized < 5µs per insert)
        assert!(
            elapsed.as_millis() < 50,
            "10K inserts took {}ms, expected < 50ms",
            elapsed.as_millis()
        );
    }

    #[test]
    fn robust_metrics_10k_inserts_performance() {
        let mut rm = RobustMetrics::new();
        let start = std::time::Instant::now();
        for i in 0..10_000 {
            rm.insert(i as f64);
        }
        let elapsed = start.elapsed();
        // Should complete in < 100ms (amortized < 10µs per insert)
        assert!(
            elapsed.as_millis() < 100,
            "10K robust inserts took {}ms, expected < 100ms",
            elapsed.as_millis()
        );
    }
}
