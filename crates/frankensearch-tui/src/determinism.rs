//! Deterministic time and seed injection for replay and testing.
//!
//! Provides a [`Clock`] trait that abstracts time sources, allowing tests
//! and replay sessions to substitute [`TickClock`] (virtual, manually
//! advanced) for [`WallClock`] (real `Instant::now()`).
//!
//! The [`DeterministicSeed`] carries a fixed random seed so that any
//! non-deterministic choices (tie-breaking, jitter, sampling) produce
//! reproducible outcomes during replay.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

// ─── Clock Trait ────────────────────────────────────────────────────────────

/// Abstraction over time sources for deterministic replay.
///
/// Production code uses [`WallClock`]. Tests and replay use [`TickClock`]
/// where time only advances when explicitly ticked.
pub trait Clock: Send {
    /// Current instant (monotonic).
    fn now(&self) -> Instant;

    /// Elapsed duration since a previous instant.
    fn elapsed_since(&self, earlier: Instant) -> Duration {
        self.now().duration_since(earlier)
    }
}

// ─── Wall Clock ─────────────────────────────────────────────────────────────

/// Production clock backed by `Instant::now()`.
#[derive(Debug, Clone, Copy)]
pub struct WallClock;

impl Clock for WallClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

// ─── Tick Clock ─────────────────────────────────────────────────────────────

/// Virtual clock for deterministic testing.
///
/// Time only advances when [`tick()`](TickClock::tick) is called. This
/// ensures frame sequences and timeout logic behave identically across
/// replays regardless of real wall-clock jitter.
pub struct TickClock {
    /// The fixed anchor instant (captured once at creation).
    anchor: Instant,
    /// Accumulated virtual elapsed time.
    elapsed: Duration,
    /// Tick size (each `tick()` call advances by this amount).
    tick_size: Duration,
    /// Total ticks advanced.
    tick_count: u64,
}

impl TickClock {
    /// Create a tick clock with the given tick interval.
    #[must_use]
    pub fn new(tick_size: Duration) -> Self {
        Self {
            anchor: Instant::now(),
            elapsed: Duration::ZERO,
            tick_size,
            tick_count: 0,
        }
    }

    /// Create a tick clock at ~60 FPS (~16.67ms per tick).
    #[must_use]
    pub fn at_60fps() -> Self {
        Self::new(Duration::from_micros(16_667))
    }

    /// Create a tick clock at ~30 FPS (~33.33ms per tick).
    #[must_use]
    pub fn at_30fps() -> Self {
        Self::new(Duration::from_micros(33_333))
    }

    /// Advance time by one tick.
    pub fn tick(&mut self) {
        self.elapsed += self.tick_size;
        self.tick_count += 1;
    }

    /// Advance time by `n` ticks.
    pub fn tick_n(&mut self, n: u64) {
        self.elapsed += self.tick_size * u32::try_from(n).unwrap_or(u32::MAX);
        self.tick_count += n;
    }

    /// Total ticks that have been advanced.
    #[must_use]
    pub const fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// The tick interval.
    #[must_use]
    pub const fn tick_size(&self) -> Duration {
        self.tick_size
    }

    /// Total virtual elapsed time.
    #[must_use]
    pub const fn virtual_elapsed(&self) -> Duration {
        self.elapsed
    }
}

impl Clock for TickClock {
    fn now(&self) -> Instant {
        self.anchor + self.elapsed
    }
}

// ─── Deterministic Seed ─────────────────────────────────────────────────────

/// Fixed random seed for deterministic replay.
///
/// Any code path that makes non-deterministic choices (tie-breaking,
/// sampling, jitter) should check `ReplayContext::seed()` and use it
/// to seed local RNGs rather than reading from the OS entropy pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeterministicSeed(pub u64);

impl DeterministicSeed {
    /// Create a new deterministic seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Get the raw seed value.
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }

    /// Derive a sub-seed for a named subsystem.
    ///
    /// Combines the base seed with a hash of the subsystem name to
    /// produce independent-but-reproducible seeds for different components.
    #[must_use]
    pub fn derive(self, subsystem: &str) -> Self {
        // Simple FNV-1a hash of the subsystem name combined with the seed.
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325 ^ self.0;
        for byte in subsystem.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        Self(hash)
    }
}

// ─── Replay Context ─────────────────────────────────────────────────────────

/// Context for a deterministic replay session.
///
/// Bundles the replay mode, clock, seed, and frame counter into a single
/// injectable context. Production code receives `None` (live mode);
/// tests and replay receive `Some(ReplayContext)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayMode {
    /// Live production mode.
    Live,
    /// Deterministic replay mode.
    Deterministic,
}

/// Replay metadata emitted alongside evidence events.
///
/// In live mode only `mode` is set. In deterministic mode all fields are
/// populated so that evidence JSONL lines satisfy the schema's conditional
/// requirement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayMetadata {
    /// Whether this is live or deterministic.
    pub mode: ReplayMode,
    /// Deterministic seed (present in deterministic mode).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Tick interval in milliseconds (present in deterministic mode).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tick_ms: Option<u64>,
    /// Current frame sequence number (present in deterministic mode).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame_seq: Option<u64>,
}

impl ReplayMetadata {
    /// Create live-mode metadata.
    #[must_use]
    pub const fn live() -> Self {
        Self {
            mode: ReplayMode::Live,
            seed: None,
            tick_ms: None,
            frame_seq: None,
        }
    }

    /// Create deterministic-mode metadata.
    #[must_use]
    pub const fn deterministic(seed: u64, tick_ms: u64, frame_seq: u64) -> Self {
        Self {
            mode: ReplayMode::Deterministic,
            seed: Some(seed),
            tick_ms: Some(tick_ms),
            frame_seq: Some(frame_seq),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn wall_clock_monotonic() {
        let clock = WallClock;
        let a = clock.now();
        let b = clock.now();
        assert!(b >= a);
    }

    #[test]
    fn tick_clock_starts_at_zero() {
        let clock = TickClock::new(Duration::from_millis(10));
        assert_eq!(clock.tick_count(), 0);
        assert_eq!(clock.virtual_elapsed(), Duration::ZERO);
    }

    #[test]
    fn tick_clock_advances() {
        let mut clock = TickClock::new(Duration::from_millis(10));
        let start = clock.now();

        clock.tick();
        assert_eq!(clock.tick_count(), 1);
        assert_eq!(clock.virtual_elapsed(), Duration::from_millis(10));
        assert_eq!(clock.elapsed_since(start), Duration::from_millis(10));

        clock.tick();
        assert_eq!(clock.tick_count(), 2);
        assert_eq!(clock.virtual_elapsed(), Duration::from_millis(20));
    }

    #[test]
    fn tick_clock_tick_n() {
        let mut clock = TickClock::new(Duration::from_millis(5));
        clock.tick_n(10);
        assert_eq!(clock.tick_count(), 10);
        assert_eq!(clock.virtual_elapsed(), Duration::from_millis(50));
    }

    #[test]
    fn tick_clock_60fps() {
        let clock = TickClock::at_60fps();
        assert_eq!(clock.tick_size(), Duration::from_micros(16_667));
    }

    #[test]
    fn tick_clock_30fps() {
        let clock = TickClock::at_30fps();
        assert_eq!(clock.tick_size(), Duration::from_micros(33_333));
    }

    #[test]
    fn deterministic_seed_derive() {
        let seed = DeterministicSeed::new(42);
        let sub_a = seed.derive("rrf");
        let sub_b = seed.derive("blend");

        // Derived seeds should differ from each other and the original.
        assert_ne!(sub_a, sub_b);
        assert_ne!(sub_a, seed);
        assert_ne!(sub_b, seed);

        // But should be deterministic.
        assert_eq!(seed.derive("rrf"), sub_a);
        assert_eq!(seed.derive("blend"), sub_b);
    }

    #[test]
    fn deterministic_seed_value() {
        let seed = DeterministicSeed::new(12345);
        assert_eq!(seed.value(), 12345);
    }

    #[test]
    fn deterministic_seed_serde_roundtrip() {
        let seed = DeterministicSeed::new(999);
        let json = serde_json::to_string(&seed).unwrap();
        let decoded: DeterministicSeed = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, seed);
    }

    #[test]
    fn replay_mode_serde_roundtrip() {
        for mode in [ReplayMode::Live, ReplayMode::Deterministic] {
            let json = serde_json::to_string(&mode).unwrap();
            let decoded: ReplayMode = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, mode);
        }
    }

    #[test]
    fn replay_metadata_live() {
        let meta = ReplayMetadata::live();
        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("\"live\""));
        assert!(!json.contains("seed"));
        assert!(!json.contains("tick_ms"));
        assert!(!json.contains("frame_seq"));
    }

    #[test]
    fn replay_metadata_deterministic() {
        let meta = ReplayMetadata::deterministic(42, 16, 100);
        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("\"deterministic\""));
        assert!(json.contains("\"seed\":42"));
        assert!(json.contains("\"tick_ms\":16"));
        assert!(json.contains("\"frame_seq\":100"));
    }

    #[test]
    fn replay_metadata_serde_roundtrip() {
        let meta = ReplayMetadata::deterministic(7, 33, 0);
        let json = serde_json::to_string(&meta).unwrap();
        let decoded: ReplayMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.seed, Some(7));
        assert_eq!(decoded.tick_ms, Some(33));
        assert_eq!(decoded.frame_seq, Some(0));
    }
}
