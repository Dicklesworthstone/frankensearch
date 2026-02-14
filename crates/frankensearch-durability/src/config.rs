use frankensearch_core::{SearchError, SearchResult};
use serde::{Deserialize, Serialize};

const MIN_SYMBOL_SIZE: u32 = 256;
const MAX_SYMBOL_SIZE: u32 = 64 * 1024;
const MIN_MAX_REPAIR_SYMBOLS: u32 = 1;

/// Durability configuration for repair-symbol generation and recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurabilityConfig {
    /// Symbol size used by the codec.
    pub symbol_size: u32,
    /// Maximum block size used by the upstream codec implementation.
    pub max_block_size: u32,
    /// Repair overhead multiplier (`1.20` = 20% repair symbols).
    pub repair_overhead: f64,
    /// Hard cap for generated repair symbols per payload.
    pub max_repair_symbols: u32,
    /// Minimum repair-symbol slack budget used during decode.
    pub slack_decode: u32,
    /// Checkpoint interval for long-running encode/decode loops.
    pub checkpoint_interval: u32,
    /// Whether verification should run on load/open paths.
    pub verify_on_open: bool,
}

impl Default for DurabilityConfig {
    fn default() -> Self {
        Self {
            symbol_size: 4096,
            max_block_size: 64 * 1024,
            repair_overhead: 1.20,
            max_repair_symbols: 250_000,
            slack_decode: 2,
            checkpoint_interval: 64,
            verify_on_open: true,
        }
    }
}

impl DurabilityConfig {
    /// Convert `repair_overhead` multiplier into an integer percent.
    ///
    /// `1.25` becomes `25`, `2.0` becomes `100`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn repair_overhead_percent(&self) -> u32 {
        let overhead = (self.repair_overhead - 1.0).max(0.0);
        (overhead * 100.0).ceil() as u32
    }

    /// Compute the expected repair symbol budget for a source symbol count.
    ///
    /// Formula:
    /// `R = min(max_repair_symbols, max(slack_decode, ceil(K * overhead_percent/100)))`
    #[must_use]
    pub fn expected_repair_symbols(&self, k_source: u32) -> u32 {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            clippy::cast_sign_loss
        )]
        let formula = ((f64::from(k_source) * f64::from(self.repair_overhead_percent())) / 100.0)
            .ceil() as u32;
        formula.max(self.slack_decode).min(self.max_repair_symbols)
    }

    /// Compute minimum symbol count required to target K+slack decode behavior.
    #[must_use]
    pub fn minimum_decode_symbols(&self, k_source: u32) -> u32 {
        k_source.saturating_add(self.slack_decode)
    }

    /// Validate user-provided configuration values.
    pub fn validate(&self) -> SearchResult<()> {
        if self.symbol_size < MIN_SYMBOL_SIZE
            || self.symbol_size > MAX_SYMBOL_SIZE
            || !self.symbol_size.is_power_of_two()
        {
            return Err(SearchError::InvalidConfig {
                field: "symbol_size".to_owned(),
                value: self.symbol_size.to_string(),
                reason: format!("must be a power of two in [{MIN_SYMBOL_SIZE}, {MAX_SYMBOL_SIZE}]"),
            });
        }
        if self.max_block_size == 0 {
            return Err(SearchError::InvalidConfig {
                field: "max_block_size".to_owned(),
                value: "0".to_owned(),
                reason: "must be greater than zero".to_owned(),
            });
        }
        if !self.repair_overhead.is_finite() || self.repair_overhead < 1.0 {
            return Err(SearchError::InvalidConfig {
                field: "repair_overhead".to_owned(),
                value: self.repair_overhead.to_string(),
                reason: "must be finite and at least 1.0".to_owned(),
            });
        }
        if self.max_repair_symbols < MIN_MAX_REPAIR_SYMBOLS {
            return Err(SearchError::InvalidConfig {
                field: "max_repair_symbols".to_owned(),
                value: self.max_repair_symbols.to_string(),
                reason: "must be at least 1".to_owned(),
            });
        }
        if self.slack_decode == 0 {
            return Err(SearchError::InvalidConfig {
                field: "slack_decode".to_owned(),
                value: "0".to_owned(),
                reason: "must be greater than zero".to_owned(),
            });
        }
        if self.slack_decode > self.max_repair_symbols {
            return Err(SearchError::InvalidConfig {
                field: "slack_decode".to_owned(),
                value: self.slack_decode.to_string(),
                reason: "must not exceed max_repair_symbols".to_owned(),
            });
        }
        if self.checkpoint_interval == 0 {
            return Err(SearchError::InvalidConfig {
                field: "checkpoint_interval".to_owned(),
                value: "0".to_owned(),
                reason: "must be greater than zero".to_owned(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::DurabilityConfig;

    #[test]
    fn default_is_valid() {
        let config = DurabilityConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn invalid_symbol_size_is_rejected() {
        let config = DurabilityConfig {
            symbol_size: 1000,
            ..DurabilityConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn invalid_repair_overhead_is_rejected() {
        let config = DurabilityConfig {
            repair_overhead: 0.5,
            ..DurabilityConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn invalid_max_repair_symbols_is_rejected() {
        let config = DurabilityConfig {
            max_repair_symbols: 0,
            ..DurabilityConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn invalid_slack_decode_is_rejected() {
        let config = DurabilityConfig {
            slack_decode: 0,
            ..DurabilityConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn slack_decode_above_max_repair_is_rejected() {
        let config = DurabilityConfig {
            slack_decode: 10,
            max_repair_symbols: 5,
            ..DurabilityConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn expected_repair_symbol_formula_applies_slack_and_cap() {
        let config = DurabilityConfig {
            repair_overhead: 1.25,
            slack_decode: 2,
            max_repair_symbols: 10_000,
            ..DurabilityConfig::default()
        };

        assert_eq!(config.repair_overhead_percent(), 25);
        assert_eq!(config.expected_repair_symbols(1), 2);
        assert_eq!(config.expected_repair_symbols(20), 5);
        assert_eq!(config.expected_repair_symbols(100_000), 10_000);
        assert_eq!(config.minimum_decode_symbols(100), 102);
    }

    #[test]
    fn nan_repair_overhead_is_rejected() {
        let config = DurabilityConfig {
            repair_overhead: f64::NAN,
            ..DurabilityConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn infinity_repair_overhead_is_rejected() {
        let config = DurabilityConfig {
            repair_overhead: f64::INFINITY,
            ..DurabilityConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn neg_infinity_repair_overhead_is_rejected() {
        let config = DurabilityConfig {
            repair_overhead: f64::NEG_INFINITY,
            ..DurabilityConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn invalid_checkpoint_interval_is_rejected() {
        let config = DurabilityConfig {
            checkpoint_interval: 0,
            ..DurabilityConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn invalid_max_block_size_is_rejected() {
        let config = DurabilityConfig {
            max_block_size: 0,
            ..DurabilityConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn minimum_decode_symbols_saturates_on_overflow() {
        let config = DurabilityConfig::default();
        let result = config.minimum_decode_symbols(u32::MAX);
        // Should saturate to u32::MAX, not wrap around.
        assert_eq!(result, u32::MAX);
    }

    #[test]
    fn repair_overhead_exactly_one_yields_zero_percent() {
        let config = DurabilityConfig {
            repair_overhead: 1.0,
            ..DurabilityConfig::default()
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.repair_overhead_percent(), 0);
    }
}
