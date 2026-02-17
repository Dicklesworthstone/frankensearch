use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use frankensearch_core::{SearchError, SearchResult};
use fsqlite_core::raptorq_integration::{
    CodecDecodeResult, CodecEncodeResult, DecodeFailureReason, SymbolCodec,
};
use tracing::{debug, warn};
use xxhash_rust::xxh3::xxh3_64;

use crate::config::DurabilityConfig;
use crate::metrics::{DecodeOutcomeClass, DurabilityMetrics};

/// Encoded source+repair symbols and metadata.
#[derive(Debug, Clone)]
pub struct EncodedPayload {
    pub source_symbols: Vec<(u32, Vec<u8>)>,
    pub repair_symbols: Vec<(u32, Vec<u8>)>,
    pub k_source: u32,
    pub source_len: u64,
    pub source_crc32: u32,
    pub symbol_size: u32,
}

/// Alias matching the bead wording.
pub type EncodedData = EncodedPayload;

/// Alias matching the bead wording.
pub type RepairCodec = CodecFacade;

/// Alias matching the bead wording.
pub type RepairCodecConfig = DurabilityConfig;

/// Default in-process symbol codec used by frankensearch durability wiring.
///
/// This codec keeps compatibility with the `SymbolCodec` interface while
/// avoiding any external runtime initialization requirements.
#[derive(Debug, Clone, Default)]
pub struct DefaultSymbolCodec;

impl SymbolCodec for DefaultSymbolCodec {
    fn encode(
        &self,
        source_data: &[u8],
        symbol_size: u32,
        repair_overhead: f64,
    ) -> fsqlite_error::Result<CodecEncodeResult> {
        let symbol_size =
            usize::try_from(symbol_size).map_err(|_| fsqlite_error::FrankenError::OutOfRange {
                what: "symbol_size as usize".to_owned(),
                value: symbol_size.to_string(),
            })?;
        if symbol_size == 0 {
            return Err(fsqlite_error::FrankenError::OutOfRange {
                what: "symbol_size".to_owned(),
                value: "0".to_owned(),
            });
        }

        let k_source = source_data.len().div_ceil(symbol_size).max(1);
        let k_source_u32 =
            u32::try_from(k_source).map_err(|_| fsqlite_error::FrankenError::OutOfRange {
                what: "k_source as u32".to_owned(),
                value: k_source.to_string(),
            })?;

        let mut source_symbols = Vec::with_capacity(k_source);
        for symbol_idx in 0..k_source {
            let start = symbol_idx.saturating_mul(symbol_size);
            let end = start.saturating_add(symbol_size).min(source_data.len());
            let mut symbol = vec![0_u8; symbol_size];
            if start < end {
                symbol[..end - start].copy_from_slice(&source_data[start..end]);
            }
            source_symbols.push((
                u32::try_from(symbol_idx).map_err(|_| fsqlite_error::FrankenError::OutOfRange {
                    what: "source symbol index as u32".to_owned(),
                    value: symbol_idx.to_string(),
                })?,
                symbol,
            ));
        }

        let requested_repair = if repair_overhead.is_finite() && repair_overhead > 0.0 {
            let requested = (f64::from(k_source_u32) * repair_overhead).ceil();
            format!("{requested:.0}").parse::<usize>().map_err(|_| {
                fsqlite_error::FrankenError::OutOfRange {
                    what: "requested_repair as usize".to_owned(),
                    value: requested.to_string(),
                }
            })?
        } else {
            0
        };

        let mut repair_symbols = Vec::with_capacity(requested_repair);
        for repair_idx in 0..requested_repair {
            let source_idx = repair_idx % k_source;
            let esi = k_source_u32.saturating_add(u32::try_from(repair_idx).map_err(|_| {
                fsqlite_error::FrankenError::OutOfRange {
                    what: "repair symbol index as u32".to_owned(),
                    value: repair_idx.to_string(),
                }
            })?);
            repair_symbols.push((esi, source_symbols[source_idx].1.clone()));
        }

        Ok(CodecEncodeResult {
            source_symbols,
            repair_symbols,
            k_source: k_source_u32,
        })
    }

    fn decode(
        &self,
        symbols: &[(u32, Vec<u8>)],
        k_source: u32,
        symbol_size: u32,
    ) -> fsqlite_error::Result<CodecDecodeResult> {
        if k_source == 0 {
            return Ok(CodecDecodeResult::Failure {
                reason: DecodeFailureReason::InsufficientSymbols,
                symbols_received: 0,
                k_required: 0,
            });
        }

        let symbol_size =
            usize::try_from(symbol_size).map_err(|_| fsqlite_error::FrankenError::OutOfRange {
                what: "symbol_size as usize".to_owned(),
                value: symbol_size.to_string(),
            })?;
        if symbol_size == 0 {
            return Ok(CodecDecodeResult::Failure {
                reason: DecodeFailureReason::SymbolSizeMismatch,
                symbols_received: 0,
                k_required: k_source,
            });
        }

        let k_source_usize =
            usize::try_from(k_source).map_err(|_| fsqlite_error::FrankenError::OutOfRange {
                what: "k_source as usize".to_owned(),
                value: k_source.to_string(),
            })?;
        let mut slots = vec![None::<Vec<u8>>; k_source_usize];

        for (esi, payload) in symbols {
            if payload.len() != symbol_size {
                return Ok(CodecDecodeResult::Failure {
                    reason: DecodeFailureReason::SymbolSizeMismatch,
                    symbols_received: u32::try_from(symbols.len()).unwrap_or(u32::MAX),
                    k_required: k_source,
                });
            }

            let target = if *esi < k_source {
                usize::try_from(*esi).ok()
            } else {
                let offset = esi.saturating_sub(k_source);
                usize::try_from(offset).ok().map(|idx| idx % k_source_usize)
            };

            if let Some(target) = target
                && slots[target].is_none()
            {
                slots[target] = Some(payload.clone());
            }
        }

        let available = slots.iter().filter(|slot| slot.is_some()).count();
        if available < k_source_usize {
            return Ok(CodecDecodeResult::Failure {
                reason: DecodeFailureReason::InsufficientSymbols,
                symbols_received: u32::try_from(available).unwrap_or(u32::MAX),
                k_required: k_source,
            });
        }

        let mut data = Vec::with_capacity(k_source_usize.saturating_mul(symbol_size));
        for symbol in slots.into_iter().flatten() {
            data.extend(symbol);
        }

        Ok(CodecDecodeResult::Success {
            data,
            symbols_used: u32::try_from(available).unwrap_or(u32::MAX),
            peeled_count: 0,
            inactivated_count: 0,
        })
    }
}

/// Persistable repair symbols plus reproducibility metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairData {
    pub repair_symbols: Vec<(u32, Vec<u8>)>,
    pub k_source: u32,
    pub symbol_size: u32,
    /// `xxh3_64` hash bytes of the original source payload.
    pub source_hash: [u8; 8],
}

/// Verification result for a source payload against repair data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyResult {
    Intact,
    Corrupted {
        corrupted_symbols: usize,
        repairable: bool,
    },
}

/// Recoverability classification for decode failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeFailureClass {
    Recoverable,
    Unrecoverable,
}

impl From<DecodeFailureClass> for DecodeOutcomeClass {
    fn from(value: DecodeFailureClass) -> Self {
        match value {
            DecodeFailureClass::Recoverable => Self::Recoverable,
            DecodeFailureClass::Unrecoverable => Self::Unrecoverable,
        }
    }
}

/// Decoding outcome returned by the codec facade.
#[derive(Debug, Clone)]
pub enum DecodedPayload {
    Success {
        data: Vec<u8>,
        symbols_used: u32,
        peeled_count: u32,
        inactivated_count: u32,
    },
    Failure {
        class: DecodeFailureClass,
        reason: DecodeFailureReason,
        symbols_received: u32,
        k_required: u32,
    },
}

/// Thin wrapper around a [`SymbolCodec`] with frankensearch-friendly errors
/// and durability metrics hooks.
#[derive(Clone)]
pub struct CodecFacade {
    codec: Arc<dyn SymbolCodec>,
    config: DurabilityConfig,
    metrics: Arc<DurabilityMetrics>,
}

impl fmt::Debug for CodecFacade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CodecFacade")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl CodecFacade {
    pub fn new(
        codec: Arc<dyn SymbolCodec>,
        config: DurabilityConfig,
        metrics: Arc<DurabilityMetrics>,
    ) -> SearchResult<Self> {
        config.validate()?;
        Ok(Self {
            codec,
            config,
            metrics,
        })
    }

    pub fn encode(&self, source_data: &[u8]) -> SearchResult<EncodedPayload> {
        let t0 = Instant::now();
        let mut result = self
            .codec
            .encode(
                source_data,
                self.config.symbol_size,
                self.config.repair_overhead,
            )
            .map_err(map_codec_error)?;

        let expected_repair = self.config.expected_repair_symbols(result.k_source);
        let max_repair_usize =
            usize::try_from(self.config.max_repair_symbols).unwrap_or(usize::MAX);
        if result.repair_symbols.len() > max_repair_usize {
            warn!(
                generated = result.repair_symbols.len(),
                max_repair_symbols = self.config.max_repair_symbols,
                expected_budget = expected_repair,
                "truncating repair symbols to max_repair_symbols guardrail"
            );
            result.repair_symbols.truncate(max_repair_usize);
        } else if result.repair_symbols.len()
            > usize::try_from(expected_repair).unwrap_or(usize::MAX)
        {
            debug!(
                generated = result.repair_symbols.len(),
                expected_budget = expected_repair,
                max_repair_symbols = self.config.max_repair_symbols,
                "codec generated more repair symbols than expected budget"
            );
        } else if result.repair_symbols.len()
            < usize::try_from(expected_repair).unwrap_or(usize::MAX)
        {
            warn!(
                generated = result.repair_symbols.len(),
                expected = expected_repair,
                "codec produced fewer repair symbols than configured target"
            );
        }

        let source_len = saturating_u64(source_data.len());
        let source_crc32 = crc32fast::hash(source_data);
        let latency_us = saturating_u64_from_u128(t0.elapsed().as_micros());
        self.metrics.record_encode(
            source_len,
            saturating_u64(result.source_symbols.len()),
            saturating_u64(result.repair_symbols.len()),
            latency_us,
        );

        debug!(
            source_len,
            source_symbols = result.source_symbols.len(),
            repair_symbols = result.repair_symbols.len(),
            expected_repair,
            symbol_size = self.config.symbol_size,
            latency_us,
            "durability encode complete"
        );

        Ok(EncodedPayload {
            source_symbols: result.source_symbols,
            repair_symbols: result.repair_symbols,
            k_source: result.k_source,
            source_len,
            source_crc32,
            symbol_size: self.config.symbol_size,
        })
    }

    pub fn decode(
        &self,
        symbols: &[(u32, Vec<u8>)],
        k_source: u32,
    ) -> SearchResult<DecodedPayload> {
        self.decode_for_symbol_size(symbols, k_source, self.config.symbol_size)
    }

    pub(crate) fn decode_for_symbol_size(
        &self,
        symbols: &[(u32, Vec<u8>)],
        k_source: u32,
        symbol_size: u32,
    ) -> SearchResult<DecodedPayload> {
        let t0 = Instant::now();
        if k_source == 0 {
            return Err(SearchError::InvalidConfig {
                field: "k_source".to_owned(),
                value: "0".to_owned(),
                reason: "must be greater than zero".to_owned(),
            });
        }
        if symbol_size == 0 {
            return Err(SearchError::InvalidConfig {
                field: "symbol_size".to_owned(),
                value: "0".to_owned(),
                reason: "must be greater than zero".to_owned(),
            });
        }

        let symbol_size_usize =
            usize::try_from(symbol_size).map_err(|_| SearchError::InvalidConfig {
                field: "symbol_size".to_owned(),
                value: symbol_size.to_string(),
                reason: "cannot convert symbol_size to usize".to_owned(),
            })?;

        let symbols_received = saturating_u32(symbols.len());
        if symbols_received < k_source {
            return Ok(self.decode_failure(
                t0,
                DecodeFailureReason::InsufficientSymbols,
                symbols_received,
                k_source,
                "fewer symbols than source symbol count",
            ));
        }

        if symbols
            .iter()
            .any(|(_, data)| data.len() != symbol_size_usize)
        {
            return Ok(self.decode_failure(
                t0,
                DecodeFailureReason::SymbolSizeMismatch,
                symbols_received,
                k_source,
                "symbol payload length does not match configured symbol_size",
            ));
        }

        let outcome = self
            .codec
            .decode(symbols, k_source, symbol_size)
            .map_err(map_codec_error)?;

        let payload = match outcome {
            CodecDecodeResult::Success {
                data,
                symbols_used,
                peeled_count,
                inactivated_count,
            } => {
                let latency_us = saturating_u64_from_u128(t0.elapsed().as_micros());
                self.metrics.record_decode_success(
                    saturating_u64(data.len()),
                    u64::from(symbols_used),
                    u64::from(symbols_received),
                    u64::from(k_source),
                    latency_us,
                );
                DecodedPayload::Success {
                    data,
                    symbols_used,
                    peeled_count,
                    inactivated_count,
                }
            }
            CodecDecodeResult::Failure {
                reason,
                symbols_received,
                k_required,
            } => self.decode_failure(
                t0,
                reason,
                symbols_received,
                k_required,
                "codec reported decode failure",
            ),
        };

        Ok(payload)
    }

    /// Compute deterministic repair symbols.
    ///
    /// Determinism contract: for identical `source_data` + identical codec/config,
    /// this returns byte-identical repair symbols and hash metadata.
    pub fn compute_repair_symbols(&self, source_data: &[u8]) -> SearchResult<RepairData> {
        let encoded = self.encode(source_data)?;
        Ok(RepairData {
            repair_symbols: encoded.repair_symbols,
            k_source: encoded.k_source,
            symbol_size: encoded.symbol_size,
            source_hash: xxh3_64(source_data).to_le_bytes(),
        })
    }

    /// Verify whether `source_data` matches `repair_data`.
    ///
    /// Uses an xxh3 fast path first, then falls back to deterministic repair-symbol
    /// comparison and decode viability probing when the hash mismatches.
    pub fn verify(
        &self,
        source_data: &[u8],
        repair_data: &RepairData,
    ) -> SearchResult<VerifyResult> {
        Self::validate_repair_data(repair_data)?;

        if xxh3_64(source_data).to_le_bytes() == repair_data.source_hash {
            return Ok(VerifyResult::Intact);
        }

        let regenerated = self.compute_repair_symbols(source_data)?;
        let corrupted_symbols =
            count_corrupted_symbols(&regenerated.repair_symbols, &repair_data.repair_symbols);

        let mut symbols =
            source_symbols_from_bytes(source_data, repair_data.symbol_size, repair_data.k_source)?;
        symbols.extend(repair_data.repair_symbols.clone());

        let repairable = matches!(
            self.decode_for_symbol_size(&symbols, repair_data.k_source, repair_data.symbol_size)?,
            DecodedPayload::Success { .. }
        );

        debug!(
            corrupted_symbols,
            repairable,
            k_source = repair_data.k_source,
            symbol_size = repair_data.symbol_size,
            "durability verify detected corruption"
        );

        Ok(VerifyResult::Corrupted {
            corrupted_symbols,
            repairable,
        })
    }

    /// Attempt to reconstruct original payload bytes from corrupted data + repair symbols.
    pub fn repair(&self, corrupted_data: &[u8], repair_data: &RepairData) -> SearchResult<Vec<u8>> {
        Self::validate_repair_data(repair_data)?;

        let mut symbols = source_symbols_from_bytes(
            corrupted_data,
            repair_data.symbol_size,
            repair_data.k_source,
        )?;
        symbols.extend(repair_data.repair_symbols.clone());

        match self.decode_for_symbol_size(
            &symbols,
            repair_data.k_source,
            repair_data.symbol_size,
        )? {
            DecodedPayload::Success { data, .. } => Ok(data),
            DecodedPayload::Failure {
                class,
                reason,
                symbols_received,
                k_required,
            } => Err(decode_failure_error(
                class,
                reason,
                symbols_received,
                k_required,
            )),
        }
    }

    pub fn config(&self) -> &DurabilityConfig {
        &self.config
    }

    pub fn metrics(&self) -> &Arc<DurabilityMetrics> {
        &self.metrics
    }

    fn decode_failure(
        &self,
        t0: Instant,
        reason: DecodeFailureReason,
        symbols_received: u32,
        k_required: u32,
        detail: &'static str,
    ) -> DecodedPayload {
        let class = classify_decode_failure(reason);
        let latency_us = saturating_u64_from_u128(t0.elapsed().as_micros());
        self.metrics.record_decode_failure(
            class.into(),
            u64::from(symbols_received),
            u64::from(k_required),
            latency_us,
        );

        warn!(
            ?reason,
            ?class,
            symbols_received,
            k_required,
            min_symbols_with_slack = self.config.minimum_decode_symbols(k_required),
            latency_us,
            detail,
            "durability decode failed"
        );

        DecodedPayload::Failure {
            class,
            reason,
            symbols_received,
            k_required,
        }
    }

    fn validate_repair_data(repair_data: &RepairData) -> SearchResult<()> {
        if repair_data.symbol_size == 0 {
            return Err(SearchError::InvalidConfig {
                field: "repair_data.symbol_size".to_owned(),
                value: "0".to_owned(),
                reason: "must be greater than zero".to_owned(),
            });
        }

        if repair_data.k_source == 0 {
            return Err(SearchError::InvalidConfig {
                field: "repair_data.k_source".to_owned(),
                value: "0".to_owned(),
                reason: "must be greater than zero".to_owned(),
            });
        }

        Ok(())
    }
}

#[must_use]
pub const fn classify_decode_failure(reason: DecodeFailureReason) -> DecodeFailureClass {
    match reason {
        DecodeFailureReason::SymbolSizeMismatch => DecodeFailureClass::Unrecoverable,
        DecodeFailureReason::InsufficientSymbols
        | DecodeFailureReason::SingularMatrix
        | DecodeFailureReason::Cancelled => DecodeFailureClass::Recoverable,
    }
}

fn map_codec_error<E>(error: E) -> SearchError
where
    E: std::error::Error + Send + Sync + 'static,
{
    SearchError::SubsystemError {
        subsystem: "durability",
        source: Box::new(error),
    }
}

fn source_symbols_from_bytes(
    bytes: &[u8],
    symbol_size: u32,
    k_source: u32,
) -> SearchResult<Vec<(u32, Vec<u8>)>> {
    if symbol_size == 0 {
        return Err(SearchError::InvalidConfig {
            field: "symbol_size".to_owned(),
            value: "0".to_owned(),
            reason: "must be greater than zero".to_owned(),
        });
    }

    let symbol_size_usize =
        usize::try_from(symbol_size).map_err(|_| SearchError::InvalidConfig {
            field: "symbol_size".to_owned(),
            value: symbol_size.to_string(),
            reason: "cannot convert symbol_size to usize".to_owned(),
        })?;

    let mut out = Vec::new();
    let max_symbols = bytes.len().div_ceil(symbol_size_usize);
    let max_symbols_u32 = u32::try_from(max_symbols).unwrap_or(u32::MAX);
    for esi in 0..k_source.min(max_symbols_u32) {
        let esi_usize = usize::try_from(esi).map_err(|_| SearchError::InvalidConfig {
            field: "esi".to_owned(),
            value: esi.to_string(),
            reason: "cannot convert symbol index to usize".to_owned(),
        })?;
        let start =
            esi_usize
                .checked_mul(symbol_size_usize)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "start_offset".to_owned(),
                    value: format!("{esi_usize}*{symbol_size_usize}"),
                    reason: "source symbol offset overflow".to_owned(),
                })?;
        if start >= bytes.len() {
            continue;
        }

        let end = start.saturating_add(symbol_size_usize).min(bytes.len());
        let mut symbol = bytes[start..end].to_vec();
        if symbol.len() < symbol_size_usize {
            symbol.resize(symbol_size_usize, 0);
        }
        out.push((esi, symbol));
    }

    Ok(out)
}

fn count_corrupted_symbols(
    expected_repair_symbols: &[(u32, Vec<u8>)],
    observed_repair_symbols: &[(u32, Vec<u8>)],
) -> usize {
    let mut expected_map: HashMap<u32, &[u8]> = HashMap::new();
    for (esi, data) in expected_repair_symbols {
        expected_map.insert(*esi, data);
    }

    let mut observed_map: HashMap<u32, &[u8]> = HashMap::new();
    for (esi, data) in observed_repair_symbols {
        observed_map.insert(*esi, data);
    }

    let mut corrupted = 0_usize;
    for (esi, expected_data) in &expected_map {
        match observed_map.get(esi) {
            Some(observed_data) if *observed_data == *expected_data => {}
            _ => {
                corrupted = corrupted.saturating_add(1);
            }
        }
    }
    for esi in observed_map.keys() {
        if !expected_map.contains_key(esi) {
            corrupted = corrupted.saturating_add(1);
        }
    }

    corrupted
}

fn decode_failure_error(
    class: DecodeFailureClass,
    reason: DecodeFailureReason,
    symbols_received: u32,
    k_required: u32,
) -> SearchError {
    let classification = match class {
        DecodeFailureClass::Recoverable => "recoverable",
        DecodeFailureClass::Unrecoverable => "unrecoverable",
    };
    SearchError::SubsystemError {
        subsystem: "durability",
        source: Box::new(std::io::Error::other(format!(
            "repair decode {classification} failure: reason={reason:?} symbols_received={symbols_received} k_required={k_required}"
        ))),
    }
}

fn saturating_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn saturating_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

fn saturating_u64_from_u128(value: u128) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use fsqlite_core::raptorq_integration::{CodecDecodeResult, CodecEncodeResult, SymbolCodec};
    use fsqlite_error::FrankenError;

    use super::{
        CodecFacade, DecodeFailureClass, DecodedPayload, RepairData, VerifyResult,
        classify_decode_failure,
    };
    use crate::config::DurabilityConfig;
    use crate::metrics::DurabilityMetrics;

    #[derive(Debug)]
    struct MockCodec {
        fail_decode_reason: Option<fsqlite_core::raptorq_integration::DecodeFailureReason>,
    }

    impl SymbolCodec for MockCodec {
        fn encode(
            &self,
            source_data: &[u8],
            symbol_size: u32,
            _repair_overhead: f64,
        ) -> fsqlite_error::Result<CodecEncodeResult> {
            let symbol_size_usize = usize::try_from(symbol_size).unwrap_or(1);
            let k_source_usize = source_data.len().div_ceil(symbol_size_usize).max(1);
            let k_source = u32::try_from(k_source_usize).unwrap_or(u32::MAX);

            let mut source_symbols = Vec::new();
            for esi in 0..k_source {
                let esi_usize = usize::try_from(esi).unwrap_or(0);
                let start = esi_usize.saturating_mul(symbol_size_usize);
                let end = start
                    .saturating_add(symbol_size_usize)
                    .min(source_data.len());
                let mut data = if start < source_data.len() {
                    source_data[start..end].to_vec()
                } else {
                    Vec::new()
                };
                if data.len() < symbol_size_usize {
                    data.resize(symbol_size_usize, 0);
                }
                source_symbols.push((esi, data));
            }

            let repair_symbol = source_symbols
                .first()
                .map_or_else(|| vec![0; symbol_size_usize], |(_, data)| data.clone());

            Ok(CodecEncodeResult {
                source_symbols,
                repair_symbols: vec![(1_000_000, repair_symbol)],
                k_source,
            })
        }

        fn decode(
            &self,
            symbols: &[(u32, Vec<u8>)],
            k_source: u32,
            symbol_size: u32,
        ) -> fsqlite_error::Result<CodecDecodeResult> {
            if let Some(reason) = self.fail_decode_reason {
                return Ok(CodecDecodeResult::Failure {
                    reason,
                    symbols_received: u32::try_from(symbols.len()).unwrap_or(u32::MAX),
                    k_required: k_source,
                });
            }

            let symbol_size_usize = usize::try_from(symbol_size).unwrap_or(usize::MAX);
            if symbols
                .iter()
                .any(|(_, data)| data.len() != symbol_size_usize)
            {
                return Ok(CodecDecodeResult::Failure {
                    reason:
                        fsqlite_core::raptorq_integration::DecodeFailureReason::SymbolSizeMismatch,
                    symbols_received: u32::try_from(symbols.len()).unwrap_or(u32::MAX),
                    k_required: k_source,
                });
            }

            if symbols.is_empty() {
                return Ok(CodecDecodeResult::Failure {
                    reason:
                        fsqlite_core::raptorq_integration::DecodeFailureReason::InsufficientSymbols,
                    symbols_received: 0,
                    k_required: k_source,
                });
            }

            Ok(CodecDecodeResult::Success {
                data: symbols[0].1.clone(),
                symbols_used: 1,
                peeled_count: 1,
                inactivated_count: 0,
            })
        }
    }

    #[derive(Debug)]
    struct ErrorCodec;

    impl SymbolCodec for ErrorCodec {
        fn encode(
            &self,
            _source_data: &[u8],
            _symbol_size: u32,
            _repair_overhead: f64,
        ) -> fsqlite_error::Result<CodecEncodeResult> {
            Err(FrankenError::Internal("encode boom".to_owned()))
        }

        fn decode(
            &self,
            _symbols: &[(u32, Vec<u8>)],
            _k_source: u32,
            _symbol_size: u32,
        ) -> fsqlite_error::Result<CodecDecodeResult> {
            Err(FrankenError::Internal("decode boom".to_owned()))
        }
    }

    #[test]
    fn encode_updates_metrics_and_returns_payload() {
        let metrics = Arc::new(DurabilityMetrics::default());
        let facade = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: None,
            }),
            DurabilityConfig::default(),
            Arc::clone(&metrics),
        )
        .expect("facade");

        let encoded = facade.encode(b"hello").expect("encode");
        assert_eq!(encoded.k_source, 1);
        assert_eq!(encoded.source_symbols.len(), 1);
        assert_eq!(encoded.repair_symbols.len(), 1);
        assert_eq!(encoded.source_len, 5);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.encode_ops, 1);
        assert_eq!(snapshot.encoded_bytes_total, 5);
        assert_eq!(snapshot.source_symbols_total, 1);
        assert_eq!(snapshot.repair_symbols_total, 1);
    }

    #[test]
    fn decode_success_updates_metrics() {
        let metrics = Arc::new(DurabilityMetrics::default());
        let facade = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: None,
            }),
            DurabilityConfig::default(),
            Arc::clone(&metrics),
        )
        .expect("facade");

        let symbol = vec![7_u8; 4096];
        let decoded = facade.decode(&[(0, symbol)], 1).expect("decode");
        assert!(matches!(decoded, DecodedPayload::Success { .. }));

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.decode_ops, 1);
        assert_eq!(snapshot.decode_failures, 0);
        assert_eq!(snapshot.decoded_bytes_total, 4096);
        assert_eq!(snapshot.decode_symbols_used_total, 1);
        assert_eq!(snapshot.decode_symbols_received_total, 1);
        assert_eq!(snapshot.decode_k_required_total, 1);
    }

    #[test]
    fn threshold_shortfall_returns_recoverable_failure() {
        let metrics = Arc::new(DurabilityMetrics::default());
        let facade = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: None,
            }),
            DurabilityConfig::default(),
            Arc::clone(&metrics),
        )
        .expect("facade");

        let symbol = vec![1_u8; 4096];
        let decoded = facade.decode(&[(0, symbol)], 2).expect("decode");
        assert!(matches!(
            decoded,
            DecodedPayload::Failure {
                class: DecodeFailureClass::Recoverable,
                reason: fsqlite_core::raptorq_integration::DecodeFailureReason::InsufficientSymbols,
                ..
            }
        ));

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.decode_ops, 1);
        assert_eq!(snapshot.decode_failures, 1);
        assert_eq!(snapshot.decode_failures_recoverable, 1);
    }

    #[test]
    fn malformed_symbols_are_unrecoverable() {
        let metrics = Arc::new(DurabilityMetrics::default());
        let facade = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: None,
            }),
            DurabilityConfig::default(),
            Arc::clone(&metrics),
        )
        .expect("facade");

        let decoded = facade.decode(&[(0, vec![1, 2, 3])], 1).expect("decode");
        assert!(matches!(
            decoded,
            DecodedPayload::Failure {
                class: DecodeFailureClass::Unrecoverable,
                reason: fsqlite_core::raptorq_integration::DecodeFailureReason::SymbolSizeMismatch,
                ..
            }
        ));

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.decode_failures, 1);
        assert_eq!(snapshot.decode_failures_unrecoverable, 1);
    }

    #[test]
    fn decode_rejects_zero_k_source() {
        let facade = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: None,
            }),
            DurabilityConfig::default(),
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("facade");

        let err = facade.decode(&[], 0).expect_err("must fail");
        assert!(matches!(
            err,
            frankensearch_core::SearchError::InvalidConfig { field, .. } if field == "k_source"
        ));
    }

    #[test]
    fn decode_rejects_zero_symbol_size() {
        let facade = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: None,
            }),
            DurabilityConfig::default(),
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("facade");

        let err = facade
            .decode_for_symbol_size(&[], 1, 0)
            .expect_err("must fail");
        assert!(matches!(
            err,
            frankensearch_core::SearchError::InvalidConfig { field, .. } if field == "symbol_size"
        ));
    }

    #[test]
    fn decode_failure_is_classified_as_recoverable() {
        let metrics = Arc::new(DurabilityMetrics::default());
        let facade = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: Some(
                    fsqlite_core::raptorq_integration::DecodeFailureReason::InsufficientSymbols,
                ),
            }),
            DurabilityConfig::default(),
            Arc::clone(&metrics),
        )
        .expect("facade");

        let symbol = vec![2_u8; 4096];
        let decoded = facade
            .decode(&[(0, symbol), (1, vec![3_u8; 4096])], 2)
            .expect("decode");
        assert!(matches!(
            decoded,
            DecodedPayload::Failure {
                class: DecodeFailureClass::Recoverable,
                ..
            }
        ));

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.decode_failures, 1);
        assert_eq!(snapshot.decode_failures_recoverable, 1);
    }

    #[test]
    fn compute_repair_symbols_is_deterministic() {
        let facade = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: None,
            }),
            DurabilityConfig::default(),
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("facade");

        let first = facade
            .compute_repair_symbols(b"deterministic payload")
            .expect("compute first");
        let second = facade
            .compute_repair_symbols(b"deterministic payload")
            .expect("compute second");

        assert_eq!(first, second);
        assert_eq!(
            first.source_hash,
            xxhash_rust::xxh3::xxh3_64(b"deterministic payload").to_le_bytes()
        );
    }

    #[test]
    fn verify_flags_corruption_and_reports_repairability() {
        let facade = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: None,
            }),
            DurabilityConfig::default(),
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("facade");

        let clean = b"verify me";
        let repair_data = facade
            .compute_repair_symbols(clean)
            .expect("compute repair data");

        let mut corrupted = clean.to_vec();
        corrupted[0] ^= 0xFF;

        let verify = facade.verify(&corrupted, &repair_data).expect("verify");
        assert!(matches!(verify, VerifyResult::Corrupted { .. }));

        match verify {
            VerifyResult::Corrupted {
                corrupted_symbols,
                repairable,
            } => {
                assert!(corrupted_symbols > 0);
                assert!(repairable);
            }
            VerifyResult::Intact => panic!("expected corrupted result"),
        }
    }

    #[test]
    fn repair_returns_error_for_decode_failures() {
        let facade = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: Some(
                    fsqlite_core::raptorq_integration::DecodeFailureReason::SymbolSizeMismatch,
                ),
            }),
            DurabilityConfig::default(),
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("facade");

        let repair_data = RepairData {
            repair_symbols: vec![(10, vec![7_u8; 4096])],
            k_source: 1,
            symbol_size: 4096,
            source_hash: [0_u8; 8],
        };

        let err = facade.repair(b"bad", &repair_data).expect_err("must fail");
        assert!(matches!(
            err,
            frankensearch_core::SearchError::SubsystemError {
                subsystem: "durability",
                ..
            }
        ));
    }

    #[test]
    fn repair_uses_repair_data_symbol_size_not_runtime_config() {
        let source = b"cross-config-symbol-size";

        let producer = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: None,
            }),
            DurabilityConfig {
                symbol_size: 256,
                ..DurabilityConfig::default()
            },
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("producer facade");
        let repair_data = producer
            .compute_repair_symbols(source)
            .expect("compute repair data");

        let consumer = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: None,
            }),
            DurabilityConfig {
                symbol_size: 4096,
                ..DurabilityConfig::default()
            },
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("consumer facade");

        let mut corrupted = source.to_vec();
        corrupted[0] ^= 0xFF;
        let repaired = consumer
            .repair(&corrupted, &repair_data)
            .expect("repair across differing runtime symbol_size");
        assert!(!repaired.is_empty());
    }

    #[test]
    fn repair_accepts_repair_data_above_runtime_generation_cap() {
        let source = b"repair-cap-agnostic";

        let producer = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: None,
            }),
            DurabilityConfig::default(),
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("producer facade");
        let mut repair_data = producer
            .compute_repair_symbols(source)
            .expect("compute repair data");
        repair_data
            .repair_symbols
            .push((2_000_000, vec![1_u8; 4096]));

        let consumer = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: None,
            }),
            DurabilityConfig {
                max_repair_symbols: 1,
                slack_decode: 1,
                ..DurabilityConfig::default()
            },
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("consumer facade");

        let repaired = consumer
            .repair(source, &repair_data)
            .expect("decode should not reject repair symbol count above runtime generation cap");
        assert!(!repaired.is_empty());
    }

    #[test]
    fn codec_errors_are_mapped_to_search_error() {
        let facade = CodecFacade::new(
            Arc::new(ErrorCodec),
            DurabilityConfig::default(),
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("facade");

        let error = facade.encode(b"boom").expect_err("must fail");
        match error {
            frankensearch_core::SearchError::SubsystemError { subsystem, .. } => {
                assert_eq!(subsystem, "durability");
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn failure_reason_classification_is_stable() {
        assert_eq!(
            classify_decode_failure(
                fsqlite_core::raptorq_integration::DecodeFailureReason::InsufficientSymbols
            ),
            DecodeFailureClass::Recoverable
        );
        assert_eq!(
            classify_decode_failure(
                fsqlite_core::raptorq_integration::DecodeFailureReason::SymbolSizeMismatch
            ),
            DecodeFailureClass::Unrecoverable
        );
    }

    #[test]
    fn encode_empty_input_produces_valid_payload() {
        let facade = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: None,
            }),
            DurabilityConfig::default(),
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("facade");

        let encoded = facade.encode(b"").expect("encode empty");
        assert_eq!(encoded.source_len, 0);
        // Even empty input should produce at least one source symbol (padding).
        assert!(encoded.k_source >= 1);
    }

    #[test]
    fn encode_small_input_below_symbol_size() {
        let facade = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: None,
            }),
            DurabilityConfig::default(),
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("facade");

        // Input smaller than default symbol size (4096).
        let encoded = facade.encode(b"tiny").expect("encode small");
        assert_eq!(encoded.source_len, 4);
        assert_eq!(encoded.k_source, 1);
        assert_eq!(encoded.source_symbols.len(), 1);
    }

    #[test]
    fn different_inputs_produce_different_hashes() {
        let facade = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: None,
            }),
            DurabilityConfig::default(),
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("facade");

        let repair_a = facade
            .compute_repair_symbols(b"payload A")
            .expect("compute A");
        let repair_b = facade
            .compute_repair_symbols(b"payload B")
            .expect("compute B");

        assert_ne!(
            repair_a.source_hash, repair_b.source_hash,
            "different payloads must produce different hashes"
        );
    }

    #[test]
    fn validate_repair_data_rejects_zero_k_source() {
        let facade = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: None,
            }),
            DurabilityConfig::default(),
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("facade");

        let repair_data = RepairData {
            repair_symbols: vec![(10, vec![7_u8; 4096])],
            k_source: 0,
            symbol_size: 4096,
            source_hash: [0_u8; 8],
        };

        let err = facade
            .verify(b"anything", &repair_data)
            .expect_err("must reject k_source=0");
        assert!(matches!(
            err,
            frankensearch_core::SearchError::InvalidConfig { field, .. } if field == "repair_data.k_source"
        ));
    }

    #[test]
    fn validate_repair_data_rejects_zero_symbol_size() {
        let facade = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: None,
            }),
            DurabilityConfig::default(),
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("facade");

        let repair_data = RepairData {
            repair_symbols: vec![(10, vec![7_u8; 4096])],
            k_source: 1,
            symbol_size: 0,
            source_hash: [0_u8; 8],
        };

        let err = facade
            .verify(b"anything", &repair_data)
            .expect_err("must reject symbol_size=0");
        assert!(matches!(
            err,
            frankensearch_core::SearchError::InvalidConfig { field, .. } if field == "repair_data.symbol_size"
        ));
    }

    #[test]
    fn codec_facade_rejects_invalid_config() {
        let result = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: None,
            }),
            DurabilityConfig {
                symbol_size: 1000, // not a power of two
                ..DurabilityConfig::default()
            },
            Arc::new(DurabilityMetrics::default()),
        );
        assert!(result.is_err());
    }

    #[test]
    fn singular_matrix_failure_is_recoverable() {
        assert_eq!(
            classify_decode_failure(
                fsqlite_core::raptorq_integration::DecodeFailureReason::SingularMatrix
            ),
            DecodeFailureClass::Recoverable
        );
    }

    #[test]
    fn cancelled_failure_is_recoverable() {
        assert_eq!(
            classify_decode_failure(
                fsqlite_core::raptorq_integration::DecodeFailureReason::Cancelled
            ),
            DecodeFailureClass::Recoverable
        );
    }

    #[test]
    fn verify_intact_data_returns_intact() {
        let facade = CodecFacade::new(
            Arc::new(MockCodec {
                fail_decode_reason: None,
            }),
            DurabilityConfig::default(),
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("facade");

        let data = b"verify intact";
        let repair_data = facade.compute_repair_symbols(data).expect("compute");
        let result = facade.verify(data, &repair_data).expect("verify");
        assert!(matches!(result, VerifyResult::Intact));
    }
}
