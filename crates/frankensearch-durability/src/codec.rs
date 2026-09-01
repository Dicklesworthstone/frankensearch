use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use frankensearch_core::{SearchError, SearchResult};
use fsqlite_core::raptorq_integration::{
    CodecDecodeResult, CodecEncodeResult, DecodeFailureReason, SymbolCodec,
};
use fsqlite_types::cx::Cx;
use raptorq::{Decoder, Encoder, EncodingPacket, ObjectTransmissionInformation, PayloadId};
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
/// This codec is backed by the `raptorq` crate (RFC 6330 `RaptorQ` fountain
/// codes). It keeps compatibility with the `SymbolCodec` interface while
/// providing real erasure recovery: any `k_source` of the generated source+
/// repair symbols can reconstruct the original payload.
#[derive(Debug, Clone, Default)]
pub struct DefaultSymbolCodec;

/// Maximum source symbols per `RaptorQ` source block (K'_max from RFC 6330).
const MAX_SOURCE_SYMBOLS_PER_BLOCK: u32 = 56403;

impl DefaultSymbolCodec {
    fn object_transmission_info(
        transfer_length: u64,
        symbol_size: u16,
    ) -> ObjectTransmissionInformation {
        // Single source block, single sub-block, alignment of 1 byte.
        ObjectTransmissionInformation::new(transfer_length, symbol_size, 1, 1, 1)
    }
}

impl SymbolCodec for DefaultSymbolCodec {
    fn encode(
        &self,
        _cx: &Cx,
        source_data: &[u8],
        symbol_size: u32,
        repair_overhead: f64,
    ) -> fsqlite_error::Result<CodecEncodeResult> {
        if symbol_size == 0 {
            return Err(fsqlite_error::FrankenError::OutOfRange {
                what: "symbol_size".to_owned(),
                value: "0".to_owned(),
            });
        }
        let symbol_size_u16 =
            u16::try_from(symbol_size).map_err(|_| fsqlite_error::FrankenError::OutOfRange {
                what: "symbol_size as u16".to_owned(),
                value: symbol_size.to_string(),
            })?;

        let k_source = source_data.len().div_ceil(symbol_size as usize).max(1);
        let k_source_u32 =
            u32::try_from(k_source).map_err(|_| fsqlite_error::FrankenError::OutOfRange {
                what: "k_source as u32".to_owned(),
                value: k_source.to_string(),
            })?;
        if k_source_u32 > MAX_SOURCE_SYMBOLS_PER_BLOCK {
            return Err(fsqlite_error::FrankenError::OutOfRange {
                what: "k_source (exceeds single RaptorQ source block limit)".to_owned(),
                value: k_source_u32.to_string(),
            });
        }

        // Empty payloads are represented by a single empty source symbol so
        // downstream hashes/lengths round-trip without invoking the encoder
        // on a zero-length object.
        if source_data.is_empty() {
            return Ok(CodecEncodeResult {
                source_symbols: vec![(0, Vec::new())],
                repair_symbols: Vec::new(),
                k_source: 1,
            });
        }

        // Encode over the symbol-aligned length; the final short symbol is
        // zero-padded by the encoder. The caller truncates decoded output back
        // to the original `source_data.len()` using the stored source hash.
        let transfer_length = u64::try_from(k_source.saturating_mul(symbol_size as usize))
            .map_err(|_| fsqlite_error::FrankenError::OutOfRange {
                what: "transfer_length".to_owned(),
                value: format!("{}*{}", k_source, symbol_size),
            })?;
        let config = Self::object_transmission_info(transfer_length, symbol_size_u16);
        let encoder = Encoder::new(source_data, config);

        let requested_repair = if repair_overhead.is_finite() && repair_overhead > 0.0 {
            let requested = (f64::from(k_source_u32) * repair_overhead).ceil();
            format!("{requested:.0}").parse::<u32>().map_err(|_| {
                fsqlite_error::FrankenError::OutOfRange {
                    what: "requested_repair as u32".to_owned(),
                    value: requested.to_string(),
                }
            })?
        } else {
            0
        };

        let packets = encoder.get_encoded_packets(requested_repair);
        let mut source_symbols = Vec::new();
        let mut repair_symbols = Vec::new();
        for packet in packets {
            let esi = packet.payload_id().encoding_symbol_id();
            let data = packet.data().to_vec();
            if esi < k_source_u32 {
                source_symbols.push((esi, data));
            } else {
                repair_symbols.push((esi, data));
            }
        }

        // Deterministic order makes tests and repair-symbol comparison stable.
        source_symbols.sort_by_key(|(esi, _)| *esi);
        repair_symbols.sort_by_key(|(esi, _)| *esi);

        Ok(CodecEncodeResult {
            source_symbols,
            repair_symbols,
            k_source: k_source_u32,
        })
    }

    fn decode(
        &self,
        _cx: &Cx,
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

        if symbol_size == 0 {
            return Ok(CodecDecodeResult::Failure {
                reason: DecodeFailureReason::SymbolSizeMismatch,
                symbols_received: 0,
                k_required: k_source,
            });
        }
        let symbol_size_u16 =
            u16::try_from(symbol_size).map_err(|_| fsqlite_error::FrankenError::OutOfRange {
                what: "symbol_size as u16".to_owned(),
                value: symbol_size.to_string(),
            })?;
        let symbol_size_usize = symbol_size as usize;

        // Empty-payload short path: the single empty source symbol is the payload.
        if symbols.len() == 1 && symbols[0].1.is_empty() {
            return Ok(CodecDecodeResult::Success {
                data: Vec::new(),
                symbols_used: 1,
                peeled_count: 0,
                inactivated_count: 0,
            });
        }

        let transfer_length = u64::from(k_source) * u64::from(symbol_size);
        let config = Self::object_transmission_info(transfer_length, symbol_size_u16);
        let mut decoder = Decoder::new(config);

        let mut recovered: Option<Vec<u8>> = None;
        for (esi, payload) in symbols {
            if payload.len() != symbol_size_usize {
                return Ok(CodecDecodeResult::Failure {
                    reason: DecodeFailureReason::SymbolSizeMismatch,
                    symbols_received: u32::try_from(symbols.len()).unwrap_or(u32::MAX),
                    k_required: k_source,
                });
            }

            let payload_id = PayloadId::new(0, *esi);
            let packet = EncodingPacket::new(payload_id, payload.clone());
            if let Some(result) = decoder.decode(packet) {
                recovered = Some(result);
                // Continue feeding the remaining symbols so the decoder is left
                // in a clean terminal state, but the result is already fixed.
            }
        }

        recovered.map_or_else(
            || {
                Ok(CodecDecodeResult::Failure {
                    reason: DecodeFailureReason::InsufficientSymbols,
                    symbols_received: u32::try_from(symbols.len()).unwrap_or(u32::MAX),
                    k_required: k_source,
                })
            },
            |data| {
                Ok(CodecDecodeResult::Success {
                    data,
                    symbols_used: k_source,
                    peeled_count: 0,
                    inactivated_count: 0,
                })
            },
        )
    }
}

/// Persistable repair symbols plus reproducibility metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairData {
    pub repair_symbols: Vec<(u32, Vec<u8>)>,
    pub k_source: u32,
    pub symbol_size: u32,
    /// Exact byte length of the original source before symbol padding.
    pub source_len: u64,
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
        let cx = Cx::new();
        let mut result = self
            .codec
            .encode(
                &cx,
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

        let cx = Cx::new();
        let outcome = self
            .codec
            .decode(&cx, symbols, k_source, symbol_size)
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
            source_len: encoded.source_len,
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
        let source_len = Self::validate_repair_data(repair_data)?;

        if source_data.len() == source_len
            && xxh3_64(source_data).to_le_bytes() == repair_data.source_hash
        {
            return Ok(VerifyResult::Intact);
        }

        let regenerated = self.compute_repair_symbols(source_data)?;
        let corrupted_symbols =
            count_corrupted_symbols(&regenerated.repair_symbols, &repair_data.repair_symbols);

        // A codec-level success is not sufficient proof of repairability: a
        // decoder can legally produce bytes from corrupt source symbols. The
        // stored whole-payload witness is the acceptance boundary.
        let repairable = self.repair(source_data, repair_data).is_ok();

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
        let source_len = Self::validate_repair_data(repair_data)?;

        if corrupted_data.len() == source_len
            && xxh3_64(corrupted_data).to_le_bytes() == repair_data.source_hash
        {
            return Ok(corrupted_data.to_vec());
        }

        // Try the stored repair symbols alone first. Source symbols have no
        // per-symbol checksums, so feeding same-length corrupt input into an
        // erasure decoder can poison an otherwise recoverable solve. The
        // default durability configuration retains at least one complete set
        // of repair symbols for this reason.
        if let DecodedPayload::Success { data, .. } = self.decode_for_symbol_size(
            &repair_data.repair_symbols,
            repair_data.k_source,
            repair_data.symbol_size,
        )? && let Some(verified) =
            Self::normalize_and_verify_repair(data, source_len, repair_data.source_hash)
        {
            return Ok(verified);
        }

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
            DecodedPayload::Success { data, .. } => {
                Self::normalize_and_verify_repair(data, source_len, repair_data.source_hash)
                    .ok_or_else(|| SearchError::SubsystemError {
                        subsystem: "durability",
                        source: Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "decoded repair payload failed the stored length/hash witness",
                        )),
                    })
            }
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

    fn normalize_and_verify_repair(
        mut data: Vec<u8>,
        source_len: usize,
        source_hash: [u8; 8],
    ) -> Option<Vec<u8>> {
        if data.len() < source_len {
            return None;
        }
        data.truncate(source_len);
        (xxh3_64(&data).to_le_bytes() == source_hash).then_some(data)
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

    fn validate_repair_data(repair_data: &RepairData) -> SearchResult<usize> {
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

        let source_len =
            usize::try_from(repair_data.source_len).map_err(|_| SearchError::InvalidConfig {
                field: "repair_data.source_len".to_owned(),
                value: repair_data.source_len.to_string(),
                reason: "cannot convert source_len to usize".to_owned(),
            })?;
        let k_source =
            usize::try_from(repair_data.k_source).map_err(|_| SearchError::InvalidConfig {
                field: "repair_data.k_source".to_owned(),
                value: repair_data.k_source.to_string(),
                reason: "cannot convert k_source to usize".to_owned(),
            })?;
        let symbol_size =
            usize::try_from(repair_data.symbol_size).map_err(|_| SearchError::InvalidConfig {
                field: "repair_data.symbol_size".to_owned(),
                value: repair_data.symbol_size.to_string(),
                reason: "cannot convert symbol_size to usize".to_owned(),
            })?;
        let symbol_capacity =
            k_source
                .checked_mul(symbol_size)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "repair_data.source_len".to_owned(),
                    value: repair_data.source_len.to_string(),
                    reason: "source symbol capacity overflows usize".to_owned(),
                })?;
        if source_len > symbol_capacity {
            return Err(SearchError::InvalidConfig {
                field: "repair_data.source_len".to_owned(),
                value: repair_data.source_len.to_string(),
                reason: format!(
                    "exceeds k_source * symbol_size capacity ({symbol_capacity} bytes)"
                ),
            });
        }

        Ok(source_len)
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
        CodecFacade, Cx, DecodeFailureClass, DecodedPayload, DefaultSymbolCodec, RepairData,
        VerifyResult, classify_decode_failure,
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
            _cx: &Cx,
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
            _cx: &Cx,
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
            _cx: &Cx,
            _source_data: &[u8],
            _symbol_size: u32,
            _repair_overhead: f64,
        ) -> fsqlite_error::Result<CodecEncodeResult> {
            Err(FrankenError::Internal("encode boom".to_owned()))
        }

        fn decode(
            &self,
            _cx: &Cx,
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
    fn default_codec_repairs_same_length_corruption_exactly() {
        let facade = CodecFacade::new(
            Arc::new(DefaultSymbolCodec),
            DurabilityConfig {
                symbol_size: 256,
                ..DurabilityConfig::default()
            },
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("facade");
        let source = (0_u16..700)
            .map(|value| value.to_le_bytes()[0])
            .collect::<Vec<_>>();
        let repair_data = facade
            .compute_repair_symbols(&source)
            .expect("compute repair data");
        let mut corrupted = source.clone();
        corrupted[0] ^= 0xFF;
        corrupted[300] ^= 0xFF;
        corrupted[699] ^= 0xFF;

        let repaired = facade
            .repair(&corrupted, &repair_data)
            .expect("repair from the stored symbols");

        assert_eq!(repaired, source);
    }

    #[test]
    fn default_codec_repair_symbols_are_independent_of_source_symbols() {
        let facade = CodecFacade::new(
            Arc::new(DefaultSymbolCodec),
            DurabilityConfig {
                symbol_size: 256,
                ..DurabilityConfig::default()
            },
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("facade");
        let source = (0_u16..700)
            .map(|value| value.to_le_bytes()[0])
            .collect::<Vec<_>>();

        let encoded = facade.encode(&source).expect("encode source");
        assert!(
            !encoded.repair_symbols.is_empty(),
            "repair budget must generate at least one repair symbol"
        );

        let source_bytes: std::collections::HashSet<_> = encoded
            .source_symbols
            .iter()
            .map(|(_, data)| data.as_slice())
            .collect();
        let collision = encoded
            .repair_symbols
            .iter()
            .any(|(_, data)| source_bytes.contains(data.as_slice()));
        assert!(
            !collision,
            "RaptorQ repair symbols must not be byte-identical copies of source symbols"
        );
    }

    #[test]
    fn default_codec_recovers_from_lost_source_symbols_using_repairs() {
        let facade = CodecFacade::new(
            Arc::new(DefaultSymbolCodec),
            DurabilityConfig {
                symbol_size: 256,
                ..DurabilityConfig::default()
            },
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("facade");
        let source = (0_u16..700)
            .map(|value| value.to_le_bytes()[0])
            .collect::<Vec<_>>();

        let repair_data = facade
            .compute_repair_symbols(&source)
            .expect("compute repair data");
        assert!(
            repair_data.repair_symbols.len() >= repair_data.k_source as usize,
            "need at least k_source repair symbols to recover without any source symbols"
        );

        // Replace the entire source payload with zeros. The repair path first
        // attempts to reconstruct from repair symbols alone; if that succeeds,
        // the zeroed source symbols are treated as erasures.
        let corrupted = vec![0_u8; source.len()];
        let repaired = facade
            .repair(&corrupted, &repair_data)
            .expect("repair from repairs only");

        assert_eq!(repaired, source);
    }

    #[test]
    fn default_codec_recovers_from_partial_source_corruption() {
        let facade = CodecFacade::new(
            Arc::new(DefaultSymbolCodec),
            DurabilityConfig {
                symbol_size: 256,
                ..DurabilityConfig::default()
            },
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("facade");
        let source = (0_u16..700)
            .map(|value| value.to_le_bytes()[0])
            .collect::<Vec<_>>();

        let repair_data = facade
            .compute_repair_symbols(&source)
            .expect("compute repair data");

        // Corrupt roughly half the payload. The decoder combines the surviving
        // source symbols with repair symbols to recover the original.
        let mut corrupted = source.clone();
        for byte in corrupted.iter_mut().take(350) {
            *byte ^= 0xFF;
        }

        let repaired = facade
            .repair(&corrupted, &repair_data)
            .expect("repair from mixed symbols");

        assert_eq!(repaired, source);
    }

    #[test]
    fn default_codec_rejects_output_when_repair_symbols_are_poisoned() {
        let facade = CodecFacade::new(
            Arc::new(DefaultSymbolCodec),
            DurabilityConfig {
                symbol_size: 256,
                ..DurabilityConfig::default()
            },
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("facade");
        let source = (0_u16..600)
            .map(|value| value.to_le_bytes()[0])
            .collect::<Vec<_>>();
        let mut repair_data = facade
            .compute_repair_symbols(&source)
            .expect("compute repair data");
        for (_, symbol) in &mut repair_data.repair_symbols {
            symbol[0] ^= 0xFF;
        }
        let mut corrupted = source.clone();
        corrupted[137] ^= 0xFF;

        assert!(matches!(
            facade.verify(&corrupted, &repair_data).expect("verify"),
            VerifyResult::Corrupted {
                repairable: false,
                ..
            }
        ));
        assert!(
            facade.repair(&corrupted, &repair_data).is_err(),
            "a codec success with the wrong bytes must fail the stored witness"
        );
    }

    #[test]
    fn verify_rejects_matching_hash_with_wrong_source_length() {
        let facade = CodecFacade::new(
            Arc::new(DefaultSymbolCodec),
            DurabilityConfig {
                symbol_size: 256,
                ..DurabilityConfig::default()
            },
            Arc::new(DurabilityMetrics::default()),
        )
        .expect("facade");
        let source = vec![0xA5; 600];
        let mut repair_data = facade
            .compute_repair_symbols(&source)
            .expect("compute repair data");
        repair_data.source_len += 1;

        assert!(matches!(
            facade.verify(&source, &repair_data).expect("verify"),
            VerifyResult::Corrupted {
                repairable: false,
                ..
            }
        ));
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
            source_len: 3,
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
            source_len: 0,
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
            source_len: 0,
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
