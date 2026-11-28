//! Explicit reads of one trusted native fixed-authority generation.
//!
//! This route never opens an ordinary fsfs catalog, follows FSFS-CURRENT, or
//! infers a producer from an artifact. The selection receipt is caller trust;
//! the admitted native snapshot owns retrieval and source hydration together.

use std::future::Future;
use std::io::{self, Write};
use std::sync::Arc;
use std::time::Instant;

use asupersync::Cx;
use frankensearch::native_ann::{
    NativeGenerationReceipt, NativeGenerationSnapshot, NativeGenerationTopology,
    NativePhaseCandidates, NativeSearchPhase,
};
use frankensearch_core::{Embedder, EmbedderTierTag, ScoredResult, SearchError, SearchResult};
use frankensearch_fusion::RrfConfig;
use serde::Serialize;

use super::{
    FsfsRuntime, SearchStreamWriter, iso_timestamp_now, pressure_timestamp_ms,
    retained_search_checkpoint,
};
use crate::adapters::format_emitter::{emit_envelope, emit_stream_frame, meta_for_format};
use crate::config::DegradationOverrideMode;
use crate::output_schema::{OutputEnvelope, output_error_from};
use crate::stream_protocol::{
    StreamEvent, StreamFrame, StreamProgressEvent, StreamResultEvent, StreamStartedEvent,
    terminal_event_completed, terminal_event_from_error,
};

const DEFAULT_IMAGE_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_SOURCE_BYTES: usize = 16 * 1024 * 1024;
const SNIPPET_CHARS: usize = 480;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuthorityMode {
    Lexical,
    Fast,
    Quality,
    Full,
    Partial,
}

impl AuthorityMode {
    fn select(receipt: &NativeGenerationReceipt, requested: Option<&str>) -> SearchResult<Self> {
        let mode = match requested {
            Some("lexical") => Self::Lexical,
            Some("fast") => Self::Fast,
            Some("quality") => Self::Quality,
            Some("full") => Self::Full,
            Some("partial") => Self::Partial,
            None => match receipt.topology() {
                NativeGenerationTopology::FastOnly
                | NativeGenerationTopology::PartialQuality { .. } => Self::Fast,
                NativeGenerationTopology::QualityOnly => Self::Quality,
                NativeGenerationTopology::FullProgressive => Self::Full,
            },
            Some(_) => {
                return Err(refused(
                    "mode",
                    "expected lexical|fast|quality|full|partial",
                ));
            }
        };
        let permitted = match receipt.topology() {
            NativeGenerationTopology::FastOnly => matches!(mode, Self::Lexical | Self::Fast),
            NativeGenerationTopology::QualityOnly => matches!(mode, Self::Lexical | Self::Quality),
            NativeGenerationTopology::FullProgressive => mode != Self::Partial,
            NativeGenerationTopology::PartialQuality { .. } => {
                matches!(mode, Self::Lexical | Self::Fast | Self::Partial)
            }
        };
        if !permitted {
            return Err(refused(
                "mode",
                "requested tier or coverage is unavailable; full and quality modes require complete quality coverage, and partial mode requires a partial-quality generation",
            ));
        }
        Ok(mode)
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Lexical => "lexical",
            Self::Fast => "fast",
            Self::Quality => "quality",
            Self::Full => "full",
            Self::Partial => "partial",
        }
    }

    const fn needs_fast(self) -> bool {
        matches!(self, Self::Fast | Self::Full | Self::Partial)
    }

    const fn needs_quality(self) -> bool {
        matches!(self, Self::Quality | Self::Full | Self::Partial)
    }
}

#[derive(Serialize)]
struct AuthoritySearchHit {
    #[serde(flatten)]
    result: ScoredResult,
    /// Content comes from the same retained Quill image as the ranked hit.
    snippet: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    /// A path is reported only when the retained source explicitly stored one.
    #[serde(skip_serializing_if = "Option::is_none")]
    source_path: Option<String>,
}

#[derive(Serialize)]
struct AuthorityPage {
    #[serde(flatten)]
    evidence: serde_json::Value,
    mode: &'static str,
    phase: &'static str,
    results: Vec<AuthoritySearchHit>,
}

fn refused(field: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("search.authority_{field}"),
        value: "native_receipt".to_owned(),
        reason: reason.to_owned(),
    }
}

impl FsfsRuntime {
    #[allow(clippy::future_not_send)]
    pub(super) async fn run_authority_search(&self, cx: &Cx) -> SearchResult<()> {
        // Keep the async command Send for the caller's runtime task. A
        // StdoutLock cannot be held across model or retrieval suspension.
        let load = |tier| async move {
            let runtime = self.clone();
            let load = move || match tier {
                EmbedderTierTag::Fast => runtime.resolve_fast_embedder(),
                EmbedderTierTag::Quality => runtime.resolve_quality_embedder()?.ok_or_else(|| {
                    SearchError::EmbedderUnavailable {
                        model: runtime.config.indexing.quality_model.clone(),
                        reason:
                            "the selected authority query requires an available quality producer"
                                .to_owned(),
                    }
                }),
            };
            if cx.blocking_pool_handle().is_some() {
                Self::search_blocking(cx, load).await?
            } else {
                load()
            }
        };
        if self.cli_input.stream {
            // The frame emitter flushes each record and owns partial writes.
            return self
                .run_authority_search_with_writer_and_loader(cx, &mut io::stdout(), load)
                .await;
        }
        let mut writer = io::BufWriter::new(io::stdout());
        self.run_authority_search_with_writer_and_loader(cx, &mut writer, load)
            .await?;
        writer.flush()?;
        Ok(())
    }

    fn validate_authority_policy(&self, mode: AuthorityMode) -> SearchResult<()> {
        if self.config.search.explain || self.cli_input.overrides.explain == Some(true) {
            return Err(refused(
                "explain",
                "native receipt search does not provide fsfs rank explanations",
            ));
        }
        if self
            .cli_input
            .overrides
            .rerank
            .unwrap_or(self.config.search.rerank)
            || self
                .cli_input
                .overrides
                .ranking_priors
                .unwrap_or(self.config.search.ranking_priors)
            || self.config.search.shadow_mode
        {
            return Err(refused(
                "policy",
                "native receipt search does not support fsfs reranking, ranking priors, or shadow observations",
            ));
        }
        if self
            .cli_input
            .overrides
            .degradation_override
            .unwrap_or(self.config.pressure.degradation_override)
            != DegradationOverrideMode::Auto
        {
            return Err(refused(
                "policy",
                "select the native mode explicitly with --authority-mode; pressure degradation overrides cannot change a trusted receipt query",
            ));
        }
        if mode.needs_quality()
            && (self
                .cli_input
                .overrides
                .fast_only
                .unwrap_or(self.config.search.fast_only)
                || self.quality_work_deferred()
                || !self
                    .cli_input
                    .overrides
                    .profile
                    .unwrap_or(self.config.pressure.profile)
                    .quality_enabled()
                || self
                    .cli_input
                    .overrides
                    .hard_pause_requested
                    .unwrap_or(self.config.pressure.hard_pause_requested)
                || self
                    .cli_input
                    .overrides
                    .quality_circuit_open
                    .unwrap_or(self.config.pressure.quality_circuit_open))
        {
            return Err(refused(
                "policy",
                "the selected mode requires quality; choose a profile that permits quality, disable search.fast_only with --no-fast-only, and clear deferred-quality, hard-pause, and quality-circuit policy",
            ));
        }
        if mode != AuthorityMode::Lexical {
            self.validate_registered_local_model_intent()?;
        }
        if mode.needs_fast() {
            self.selected_fast_model_id()?;
        }
        if mode.needs_quality()
            && super::select_quality_model(&self.config.indexing.quality_model)?.is_none()
        {
            return Err(refused(
                "quality",
                "the selected mode requires indexing.quality_model to name its independently configured producer",
            ));
        }
        Ok(())
    }

    /// The injectable loader changes only model acquisition. Receipt/root,
    /// producer, native query, and retained source admission always run here.
    #[allow(clippy::future_not_send, clippy::too_many_lines)]
    pub(super) async fn run_authority_search_with_writer_and_loader<W, F, Fut>(
        &self,
        cx: &Cx,
        writer: &mut W,
        load: F,
    ) -> SearchResult<()>
    where
        W: Write,
        F: FnMut(EmbedderTierTag) -> Fut,
        Fut: Future<Output = SearchResult<Arc<dyn Embedder>>>,
    {
        if !self.cli_input.stream {
            let started = Instant::now();
            return self
                .execute_authority_query(cx, load, |payload| {
                    let elapsed_ms =
                        u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                    let envelope = OutputEnvelope::success(
                        payload,
                        meta_for_format("search", self.cli_input.format)
                            .with_duration_ms(elapsed_ms),
                        iso_timestamp_now(),
                    );
                    emit_envelope(&envelope, self.cli_input.format, writer)
                })
                .await;
        }
        self.cli_input.validate_authority_selection()?;
        let stream_id = format!("search-{}-{}", pressure_timestamp_ms(), std::process::id());
        let mut writer = SearchStreamWriter {
            inner: writer,
            failed: false,
            wrote: false,
        };
        let format = self.cli_input.format;
        let mut seq = 0;
        let mut send = |writer: &mut SearchStreamWriter<'_, W>,
                        event: StreamEvent<AuthoritySearchHit>,
                        evidence| {
            let mut frame = StreamFrame::new(&stream_id, seq, iso_timestamp_now(), "search", event);
            frame.native_authority = evidence;
            emit_stream_frame(&frame, format, writer)?;
            seq += 1;
            Ok::<(), SearchError>(())
        };
        let started = StreamStartedEvent::new(
            &stream_id,
            self.cli_input.query.as_deref().unwrap_or_default(),
            format.to_string(),
        );
        let result = async {
            send(&mut writer, StreamEvent::Started(started), None)?;
            self.execute_authority_query(cx, load, |page| {
                let AuthorityPage {
                    mut evidence,
                    mode,
                    phase,
                    results,
                } = page;
                let stage = match (phase, mode) {
                    ("refined", _) | (_, "quality") => "retrieve.quality",
                    (_, "lexical") => "retrieve.lexical",
                    _ => "retrieve.fast",
                };
                let progress = StreamProgressEvent {
                    stage: stage.into(),
                    completed_units: results.len() as u64,
                    total_units: None,
                    reason_code: format!("query.stream.{phase}_ready"),
                    message: format!("{phase} results ready"),
                    zero_signal: Default::default(),
                    recovery_plans: Default::default(),
                };
                evidence["mode"] = mode.into();
                evidence["phase"] = phase.into();
                send(&mut writer, StreamEvent::Progress(progress), Some(evidence))?;
                for (index, item) in results.into_iter().enumerate() {
                    retained_search_checkpoint(cx)?;
                    send(
                        &mut writer,
                        StreamEvent::Result(StreamResultEvent {
                            rank: index as u64 + 1,
                            item,
                        }),
                        None,
                    )?;
                }
                Ok(())
            })
            .await
        }
        .await;
        let terminal = match &result {
            Ok(()) => terminal_event_completed(),
            Err(err) => terminal_event_from_error(err, 0, 3),
        };
        if writer.wrote && !writer.failed {
            if let Err(err) = send(&mut writer, StreamEvent::Terminal(terminal), None) {
                // TOON can reject a user-controlled error string before writing any bytes.
                if !writer.failed {
                    send(
                        &mut writer,
                        StreamEvent::Terminal(terminal_event_from_error(&err, 0, 3)),
                        None,
                    )
                    .map_err(|error| writer.mark_error_if_output_owned(error, output_error_from))?;
                }
                return Err(writer.mark_error_if_output_owned(err, output_error_from));
            }
        }
        result.map_err(|error| writer.mark_error_if_output_owned(error, output_error_from))
    }

    #[allow(clippy::future_not_send, clippy::too_many_lines)]
    async fn execute_authority_query<F, Fut>(
        &self,
        cx: &Cx,
        mut load: F,
        mut publish: impl FnMut(AuthorityPage) -> SearchResult<()>,
    ) -> SearchResult<()>
    where
        F: FnMut(EmbedderTierTag) -> Fut,
        Fut: Future<Output = SearchResult<Arc<dyn Embedder>>>,
    {
        retained_search_checkpoint(cx)?;
        self.cli_input.validate_authority_selection()?;
        let path = self
            .cli_input
            .authority_receipt
            .as_deref()
            .ok_or_else(|| refused("receipt", "search requires --authority-receipt PATH"))?;
        let query = self
            .cli_input
            .query
            .as_deref()
            .ok_or_else(|| refused("query", "missing search query argument"))?;
        let query = Self::normalize_search_query(query);
        if query.is_empty() {
            return Err(refused("query", "search query must not be empty"));
        }
        let receipt = NativeGenerationReceipt::read(path)?;
        let mode = AuthorityMode::select(&receipt, self.cli_input.authority_mode.as_deref())?;
        self.validate_authority_policy(mode)?;
        let max_image_bytes = self
            .cli_input
            .authority_max_image_bytes
            .unwrap_or(DEFAULT_IMAGE_BYTES);
        let max_source_bytes = self
            .cli_input
            .authority_max_source_bytes
            .unwrap_or(DEFAULT_SOURCE_BYTES);
        let current = self.cli_input.authority_current_receipt.as_deref();
        let (_root, snapshot) = if cx.blocking_pool_handle().is_some() {
            let selected = receipt.clone();
            let context = cx.clone();
            let current = current.map(std::path::Path::to_path_buf);
            Self::search_blocking(cx, move || {
                selected.open_read_selection(&context, current.as_deref(), max_image_bytes)
            })
            .await??
        } else {
            receipt.open_read_selection(cx, current, max_image_bytes)?
        };
        retained_search_checkpoint(cx)?;
        let total_documents =
            usize::try_from(snapshot.metadata().total_documents).map_err(|_| {
                refused(
                    "limit",
                    "the admitted live document count does not fit this platform",
                )
            })?;
        let requested = self
            .cli_input
            .overrides
            .limit
            .unwrap_or(self.config.search.default_limit);
        let limit = if requested == 0 {
            total_documents
        } else {
            requested.min(total_documents)
        };
        let fast = if mode.needs_fast() {
            Some(load(EmbedderTierTag::Fast).await?)
        } else {
            None
        };
        retained_search_checkpoint(cx)?;
        let mut quality_runtime = self.clone();
        let defer_quality =
            self.cli_input.stream && matches!(mode, AuthorityMode::Full | AuthorityMode::Partial);
        let quality = if mode.needs_quality() && !defer_quality {
            // Model loading and every required quality phase share one
            // absolute budget. Buffered full queries admit both producers before
            // Initial, so that admission and Initial also consume this budget.
            quality_runtime.quality_budget_started_at = Some(cx.now());
            Some(
                quality_runtime
                    .with_quality_deadline(cx, load(EmbedderTierTag::Quality))
                    .await?,
            )
        } else {
            None
        };
        retained_search_checkpoint(cx)?;

        let hydrate = async |phase: &'static str, results, candidates| {
            retained_search_checkpoint(cx)?;
            if cx.blocking_pool_handle().is_some() {
                let selected = receipt.clone();
                let snapshot = Arc::clone(&snapshot);
                let context = cx.clone();
                let query = query.clone();
                Self::search_blocking(cx, move || {
                    authority_payload(
                        &context,
                        &selected,
                        &snapshot,
                        &query,
                        mode,
                        phase,
                        results,
                        candidates,
                        max_source_bytes,
                    )
                })
                .await?
            } else {
                authority_payload(
                    cx,
                    &receipt,
                    &snapshot,
                    &query,
                    mode,
                    phase,
                    results,
                    candidates,
                    max_source_bytes,
                )
            }
        };
        let (phase, results, candidates) = if mode == AuthorityMode::Lexical {
            (
                "initial",
                snapshot.search_lexical_text(cx, &query, limit).await?,
                None,
            )
        } else {
            let pool = self
                .search_blocking_pool
                .clone()
                .or_else(|| cx.blocking_pool_handle());
            let deferred_producer = std::sync::OnceLock::new();
            let mut search = match mode {
                AuthorityMode::Fast | AuthorityMode::Quality => {
                    let (tier, producer) = if mode == AuthorityMode::Quality {
                        (EmbedderTierTag::Quality, quality.as_deref())
                    } else {
                        (EmbedderTierTag::Fast, fast.as_deref())
                    };
                    snapshot.search_tier_progressive(
                        cx,
                        tier,
                        producer.ok_or_else(|| {
                            refused("producer", "selected producer is unavailable")
                        })?,
                        &query,
                        limit,
                        pool,
                    )?
                }
                AuthorityMode::Full | AuthorityMode::Partial if defer_quality => snapshot
                    .search_progressive_with_deferred_quality(
                        cx,
                        fast.as_deref()
                            .ok_or_else(|| refused("producer", "fast producer is unavailable"))?,
                        &query,
                        limit,
                        pool,
                    )?,
                AuthorityMode::Full | AuthorityMode::Partial => {
                    let fast = fast
                        .as_deref()
                        .ok_or_else(|| refused("producer", "fast producer is unavailable"))?;
                    let quality = quality
                        .as_deref()
                        .ok_or_else(|| refused("producer", "quality producer is unavailable"))?;
                    match (mode, pool) {
                        (AuthorityMode::Partial, Some(pool)) => snapshot
                            .search_progressive_with_partial_quality_and_pool(
                                cx, fast, quality, &query, limit, pool,
                            )?,
                        (AuthorityMode::Partial, None) => snapshot
                            .search_progressive_with_partial_quality(
                                cx, fast, quality, &query, limit,
                            )?,
                        (_, Some(pool)) => snapshot.search_progressive_with_quality_and_pool(
                            cx, fast, quality, &query, limit, pool,
                        )?,
                        (_, None) => snapshot
                            .search_progressive_with_quality(cx, fast, quality, &query, limit)?,
                    }
                }
                AuthorityMode::Lexical => {
                    return Err(refused(
                        "mode",
                        "lexical mode cannot construct a vector query",
                    ));
                }
            }
            .with_fusion(
                RrfConfig {
                    k: self.config.search.rrf_k,
                    ..RrfConfig::default()
                },
                self.effective_quality_weight(),
            )?;
            let mut final_page = None;
            let mut quality_started = mode.needs_quality() && !defer_quality;
            loop {
                let next = if quality_started {
                    quality_runtime
                        .with_quality_deadline(cx, search.next_phase())
                        .await?
                } else {
                    search.next_phase().await?
                };
                let Some(next) = next else { break };
                let complete = matches!(&next, NativeSearchPhase::Refined { .. })
                    || matches!(mode, AuthorityMode::Fast | AuthorityMode::Quality);
                let page = match next {
                    NativeSearchPhase::Initial {
                        results,
                        candidates,
                    } => ("initial", results, Some(candidates)),
                    NativeSearchPhase::Refined {
                        results,
                        candidates,
                    } => ("refined", results, Some(candidates)),
                    NativeSearchPhase::RefinementFailed { error, .. }
                    | NativeSearchPhase::RerankFailed { error, .. } => return Err(error),
                    NativeSearchPhase::Reranked { .. } => {
                        return Err(refused("phase", "unrequested reranking phase"));
                    }
                };
                if defer_quality && !quality_started {
                    let payload = hydrate(page.0, page.1, page.2).await?;
                    retained_search_checkpoint(cx)?;
                    publish(payload)?;
                    retained_search_checkpoint(cx)?;
                    quality_runtime.quality_budget_started_at = Some(cx.now());
                    let producer = quality_runtime
                        .with_quality_deadline(cx, load(EmbedderTierTag::Quality))
                        .await?;
                    let producer = deferred_producer.get_or_init(|| producer);
                    search.bind_deferred_quality(producer.as_ref())?;
                    quality_started = true;
                } else {
                    final_page = Some(page);
                }
                // Reranking is not requested on this route. Do not start a
                // fresh timeout check just to drain a completed query to None.
                if complete {
                    break;
                }
            }
            if defer_quality && final_page.is_none() && quality_started {
                // An empty Initial can exhaust the iterator without a Refined page.
                return Ok(());
            }
            final_page.ok_or_else(|| {
                refused("phase", "native query completed without an Initial phase")
            })?
        };
        retained_search_checkpoint(cx)?;
        let payload = hydrate(phase, results, candidates).await?;
        retained_search_checkpoint(cx)?;
        publish(payload)
    }
}

#[allow(clippy::too_many_arguments)]
fn authority_payload(
    cx: &Cx,
    receipt: &NativeGenerationReceipt,
    snapshot: &NativeGenerationSnapshot,
    query: &str,
    mode: AuthorityMode,
    phase: &'static str,
    results: Vec<ScoredResult>,
    candidates: Option<NativePhaseCandidates>,
    max_source_bytes: usize,
) -> SearchResult<AuthorityPage> {
    let ids: Vec<&str> = results
        .iter()
        .map(|result| result.doc_id.as_str())
        .collect();
    let sources = snapshot.source_documents_by_ids(cx, &ids, max_source_bytes)?;
    if sources.len() != results.len() {
        return Err(refused(
            "source",
            "retained hydration changed the ranked page length",
        ));
    }
    let mut hits = Vec::with_capacity(results.len());
    for (result, source) in results.into_iter().zip(sources) {
        retained_search_checkpoint(cx)?;
        let source = source.ok_or_else(|| {
            refused(
                "source",
                "a ranked document is absent from the retained source image",
            )
        })?;
        if source.id != result.doc_id.as_str() {
            return Err(refused(
                "source",
                "retained hydration changed the ranked document identity",
            ));
        }
        hits.push(AuthoritySearchHit {
            result,
            snippet: source.content.chars().take(SNIPPET_CHARS).collect(),
            title: source.title,
            source_path: source.metadata.get("path").cloned(),
        });
    }
    let mut payload = serde_json::json!({
        "schema": "fsfs.native-authority-search.v1",
        "query": query,
        "selection_kind": "current",
        "topology": receipt.topology_name(),
        "generation": receipt.generation,
        "root_id": receipt.root_id,
        "authority": snapshot.authority().canonical_bytes().as_slice(),
        "profile": receipt.profile,
        "total_documents": snapshot.metadata().total_documents,
        "returned_hits": hits.len(),
    });
    if let Some(candidates) = candidates {
        payload["candidates"] = serde_json::json!({
            "fast": candidates.fast, "quality": candidates.quality, "lexical": candidates.lexical,
        });
    }
    let retained = snapshot.retained_snapshot();
    if let Some(current) = retained.selected_history_of() {
        payload["selected_history_of"] = serde_json::json!(current.canonical_bytes().as_slice());
        payload["selection_kind"] = if retained.selected_predecessor_of().is_some() {
            serde_json::json!("retained_predecessor")
        } else {
            serde_json::json!("retained_ancestor")
        };
    }
    if let NativeGenerationTopology::PartialQuality {
        quality_documents,
        total_documents,
        coverage_ppm,
    } = snapshot.topology()
    {
        payload["quality_coverage"] = serde_json::json!({
            "kind": "partial_quality", "quality_documents": quality_documents,
            "total_documents": total_documents, "parts_per_million": coverage_ppm,
        });
    }
    Ok(AuthorityPage {
        evidence: payload,
        mode: mode.as_str(),
        phase,
        results: hits,
    })
}

#[cfg(test)]
#[path = "authority_search_tests.rs"]
mod tests;
