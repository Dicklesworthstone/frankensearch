//! Complete-generation command routing. A selected store must never fall back
//! to the legacy mutable layout, including when its selection is corrupt.

use std::fs;
use std::io::{ErrorKind, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};

#[cfg(unix)]
use super::{FSFS_DAEMON_REQUEST_MAX_BYTES, SearchServeFrameBuffer};
use super::{
    FSFS_SEARCH_UNBOUNDED_LIMIT_SENTINEL, FSFS_TUI_INTERACTIVE_RESULT_LIMIT, FsfsRuntime,
    FtuiSession, InterfaceMode, SearchDashboardState, SearchExecutionFlags, iso_timestamp_now,
    pressure_timestamp_ms, retained_search_checkpoint, validate_retained_catalog_path,
};
use crate::adapters::format_emitter::{emit_envelope, meta_for_format};
use crate::generation_store::{
    COMPLETE_GENERATION_MANIFEST, COMPLETE_GENERATION_POINTER, CompleteGenerationStore,
    GenerationPublication,
};
use crate::output_schema::OutputEnvelope;
use crate::{CliCommand, OutputFormat, ShutdownCoordinator};

pub(super) fn require_durable_publication(
    publication: GenerationPublication,
) -> SearchResult<crate::generation_store::PublishedGeneration> {
    match publication {
        GenerationPublication::Durable(generation) => Ok(generation),
        GenerationPublication::VisibleButDurabilityUncertain { generation, source } => {
            // The rename is already visible. Never describe this as an aborted
            // rebuild or restore the predecessor after a directory-sync failure.
            Err(SearchError::SubsystemError {
                subsystem: "fsfs.complete_generation.durability",
                source: Box::new(std::io::Error::other(format!(
                    "generation {} is already visible at {}, but directory synchronization failed: {source}",
                    generation.id(),
                    generation.path().display(),
                ))),
            })
        }
    }
}

impl FsfsRuntime {
    /// Run a command with complete-generation publication and reader admission.
    ///
    /// `initialize_store` explicitly opts a new or legacy index root into this
    /// layout. Once a root has a selection or staging tree, commands recognize
    /// it without that opt-in. Rebuilds retain the predecessor until publication;
    /// searches pin one admitted bundle and do not write caches into it.
    ///
    /// Legacy roots retain their ordinary dispatch. Unsupported complete-store
    /// commands are refused instead of treating the store as a mutable index.
    ///
    /// # Errors
    /// Returns command, selection, indexing, search, output or cancellation errors.
    #[allow(clippy::future_not_send, clippy::too_many_lines)]
    pub async fn run_mode_with_complete_generations(
        &self,
        cx: &Cx,
        mode: InterfaceMode,
        shutdown: Option<&ShutdownCoordinator>,
        initialize_store: bool,
    ) -> SearchResult<()> {
        let Some(root) = self.complete_generation_command_root(initialize_store)? else {
            return match shutdown {
                Some(shutdown) => self.run_mode_with_shutdown(cx, mode, shutdown).await,
                None => self.run_mode(cx, mode).await,
            };
        };
        let dashboard = mode == InterfaceMode::Tui || self.cli_input.command == CliCommand::Tui;
        self.validate_command_inputs(if dashboard {
            CliCommand::Tui
        } else {
            self.cli_input.command
        })?;
        let _cancellation_scope = shutdown.map(|shutdown| shutdown.cancellation_scope(cx));
        retained_search_checkpoint(cx)?;
        validate_retained_catalog_path(&self.config.storage.db_path)?;
        if dashboard {
            return self.run_complete_generation_tui(cx, &root).await;
        }
        let mut stdout = std::io::stdout();
        match self.cli_input.command {
            CliCommand::Index if !self.cli_input.watch && !self.config.indexing.watch_mode => {
                self.run_complete_generation_index_with_writer(cx, &root, &mut stdout)
                    .await
            }
            #[cfg(unix)]
            CliCommand::Watch | CliCommand::Index => {
                self.run_complete_generation_watch_with_writer(cx, &root, &mut stdout)
                    .await
            }
            CliCommand::Search => {
                self.run_complete_generation_search_with_writer(cx, &root, &mut stdout)
                    .await
            }
            CliCommand::Explain => {
                self.run_complete_generation_explain_with_writer(cx, &root, &mut stdout)
                    .await
            }
            CliCommand::Flush => {
                self.run_complete_generation_flush_with_writer(cx, &root, &mut stdout)
            }
            CliCommand::Delete => {
                self.run_complete_generation_delete_with_writer(cx, &root, &mut stdout)
                    .await
            }
            CliCommand::AppendBatch => {
                self.run_complete_generation_append_batch_with_writer(cx, &root, &mut stdout)
                    .await
            }
            CliCommand::Compact => {
                self.run_complete_generation_compact_with_writer(cx, &root, &mut stdout)
                    .await
            }
            #[cfg(unix)]
            CliCommand::Serve => self.run_complete_generation_serve(cx, &root).await,
            #[cfg(unix)]
            CliCommand::Daemon => self.run_complete_generation_daemon(cx, &root).await,
            CliCommand::Status | CliCommand::Doctor => {
                let store = CompleteGenerationStore::open(cx, &root)?;
                let selected = store.active(cx)?.ok_or_else(|| {
                    complete_cli_error("selection", "no complete generation has been published")
                })?;
                let mut input = self.cli_input.clone();
                input.index_dir = Some(selected.path().to_path_buf());
                let reader_runtime = self.clone().with_cli_input(input);
                if self.cli_input.command == CliCommand::Status {
                    reader_runtime.run_status_command()
                } else {
                    reader_runtime.run_doctor_command()
                }
            }
            _ => Err(complete_cli_error(
                "command",
                "this command cannot mutate a complete-generation store in place; use a one-shot index rebuild, complete-generation watch, append-batch, delete, compact, flush, search, explain, tui, serve, daemon, status, or doctor",
            )),
        }
    }

    /// Retain one complete generation for the dashboard's entire session, just
    /// as the ordinary dashboard retains its initially opened search resources.
    /// Reopening the dashboard observes a successor; editing a query does not
    /// mix a new catalog or lexical snapshot into an earlier vector generation.
    async fn run_complete_generation_tui(&self, cx: &Cx, root: &Path) -> SearchResult<()> {
        retained_search_checkpoint(cx)?;
        if self.config.search.shadow_mode {
            return Err(SearchError::InvalidConfig {
                field: "search.shadow_mode".to_owned(),
                value: "true".to_owned(),
                reason: "shadow mode writes observation artifacts and cannot run against a sealed complete generation"
                    .to_owned(),
            });
        }
        if !std::io::stdout().is_terminal() || !std::io::stdin().is_terminal() {
            return self.run_complete_generation_tui_status(cx, root);
        }
        let mut reader = Box::pin(self.open_retained_search(cx, root)).await?;
        retained_search_checkpoint(cx)?;
        eprintln!(
            "fsfs: dashboard is pinned to complete generation {}; reopen it to see a newer generation ('/' to search, 'q' to quit)",
            reader.generation().id(),
        );
        // Do not enter generic first-run onboarding: this runtime names a sealed
        // generation, whose catalog and search resources are already admitted.
        // Initialize the dashboard from that runtime, then keep the retained
        // resources in the existing render loops for the entire session.
        let no_color = reader.runtime.cli_input.no_color || std::env::var_os("NO_COLOR").is_some();
        let status_payload = reader.runtime.collect_status_payload()?;
        let mode_hint = reader.runtime.search_mode_hint()?;
        let configured_limit = reader
            .runtime
            .cli_input
            .overrides
            .limit
            .unwrap_or(reader.runtime.config.search.default_limit);
        let result_limit = if configured_limit == FSFS_SEARCH_UNBOUNDED_LIMIT_SENTINEL {
            FSFS_TUI_INTERACTIVE_RESULT_LIMIT
        } else {
            configured_limit.max(1)
        };
        let mut state =
            SearchDashboardState::new(status_payload, mode_hint, result_limit, no_color);
        match FtuiSession::enter() {
            Ok(mut session) => {
                reader
                    .runtime
                    .run_search_dashboard_ftui(
                        cx,
                        &mut session,
                        &mut state,
                        &mut reader.resources,
                        no_color,
                    )
                    .await
            }
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "interactive fsfs search cockpit unavailable; falling back to ansi interaction"
                );
                reader
                    .runtime
                    .run_search_dashboard_ansi(cx, &mut state, &mut reader.resources, no_color)
                    .await
            }
        }
    }

    fn run_complete_generation_tui_status(&self, cx: &Cx, root: &Path) -> SearchResult<()> {
        retained_search_checkpoint(cx)?;
        // Status needs the authenticated inventory, not semantic resource
        // admission or model loading. A missing model must remain visible
        // as status rather than prevent noninteractive TUI fallback.
        let store = CompleteGenerationStore::open(cx, root)?;
        let selected = store.active(cx)?.ok_or_else(|| {
            complete_cli_error("selection", "no complete generation has been published")
        })?;
        let mut input = self.cli_input.clone();
        input.index_dir = Some(selected.path().to_path_buf());
        input.daemon = false;
        retained_search_checkpoint(cx)?;
        self.clone().with_cli_input(input).run_status_command()
    }

    fn complete_generation_command_root(
        &self,
        initialize_store: bool,
    ) -> SearchResult<Option<PathBuf>> {
        if !matches!(
            self.cli_input.command,
            CliCommand::Index
                | CliCommand::Watch
                | CliCommand::Search
                | CliCommand::Serve
                | CliCommand::Status
                | CliCommand::Doctor
                | CliCommand::Flush
                | CliCommand::AppendBatch
                | CliCommand::Delete
                | CliCommand::Compact
                | CliCommand::Daemon
                | CliCommand::Explain
                | CliCommand::Tui
        ) {
            return Ok(None);
        }
        let root = if matches!(
            self.cli_input.command,
            CliCommand::Index | CliCommand::Watch
        ) {
            self.resolve_index_root(&self.resolve_target_root()?)?
        } else {
            self.resolve_status_index_root()?
        };
        // An otherwise new legacy index below a sealed bundle would still
        // change its inventory. Check every ancestor before allowing fallback,
        // including when --index-dir names a child that does not exist yet.
        for ancestor in root.ancestors() {
            if complete_entry_exists(&ancestor.join(COMPLETE_GENERATION_MANIFEST))? {
                return Err(complete_cli_error(
                    "root",
                    "a sealed generation and its descendants cannot be used as a mutable index root; select its complete-generation store instead",
                ));
            }
        }
        // Check directory entries, not Path::exists: a dangling symlink or an
        // unreadable descriptor is not permission to enter the legacy path.
        // A staging tree also prevents fallback after an interrupted first build
        // or after the active descriptor has been removed out of protocol.
        let selected = complete_entry_exists(&root.join(COMPLETE_GENERATION_POINTER))?
            || complete_entry_exists(&root.join("generations"))?;
        Ok((initialize_store || selected).then_some(root))
    }

    /// Confirm durability of the selected immutable bundle without rescanning
    /// sources, loading models or opening engine artifacts for mutation. This
    /// barrier does not drain a separate watch process's queue. Publisher
    /// contention and uncertain final synchronization remain errors, not success
    /// receipts for unpublished or incompletely synchronized work.
    fn run_complete_generation_flush_with_writer<W: Write>(
        &self,
        cx: &Cx,
        root: &Path,
        writer: &mut W,
    ) -> SearchResult<()> {
        retained_search_checkpoint(cx)?;
        let store = CompleteGenerationStore::open(cx, root)?;
        let generation = require_durable_publication(store.flush_selected(cx)?)?;
        // The barrier has completed. A late cancellation cannot revoke it, and
        // receipt output must not describe this as a newly published generation.
        if self.cli_input.format == OutputFormat::Table {
            writeln!(
                writer,
                "Confirmed durability of complete generation {} at {} (selection unchanged; watch queue not drained)",
                generation.id(),
                generation.path().display(),
            )?;
        } else {
            let payload = serde_json::json!({
                "generation_id": generation.id(),
                "generation_path": generation.path(),
                "store_root": root,
                "manifest_sha256": generation.manifest_sha256(),
                "durability": "confirmed",
                "scope": "selected_generation",
                "generation_changed": false,
                "watch_queue_drained": false,
            });
            let envelope = OutputEnvelope::success(
                payload,
                meta_for_format("flush", self.cli_input.format),
                iso_timestamp_now(),
            );
            emit_envelope(&envelope, self.cli_input.format, writer)?;
            if !matches!(
                self.cli_input.format,
                OutputFormat::Jsonl | OutputFormat::Csv
            ) {
                writer.write_all(b"\n")?;
            }
        }
        writer.flush().map_err(SearchError::Io)
    }

    #[allow(clippy::future_not_send)]
    async fn run_complete_generation_index_with_writer<W: Write>(
        &self,
        cx: &Cx,
        root: &Path,
        writer: &mut W,
    ) -> SearchResult<()> {
        let generation =
            require_durable_publication(self.rebuild_retained_generation(cx, root).await?)?;
        self.emit_complete_generation_receipt(root, &generation, "index", writer)
    }

    pub(super) fn emit_complete_generation_receipt<W: Write>(
        &self,
        root: &Path,
        generation: &crate::generation_store::PublishedGeneration,
        command: &str,
        writer: &mut W,
    ) -> SearchResult<()> {
        if self.cli_input.format == OutputFormat::Table {
            writeln!(
                writer,
                "Published complete generation {} at {} (durable; predecessors retained)",
                generation.id(),
                generation.path().display(),
            )?;
        } else {
            let payload = serde_json::json!({
                "generation_id": generation.id(),
                "generation_path": generation.path(),
                "store_root": root,
                "manifest_sha256": generation.manifest_sha256(),
                "publication": "durable",
                "generation_complete": true,
            });
            let envelope = OutputEnvelope::success(
                payload,
                meta_for_format(command, self.cli_input.format),
                iso_timestamp_now(),
            );
            emit_envelope(&envelope, self.cli_input.format, writer)?;
            if !matches!(
                self.cli_input.format,
                OutputFormat::Jsonl | OutputFormat::Csv
            ) {
                writer.write_all(b"\n")?;
            }
        }
        writer.flush().map_err(SearchError::Io)
    }

    async fn run_complete_generation_delete_with_writer<W: Write>(
        &self,
        cx: &Cx,
        root: &Path,
        writer: &mut W,
    ) -> SearchResult<()> {
        let (publication, deleted) = self.delete_retained_generation(cx, root).await?;
        let generation = publication.map(require_durable_publication).transpose()?;
        if self.cli_input.format == OutputFormat::Table {
            writeln!(writer, "{deleted} documents deleted")?;
            if let Some(generation) = &generation {
                writeln!(
                    writer,
                    "Published complete generation {} (durable; predecessors retained)",
                    generation.id(),
                )?;
            }
        } else {
            let mut payload = serde_json::json!({
                "deleted": deleted,
                "generation_changed": generation.is_some(),
            });
            if let Some(generation) = &generation {
                payload["generation_id"] = serde_json::json!(generation.id());
                payload["generation_path"] = serde_json::json!(generation.path());
                payload["manifest_sha256"] = serde_json::json!(generation.manifest_sha256());
                payload["publication"] = serde_json::json!("durable");
                payload["generation_complete"] = serde_json::json!(true);
            }
            let envelope = OutputEnvelope::success(
                payload,
                meta_for_format("delete", self.cli_input.format),
                iso_timestamp_now(),
            );
            emit_envelope(&envelope, self.cli_input.format, writer)?;
            if !matches!(
                self.cli_input.format,
                OutputFormat::Jsonl | OutputFormat::Csv
            ) {
                writer.write_all(b"\n")?;
            }
        }
        writer.flush().map_err(SearchError::Io)
    }

    async fn run_complete_generation_append_batch_with_writer<W: Write>(
        &self,
        cx: &Cx,
        root: &Path,
        writer: &mut W,
    ) -> SearchResult<()> {
        let (publication, appended) = self.append_retained_generation(cx, root).await?;
        let generation = publication.map(require_durable_publication).transpose()?;
        if self.cli_input.format == OutputFormat::Table {
            writeln!(writer, "{appended} documents inserted or replaced")?;
            if let Some(generation) = &generation {
                return self.emit_complete_generation_receipt(
                    root,
                    generation,
                    "append-batch",
                    writer,
                );
            }
        } else {
            let mut payload = serde_json::json!({
                "appended": appended,
                "generation_changed": generation.is_some(),
            });
            if let Some(generation) = &generation {
                payload["generation_id"] = serde_json::json!(generation.id());
                payload["generation_path"] = serde_json::json!(generation.path());
                payload["manifest_sha256"] = serde_json::json!(generation.manifest_sha256());
                payload["publication"] = serde_json::json!("durable");
                payload["generation_complete"] = serde_json::json!(true);
            }
            let envelope = OutputEnvelope::success(
                payload,
                meta_for_format("append-batch", self.cli_input.format),
                iso_timestamp_now(),
            );
            emit_envelope(&envelope, self.cli_input.format, writer)?;
            if !matches!(
                self.cli_input.format,
                OutputFormat::Jsonl | OutputFormat::Csv
            ) {
                writer.write_all(b"\n")?;
            }
        }
        writer.flush().map_err(SearchError::Io)
    }

    async fn run_complete_generation_compact_with_writer<W: Write>(
        &self,
        cx: &Cx,
        root: &Path,
        writer: &mut W,
    ) -> SearchResult<()> {
        let (publication, mut payload) = self.compact_retained_generation(cx, root).await?;
        let generation = require_durable_publication(publication)?;
        if self.cli_input.format == OutputFormat::Table {
            writeln!(
                writer,
                "Compacted all present vector tiers into an isolated successor"
            )?;
            return self.emit_complete_generation_receipt(root, &generation, "compact", writer);
        }
        payload["generation_id"] = serde_json::json!(generation.id());
        payload["generation_path"] = serde_json::json!(generation.path());
        payload["store_root"] = serde_json::json!(root);
        payload["manifest_sha256"] = serde_json::json!(generation.manifest_sha256());
        payload["publication"] = serde_json::json!("durable");
        payload["generation_complete"] = serde_json::json!(true);
        let envelope = OutputEnvelope::success(
            payload,
            meta_for_format("compact", self.cli_input.format),
            iso_timestamp_now(),
        );
        emit_envelope(&envelope, self.cli_input.format, writer)?;
        if !matches!(
            self.cli_input.format,
            OutputFormat::Jsonl | OutputFormat::Csv
        ) {
            writer.write_all(b"\n")?;
        }
        writer.flush().map_err(SearchError::Io)
    }

    async fn run_complete_generation_search_with_writer<W: Write + Send>(
        &self,
        cx: &Cx,
        root: &Path,
        writer: &mut W,
    ) -> SearchResult<()> {
        let query = self
            .cli_input
            .query
            .as_deref()
            .ok_or_else(|| complete_cli_error("query", "missing search query argument"))?;
        let limit = self
            .cli_input
            .overrides
            .limit
            .unwrap_or(self.config.search.default_limit);
        if self.cli_input.stream
            && !matches!(
                self.cli_input.format,
                OutputFormat::Jsonl | OutputFormat::Toon
            )
        {
            return Err(complete_cli_error(
                "format",
                "stream mode requires --format jsonl or --format toon",
            ));
        }
        if self.config.search.explain {
            // Ordinary stores attach per-hit explanations (bd-7l7si); a sealed
            // bundle keeps its context at the store root, not wired here yet.
            return Err(complete_cli_error(
                "explain",
                "--explain is not available for a complete-generation store yet; search without it, then run `fsfs explain R<n>` for a hit",
            ));
        }
        let started = Instant::now();
        if self.cli_input.expand {
            return self
                .run_complete_expanded_search(cx, root, query, limit, started, writer)
                .await;
        }
        if self.cli_input.daemon || self.cli_input.daemon_socket.is_some() {
            #[cfg(unix)]
            {
                if self.cli_input.stream {
                    return self
                        .stream_complete_generation_daemon(cx, root, query, limit, writer)
                        .await;
                }
                let payload = self
                    .query_complete_generation_daemon(cx, root, query, limit)
                    .await?;
                return self.emit_complete_search_payload(payload, started, writer);
            }
            #[cfg(not(unix))]
            return Err(complete_cli_error(
                "daemon_transport",
                "complete-generation daemon forwarding requires Unix sockets",
            ));
        }
        let mut reader = self.open_retained_search(cx, root).await?;
        reader
            .runtime
            .enable_complete_generation_explanations(root, &reader.generation)?;
        if self.cli_input.stream {
            let stream_id = format!("search-{}-{}", pressure_timestamp_ms(), std::process::id());
            return reader
                .runtime
                .run_search_stream_command_with_writer(
                    cx,
                    query,
                    limit,
                    &stream_id,
                    writer,
                    Some((
                        &mut reader.resources,
                        SearchExecutionFlags {
                            include_snippets: true,
                            persist_explain_session: true,
                        },
                    )),
                )
                .await;
        }
        let payloads = reader
            .runtime
            .execute_search_payloads_with_mode_using_resources(
                cx,
                query,
                limit,
                super::SearchExecutionMode::Full,
                &mut reader.resources,
                SearchExecutionFlags {
                    include_snippets: true,
                    persist_explain_session: true,
                },
            )
            .await?;
        let payload = payloads.last().cloned().ok_or_else(|| {
            complete_cli_error("search", "search completed without an Initial phase")
        })?;
        self.emit_complete_search_payload(payload, started, writer)
    }

    #[allow(clippy::too_many_lines)]
    async fn run_complete_expanded_search<W: Write + Send>(
        &self,
        cx: &Cx,
        root: &Path,
        query: &str,
        limit: usize,
        started: Instant,
        writer: &mut W,
    ) -> SearchResult<()> {
        // Pin before the expansion request. A publication during the LLM call
        // or between variants must not mix lexical, vector or catalog inputs.
        let mut reader = self.open_retained_search(cx, root).await?;
        reader
            .runtime
            .enable_complete_generation_explanations(root, &reader.generation)?;
        if self.cli_input.daemon || self.cli_input.daemon_socket.is_some() {
            tracing::warn!(
                "--expand executes directly against one retained generation; daemon forwarding is ignored"
            );
        }
        if !self.cli_input.stream {
            let expansion = Self::expand_search_query(cx, query).await?;
            let payload = Self::execute_retained_expanded_queries(
                cx,
                &mut reader,
                query,
                limit,
                &expansion.queries,
            )
            .await?;
            if expansion.queries.len() > 1 {
                reader
                    .runtime
                    .invalidate_complete_generation_explanation(query)?;
            }
            return self.emit_complete_search_payload(payload, started, writer);
        }

        let stream_id = format!("search-{}-{}", pressure_timestamp_ms(), std::process::id());
        let mut seq = 0;
        reader
            .runtime
            .emit_search_stream_started(query, &stream_id, &mut seq, writer)?;
        let result = async {
            // Deliver Initial immediately. Retain the ordinary terminal phase
            // until expansion chooses the single final ranking for this stream.
            let mut sink = |payload: &crate::output_schema::SearchPayload| {
                if payload.phase == crate::output_schema::SearchOutputPhase::Initial {
                    self.emit_search_stream_payload(payload, &stream_id, &mut seq, writer)?;
                }
                Ok(())
            };
            let original = reader
                .runtime
                .execute_search_phase_artifacts_with_mode_using_resources(
                    cx,
                    query,
                    limit,
                    super::SearchExecutionMode::Full,
                    &mut reader.resources,
                    SearchExecutionFlags {
                        include_snippets: true,
                        persist_explain_session: false,
                    },
                    Some(&mut sink),
                )
                .await?;
            let expansion = Self::expand_search_query(cx, query).await?;
            if expansion.queries.len() > 1 {
                let payload = Self::execute_retained_expanded_queries(
                    cx,
                    &mut reader,
                    query,
                    limit,
                    &expansion.queries,
                )
                .await?;
                self.emit_search_stream_payload_with_stage(
                    &payload,
                    &stream_id,
                    &mut seq,
                    writer,
                    Some((
                        "retrieve.expansion",
                        "query.stream.expanded_ready",
                        "expanded query rankings fused",
                    )),
                )?;
                retained_search_checkpoint(cx)?;
                reader
                    .runtime
                    .invalidate_complete_generation_explanation(query)?;
            } else {
                let last = original
                    .last()
                    .ok_or_else(|| complete_cli_error("search", "search returned no phase"))?;
                if last.phase != crate::output_schema::SearchOutputPhase::Initial {
                    self.emit_search_stream_payload(&last.payload, &stream_id, &mut seq, writer)?;
                }
                retained_search_checkpoint(cx)?;
                reader.runtime.persist_search_artifact_explanation(
                    &reader.resources.index_root,
                    query,
                    last,
                );
            }
            Ok(())
        }
        .await;
        match result {
            Ok(()) => self.emit_search_stream_terminal_completed(&stream_id, &mut seq, writer),
            Err(error) => {
                self.emit_search_stream_terminal_error(&stream_id, &error, &mut seq, writer)?;
                Err(error)
            }
        }
    }

    async fn execute_retained_expanded_queries(
        cx: &Cx,
        reader: &mut super::RetainedSearchReader,
        query: &str,
        limit: usize,
        queries: &[crate::query_expansion::ExpandedQuery],
    ) -> SearchResult<crate::output_schema::SearchPayload> {
        retained_search_checkpoint(cx)?;
        if queries.len() <= 1 {
            return reader
                .runtime
                .execute_search_payloads_with_mode_using_resources(
                    cx,
                    query,
                    limit,
                    super::SearchExecutionMode::Full,
                    &mut reader.resources,
                    SearchExecutionFlags {
                        include_snippets: true,
                        persist_explain_session: reader.runtime.complete_explain_target.is_some(),
                    },
                )
                .await?
                .pop()
                .ok_or_else(|| complete_cli_error("search", "search returned no phase"));
        }
        let fingerprint = reader.resources.generation_fingerprint.clone();
        let payloads = reader
            .runtime
            .execute_expanded_query_variants(
                cx,
                queries,
                limit.saturating_mul(2).max(20),
                &mut reader.resources,
                &fingerprint,
                SearchExecutionFlags {
                    include_snippets: true,
                    persist_explain_session: false,
                },
            )
            .await?;
        retained_search_checkpoint(cx)?;
        Self::validate_search_generation_fingerprint(
            &reader.resources.index_root,
            &fingerprint,
            super::SearchExecutionMode::Full,
        )?;
        Ok(Self::fuse_expanded_payloads(
            query,
            &payloads,
            limit,
            reader.runtime.config.search.rrf_k,
        ))
    }

    fn emit_complete_search_payload<W: Write>(
        &self,
        payload: crate::output_schema::SearchPayload,
        started: Instant,
        writer: &mut W,
    ) -> SearchResult<()> {
        let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        if self.cli_input.format == OutputFormat::Table {
            let table = crate::adapters::format_emitter::render_search_table_for_cli(
                &payload,
                Some(elapsed_ms),
                self.cli_input.no_color,
            );
            writer.write_all(table.as_bytes())?;
        } else {
            let warnings = Self::search_generation_warnings(&payload);
            let envelope = OutputEnvelope::success(
                payload,
                meta_for_format("search", self.cli_input.format).with_duration_ms(elapsed_ms),
                iso_timestamp_now(),
            )
            .with_warnings(warnings);
            if !(self.cli_input.compact
                && crate::adapters::format_emitter::emit_compact_search_envelope(
                    &envelope,
                    self.cli_input.format,
                    writer,
                )?)
            {
                emit_envelope(&envelope, self.cli_input.format, writer)?;
            }
            if !matches!(
                self.cli_input.format,
                OutputFormat::Jsonl | OutputFormat::Csv
            ) {
                writer.write_all(b"\n")?;
            }
        }
        writer.flush().map_err(SearchError::Io)
    }
}

#[cfg(unix)]
impl FsfsRuntime {
    #[allow(clippy::future_not_send)]
    async fn run_complete_generation_serve(&self, cx: &Cx, root: &Path) -> SearchResult<()> {
        if self.cli_input.daemon || self.cli_input.daemon_socket.is_some() {
            return self.run_complete_generation_daemon(cx, root).await;
        }
        // Keep the command future Send: StdinLock contains a non-Send guard
        // that cannot be retained while a query awaits model/search work.
        let mut input = std::io::BufReader::new(std::io::stdin());
        let mut output = std::io::stdout();
        let cache_enabled = std::env::var_os("FSFS_DISABLE_QUERY_CACHE").is_none();
        self.run_complete_generation_serve_with_io(cx, root, &mut input, &mut output, cache_enabled)
            .await
    }

    // Like the existing stdio server, input is read on the owning command lane.
    // Cancellation is observed between reads/requests; a blocked stdin read is
    // not claimed to be preemptible. No detached input worker is introduced.
    #[allow(clippy::future_not_send)]
    async fn run_complete_generation_serve_with_io<R: std::io::BufRead, W: Write>(
        &self,
        cx: &Cx,
        root: &Path,
        input: &mut R,
        output: &mut W,
        cache_enabled: bool,
    ) -> SearchResult<()> {
        use std::collections::HashMap;
        use std::io::{BufRead, Read};

        let mut session = self.open_live_retained_search(cx, root).await?;
        let mut cache = HashMap::new();
        let ready = Self::search_serve_ready_event(
            self.cli_input.format.to_string(),
            &session.reader.resources,
        );
        emit_complete_serve_line(&ready, output)?;
        let mut line = Vec::new();
        loop {
            retained_search_checkpoint(cx)?;
            line.clear();
            // Bound allocation before JSON parsing, including partial clients.
            // Do not drain an arbitrarily long invalid line before refusing it.
            let count = (&mut *input)
                .take(FSFS_DAEMON_REQUEST_MAX_BYTES as u64 + 1)
                .read_until(b'\n', &mut line)?;
            if count == 0 {
                return Ok(());
            }
            if line.len() > FSFS_DAEMON_REQUEST_MAX_BYTES {
                return Err(complete_cli_error(
                    "serve_request",
                    "request exceeds 1 MiB limit",
                ));
            }
            let raw = match std::str::from_utf8(&line) {
                Ok(raw) => raw.trim(),
                Err(_) => {
                    emit_complete_serve_line(
                        &Self::search_serve_error_response("", "full", "request is not UTF-8"),
                        output,
                    )?;
                    continue;
                }
            };
            if raw.is_empty() {
                continue;
            }
            if matches!(raw, "quit" | "exit" | ":quit" | ":exit") {
                return Ok(());
            }
            let request = match Self::parse_search_serve_request(raw) {
                Ok(request) => request,
                Err(error) => {
                    emit_complete_serve_line(
                        &Self::search_serve_error_response("", "full", error.to_string()),
                        output,
                    )?;
                    continue;
                }
            };
            let query = request.query.clone();
            let mode = request.mode.clone().unwrap_or_else(|| "full".to_owned());
            let result = async {
                if session.refresh(cx).await? {
                    cache.clear();
                }
                // Both the runtime's hydration paths and the resources belong
                // to the same admitted generation. Never invoke this handler
                // with the outer store-root runtime or after a failed refresh.
                session
                    .reader
                    .runtime
                    .execute_search_serve_request(
                        cx,
                        request,
                        &mut session.reader.resources,
                        &mut cache,
                        cache_enabled,
                    )
                    .await
            }
            .await;
            let response = match result {
                Ok(response) => response,
                Err(error) => Self::search_serve_error_response(query, mode, error.to_string()),
            };
            emit_complete_serve_line(&response, output)?;
        }
    }
}

#[cfg(unix)]
fn emit_complete_serve_line<T: serde::Serialize, W: Write>(
    value: &T,
    output: &mut W,
) -> SearchResult<()> {
    // Encode fully under the existing daemon response bound before exposing
    // bytes, so an oversized response cannot leave a partial JSON record.
    let mut bytes = SearchServeFrameBuffer::default();
    serde_json::to_writer(&mut bytes, value).map_err(|error| {
        complete_cli_error(
            "serve_response",
            &format!("cannot encode response: {error}"),
        )
    })?;
    bytes.write_all(b"\n")?;
    output.write_all(&bytes.0)?;
    output.flush().map_err(SearchError::Io)
}

fn complete_entry_exists(path: &Path) -> SearchResult<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

pub(super) fn complete_cli_error(field: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("complete_generation.{field}"),
        value: String::new(),
        reason: reason.to_owned(),
    }
}

#[cfg(all(test, unix, not(feature = "embedded-models")))]
mod tests {
    use super::super::{
        DashboardSearchStage, FSFS_EXPLAIN_SESSION_FILE, SearchDashboardState, SearchExecutionMode,
        set_test_fast_embedder, set_test_quality_embedder, test_fast_embedder_override,
        test_quality_embedder_override,
    };
    use super::*;
    use crate::output_schema::{SearchHitPayload, SearchOutputPhase};
    use crate::stream_protocol::StreamFrame;
    use crate::{CliInput, FsfsConfig};
    use asupersync::test_utils::run_test_with_cx;
    use frankensearch_core::{Embedder, EmbeddingIdentityBundleV1, ModelCategory, SearchFuture};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    fn fixture(parent: &Path) -> (FsfsRuntime, PathBuf, PathBuf) {
        let source = parent.join("source");
        let root = parent.join("store");
        fs::create_dir(&source).unwrap();
        fs::write(source.join("alpha.md"), "sharedtoken alpha document").unwrap();
        let mut config = FsfsConfig::default();
        "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
        config.indexing.offline = true;
        config.indexing.quality_model.clear();
        config.search.fast_only = true;
        config.search.rerank = false;
        let input = CliInput {
            command: CliCommand::Index,
            target_path: Some(source.clone()),
            index_dir: Some(root.clone()),
            format: OutputFormat::Json,
            quiet: true,
            ..CliInput::default()
        };
        (FsfsRuntime::new(config).with_cli_input(input), source, root)
    }

    fn search_runtime(runtime: &FsfsRuntime) -> FsfsRuntime {
        let mut input = runtime.cli_input.clone();
        input.command = CliCommand::Search;
        input.query = Some("sharedtoken".to_owned());
        input.daemon = false;
        runtime.clone().with_cli_input(input)
    }

    async fn publish(runtime: &FsfsRuntime, cx: &Cx, root: &Path) -> serde_json::Value {
        let mut output = Vec::new();
        runtime
            .run_complete_generation_index_with_writer(cx, root, &mut output)
            .await
            .unwrap();
        let receipt: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(receipt["ok"], true);
        assert_eq!(receipt["data"]["publication"], "durable");
        receipt
    }

    fn sealed_inventory(root: &Path) -> BTreeMap<PathBuf, Option<Vec<u8>>> {
        let mut entries = BTreeMap::new();
        let mut pending = vec![root.to_path_buf()];
        while let Some(directory) = pending.pop() {
            for entry in fs::read_dir(directory).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();
                let relative = path.strip_prefix(root).unwrap().to_path_buf();
                if entry.file_type().unwrap().is_dir() {
                    entries.insert(relative, None);
                    pending.push(path);
                } else {
                    entries.insert(relative, Some(fs::read(path).unwrap()));
                }
            }
        }
        entries
    }

    // Deterministic, independently identified semantic fixtures exercise the
    // real dashboard's stage routing; they make no model-quality claim.
    struct DashboardEmbedder {
        identity: EmbeddingIdentityBundleV1,
        category: ModelCategory,
    }

    impl Embedder for DashboardEmbedder {
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
        }

        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async { Ok(vec![1.0, 0.0, 0.0]) })
        }

        fn dimension(&self) -> usize {
            3
        }

        fn id(&self) -> &str {
            &self.identity.space.logical_model_id
        }

        fn model_name(&self) -> &str {
            self.id()
        }

        fn is_semantic(&self) -> bool {
            true
        }

        fn category(&self) -> ModelCategory {
            self.category
        }
    }

    struct RestoreDashboardEmbedders {
        fast: Option<Arc<dyn Embedder>>,
        quality: Option<Arc<dyn Embedder>>,
    }

    impl RestoreDashboardEmbedders {
        fn install() -> Self {
            let previous = Self {
                fast: test_fast_embedder_override(),
                quality: test_quality_embedder_override(),
            };
            set_test_fast_embedder(Some(Arc::new(DashboardEmbedder {
                identity: EmbeddingIdentityBundleV1::explicit_test_model("dashboard-fast", 3),
                category: ModelCategory::StaticEmbedder,
            })));
            set_test_quality_embedder(Some(Arc::new(DashboardEmbedder {
                identity: EmbeddingIdentityBundleV1::explicit_test_model("dashboard-quality", 3),
                category: ModelCategory::TransformerEmbedder,
            })));
            previous
        }
    }

    impl Drop for RestoreDashboardEmbedders {
        fn drop(&mut self) {
            set_test_fast_embedder(self.fast.take());
            set_test_quality_embedder(self.quality.take());
        }
    }

    #[test]
    fn complete_tui_stages_pin_vectors_lexical_and_snippets_across_publication() {
        let scheduler = asupersync::runtime::RuntimeBuilder::current_thread()
            .blocking_threads(0, 2)
            .build()
            .unwrap();
        scheduler.block_on(async move {
            let cx = Cx::current().expect("current-thread root installs a spawn-capable context");
            let _embedders = RestoreDashboardEmbedders::install();
            let directory = tempfile::tempdir().unwrap();
            let (mut runtime, source, root) = fixture(directory.path());
            runtime.config.search.fast_only = false;
            runtime.config.search.quality_timeout_ms = 5_000;
            "dashboard-quality".clone_into(&mut runtime.config.indexing.quality_model);
            fs::write(
                source.join("alpha.md"),
                "sharedtoken original document body",
            )
            .unwrap();
            publish(&runtime, &cx, &root).await;
            let mut reader = runtime.open_retained_search(&cx, &root).await.unwrap();
            let predecessor = reader.generation().clone();
            let before = sealed_inventory(predecessor.path());
            let status = reader.runtime.collect_status_payload().unwrap();
            assert_eq!(status.index.path, predecessor.path().display().to_string());
            let mut state = SearchDashboardState::new(status, None, 10, true);
            state
                .query_input
                .set_value("how to find sharedtoken document content");
            reader
                .runtime
                .refresh_search_dashboard_lexical(&cx, &mut state, &mut reader.resources)
                .await;
            assert_eq!(state.stage_chain, vec![DashboardSearchStage::Lexical]);
            assert_eq!(state.latest_hits().len(), 1);
            assert_eq!(state.latest_hits()[0].path, "alpha.md");
            assert!(state.latest_hits()[0].semantic_rank.is_none());

            fs::write(
                source.join("alpha.md"),
                "sharedtoken rewritten successor body",
            )
            .unwrap();
            fs::write(source.join("beta.md"), "sharedtoken new successor document").unwrap();
            publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let successor = store.active(&cx).unwrap().unwrap();
            let successor_before = sealed_inventory(successor.path());
            assert_ne!(predecessor, successor);

            reader
                .runtime
                .refresh_search_dashboard_semantic_fast(&cx, &mut state, &mut reader.resources)
                .await;
            assert!(state.last_error.is_none(), "{:?}", state.last_error);
            assert_eq!(state.latest_hits().len(), 1);
            assert!(state.latest_hits()[0].semantic_rank.is_some());
            reader
                .runtime
                .refresh_search_dashboard_quality(&cx, &mut state, &mut reader.resources)
                .await;
            assert!(state.last_error.is_none(), "{:?}", state.last_error);
            let refined = state.latest_payload().unwrap();
            assert_eq!(refined.phase, SearchOutputPhase::Refined);
            assert_eq!(refined.hits.len(), 1);
            assert_eq!(refined.hits[0].path, "alpha.md");
            assert!(
                refined.hits[0]
                    .snippet
                    .as_ref()
                    .unwrap()
                    .contains("original")
            );
            let blend = refined.semantic_blend.as_ref().unwrap();
            assert_eq!(blend.fast_embedder, "dashboard-fast");
            assert_eq!(blend.quality_embedder, "dashboard-quality");
            assert_eq!(reader.generation(), &predecessor);

            let mut fresh = runtime.open_retained_search(&cx, &root).await.unwrap();
            let mut fresh_state = SearchDashboardState::new(
                fresh.runtime.collect_status_payload().unwrap(),
                None,
                10,
                true,
            );
            fresh_state.query_input.set_value(state.query_input.value());
            fresh
                .runtime
                .refresh_search_dashboard_full(&cx, &mut fresh_state, &mut fresh.resources)
                .await;
            assert!(
                fresh_state.last_error.is_none(),
                "{:?}",
                fresh_state.last_error
            );
            assert_eq!(fresh_state.latest_hits().len(), 2);
            assert_eq!(
                fresh_state.latest_payload().unwrap().phase,
                SearchOutputPhase::Refined
            );
            assert!(
                fresh_state
                    .latest_hits()
                    .iter()
                    .find(|hit| hit.path == "alpha.md")
                    .unwrap()
                    .snippet
                    .as_ref()
                    .unwrap()
                    .contains("rewritten")
            );
            cx.set_cancel_requested(true);
            let error = reader
                .runtime
                .refresh_search_dashboard_mode_with_limit(
                    &cx,
                    &mut state,
                    SearchExecutionMode::Full,
                    &mut reader.resources,
                    10,
                    true,
                )
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::Cancelled { .. }));
            cx.set_cancel_requested(false);
            assert_eq!(reader.generation(), &predecessor);
            assert_eq!(sealed_inventory(predecessor.path()), before);
            assert_eq!(sealed_inventory(successor.path()), successor_before);
            assert!(!root.join(FSFS_EXPLAIN_SESSION_FILE).exists());
        });
    }

    #[test]
    fn complete_tui_noninteractive_status_preserves_the_selected_bundle() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let before = sealed_inventory(&root);
            let mut tui = search_runtime(&runtime);
            tui.cli_input.query = None;
            tui.config.indexing.model_dir = directory
                .path()
                .join("missing-models")
                .display()
                .to_string();
            // A native quality loader would require both model artifacts and
            // a caller-owned blocking pool; status needs neither.
            super::super::FSFS_NATIVE_QUALITY_MODEL_ID
                .clone_into(&mut tui.config.indexing.quality_model);
            tui.run_complete_generation_tui_status(&cx, &root).unwrap();
            assert_eq!(sealed_inventory(&root), before);
            assert!(!directory.path().join("missing-models").exists());
        });
    }

    #[test]
    fn complete_tui_refuses_corrupt_selection_shadow_and_cancellation_before_onboarding() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let before = sealed_inventory(&root);
            let mut tui = search_runtime(&runtime);
            // Auto TUI mode can originate from a Search command without a
            // query. It must reach TUI admission, not CLI query validation.
            tui.cli_input.query = None;
            tui.config.search.shadow_mode = true;
            let error = tui
                .run_mode_with_complete_generations(&cx, InterfaceMode::Tui, None, false)
                .await
                .unwrap_err();
            assert!(
                matches!(error, SearchError::InvalidConfig { field, .. } if field == "search.shadow_mode")
            );
            tui.config.search.shadow_mode = false;
            tui.cli_input.command = CliCommand::Tui;
            cx.set_cancel_requested(true);
            let error = tui
                .run_mode_with_complete_generations(&cx, InterfaceMode::Cli, None, false)
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::Cancelled { .. }));
            cx.set_cancel_requested(false);
            assert_eq!(sealed_inventory(&root), before);

            fs::write(root.join(COMPLETE_GENERATION_POINTER), "corrupt").unwrap();
            let corrupt_before = sealed_inventory(&root);
            tui.run_mode_with_complete_generations(&cx, InterfaceMode::Tui, None, false)
                .await
                .unwrap_err();
            assert_eq!(sealed_inventory(&root), corrupt_before);
        });
    }

    #[test]
    fn complete_expansion_pins_all_variants_and_preserves_sealed_inventories() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let mut reader = runtime.open_retained_search(&cx, &root).await.unwrap();
            let predecessor = reader.generation().clone();
            let before = fs::read(predecessor.path().join(COMPLETE_GENERATION_MANIFEST)).unwrap();
            fs::write(source.join("beta.md"), "sharedtoken beta successor").unwrap();
            publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let selected = store.active(&cx).unwrap().unwrap();
            let queries = [
                crate::query_expansion::ExpandedQuery {
                    text: "sharedtoken".to_owned(),
                    strategy: crate::query_expansion::ExpansionStrategy::Original,
                },
                crate::query_expansion::ExpandedQuery {
                    text: "alpha beta".to_owned(),
                    strategy: crate::query_expansion::ExpansionStrategy::Keyword,
                },
            ];
            let old = FsfsRuntime::execute_retained_expanded_queries(
                &cx,
                &mut reader,
                "sharedtoken",
                10,
                &queries,
            )
            .await
            .unwrap();
            assert_eq!(old.hits.len(), 1);
            assert_eq!(old.hits[0].path, "alpha.md");
            assert_eq!(reader.generation(), &predecessor);
            let mut fresh = runtime.open_retained_search(&cx, &root).await.unwrap();
            let new = FsfsRuntime::execute_retained_expanded_queries(
                &cx,
                &mut fresh,
                "sharedtoken",
                10,
                &queries,
            )
            .await
            .unwrap();
            assert_eq!(new.hits.len(), 2);
            assert_eq!(new.query, "sharedtoken");
            assert_eq!(store.active(&cx).unwrap(), Some(selected));
            assert_eq!(
                fs::read(predecessor.path().join(COMPLETE_GENERATION_MANIFEST)).unwrap(),
                before,
            );
            for generation in [reader.generation(), fresh.generation()] {
                assert!(
                    !generation
                        .path()
                        .join(super::super::FSFS_EXPLAIN_SESSION_FILE)
                        .exists()
                );
                assert!(!generation.path().join("query_cache").exists());
            }
        });
    }

    #[test]
    fn complete_expansion_without_provider_preserves_original_phase_and_cancellation() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let mut reader = runtime.open_retained_search(&cx, &root).await.unwrap();
            let original = reader
                .search(&cx, "sharedtoken", 1)
                .await
                .unwrap()
                .pop()
                .unwrap();
            let expansion = crate::query_expansion::expand_query(
                "sharedtoken",
                &std::collections::HashMap::new(),
            );
            assert_eq!(expansion.queries.len(), 1);
            let fallback = FsfsRuntime::execute_retained_expanded_queries(
                &cx,
                &mut reader,
                "sharedtoken",
                1,
                &expansion.queries,
            )
            .await
            .unwrap();
            assert_eq!(fallback.phase, original.phase);
            assert_eq!(fallback.hits, original.hits);
            let selected = reader.generation().clone();
            cx.set_cancel_requested(true);
            let error = FsfsRuntime::execute_retained_expanded_queries(
                &cx,
                &mut reader,
                "sharedtoken",
                1,
                &expansion.queries,
            )
            .await
            .unwrap_err();
            assert!(matches!(error, SearchError::Cancelled { .. }));
            cx.set_cancel_requested(false);
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(selected),
            );
        });
    }

    struct PublicationWriter {
        bytes: Vec<u8>,
        root: PathBuf,
        flushes: usize,
        switches: std::collections::VecDeque<(usize, Vec<u8>)>,
    }

    impl Write for PublicationWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.flushes += 1;
            if self
                .switches
                .front()
                .is_some_and(|(after, _)| *after == self.flushes)
            {
                let (_, pointer) = self.switches.pop_front().expect("scheduled switch");
                let temporary = self.root.join("test-selection-switch");
                fs::write(&temporary, pointer)?;
                fs::rename(temporary, self.root.join(COMPLETE_GENERATION_POINTER))?;
            }
            Ok(())
        }
    }

    fn serve_lines(bytes: &[u8]) -> Vec<serde_json::Value> {
        std::str::from_utf8(bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn complete_serve_reuses_warm_cache_then_observes_successor_at_request_boundary() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let pointer_path = root.join(COMPLETE_GENERATION_POINTER);
            let first = fs::read(&pointer_path).unwrap();
            fs::write(source.join("beta.md"), "sharedtoken beta document").unwrap();
            publish(&runtime, &cx, &root).await;
            let successor = fs::read(&pointer_path).unwrap();
            // Replay genuine, fully admitted bundles; the output flush places
            // publication exactly between complete requests, not inside search.
            fs::write(&pointer_path, first).unwrap();
            let mut output = PublicationWriter {
                bytes: Vec::new(),
                root: root.clone(),
                flushes: 0,
                switches: [(3, successor)].into(),
            };
            let mut input =
                std::io::Cursor::new(b"sharedtoken\nsharedtoken\nsharedtoken\nsharedtoken\nquit\n");
            runtime
                .run_complete_generation_serve_with_io(&cx, &root, &mut input, &mut output, true)
                .await
                .unwrap();
            let lines = serve_lines(&output.bytes);
            assert_eq!(lines.len(), 5);
            assert_eq!(lines[0]["event"], "ready");
            for (row, cached, count) in [(1, false, 1), (2, true, 1), (3, false, 2), (4, true, 2)] {
                assert_eq!(lines[row]["ok"], true);
                assert_eq!(lines[row]["cached"], cached);
                let phases = lines[row]["payloads"].as_array().unwrap();
                assert_eq!(
                    phases.last().unwrap()["hits"].as_array().unwrap().len(),
                    count
                );
            }
            assert!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap()
                    .is_some()
            );
        });
    }

    #[test]
    fn complete_serve_refuses_stale_cache_on_bad_selection_and_recovers_after_repair() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let pointer_path = root.join(COMPLETE_GENERATION_POINTER);
            let first = fs::read(&pointer_path).unwrap();
            fs::write(source.join("beta.md"), "sharedtoken beta document").unwrap();
            publish(&runtime, &cx, &root).await;
            let successor = fs::read(&pointer_path).unwrap();
            fs::write(&pointer_path, first).unwrap();
            let mut output = PublicationWriter {
                bytes: Vec::new(),
                root: root.clone(),
                flushes: 0,
                switches: [(2, b"corrupt selection".to_vec()), (3, successor)].into(),
            };
            let mut input = std::io::Cursor::new(b"sharedtoken\nsharedtoken\nsharedtoken\nquit\n");
            runtime
                .run_complete_generation_serve_with_io(&cx, &root, &mut input, &mut output, true)
                .await
                .unwrap();
            let lines = serve_lines(&output.bytes);
            assert_eq!(lines.len(), 4);
            assert_eq!(lines[1]["ok"], true);
            assert_eq!(lines[2]["ok"], false);
            assert_eq!(lines[2]["cached"], false);
            assert!(lines[2]["payloads"].as_array().unwrap().is_empty());
            assert_eq!(lines[3]["ok"], true);
            assert_eq!(lines[3]["cached"], false);
            let phases = lines[3]["payloads"].as_array().unwrap();
            assert_eq!(phases.last().unwrap()["hits"].as_array().unwrap().len(), 2);
            assert!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap()
                    .is_some()
            );
        });
    }

    #[test]
    fn complete_serve_bounds_unterminated_input_without_mutating_the_bundle() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let before = store.active(&cx).unwrap();
            let mut input = std::io::Cursor::new(vec![b'a'; FSFS_DAEMON_REQUEST_MAX_BYTES + 1]);
            let mut output = Vec::new();
            let error = runtime
                .run_complete_generation_serve_with_io(&cx, &root, &mut input, &mut output, true)
                .await
                .unwrap_err();
            assert!(matches!(
                error,
                SearchError::InvalidConfig { field, .. } if field == "complete_generation.serve_request"
            ));
            assert_eq!(
                serve_lines(&output).len(),
                1,
                "only the ready event is visible"
            );
            assert_eq!(store.active(&cx).unwrap(), before);
        });
    }

    #[test]
    fn complete_serve_bad_utf8_and_json_do_not_poison_the_next_request() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let mut input = std::io::Cursor::new(b"\xff\n{invalid json\nsharedtoken\nquit\n");
            let mut output = Vec::new();
            runtime
                .run_complete_generation_serve_with_io(&cx, &root, &mut input, &mut output, true)
                .await
                .unwrap();
            let lines = serve_lines(&output);
            assert_eq!(lines.len(), 4);
            assert_eq!(lines[1]["ok"], false);
            assert_eq!(lines[2]["ok"], false);
            assert_eq!(lines[3]["ok"], true);
            assert_eq!(lines[3]["cached"], false);
        });
    }

    #[test]
    fn complete_serve_oversized_response_exposes_no_partial_record() {
        let oversized = "x".repeat(super::super::FSFS_DAEMON_RESPONSE_MAX_BYTES);
        let mut output = Vec::new();
        emit_complete_serve_line(&oversized, &mut output).unwrap_err();
        assert!(output.is_empty());
    }

    #[test]
    fn complete_cli_routes_only_opted_in_or_previously_initialized_roots() {
        let directory = tempfile::tempdir().unwrap();
        let (runtime, _, root) = fixture(directory.path());
        assert!(
            runtime
                .complete_generation_command_root(false)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            runtime.complete_generation_command_root(true).unwrap(),
            Some(root.clone())
        );
        assert!(!root.exists(), "route selection is read-only");
        fs::create_dir(&root).unwrap();
        fs::create_dir(root.join("generations")).unwrap();
        assert_eq!(
            runtime.complete_generation_command_root(false).unwrap(),
            Some(root)
        );
    }

    #[test]
    fn complete_cli_does_not_treat_a_dangling_selection_as_a_legacy_root() {
        let directory = tempfile::tempdir().unwrap();
        let (runtime, _, root) = fixture(directory.path());
        fs::create_dir(&root).unwrap();
        std::os::unix::fs::symlink("absent", root.join(COMPLETE_GENERATION_POINTER)).unwrap();
        assert_eq!(
            runtime.complete_generation_command_root(false).unwrap(),
            Some(root)
        );
    }

    #[test]
    fn complete_cli_refuses_new_indexes_beneath_published_generations() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let generation = store.active(&cx).unwrap().unwrap();
            let manifest_path = generation.path().join(COMPLETE_GENERATION_MANIFEST);
            let before = fs::read(&manifest_path).unwrap();
            let pointer = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            let explicit_child = generation.path().join("new-explicit-index");
            let mut explicit_input = runtime.cli_input.clone();
            explicit_input.index_dir = Some(explicit_child.clone());
            let explicit = runtime.clone().with_cli_input(explicit_input);

            // A real existing component supplies the nested target. The
            // default relative index directory would be a new child beneath
            // that target, so an exact-root-only marker check misses it.
            let target = generation.path().join("lexical");
            assert!(target.is_dir());
            let mut nested_input = runtime.cli_input.clone();
            nested_input.target_path = Some(target.clone());
            nested_input.index_dir = None;
            let mut nested = runtime.clone().with_cli_input(nested_input);
            "new-relative-index".clone_into(&mut nested.config.storage.index_dir);
            let relative_child = target.join("new-relative-index");

            for attempted in [&explicit, &nested] {
                for initialize in [false, true] {
                    let error = attempted
                        .run_mode_with_complete_generations(
                            &cx,
                            InterfaceMode::Cli,
                            None,
                            initialize,
                        )
                        .await
                        .expect_err("dispatch must refuse a write below a sealed generation");
                    assert!(
                        matches!(
                            error,
                            SearchError::InvalidConfig { ref field, .. }
                                if field == "complete_generation.root"
                        ),
                        "{error}",
                    );
                    // active() rehashes the complete inventory, so this
                    // verifies every original file and excludes new entries.
                    assert_eq!(store.active(&cx).unwrap(), Some(generation.clone()));
                    assert_eq!(fs::read(&manifest_path).unwrap(), before);
                    assert_eq!(
                        fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                        pointer
                    );
                    assert!(!explicit_child.exists());
                    assert!(!relative_child.exists());
                }
            }
        });
    }

    #[test]
    fn complete_cli_rebuild_and_json_search_preserve_an_open_predecessor() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let first_receipt = publish(&runtime, &cx, &root).await;
            let mut old = runtime.open_retained_search(&cx, &root).await.unwrap();
            fs::write(source.join("beta.md"), "sharedtoken beta document").unwrap();
            let second_receipt = publish(&runtime, &cx, &root).await;
            assert_ne!(
                first_receipt["data"]["generation_id"],
                second_receipt["data"]["generation_id"]
            );
            let mut output = Vec::new();
            search_runtime(&runtime)
                .run_complete_generation_search_with_writer(&cx, &root, &mut output)
                .await
                .unwrap();
            let result: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(result["data"]["hits"].as_array().unwrap().len(), 2);
            let old_results = old.search(&cx, "sharedtoken", 10).await.unwrap();
            assert_eq!(old_results.last().unwrap().hits.len(), 1);
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let current = store.active(&cx).unwrap().unwrap();
            assert_eq!(
                current.id(),
                second_receipt["data"]["generation_id"].as_str().unwrap()
            );
            assert!(!current.path().join(FSFS_EXPLAIN_SESSION_FILE).exists());
        });
    }

    #[test]
    fn complete_cli_cancelled_rebuild_cannot_emit_success_or_replace_selection() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let pointer = root.join(COMPLETE_GENERATION_POINTER);
            let before = fs::read(&pointer).unwrap();
            let mut output = Vec::new();
            cx.set_cancel_requested(true);
            let error = runtime
                .run_complete_generation_index_with_writer(&cx, &root, &mut output)
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::Cancelled { .. }));
            cx.set_cancel_requested(false);
            assert!(output.is_empty());
            assert_eq!(fs::read(pointer).unwrap(), before);
            assert!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap()
                    .is_some()
            );
        });
    }

    #[test]
    fn complete_cli_stream_uses_the_existing_protocol_without_changing_the_bundle() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let before = store.active(&cx).unwrap();
            let mut search = search_runtime(&runtime);
            search.cli_input.stream = true;
            search.cli_input.format = OutputFormat::Jsonl;
            let mut output = Vec::new();
            search
                .run_complete_generation_search_with_writer(&cx, &root, &mut output)
                .await
                .unwrap();
            let frames: Vec<serde_json::Value> = String::from_utf8(output)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            assert!(!frames.is_empty());
            // Deserialize the actual public stream type as well as inspecting
            // the serialized sequence; this catches accidental envelope output.
            for frame in &frames {
                let _: StreamFrame<SearchHitPayload> =
                    serde_json::from_value(frame.clone()).unwrap();
            }
            assert_eq!(store.active(&cx).unwrap(), before);
        });
    }

    #[test]
    fn complete_cli_refuses_corrupt_selection_before_any_search_output() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            fs::write(root.join(COMPLETE_GENERATION_POINTER), "corrupt").unwrap();
            let search = search_runtime(&runtime);
            assert_eq!(
                search.complete_generation_command_root(false).unwrap(),
                Some(root.clone())
            );
            let mut output = Vec::new();
            search
                .run_complete_generation_search_with_writer(&cx, &root, &mut output)
                .await
                .unwrap_err();
            assert!(output.is_empty());
            assert_eq!(
                fs::read_to_string(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                "corrupt"
            );
        });
    }

    #[test]
    fn complete_cli_flush_dispatch_preserves_the_selected_bundle() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let before = store.active(&cx).unwrap();
            let mut input = runtime.cli_input.clone();
            input.command = CliCommand::Flush;
            runtime
                .clone()
                .with_cli_input(input)
                .run_mode_with_complete_generations(&cx, InterfaceMode::Cli, None, false)
                .await
                .unwrap();
            assert_eq!(store.active(&cx).unwrap(), before);
        });
    }

    #[test]
    fn complete_cli_flush_receipts_bind_selection_without_rescan_or_model_loading() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (mut runtime, source, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let selected = store.active(&cx).unwrap().unwrap();
            let before = sealed_inventory(selected.path());
            let pointer = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            let count = fs::read_dir(root.join("generations")).unwrap().count();
            fs::write(source.join("alpha.md"), "unindexed replacement body").unwrap();
            let models = directory.path().join("missing-models");
            runtime.config.indexing.model_dir = models.display().to_string();
            super::super::FSFS_NATIVE_QUALITY_MODEL_ID
                .clone_into(&mut runtime.config.indexing.quality_model);
            runtime.cli_input.command = CliCommand::Flush;
            for format in [OutputFormat::Json, OutputFormat::Jsonl, OutputFormat::Table] {
                runtime.cli_input.format = format;
                let mut output = Vec::new();
                runtime
                    .run_complete_generation_flush_with_writer(&cx, &root, &mut output)
                    .unwrap();
                assert_eq!(output.last(), Some(&b'\n'));
                if format == OutputFormat::Table {
                    let text = std::str::from_utf8(&output).unwrap();
                    assert!(text.contains("Confirmed durability"));
                    assert!(text.contains(selected.id()));
                    assert!(text.contains("watch queue not drained"));
                    assert!(!text.contains("Published"));
                } else {
                    let receipt: serde_json::Value = serde_json::from_slice(&output).unwrap();
                    assert_eq!(receipt["ok"], true);
                    assert_eq!(receipt["data"]["generation_id"], selected.id());
                    assert_eq!(
                        receipt["data"]["manifest_sha256"],
                        selected.manifest_sha256()
                    );
                    assert_eq!(receipt["data"]["durability"], "confirmed");
                    assert_eq!(receipt["data"]["scope"], "selected_generation");
                    assert_eq!(receipt["data"]["generation_changed"], false);
                    assert_eq!(receipt["data"]["watch_queue_drained"], false);
                    if format == OutputFormat::Jsonl {
                        assert_eq!(String::from_utf8_lossy(&output).matches('\n').count(), 1);
                    }
                }
                assert_eq!(store.active(&cx).unwrap(), Some(selected.clone()));
                assert_eq!(sealed_inventory(selected.path()), before);
                assert_eq!(
                    fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                    pointer
                );
                assert_eq!(
                    fs::read_dir(root.join("generations")).unwrap().count(),
                    count
                );
                assert!(!models.exists());
            }
        });
    }

    #[test]
    fn complete_cli_flush_refuses_absent_corrupt_selection_and_damaged_bundles() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (mut runtime, _, root) = fixture(directory.path());
            runtime.cli_input.command = CliCommand::Flush;
            let mut output = Vec::new();
            assert!(
                runtime
                    .run_complete_generation_flush_with_writer(&cx, &root, &mut output)
                    .is_err()
            );
            assert!(output.is_empty());
            assert!(!root.exists());
            CompleteGenerationStore::create(&cx, &root).unwrap();
            assert!(
                runtime
                    .run_complete_generation_flush_with_writer(&cx, &root, &mut output)
                    .is_err()
            );
            assert!(output.is_empty());
            assert!(!root.join(COMPLETE_GENERATION_POINTER).exists());
            publish(&runtime, &cx, &root).await;
            let pointer_path = root.join(COMPLETE_GENERATION_POINTER);
            let pointer = fs::read(&pointer_path).unwrap();
            fs::write(&pointer_path, b"corrupt selection").unwrap();
            assert!(
                runtime
                    .run_complete_generation_flush_with_writer(&cx, &root, &mut output)
                    .is_err()
            );
            assert!(output.is_empty());
            assert_eq!(fs::read(&pointer_path).unwrap(), b"corrupt selection");
            fs::write(&pointer_path, &pointer).unwrap();
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let selected = store.active(&cx).unwrap().unwrap();
            fs::write(selected.path().join("unexpected-artifact"), b"damage").unwrap();
            let damaged = sealed_inventory(selected.path());
            assert!(
                runtime
                    .run_complete_generation_flush_with_writer(&cx, &root, &mut output)
                    .is_err()
            );
            assert!(output.is_empty());
            assert_eq!(sealed_inventory(selected.path()), damaged);
            assert_eq!(fs::read(pointer_path).unwrap(), pointer);
        });
    }

    #[test]
    fn complete_cli_flush_cancellation_and_publisher_contention_emit_no_success() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let selected = store.active(&cx).unwrap();
            let pointer = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            let pending = store.begin(&cx).unwrap();
            let mut output = Vec::new();
            assert!(
                runtime
                    .run_complete_generation_flush_with_writer(&cx, &root, &mut output)
                    .is_err()
            );
            assert!(output.is_empty());
            drop(pending);
            cx.set_cancel_requested(true);
            assert!(matches!(
                runtime.run_complete_generation_flush_with_writer(&cx, &root, &mut output),
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            assert!(output.is_empty());
            assert_eq!(store.active(&cx).unwrap(), selected);
            assert_eq!(
                fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                pointer
            );
            runtime
                .run_complete_generation_flush_with_writer(&cx, &root, &mut output)
                .unwrap();
            let receipt: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(receipt["data"]["durability"], "confirmed");
        });
    }

    #[test]
    fn complete_cli_flush_output_failure_does_not_undo_the_durable_selection() {
        struct BrokenWriter;
        impl Write for BrokenWriter {
            fn write(&mut self, _bytes: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(ErrorKind::BrokenPipe, "closed output"))
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (mut runtime, _, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            runtime.cli_input.format = OutputFormat::Table;
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let selected = store.active(&cx).unwrap();
            let pointer = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            assert!(matches!(
                runtime.run_complete_generation_flush_with_writer(&cx, &root, &mut BrokenWriter),
                Err(SearchError::Io(error)) if error.kind() == ErrorKind::BrokenPipe
            ));
            assert_eq!(store.active(&cx).unwrap(), selected);
            assert_eq!(
                fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                pointer
            );
        });
    }

    #[test]
    fn complete_cli_flush_confirms_selected_bundle_and_preserves_pinned_readers() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let before = store.active(&cx).unwrap().unwrap();
            let inventory = sealed_inventory(before.path());
            let pointer = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            let mut reader = runtime.open_retained_search(&cx, &root).await.unwrap();
            let hits_before = reader.search(&cx, "sharedtoken", 10).await.unwrap();
            let mut input = runtime.cli_input.clone();
            input.command = CliCommand::Flush;
            let command = runtime.clone().with_cli_input(input);
            let mut output = Vec::new();
            command
                .run_complete_generation_flush_with_writer(&cx, &root, &mut output)
                .unwrap();
            let receipt: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(receipt["ok"], true);
            assert_eq!(receipt["data"]["generation_id"], before.id());
            assert_eq!(receipt["data"]["manifest_sha256"], before.manifest_sha256());
            assert_eq!(receipt["data"]["durability"], "confirmed");
            assert_eq!(receipt["data"]["scope"], "selected_generation");
            assert_eq!(receipt["data"]["generation_changed"], false);
            assert_eq!(receipt["data"]["watch_queue_drained"], false);
            // Exercise the public dispatcher too: the selected store must no
            // longer hit the legacy writer path or the unsupported-command arm.
            command
                .run_mode_with_complete_generations(&cx, InterfaceMode::Cli, None, false)
                .await
                .unwrap();
            assert_eq!(sealed_inventory(before.path()), inventory);
            assert_eq!(
                fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                pointer
            );
            assert_eq!(store.active(&cx).unwrap(), Some(before));
            let hits_after = reader.search(&cx, "sharedtoken", 10).await.unwrap();
            assert_eq!(
                hits_after.last().unwrap().hits,
                hits_before.last().unwrap().hits
            );
        });
    }

    #[test]
    fn complete_cli_flush_and_failed_readers_never_adopt_an_unpublished_lexical_child() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            let store = CompleteGenerationStore::create(&cx, &root).unwrap();
            let candidate = store.begin(&cx).unwrap();
            let lexical = candidate.path().join("lexical");
            fs::create_dir_all(lexical.join("quill-v1")).unwrap();
            fs::write(lexical.join("quill-v1/MANIFEST"), b"adoption marker").unwrap();
            // A generic complete store validates inventory, while engine
            // admission belongs to the consumer. Plant exactly the layout
            // that the legacy resolver would adopt by writing a new CURRENT.
            let selected =
                require_durable_publication(candidate.publish(&cx, |_, _| Ok(())).unwrap())
                    .unwrap();
            let before = sealed_inventory(selected.path());
            let mut output = Vec::new();
            runtime
                .run_complete_generation_flush_with_writer(&cx, &root, &mut output)
                .unwrap();
            let receipt: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(receipt["ok"], true);
            assert_eq!(receipt["data"]["generation_id"], selected.id());
            assert_eq!(receipt["data"]["durability"], "confirmed");
            assert_eq!(receipt["data"]["scope"], "selected_generation");
            assert_eq!(receipt["data"]["generation_changed"], false);
            assert_eq!(sealed_inventory(selected.path()), before);
            assert!(!selected.path().join("lexical/CURRENT").exists());
            let error = runtime.open_retained_search(&cx, &root).await.unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                if field == "complete_generation.lexical_layout"));
            let mut input = runtime.cli_input.clone();
            input.index_dir = Some(selected.path().to_path_buf());
            let error = runtime
                .clone()
                .with_cli_input(input)
                .collect_status_payload()
                .unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                if field == "complete_generation.lexical_layout"));
            assert_eq!(sealed_inventory(selected.path()), before);
            assert_eq!(store.active(&cx).unwrap(), Some(selected));
        });
    }

    #[test]
    fn complete_cli_append_batch_publishes_truthful_receipts_and_preserves_old_readers() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let predecessor = store.active(&cx).unwrap().unwrap();
            let mut old_reader = runtime.open_retained_search(&cx, &root).await.unwrap();
            let batch = directory.path().join("batch.jsonl");
            fs::write(
                &batch,
                concat!(
                    "{\"id\":\"alpha.md\",\"text\":\"sharedtoken rewritten alpha\"}\n",
                    "{\"id\":\"beta.md\",\"text\":\"sharedtoken supplied beta\"}\n",
                ),
            )
            .unwrap();
            let mut input = runtime.cli_input.clone();
            input.command = CliCommand::AppendBatch;
            input.input_file = Some(batch.clone());
            let append = runtime.clone().with_cli_input(input);
            let mut output = Vec::new();
            append
                .run_complete_generation_append_batch_with_writer(&cx, &root, &mut output)
                .await
                .unwrap();
            let receipt: serde_json::Value = serde_json::from_slice(&output).unwrap();
            let successor = store.active(&cx).unwrap().unwrap();
            assert_eq!(receipt["ok"], true);
            assert_eq!(receipt["data"]["appended"], 2);
            assert_eq!(receipt["data"]["generation_changed"], true);
            assert_eq!(receipt["data"]["publication"], "durable");
            assert_eq!(receipt["data"]["generation_id"], successor.id());
            assert_ne!(successor.id(), predecessor.id());
            let mut fresh = runtime.open_retained_search(&cx, &root).await.unwrap();
            assert_eq!(
                fresh
                    .search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .len(),
                2
            );
            assert_eq!(
                old_reader
                    .search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .len(),
                1
            );
            assert_eq!(
                fs::read_to_string(source.join("alpha.md")).unwrap(),
                "sharedtoken alpha document"
            );
            assert!(!source.join("beta.md").exists());

            fs::write(&batch, " \n\t\n").unwrap();
            output.clear();
            append
                .run_complete_generation_append_batch_with_writer(&cx, &root, &mut output)
                .await
                .unwrap();
            let receipt: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(receipt["data"]["appended"], 0);
            assert_eq!(receipt["data"]["generation_changed"], false);
            assert_eq!(store.active(&cx).unwrap(), Some(successor.clone()));

            fs::write(
                &batch,
                "{\"id\":\"unpublished.md\",\"text\":\"sharedtoken rejected batch\"}\ninvalid JSON\n",
            )
            .unwrap();
            output.clear();
            append
                .run_complete_generation_append_batch_with_writer(&cx, &root, &mut output)
                .await
                .unwrap_err();
            assert!(output.is_empty());
            assert_eq!(store.active(&cx).unwrap(), Some(successor));
        });
    }

    #[test]
    fn complete_cli_mutation_receipts_match_selection_and_preserve_old_readers() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let predecessor = store.active(&cx).unwrap().unwrap();
            let inventory =
                fs::read(predecessor.path().join(COMPLETE_GENERATION_MANIFEST)).unwrap();
            let mut old_reader = runtime.open_retained_search(&cx, &root).await.unwrap();
            let mut input = runtime.cli_input.clone();
            input.command = CliCommand::Delete;
            input.delete_ids = vec!["missing.md".to_owned()];
            let mut delete = runtime.clone().with_cli_input(input);
            let mut output = Vec::new();
            delete
                .run_complete_generation_delete_with_writer(&cx, &root, &mut output)
                .await
                .unwrap();
            let receipt: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(receipt["ok"], true);
            assert_eq!(receipt["data"]["deleted"], 0);
            assert_eq!(receipt["data"]["generation_changed"], false);
            assert_eq!(store.active(&cx).unwrap(), Some(predecessor.clone()));

            delete.cli_input.delete_ids = vec!["alpha.md".to_owned()];
            output.clear();
            delete
                .run_complete_generation_delete_with_writer(&cx, &root, &mut output)
                .await
                .unwrap();
            let receipt: serde_json::Value = serde_json::from_slice(&output).unwrap();
            let deleted = store.active(&cx).unwrap().unwrap();
            assert_eq!(receipt["data"]["deleted"], 1);
            assert_eq!(receipt["data"]["generation_changed"], true);
            assert_eq!(receipt["data"]["publication"], "durable");
            assert_eq!(receipt["data"]["generation_id"], deleted.id());
            assert_ne!(deleted.id(), predecessor.id());
            assert!(source.join("alpha.md").is_file());

            output.clear();
            runtime
                .run_complete_generation_compact_with_writer(&cx, &root, &mut output)
                .await
                .unwrap();
            let receipt: serde_json::Value = serde_json::from_slice(&output).unwrap();
            let compacted = store.active(&cx).unwrap().unwrap();
            assert_eq!(receipt["ok"], true);
            assert_eq!(receipt["data"]["publication"], "durable");
            assert_eq!(receipt["data"]["generation_id"], compacted.id());
            assert_ne!(compacted.id(), deleted.id());
            let mut fresh = runtime.open_retained_search(&cx, &root).await.unwrap();
            assert!(
                fresh
                    .search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .is_empty()
            );
            assert_eq!(
                old_reader
                    .search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .len(),
                1
            );
            assert_eq!(
                fs::read(predecessor.path().join(COMPLETE_GENERATION_MANIFEST)).unwrap(),
                inventory
            );
        });
    }
}

#[cfg(unix)]
#[path = "complete_daemon.rs"]
mod complete_daemon;
