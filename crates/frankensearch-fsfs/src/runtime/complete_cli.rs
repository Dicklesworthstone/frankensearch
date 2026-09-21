//! Complete-generation command routing. A selected store must never fall back
//! to the legacy mutable layout, including when its selection is corrupt.

use std::fs;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};

use super::{
    FsfsRuntime, InterfaceMode, SearchExecutionFlags, iso_timestamp_now, pressure_timestamp_ms,
    retained_search_checkpoint, validate_retained_catalog_path,
};
#[cfg(unix)]
use super::{FSFS_DAEMON_REQUEST_MAX_BYTES, SearchServeFrameBuffer};
use crate::adapters::format_emitter::{emit_envelope, meta_for_format};
use crate::generation_store::{
    COMPLETE_GENERATION_MANIFEST, COMPLETE_GENERATION_POINTER, CompleteGenerationStore,
    GenerationPublication,
};
use crate::output_schema::OutputEnvelope;
use crate::{CliCommand, OutputFormat, ShutdownCoordinator};

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
    #[allow(clippy::future_not_send)]
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
        if mode != InterfaceMode::Cli {
            return Err(complete_cli_error(
                "interface",
                "complete-generation stores require a CLI command; TUI routing is not yet supported",
            ));
        }
        self.validate_command_inputs(self.cli_input.command)?;
        let _cancellation_scope = shutdown.map(|shutdown| shutdown.cancellation_scope(cx));
        retained_search_checkpoint(cx)?;
        validate_retained_catalog_path(&self.config.storage.db_path)?;
        let mut stdout = std::io::stdout();
        match self.cli_input.command {
            CliCommand::Index if !self.cli_input.watch && !self.config.indexing.watch_mode => {
                self.run_complete_generation_index_with_writer(cx, &root, &mut stdout)
                    .await
            }
            CliCommand::Search => {
                self.run_complete_generation_search_with_writer(cx, &root, &mut stdout)
                    .await
            }
            #[cfg(unix)]
            CliCommand::Serve => self.run_complete_generation_serve(cx, &root).await,
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
                "this command cannot mutate a complete-generation store in place; use a one-shot index rebuild, direct search, stdio serve, status, or doctor",
            )),
        }
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
        let root = if matches!(self.cli_input.command, CliCommand::Index | CliCommand::Watch) {
            self.resolve_index_root(&self.resolve_target_root()?)?
        } else {
            self.resolve_status_index_root()?
        };
        if complete_entry_exists(&root.join(COMPLETE_GENERATION_MANIFEST))? {
            return Err(complete_cli_error(
                "root",
                "a sealed generation is not a mutable index root; select its complete-generation store instead",
            ));
        }
        // Check directory entries, not Path::exists: a dangling symlink or an
        // unreadable descriptor is not permission to enter the legacy path.
        // A staging tree also prevents fallback after an interrupted first build
        // or after the active descriptor has been removed out of protocol.
        let selected = complete_entry_exists(&root.join(COMPLETE_GENERATION_POINTER))?
            || complete_entry_exists(&root.join("generations"))?;
        Ok((initialize_store || selected).then_some(root))
    }

    #[allow(clippy::future_not_send)]
    async fn run_complete_generation_index_with_writer<W: Write>(
        &self,
        cx: &Cx,
        root: &Path,
        writer: &mut W,
    ) -> SearchResult<()> {
        let generation = match self.rebuild_retained_generation(cx, root).await? {
            GenerationPublication::Durable(generation) => generation,
            GenerationPublication::VisibleButDurabilityUncertain { generation, source } => {
                // The rename already happened. In particular, do not label this
                // an aborted rebuild or try to restore the predecessor pointer.
                return Err(SearchError::SubsystemError {
                    subsystem: "fsfs.complete_generation.durability",
                    source: Box::new(std::io::Error::other(format!(
                        "generation {} is already visible at {}, but directory synchronization failed: {source}",
                        generation.id(),
                        generation.path().display(),
                    ))),
                });
            }
        };
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
                meta_for_format("index", self.cli_input.format),
                iso_timestamp_now(),
            );
            emit_envelope(&envelope, self.cli_input.format, writer)?;
            if !matches!(self.cli_input.format, OutputFormat::Jsonl | OutputFormat::Csv) {
                writer.write_all(b"\n")?;
            }
        }
        writer.flush().map_err(SearchError::Io)
    }

    async fn run_complete_generation_search_with_writer<W: Write + Send>(
        &self,
        cx: &Cx,
        root: &Path,
        writer: &mut W,
    ) -> SearchResult<()> {
        if self.cli_input.daemon
            || self.cli_input.daemon_socket.is_some()
            || self.cli_input.expand
        {
            return Err(complete_cli_error(
                "search_options",
                "complete-generation CLI search currently requires --no-daemon and does not support --expand; no fallback was attempted",
            ));
        }
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
            && !matches!(self.cli_input.format, OutputFormat::Jsonl | OutputFormat::Toon)
        {
            return Err(complete_cli_error(
                "format",
                "stream mode requires --format jsonl or --format toon",
            ));
        }
        let started = Instant::now();
        let mut reader = self.open_retained_search(cx, root).await?;
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
                            persist_explain_session: false,
                        },
                    )),
                )
                .await;
        }
        let payloads = reader.search(cx, query, limit).await?;
        let payload = payloads.last().cloned().ok_or_else(|| {
            complete_cli_error("search", "search completed without an Initial phase")
        })?;
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
            emit_envelope(&envelope, self.cli_input.format, writer)?;
            if !matches!(self.cli_input.format, OutputFormat::Jsonl | OutputFormat::Csv) {
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
            return Err(complete_cli_error(
                "serve_transport",
                "complete-generation serve currently supports stdin/stdout only; no socket daemon fallback was attempted",
            ));
        }
        let stdin = std::io::stdin();
        let mut input = stdin.lock();
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

        let mut live = self.open_live_retained_search(cx, root).await?;
        let mut cache = HashMap::new();
        let ready = Self::search_serve_ready_event(
            self.cli_input.format.to_string(),
            &live.reader.resources,
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
                return Err(complete_cli_error("serve_request", "request exceeds 1 MiB limit"));
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
                if live.refresh(cx).await? {
                    cache.clear();
                }
                // Both the runtime's hydration paths and the resources belong
                // to the same admitted generation. Never invoke this handler
                // with the outer store-root runtime or after a failed refresh.
                live.reader
                    .runtime
                    .execute_search_serve_request(
                        cx,
                        request,
                        &mut live.reader.resources,
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
        complete_cli_error("serve_response", &format!("cannot encode response: {error}"))
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

fn complete_cli_error(field: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("complete_generation.{field}"),
        value: String::new(),
        reason: reason.to_owned(),
    }
}

#[cfg(all(test, unix, not(feature = "embedded-models")))]
mod tests {
    use super::*;
    use super::super::FSFS_EXPLAIN_SESSION_FILE;
    use asupersync::test_utils::run_test_with_cx;
    use crate::output_schema::SearchHitPayload;
    use crate::stream_protocol::StreamFrame;
    use crate::{CliInput, FsfsConfig};

    fn fixture(parent: &Path) -> (FsfsRuntime, PathBuf, PathBuf) {
        let source = parent.join("source");
        let root = parent.join("store");
        fs::create_dir(&source).unwrap();
        fs::write(source.join("alpha.md"), "sharedtoken alpha document").unwrap();
        let mut config = FsfsConfig::default();
        config.storage.db_path = "{index_dir}/catalog.sqlite".to_owned();
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
            if self.switches.front().is_some_and(|(after, _)| *after == self.flushes) {
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
            let mut input = std::io::Cursor::new(
                b"sharedtoken\nsharedtoken\nsharedtoken\nsharedtoken\nquit\n",
            );
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
                assert_eq!(phases.last().unwrap()["hits"].as_array().unwrap().len(), count);
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
            assert_eq!(serve_lines(&output).len(), 1, "only the ready event is visible");
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
    fn complete_cli_refuses_legacy_mutators_without_touching_the_selected_bundle() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let before = store.active(&cx).unwrap();
            for command in [CliCommand::Compact, CliCommand::Flush, CliCommand::Daemon] {
                let mut input = runtime.cli_input.clone();
                input.command = command;
                runtime
                    .clone()
                    .with_cli_input(input)
                    .run_mode_with_complete_generations(&cx, InterfaceMode::Cli, None, false)
                    .await
                    .unwrap_err();
                assert_eq!(store.active(&cx).unwrap(), before);
            }
        });
    }
}
