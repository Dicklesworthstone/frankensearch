//! Explicit hybrid command admission and progressive NDJSON transport.
//!
//! Retrieval and phase ordering stay in `RetainedLiveSearchSession`. This module
//! only selects that existing pipeline and acknowledges fully flushed frames.

use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use frankensearch_fsfs::adapters::quill_live_search::QuillLiveSearchSession;
use frankensearch_fsfs::adapters::retained_live_search::{
    RetainedLiveSearchFrame, RetainedLiveSearchSession,
};
use frankensearch_fsfs::generation_store::CompleteGenerationStore;
use frankensearch_fsfs::{
    CliCommand, CliInput, CliOverrides, FsfsRuntime, OutputFormat, current_unicode_environment,
    default_project_config_file_path, default_user_config_file_path, load_from_layered_sources,
    load_from_sources,
};
use frankensearch_quill::QuillConfig;
use serde::Serialize;

use super::{Options, invalid};

const HYBRID_SCHEMA_VERSION: &str = "fsfs.stream.live_search.hybrid.v1";
const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

pub(super) enum Subscription {
    Lexical(Box<QuillLiveSearchSession>),
    Hybrid(Box<RetainedLiveSearchSession>),
}

impl Subscription {
    pub(super) fn new(
        store: CompleteGenerationStore,
        options: &Options,
        runtime: Option<FsfsRuntime>,
    ) -> SearchResult<Self> {
        match (options.hybrid, runtime) {
            (true, Some(runtime)) => Ok(Self::Hybrid(Box::new(RetainedLiveSearchSession::new(
                runtime,
                store,
                options.query.clone(),
                options.limits,
                options.refresh,
            )?))),
            (false, None) => Ok(Self::Lexical(Box::new(QuillLiveSearchSession::new(
                store,
                options.query.clone(),
                options.limits,
                options.refresh,
                QuillConfig::default(),
            )?))),
            _ => Err(invalid(
                "subscription mode does not match its admitted runtime",
            )),
        }
    }

    /// One update is one physical generation's complete phase sequence, not one
    /// phase frame. A max-update limit cannot turn Initial into a final result by
    /// cutting off refinement. Cancellation or failed output still stops promptly.
    pub(super) async fn poll_ndjson<W: Write + Send>(
        &mut self,
        cx: &Cx,
        now: Instant,
        writer: &mut W,
    ) -> SearchResult<bool> {
        match self {
            Self::Lexical(session) => Ok(session.poll_ndjson(cx, now, writer).await?.is_some()),
            Self::Hybrid(session) => {
                let mut sink = |frame: &RetainedLiveSearchFrame| emit_frame(frame, writer);
                Ok(session.poll_with_sink(cx, now, &mut sink).await? != 0)
            }
        }
    }
}

/// Lexical mode returns before inspecting environment/configuration. Hybrid mode
/// uses the standard policy loader but explicitly disables model downloads and
/// indexing. This does not certify semantic readiness: the real pipeline must
/// admit each producer and preserve any typed degradation/failure annotations.
pub(super) fn configured_runtime(options: &Options) -> SearchResult<Option<FsfsRuntime>> {
    if !options.hybrid {
        return Ok(None);
    }
    let home = frankensearch_core::platform_dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"));
    let env = current_unicode_environment();
    let overrides = CliOverrides::default();
    let loaded = if let Some(path) = &options.config_path {
        let path = super::super::expand_cli_config_path(path, &home);
        if !std::fs::metadata(&path)?.is_file() {
            return Err(invalid("--config must name a regular configuration file"));
        }
        load_from_sources(Some(&path), &env, &overrides, &home)?
    } else {
        let project = default_project_config_file_path(&std::env::current_dir()?);
        let user = default_user_config_file_path(&home);
        load_from_layered_sources(Some(&project), Some(&user), &env, &overrides, &home)?
    };
    let mut config = loaded.config;
    config.indexing.offline = true;
    config.indexing.watch_mode = false;
    Ok(Some(FsfsRuntime::new(config).with_cli_input(CliInput {
        command: CliCommand::Search,
        index_dir: Some(options.root.clone()),
        format: OutputFormat::Jsonl,
        quiet: true,
        no_color: true,
        ..CliInput::default()
    })))
}

#[derive(Serialize)]
struct Envelope<'a> {
    schema_version: &'static str,
    #[serde(flatten)]
    frame: &'a RetainedLiveSearchFrame,
}

struct BoundedFrame {
    bytes: Vec<u8>,
    limit: usize,
}

impl Write for BoundedFrame {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes
            .len()
            .checked_add(bytes.len())
            .filter(|length| *length <= self.limit)
            .ok_or_else(|| io::Error::other("hybrid live-search frame exceeds its byte limit"))?;
        self.bytes
            .try_reserve(bytes.len())
            .map_err(|_| io::Error::other("cannot reserve hybrid live-search output"))?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn encode_frame(frame: &RetainedLiveSearchFrame, limit: usize) -> SearchResult<Vec<u8>> {
    let mut output = BoundedFrame {
        bytes: Vec::new(),
        limit,
    };
    serde_json::to_writer(
        &mut output,
        &Envelope {
            schema_version: HYBRID_SCHEMA_VERSION,
            frame,
        },
    )
    .map_err(|source| SearchError::SubsystemError {
        subsystem: "fsfs.live_search.hybrid_output",
        source: Box::new(source),
    })?;
    output.write_all(b"\n")?;
    Ok(output.bytes)
}

fn emit_frame<W: Write>(frame: &RetainedLiveSearchFrame, writer: &mut W) -> SearchResult<()> {
    // Finish serialization (including its size check) before exposing any bytes.
    // The caller's GuardedOutput checks cancellation/budget before each syscall.
    let bytes = encode_frame(frame, MAX_FRAME_BYTES)?;
    writer.write_all(&bytes)?;
    writer.flush()?;
    // No fallible work follows acknowledgment. The retained session advances its
    // baseline only after this returns Ok; output failures are never retried.
    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use frankensearch_fsfs::FsfsConfig;
    use frankensearch_fsfs::adapters::live_search::{
        LiveSearchConfig, LiveSearchEvent, LiveSearchHit, LiveSearchTracker,
    };
    use frankensearch_fsfs::output_schema::SearchOutputPhase;
    use serde_json::{Value, json};

    use super::*;

    fn options(root: &std::path::Path) -> Options {
        Options::parse(vec![
            "--hybrid".into(),
            "--index-dir".into(),
            root.as_os_str().to_owned(),
            "--query".into(),
            "alpha".into(),
            "--once".into(),
        ])
        .unwrap()
    }

    fn frames() -> [RetainedLiveSearchFrame; 2] {
        let mut tracker = LiveSearchTracker::new("alpha", LiveSearchConfig::default()).unwrap();
        let mut frame = |phase: SearchOutputPhase, id: &str| RetainedLiveSearchFrame {
            generation_id: "physical-generation".to_owned(),
            manifest_sha256: "a".repeat(64),
            phase,
            annotations: json!({"query": "alpha", "phase": phase, "semantic_admitted": true}),
            update: tracker
                .apply(
                    &format!("physical-generation@digest:{phase}"),
                    vec![LiveSearchHit {
                        doc_id: id.to_owned(),
                        score: 0.5,
                        item: json!({"path": id, "snippet": "alpha\nquoted \"text\""}),
                    }],
                )
                .unwrap(),
        };
        [
            frame(SearchOutputPhase::Initial, "a.rs"),
            frame(SearchOutputPhase::Refined, "b.rs"),
        ]
    }

    #[test]
    fn progressive_transport_preserves_receipt_phase_and_delta_chain() {
        let frames = frames();
        let mut bytes = Vec::new();
        emit_frame(&frames[0], &mut bytes).unwrap();
        emit_frame(&frames[1], &mut bytes).unwrap();
        let records = bytes
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["schema_version"], HYBRID_SCHEMA_VERSION);
        for (record, expected) in records.iter().zip(&frames) {
            let decoded: RetainedLiveSearchFrame = serde_json::from_value(record.clone()).unwrap();
            assert_eq!(&decoded, expected);
        }
        assert!(matches!(
            &frames[0].update.as_ref().unwrap().event,
            LiveSearchEvent::Snapshot { .. }
        ));
        assert_eq!(
            records[1]["update"]["previous_generation"],
            records[0]["update"]["generation"]
        );
        assert_eq!(records[0]["generation_id"], records[1]["generation_id"]);
        assert_ne!(
            records[0]["update"]["generation"],
            records[1]["update"]["generation"]
        );
    }

    #[test]
    fn refinement_failure_is_not_an_empty_result_delta() {
        let frame = RetainedLiveSearchFrame {
            generation_id: "retained".to_owned(),
            manifest_sha256: "b".repeat(64),
            phase: SearchOutputPhase::RefinementFailed,
            annotations: json!({"skip_reason": "quality_timeout", "rerank_applied": false}),
            update: None,
        };
        let bytes = encode_frame(&frame, MAX_FRAME_BYTES).unwrap();
        let decoded: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded["phase"], "refinement_failed");
        assert!(decoded["update"].is_null());
        assert_eq!(decoded["annotations"]["skip_reason"], "quality_timeout");
        assert_eq!(decoded["annotations"]["rerank_applied"], false);
    }

    #[test]
    fn serialization_bound_includes_the_record_separator() {
        let frame = &frames()[0];
        let encoded = encode_frame(frame, MAX_FRAME_BYTES).unwrap();
        assert_eq!(encode_frame(frame, encoded.len()).unwrap(), encoded);
        assert!(encode_frame(frame, encoded.len() - 1).is_err());
        assert!(encode_frame(frame, 0).is_err());
    }

    struct FailOutput {
        bytes: Vec<u8>,
        writes: usize,
        flushes: usize,
        fail_write: bool,
    }

    impl Write for FailOutput {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            if self.fail_write {
                self.bytes.extend_from_slice(&bytes[..1]);
                return Err(io::Error::from(io::ErrorKind::BrokenPipe));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            self.flushes += 1;
            Err(io::Error::other("injected flush failure"))
        }
    }

    #[test]
    fn transport_failures_preserve_the_io_error_and_are_never_retried() {
        for fail_write in [true, false] {
            let mut output = FailOutput {
                bytes: Vec::new(),
                writes: 0,
                flushes: 0,
                fail_write,
            };
            let error = emit_frame(&frames()[0], &mut output).unwrap_err();
            let SearchError::Io(error) = error else {
                panic!("transport failure lost its original I/O type");
            };
            assert_eq!(output.writes, 1);
            if fail_write {
                assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
                assert_eq!(output.bytes.len(), 1);
                assert_eq!(output.flushes, 0);
            } else {
                assert_eq!(error.kind(), io::ErrorKind::Other);
                assert_eq!(output.flushes, 1);
                assert!(serde_json::from_slice::<Value>(&output.bytes).is_ok());
            }
        }
    }

    #[test]
    fn oversized_payload_is_refused_before_transport() {
        let mut frame = frames()[0].clone();
        frame.annotations = json!({"oversized": "x".repeat(MAX_FRAME_BYTES)});
        let mut output = Vec::new();
        assert!(emit_frame(&frame, &mut output).is_err());
        assert!(output.is_empty());
    }

    #[test]
    fn lexical_mode_never_opens_a_configuration_file() {
        let root = tempfile::tempdir().unwrap();
        let mut options = options(root.path());
        options.hybrid = false;
        // The parser refuses this combination; the boundary also does no I/O
        // if options were constructed directly by an internal caller.
        options.config_path = Some(root.path().join("does-not-exist"));
        assert!(configured_runtime(&options).unwrap().is_none());
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    }

    #[test]
    fn explicit_hybrid_configuration_preserves_search_policy_and_disables_indexing() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("fsfs.toml");
        std::fs::write(
            &path,
            "[search]\nfast_only = true\n[indexing]\nwatch_mode = true\noffline = false\n",
        )
        .unwrap();
        let mut options = options(root.path());
        options.config_path = Some(path);
        let runtime = configured_runtime(&options).unwrap().unwrap();
        assert!(runtime.config().search.fast_only);
        assert!(runtime.config().indexing.offline);
        assert!(!runtime.config().indexing.watch_mode);
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
    }

    #[test]
    fn hybrid_rejects_missing_or_directory_configuration() {
        let root = tempfile::tempdir().unwrap();
        let mut options = options(root.path());
        for path in [root.path().to_path_buf(), root.path().join("missing")] {
            options.config_path = Some(path);
            assert!(configured_runtime(&options).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn hybrid_dispatch_waits_without_models_and_honors_cancellation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let mut session = Subscription::new(
                store,
                &options(root.path()),
                Some(FsfsRuntime::new(FsfsConfig::default())),
            )
            .unwrap();
            let mut bytes = Vec::new();
            assert!(
                !session
                    .poll_ndjson(&cx, Instant::now(), &mut bytes)
                    .await
                    .unwrap()
            );
            cx.set_cancel_requested(true);
            assert!(matches!(
                session.poll_ndjson(&cx, Instant::now(), &mut bytes).await,
                Err(SearchError::Cancelled { .. })
            ));
            assert!(bytes.is_empty());
            assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
        });
    }

    #[cfg(unix)]
    #[test]
    fn hybrid_dispatch_requires_its_runtime_instead_of_silently_using_lexical() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            assert!(Subscription::new(store, &options(root.path()), None).is_err());
        });
    }

    #[cfg(unix)]
    #[test]
    fn hybrid_command_deadline_applies_while_waiting_for_the_first_generation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let _store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let mut options = options(root.path());
            options.once = false;
            options.timeout = Some(std::time::Duration::from_millis(10));
            options.poll_interval = std::time::Duration::from_secs(60);
            let mut bytes = Vec::new();
            let outcome = super::super::execute_with_runtime(
                &cx,
                &options,
                &mut bytes,
                Some(FsfsRuntime::new(FsfsConfig::default())),
            )
            .await;
            assert!(matches!(outcome, Err(SearchError::SearchTimeout { .. })));
            assert!(bytes.is_empty());
        });
    }

    #[cfg(unix)]
    #[test]
    fn hybrid_arguments_keep_non_utf8_path_support() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let path = OsString::from_vec(b"config-\xff.toml".to_vec());
        let args = vec![
            "--hybrid".into(),
            "--config".into(),
            path.clone(),
            "--index-dir".into(),
            "/store".into(),
            "--query".into(),
            "alpha".into(),
        ];
        let options = Options::parse(args).unwrap();
        assert_eq!(options.config_path, Some(PathBuf::from(path)));
    }
}
