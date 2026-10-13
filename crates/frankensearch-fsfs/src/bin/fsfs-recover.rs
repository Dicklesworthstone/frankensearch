//! Receipt-directed complete-store recovery, without normal CLI root discovery.
//! Inspection is the default; only an explicit --apply can change selection.
//! Vector reconstruction additionally requires --repair-vectors; originals stay intact.

#![recursion_limit = "512"]
#![deny(unsafe_code)]

use std::collections::HashSet;
use std::ffi::OsString;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use asupersync::Cx;
use asupersync::runtime::RuntimeBuilder;
use frankensearch_core::rfc3339::{format_unix_nanos, now_unix_nanos};
use frankensearch_core::{SearchError, SearchResult};
use frankensearch_fsfs::generation_store::{
    CompleteGenerationStore, GenerationPublication, PublishedGeneration,
};
use frankensearch_fsfs::output_schema::{SearchOutputPhase, SearchPayload};
use frankensearch_fsfs::runtime::SearchBlockingPool;
use frankensearch_fsfs::{
    CliCommand, CliInput, CliOverrides, FsfsRuntime, OutputEnvelope, OutputError, OutputFormat,
    ShutdownCoordinator, Verbosity, current_unicode_environment, default_project_config_file_path,
    default_user_config_file_path, exit_code_for, init_subscriber, load_from_layered_sources,
    load_from_sources, meta_for_format, output_error_from,
};
use serde::Serialize;

const MAX_QUERY_BYTES: usize = 64 * 1024;
const MAX_REPORT_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_REPAIR_COPY_BYTES: u64 = 1024 * 1024 * 1024;
const HELP: &str =
    "fsfs-recover --index-dir STORE --generation ID --manifest-sha256 SHA256 [OPTIONS]

Inspect an explicitly receipted complete generation, even if FSFS-CURRENT is damaged.
No directory scan, source rebuild, model download or automatic fallback is performed.

  --config FILE       Use the normal fsfs configuration loader and model selection
  --query TEXT        Preview a query against the target before any restoration
  --limit N           Preview results per phase (1..=100; default 10)
  --apply             Explicitly restore the admitted target; default is read-only inspection
  --repair-vectors    With --apply, reconstruct damaged vectors in a NEW copied generation
  --max-copy-bytes N  Repair payload copy limit (positive integer; default 1073741824)
  --confirm-durability Confirm the exact selected target without models or republication
  --format FORMAT     json (default), jsonl, or table
  --help              Show this help (use alone)

Use the generation ID and manifest SHA-256 from a trusted publication receipt.
All stored vector tiers must admit their configured producers, even in fast-only mode.
A competing publication refuses restoration; no attempt is automatically retried.
Predecessors and damaged bundles are retained. No file is deleted.
Repair requires intact receipted FEC, catalog, lexical and metadata artifacts (Unix only).
Each damaged vector is limited to 128 MiB and its sidecar to 256 MiB; copy limits are not RSS limits.
Failed partial copies remain unselected. Repaired results report a NEW receipt plus repaired_from.
--max-copy-bytes requires --repair-vectors; inspection never creates a repair copy.
--confirm-durability cannot be combined with --apply, --config, --query, or --limit.
It also cannot be combined with repair options; confirmation performs no reconstruction.
Durability confirmation is not model admission or a drained watcher queue.
This operates on cooperative complete stores, not the fixed-authority antirollback layout.
";

#[derive(Clone, Debug)]
struct Options {
    root: PathBuf,
    generation: String,
    digest: String,
    config: Option<PathBuf>,
    query: Option<String>,
    limit: usize,
    apply: bool,
    repair_vectors: bool,
    max_copy_bytes: u64,
    confirm_durability: bool,
    format: OutputFormat,
}

fn invalid(reason: impl Into<String>) -> SearchError {
    SearchError::InvalidConfig {
        field: "recovery.arguments".to_owned(),
        value: String::new(),
        reason: reason.into(),
    }
}

fn text(value: OsString) -> SearchResult<String> {
    value
        .into_string()
        .map_err(|_| invalid("expected UTF-8 text"))
}

impl Options {
    fn parse(args: Vec<OsString>) -> SearchResult<Self> {
        let mut root = None;
        let mut generation = None;
        let mut digest = None;
        let mut config = None;
        let mut query = None;
        let mut limit = 10;
        let mut apply = false;
        let mut repair_vectors = false;
        let mut max_copy_bytes = DEFAULT_REPAIR_COPY_BYTES;
        let mut confirm_durability = false;
        let mut format = OutputFormat::Json;
        let mut seen = HashSet::new();
        let mut args = args.into_iter();
        while let Some(flag) = args.next() {
            let flag = text(flag)?;
            if !matches!(
                flag.as_str(),
                "--index-dir"
                    | "--generation"
                    | "--manifest-sha256"
                    | "--config"
                    | "--query"
                    | "--limit"
                    | "--apply"
                    | "--repair-vectors"
                    | "--max-copy-bytes"
                    | "--confirm-durability"
                    | "--format"
            ) {
                return Err(invalid("unknown argument; run fsfs-recover --help"));
            }
            if !seen.insert(flag.clone()) {
                return Err(invalid(format!("duplicate {flag} is not allowed")));
            }
            if flag == "--apply" {
                apply = true;
                continue;
            }
            if flag == "--confirm-durability" {
                confirm_durability = true;
                continue;
            }
            if flag == "--repair-vectors" {
                repair_vectors = true;
                continue;
            }
            let value = args
                .next()
                .ok_or_else(|| invalid(format!("missing value for {flag}")))?;
            if value.is_empty() {
                return Err(invalid(format!("empty value for {flag}")));
            }
            match flag.as_str() {
                "--index-dir" => root = Some(PathBuf::from(value)),
                "--config" => config = Some(PathBuf::from(value)),
                "--generation" => generation = Some(text(value)?),
                "--manifest-sha256" => digest = Some(text(value)?),
                "--query" => query = Some(text(value)?),
                "--max-copy-bytes" => {
                    let value = text(value)?;
                    max_copy_bytes = value
                        .bytes()
                        .all(|byte| byte.is_ascii_digit())
                        .then(|| value.parse::<u64>().ok())
                        .flatten()
                        .filter(|bytes| *bytes > 0)
                        .ok_or_else(|| invalid("--max-copy-bytes must be a positive u64 decimal integer"))?;
                }
                "--limit" => {
                    limit = text(value)?
                        .parse::<usize>()
                        .ok()
                        .filter(|value| (1..=100).contains(value))
                        .ok_or_else(|| invalid("--limit must be between 1 and 100"))?;
                }
                "--format" => {
                    format = match text(value)?.as_str() {
                        "json" => OutputFormat::Json,
                        "jsonl" => OutputFormat::Jsonl,
                        "table" => OutputFormat::Table,
                        _ => return Err(invalid("--format must be json, jsonl, or table")),
                    };
                }
                _ => return Err(invalid("invalid value-bearing argument")),
            }
        }
        Self {
            root: root.ok_or_else(|| {
                invalid("--index-dir is required; recovery never guesses a store")
            })?,
            generation: generation.ok_or_else(|| invalid("--generation is required"))?,
            digest: digest.ok_or_else(|| invalid("--manifest-sha256 is required"))?,
            config,
            query,
            limit,
            apply,
            repair_vectors,
            max_copy_bytes,
            confirm_durability,
            format,
        }
        .validate(seen.contains("--limit"), seen.contains("--max-copy-bytes"))
    }

    fn validate(mut self, explicit_limit: bool, explicit_copy_limit: bool) -> SearchResult<Self> {
        // Bound identity text here; the store's canonical pointer decoder owns
        // the exact generation grammar. Do not accept a pathname as an identity.
        if self.generation.len() != 60
            || self
                .generation
                .bytes()
                .any(|byte| !byte.is_ascii_alphanumeric() && byte != b'-')
        {
            return Err(invalid(
                "invalid generation ID; use the trusted publication receipt",
            ));
        }
        if self.digest.len() != 64 || !self.digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(invalid(
                "manifest SHA-256 must be exactly 64 hexadecimal characters",
            ));
        }
        if self
            .query
            .as_ref()
            .is_some_and(|query| query.trim().is_empty() || query.len() > MAX_QUERY_BYTES)
        {
            return Err(invalid("preview query must be nonblank and at most 64 KiB"));
        }
        if explicit_limit && self.query.is_none() {
            return Err(invalid("--limit requires --query"));
        }
        if self.repair_vectors && !self.apply {
            return Err(invalid("--repair-vectors requires --apply; inspection cannot allocate a repair copy"));
        }
        if explicit_copy_limit && !self.repair_vectors {
            return Err(invalid("--max-copy-bytes requires --repair-vectors"));
        }
        if self.repair_vectors && self.max_copy_bytes == 0 {
            return Err(invalid("vector repair requires a positive copy budget"));
        }
        #[cfg(not(unix))]
        if self.repair_vectors {
            return Err(invalid("vector repair requires Unix directory durability support"));
        }
        if self.confirm_durability
            && (self.apply || self.config.is_some() || self.query.is_some() || explicit_limit
                || self.repair_vectors || explicit_copy_limit)
        {
            return Err(invalid(
                "--confirm-durability cannot be combined with --apply, --config, --query, or --limit",
            ));
        }
        self.digest.make_ascii_lowercase();
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum RecoveryState {
    Inspected,
    RestoredDurable,
    RestoredDurabilityUncertain,
    DurabilityConfirmed,
    DurabilityStillUncertain,
}

#[derive(Debug, Serialize)]
struct RepairSource {
    generation_id: String,
    manifest_sha256: String,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u16,
    state: RecoveryState,
    generation_id: String,
    manifest_sha256: String,
    selection_write_performed: bool,
    durability_confirmed: bool,
    producer_admission_checked: bool,
    // The output generation is the new copy, not the damaged original. Keep
    // its original trusted receipt separate, including uncertain publication.
    #[serde(skip_serializing_if = "Option::is_none")]
    repaired_from: Option<RepairSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sync_error_kind: Option<String>,
    preview: Vec<SearchPayload>,
}

impl Report {
    fn inspected(generation: &PublishedGeneration) -> Self {
        Self {
            schema_version: 1,
            state: RecoveryState::Inspected,
            generation_id: generation.id().to_owned(),
            manifest_sha256: generation.manifest_sha256().to_owned(),
            selection_write_performed: false,
            durability_confirmed: false,
            producer_admission_checked: true,
            repaired_from: None,
            sync_error_kind: None,
            preview: Vec::new(),
        }
    }

    fn completed(mut self, publication: GenerationPublication, restored: bool) -> Self {
        // Nothing fallible follows publication. Keep its original visible
        // outcome even if a later output write or cancellation fails.
        self.selection_write_performed = restored;
        self.producer_admission_checked = restored;
        match publication {
            GenerationPublication::Durable(_) => {
                self.state = if restored {
                    RecoveryState::RestoredDurable
                } else {
                    RecoveryState::DurabilityConfirmed
                };
                self.durability_confirmed = true;
            }
            GenerationPublication::VisibleButDurabilityUncertain { source, .. } => {
                self.state = if restored {
                    RecoveryState::RestoredDurabilityUncertain
                } else {
                    RecoveryState::DurabilityStillUncertain
                };
                self.sync_error_kind = Some(format!("{:?}", source.kind()));
            }
        }
        self
    }
}

// Preserve the ordinary envelope's success/error invariant. A visible but
// uncertain restore is an error with typed recovery facts, not success data.
#[derive(Serialize)]
struct Response {
    #[serde(flatten)]
    envelope: OutputEnvelope<Report>,
    #[serde(skip_serializing_if = "Option::is_none")]
    recovery: Option<Report>,
}

impl Response {
    fn from_report(report: Report, format: OutputFormat) -> (Self, u8) {
        let meta = meta_for_format("recover", format);
        let now = format_unix_nanos(now_unix_nanos());
        if matches!(
            report.state,
            RecoveryState::RestoredDurabilityUncertain | RecoveryState::DurabilityStillUncertain
        ) {
            let error = OutputError::new(
                "subsystem_error",
                "The selected generation is visible, but durability is not confirmed.",
                1,
            )
            .with_field("complete_generation.recovery.durability")
            .with_suggestion(if report.repaired_from.is_some() {
                "Do not repeat the repair or assume an abort. Use fsfs-recover --confirm-durability with the NEW generation_id and manifest_sha256 in this recovery report, not the original repaired_from receipt; a changed selection will be refused."
            } else {
                "Do not assume an abort or blindly restore again. Use fsfs-recover --confirm-durability with the same trusted generation receipt; a changed selection will be refused."
            });
            (
                Self {
                    envelope: OutputEnvelope::error(error, meta, now),
                    recovery: Some(report),
                },
                1,
            )
        } else {
            (
                Self {
                    envelope: OutputEnvelope::success(report, meta, now),
                    recovery: None,
                },
                0,
            )
        }
    }
}

struct BoundedOutput {
    bytes: Vec<u8>,
    limit: usize,
}

impl BoundedOutput {
    const fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
        }
    }
}

impl Write for BoundedOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let length = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .filter(|length| *length <= self.limit)
            .ok_or_else(|| io::Error::other("recovery output exceeds its byte limit"))?;
        self.bytes
            .try_reserve(length - self.bytes.len())
            .map_err(|_| io::Error::other("cannot reserve recovery output buffer"))?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn encode<T: Serialize>(value: &T) -> SearchResult<Vec<u8>> {
    let mut buffer = BoundedOutput::new(MAX_REPORT_BYTES);
    serde_json::to_writer(&mut buffer, value).map_err(|source| SearchError::SubsystemError {
        subsystem: "fsfs.recovery.output",
        source: Box::new(source),
    })?;
    buffer.write_all(b"\n")?;
    Ok(buffer.bytes)
}

fn validate_preview(preview: &[SearchPayload]) -> SearchResult<()> {
    if preview.is_empty()
        || preview
            .iter()
            .any(|phase| phase.phase == SearchOutputPhase::RefinementFailed)
    {
        return Err(invalid(
            "preview did not complete successfully; no restoration was performed",
        ));
    }
    // Bound serialization before any selection write, leaving space for the
    // receipt and envelope. A giant preview cannot commit and then fail here.
    let mut buffer = BoundedOutput::new(MAX_REPORT_BYTES / 2);
    serde_json::to_writer(&mut buffer, preview).map_err(|source| SearchError::SubsystemError {
        subsystem: "fsfs.recovery.preview",
        source: Box::new(source),
    })?;
    Ok(())
}

async fn execute(
    cx: &Cx,
    runtime: Option<&FsfsRuntime>,
    options: &Options,
) -> SearchResult<Report> {
    if options.confirm_durability {
        let store = CompleteGenerationStore::open(cx, &options.root)?;
        let expected = store.open_retained(cx, &options.generation, &options.digest)?;
        let report = Report::inspected(&expected);
        // The API rechecks this exact target under the publication lease.
        // Do not substitute a separate is_selected() check plus unbound flush.
        let publication = store.confirm_retained_durability(cx, &expected)?;
        return Ok(report.completed(publication, false));
    }
    let runtime = runtime.ok_or_else(|| invalid("recovery runtime was not initialized"))?;
    if options.apply {
        let mut plan = if options.repair_vectors {
            #[cfg(unix)]
            {
                runtime.prepare_retained_vector_repair(
                    cx, &options.root, &options.generation, &options.digest, options.max_copy_bytes,
                ).await?
            }
            #[cfg(not(unix))]
            return Err(invalid("vector repair requires Unix directory durability support"));
        } else {
            runtime
                .prepare_retained_recovery(cx, &options.root, &options.generation, &options.digest)
                .await?
        };
        let mut report = Report::inspected(plan.generation());
        if options.repair_vectors {
            report.repaired_from = Some(RepairSource {
                generation_id: options.generation.clone(),
                manifest_sha256: options.digest.clone(),
            });
        }
        if let Some(query) = &options.query {
            report.preview = plan.search(cx, query, options.limit).await?;
            validate_preview(&report.preview)?;
        }
        let (publication, _reader) = plan.restore(cx)?;
        Ok(report.completed(publication, true))
    } else {
        // Inspection never takes publication ownership and never replaces a
        // corrupt descriptor. The caller must separately authorize --apply.
        let mut reader = runtime
            .open_retained_search_from_receipt(
                cx,
                &options.root,
                &options.generation,
                &options.digest,
            )
            .await?;
        let mut report = Report::inspected(reader.generation());
        if let Some(query) = &options.query {
            report.preview = reader.search(cx, query, options.limit).await?;
            validate_preview(&report.preview)?;
        }
        Ok(report)
    }
}

fn configured_runtime(options: &Options) -> SearchResult<FsfsRuntime> {
    let home = frankensearch_core::platform_dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"));
    let env = current_unicode_environment();
    let overrides = CliOverrides::default();
    let loaded = if let Some(path) = &options.config {
        if !std::fs::metadata(path)?.is_file() {
            return Err(invalid("--config must name a regular configuration file"));
        }
        load_from_sources(Some(path), &env, &overrides, &home)?
    } else {
        let project = default_project_config_file_path(&std::env::current_dir()?);
        let user = default_user_config_file_path(&home);
        load_from_layered_sources(Some(&project), Some(&user), &env, &overrides, &home)?
    };
    let mut config = loaded.config;
    // Recovery uses already available producers. It is not a download,
    // indexing, automatic fallback, update check or watch-mode entrypoint.
    config.indexing.offline = true;
    config.indexing.watch_mode = false;
    Ok(FsfsRuntime::new(config).with_cli_input(CliInput {
        command: CliCommand::Doctor,
        index_dir: Some(options.root.clone()),
        quiet: true,
        no_color: true,
        format: options.format,
        ..CliInput::default()
    }))
}

fn run(options: Options) -> SearchResult<Report> {
    // Pure durability confirmation must work when model/configuration state is
    // unavailable. It never claims the selected generation is semantic-ready.
    let runtime = if options.confirm_durability {
        None
    } else {
        Some(configured_runtime(&options)?)
    };
    let pool = Arc::new(SearchBlockingPool::default());
    #[cfg(feature = "rerank")]
    let runtime = runtime.map(|runtime| runtime.with_native_blocking_pool(pool.handle()));
    let scheduler = RuntimeBuilder::current_thread()
        .blocking_threads(0, 2)
        .build()
        .map_err(|source| SearchError::SubsystemError {
            subsystem: "fsfs.recovery.runtime",
            source: Box::new(io::Error::other(source.to_string())),
        })?;
    let shutdown = Arc::new(ShutdownCoordinator::new());
    shutdown.register_signals()?;
    let worker_shutdown = Arc::clone(&shutdown);
    let worker_pool = Arc::clone(&pool);
    let task = scheduler.handle().spawn(async move {
        let cx = worker_pool.context(Cx::current().expect("runtime installs a request context"));
        let _cancellation = worker_shutdown.cancellation_scope(&cx);
        execute(&cx, runtime.as_ref(), &options).await
    });
    let result = scheduler.block_on(task);
    shutdown.stop_signal_listener();
    // A late signal cannot turn an already-visible restore into a claimed abort.
    // The result retains its publication facts; no process::exit skips cleanup.
    drop(scheduler);
    drop(pool);
    result
}

fn render(response: &Response, format: OutputFormat) -> SearchResult<Vec<u8>> {
    if format != OutputFormat::Table {
        return encode(response);
    }
    let mut output = BoundedOutput::new(MAX_REPORT_BYTES);
    if let Some(report) = response
        .envelope
        .data
        .as_ref()
        .or(response.recovery.as_ref())
    {
        writeln!(output, "Recovery state: {:?}", report.state)?;
        writeln!(
            output,
            "Generation: {}\nManifest SHA-256: {}",
            report.generation_id, report.manifest_sha256,
        )?;
        if let Some(original) = &report.repaired_from {
            writeln!(
                output,
                "Reconstructed from: {}\nOriginal manifest SHA-256: {}",
                original.generation_id, original.manifest_sha256,
            )?;
        }
        writeln!(
            output,
            "Selection write performed: {}\nDurability confirmed: {}",
            report.selection_write_performed, report.durability_confirmed,
        )?;
        writeln!(
            output,
            "Producer admission checked: {}",
            report.producer_admission_checked,
        )?;
        for phase in &report.preview {
            writeln!(
                output,
                "Preview {:?}: {} hits",
                phase.phase,
                phase.hits.len()
            )?;
            for hit in &phase.hits {
                writeln!(output, "  {:?}", hit.path)?;
            }
        }
    }
    if let Some(error) = &response.envelope.error {
        writeln!(output, "{}", error.message)?;
        if let Some(suggestion) = &error.suggestion {
            writeln!(output, "{suggestion}")?;
        }
    }
    Ok(output.bytes)
}

fn main() -> ExitCode {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() == 1 && args[0] == "--help" {
        return if io::stdout().lock().write_all(HELP.as_bytes()).is_ok() {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        };
    }
    let parsed = Options::parse(args);
    let format = parsed
        .as_ref()
        .map_or(OutputFormat::Json, |options| options.format);
    init_subscriber(Verbosity::from_flags(false, true), true);
    let (response, code) = match parsed.and_then(run) {
        Ok(report) => Response::from_report(report, format),
        Err(error) => {
            let code = u8::try_from(exit_code_for(&error)).unwrap_or(1);
            (
                Response {
                    envelope: OutputEnvelope::error(
                        output_error_from(&error),
                        meta_for_format("recover", format),
                        format_unix_nanos(now_unix_nanos()),
                    ),
                    recovery: None,
                },
                code,
            )
        }
    };
    // Encode completely before exposing bytes. An OS output failure may still
    // leave a partial frame; do not append a second, misleading error envelope.
    let emission = render(&response, format).and_then(|bytes| {
        let mut stdout = io::stdout().lock();
        stdout.write_all(&bytes)?;
        stdout.flush().map_err(SearchError::Io)
    });
    if emission.is_err() {
        eprintln!(
            "fsfs-recover: output failed; a requested restore may already be visible. Inspect the selection before retrying."
        );
        return ExitCode::FAILURE;
    }
    ExitCode::from(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Vec<OsString> {
        [
            "--index-dir".to_owned(),
            "store".to_owned(),
            "--generation".to_owned(),
            format!("g-{}-{}-{}", "a".repeat(32), "b".repeat(8), "c".repeat(16)),
            "--manifest-sha256".to_owned(),
            "a".repeat(64),
        ]
        .into_iter()
        .map(OsString::from)
        .collect()
    }

    #[test]
    fn inspection_is_default_and_apply_is_explicit() {
        let options = Options::parse(args()).unwrap();
        assert!(!options.apply);
        assert_eq!(options.limit, 10);
        assert_eq!(options.format, OutputFormat::Json);
        let mut input = args();
        input.push("--apply".into());
        assert!(Options::parse(input).unwrap().apply);
    }

    #[test]
    fn missing_forged_and_duplicate_authority_is_rejected() {
        for index in [1, 3, 5] {
            let mut input = args();
            input[index] = "".into();
            assert!(Options::parse(input).is_err());
        }
        for invalid_id in [
            "../outside".to_owned(),
            "/tmp/target".to_owned(),
            "x".repeat(61),
        ] {
            let mut input = args();
            input[3] = invalid_id.into();
            assert!(Options::parse(input).is_err());
        }
        for digest in ["a".repeat(63), "g".repeat(64)] {
            let mut input = args();
            input[5] = digest.into();
            assert!(Options::parse(input).is_err());
        }
        for suffix in [
            vec!["--apply", "--apply"],
            vec!["--index-dir", "other"],
            vec!["--unknown"],
            vec!["--query"],
            vec!["--apply", "--help"],
        ] {
            let mut input = args();
            input.extend(suffix.into_iter().map(OsString::from));
            assert!(Options::parse(input).is_err());
        }
    }

    #[test]
    fn query_values_cannot_become_apply_flags_and_limits_are_bounded() {
        let mut input = args();
        input.extend(["--query", "--apply"].map(OsString::from));
        let options = Options::parse(input).unwrap();
        assert!(!options.apply);
        assert_eq!(options.query.as_deref(), Some("--apply"));
        for limit in ["0", "101", "all", "-1", "18446744073709551616"] {
            let mut input = args();
            input.extend(["--query", "hello", "--limit", limit].map(OsString::from));
            assert!(Options::parse(input).is_err());
        }
        let mut input = args();
        input.extend(["--limit", "10"].map(OsString::from));
        assert!(Options::parse(input).is_err());
        let mut input = args();
        input.push("--query".into());
        input.push("x".repeat(MAX_QUERY_BYTES + 1).into());
        assert!(Options::parse(input).is_err());
    }

    #[test]
    fn output_buffer_refuses_before_partial_admission() {
        let mut output = BoundedOutput::new(3);
        output.write_all(b"abc").unwrap();
        assert!(output.write_all(b"d").is_err());
        assert_eq!(output.bytes, b"abc");
    }

    #[test]
    fn repair_is_never_implicit_and_conflicting_copy_authority_is_refused() {
        assert!(!Options::parse(args()).unwrap().repair_vectors);
        for suffix in [
            vec!["--repair-vectors"],
            vec!["--max-copy-bytes", "1"],
            vec!["--apply", "--max-copy-bytes", "1"],
            vec!["--apply", "--repair-vectors", "--repair-vectors"],
            vec!["--apply", "--repair-vectors", "--confirm-durability"],
            vec!["--apply", "--repair-vectors", "--max-copy-bytes", "0"],
            vec!["--apply", "--repair-vectors", "--max-copy-bytes", "-1"],
            vec!["--apply", "--repair-vectors", "--max-copy-bytes", "+1"],
            vec!["--apply", "--repair-vectors", "--max-copy-bytes", "18446744073709551616"],
            vec!["--apply", "--repair-vectors", "--max-copy-bytes", "1", "--max-copy-bytes", "2"],
        ] {
            let mut input = args();
            input.extend(suffix.iter().map(|value| OsString::from(*value)));
            assert!(Options::parse(input).is_err(), "{suffix:?}");
        }
        let mut input = args();
        input.extend(["--query", "--repair-vectors"].map(OsString::from));
        let parsed = Options::parse(input).unwrap();
        assert!(!parsed.repair_vectors && !parsed.apply);
        assert_eq!(parsed.query.as_deref(), Some("--repair-vectors"));
    }

    #[cfg(unix)]
    #[test]
    fn explicit_repair_preserves_receipt_preview_and_the_selected_copy_budget() {
        for budget in ["1", "1073741824", "18446744073709551615"] {
            let mut input = args();
            input.extend([
                "--apply", "--repair-vectors", "--max-copy-bytes", budget,
                "--query", "sharedtoken", "--limit", "3",
            ].map(OsString::from));
            let options = Options::parse(input).unwrap();
            assert!(options.apply && options.repair_vectors && !options.confirm_durability);
            assert_eq!(options.max_copy_bytes, budget.parse::<u64>().unwrap());
            assert_eq!(options.query.as_deref(), Some("sharedtoken"));
            assert_eq!(options.limit, 3);
            assert_eq!(options.digest, "a".repeat(64));
        }
        let mut input = args();
        input.extend(["--apply", "--repair-vectors"].map(OsString::from));
        assert_eq!(Options::parse(input).unwrap().max_copy_bytes, DEFAULT_REPAIR_COPY_BYTES);
    }

    #[test]
    fn repair_reports_new_receipt_and_preserves_original_provenance_on_uncertainty() {
        for state in [RecoveryState::RestoredDurable, RecoveryState::RestoredDurabilityUncertain] {
            let mut facts = report(state);
            facts.generation_id = "new-repaired-generation".to_owned();
            facts.manifest_sha256 = "b".repeat(64);
            facts.repaired_from = Some(RepairSource {
                generation_id: "original-damaged-generation".to_owned(),
                manifest_sha256: "a".repeat(64),
            });
            let (response, code) = Response::from_report(facts, OutputFormat::Json);
            let value: serde_json::Value = serde_json::from_slice(&encode(&response).unwrap()).unwrap();
            let durable = state == RecoveryState::RestoredDurable;
            assert_eq!(code, u8::from(!durable));
            assert_eq!(value["ok"], durable);
            let facts = if durable { &value["data"] } else { &value["recovery"] };
            assert_eq!(facts["generation_id"], "new-repaired-generation");
            assert_eq!(facts["manifest_sha256"], "b".repeat(64));
            assert_eq!(facts["repaired_from"]["generation_id"], "original-damaged-generation");
            assert_eq!(facts["repaired_from"]["manifest_sha256"], "a".repeat(64));
            assert_eq!(facts["selection_write_performed"], true);
            assert_eq!(facts["durability_confirmed"], durable);
            if !durable {
                assert!(value["error"]["suggestion"].as_str().unwrap().contains("NEW generation_id"));
            }
            let table = String::from_utf8(render(&response, OutputFormat::Table).unwrap()).unwrap();
            assert!(table.contains("Generation: new-repaired-generation"));
            assert!(table.contains("Reconstructed from: original-damaged-generation"));
        }
        let (ordinary, _) = Response::from_report(report(RecoveryState::Inspected), OutputFormat::Json);
        let value: serde_json::Value = serde_json::from_slice(&encode(&ordinary).unwrap()).unwrap();
        assert!(value["data"].get("repaired_from").is_none());
    }

    #[test]
    fn confirmation_is_explicit_and_rejects_ignored_or_conflicting_options() {
        let mut input = args();
        input.push("--confirm-durability".into());
        let parsed = Options::parse(input.clone()).unwrap();
        assert!(parsed.confirm_durability);
        assert!(!parsed.apply);
        for suffix in [
            vec!["--apply"],
            vec!["--config", "configuration.toml"],
            vec!["--query", "hello"],
            vec!["--limit", "10"],
            vec!["--confirm-durability"],
        ] {
            let mut conflicting = input.clone();
            conflicting.extend(suffix.into_iter().map(OsString::from));
            assert!(Options::parse(conflicting).is_err());
        }
        let mut query = args();
        query.extend(["--query", "--confirm-durability"].map(OsString::from));
        assert!(!Options::parse(query).unwrap().confirm_durability);
    }

    fn report(state: RecoveryState) -> Report {
        Report {
            schema_version: 1,
            state,
            generation_id: "g-test".to_owned(),
            manifest_sha256: "a".repeat(64),
            selection_write_performed: matches!(
                state,
                RecoveryState::RestoredDurable | RecoveryState::RestoredDurabilityUncertain
            ),
            durability_confirmed: matches!(
                state,
                RecoveryState::RestoredDurable | RecoveryState::DurabilityConfirmed
            ),
            producer_admission_checked: !matches!(
                state,
                RecoveryState::DurabilityConfirmed | RecoveryState::DurabilityStillUncertain
            ),
            repaired_from: None,
            sync_error_kind: None,
            preview: Vec::new(),
        }
    }

    #[test]
    fn uncertain_restore_has_error_envelope_and_machine_readable_visible_outcome() {
        let (response, code) = Response::from_report(
            report(RecoveryState::RestoredDurabilityUncertain),
            OutputFormat::Json,
        );
        let encoded = encode(&response).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(code, 1);
        assert_eq!(value["ok"], false);
        assert!(value.get("data").is_none());
        assert_eq!(value["recovery"]["state"], "restored_durability_uncertain");
        assert_eq!(value["recovery"]["selection_write_performed"], true);
        assert_eq!(value["recovery"]["durability_confirmed"], false);
        assert_eq!(String::from_utf8_lossy(&encoded).matches('\n').count(), 1);
    }

    #[test]
    fn inspection_never_claims_durability_and_durable_restore_has_no_error() {
        for state in [RecoveryState::Inspected, RecoveryState::RestoredDurable] {
            let (response, code) = Response::from_report(report(state), OutputFormat::Jsonl);
            assert_eq!(code, 0);
            let value: serde_json::Value =
                serde_json::from_slice(&encode(&response).unwrap()).unwrap();
            assert_eq!(value["ok"], true);
            assert!(value.get("error").is_none());
            assert!(value.get("recovery").is_none());
            assert_eq!(
                value["data"]["durability_confirmed"],
                state == RecoveryState::RestoredDurable
            );
        }
    }

    #[test]
    fn confirmation_outcomes_never_claim_model_admission_or_selection_writes() {
        for state in [
            RecoveryState::DurabilityConfirmed,
            RecoveryState::DurabilityStillUncertain,
        ] {
            let (response, code) = Response::from_report(report(state), OutputFormat::Json);
            let encoded = encode(&response).unwrap();
            let value: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
            let confirmed = state == RecoveryState::DurabilityConfirmed;
            assert_eq!(code, u8::from(!confirmed));
            assert_eq!(value["ok"], confirmed);
            let facts = if confirmed {
                &value["data"]
            } else {
                &value["recovery"]
            };
            assert_eq!(facts["selection_write_performed"], false);
            assert_eq!(facts["producer_admission_checked"], false);
            assert_eq!(facts["durability_confirmed"], confirmed);
        }
    }

    #[cfg(unix)]
    #[test]
    fn confirmation_executes_without_a_model_runtime_and_refuses_stale_targets() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            let publish = || {
                let build = store.begin(&cx).unwrap();
                std::fs::write(build.path().join("payload"), b"container fixture").unwrap();
                let GenerationPublication::Durable(generation) =
                    build.publish(&cx, |_, _| Ok(())).unwrap()
                else {
                    panic!("fixture publication must be durable"); // ubs:ignore — test assertion.
                };
                generation
            };
            let first = publish();
            let mut input = args();
            input[1] = directory.path().as_os_str().to_owned();
            input[3] = first.id().into();
            input[5] = first.manifest_sha256().into();
            input.push("--confirm-durability".into());
            let options = Options::parse(input).unwrap();
            let report = execute(&cx, None, &options).await.unwrap();
            assert_eq!(report.state, RecoveryState::DurabilityConfirmed);
            assert!(!report.producer_admission_checked);
            assert!(!report.selection_write_performed);
            let successor = publish();
            assert!(execute(&cx, None, &options).await.is_err());
            assert_eq!(store.active(&cx).unwrap(), Some(successor));
        });
    }
}
