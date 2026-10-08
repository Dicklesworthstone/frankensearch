//! `fsfs live-search` process boundary with explicit source-watch opt-in.
//!
//! Keep this outside the ordinary CLI dispatcher: live subscriptions require an
//! explicitly selected complete store, never auto-discovery, migration or an
//! update check. Models/configuration require `--hybrid`; only `--watch-source`
//! authorizes indexing and complete-generation publication.

mod hybrid;
#[cfg(unix)]
mod terminal;
#[cfg(unix)]
mod watch;

use std::collections::HashSet;
use std::ffi::OsString;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use asupersync::Cx;
use asupersync::runtime::RuntimeBuilder;
use frankensearch_core::{SearchError, SearchResult};
use frankensearch_fsfs::adapters::live_search::LiveSearchConfig;
use frankensearch_fsfs::adapters::live_search_session::LiveSearchRefreshConfig;
use frankensearch_fsfs::generation_store::CompleteGenerationStore;
use frankensearch_fsfs::runtime::SearchBlockingPool;
use frankensearch_fsfs::{FsfsRuntime, ShutdownCoordinator, exit_code, exit_code_for};
#[cfg(all(test, unix))]
use frankensearch_quill::QuillConfig;

const MAX_QUERY_BYTES: usize = 64 * 1024;
const HELP: &str = "fsfs live-search --index-dir STORE --query TEXT [OPTIONS]

Subscribe to committed complete generations; native Quill lexical search is default.
Without --watch-source, the existing store must contain FSFS-CURRENT for a snapshot.
Read-only by default. No downloads, root discovery, or sealed-index migration.
Run a publisher separately, or explicitly opt into indexing with --watch-source.

  --index-dir STORE       Explicit complete-generation store root (required)
  --query TEXT            Nonblank UTF-8 query, at most 64 KiB (required)
  --hybrid                Use the retained progressive hybrid pipeline and local models
  --config FILE           Hybrid configuration/model policy; requires --hybrid
  --tui                   Live terminal view and query editor (Unix terminals required)
  --watch-source DIR      Index and watch DIR, publishing successors; requires --hybrid
  --limit N               Result window, 1..=10000 (default 20)
  --poll-ms N             Selection probe interval, 10..=60000 (default 100)
  --debounce-ms N         Quiet period before a refresh, 0..=60000 (default 100)
  --max-wait-ms N         Churn coalescing bound, 1..=60000 (default 1000)
  --min-score-delta N     Finite, non-negative score-only threshold (default 0)
  --max-updates N         Exit after N completed updates (all phases; see TUI notes)
  --once                 Deliver one generation, including refinement, then exit
  --timeout-ms N         Overall cooperative deadline, 1..=86400000
  --format jsonl         Explicit machine output (default; incompatible with --tui)
  --help                 Print this help (use alone)

Default stdout contains only fsfs.stream.live_search.v1 frames: an initial snapshot,
then atomic deltas. Empty deltas still identify committed generation changes.
Apply every delta in full before rendering. The window is top-k, not all matches.
Generation IDs are opaque; an explicitly restored predecessor is a new update.
Each connection starts at sequence 1 with a full snapshot; there is no replay log.

The default waits for an initial selection in an EXISTING store. A later absent
selection never creates synthetic removals (hybrid mode fails closed). Corruption,
unreadable descriptors, query failures, and output failures stop the subscription.
Errors are JSON on stderr; no error envelope is appended to the result stream.
Ctrl-C/SIGTERM exits 130. A closed output pipe exits successfully without retry.
The timeout and cancellation are cooperative; they cannot preempt an already
running filesystem syscall, synchronous query, or blocked output syscall.
The default is lexical-only and never loads models or configuration. --hybrid
loads normal fsfs configuration (or --config) and uses already available models,
with downloads disabled. Configure ranking, fast-only and rerank policy there.
Hybrid stdout uses fsfs.stream.live_search.hybrid.v1, preserving Initial,
Refined and RefinementFailed annotations and a generation-pinned result delta.
Initial is flushed before refinement. A failed refinement preserves its Initial
window; it is not a successful Refined update. --once/--max-updates never split
that phase sequence. Each hybrid record is bounded to 16 MiB before output.
--filter, --rerank, and --semantic CLI flags are refused rather than ignored.

--tui displays this same subscription in an alternate-screen terminal view.
Up/Down or j/k select results; PgUp/PgDn move ten rows; Home/End jump; q/Esc quit.
Press / or Enter to edit the query. In the editor, Enter submits, Esc cancels the
draft, Ctrl-U clears it, and Left/Right/Home/End/Backspace/Delete edit UTF-8 text.
Bracketed paste is literal input: newlines/tabs become spaces, control characters
are refused, and an oversized paste is rejected whole. Queries remain bounded to
64 KiB. Ctrl-C stops in either mode; q and j/k are ordinary text while editing.
Typing does not launch searches. Only the latest submitted query is retained while
an existing phase sequence finishes. Then a fresh query snapshot replaces the old
window, reusing admitted readers/models; no process, model reload or index rebuild
is needed solely to change the query. Unchanged submissions are no-ops. Query
changes never revive a failed transport, reset the overall timeout, or split
Initial/refinement. Selection follows identity within a query; a new query starts
at its first result. Failed refinement retains that query's Initial results.
For a read-only TUI subscription, --max-updates counts completed query/generation
refreshes. With --watch-source it still counts only actual published generations:
query editing does not publish or spend that limit. Machine NDJSON mode remains a
fixed-query subscription. Existing limits restore the terminal when they stop it.

--watch-source is supported on Unix and requires disjoint source/store trees.
It builds the initial generation, then streams each durable publication through
the same hybrid pipeline. Models must already be available. Because complete
generations are retained, --once or --max-updates is REQUIRED in this mode.
--once builds and queries once; --max-updates counts published generations, not
phases. These limits are not a disk quota or automatic garbage collection.
The native watcher owns source debounce/reconciliation; --poll-ms, --debounce-ms
and --max-wait-ms are therefore refused in this mode. Query/output backpressure
pauses new builds, not native change registration. Failed query/output or timeout
can follow a successful publication: they never roll it back or delete its files.
";

#[derive(Debug, Clone)]
struct Options {
    root: PathBuf,
    query: String,
    limits: LiveSearchConfig,
    refresh: LiveSearchRefreshConfig,
    poll_interval: Duration,
    max_updates: Option<u64>,
    timeout: Option<Duration>,
    once: bool,
    hybrid: bool,
    config_path: Option<PathBuf>,
    watch_source: Option<PathBuf>,
    tui: bool,
}

impl Options {
    #[allow(clippy::too_many_lines)]
    fn parse(args: Vec<OsString>) -> SearchResult<Self> {
        let mut root = None;
        let mut query = None;
        let mut limits = LiveSearchConfig::default();
        let mut refresh = LiveSearchRefreshConfig::default();
        let mut poll_interval = Duration::from_millis(100);
        let mut max_updates = None;
        let mut timeout = None;
        let mut once = false;
        let mut hybrid = false;
        let mut config_path = None;
        let mut watch_source = None;
        let mut tui = false;
        let mut seen = HashSet::new();
        let mut args = args.into_iter();
        while let Some(flag) = args.next() {
            let flag = text(flag)?;
            if !matches!(
                flag.as_str(),
                "--index-dir"
                    | "--query"
                    | "--limit"
                    | "--poll-ms"
                    | "--debounce-ms"
                    | "--max-wait-ms"
                    | "--min-score-delta"
                    | "--max-updates"
                    | "--once"
                    | "--timeout-ms"
                    | "--format"
                    | "--hybrid"
                    | "--config"
                    | "--watch-source"
                    | "--tui"
            ) {
                return Err(invalid("unknown argument; run fsfs live-search --help"));
            }
            if !seen.insert(flag.clone()) {
                return Err(invalid(format!("duplicate {flag} is not allowed")));
            }
            if flag == "--once" {
                once = true;
                continue;
            }
            if flag == "--hybrid" {
                hybrid = true;
                continue;
            }
            if flag == "--tui" {
                tui = true;
                continue;
            }
            let value = args
                .next()
                .filter(|value| !value.is_empty())
                .ok_or_else(|| invalid(format!("missing or empty value for {flag}")))?;
            match flag.as_str() {
                "--index-dir" => root = Some(PathBuf::from(value)),
                "--config" => config_path = Some(PathBuf::from(value)),
                "--watch-source" => watch_source = Some(PathBuf::from(value)),
                "--query" => query = Some(text(value)?),
                "--limit" => {
                    limits.max_results = usize::try_from(number(value, &flag, 1, 10_000)?)
                        .map_err(|_| invalid("result limit does not fit this platform"))?;
                }
                "--poll-ms" => {
                    poll_interval = Duration::from_millis(number(value, &flag, 10, 60_000)?);
                }
                "--debounce-ms" => {
                    refresh.debounce = Duration::from_millis(number(value, &flag, 0, 60_000)?);
                }
                "--max-wait-ms" => {
                    refresh.max_wait = Duration::from_millis(number(value, &flag, 1, 60_000)?);
                }
                "--min-score-delta" => {
                    limits.min_score_delta = text(value)?
                        .parse::<f64>()
                        .ok()
                        .filter(|value| value.is_finite() && *value >= 0.0)
                        .ok_or_else(|| {
                            invalid("--min-score-delta must be finite and non-negative")
                        })?;
                }
                "--max-updates" => max_updates = Some(number(value, &flag, 1, u64::MAX)?),
                "--timeout-ms" => {
                    timeout = Some(Duration::from_millis(number(value, &flag, 1, 86_400_000)?));
                }
                "--format" => {
                    if text(value)? != "jsonl" {
                        return Err(invalid("live-search supports only --format jsonl"));
                    }
                }
                _ => return Err(invalid("invalid value-bearing argument")),
            }
        }
        let query = query.ok_or_else(|| invalid("--query is required"))?;
        if query.trim().is_empty() || query.len() > MAX_QUERY_BYTES {
            return Err(invalid("query must be nonblank and at most 64 KiB"));
        }
        if refresh.max_wait < refresh.debounce {
            return Err(invalid("--max-wait-ms must be at least --debounce-ms"));
        }
        if once && max_updates.is_some() {
            return Err(invalid("--once cannot be combined with --max-updates"));
        }
        if config_path.is_some() && !hybrid {
            return Err(invalid(
                "--config requires --hybrid; lexical mode never loads configuration",
            ));
        }
        if tui {
            if !cfg!(unix) {
                return Err(invalid("--tui requires the Unix terminal backend"));
            }
            if seen.contains("--format") {
                return Err(invalid("--tui cannot be combined with --format"));
            }
        }
        if watch_source.is_some() {
            if !cfg!(unix) {
                return Err(invalid(
                    "--watch-source requires a Unix complete-generation watcher",
                ));
            }
            if !hybrid {
                return Err(invalid(
                    "--watch-source requires --hybrid and already available models",
                ));
            }
            if !once && max_updates.is_none() {
                return Err(invalid(
                    "--watch-source requires --once or --max-updates; retained generations are not garbage-collected",
                ));
            }
            if ["--poll-ms", "--debounce-ms", "--max-wait-ms"]
                .iter()
                .any(|flag| seen.contains(*flag))
            {
                return Err(invalid(
                    "--watch-source uses native source debounce/reconciliation, not selection-poll timing flags",
                ));
            }
        }
        Ok(Self {
            root: root.ok_or_else(|| invalid("--index-dir is required; no store is guessed"))?,
            query,
            limits,
            refresh,
            poll_interval,
            max_updates: if once { Some(1) } else { max_updates },
            timeout,
            once,
            hybrid,
            config_path,
            watch_source,
            tui,
        })
    }
}

fn text(value: OsString) -> SearchResult<String> {
    value
        .into_string()
        .map_err(|_| invalid("expected UTF-8 option or query text"))
}

fn number(value: OsString, flag: &str, min: u64, max: u64) -> SearchResult<u64> {
    text(value)?
        .parse::<u64>()
        .ok()
        .filter(|value| (min..=max).contains(value))
        .ok_or_else(|| invalid(format!("{flag} must be an integer in {min}..={max}")))
}

fn invalid(reason: impl Into<String>) -> SearchError {
    SearchError::InvalidConfig {
        field: "live_search.arguments".to_owned(),
        value: String::new(),
        reason: reason.into(),
    }
}

/// A process-lifetime budget, also checked at the final output boundary.
struct Budget {
    started: Instant,
    timeout: Option<Duration>,
}

impl Budget {
    fn remaining(&self) -> Option<Duration> {
        self.timeout
            .map(|timeout| timeout.saturating_sub(self.started.elapsed()))
    }

    fn check(&self, cx: &Cx) -> SearchResult<()> {
        cx.checkpoint().map_err(|error| SearchError::Cancelled {
            phase: "fsfs.live_search.command".to_owned(),
            reason: error.to_string(),
        })?;
        if self
            .remaining()
            .is_some_and(|remaining| remaining.is_zero())
        {
            return Err(SearchError::SearchTimeout {
                elapsed_ms: u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX),
                budget_ms: u64::try_from(self.timeout.unwrap_or_default().as_millis())
                    .unwrap_or(u64::MAX),
            });
        }
        Ok(())
    }
}

/// This guard does not buffer or retry. The session owns serialization and only
/// acknowledges its baseline after this writer's flush succeeds.
struct GuardedOutput<'a, W> {
    writer: &'a mut W,
    cx: &'a Cx,
    budget: &'a Budget,
}

impl<W: Write> Write for GuardedOutput<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.budget.check(self.cx).map_err(io::Error::other)?;
        self.writer.write(bytes)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.budget.check(self.cx).map_err(io::Error::other)?;
        self.writer.flush()
    }
}

#[cfg(test)]
async fn execute<W: Write + Send>(cx: &Cx, options: &Options, writer: &mut W) -> SearchResult<u64> {
    execute_with_runtime(cx, options, writer, None).await
}

async fn execute_with_runtime<W: Write + Send>(
    cx: &Cx,
    options: &Options,
    writer: &mut W,
    runtime: Option<FsfsRuntime>,
) -> SearchResult<u64> {
    let budget = Budget {
        started: Instant::now(),
        timeout: options.timeout,
    };
    budget.check(cx)?;
    if options.watch_source.is_some() {
        #[cfg(unix)]
        return Box::pin(watch::execute(cx, &budget, options, writer, runtime)).await;
        #[cfg(not(unix))]
        return Err(invalid("--watch-source is unsupported on this platform"));
    }
    // Opening an explicit existing root never creates a generation or writer lease.
    let store = CompleteGenerationStore::open(cx, &options.root)?;
    if options.once && store.active(cx)?.is_none() {
        return Err(SearchError::IndexNotFound {
            path: options.root.join("FSFS-CURRENT"),
        });
    }
    let mut session = hybrid::Subscription::new(store, options, runtime)?;
    let mut delivered = 0_u64;
    loop {
        budget.check(cx)?;
        let result = {
            let mut output = GuardedOutput {
                writer: &mut *writer,
                cx,
                budget: &budget,
            };
            Box::pin(session.poll_ndjson(cx, Instant::now(), &mut output)).await
        };
        // Cancellation and timeout stay typed even when the output guard was
        // the first boundary to observe them. No retry can extend a torn frame.
        budget.check(cx)?;
        if result? {
            delivered = delivered
                .checked_add(1)
                .ok_or_else(|| invalid("delivered generation counter exhausted"))?;
            if options
                .max_updates
                .is_some_and(|maximum| delivered >= maximum)
            {
                return Ok(delivered);
            }
        } else if options.once {
            // Selection can disappear between the preflight and first poll.
            return Err(SearchError::IndexNotFound {
                path: options.root.join("FSFS-CURRENT"),
            });
        }
        let delay = budget
            .remaining()
            .map_or(options.poll_interval, |remaining| {
                remaining.min(options.poll_interval)
            });
        // Bound signal responsiveness independently of a long user poll period.
        let mut remaining = delay;
        while !remaining.is_zero() {
            budget.check(cx)?;
            let slice = remaining.min(Duration::from_millis(25));
            asupersync::time::sleep(cx.now(), slice).await;
            remaining = remaining.saturating_sub(slice);
        }
    }
}

fn run(options: Options) -> SearchResult<u64> {
    // Refuse redirected terminal mode before configuration or a source build
    // can run. A headless caller continues to use the unchanged NDJSON path.
    if options.tui {
        #[cfg(unix)]
        terminal::preflight()?;
        #[cfg(not(unix))]
        return Err(invalid("--tui requires the Unix terminal backend"));
    }
    // This command owns a dedicated single-request runtime on the calling CLI
    // thread. There are no interactive/request tasks sharing its synchronous
    // filesystem/query/output lane. Quill's async admission uses this same
    // runtime and the established host-owned blocking pool; no nested runtime.
    let pool = Arc::new(SearchBlockingPool::default());
    let app_runtime = hybrid::configured_runtime(&options)?;
    #[cfg(feature = "rerank")]
    let app_runtime = app_runtime.map(|runtime| runtime.with_native_blocking_pool(pool.handle()));
    let scheduler = RuntimeBuilder::current_thread()
        .blocking_threads(0, 2)
        .build()
        .map_err(|error| SearchError::SubsystemError {
            subsystem: "fsfs.live_search.runtime",
            source: Box::new(io::Error::other(error.to_string())),
        })?;
    let shutdown = Arc::new(ShutdownCoordinator::new());
    shutdown.register_signals()?;
    let request_shutdown = Arc::clone(&shutdown);
    let request_pool = Arc::clone(&pool);
    let task = scheduler.handle().spawn(async move {
        let current =
            Cx::current().ok_or_else(|| invalid("runtime did not install a request context"))?;
        let cx = request_pool.context(current);
        let scope = request_shutdown.cancellation_scope(&cx);
        if options.tui {
            #[cfg(unix)]
            return Box::pin(terminal::execute(&scope, &options, app_runtime)).await;
            #[cfg(not(unix))]
            return Err(invalid("--tui requires the Unix terminal backend"));
        }
        // Stdout (rather than its non-Send lock guard) lives across admission awaits.
        // Only this task writes stdout, so frames cannot interleave in this process.
        execute_with_runtime(&scope, &options, &mut io::stdout(), app_runtime).await
    });
    let result = scheduler.block_on(task);
    shutdown.stop_signal_listener();
    drop(scheduler);
    drop(pool);
    result
}

fn result_exit_code(result: &SearchResult<u64>) -> i32 {
    match result {
        Ok(_) => exit_code::OK,
        Err(SearchError::Io(error)) if error.kind() == io::ErrorKind::BrokenPipe => exit_code::OK,
        Err(SearchError::Cancelled { .. }) => exit_code::INTERRUPTED,
        Err(error) => exit_code_for(error),
    }
}

fn report_error<W: Write>(
    writer: &mut W,
    phase: &str,
    code: i32,
    error: &SearchError,
) -> io::Result<()> {
    let value = serde_json::json!({
        "schema_version": "fsfs.live_search.error.v1",
        "command": "live-search",
        "phase": phase,
        "exit_code": code,
        "message": error.to_string(),
    });
    serde_json::to_writer(&mut *writer, &value).map_err(io::Error::other)?;
    writer.write_all(b"\n")?;
    writer.flush()
}

/// Called before ordinary argument/config processing so live errors never append
/// a normal command envelope to an already-started snapshot/delta stream.
pub fn entry(args: Vec<OsString>) -> i32 {
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        let result = io::stdout()
            .write_all(HELP.as_bytes())
            .map(|()| 0)
            .map_err(SearchError::Io);
        return result_exit_code(&result);
    }
    let options = match Options::parse(args) {
        Ok(options) => options,
        Err(error) => {
            let _ = report_error(
                &mut io::stderr(),
                "arguments",
                exit_code::USAGE_ERROR,
                &error,
            );
            return exit_code::USAGE_ERROR;
        }
    };
    let phase = if options.watch_source.is_some() {
        "watch_subscription"
    } else {
        "subscription"
    };
    let result = run(options);
    let code = result_exit_code(&result);
    if code != exit_code::OK
        && let Err(error) = &result
    {
        let _ = report_error(&mut io::stderr(), phase, code, error);
    }
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(extra: &[&str]) -> Vec<OsString> {
        ["--index-dir", "/explicit/store", "--query", "alpha"]
            .into_iter()
            .chain(extra.iter().copied())
            .map(OsString::from)
            .collect()
    }

    #[test]
    fn defaults_are_bounded_and_do_not_imply_a_model() {
        let options = Options::parse(args(&[])).unwrap();
        assert_eq!(options.root, PathBuf::from("/explicit/store"));
        assert_eq!(options.query, "alpha");
        assert_eq!(options.limits.max_results, 20);
        assert_eq!(options.poll_interval, Duration::from_millis(100));
        assert_eq!(options.refresh, LiveSearchRefreshConfig::default());
        assert_eq!(options.max_updates, None);
        assert_eq!(options.timeout, None);
        assert!(!options.once);
        assert!(!options.hybrid);
        assert!(options.config_path.is_none());
        assert!(options.watch_source.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn watch_requires_explicit_indexing_and_model_policy() {
        assert!(Options::parse(args(&["--watch-source", "/source"])).is_err());
        assert!(Options::parse(args(&["--watch-source", "/source", "--hybrid"])).is_err());
        let options =
            Options::parse(args(&["--watch-source", "/source", "--hybrid", "--once"])).unwrap();
        assert_eq!(options.watch_source, Some(PathBuf::from("/source")));
        assert_eq!(options.max_updates, Some(1));
        assert!(Options::parse(args(&["--hybrid", "--watch-source"])).is_err());
        assert!(
            Options::parse(args(&[
                "--hybrid",
                "--watch-source",
                "/a",
                "--watch-source",
                "/b",
            ]))
            .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn watch_does_not_silently_ignore_selection_poll_tuning() {
        for flag in ["--poll-ms", "--debounce-ms", "--max-wait-ms"] {
            assert!(
                Options::parse(args(&[
                    "--hybrid",
                    "--watch-source",
                    "/source",
                    "--once",
                    flag,
                    "100",
                ]))
                .is_err(),
                "{flag}"
            );
        }
        let options = Options::parse(args(&[
            "--hybrid",
            "--watch-source",
            "/source",
            "--min-score-delta",
            "0.1",
            "--limit",
            "5",
            "--max-updates",
            "2",
            "--timeout-ms",
            "10000",
        ]))
        .unwrap();
        assert_eq!(options.limits.max_results, 5);
        assert_eq!(options.max_updates, Some(2));
        assert_eq!(options.timeout, Some(Duration::from_secs(10)));
    }

    #[cfg(not(unix))]
    #[test]
    fn watch_is_refused_on_unsupported_platforms_before_startup() {
        assert!(Options::parse(args(&["--hybrid", "--watch-source", "source"])).is_err());
    }

    #[test]
    fn hybrid_and_configuration_are_explicit_and_order_independent() {
        let options = Options::parse(args(&[
            "--config",
            "/models/fsfs.toml",
            "--hybrid",
            "--once",
        ]))
        .unwrap();
        assert!(options.hybrid);
        assert_eq!(
            options.config_path,
            Some(PathBuf::from("/models/fsfs.toml"))
        );
        assert_eq!(options.max_updates, Some(1));
        assert!(Options::parse(args(&["--config", "/not-opened"])).is_err());
        assert!(Options::parse(args(&["--hybrid", "--hybrid"])).is_err());
        assert!(Options::parse(args(&["--hybrid", "--config"])).is_err());
    }

    #[test]
    fn explicit_options_reach_the_session_configuration() {
        let options = Options::parse(args(&[
            "--limit",
            "100",
            "--poll-ms",
            "25",
            "--debounce-ms",
            "50",
            "--max-wait-ms",
            "500",
            "--min-score-delta",
            "0.025",
            "--max-updates",
            "3",
            "--timeout-ms",
            "2000",
            "--format",
            "jsonl",
        ]))
        .unwrap();
        assert_eq!(options.limits.max_results, 100);
        assert_eq!(
            options.limits.min_score_delta.to_bits(),
            0.025_f64.to_bits()
        );
        assert_eq!(options.refresh.debounce, Duration::from_millis(50));
        assert_eq!(options.refresh.max_wait, Duration::from_millis(500));
        assert_eq!(options.poll_interval, Duration::from_millis(25));
        assert_eq!(options.timeout, Some(Duration::from_secs(2)));
        assert_eq!(options.max_updates, Some(3));
    }

    #[test]
    fn flags_cannot_silently_enable_unsupported_search_modes() {
        for flag in [
            "--rerank",
            "--filter",
            "--semantic",
            "--daemon",
            "--config",
            "--watch",
        ] {
            assert!(Options::parse(args(&[flag])).is_err(), "{flag}");
        }
        for format in ["json", "table", "csv", "toon"] {
            assert!(Options::parse(args(&["--format", format])).is_err());
        }
    }

    #[test]
    fn invalid_limits_durations_and_score_thresholds_are_refused() {
        for (flag, value) in [
            ("--limit", "0"),
            ("--limit", "10001"),
            ("--poll-ms", "0"),
            ("--poll-ms", "60001"),
            ("--debounce-ms", "-1"),
            ("--max-wait-ms", "0"),
            ("--max-updates", "0"),
            ("--max-updates", "18446744073709551616"),
            ("--timeout-ms", "0"),
            ("--timeout-ms", "86400001"),
            ("--min-score-delta", "NaN"),
            ("--min-score-delta", "inf"),
            ("--min-score-delta", "-0.01"),
            ("--limit", "one"),
        ] {
            assert!(
                Options::parse(args(&[flag, value])).is_err(),
                "{flag} {value}"
            );
        }
        assert!(Options::parse(args(&["--debounce-ms", "501", "--max-wait-ms", "500"])).is_err());
        assert!(Options::parse(args(&["--once", "--max-updates", "1"])).is_err());
        assert_eq!(
            Options::parse(args(&["--once"])).unwrap().max_updates,
            Some(1)
        );
    }

    #[test]
    fn duplicate_missing_and_blank_arguments_are_refused() {
        for extra in [
            vec!["--query", "beta"],
            vec!["--limit"],
            vec!["--once", "--once"],
        ] {
            assert!(Options::parse(args(&extra)).is_err());
        }
        assert!(Options::parse(Vec::new()).is_err());
        assert!(Options::parse(vec!["--query".into(), "alpha".into()]).is_err());
        for query in [
            String::new(),
            "   ".to_owned(),
            "x".repeat(MAX_QUERY_BYTES + 1),
        ] {
            assert!(
                Options::parse(vec![
                    "--index-dir".into(),
                    "/store".into(),
                    "--query".into(),
                    query.into(),
                ])
                .is_err()
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn filesystem_paths_need_not_be_utf8_but_query_text_must_be() {
        use std::os::unix::ffi::OsStringExt;
        let path = OsString::from_vec(b"/store-\xff".to_vec());
        let options = Options::parse(vec![
            "--index-dir".into(),
            path.clone(),
            "--query".into(),
            "alpha".into(),
        ])
        .unwrap();
        assert_eq!(options.root.into_os_string(), path);
        assert!(
            Options::parse(vec![
                "--index-dir".into(),
                "/store".into(),
                "--query".into(),
                OsString::from_vec(vec![255]),
            ])
            .is_err()
        );
    }

    #[test]
    fn error_output_is_one_json_record_even_with_control_characters() {
        let mut bytes = Vec::new();
        report_error(
            &mut bytes,
            "arguments",
            2,
            &invalid("bad\n\u{1b}[2J\"query"),
        )
        .unwrap();
        assert_eq!(bytes.split(|&byte| byte == b'\n').count(), 2);
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["command"], "live-search");
        assert_eq!(value["exit_code"], 2);
        assert!(!bytes.contains(&0x1b));
    }

    #[test]
    fn broken_pipe_is_success_but_other_output_errors_are_failures() {
        let broken = Err(SearchError::Io(io::Error::from(io::ErrorKind::BrokenPipe)));
        let full = Err(SearchError::Io(io::Error::other("disk full")));
        assert_eq!(result_exit_code(&broken), 0);
        assert_ne!(result_exit_code(&full), 0);
        let cancelled = Err(SearchError::Cancelled {
            phase: "test".into(),
            reason: "stop".into(),
        });
        assert_eq!(result_exit_code(&cancelled), 130);
    }

    #[test]
    fn expired_budget_and_cancellation_block_output_before_any_bytes() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let budget = Budget {
                started: Instant::now(),
                timeout: Some(Duration::ZERO),
            };
            let mut bytes = Vec::new();
            let mut writer = GuardedOutput {
                writer: &mut bytes,
                cx: &cx,
                budget: &budget,
            };
            assert!(writer.write_all(b"must not escape").is_err());
            assert!(bytes.is_empty());
            cx.set_cancel_requested(true);
            let budget = Budget {
                started: Instant::now(),
                timeout: None,
            };
            let mut writer = GuardedOutput {
                writer: &mut bytes,
                cx: &cx,
                budget: &budget,
            };
            assert!(writer.write_all(b"also refused").is_err());
            assert!(bytes.is_empty());
            assert!(matches!(
                budget.check(&cx),
                Err(SearchError::Cancelled { .. })
            ));
        });
    }

    #[cfg(unix)]
    mod native {
        use asupersync::test_utils::run_test_with_cx;
        use frankensearch_core::{IndexableDocument, LexicalWrite};
        use frankensearch_fsfs::adapters::live_search::{LiveSearchChange, LiveSearchEvent};
        use frankensearch_fsfs::adapters::quill_live_search::QuillLiveSearchFrame;
        use frankensearch_fsfs::generation_store::{
            COMPLETE_GENERATION_POINTER, GenerationBuild, GenerationPublication,
        };
        use frankensearch_quill::QuillIndex;

        use super::*;

        fn options(root: &std::path::Path, extra: &[&str]) -> Options {
            let mut values = vec![
                OsString::from("--index-dir"),
                root.as_os_str().to_owned(),
                OsString::from("--query"),
                OsString::from("alpha"),
                OsString::from("--poll-ms"),
                OsString::from("10"),
                OsString::from("--debounce-ms"),
                OsString::from("0"),
                OsString::from("--timeout-ms"),
                OsString::from("2000"),
            ];
            values.extend(extra.iter().map(OsString::from));
            Options::parse(values).unwrap()
        }

        async fn staged(cx: &Cx, store: &CompleteGenerationStore, id: &str) -> GenerationBuild {
            let build = store.begin(cx).unwrap();
            let config = QuillConfig {
                deterministic_ingest: true,
                ..QuillConfig::default()
            };
            let index = QuillIndex::create(cx, &build.path().join("lexical"), config)
                .await
                .unwrap();
            let document = IndexableDocument::new(id, "alpha immutable search result")
                .with_metadata("path", format!("/{id}.rs"));
            LexicalWrite::index_documents(&index, cx, &[document])
                .await
                .unwrap();
            LexicalWrite::commit(&index, cx).await.unwrap();
            drop(index);
            build
        }

        fn publish(cx: &Cx, build: GenerationBuild) {
            assert!(matches!(
                build.publish(cx, |_, _| Ok(())).unwrap(),
                GenerationPublication::Durable(_)
            ));
        }

        fn frames(bytes: &[u8]) -> Vec<QuillLiveSearchFrame> {
            bytes
                .split(|byte| *byte == b'\n')
                .filter(|line| !line.is_empty())
                .map(|line| serde_json::from_slice(line).unwrap())
                .collect()
        }

        #[test]
        fn once_emits_a_real_snapshot_without_changing_the_selection() {
            run_test_with_cx(|cx| async move {
                let root = tempfile::tempdir().unwrap();
                let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
                publish(&cx, staged(&cx, &store, "old").await);
                let pointer = std::fs::read(root.path().join(COMPLETE_GENERATION_POINTER)).unwrap();
                let mut output = Vec::new();
                assert_eq!(
                    execute(&cx, &options(root.path(), &["--once"]), &mut output)
                        .await
                        .unwrap(),
                    1
                );
                let frames = frames(&output);
                assert_eq!(frames.len(), 1);
                assert_eq!(frames[0].sequence, 1);
                assert_eq!(frames[0].previous_generation, None);
                let LiveSearchEvent::Snapshot { results } = &frames[0].event else {
                    panic!("first live frame must be a snapshot");
                };
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].rank, 1);
                assert_eq!(results[0].hit.doc_id, "old");
                assert_eq!(results[0].hit.item.as_ref().unwrap()["path"], "/old.rs");
                assert_eq!(
                    std::fs::read(root.path().join(COMPLETE_GENERATION_POINTER)).unwrap(),
                    pointer
                );
                // Querying never retained the publisher lease.
                drop(store.begin(&cx).unwrap());
            });
        }

        struct PublishAfterSnapshot {
            bytes: Vec<u8>,
            successor: Option<GenerationBuild>,
            cx: Cx,
        }

        impl Write for PublishAfterSnapshot {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.bytes.extend_from_slice(bytes);
                Ok(bytes.len())
            }

            fn flush(&mut self) -> io::Result<()> {
                if let Some(successor) = self.successor.take() {
                    match successor
                        .publish(&self.cx, |_, _| Ok(()))
                        .map_err(io::Error::other)?
                    {
                        GenerationPublication::Durable(_) => {}
                        GenerationPublication::VisibleButDurabilityUncertain { source, .. } => {
                            return Err(source);
                        }
                    }
                }
                Ok(())
            }
        }

        #[test]
        fn a_real_successor_produces_additions_and_removals_in_the_second_frame() {
            run_test_with_cx(|cx| async move {
                let root = tempfile::tempdir().unwrap();
                let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
                publish(&cx, staged(&cx, &store, "old").await);
                let successor = staged(&cx, &store, "new").await;
                // Only publish after the first frame is delivered. This tests
                // the actual long-lived command, not two independent searches.
                let mut output = PublishAfterSnapshot {
                    bytes: Vec::new(),
                    successor: Some(successor),
                    cx: cx.clone(),
                };
                let count = execute(
                    &cx,
                    &options(root.path(), &["--max-updates", "2"]),
                    &mut output,
                )
                .await
                .unwrap();
                assert_eq!(count, 2);
                let frames = frames(&output.bytes);
                assert_eq!(frames.len(), 2);
                assert_eq!(frames[1].sequence, 2);
                assert_eq!(
                    frames[1].previous_generation.as_deref(),
                    Some(frames[0].generation.as_str())
                );
                assert_ne!(frames[0].generation, frames[1].generation);
                assert_eq!(frames[1].result_count, 1);
                let LiveSearchEvent::Delta { changes } = &frames[1].event else {
                    panic!("successor must produce a delta");
                };
                assert_eq!(changes.len(), 2);
                assert!(changes.iter().any(|change| matches!(change,
                    LiveSearchChange::Removed { doc_id, previous_rank: 1 } if doc_id == "old")));
                assert!(changes.iter().any(|change| matches!(change,
                    LiveSearchChange::Added { result } if result.hit.doc_id == "new")));
            });
        }

        #[test]
        fn once_refuses_an_unselected_store_without_stdout_or_new_files() {
            run_test_with_cx(|cx| async move {
                let root = tempfile::tempdir().unwrap();
                let _store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
                let mut output = Vec::new();
                assert!(matches!(
                    execute(&cx, &options(root.path(), &["--once"]), &mut output).await,
                    Err(SearchError::IndexNotFound { .. })
                ));
                assert!(output.is_empty());
                assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
            });
        }

        #[test]
        fn missing_root_is_not_initialized_by_the_command() {
            run_test_with_cx(|cx| async move {
                let parent = tempfile::tempdir().unwrap();
                let root = parent.path().join("absent");
                let mut output = Vec::new();
                assert!(
                    execute(&cx, &options(&root, &["--once"]), &mut output)
                        .await
                        .is_err()
                );
                assert!(!root.exists());
                assert!(output.is_empty());
            });
        }

        #[test]
        fn cancelled_command_cannot_publish_its_first_frame() {
            run_test_with_cx(|cx| async move {
                let root = tempfile::tempdir().unwrap();
                let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
                publish(&cx, staged(&cx, &store, "retained").await);
                cx.set_cancel_requested(true);
                let mut output = Vec::new();
                assert!(matches!(
                    execute(&cx, &options(root.path(), &["--once"]), &mut output).await,
                    Err(SearchError::Cancelled { .. })
                ));
                assert!(output.is_empty());
                cx.set_cancel_requested(false);
                assert_eq!(
                    execute(&cx, &options(root.path(), &["--once"]), &mut output)
                        .await
                        .unwrap(),
                    1
                );
            });
        }

        struct FailFlush {
            bytes: Vec<u8>,
            flushes: usize,
        }

        impl Write for FailFlush {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.bytes.extend_from_slice(bytes);
                Ok(bytes.len())
            }

            fn flush(&mut self) -> io::Result<()> {
                self.flushes += 1;
                Err(io::Error::other("injected flush failure"))
            }
        }

        #[test]
        fn failed_transport_stops_without_retry_or_appended_error_frame() {
            run_test_with_cx(|cx| async move {
                let root = tempfile::tempdir().unwrap();
                let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
                publish(&cx, staged(&cx, &store, "old").await);
                let mut output = FailFlush {
                    bytes: Vec::new(),
                    flushes: 0,
                };
                let error = execute(&cx, &options(root.path(), &[]), &mut output)
                    .await
                    .unwrap_err();
                assert!(matches!(error, SearchError::Io(_)));
                assert_eq!(output.flushes, 1);
                // Full bytes may have escaped before a flush failure, but are
                // never followed by a retry or a different envelope schema.
                assert_eq!(frames(&output.bytes).len(), 1);
                drop(store.begin(&cx).unwrap());
            });
        }

        #[test]
        fn corrupted_descriptor_is_not_replaced_or_reported_as_empty_results() {
            run_test_with_cx(|cx| async move {
                let root = tempfile::tempdir().unwrap();
                let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
                publish(&cx, staged(&cx, &store, "old").await);
                let pointer = root.path().join(COMPLETE_GENERATION_POINTER);
                std::fs::write(&pointer, b"fault injection: malformed selection").unwrap();
                let mut output = Vec::new();
                assert!(
                    execute(&cx, &options(root.path(), &[]), &mut output)
                        .await
                        .is_err()
                );
                assert!(output.is_empty());
                assert_eq!(
                    std::fs::read(&pointer).unwrap(),
                    b"fault injection: malformed selection"
                );
            });
        }
    }
}
