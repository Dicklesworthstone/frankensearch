//! Interactive, read-only subscriptions to an explicit complete-generation store.
//! No source discovery, publisher, persistent query cache, or private model loader.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use frankensearch_fsfs::adapters::live_search::{
    LIVE_SEARCH_SCHEMA_VERSION, LiveSearchChange, LiveSearchConfig, LiveSearchEvent,
    LiveSearchFrame,
};
use frankensearch_fsfs::adapters::live_search_session::LiveSearchRefreshConfig;
use frankensearch_fsfs::adapters::quill_live_search::QuillLiveSearchSession;
use frankensearch_fsfs::adapters::retained_live_search::{
    RetainedLiveSearchFrame, RetainedLiveSearchSession,
};
use frankensearch_fsfs::generation_store::CompleteGenerationStore;
use frankensearch_fsfs::{FsfsRuntime, exit_code_for};
use frankensearch_quill::QuillConfig;
use ftui_core::event::{Event, KeyCode, KeyEventKind, Modifiers};
use serde_json::Value;

const MAX_QUERY_BYTES: usize = 64 * 1024;
const MAX_RESULTS: usize = 1_000;
const INPUT_TICK: Duration = Duration::from_millis(25);
const HELP: &str = "fsfs-live --index-dir STORE --query TEXT [--hybrid] [--config FILE] [--limit N]

Interactive live search of an EXISTING complete-generation store (Unix terminals).
Default: native, model-free Quill. --hybrid uses configured, already available models.
--config requires --hybrid. --limit is 1..=1000 (default 20). No indexing or downloads.

Up/Down or j/k select; PageUp/PageDown scroll; Home/End jump.
/ edits the query; Ctrl-U clears; Enter searches; Escape cancels editing.
r explicitly retries/refreshes; q or Escape quits; Ctrl-C cancels.
A query edit abandons the previous request. Selection follows document identity,
not rank, through live updates. Refinement failure retains the Initial window.
Failed admission is shown explicitly; old results are marked stale until retry.
Both stdin and stdout must be terminals. No source file is modified.
";

#[derive(Debug, Clone)]
struct Options {
    root: PathBuf,
    query: String,
    limit: usize,
    hybrid: bool,
    config: Option<PathBuf>,
}

impl Options {
    fn parse(args: Vec<OsString>) -> SearchResult<Self> {
        let (mut root, mut query, mut config) = (None, None, None);
        let mut limit = 20;
        let mut hybrid = false;
        let mut seen = BTreeSet::new();
        let mut args = args.into_iter();
        while let Some(flag) = args.next() {
            let flag = flag
                .into_string()
                .map_err(|_| invalid("option names must be UTF-8"))?;
            if !seen.insert(flag.clone()) {
                return Err(invalid("duplicate option"));
            }
            if flag == "--hybrid" {
                hybrid = true;
                continue;
            }
            if !matches!(
                flag.as_str(),
                "--index-dir" | "--query" | "--config" | "--limit"
            ) {
                return Err(invalid("unknown option; run fsfs-live --help"));
            }
            let value = args
                .next()
                .filter(|value| !value.is_empty())
                .ok_or_else(|| invalid("missing or empty option value"))?;
            match flag.as_str() {
                "--index-dir" => root = Some(PathBuf::from(value)),
                "--config" => config = Some(PathBuf::from(value)),
                "--query" => {
                    query = Some(
                        value
                            .into_string()
                            .map_err(|_| invalid("query must be UTF-8"))?,
                    );
                }
                "--limit" => {
                    limit = value
                        .to_str()
                        .and_then(|value| value.parse::<usize>().ok())
                        .filter(|limit| (1..=MAX_RESULTS).contains(limit))
                        .ok_or_else(|| invalid("--limit must be in 1..=1000"))?;
                }
                _ => return Err(invalid("invalid value-bearing option")),
            }
        }
        let query = query.ok_or_else(|| invalid("--query is required"))?;
        validate_query(&query)?;
        if config.is_some() && !hybrid {
            return Err(invalid("--config requires --hybrid"));
        }
        Ok(Self {
            root: root.ok_or_else(|| invalid("--index-dir is required; no store is guessed"))?,
            query,
            limit,
            hybrid,
            config,
        })
    }
}

fn invalid(reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "live_search.tui".to_owned(),
        value: String::new(),
        reason: reason.to_owned(),
    }
}

// A hybrid frame's item is its metadata value. `Some` itself does not fit the
// higher-ranked `Fn(&T) -> Option<&Value>` bound; a named fn does.
#[allow(clippy::unnecessary_wraps)]
fn value_metadata(value: &Value) -> Option<&Value> {
    Some(value)
}

fn validate_query(query: &str) -> SearchResult<()> {
    if query.trim().is_empty() || query.len() > MAX_QUERY_BYTES {
        return Err(invalid("query must be nonblank and at most 64 KiB"));
    }
    Ok(())
}

fn checkpoint(cx: &Cx) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: "fsfs.live_search.tui".to_owned(),
        reason: error.to_string(),
    })
}

// All corpus/config/query text reaches the terminal as bounded literal widget
// text. Escape controls, OSC/CSI introducers and bidi controls, not just ESC.
fn literal(value: &str, max_chars: usize) -> String {
    let mut result = String::new();
    for ch in value.chars().take(max_chars) {
        if ch.is_control()
            || matches!(ch,
                '\u{061c}' | '\u{200e}' | '\u{200f}'
                | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
        {
            result.extend(ch.escape_default());
        } else {
            result.push(ch);
        }
    }
    result
}

#[derive(Debug, Clone, PartialEq)]
struct Row {
    id: String,
    rank: u64,
    score: f64,
    path: String,
    detail: String,
}

#[derive(Debug, Clone, PartialEq)]
struct View {
    query: String,
    draft: Option<String>,
    limit: usize,
    rows: Vec<Row>,
    selected: usize,
    sequence: u64,
    generation: Option<String>,
    status: String,
}

#[derive(Debug, PartialEq, Eq)]
enum Action {
    Continue,
    Query(String),
    Quit,
    Cancel,
}

impl View {
    fn new(query: String, limit: usize) -> Self {
        Self {
            query,
            limit,
            draft: None,
            rows: Vec::new(),
            selected: 0,
            sequence: 0,
            generation: None,
            status: "Waiting for a published generation".to_owned(),
        }
    }

    // Apply an entire frame transactionally; malformed or out-of-order updates
    // never leave a partially changed result window on screen.
    #[allow(clippy::too_many_lines)]
    fn update<T>(
        &mut self,
        frame: &LiveSearchFrame<T>,
        item: impl Fn(&T) -> Option<&Value>,
    ) -> SearchResult<()> {
        if frame.schema_version != LIVE_SEARCH_SCHEMA_VERSION
            || frame.query != self.query
            || frame.sequence
                != self
                    .sequence
                    .checked_add(1)
                    .ok_or_else(|| invalid("sequence exhausted"))?
            || frame.previous_generation != self.generation
            || frame.generation.is_empty()
            || self.generation.as_deref() == Some(frame.generation.as_str())
            || frame.result_count > self.limit
        {
            return Err(invalid(
                "live result revision does not extend the visible query",
            ));
        }
        let selected_id = self.rows.get(self.selected).map(|row| row.id.clone());
        let row = |hit: &frankensearch_fsfs::adapters::live_search::RankedLiveSearchHit<T>| -> SearchResult<Row> {
            if hit.hit.doc_id.is_empty()
                || hit.hit.doc_id.len() > MAX_QUERY_BYTES
                || !hit.hit.score.is_finite()
                || hit.rank == 0
                || hit.rank > self.limit as u64
            {
                return Err(invalid("invalid live result identity, rank or score"));
            }
            let metadata = item(&hit.hit.item);
            let path = metadata
                .and_then(|value| value.get("path"))
                .and_then(Value::as_str)
                .unwrap_or(&hit.hit.doc_id);
            let snippet = metadata
                .and_then(|value| value.get("snippet"))
                .and_then(Value::as_str)
                .unwrap_or("");
            Ok(Row {
                id: hit.hit.doc_id.clone(),
                rank: hit.rank,
                score: hit.hit.score,
                path: literal(path, 4_096),
                detail: literal(snippet, 4_096),
            })
        };
        let mut rows: BTreeMap<String, Row> = match &frame.event {
            LiveSearchEvent::Snapshot { .. } if self.sequence == 0 => BTreeMap::new(),
            LiveSearchEvent::Delta { .. } if self.sequence != 0 => self
                .rows
                .iter()
                .cloned()
                .map(|row| (row.id.clone(), row))
                .collect(),
            _ => return Err(invalid("expected one initial snapshot followed by deltas")),
        };
        let mut changed = BTreeSet::new();
        match &frame.event {
            LiveSearchEvent::Snapshot { results } => {
                if results.len() > self.limit {
                    return Err(invalid("snapshot exceeds window"));
                }
                for hit in results {
                    let hit = row(hit)?;
                    if rows.insert(hit.id.clone(), hit).is_some() {
                        return Err(invalid("duplicate snapshot identity"));
                    }
                }
            }
            LiveSearchEvent::Delta { changes } => {
                if changes.len() > self.limit.saturating_mul(2) {
                    return Err(invalid("delta exceeds window"));
                }
                for change in changes {
                    let id = match change {
                        LiveSearchChange::Added { result }
                        | LiveSearchChange::Updated { result, .. } => &result.hit.doc_id,
                        LiveSearchChange::Removed { doc_id, .. } => doc_id,
                    };
                    if !changed.insert(id) {
                        return Err(invalid("duplicate delta identity"));
                    }
                    match change {
                        LiveSearchChange::Added { result } => {
                            let hit = row(result)?;
                            if rows.insert(hit.id.clone(), hit).is_some() {
                                return Err(invalid("added identity already exists"));
                            }
                        }
                        LiveSearchChange::Removed {
                            doc_id,
                            previous_rank,
                        } => {
                            if rows
                                .remove(doc_id)
                                .is_none_or(|row| row.rank != *previous_rank)
                            {
                                return Err(invalid("removal does not match the visible baseline"));
                            }
                        }
                        LiveSearchChange::Updated {
                            result,
                            previous_rank,
                        } => {
                            let hit = row(result)?;
                            if rows
                                .insert(hit.id.clone(), hit)
                                .is_none_or(|old| old.rank != *previous_rank)
                            {
                                return Err(invalid("update does not match the visible baseline"));
                            }
                        }
                    }
                }
            }
        }
        let mut rows = rows.into_values().collect::<Vec<_>>();
        rows.sort_by_key(|row| row.rank);
        if rows.len() != frame.result_count
            || rows
                .iter()
                .enumerate()
                .any(|(offset, row)| row.rank != offset as u64 + 1)
        {
            return Err(invalid("live result ranks or count are inconsistent"));
        }
        self.selected = selected_id
            .as_ref()
            .and_then(|id| rows.iter().position(|row| &row.id == id))
            .unwrap_or_else(|| self.selected.min(rows.len().saturating_sub(1)));
        self.rows = rows;
        self.sequence = frame.sequence;
        self.generation = Some(frame.generation.clone());
        Ok(())
    }

    fn hybrid(&mut self, frame: &RetainedLiveSearchFrame) -> SearchResult<()> {
        if let Some(update) = &frame.update {
            self.update(update, value_metadata)?;
        }
        let semantic = frame
            .annotations
            .get("semantic_admitted")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let reason = frame
            .annotations
            .get("skip_reason")
            .and_then(Value::as_str)
            .unwrap_or("");
        self.status = format!(
            "{} | semantic admitted: {semantic} | {}",
            frame.phase,
            literal(reason, 256)
        );
        Ok(())
    }

    #[allow(clippy::needless_pass_by_value)]
    fn event(&mut self, event: Event) -> Action {
        if let Event::Paste(paste) = &event {
            if let Some(draft) = self.draft.as_mut() {
                for ch in paste.text.chars() {
                    let ch = if ch.is_whitespace() { ' ' } else { ch };
                    if ch.is_control() {
                        continue;
                    }
                    if draft.len().saturating_add(ch.len_utf8()) > MAX_QUERY_BYTES {
                        break;
                    }
                    draft.push(ch);
                }
            }
            return Action::Continue;
        }
        let Event::Key(key) = event else {
            return Action::Continue;
        };
        if key.kind == KeyEventKind::Release {
            return Action::Continue;
        }
        if key.code == KeyCode::Char('c') && key.modifiers.contains(Modifiers::CTRL) {
            return Action::Cancel;
        }
        if let Some(draft) = self.draft.as_mut() {
            match key.code {
                KeyCode::Escape => self.draft = None,
                KeyCode::Enter => {
                    if validate_query(draft).is_ok() {
                        return Action::Query(draft.clone());
                    }
                    "Query must be nonblank and at most 64 KiB".clone_into(&mut self.status);
                }
                KeyCode::Backspace => {
                    draft.pop();
                }
                KeyCode::Char('u') if key.modifiers.contains(Modifiers::CTRL) => draft.clear(),
                KeyCode::Char(ch)
                    if !ch.is_control()
                        && !key
                            .modifiers
                            .intersects(Modifiers::CTRL | Modifiers::ALT | Modifiers::SUPER)
                        && draft.len().saturating_add(ch.len_utf8()) <= MAX_QUERY_BYTES =>
                {
                    draft.push(ch);
                }
                _ => {}
            }
            return Action::Continue;
        }
        match key.code {
            KeyCode::Char('q') | KeyCode::Escape => return Action::Quit,
            KeyCode::Char('/') => self.draft = Some(self.query.clone()),
            KeyCode::Char('r') => return Action::Query(self.query.clone()),
            KeyCode::Up | KeyCode::Char('k') => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => self.selected = self.selected.saturating_add(1),
            KeyCode::PageUp => self.selected = self.selected.saturating_sub(10),
            KeyCode::PageDown => self.selected = self.selected.saturating_add(10),
            KeyCode::Home => self.selected = 0,
            KeyCode::End => self.selected = self.rows.len().saturating_sub(1),
            _ => {}
        }
        self.selected = self.selected.min(self.rows.len().saturating_sub(1));
        Action::Continue
    }
}

trait Screen: Send {
    fn draw(&mut self, view: &View) -> SearchResult<()>;
    fn event(&mut self) -> SearchResult<Option<Event>>;
}

struct Ui<S> {
    screen: S,
    view: View,
}

impl<S: Screen> Ui<S> {
    fn draw(&mut self) -> SearchResult<()> {
        self.screen.draw(&self.view).map_err(terminal_error)
    }

    fn input(&mut self) -> SearchResult<Action> {
        // Bound each drain so sustained keyboard input cannot starve search.
        for _ in 0..32 {
            let Some(event) = self.screen.event().map_err(terminal_error)? else {
                break;
            };
            let action = self.view.event(event);
            self.draw()?;
            if action != Action::Continue {
                return Ok(action);
            }
        }
        Ok(Action::Continue)
    }
}

fn terminal_error(source: SearchError) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "fsfs.live_search.tui.terminal",
        source: Box::new(source),
    }
}

fn lock<S>(ui: &Mutex<Ui<S>>) -> SearchResult<MutexGuard<'_, Ui<S>>> {
    ui.lock()
        .map_err(|_| invalid("terminal state was poisoned"))
}

enum Session {
    Lexical(Box<QuillLiveSearchSession>),
    Hybrid(Box<RetainedLiveSearchSession>),
}

impl Session {
    fn new(
        store: CompleteGenerationStore,
        options: &Options,
        query: &str,
        runtime: Option<FsfsRuntime>,
    ) -> SearchResult<Self> {
        let limits = LiveSearchConfig {
            max_results: options.limit,
            min_score_delta: 0.0,
        };
        match (options.hybrid, runtime) {
            (false, None) => Ok(Self::Lexical(Box::new(QuillLiveSearchSession::new(
                store,
                query,
                limits,
                LiveSearchRefreshConfig::default(),
                QuillConfig::default(),
            )?))),
            (true, Some(runtime)) => Ok(Self::Hybrid(Box::new(RetainedLiveSearchSession::new(
                runtime,
                store,
                query,
                limits,
                LiveSearchRefreshConfig::default(),
            )?))),
            _ => Err(invalid("query mode and runtime disagree")),
        }
    }

    async fn poll<S: Screen>(
        &mut self,
        cx: &Cx,
        now: Instant,
        ui: &Mutex<Ui<S>>,
    ) -> SearchResult<()> {
        match self {
            Self::Lexical(session) => {
                if let Some(frame) = session.poll(cx, now).await? {
                    let mut ui = lock(ui)?;
                    ui.view.update(&frame, Option::as_ref)?;
                    "Lexical | no semantic model used".clone_into(&mut ui.view.status);
                    ui.draw()?;
                }
            }
            Self::Hybrid(session) => {
                let mut sink = |frame: &RetainedLiveSearchFrame| {
                    let mut ui = lock(ui)?;
                    ui.view.hybrid(frame)?;
                    // Initial reaches the terminal before the reader awaits
                    // refinement. A failed draw does not acknowledge that phase.
                    ui.draw()
                };
                session.poll_with_sink(cx, now, &mut sink).await?;
            }
        }
        Ok(())
    }
}

// Own and drop the query future BEFORE returning a new-query/quit action.
// No old sink can run after the caller resets the visible query and subscription.
async fn responsive<S: Screen, F: std::future::Future<Output = SearchResult<()>> + Send>(
    cx: &Cx,
    ui: &Mutex<Ui<S>>,
    work: F,
) -> SearchResult<Option<Action>> {
    let mut work = std::pin::pin!(work);
    loop {
        checkpoint(cx)?;
        let action = lock(ui)?.input()?;
        if action != Action::Continue {
            return Ok(Some(action));
        }
        let mut tick = std::pin::pin!(asupersync::time::sleep(cx.now(), INPUT_TICK));
        let completed = std::future::poll_fn(|task_cx| {
            if let std::task::Poll::Ready(result) =
                std::future::Future::poll(work.as_mut(), task_cx)
            {
                return std::task::Poll::Ready(Some(result));
            }
            if std::future::Future::poll(tick.as_mut(), task_cx).is_ready() {
                return std::task::Poll::Ready(None);
            }
            std::task::Poll::Pending
        })
        .await;
        if let Some(result) = completed {
            result?;
            return Ok(None);
        }
    }
}

async fn execute<S: Screen>(
    cx: &Cx,
    options: &Options,
    runtime: Option<FsfsRuntime>,
    screen: S,
) -> SearchResult<()> {
    checkpoint(cx)?;
    let store = CompleteGenerationStore::open(cx, &options.root)?;
    let ui = Mutex::new(Ui {
        screen,
        view: View::new(options.query.clone(), options.limit),
    });
    lock(&ui)?.draw()?;
    let mut session = Session::new(store.clone(), options, &options.query, runtime.clone())?;
    let mut failed = false;
    loop {
        let outcome = if failed {
            responsive(cx, &ui, std::future::pending::<SearchResult<()>>()).await
        } else {
            responsive(cx, &ui, async {
                session.poll(cx, Instant::now(), &ui).await?;
                asupersync::time::sleep(cx.now(), Duration::from_millis(100)).await;
                Ok(())
            })
            .await
        };
        let action = match outcome {
            Ok(action) => action,
            Err(error @ SearchError::Cancelled { .. }) => return Err(error),
            Err(
                error @ SearchError::SubsystemError {
                    subsystem: "fsfs.live_search.tui.terminal",
                    ..
                },
            ) => return Err(error),
            Err(error) => {
                // No implicit retry or stale-result fallback. The last window
                // remains explicitly stale; only r or a new query tries again.
                let mut ui = lock(&ui)?;
                ui.view.status = format!(
                    "STALE / query stopped: {} | r retries",
                    literal(&error.to_string(), 512)
                );
                ui.draw()?;
                drop(ui);
                failed = true;
                continue;
            }
        };
        match action {
            Some(Action::Quit) => return Ok(()),
            Some(Action::Cancel) => {
                return Err(SearchError::Cancelled {
                    phase: "fsfs.live_search.tui".to_owned(),
                    reason: "Ctrl-C".to_owned(),
                });
            }
            Some(Action::Query(query)) => {
                validate_query(&query)?;
                session = Session::new(store.clone(), options, &query, runtime.clone())?;
                let mut ui = lock(&ui)?;
                ui.view = View::new(query, options.limit);
                ui.draw()?;
                drop(ui);
                failed = false;
            }
            _ => {}
        }
    }
}

fn configured_runtime(options: &Options) -> SearchResult<Option<FsfsRuntime>> {
    if !options.hybrid {
        return Ok(None);
    }
    use frankensearch_fsfs::{
        CliCommand, CliInput, CliOverrides, OutputFormat, current_unicode_environment,
        default_project_config_file_path, default_user_config_file_path, load_from_layered_sources,
        load_from_sources,
    };
    let home = frankensearch_core::platform_dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"));
    let env = current_unicode_environment();
    let overrides = CliOverrides::default();
    let loaded = if let Some(path) = &options.config {
        if !std::fs::metadata(path)?.is_file() {
            return Err(invalid("--config must name a regular file"));
        }
        load_from_sources(Some(path), &env, &overrides, &home)?
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
        quiet: true,
        no_color: true,
        format: OutputFormat::Jsonl,
        ..CliInput::default()
    })))
}

#[cfg(unix)]
mod terminal {
    use super::{Duration, Event, Screen, SearchResult, View, literal};
    use ftui_backend::{Backend, BackendEventSource, BackendFeatures, BackendPresenter};
    use ftui_core::geometry::Rect;
    use ftui_render::grapheme_pool::GraphemePool;
    use ftui_render::{buffer::Buffer, diff::BufferDiff, frame::Frame};
    use ftui_text::{Line, Text};
    use ftui_tty::{TtyBackend, TtySessionOptions};
    use ftui_widgets::{Widget, paragraph::Paragraph};

    pub(super) struct Terminal {
        backend: TtyBackend,
        // ftui-tty 0.7 exposes no presenter pool and presents without one, so
        // the terminal owns its frames' pool, as the fsfs dashboard does.
        grapheme_pool: GraphemePool,
        previous: Option<(u16, u16, Buffer)>,
    }

    impl Terminal {
        pub(super) fn open() -> SearchResult<Self> {
            let backend = TtyBackend::open(
                80,
                24,
                TtySessionOptions {
                    alternate_screen: true,
                    intercept_signals: false,
                    features: BackendFeatures {
                        bracketed_paste: true,
                        ..BackendFeatures::default()
                    },
                },
            )?;
            Ok(Self {
                backend,
                grapheme_pool: GraphemePool::new(),
                previous: None,
            })
        }
    }

    impl Screen for Terminal {
        fn event(&mut self) -> SearchResult<Option<Event>> {
            if !self.backend.poll_event(Duration::ZERO)? {
                return Ok(None);
            }
            Ok(self.backend.read_event()?)
        }

        fn draw(&mut self, view: &View) -> SearchResult<()> {
            let (width, height) = self.backend.size()?;
            let (width, height) = (width.min(512), height.min(256));
            if width == 0 || height == 0 {
                return Ok(());
            }
            let buffer = {
                let mut frame = Frame::new(width, height, &mut self.grapheme_pool);
                let query = view.draft.as_deref().unwrap_or(&view.query);
                let heading = if view.draft.is_some() {
                    "EDIT QUERY (Enter submits, Esc cancels)"
                } else {
                    "fsfs-live | read-only live search"
                };
                let generation = view.generation.as_deref().map_or("none", |value| value);
                let lines = vec![
                    Line::from(heading),
                    Line::from(literal(query, 512)),
                    Line::from(literal(&view.status, 512)),
                    Line::from(format!(
                        "{} results | revision {} | {}",
                        view.rows.len(),
                        view.sequence,
                        literal(generation, 80)
                    )),
                ];
                Paragraph::new(Text::from_lines(lines))
                    .render(Rect::new(0, 0, width, height.min(4)), &mut frame);
                let body_height = height.saturating_sub(7);
                let first = view
                    .selected
                    .saturating_sub(usize::from(body_height).saturating_sub(1));
                let rows = view
                    .rows
                    .iter()
                    .enumerate()
                    .skip(first)
                    .take(usize::from(body_height))
                    .map(|(offset, row)| {
                        Line::from(format!(
                            "{} {:>4}  {:.5}  {}",
                            if offset == view.selected { ">" } else { " " },
                            row.rank,
                            row.score,
                            row.path
                        ))
                    })
                    .collect::<Vec<_>>();
                if body_height > 0 {
                    Paragraph::new(Text::from_lines(rows))
                        .render(Rect::new(0, 4, width, body_height), &mut frame);
                }
                if height >= 3 {
                    let detail = view
                        .rows
                        .get(view.selected)
                        .map_or("", |row| row.detail.as_str());
                    Paragraph::new(Text::from_lines(vec![
                        Line::from(detail),
                        Line::from(
                            "Up/Down j/k select | / query | r retry | q/Esc quit | Ctrl-C cancel",
                        ),
                    ]))
                    .render(Rect::new(0, height - 3, width, 3), &mut frame);
                }
                frame.buffer
            };
            let diff = self
                .previous
                .as_ref()
                .filter(|(old_width, old_height, _)| *old_width == width && *old_height == height)
                .map(|(_, _, old)| BufferDiff::compute(old, &buffer));
            self.backend
                .presenter()
                .present_ui(&buffer, diff.as_ref(), diff.is_none())?;
            self.previous = Some((width, height, buffer));
            Ok(())
        }
    }
}

#[cfg(unix)]
fn run(options: Options) -> SearchResult<()> {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return Err(invalid(
            "fsfs-live requires terminal stdin and stdout; use fsfs live-search for NDJSON",
        ));
    }
    use asupersync::runtime::RuntimeBuilder;
    use frankensearch_fsfs::{ShutdownCoordinator, runtime::SearchBlockingPool};
    let pool = Arc::new(SearchBlockingPool::default());
    let runtime = configured_runtime(&options)?;
    #[cfg(feature = "rerank")]
    let runtime = runtime.map(|runtime| runtime.with_native_blocking_pool(pool.handle()));
    let scheduler = RuntimeBuilder::current_thread()
        .blocking_threads(0, 2)
        .build()
        .map_err(|error| invalid(&error.to_string()))?;
    let shutdown = Arc::new(ShutdownCoordinator::new());
    shutdown.register_signals()?;
    let request_shutdown = Arc::clone(&shutdown);
    let request_pool = Arc::clone(&pool);
    let task = scheduler.handle().spawn(async move {
        let current = Cx::current().ok_or_else(|| invalid("runtime did not install Cx"))?;
        let cx = request_pool.context(current);
        let scope = request_shutdown.cancellation_scope(&cx);
        // The backend is owned inside the task. Its Drop restores the terminal
        // before errors are printed, including failed opens/queries and signals.
        let screen = terminal::Terminal::open()?;
        execute(&scope, &options, runtime, screen).await
    });
    let result = scheduler.block_on(task);
    shutdown.stop_signal_listener();
    drop(scheduler);
    drop(pool);
    result
}

pub fn entry() {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        let _ = io::stdout().write_all(HELP.as_bytes());
        return;
    }
    let result = Options::parse(args).and_then(run);
    if let Err(error) = result {
        let code = exit_code_for(&error);
        let _ = writeln!(
            io::stderr(),
            "fsfs-live: {}",
            literal(&error.to_string(), 2_048)
        );
        std::process::exit(code);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering};

    use asupersync::test_utils::run_test_with_cx;
    use frankensearch_core::{IndexableDocument, LexicalWrite};
    use frankensearch_fsfs::adapters::live_search::{LiveSearchHit, LiveSearchTracker};
    use frankensearch_fsfs::generation_store::{GenerationPublication, PublishedGeneration};
    use frankensearch_fsfs::output_schema::SearchOutputPhase;
    use frankensearch_quill::QuillIndex;
    use ftui_core::event::{KeyEvent, PasteEvent};
    use serde_json::json;

    use super::*;

    fn options(root: &std::path::Path) -> Options {
        Options {
            root: root.to_path_buf(),
            query: "alpha".to_owned(),
            limit: 20,
            hybrid: false,
            config: None,
        }
    }

    fn key(code: KeyCode) -> Event {
        Event::Key(KeyEvent::new(code))
    }

    fn hits(ids: &[&str]) -> Vec<LiveSearchHit<Value>> {
        ids.iter()
            .map(|id| LiveSearchHit {
                doc_id: (*id).to_owned(),
                score: 1.0,
                item: json!({"path": format!("/{id}.rs"), "snippet": format!("body of {id}")}),
            })
            .collect()
    }

    fn tracker() -> LiveSearchTracker<Value> {
        LiveSearchTracker::new("alpha", LiveSearchConfig::default()).unwrap()
    }

    #[test]
    fn selection_follows_identity_across_reordering_and_metadata_updates() {
        let mut tracker = tracker();
        let mut view = View::new("alpha".to_owned(), 20);
        view.update(
            &tracker
                .apply("a", hits(&["one", "two", "three"]))
                .unwrap()
                .unwrap(),
            value_metadata,
        )
        .unwrap();
        view.selected = 1;
        let mut next = hits(&["two", "three", "one"]);
        next[0].item["snippet"] = json!("new indexed body");
        view.update(&tracker.apply("b", next).unwrap().unwrap(), value_metadata)
            .unwrap();
        assert_eq!(view.selected, 0);
        assert_eq!(view.rows[view.selected].id, "two");
        assert_eq!(view.rows[0].detail, "new indexed body");
    }

    #[test]
    fn removals_and_empty_successors_clamp_selection_without_stale_hits() {
        let mut tracker = tracker();
        let mut view = View::new("alpha".to_owned(), 20);
        view.update(
            &tracker.apply("a", hits(&["one", "two"])).unwrap().unwrap(),
            value_metadata,
        )
        .unwrap();
        view.selected = 1;
        view.update(
            &tracker.apply("b", hits(&["one"])).unwrap().unwrap(),
            value_metadata,
        )
        .unwrap();
        assert_eq!(view.selected, 0);
        assert_eq!(view.rows.len(), 1);
        view.update(
            &tracker.apply("c", hits(&[])).unwrap().unwrap(),
            value_metadata,
        )
        .unwrap();
        assert!(view.rows.is_empty());
        assert_eq!(view.selected, 0);
        assert_eq!(view.sequence, 3);
    }

    #[test]
    fn rejected_revisions_leave_the_entire_window_unchanged() {
        let mut tracker = tracker();
        let mut view = View::new("alpha".to_owned(), 20);
        view.update(
            &tracker.apply("a", hits(&["one", "two"])).unwrap().unwrap(),
            value_metadata,
        )
        .unwrap();
        let frame = tracker.apply("b", hits(&["two", "one"])).unwrap().unwrap();
        for case in 0..5 {
            let mut bad = frame.clone();
            match case {
                0 => bad.sequence += 1,
                1 => bad.previous_generation = Some("foreign".to_owned()),
                2 => bad.query = "another query".to_owned(),
                3 => bad.result_count += 1,
                _ => bad.schema_version = "foreign".to_owned(),
            }
            let before = view.clone();
            assert!(view.update(&bad, value_metadata).is_err());
            assert_eq!(view, before);
        }
        view.update(&frame, value_metadata).unwrap();
        let before = view.clone();
        assert!(view.update(&frame, value_metadata).is_err());
        assert_eq!(view, before);
    }

    #[test]
    fn partial_delta_failure_does_not_apply_its_valid_prefix() {
        let mut tracker = tracker();
        let mut view = View::new("alpha".to_owned(), 20);
        view.update(
            &tracker.apply("a", hits(&["one", "two"])).unwrap().unwrap(),
            value_metadata,
        )
        .unwrap();
        let mut frame = tracker.apply("b", hits(&[])).unwrap().unwrap();
        let LiveSearchEvent::Delta { changes } = &mut frame.event else {
            panic!("expected delta");
        };
        changes.push(LiveSearchChange::Removed {
            doc_id: "missing".to_owned(),
            previous_rank: 3,
        });
        let before = view.clone();
        assert!(view.update(&frame, value_metadata).is_err());
        assert_eq!(view, before);
    }

    #[test]
    fn snapshot_rank_gaps_and_nonfinite_scores_are_refused() {
        let mut tracker = tracker();
        let frame = tracker.apply("a", hits(&["one", "two"])).unwrap().unwrap();
        for bad_score in [false, true] {
            let mut frame = frame.clone();
            let LiveSearchEvent::Snapshot { results } = &mut frame.event else {
                panic!("expected snapshot");
            };
            if bad_score {
                results[0].hit.score = f64::NAN;
            } else {
                results[0].rank = 2;
            }
            let mut view = View::new("alpha".to_owned(), 20);
            assert!(view.update(&frame, value_metadata).is_err());
            assert_eq!(view.sequence, 0);
            assert!(view.rows.is_empty());
        }
    }

    #[test]
    fn failed_refinement_preserves_initial_and_reports_actual_admission() {
        let mut tracker = tracker();
        let mut view = View::new("alpha".to_owned(), 20);
        view.update(
            &tracker.apply("g:initial", hits(&["one"])).unwrap().unwrap(),
            value_metadata,
        )
        .unwrap();
        let rows = view.rows.clone();
        view.hybrid(&RetainedLiveSearchFrame {
            generation_id: "g".to_owned(),
            manifest_sha256: "a".repeat(64),
            phase: SearchOutputPhase::RefinementFailed,
            annotations: json!({"semantic_admitted": false, "skip_reason": "quality_timeout"}),
            update: None,
        })
        .unwrap();
        assert_eq!(view.rows, rows);
        assert_eq!(view.sequence, 1);
        assert!(view.status.contains("quality_timeout"));
        assert!(view.status.contains("false"));
    }

    #[test]
    fn terminal_controls_are_literal_and_display_work_is_bounded() {
        let text = "safe\x1b]52;c;secret\x07\n\r\u{009b}\u{202e}filename";
        let rendered = literal(text, 200);
        assert!(!rendered.chars().any(char::is_control));
        assert!(!rendered.contains('\u{202e}'));
        assert!(rendered.contains("secret"));
        assert_eq!(literal("日本語🌍", 2), "日本");
        assert_eq!(literal(&"a".repeat(10_000), 12).len(), 12);
    }

    #[test]
    fn query_edits_are_explicit_and_escape_retains_the_current_query() {
        let mut view = View::new("alpha".to_owned(), 20);
        view.event(key(KeyCode::Char('/')));
        view.event(Event::Key(
            KeyEvent::new(KeyCode::Char('u')).with_modifiers(Modifiers::CTRL),
        ));
        view.event(key(KeyCode::Char('β')));
        assert_eq!(
            view.event(key(KeyCode::Enter)),
            Action::Query("β".to_owned())
        );
        assert_eq!(
            view.query, "alpha",
            "the driver owns the actual session switch"
        );
        assert_eq!(view.event(key(KeyCode::Escape)), Action::Continue);
        assert!(view.draft.is_none());
        assert_eq!(view.query, "alpha");
        assert_eq!(view.event(key(KeyCode::Char('q'))), Action::Quit);
    }

    #[test]
    fn paste_is_bounded_text_not_commands_and_key_releases_do_nothing() {
        let mut view = View::new("alpha".to_owned(), 20);
        view.event(Event::Paste(PasteEvent::bracketed("q/r\n")));
        assert!(view.draft.is_none());
        view.event(key(KeyCode::Char('/')));
        view.event(Event::Paste(PasteEvent::bracketed("\nβ\x1b")));
        assert_eq!(view.draft.as_deref(), Some("alpha β"));
        view.event(Event::Key(
            KeyEvent::new(KeyCode::Backspace).with_kind(KeyEventKind::Release),
        ));
        assert_eq!(view.draft.as_deref(), Some("alpha β"));
        view.draft = Some("x".repeat(MAX_QUERY_BYTES - 1));
        view.event(Event::Paste(PasteEvent::bracketed("🌍")));
        assert_eq!(view.draft.as_ref().unwrap().len(), MAX_QUERY_BYTES - 1);
    }

    #[test]
    fn arguments_require_explicit_store_and_do_not_authorize_indexing() {
        let args = |extra: &[&str]| {
            ["--index-dir", "/store", "--query", "alpha"]
                .into_iter()
                .chain(extra.iter().copied())
                .map(OsString::from)
                .collect()
        };
        assert!(Options::parse(args(&[])).is_ok());
        for extra in [
            vec!["--limit", "0"],
            vec!["--limit", "1001"],
            vec!["--watch-source", "/source"],
            vec!["--config", "/file"],
            vec!["--hybrid", "--hybrid"],
            vec!["--query", "beta"],
        ] {
            assert!(Options::parse(args(&extra)).is_err());
        }
        assert!(Options::parse(Vec::new()).is_err());
        assert!(Options::parse(args(&["--hybrid", "--config", "/file"])).is_ok());
    }

    #[derive(Default)]
    struct Capture {
        views: Arc<Mutex<Vec<View>>>,
        events: VecDeque<Event>,
        switch_query: bool,
        switched: bool,
        quit_queued: bool,
    }

    impl Screen for Capture {
        fn draw(&mut self, view: &View) -> SearchResult<()> {
            self.views.lock().unwrap().push(view.clone());
            if self.switch_query && view.sequence > 0 {
                if view.query == "alpha" && !self.switched {
                    self.switched = true;
                    self.events.extend([
                        key(KeyCode::Char('/')),
                        Event::Key(
                            KeyEvent::new(KeyCode::Char('u')).with_modifiers(Modifiers::CTRL),
                        ),
                        key(KeyCode::Char('b')),
                        key(KeyCode::Char('e')),
                        key(KeyCode::Char('t')),
                        key(KeyCode::Char('a')),
                        key(KeyCode::Enter),
                    ]);
                } else if view.query == "beta" && !self.quit_queued {
                    self.quit_queued = true;
                    self.events.push_back(key(KeyCode::Char('q')));
                }
            }
            Ok(())
        }
        fn event(&mut self) -> SearchResult<Option<Event>> {
            Ok(self.events.pop_front())
        }
    }

    async fn publish(
        cx: &Cx,
        store: &CompleteGenerationStore,
        docs: &[IndexableDocument],
    ) -> PublishedGeneration {
        let build = store.begin(cx).unwrap();
        let index = QuillIndex::create(
            cx,
            &build.path().join("lexical"),
            QuillConfig {
                deterministic_ingest: true,
                ..QuillConfig::default()
            },
        )
        .await
        .unwrap();
        LexicalWrite::index_documents(&index, cx, docs)
            .await
            .unwrap();
        LexicalWrite::commit(&index, cx).await.unwrap();
        drop(index);
        let GenerationPublication::Durable(generation) = build.publish(cx, |_, _| Ok(())).unwrap()
        else {
            panic!("fixture did not reach durability");
        };
        generation
    }

    #[test]
    fn native_generation_refresh_replaces_results_without_a_writer_or_source_read() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let first = publish(&cx, &store, &[IndexableDocument::new("old", "alpha first")]).await;
            let ui = Mutex::new(Ui {
                screen: Capture::default(),
                view: View::new("alpha".to_owned(), 20),
            });
            let mut session =
                Session::new(store.clone(), &options(root.path()), "alpha", None).unwrap();
            let now = Instant::now();
            session.poll(&cx, now, &ui).await.unwrap();
            assert_eq!(lock(&ui).unwrap().view.rows[0].id, "old");
            let successor = publish(
                &cx,
                &store,
                &[IndexableDocument::new("new", "alpha second")],
            )
            .await;
            session
                .poll(&cx, now + Duration::from_millis(1), &ui)
                .await
                .unwrap();
            session
                .poll(&cx, now + Duration::from_millis(101), &ui)
                .await
                .unwrap();
            assert_eq!(lock(&ui).unwrap().view.rows[0].id, "new");
            assert_eq!(lock(&ui).unwrap().view.rows.len(), 1);
            assert_eq!(store.active(&cx).unwrap(), Some(successor));
            assert!(
                first.path().exists(),
                "old pinned generation remains intact"
            );
            drop(store.begin(&cx).unwrap());
        });
    }

    #[test]
    fn production_driver_switches_native_queries_and_resets_the_revision_chain() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let generation = publish(
                &cx,
                &store,
                &[
                    IndexableDocument::new("a", "alpha only"),
                    IndexableDocument::new("b", "beta only"),
                ],
            )
            .await;
            let capture = Capture {
                switch_query: true,
                ..Capture::default()
            };
            let views = Arc::clone(&capture.views);
            execute(&cx, &options(root.path()), None, capture)
                .await
                .unwrap();
            let views = views.lock().unwrap();
            let alpha = views
                .iter()
                .find(|view| view.query == "alpha" && view.sequence == 1)
                .unwrap();
            assert_eq!(alpha.rows[0].id, "a");
            let beta = views
                .iter()
                .find(|view| view.query == "beta" && view.sequence == 1)
                .unwrap();
            assert_eq!(beta.rows[0].id, "b");
            assert_eq!(beta.rows.len(), 1);
            drop(views);
            assert_eq!(store.active(&cx).unwrap(), Some(generation));
        });
    }

    struct PendingQuery {
        dropped: Arc<AtomicBool>,
        polled: Arc<AtomicBool>,
    }
    impl std::future::Future for PendingQuery {
        type Output = SearchResult<()>;
        fn poll(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Self::Output> {
            self.polled.store(true, Ordering::SeqCst);
            std::task::Poll::Pending
        }
    }
    impl Drop for PendingQuery {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    struct QuitAfterPoll(Arc<AtomicBool>);

    impl Screen for QuitAfterPoll {
        fn draw(&mut self, _view: &View) -> SearchResult<()> {
            Ok(())
        }
        fn event(&mut self) -> SearchResult<Option<Event>> {
            Ok(self
                .0
                .load(Ordering::SeqCst)
                .then(|| key(KeyCode::Char('q'))))
        }
    }

    #[test]
    fn quit_drops_an_unfinished_query_before_returning_to_the_owner() {
        run_test_with_cx(|cx| async move {
            let dropped = Arc::new(AtomicBool::new(false));
            let polled = Arc::new(AtomicBool::new(false));
            let screen = QuitAfterPoll(Arc::clone(&polled));
            let ui = Mutex::new(Ui {
                screen,
                view: View::new("alpha".to_owned(), 20),
            });
            let work = PendingQuery {
                dropped: Arc::clone(&dropped),
                polled: Arc::clone(&polled),
            };
            assert_eq!(
                responsive(&cx, &ui, work).await.unwrap(),
                Some(Action::Quit)
            );
            assert!(
                polled.load(Ordering::SeqCst),
                "quit arrived after the query yielded Pending"
            );
            assert!(dropped.load(Ordering::SeqCst));
        });
    }

    #[test]
    fn cancellation_drops_a_pending_query_without_any_frame() {
        run_test_with_cx(|cx| async move {
            let dropped = Arc::new(AtomicBool::new(false));
            let ui = Mutex::new(Ui {
                screen: Capture::default(),
                view: View::new("alpha".to_owned(), 20),
            });
            cx.set_cancel_requested(true);
            let work = PendingQuery {
                dropped: Arc::clone(&dropped),
                polled: Arc::new(AtomicBool::new(false)),
            };
            assert!(matches!(
                responsive(&cx, &ui, work).await,
                Err(SearchError::Cancelled { .. })
            ));
            assert!(dropped.load(Ordering::SeqCst));
            assert_eq!(lock(&ui).unwrap().view.sequence, 0);
        });
    }

    struct BrokenInput(Arc<std::sync::atomic::AtomicUsize>);

    impl Screen for BrokenInput {
        fn draw(&mut self, _view: &View) -> SearchResult<()> {
            Ok(())
        }
        fn event(&mut self) -> SearchResult<Option<Event>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Err(io::Error::from(io::ErrorKind::BrokenPipe).into())
        }
    }

    #[test]
    fn terminal_failure_exits_instead_of_spinning_a_query_retry_loop() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let _store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let result = execute(
                &cx,
                &options(root.path()),
                None,
                BrokenInput(Arc::clone(&calls)),
            )
            .await;
            assert!(matches!(
                result,
                Err(SearchError::SubsystemError {
                    subsystem: "fsfs.live_search.tui.terminal",
                    ..
                })
            ));
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        });
    }
}
