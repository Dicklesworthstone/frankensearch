//! Live terminal subscriber for the existing command, not a second search engine.
//!
//! The same bounded NDJSON producer feeds this consumer in-process. Each complete
//! record is validated and presented before its producer receives a successful
//! flush. Input and the owned subscription future are driven by the same task;
//! no input thread, detached publisher, or unbounded event queue is introduced.

mod model;

use std::future::{Future, poll_fn};
use std::io::{self, IsTerminal, Write};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::Poll;
use std::time::Duration;

use asupersync::Cx;
use asupersync::types::CancelKind;
use frankensearch_core::{SearchError, SearchResult};
use frankensearch_fsfs::FsfsRuntime;
use ftui_backend::{Backend, BackendEventSource, BackendFeatures, BackendPresenter};
use ftui_core::event::{Event, KeyCode, KeyEventKind};
use ftui_core::geometry::Rect;
use ftui_render::diff::BufferDiff;
use ftui_render::frame::Frame;
use ftui_render::grapheme_pool::GraphemePool;
use ftui_render::presenter::Presenter;
use ftui_widgets::{Widget, paragraph::Paragraph};
use ftui_tty::{TtyBackend, TtySessionOptions};

use super::Options;
use model::{Model, display_text};

const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;
const INPUT_TICK: Duration = Duration::from_millis(20);
const MAX_EVENTS_PER_TICK: usize = 32;

/// Refuse redirected output before configuration, model admission or source writes.
pub(super) fn preflight() -> SearchResult<()> {
    validate_terminals(io::stdin().is_terminal(), io::stdout().is_terminal())
}

fn validate_terminals(input: bool, output: bool) -> SearchResult<()> {
    if !input || !output {
        return Err(super::invalid(
            "--tui requires terminal stdin and stdout; omit it for NDJSON output",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Move(isize),
    First,
    Last,
    Redraw,
    Ignore,
    Quit,
    Interrupt,
}

fn action_for(event: &Event) -> Option<Action> {
    match event {
        Event::Resize { .. } | Event::Focus(true) => Some(Action::Redraw),
        Event::Key(key) if key.kind != KeyEventKind::Release => {
            if key.ctrl() && key.code == KeyCode::Char('c') {
                return Some(Action::Interrupt);
            }
            if key.ctrl() || key.alt() || key.super_key() {
                return None;
            }
            match key.code {
                KeyCode::Char('q') | KeyCode::Escape => Some(Action::Quit),
                KeyCode::Up | KeyCode::Char('k') => Some(Action::Move(-1)),
                KeyCode::Down | KeyCode::Char('j') => Some(Action::Move(1)),
                KeyCode::PageUp => Some(Action::Move(-10)),
                KeyCode::PageDown => Some(Action::Move(10)),
                KeyCode::Home => Some(Action::First),
                KeyCode::End => Some(Action::Last),
                _ => None,
            }
        }
        _ => None,
    }
}

trait Surface: Send {
    fn present(&mut self, model: &Model) -> io::Result<()>;
    /// None means no decoded input is currently available.
    fn poll_action(&mut self) -> io::Result<Option<Action>>;
}

struct TerminalSurface {
    // Flush presenter teardown before the backend restores raw/alternate mode.
    presenter: Presenter<io::Stdout>,
    backend: TtyBackend,
}

impl TerminalSurface {
    fn open() -> io::Result<Self> {
        let mut backend = TtyBackend::open(80, 24, TtySessionOptions {
            alternate_screen: true,
            // Normal exits restore via TtyBackend's drop. Its existing signal
            // cleanup also protects force-exit during an uninterruptible syscall.
            intercept_signals: true,
            features: BackendFeatures::default(),
        })?;
        let presenter = Presenter::new(io::stdout(), *backend.presenter().capabilities());
        Ok(Self { presenter, backend })
    }
}

impl Surface for TerminalSurface {
    fn present(&mut self, model: &Model) -> io::Result<()> {
        let (width, height) = self.backend.size()?;
        let size = (width.clamp(1, 512), height.clamp(1, 256));
        // A fresh, frame-local pool bounds memory during arbitrarily long
        // subscriptions. Full repaint avoids comparing IDs from different pools.
        // The 0.7 presenter accepts the exact pool that produced this buffer.
        let mut pool = GraphemePool::new();
        let buffer = {
            let mut frame = Frame::new(size.0, size.1, &mut pool);
            render(model, size.0, size.1, &mut frame);
            frame.buffer
        };
        let diff = BufferDiff::full(size.0, size.1);
        self.presenter.present_with_pool(&buffer, &diff, Some(&pool), None)?;
        Ok(())
    }

    fn poll_action(&mut self) -> io::Result<Option<Action>> {
        if !self.backend.poll_event(Duration::ZERO)? {
            return Ok(None);
        }
        Ok(self.backend.read_event()?.as_ref().map(|event| {
            action_for(event).unwrap_or(Action::Ignore)
        }))
    }
}

fn render(model: &Model, width: u16, height: u16, frame: &mut Frame<'_>) {
    let line = |frame: &mut Frame<'_>, row, text: String| {
        if row < height {
            Paragraph::new(text).render(Rect::new(0, row, width, 1), frame);
        }
    };
    line(frame, 0, format!("fsfs LIVE | {} | {}",
        if model.hybrid { "hybrid subscription" } else { "lexical subscription" },
        display_text(&model.query, 120)));
    if height < 6 {
        line(frame, height.saturating_sub(1), "Resize terminal | q/Esc: quit".to_owned());
        return;
    }
    line(frame, 1, model.status());
    line(frame, 2, format!("Committed revision: {}",
        display_text(model.revision.as_deref().unwrap_or("none"), 180)));
    let list_rows = usize::from(if height >= 10 { height - 8 } else { height - 4 });
    let start = model.selected.saturating_sub(list_rows.saturating_sub(1));
    if model.results.is_empty() {
        let message = if model.sequence == 0 { "Waiting for publication..." } else { "No matching results" };
        line(frame, 3, message.to_owned());
    }
    for (offset, result) in model.results.iter().enumerate().skip(start).take(list_rows) {
        let location = result.hit.item.get("path").and_then(serde_json::Value::as_str)
            .unwrap_or(&result.hit.doc_id);
        let location = display_text(location, 512);
        let location = result.hit.item.get("line").and_then(serde_json::Value::as_u64)
            .map_or_else(|| location.clone(), |number| format!("{location}:{number}"));
        let row = 3_u16.saturating_add(u16::try_from(offset - start).unwrap_or(u16::MAX));
        line(frame, row, format!("{} {:>4}  {:>10.5}  {location}",
            if offset == model.selected { ">" } else { " " }, result.rank, result.hit.score));
    }
    if height >= 10 && let Some(selected) = model.results.get(model.selected) {
        line(frame, height - 4, format!("Selected: {}", display_text(&selected.hit.doc_id, 512)));
        let snippet = selected.hit.item.get("snippet").and_then(serde_json::Value::as_str)
            .unwrap_or("No stored snippet in this result payload");
        Paragraph::new(display_text(snippet, 2_000))
            .render(Rect::new(0, height - 3, width, 2), frame);
    }
    line(frame, height - 1, "Up/Down or j/k: select | PgUp/PgDn: 10 rows | Home/End | q/Esc: quit".to_owned());
}

struct FrameOutput<S> {
    surface: S,
    model: Model,
    pending: Vec<u8>,
}

impl<S: Surface> FrameOutput<S> {
    fn new(mut surface: S, model: Model) -> io::Result<Self> {
        surface.present(&model)?;
        Ok(Self { surface, model, pending: Vec::new() })
    }

    fn input(&mut self) -> io::Result<Option<Action>> {
        let mut redraw = false;
        for _ in 0..MAX_EVENTS_PER_TICK {
            let Some(action) = self.surface.poll_action()? else { break };
            match action {
                Action::Quit | Action::Interrupt => return Ok(Some(action)),
                Action::Ignore => continue,
                Action::Move(distance) => self.model.move_by(distance),
                Action::First => self.model.selected = 0,
                Action::Last => self.model.selected = self.model.results.len().saturating_sub(1),
                Action::Redraw => {}
            }
            redraw = true;
        }
        if redraw {
            self.surface.present(&self.model)?;
        }
        Ok(None)
    }
}

impl<S: Surface> Write for FrameOutput<S> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.pending.len().checked_add(bytes.len())
            .filter(|length| *length <= MAX_FRAME_BYTES)
            .ok_or_else(|| model::invalid("terminal live-search frame exceeds 16 MiB"))?;
        self.pending.try_reserve(bytes.len())
            .map_err(|_| io::Error::other("cannot reserve terminal live-search frame"))?;
        self.pending.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let candidate = self.model.prepare(&self.pending)?;
        self.surface.present(&candidate)?;
        // Rendering is the acknowledgment. Failed presentation cannot move
        // either the terminal model or the upstream subscription baseline.
        self.model = candidate;
        self.pending.clear();
        Ok(())
    }
}

struct SharedOutput<S>(Arc<Mutex<FrameOutput<S>>>);

fn lock<S>(shared: &Mutex<FrameOutput<S>>) -> io::Result<MutexGuard<'_, FrameOutput<S>>> {
    shared.lock().map_err(|_| io::Error::other("terminal live-search state was poisoned"))
}

impl<S: Surface> Write for SharedOutput<S> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        lock(&self.0)?.write(bytes)
    }

    fn flush(&mut self) -> io::Result<()> {
        lock(&self.0)?.flush()
    }
}

fn checkpoint(cx: &Cx) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: "fsfs.live_search.terminal".to_owned(),
        reason: error.to_string(),
    })
}

async fn drive<S: Surface, F: Future<Output = SearchResult<u64>>>(
    cx: &Cx,
    shared: &Mutex<FrameOutput<S>>,
    work: F,
) -> SearchResult<u64> {
    let mut work = std::pin::pin!(work);
    loop {
        checkpoint(cx)?;
        let action = lock(shared)?.input()?;
        match action {
            Some(Action::Quit) => {
                // Cancel existing worker operations as well as dropping their
                // command future. The caller still owns/drains its blocking pool.
                cx.cancel_fast(CancelKind::User);
                return Ok(0);
            }
            Some(Action::Interrupt) => {
                cx.cancel_fast(CancelKind::User);
                return Err(SearchError::Cancelled {
                    phase: "fsfs.live_search.terminal".to_owned(),
                    reason: "terminal interrupt".to_owned(),
                });
            }
            _ => {}
        }
        // Poll exactly once per input tick. The underlying future retains the
        // real task waker; the timer also guarantees a bounded input interval
        // even when the subscriber is waiting on a long configured poll delay.
        if let Poll::Ready(result) = poll_fn(|context| {
            Poll::Ready(work.as_mut().poll(context))
        }).await {
            return result;
        }
        asupersync::time::sleep(cx.now(), INPUT_TICK).await;
    }
}

pub(super) async fn execute(
    cx: &Cx,
    options: &Options,
    runtime: Option<FsfsRuntime>,
) -> SearchResult<u64> {
    checkpoint(cx)?;
    preflight()?;
    let model = Model::new(options.query.clone(), options.hybrid, options.limits.max_results);
    let output = FrameOutput::new(TerminalSurface::open()?, model)?;
    let shared = Arc::new(Mutex::new(output));
    let mut writer = SharedOutput(Arc::clone(&shared));
    let work = super::execute_with_runtime(cx, options, &mut writer, runtime);
    // Backend drop restores the terminal on every return, including an input,
    // admission, decoding, rendering or output error. No raw guard escapes.
    drive(cx, shared.as_ref(), work).await
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering};

    use asupersync::test_utils::run_test_with_cx;
    use frankensearch_core::{IndexableDocument, LexicalWrite};
    use frankensearch_fsfs::adapters::live_search::{LiveSearchConfig, LiveSearchHit, LiveSearchTracker};
    use frankensearch_fsfs::generation_store::{CompleteGenerationStore, GenerationPublication};
    use frankensearch_quill::{QuillConfig, QuillIndex};
    use ftui_core::event::{KeyEvent, Modifiers};
    use ftui_core::terminal_capabilities::TerminalCapabilities;
    use serde_json::json;

    use super::*;

    #[derive(Default)]
    struct RecordingSurface {
        frames: Vec<Model>,
        actions: VecDeque<Action>,
        fail_after: Option<usize>,
        quit_when_polled: Option<Arc<AtomicBool>>,
    }

    impl Surface for RecordingSurface {
        fn present(&mut self, model: &Model) -> io::Result<()> {
            if self.fail_after.is_some_and(|count| self.frames.len() >= count) {
                return Err(io::Error::other("injected terminal failure"));
            }
            self.frames.push(model.clone());
            Ok(())
        }

        fn poll_action(&mut self) -> io::Result<Option<Action>> {
            if self.quit_when_polled.as_ref().is_some_and(|flag| flag.load(Ordering::SeqCst)) {
                return Ok(Some(Action::Quit));
            }
            Ok(self.actions.pop_front())
        }
    }

    fn output(surface: RecordingSurface) -> FrameOutput<RecordingSurface> {
        FrameOutput::new(surface, Model::new("alpha".to_owned(), false, 5)).unwrap()
    }

    fn record() -> Vec<u8> {
        let mut tracker = LiveSearchTracker::new("alpha", LiveSearchConfig::default()).unwrap();
        let frame = tracker.apply("generation", vec![LiveSearchHit {
            doc_id: "doc-a".to_owned(), score: 1.0, item: json!({"path": "a.rs"}),
        }]).unwrap().unwrap();
        let mut bytes = serde_json::to_vec(&frame).unwrap();
        bytes.push(b'\n');
        bytes
    }

    #[test]
    fn complete_record_is_presented_before_its_baseline_is_acknowledged() {
        let mut output = output(RecordingSurface::default());
        let record = record();
        for chunk in record.chunks(7) {
            output.write_all(chunk).unwrap();
            assert_eq!(output.model.sequence, 0);
        }
        output.flush().unwrap();
        assert_eq!(output.model.sequence, 1);
        assert_eq!(output.surface.frames.len(), 2); // waiting, then snapshot
        assert_eq!(output.surface.frames[1].results[0].hit.doc_id, "doc-a");
        assert!(output.pending.is_empty());
    }

    #[test]
    fn failed_presentation_and_partial_records_leave_the_display_baseline_unchanged() {
        let mut failed = output(RecordingSurface { fail_after: Some(1), ..RecordingSurface::default() });
        failed.write_all(&record()).unwrap();
        assert!(failed.flush().is_err());
        assert_eq!(failed.model.sequence, 0);
        assert_eq!(failed.surface.frames.len(), 1);
        let mut partial = output(RecordingSurface::default());
        partial.write_all(b"{\"schema_version\":").unwrap();
        assert!(partial.flush().is_err());
        assert_eq!(partial.model.sequence, 0);
    }

    #[test]
    fn oversized_records_are_refused_without_rendering_or_growing_the_buffer() {
        let mut output = output(RecordingSurface::default());
        assert!(output.write_all(&vec![b'x'; MAX_FRAME_BYTES + 1]).is_err());
        assert!(output.pending.is_empty());
        assert_eq!(output.surface.frames.len(), 1);
    }

    #[test]
    fn terminal_flags_are_explicit_and_do_not_enable_indexing_or_hybrid_models() {
        let options = Options::parse(vec!["--index-dir".into(), "/store".into(),
            "--query".into(), "alpha".into(), "--tui".into()]).unwrap();
        assert!(options.tui);
        assert!(!options.hybrid);
        assert!(options.watch_source.is_none());
        for suffix in [vec!["--tui"], vec!["--format", "jsonl"]] {
            let args = ["--index-dir", "/store", "--query", "alpha", "--tui"].into_iter()
                .chain(suffix).map(std::ffi::OsString::from).collect();
            assert!(Options::parse(args).is_err());
        }
        for (input, output) in [(false, false), (true, false), (false, true)] {
            assert!(validate_terminals(input, output).is_err());
        }
        assert!(validate_terminals(true, true).is_ok());
    }

    #[test]
    fn key_releases_and_modified_navigation_do_not_move_or_quit() {
        let press = KeyEvent::new(KeyCode::Char('q'));
        assert_eq!(action_for(&Event::Key(press)), Some(Action::Quit));
        assert_eq!(action_for(&Event::Key(press.with_kind(KeyEventKind::Release))), None);
        assert_eq!(action_for(&Event::Key(press.with_modifiers(Modifiers::ALT))), None);
        assert_eq!(action_for(&Event::Key(KeyEvent::new(KeyCode::Char('c')).with_modifiers(Modifiers::CTRL))), Some(Action::Interrupt));
    }

    #[test]
    fn ignored_input_is_drained_but_a_flood_cannot_starve_the_subscription() {
        let mut output = output(RecordingSurface {
            actions: std::iter::repeat_n(Action::Ignore, MAX_EVENTS_PER_TICK + 1)
                .chain([Action::Quit]).collect(),
            ..RecordingSurface::default()
        });
        assert_eq!(output.input().unwrap(), None);
        assert_eq!(output.surface.actions.len(), 2);
        assert_eq!(output.input().unwrap(), Some(Action::Quit));
    }

    struct PendingWork {
        dropped: Arc<AtomicBool>,
        polled: Arc<AtomicBool>,
    }

    impl Future for PendingWork {
        type Output = SearchResult<u64>;
        fn poll(self: std::pin::Pin<&mut Self>, _: &mut std::task::Context<'_>) -> Poll<Self::Output> {
            self.polled.store(true, Ordering::SeqCst);
            Poll::Pending
        }
    }

    impl Drop for PendingWork {
        fn drop(&mut self) { self.dropped.store(true, Ordering::SeqCst); }
    }

    #[test]
    fn quit_drops_the_owned_command_future_and_cancels_its_workers() {
        run_test_with_cx(|cx| async move {
            let dropped = Arc::new(AtomicBool::new(false));
            let polled = Arc::new(AtomicBool::new(false));
            let output = Mutex::new(output(RecordingSurface {
                actions: [Action::Quit].into(), ..RecordingSurface::default()
            }));
            let work = PendingWork { dropped: Arc::clone(&dropped), polled: Arc::clone(&polled) };
            assert_eq!(drive(&cx, &output, work).await.unwrap(), 0);
            assert!(dropped.load(Ordering::SeqCst));
            assert!(!polled.load(Ordering::SeqCst));
            assert!(cx.is_cancel_requested());
        });
    }

    #[test]
    fn input_can_stop_an_already_polled_subscription_that_never_becomes_ready() {
        run_test_with_cx(|cx| async move {
            let dropped = Arc::new(AtomicBool::new(false));
            let polled = Arc::new(AtomicBool::new(false));
            let output = Mutex::new(output(RecordingSurface {
                quit_when_polled: Some(Arc::clone(&polled)),
                ..RecordingSurface::default()
            }));
            let work = PendingWork {
                dropped: Arc::clone(&dropped),
                polled: Arc::clone(&polled),
            };
            assert_eq!(drive(&cx, &output, work).await.unwrap(), 0);
            assert!(polled.load(Ordering::SeqCst));
            assert!(dropped.load(Ordering::SeqCst));
            assert!(cx.is_cancel_requested());
            assert_eq!(lock(&output).unwrap().model.sequence, 0);
        });
    }

    #[test]
    fn render_uses_the_correct_grapheme_pool_at_tiny_and_normal_sizes() {
        let model = Model::new("café 👩‍💻 世界".to_owned(), false, 5);
        for (width, height) in [(1, 1), (20, 6), (80, 24), (140, 40)] {
            let mut pool = GraphemePool::new();
            let buffer = {
                let mut frame = Frame::new(width, height, &mut pool);
                render(&model, width, height, &mut frame);
                frame.buffer
            };
            let mut presenter = Presenter::new(Vec::new(), TerminalCapabilities::detect());
            presenter.present_with_pool(&buffer, &BufferDiff::full(width, height), Some(&pool), None).unwrap();
            let rendered = String::from_utf8(presenter.writer_mut().clone()).unwrap();
            if width >= 80 {
                assert!(rendered.contains("👩‍💻"));
                assert!(rendered.contains("café"));
            }
        }
    }

    #[test]
    fn terminal_backend_and_shared_output_satisfy_the_real_task_send_contract() {
        fn is_send<T: Send>() {}
        is_send::<TerminalSurface>();
        is_send::<SharedOutput<TerminalSurface>>();
    }

    #[test]
    fn native_quill_subscription_reaches_the_terminal_consumer_without_a_second_ranker() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let build = store.begin(&cx).unwrap();
            let index = QuillIndex::create(&cx, &build.path().join("lexical"), QuillConfig {
                deterministic_ingest: true, ..QuillConfig::default()
            }).await.unwrap();
            LexicalWrite::index_documents(&index, &cx, &[
                IndexableDocument::new("source-a", "alpha searchable source").with_metadata("path", "source-a.rs"),
            ]).await.unwrap();
            LexicalWrite::commit(&index, &cx).await.unwrap();
            drop(index);
            assert!(matches!(build.publish(&cx, |_, _| Ok(())).unwrap(), GenerationPublication::Durable(_)));
            let options = Options::parse(vec!["--index-dir".into(), root.path().as_os_str().to_owned(),
                "--query".into(), "alpha".into(), "--once".into(), "--tui".into()]).unwrap();
            let shared = Arc::new(Mutex::new(output(RecordingSurface::default())));
            let mut writer = SharedOutput(Arc::clone(&shared));
            let work = super::super::execute_with_runtime(&cx, &options, &mut writer, None);
            assert_eq!(drive(&cx, shared.as_ref(), work).await.unwrap(), 1);
            let view = lock(shared.as_ref()).unwrap();
            assert_eq!(view.model.results.len(), 1);
            assert_eq!(view.model.results[0].hit.doc_id, "source-a");
            assert_eq!(view.model.results[0].hit.item["path"], "source-a.rs");
            assert_eq!(view.model.sequence, 1);
            drop(view);
            // Reading and rendering did not retain publication ownership.
            drop(store.begin(&cx).unwrap());
        });
    }
}
