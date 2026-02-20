//! Input recording and deterministic replay.
//!
//! Provides [`ReplayRecorder`] for capturing input events with timestamps
//! and [`ReplayPlayer`] for replaying them deterministically. This enables
//! automated testing, bug reproduction, and demo playback.

use std::time::Duration;

use ftui_core::event::{KeyCode, Modifiers, MouseButton, MouseEventKind};
use serde::{Deserialize, Serialize};

use crate::input::InputEvent;

// ─── Replay State ────────────────────────────────────────────────────────────

/// State of the replay system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayState {
    /// Idle — not recording or playing.
    Idle,
    /// Recording input events.
    Recording,
    /// Playing back recorded events.
    Playing,
    /// Playback paused.
    Paused,
}

// ─── Input Record ────────────────────────────────────────────────────────────

/// A recorded input event with a timestamp offset from the start.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputRecord {
    /// Time offset from the start of recording.
    pub offset: Duration,
    /// The recorded event kind.
    pub event: RecordedEvent,
}

/// Serializable input event (subset of `InputEvent` that can be replayed).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RecordedEvent {
    /// A key press.
    Key {
        /// Key code in stable replay encoding.
        key: String,
        /// Modifier bits.
        modifiers: u8,
    },
    /// A mouse event.
    Mouse {
        /// Mouse kind in stable replay encoding.
        kind: String,
        /// Column position.
        col: u16,
        /// Row position.
        row: u16,
    },
    /// Terminal resize.
    Resize {
        /// New width.
        width: u16,
        /// New height.
        height: u16,
    },
}

impl RecordedEvent {
    /// Create a key event record from key code and modifiers.
    #[must_use]
    pub fn from_key(key: KeyCode, modifiers: Modifiers) -> Self {
        Self::Key {
            key: encode_key_code(key),
            modifiers: modifiers.bits(),
        }
    }

    /// Create a mouse event record.
    #[must_use]
    pub fn from_mouse(kind: MouseEventKind, col: u16, row: u16) -> Self {
        Self::Mouse {
            kind: encode_mouse_event(kind),
            col,
            row,
        }
    }

    /// Create a resize event record.
    #[must_use]
    pub const fn from_resize(width: u16, height: u16) -> Self {
        Self::Resize { width, height }
    }

    /// Convert a recorded event back into a runtime input event.
    #[must_use]
    pub fn to_input_event(&self) -> Option<InputEvent> {
        match self {
            Self::Key { key, modifiers } => {
                let key = decode_key_code(key)?;
                let mods = Modifiers::from_bits(*modifiers)?;
                Some(InputEvent::Key(key, mods))
            }
            Self::Mouse { kind, col, row } => {
                let kind = decode_mouse_event(kind)?;
                Some(InputEvent::Mouse(kind, *col, *row))
            }
            Self::Resize { width, height } => Some(InputEvent::Resize(*width, *height)),
        }
    }
}

// ─── Replay Recorder ─────────────────────────────────────────────────────────

/// Records input events with timestamps for later replay.
pub struct ReplayRecorder {
    /// Recorded events.
    records: Vec<InputRecord>,
    /// When recording started.
    start: Option<std::time::Instant>,
    /// Current state.
    state: ReplayState,
}

impl ReplayRecorder {
    /// Create a new recorder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            records: Vec::new(),
            start: None,
            state: ReplayState::Idle,
        }
    }

    /// Start recording.
    pub fn start(&mut self) {
        self.records.clear();
        self.start = Some(std::time::Instant::now());
        self.state = ReplayState::Recording;
    }

    /// Stop recording.
    pub const fn stop(&mut self) {
        self.state = ReplayState::Idle;
        self.start = None;
    }

    /// Record an input event.
    pub fn record(&mut self, event: &InputEvent) {
        if self.state != ReplayState::Recording {
            return;
        }

        let offset = self.start.map_or(Duration::ZERO, |s| s.elapsed());

        let recorded = match event {
            InputEvent::Key(key, mods) => RecordedEvent::from_key(*key, *mods),
            InputEvent::Mouse(kind, col, row) => RecordedEvent::from_mouse(*kind, *col, *row),
            InputEvent::Resize(w, h) => RecordedEvent::from_resize(*w, *h),
            InputEvent::Action(_) => return, // Don't record resolved actions.
        };

        self.records.push(InputRecord {
            offset,
            event: recorded,
        });
    }

    /// Get the recorded events.
    #[must_use]
    pub fn records(&self) -> &[InputRecord] {
        &self.records
    }

    /// Current state.
    #[must_use]
    pub const fn state(&self) -> ReplayState {
        self.state
    }

    /// Number of recorded events.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether no events have been recorded.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Export records as JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.records)
    }
}

impl Default for ReplayRecorder {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Replay Player ───────────────────────────────────────────────────────────

/// Plays back recorded input events.
pub struct ReplayPlayer {
    /// Events to play.
    records: Vec<InputRecord>,
    /// Current position in the playback.
    position: usize,
    /// Current state.
    state: ReplayState,
}

impl ReplayPlayer {
    /// Create a new player with the given records.
    #[must_use]
    pub const fn new(records: Vec<InputRecord>) -> Self {
        Self {
            records,
            position: 0,
            state: ReplayState::Idle,
        }
    }

    /// Load records from JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let records: Vec<InputRecord> = serde_json::from_str(json)?;
        Ok(Self::new(records))
    }

    /// Start playback from the beginning.
    pub const fn play(&mut self) {
        self.position = 0;
        self.state = ReplayState::Playing;
    }

    /// Pause playback.
    pub fn pause(&mut self) {
        if self.state == ReplayState::Playing {
            self.state = ReplayState::Paused;
        }
    }

    /// Resume playback.
    pub fn resume(&mut self) {
        if self.state == ReplayState::Paused {
            self.state = ReplayState::Playing;
        }
    }

    /// Stop playback.
    pub const fn stop(&mut self) {
        self.state = ReplayState::Idle;
        self.position = 0;
    }

    /// Get the next event to play, advancing the position.
    #[must_use]
    pub fn advance(&mut self) -> Option<&InputRecord> {
        if self.state != ReplayState::Playing {
            return None;
        }

        if self.position >= self.records.len() {
            self.state = ReplayState::Idle;
            return None;
        }

        let record = &self.records[self.position];
        self.position += 1;
        Some(record)
    }

    /// Advance playback and decode the next replayable input event.
    ///
    /// Returns the event timestamp offset plus decoded input. Records that
    /// cannot be decoded are skipped.
    #[must_use]
    pub fn advance_input(&mut self) -> Option<(Duration, InputEvent)> {
        while let Some(record) = self.advance().cloned() {
            if let Some(event) = record.event.to_input_event() {
                return Some((record.offset, event));
            }
        }
        None
    }

    /// Current playback position.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Total number of events.
    #[must_use]
    pub const fn total(&self) -> usize {
        self.records.len()
    }

    /// Current state.
    #[must_use]
    pub const fn state(&self) -> ReplayState {
        self.state
    }

    /// Whether playback is complete.
    #[must_use]
    pub const fn is_done(&self) -> bool {
        self.position >= self.records.len()
    }

    /// Playback progress as a ratio (0.0 to 1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn progress(&self) -> f64 {
        if self.records.is_empty() {
            return 1.0;
        }
        self.position as f64 / self.records.len() as f64
    }
}

fn encode_key_code(key: KeyCode) -> String {
    match key {
        KeyCode::Backspace => "backspace".to_owned(),
        KeyCode::Enter => "enter".to_owned(),
        KeyCode::Left => "left".to_owned(),
        KeyCode::Right => "right".to_owned(),
        KeyCode::Up => "up".to_owned(),
        KeyCode::Down => "down".to_owned(),
        KeyCode::Home => "home".to_owned(),
        KeyCode::End => "end".to_owned(),
        KeyCode::PageUp => "page_up".to_owned(),
        KeyCode::PageDown => "page_down".to_owned(),
        KeyCode::Tab => "tab".to_owned(),
        KeyCode::BackTab => "back_tab".to_owned(),
        KeyCode::Delete => "delete".to_owned(),
        KeyCode::Insert => "insert".to_owned(),
        KeyCode::Null => "null".to_owned(),
        KeyCode::Escape => "esc".to_owned(),
        KeyCode::MediaPlayPause => "media_play_pause".to_owned(),
        KeyCode::MediaStop => "media_stop".to_owned(),
        KeyCode::MediaNextTrack => "media_next_track".to_owned(),
        KeyCode::MediaPrevTrack => "media_prev_track".to_owned(),
        KeyCode::F(n) => format!("f:{n}"),
        KeyCode::Char(ch) => format!("char:{}", u32::from(ch)),
    }
}

fn decode_key_code(encoded: &str) -> Option<KeyCode> {
    if let Some(num) = encoded.strip_prefix("f:") {
        return num.parse::<u8>().ok().map(KeyCode::F);
    }
    if let Some(ch) = encoded
        .strip_prefix("char:")
        .and_then(|value| value.parse::<u32>().ok())
        .and_then(char::from_u32)
    {
        return Some(KeyCode::Char(ch));
    }

    Some(match encoded {
        "backspace" | "Backspace" => KeyCode::Backspace,
        "enter" | "Enter" => KeyCode::Enter,
        "left" | "Left" => KeyCode::Left,
        "right" | "Right" => KeyCode::Right,
        "up" | "Up" => KeyCode::Up,
        "down" | "Down" => KeyCode::Down,
        "home" | "Home" => KeyCode::Home,
        "end" | "End" => KeyCode::End,
        "page_up" | "PageUp" => KeyCode::PageUp,
        "page_down" | "PageDown" => KeyCode::PageDown,
        "tab" | "Tab" => KeyCode::Tab,
        "back_tab" | "BackTab" => KeyCode::BackTab,
        "delete" | "Delete" => KeyCode::Delete,
        "insert" | "Insert" => KeyCode::Insert,
        "null" | "Null" => KeyCode::Null,
        "esc" | "Esc" | "escape" | "Escape" => KeyCode::Escape,
        "media_play_pause" => KeyCode::MediaPlayPause,
        "media_stop" => KeyCode::MediaStop,
        "media_next_track" => KeyCode::MediaNextTrack,
        "media_prev_track" => KeyCode::MediaPrevTrack,
        _ => return None,
    })
}

fn encode_mouse_event(kind: MouseEventKind) -> String {
    match kind {
        MouseEventKind::Down(button) => format!("down:{}", encode_mouse_button(button)),
        MouseEventKind::Up(button) => format!("up:{}", encode_mouse_button(button)),
        MouseEventKind::Drag(button) => format!("drag:{}", encode_mouse_button(button)),
        MouseEventKind::Moved => "moved".to_owned(),
        MouseEventKind::ScrollDown => "scroll_down".to_owned(),
        MouseEventKind::ScrollUp => "scroll_up".to_owned(),
        MouseEventKind::ScrollLeft => "scroll_left".to_owned(),
        MouseEventKind::ScrollRight => "scroll_right".to_owned(),
    }
}

fn decode_mouse_event(encoded: &str) -> Option<MouseEventKind> {
    if let Some(button) = encoded.strip_prefix("down:").and_then(decode_mouse_button) {
        return Some(MouseEventKind::Down(button));
    }
    if let Some(button) = encoded.strip_prefix("up:").and_then(decode_mouse_button) {
        return Some(MouseEventKind::Up(button));
    }
    if let Some(button) = encoded.strip_prefix("drag:").and_then(decode_mouse_button) {
        return Some(MouseEventKind::Drag(button));
    }

    Some(match encoded {
        "moved" | "Moved" => MouseEventKind::Moved,
        "scroll_down" | "ScrollDown" => MouseEventKind::ScrollDown,
        "scroll_up" | "ScrollUp" => MouseEventKind::ScrollUp,
        "scroll_left" | "ScrollLeft" => MouseEventKind::ScrollLeft,
        "scroll_right" | "ScrollRight" => MouseEventKind::ScrollRight,
        _ => return None,
    })
}

const fn encode_mouse_button(button: MouseButton) -> &'static str {
    match button {
        MouseButton::Left => "left",
        MouseButton::Right => "right",
        MouseButton::Middle => "middle",
    }
}

fn decode_mouse_button(encoded: &str) -> Option<MouseButton> {
    match encoded {
        "left" | "Left" => Some(MouseButton::Left),
        "right" | "Right" => Some(MouseButton::Right),
        "middle" | "Middle" => Some(MouseButton::Middle),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use ftui_core::event::{KeyCode, Modifiers};

    use super::*;

    #[test]
    fn recorder_starts_idle() {
        let recorder = ReplayRecorder::new();
        assert_eq!(recorder.state(), ReplayState::Idle);
        assert!(recorder.is_empty());
    }

    #[test]
    fn recorder_records_events() {
        let mut recorder = ReplayRecorder::new();
        recorder.start();

        let event = InputEvent::Key(KeyCode::Char('a'), Modifiers::NONE);
        recorder.record(&event);
        recorder.record(&event);

        assert_eq!(recorder.len(), 2);
        assert_eq!(recorder.state(), ReplayState::Recording);

        recorder.stop();
        assert_eq!(recorder.state(), ReplayState::Idle);
    }

    #[test]
    fn recorder_ignores_when_idle() {
        let mut recorder = ReplayRecorder::new();
        let event = InputEvent::Key(KeyCode::Char('a'), Modifiers::NONE);
        recorder.record(&event);
        assert!(recorder.is_empty());
    }

    #[test]
    fn recorder_ignores_action_events() {
        let mut recorder = ReplayRecorder::new();
        recorder.start();
        let event = InputEvent::Action(crate::input::KeyAction::Quit);
        recorder.record(&event);
        assert!(recorder.is_empty());
    }

    #[test]
    fn recorder_export_json() {
        let mut recorder = ReplayRecorder::new();
        recorder.start();
        let event = InputEvent::Key(KeyCode::Enter, Modifiers::NONE);
        recorder.record(&event);
        recorder.stop();

        let json = recorder.export_json().unwrap();
        assert!(json.contains("key"));
    }

    #[test]
    fn player_playback() {
        let records = vec![
            InputRecord {
                offset: Duration::from_millis(0),
                event: RecordedEvent::from_key(KeyCode::Char('a'), Modifiers::NONE),
            },
            InputRecord {
                offset: Duration::from_millis(100),
                event: RecordedEvent::from_key(KeyCode::Enter, Modifiers::NONE),
            },
        ];

        let mut player = ReplayPlayer::new(records);
        assert_eq!(player.state(), ReplayState::Idle);
        assert_eq!(player.total(), 2);

        player.play();
        assert_eq!(player.state(), ReplayState::Playing);

        let first = player.advance().unwrap();
        assert_eq!(first.offset, Duration::from_millis(0));
        assert_eq!(player.position(), 1);

        let second = player.advance().unwrap();
        assert_eq!(second.offset, Duration::from_millis(100));
        assert_eq!(player.position(), 2);

        // Exhausted.
        assert!(player.advance().is_none());
        assert!(player.is_done());
    }

    #[test]
    fn player_pause_resume() {
        let records = vec![InputRecord {
            offset: Duration::ZERO,
            event: RecordedEvent::from_key(KeyCode::Char('x'), Modifiers::NONE),
        }];

        let mut player = ReplayPlayer::new(records);
        player.play();
        player.pause();
        assert_eq!(player.state(), ReplayState::Paused);
        assert!(player.advance().is_none()); // Paused, no events.

        player.resume();
        assert_eq!(player.state(), ReplayState::Playing);
        assert!(player.advance().is_some());
    }

    #[test]
    fn player_progress() {
        let records = vec![
            InputRecord {
                offset: Duration::ZERO,
                event: RecordedEvent::from_resize(80, 24),
            },
            InputRecord {
                offset: Duration::from_millis(50),
                event: RecordedEvent::from_resize(120, 40),
            },
        ];

        let mut player = ReplayPlayer::new(records);
        assert!((player.progress() - 0.0).abs() < f64::EPSILON);

        player.play();
        let _ = player.advance();
        assert!((player.progress() - 0.5).abs() < f64::EPSILON);

        let _ = player.advance();
        assert!((player.progress() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn player_from_json_roundtrip() {
        let records = vec![InputRecord {
            offset: Duration::from_millis(42),
            event: RecordedEvent::from_key(KeyCode::Char('q'), Modifiers::CTRL),
        }];

        let json = serde_json::to_string(&records).unwrap();
        let player = ReplayPlayer::from_json(&json).unwrap();
        assert_eq!(player.total(), 1);
    }

    #[test]
    fn recorded_event_serde() {
        let event = RecordedEvent::from_key(KeyCode::Enter, Modifiers::SHIFT);
        let json = serde_json::to_string(&event).unwrap();
        let decoded: RecordedEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(decoded, RecordedEvent::Key { .. }));
        if let RecordedEvent::Key { modifiers, .. } = decoded {
            assert_eq!(modifiers, Modifiers::SHIFT.bits());
        }
    }

    #[test]
    fn recorded_event_mouse() {
        let event = RecordedEvent::from_mouse(MouseEventKind::Down(MouseButton::Left), 10, 20);
        let json = serde_json::to_string(&event).unwrap();
        let decoded: RecordedEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(decoded, RecordedEvent::Mouse { .. }));
        if let RecordedEvent::Mouse { col, row, .. } = decoded {
            assert_eq!(col, 10);
            assert_eq!(row, 20);
        }
    }

    #[test]
    fn recorded_event_to_input_event_roundtrip() {
        let key = RecordedEvent::from_key(KeyCode::Char('x'), Modifiers::CTRL);
        assert_eq!(
            key.to_input_event(),
            Some(InputEvent::Key(KeyCode::Char('x'), Modifiers::CTRL))
        );

        let mouse = RecordedEvent::from_mouse(MouseEventKind::ScrollDown, 4, 8);
        assert_eq!(
            mouse.to_input_event(),
            Some(InputEvent::Mouse(MouseEventKind::ScrollDown, 4, 8))
        );

        let resize = RecordedEvent::from_resize(120, 40);
        assert_eq!(resize.to_input_event(), Some(InputEvent::Resize(120, 40)));
    }

    #[test]
    fn player_advance_input_decodes_recorded_events() {
        let mut player = ReplayPlayer::new(vec![InputRecord {
            offset: Duration::from_millis(7),
            event: RecordedEvent::from_key(KeyCode::Enter, Modifiers::NONE),
        }]);
        player.play();

        let (offset, event) = player.advance_input().expect("decoded event");
        assert_eq!(offset, Duration::from_millis(7));
        assert_eq!(event, InputEvent::Key(KeyCode::Enter, Modifiers::NONE));
    }

    #[test]
    fn empty_player_progress() {
        let player = ReplayPlayer::new(vec![]);
        assert!((player.progress() - 1.0).abs() < f64::EPSILON);
    }
}
