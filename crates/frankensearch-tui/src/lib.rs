//! Shared TUI framework for frankensearch products.
//!
//! This crate provides reusable terminal UI primitives shared by both the
//! fsfs deluxe TUI and the ops observability TUI. It ensures consistent UX,
//! keyboard shortcuts, theming, and accessibility across all frankensearch
//! TUI products.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │  Product crates (fsfs, ops)                     │
//! │  └─ product-specific screens + data sources     │
//! ├─────────────────────────────────────────────────┤
//! │  frankensearch-tui (this crate)                 │
//! │  ├─ screen: Screen trait, ScreenId, registry    │
//! │  ├─ shell: App shell, status bar, breadcrumbs   │
//! │  ├─ palette: Command palette, action routing    │
//! │  ├─ input: Keymap, bindings, mouse model        │
//! │  ├─ theme: Color schemes, dark/light presets    │
//! │  ├─ overlay: Help, alerts, confirmation dialogs │
//! │  ├─ accessibility: Focus, semantic annotations  │
//! │  ├─ frame: Budget enforcement, jank detection   │
//! │  ├─ replay: Input recording, deterministic play │
//! │  ├─ determinism: Clock trait, seeds, replay ctx  │
//! │  ├─ evidence: JSONL evidence hooks + redaction   │
//! │  └─ terminal: Mode detection, reconnect handler │
//! ├─────────────────────────────────────────────────┤
//! │  ratatui + crossterm                            │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! Product crates implement the [`Screen`] trait for their views, register
//! them in a [`ScreenRegistry`], and hand control to the [`AppShell`] which
//! manages navigation, overlays, input dispatch, and frame timing.

#![forbid(unsafe_code)]

pub mod accessibility;
pub mod determinism;
pub mod evidence;
pub mod frame;
pub mod input;
pub mod overlay;
pub mod palette;
pub mod replay;
pub mod screen;
pub mod shell;
pub mod terminal;
pub mod theme;

// ─── Re-exports ─────────────────────────────────────────────────────────────

pub use accessibility::{FocusDirection, FocusManager, SemanticRole};
pub use determinism::{Clock, DeterministicSeed, ReplayMetadata, ReplayMode, TickClock, WallClock};
pub use evidence::{
    EvidenceEnvelope, EvidenceEvent, EvidenceEventType, EvidencePayload, EvidenceReason,
    EvidenceRedaction, EvidenceSeverity, EvidenceSink, EvidenceTrace, EvidenceWriteError,
    NoopWriter, RedactionTransform, VecWriter,
};
pub use frame::{FrameBudget, FrameMetrics, JankCallback};
pub use input::{InputEvent, KeyAction, KeyBinding, Keymap};
pub use overlay::{OverlayKind, OverlayManager, OverlayRequest};
pub use palette::{Action, ActionCategory, CommandPalette, PaletteState};
pub use replay::{InputRecord, ReplayPlayer, ReplayRecorder, ReplayState};
pub use screen::{Screen, ScreenContext, ScreenId, ScreenRegistry};
pub use shell::{AppShell, ShellConfig, StatusLine};
pub use terminal::{TerminalEvent, TerminalMode, TerminalState};
pub use theme::{ColorScheme, Theme, ThemePreset};
