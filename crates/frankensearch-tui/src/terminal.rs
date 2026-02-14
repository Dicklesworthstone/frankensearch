//! Terminal mode detection and inline/alt-screen resilience.
//!
//! Handles the difference between alt-screen mode (full TUI takeover)
//! and inline mode (renders within the existing scrollback). Provides
//! reconnect/restart event handling so the TUI recovers gracefully from
//! terminal disconnects, SIGWINCH bursts, and mode transitions.

use serde::{Deserialize, Serialize};

// ─── Terminal Mode ──────────────────────────────────────────────────────────

/// Terminal rendering mode.
///
/// The TUI must behave correctly in both modes. Alt-screen mode owns the
/// entire terminal; inline mode coexists with scrollback history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalMode {
    /// Alt-screen mode: full terminal takeover, no scrollback.
    #[default]
    AltScreen,
    /// Inline mode: renders within existing scrollback.
    Inline,
}

impl TerminalMode {
    /// Whether this mode uses the alternate screen buffer.
    #[must_use]
    pub const fn uses_alt_screen(self) -> bool {
        matches!(self, Self::AltScreen)
    }

    /// Human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::AltScreen => "Alt Screen",
            Self::Inline => "Inline",
        }
    }
}

impl std::fmt::Display for TerminalMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ─── Terminal Event ─────────────────────────────────────────────────────────

/// Terminal lifecycle events that the shell must handle for resilience.
///
/// These events represent conditions where the terminal state may have
/// changed in ways that require the TUI to re-sync its rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminalEvent {
    /// Terminal was resized. The shell should re-layout.
    Resized {
        /// New width in columns.
        width: u16,
        /// New height in rows.
        height: u16,
    },
    /// Terminal was reconnected after a disconnect (e.g., SSH dropped).
    /// The shell should re-enter raw mode and clear the screen.
    Reconnected,
    /// Terminal focus was gained (window came to foreground).
    FocusGained,
    /// Terminal focus was lost (window went to background).
    FocusLost,
    /// Mode transition requested (e.g., alt-screen to inline or vice versa).
    ModeChange(TerminalMode),
    /// SIGWINCH burst detected — multiple rapid resizes. The shell should
    /// debounce and only process the final size.
    ResizeBurst {
        /// Final width after the burst.
        final_width: u16,
        /// Final height after the burst.
        final_height: u16,
        /// Number of resize events in the burst.
        burst_count: u32,
    },
}

// ─── Terminal State ─────────────────────────────────────────────────────────

/// Tracks the current terminal state for resilience logic.
///
/// The shell consults this to decide whether a full re-render is needed
/// after lifecycle events, and to debounce SIGWINCH bursts.
#[derive(Debug, Clone)]
pub struct TerminalState {
    /// Current rendering mode.
    pub mode: TerminalMode,
    /// Current terminal width.
    pub width: u16,
    /// Current terminal height.
    pub height: u16,
    /// Whether the terminal currently has focus.
    pub focused: bool,
    /// Whether the terminal is connected.
    pub connected: bool,
    /// Count of resize events since last render.
    pending_resizes: u32,
    /// Whether a full re-render is needed (e.g., after reconnect).
    needs_full_redraw: bool,
}

impl TerminalState {
    /// Create a new terminal state with default dimensions.
    #[must_use]
    pub const fn new(mode: TerminalMode) -> Self {
        Self {
            mode,
            width: 80,
            height: 24,
            focused: true,
            connected: true,
            pending_resizes: 0,
            needs_full_redraw: true, // First render is always full.
        }
    }

    /// Process a terminal event and update state.
    ///
    /// Returns `true` if the event requires a re-render.
    pub fn handle_event(&mut self, event: &TerminalEvent) -> bool {
        match event {
            TerminalEvent::Resized { width, height } => {
                let changed = self.width != *width || self.height != *height;
                self.width = *width;
                self.height = *height;
                self.pending_resizes += 1;
                if changed {
                    self.needs_full_redraw = true;
                }
                changed
            }
            TerminalEvent::Reconnected => {
                self.connected = true;
                self.needs_full_redraw = true;
                true
            }
            TerminalEvent::FocusGained => {
                let changed = !self.focused;
                self.focused = true;
                changed
            }
            TerminalEvent::FocusLost => {
                let changed = self.focused;
                self.focused = false;
                changed
            }
            TerminalEvent::ModeChange(new_mode) => {
                let changed = self.mode != *new_mode;
                self.mode = *new_mode;
                if changed {
                    self.needs_full_redraw = true;
                }
                changed
            }
            TerminalEvent::ResizeBurst {
                final_width,
                final_height,
                burst_count,
            } => {
                let changed = self.width != *final_width || self.height != *final_height;
                self.width = *final_width;
                self.height = *final_height;
                self.pending_resizes += burst_count;
                if changed {
                    self.needs_full_redraw = true;
                }
                changed
            }
        }
    }

    /// Whether a full redraw is needed.
    #[must_use]
    pub const fn needs_full_redraw(&self) -> bool {
        self.needs_full_redraw
    }

    /// Acknowledge the full redraw (call after rendering).
    pub const fn acknowledge_redraw(&mut self) {
        self.needs_full_redraw = false;
        self.pending_resizes = 0;
    }

    /// Mark the terminal as disconnected.
    pub const fn mark_disconnected(&mut self) {
        self.connected = false;
    }

    /// Number of pending resize events since last render.
    #[must_use]
    pub const fn pending_resizes(&self) -> u32 {
        self.pending_resizes
    }

    /// Whether the terminal is currently connected.
    #[must_use]
    pub const fn is_connected(&self) -> bool {
        self.connected
    }
}

impl Default for TerminalState {
    fn default() -> Self {
        Self::new(TerminalMode::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_mode_default() {
        assert_eq!(TerminalMode::default(), TerminalMode::AltScreen);
    }

    #[test]
    fn terminal_mode_alt_screen() {
        assert!(TerminalMode::AltScreen.uses_alt_screen());
        assert!(!TerminalMode::Inline.uses_alt_screen());
    }

    #[test]
    fn terminal_mode_labels() {
        assert_eq!(TerminalMode::AltScreen.label(), "Alt Screen");
        assert_eq!(TerminalMode::Inline.label(), "Inline");
    }

    #[test]
    fn terminal_mode_display() {
        assert_eq!(TerminalMode::AltScreen.to_string(), "Alt Screen");
        assert_eq!(TerminalMode::Inline.to_string(), "Inline");
    }

    #[test]
    fn terminal_mode_serde_roundtrip() {
        for mode in [TerminalMode::AltScreen, TerminalMode::Inline] {
            let json = serde_json::to_string(&mode).unwrap();
            let decoded: TerminalMode = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, mode);
        }
    }

    #[test]
    fn terminal_state_defaults() {
        let state = TerminalState::new(TerminalMode::AltScreen);
        assert_eq!(state.mode, TerminalMode::AltScreen);
        assert_eq!(state.width, 80);
        assert_eq!(state.height, 24);
        assert!(state.focused);
        assert!(state.connected);
        assert!(state.needs_full_redraw()); // First render.
    }

    #[test]
    fn terminal_state_resize() {
        let mut state = TerminalState::new(TerminalMode::AltScreen);
        state.acknowledge_redraw();
        assert!(!state.needs_full_redraw());

        let changed = state.handle_event(&TerminalEvent::Resized {
            width: 120,
            height: 40,
        });
        assert!(changed);
        assert_eq!(state.width, 120);
        assert_eq!(state.height, 40);
        assert!(state.needs_full_redraw());
        assert_eq!(state.pending_resizes(), 1);
    }

    #[test]
    fn terminal_state_same_size_no_change() {
        let mut state = TerminalState::new(TerminalMode::AltScreen);
        state.acknowledge_redraw();

        let changed = state.handle_event(&TerminalEvent::Resized {
            width: 80,
            height: 24,
        });
        assert!(!changed);
    }

    #[test]
    fn terminal_state_reconnect() {
        let mut state = TerminalState::new(TerminalMode::AltScreen);
        state.acknowledge_redraw();
        state.mark_disconnected();
        assert!(!state.is_connected());

        let changed = state.handle_event(&TerminalEvent::Reconnected);
        assert!(changed);
        assert!(state.is_connected());
        assert!(state.needs_full_redraw());
    }

    #[test]
    fn terminal_state_focus_gain_loss() {
        let mut state = TerminalState::new(TerminalMode::AltScreen);

        // Losing focus.
        let changed = state.handle_event(&TerminalEvent::FocusLost);
        assert!(changed);
        assert!(!state.focused);

        // Gaining focus.
        let changed = state.handle_event(&TerminalEvent::FocusGained);
        assert!(changed);
        assert!(state.focused);

        // Gaining focus again (no change).
        let changed = state.handle_event(&TerminalEvent::FocusGained);
        assert!(!changed);
    }

    #[test]
    fn terminal_state_mode_change() {
        let mut state = TerminalState::new(TerminalMode::AltScreen);
        state.acknowledge_redraw();

        let changed = state.handle_event(&TerminalEvent::ModeChange(TerminalMode::Inline));
        assert!(changed);
        assert_eq!(state.mode, TerminalMode::Inline);
        assert!(state.needs_full_redraw());

        // Same mode (no change).
        state.acknowledge_redraw();
        let changed = state.handle_event(&TerminalEvent::ModeChange(TerminalMode::Inline));
        assert!(!changed);
    }

    #[test]
    fn terminal_state_resize_burst() {
        let mut state = TerminalState::new(TerminalMode::AltScreen);
        state.acknowledge_redraw();

        let changed = state.handle_event(&TerminalEvent::ResizeBurst {
            final_width: 200,
            final_height: 50,
            burst_count: 5,
        });
        assert!(changed);
        assert_eq!(state.width, 200);
        assert_eq!(state.height, 50);
        assert_eq!(state.pending_resizes(), 5);
        assert!(state.needs_full_redraw());
    }

    #[test]
    fn terminal_state_acknowledge_redraw() {
        let mut state = TerminalState::new(TerminalMode::AltScreen);
        assert!(state.needs_full_redraw());

        state.acknowledge_redraw();
        assert!(!state.needs_full_redraw());
        assert_eq!(state.pending_resizes(), 0);
    }

    #[test]
    fn terminal_event_resize_equality() {
        let a = TerminalEvent::Resized {
            width: 80,
            height: 24,
        };
        let b = TerminalEvent::Resized {
            width: 80,
            height: 24,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn terminal_event_reconnected() {
        assert_eq!(TerminalEvent::Reconnected, TerminalEvent::Reconnected);
    }

    #[test]
    fn terminal_state_default_impl() {
        let state = TerminalState::default();
        assert_eq!(state.mode, TerminalMode::AltScreen);
    }

    #[test]
    fn terminal_state_multiple_events() {
        let mut state = TerminalState::new(TerminalMode::AltScreen);
        state.acknowledge_redraw();

        // Resize, then focus loss, then reconnect.
        state.handle_event(&TerminalEvent::Resized {
            width: 100,
            height: 30,
        });
        state.handle_event(&TerminalEvent::FocusLost);
        state.handle_event(&TerminalEvent::Reconnected);

        assert_eq!(state.width, 100);
        assert_eq!(state.height, 30);
        assert!(!state.focused);
        assert!(state.connected);
        assert!(state.needs_full_redraw());
    }
}
