//! Unified input model: keymap, bindings, mouse support.
//!
//! Provides a configurable keymap that maps terminal events to semantic
//! [`KeyAction`] values. Product crates extend the action set; the shell
//! handles navigation-level actions (quit, tab switch, palette toggle).

use std::collections::HashMap;

use crossterm::event::{KeyCode, KeyModifiers, MouseEventKind};
use serde::{Deserialize, Serialize};

// ─── Input Event Abstraction ────────────────────────────────────────────────

/// High-level input event consumed by screens and the shell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputEvent {
    /// A key press with modifiers.
    Key(KeyCode, KeyModifiers),
    /// A mouse event at a position.
    Mouse(MouseEventKind, u16, u16),
    /// Terminal resize.
    Resize(u16, u16),
    /// A resolved semantic action (after keymap lookup).
    Action(KeyAction),
}

// ─── Semantic Key Actions ───────────────────────────────────────────────────

/// Semantic action resolved from key bindings.
///
/// Shell-level actions are handled by the app shell. Screen-level actions
/// are forwarded to the active screen. Product crates can define custom
/// actions using the `Custom` variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyAction {
    // ── Shell-level ─────────────────────────────────────────────────
    /// Quit the application.
    Quit,
    /// Toggle the command palette.
    TogglePalette,
    /// Navigate to the next screen (tab).
    NextScreen,
    /// Navigate to the previous screen (shift-tab).
    PrevScreen,
    /// Toggle help overlay.
    ToggleHelp,
    /// Dismiss current overlay / cancel.
    Dismiss,

    // ── Navigation ──────────────────────────────────────────────────
    /// Move focus up.
    Up,
    /// Move focus down.
    Down,
    /// Move focus left.
    Left,
    /// Move focus right.
    Right,
    /// Page up.
    PageUp,
    /// Page down.
    PageDown,
    /// Go to first item.
    Home,
    /// Go to last item.
    End,

    // ── Interaction ─────────────────────────────────────────────────
    /// Confirm / select / enter.
    Confirm,
    /// Delete / backspace.
    Delete,
    /// Copy to clipboard.
    Copy,

    // ── Product-specific ────────────────────────────────────────────
    /// Custom action defined by product crates.
    Custom(String),
}

// ─── Key Binding ────────────────────────────────────────────────────────────

/// A key binding maps a key+modifier combination to a semantic action.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KeyBinding {
    /// The key code.
    pub key: String,
    /// Modifier keys (ctrl, alt, shift).
    pub modifiers: Vec<String>,
    /// The action this binding triggers.
    pub action: KeyAction,
}

// ─── Keymap ─────────────────────────────────────────────────────────────────

/// Configurable keymap that resolves key events to semantic actions.
pub struct Keymap {
    bindings: HashMap<(KeyCode, KeyModifiers), KeyAction>,
}

impl Keymap {
    /// Create a keymap with the default bindings.
    #[must_use]
    pub fn default_bindings() -> Self {
        let mut bindings = HashMap::new();

        // Quit
        bindings.insert((KeyCode::Char('q'), KeyModifiers::NONE), KeyAction::Quit);
        bindings.insert((KeyCode::Char('c'), KeyModifiers::CONTROL), KeyAction::Quit);

        // Command palette
        bindings.insert(
            (KeyCode::Char('p'), KeyModifiers::CONTROL),
            KeyAction::TogglePalette,
        );
        bindings.insert(
            (KeyCode::Char(':'), KeyModifiers::NONE),
            KeyAction::TogglePalette,
        );

        // Navigation
        bindings.insert((KeyCode::Tab, KeyModifiers::NONE), KeyAction::NextScreen);
        bindings.insert(
            (KeyCode::BackTab, KeyModifiers::SHIFT),
            KeyAction::PrevScreen,
        );

        // Help
        bindings.insert(
            (KeyCode::Char('?'), KeyModifiers::NONE),
            KeyAction::ToggleHelp,
        );
        bindings.insert((KeyCode::F(1), KeyModifiers::NONE), KeyAction::ToggleHelp);

        // Dismiss
        bindings.insert((KeyCode::Esc, KeyModifiers::NONE), KeyAction::Dismiss);

        // Movement
        bindings.insert((KeyCode::Up, KeyModifiers::NONE), KeyAction::Up);
        bindings.insert((KeyCode::Down, KeyModifiers::NONE), KeyAction::Down);
        bindings.insert((KeyCode::Left, KeyModifiers::NONE), KeyAction::Left);
        bindings.insert((KeyCode::Right, KeyModifiers::NONE), KeyAction::Right);
        bindings.insert((KeyCode::Char('k'), KeyModifiers::NONE), KeyAction::Up);
        bindings.insert((KeyCode::Char('j'), KeyModifiers::NONE), KeyAction::Down);
        bindings.insert((KeyCode::Char('h'), KeyModifiers::NONE), KeyAction::Left);
        bindings.insert((KeyCode::Char('l'), KeyModifiers::NONE), KeyAction::Right);

        // Page navigation
        bindings.insert((KeyCode::PageUp, KeyModifiers::NONE), KeyAction::PageUp);
        bindings.insert((KeyCode::PageDown, KeyModifiers::NONE), KeyAction::PageDown);
        bindings.insert((KeyCode::Home, KeyModifiers::NONE), KeyAction::Home);
        bindings.insert((KeyCode::End, KeyModifiers::NONE), KeyAction::End);

        // Interaction
        bindings.insert((KeyCode::Enter, KeyModifiers::NONE), KeyAction::Confirm);
        bindings.insert((KeyCode::Backspace, KeyModifiers::NONE), KeyAction::Delete);
        bindings.insert((KeyCode::Char('y'), KeyModifiers::CONTROL), KeyAction::Copy);

        Self { bindings }
    }

    /// Resolve a key event to a semantic action.
    #[must_use]
    pub fn resolve(&self, key: KeyCode, modifiers: KeyModifiers) -> Option<&KeyAction> {
        self.bindings.get(&(key, modifiers))
    }

    /// Add or override a binding.
    pub fn bind(&mut self, key: KeyCode, modifiers: KeyModifiers, action: KeyAction) {
        self.bindings.insert((key, modifiers), action);
    }

    /// Remove a binding.
    pub fn unbind(&mut self, key: KeyCode, modifiers: KeyModifiers) {
        self.bindings.remove(&(key, modifiers));
    }

    /// Number of active bindings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Whether the keymap is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

impl Default for Keymap {
    fn default() -> Self {
        Self::default_bindings()
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyModifiers};

    use super::*;

    #[test]
    fn default_keymap_has_bindings() {
        let keymap = Keymap::default_bindings();
        assert!(!keymap.is_empty());
        assert!(keymap.len() > 15);
    }

    #[test]
    fn resolve_quit_q() {
        let keymap = Keymap::default_bindings();
        let action = keymap.resolve(KeyCode::Char('q'), KeyModifiers::NONE);
        assert_eq!(action, Some(&KeyAction::Quit));
    }

    #[test]
    fn resolve_quit_ctrl_c() {
        let keymap = Keymap::default_bindings();
        let action = keymap.resolve(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(action, Some(&KeyAction::Quit));
    }

    #[test]
    fn resolve_palette_ctrl_p() {
        let keymap = Keymap::default_bindings();
        let action = keymap.resolve(KeyCode::Char('p'), KeyModifiers::CONTROL);
        assert_eq!(action, Some(&KeyAction::TogglePalette));
    }

    #[test]
    fn resolve_vim_movement() {
        let keymap = Keymap::default_bindings();
        assert_eq!(
            keymap.resolve(KeyCode::Char('j'), KeyModifiers::NONE),
            Some(&KeyAction::Down)
        );
        assert_eq!(
            keymap.resolve(KeyCode::Char('k'), KeyModifiers::NONE),
            Some(&KeyAction::Up)
        );
    }

    #[test]
    fn resolve_unknown_returns_none() {
        let keymap = Keymap::default_bindings();
        assert!(
            keymap
                .resolve(KeyCode::Char('z'), KeyModifiers::NONE)
                .is_none()
        );
    }

    #[test]
    fn custom_binding() {
        let mut keymap = Keymap::default_bindings();
        keymap.bind(
            KeyCode::Char('s'),
            KeyModifiers::CONTROL,
            KeyAction::Custom("save".to_string()),
        );
        let action = keymap.resolve(KeyCode::Char('s'), KeyModifiers::CONTROL);
        assert_eq!(action, Some(&KeyAction::Custom("save".to_string())));
    }

    #[test]
    fn unbind_removes_binding() {
        let mut keymap = Keymap::default_bindings();
        assert!(
            keymap
                .resolve(KeyCode::Char('q'), KeyModifiers::NONE)
                .is_some()
        );
        keymap.unbind(KeyCode::Char('q'), KeyModifiers::NONE);
        assert!(
            keymap
                .resolve(KeyCode::Char('q'), KeyModifiers::NONE)
                .is_none()
        );
    }

    #[test]
    fn key_action_serde_roundtrip() {
        for action in [
            KeyAction::Quit,
            KeyAction::TogglePalette,
            KeyAction::Up,
            KeyAction::Custom("test".to_string()),
        ] {
            let json = serde_json::to_string(&action).unwrap();
            let decoded: KeyAction = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, action);
        }
    }
}
