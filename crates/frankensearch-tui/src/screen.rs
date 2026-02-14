//! Screen trait, screen IDs, and registry for TUI navigation.
//!
//! Product crates implement [`Screen`] for each view (search results,
//! indexing progress, fleet overview, etc.) and register them in a
//! [`ScreenRegistry`]. The app shell uses the registry to navigate
//! between screens while preserving context.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;

use ratatui::Frame;
use serde::{Deserialize, Serialize};

use crate::input::InputEvent;

// ─── Screen Identity ────────────────────────────────────────────────────────

/// Unique identifier for a screen within the TUI.
///
/// Screen IDs use a `namespace.name` convention:
/// - `fsfs.search` — fsfs search results screen
/// - `fsfs.indexing` — fsfs indexing progress screen
/// - `ops.fleet` — ops fleet overview screen
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScreenId(pub String);

impl ScreenId {
    /// Create a new screen ID.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl fmt::Display for ScreenId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ─── Screen Context ─────────────────────────────────────────────────────────

/// Context passed to a screen during rendering and event handling.
///
/// Contains navigation state, current theme, and shared metadata that
/// screens need to render correctly.
#[derive(Debug, Clone)]
pub struct ScreenContext {
    /// The ID of the currently active screen.
    pub active_screen: ScreenId,
    /// Width of the terminal in columns.
    pub terminal_width: u16,
    /// Height of the terminal in rows.
    pub terminal_height: u16,
    /// Whether the screen is focused (receives input).
    pub focused: bool,
}

// ─── Screen Trait ───────────────────────────────────────────────────────────

/// Trait that product crates implement for each TUI view.
///
/// Screens handle rendering to a ratatui `Frame` and processing input
/// events. The app shell manages lifecycle, focus, and navigation.
pub trait Screen: Send {
    /// Unique identifier for this screen.
    fn id(&self) -> &ScreenId;

    /// Human-readable title for the status bar / breadcrumbs.
    fn title(&self) -> &str;

    /// Render the screen content into the provided frame area.
    fn render(&self, frame: &mut Frame<'_>, ctx: &ScreenContext);

    /// Handle an input event. Returns a [`ScreenAction`] indicating
    /// what the shell should do next.
    fn handle_input(&mut self, event: &InputEvent, ctx: &ScreenContext) -> ScreenAction;

    /// Called when this screen gains focus.
    fn on_focus(&mut self) {}

    /// Called when this screen loses focus.
    fn on_blur(&mut self) {}

    /// Semantic role for accessibility (screen reader hint).
    fn semantic_role(&self) -> &'static str {
        "region"
    }

    /// Downcast support for product crates that need concrete screen state.
    fn as_any(&self) -> &dyn Any;

    /// Mutable downcast support for product crates that need concrete screen state.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Action returned by a screen's input handler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScreenAction {
    /// Input was consumed; no navigation needed.
    Consumed,
    /// Input was not handled; pass to shell.
    Ignored,
    /// Navigate to a different screen.
    Navigate(ScreenId),
    /// Open an overlay (help, confirmation, etc.).
    OpenOverlay(String),
    /// Request application exit.
    Quit,
}

// ─── Screen Registry ────────────────────────────────────────────────────────

/// Registry of available screens for navigation.
///
/// Product crates register their screens at startup. The app shell
/// uses the registry to look up screens by ID and manage navigation
/// history.
pub struct ScreenRegistry {
    screens: HashMap<ScreenId, Box<dyn Screen>>,
    order: Vec<ScreenId>,
}

impl ScreenRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            screens: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Register a screen. The insertion order determines the default
    /// navigation order (tab cycling).
    pub fn register(&mut self, screen: Box<dyn Screen>) {
        let id = screen.id().clone();
        self.order.push(id.clone());
        self.screens.insert(id, screen);
    }

    /// Look up a screen by ID.
    #[must_use]
    pub fn get(&self, id: &ScreenId) -> Option<&dyn Screen> {
        self.screens.get(id).map(AsRef::as_ref)
    }

    /// Look up a screen mutably by ID.
    pub fn get_mut(&mut self, id: &ScreenId) -> Option<&mut Box<dyn Screen>> {
        self.screens.get_mut(id)
    }

    /// Get the ordered list of screen IDs.
    #[must_use]
    pub fn screen_ids(&self) -> &[ScreenId] {
        &self.order
    }

    /// Get the next screen ID in tab order (wraps around).
    #[must_use]
    pub fn next_screen(&self, current: &ScreenId) -> Option<&ScreenId> {
        let pos = self.order.iter().position(|id| id == current)?;
        let next = (pos + 1) % self.order.len();
        self.order.get(next)
    }

    /// Get the previous screen ID in tab order (wraps around).
    #[must_use]
    pub fn prev_screen(&self, current: &ScreenId) -> Option<&ScreenId> {
        let pos = self.order.iter().position(|id| id == current)?;
        let prev = if pos == 0 {
            self.order.len() - 1
        } else {
            pos - 1
        };
        self.order.get(prev)
    }

    /// Number of registered screens.
    #[must_use]
    pub fn len(&self) -> usize {
        self.screens.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.screens.is_empty()
    }
}

impl Default for ScreenRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use ratatui::Frame;

    use super::*;

    struct TestScreen {
        id: ScreenId,
        title: String,
    }

    impl TestScreen {
        fn new(id: &str, title: &str) -> Self {
            Self {
                id: ScreenId::new(id),
                title: title.to_string(),
            }
        }
    }

    impl Screen for TestScreen {
        fn id(&self) -> &ScreenId {
            &self.id
        }

        fn title(&self) -> &str {
            &self.title
        }

        fn render(&self, _frame: &mut Frame<'_>, _ctx: &ScreenContext) {}

        fn handle_input(&mut self, _event: &InputEvent, _ctx: &ScreenContext) -> ScreenAction {
            ScreenAction::Ignored
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[test]
    fn screen_id_display() {
        let id = ScreenId::new("fsfs.search");
        assert_eq!(id.to_string(), "fsfs.search");
    }

    #[test]
    fn screen_id_serde_roundtrip() {
        let id = ScreenId::new("ops.fleet");
        let json = serde_json::to_string(&id).unwrap();
        let decoded: ScreenId = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, id);
    }

    #[test]
    fn registry_register_and_lookup() {
        let mut reg = ScreenRegistry::new();
        reg.register(Box::new(TestScreen::new("a", "Screen A")));
        reg.register(Box::new(TestScreen::new("b", "Screen B")));

        assert_eq!(reg.len(), 2);
        assert!(!reg.is_empty());
        assert_eq!(reg.get(&ScreenId::new("a")).unwrap().title(), "Screen A");
        assert_eq!(reg.get(&ScreenId::new("b")).unwrap().title(), "Screen B");
        assert!(reg.get(&ScreenId::new("c")).is_none());
    }

    #[test]
    fn registry_screen_ids_preserves_order() {
        let mut reg = ScreenRegistry::new();
        reg.register(Box::new(TestScreen::new("x", "X")));
        reg.register(Box::new(TestScreen::new("y", "Y")));
        reg.register(Box::new(TestScreen::new("z", "Z")));

        let ids: Vec<&str> = reg.screen_ids().iter().map(|id| id.0.as_str()).collect();
        assert_eq!(ids, vec!["x", "y", "z"]);
    }

    #[test]
    fn registry_next_wraps_around() {
        let mut reg = ScreenRegistry::new();
        reg.register(Box::new(TestScreen::new("a", "A")));
        reg.register(Box::new(TestScreen::new("b", "B")));
        reg.register(Box::new(TestScreen::new("c", "C")));

        let next = reg.next_screen(&ScreenId::new("c")).unwrap();
        assert_eq!(next, &ScreenId::new("a"));
    }

    #[test]
    fn registry_prev_wraps_around() {
        let mut reg = ScreenRegistry::new();
        reg.register(Box::new(TestScreen::new("a", "A")));
        reg.register(Box::new(TestScreen::new("b", "B")));

        let prev = reg.prev_screen(&ScreenId::new("a")).unwrap();
        assert_eq!(prev, &ScreenId::new("b"));
    }

    #[test]
    fn empty_registry() {
        let reg = ScreenRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }
}
