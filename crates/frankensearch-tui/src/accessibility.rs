//! Accessibility: focus management and semantic annotations.
//!
//! Provides a [`FocusManager`] that tracks which widget has keyboard
//! focus and [`SemanticRole`] annotations for screen readers. Product
//! crates annotate their widgets; the shell manages focus transitions.

use serde::{Deserialize, Serialize};

// ─── Semantic Role ───────────────────────────────────────────────────────────

/// Semantic role for accessibility / screen reader hints.
///
/// Based loosely on WAI-ARIA roles, adapted for terminal UIs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticRole {
    /// A named region or panel.
    Region,
    /// Navigation container (tabs, breadcrumbs).
    Navigation,
    /// A list of items.
    List,
    /// A single item in a list.
    ListItem,
    /// A search input field.
    Search,
    /// A status indicator / progress bar.
    Status,
    /// An alert or notification.
    Alert,
    /// A dialog or overlay.
    Dialog,
    /// A toolbar.
    Toolbar,
    /// A data table or grid.
    Grid,
}

impl std::fmt::Display for SemanticRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Region => write!(f, "region"),
            Self::Navigation => write!(f, "navigation"),
            Self::List => write!(f, "list"),
            Self::ListItem => write!(f, "listitem"),
            Self::Search => write!(f, "search"),
            Self::Status => write!(f, "status"),
            Self::Alert => write!(f, "alert"),
            Self::Dialog => write!(f, "dialog"),
            Self::Toolbar => write!(f, "toolbar"),
            Self::Grid => write!(f, "grid"),
        }
    }
}

// ─── Focus Direction ─────────────────────────────────────────────────────────

/// Direction for focus movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusDirection {
    /// Move focus to the next widget.
    Next,
    /// Move focus to the previous widget.
    Previous,
    /// Move focus up.
    Up,
    /// Move focus down.
    Down,
    /// Move focus left.
    Left,
    /// Move focus right.
    Right,
}

// ─── Focus Manager ───────────────────────────────────────────────────────────

/// Manages keyboard focus across widgets within a screen.
///
/// Widgets register with string IDs. The focus manager tracks which
/// widget is currently focused and handles directional navigation.
pub struct FocusManager {
    /// Ordered list of focusable widget IDs.
    widgets: Vec<String>,
    /// Index of the currently focused widget.
    focused: usize,
}

impl FocusManager {
    /// Create an empty focus manager.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            widgets: Vec::new(),
            focused: 0,
        }
    }

    /// Register a focusable widget.
    pub fn register(&mut self, id: impl Into<String>) {
        self.widgets.push(id.into());
    }

    /// Get the currently focused widget ID.
    #[must_use]
    pub fn focused(&self) -> Option<&str> {
        self.widgets.get(self.focused).map(String::as_str)
    }

    /// Check if a specific widget is focused.
    #[must_use]
    pub fn is_focused(&self, id: &str) -> bool {
        self.focused() == Some(id)
    }

    /// Move focus in the given direction.
    pub const fn move_focus(&mut self, direction: FocusDirection) {
        if self.widgets.is_empty() {
            return;
        }

        match direction {
            FocusDirection::Next | FocusDirection::Down | FocusDirection::Right => {
                self.focused = (self.focused + 1) % self.widgets.len();
            }
            FocusDirection::Previous | FocusDirection::Up | FocusDirection::Left => {
                self.focused = if self.focused == 0 {
                    self.widgets.len() - 1
                } else {
                    self.focused - 1
                };
            }
        }
    }

    /// Focus a specific widget by ID. Returns `true` if found.
    pub fn focus(&mut self, id: &str) -> bool {
        if let Some(pos) = self.widgets.iter().position(|w| w == id) {
            self.focused = pos;
            true
        } else {
            false
        }
    }

    /// Number of focusable widgets.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.widgets.len()
    }

    /// Whether there are no focusable widgets.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.widgets.is_empty()
    }

    /// Clear all registered widgets and reset focus.
    pub fn clear(&mut self) {
        self.widgets.clear();
        self.focused = 0;
    }
}

impl Default for FocusManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn focus_manager_empty() {
        let fm = FocusManager::new();
        assert!(fm.is_empty());
        assert!(fm.focused().is_none());
    }

    #[test]
    fn focus_manager_register_and_focus() {
        let mut fm = FocusManager::new();
        fm.register("search");
        fm.register("results");
        fm.register("details");

        assert_eq!(fm.len(), 3);
        assert_eq!(fm.focused(), Some("search"));
        assert!(fm.is_focused("search"));
    }

    #[test]
    fn focus_manager_next_wraps() {
        let mut fm = FocusManager::new();
        fm.register("a");
        fm.register("b");
        fm.register("c");

        fm.move_focus(FocusDirection::Next);
        assert_eq!(fm.focused(), Some("b"));
        fm.move_focus(FocusDirection::Next);
        assert_eq!(fm.focused(), Some("c"));
        fm.move_focus(FocusDirection::Next);
        assert_eq!(fm.focused(), Some("a")); // Wrap.
    }

    #[test]
    fn focus_manager_prev_wraps() {
        let mut fm = FocusManager::new();
        fm.register("a");
        fm.register("b");

        fm.move_focus(FocusDirection::Previous);
        assert_eq!(fm.focused(), Some("b")); // Wrap backward.
    }

    #[test]
    fn focus_manager_focus_by_id() {
        let mut fm = FocusManager::new();
        fm.register("x");
        fm.register("y");
        fm.register("z");

        assert!(fm.focus("z"));
        assert_eq!(fm.focused(), Some("z"));
        assert!(!fm.focus("nonexistent"));
    }

    #[test]
    fn focus_manager_clear() {
        let mut fm = FocusManager::new();
        fm.register("a");
        fm.register("b");
        fm.clear();
        assert!(fm.is_empty());
        assert!(fm.focused().is_none());
    }

    #[test]
    fn semantic_role_display() {
        assert_eq!(SemanticRole::Region.to_string(), "region");
        assert_eq!(SemanticRole::Navigation.to_string(), "navigation");
        assert_eq!(SemanticRole::Grid.to_string(), "grid");
    }

    #[test]
    fn semantic_role_serde_roundtrip() {
        for role in [
            SemanticRole::Region,
            SemanticRole::List,
            SemanticRole::Search,
            SemanticRole::Dialog,
        ] {
            let json = serde_json::to_string(&role).unwrap();
            let decoded: SemanticRole = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, role);
        }
    }

    #[test]
    fn focus_direction_aliases() {
        let mut fm = FocusManager::new();
        fm.register("a");
        fm.register("b");

        // Down should behave like Next.
        fm.move_focus(FocusDirection::Down);
        assert_eq!(fm.focused(), Some("b"));

        // Up should behave like Previous.
        fm.move_focus(FocusDirection::Up);
        assert_eq!(fm.focused(), Some("a"));
    }
}
