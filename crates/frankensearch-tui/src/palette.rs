//! Command palette: fuzzy-matched action lookup and dispatch.
//!
//! The [`CommandPalette`] provides a searchable list of actions that the
//! user can invoke by name. Product crates register domain-specific actions;
//! the shell handles navigation-level actions.

use serde::{Deserialize, Serialize};

// ─── Action Category ─────────────────────────────────────────────────────────

/// Category for grouping palette actions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionCategory {
    /// Navigation actions (go to screen, tab switch).
    Navigation,
    /// Search-related actions.
    Search,
    /// Configuration / settings.
    Settings,
    /// Diagnostic / debug actions.
    Debug,
    /// Product-specific category.
    Custom(String),
}

impl std::fmt::Display for ActionCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Navigation => write!(f, "Navigation"),
            Self::Search => write!(f, "Search"),
            Self::Settings => write!(f, "Settings"),
            Self::Debug => write!(f, "Debug"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

// ─── Action ──────────────────────────────────────────────────────────────────

/// A named action that can be invoked from the command palette.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    /// Unique identifier for the action.
    pub id: String,
    /// Human-readable label shown in the palette.
    pub label: String,
    /// Optional description / help text.
    pub description: Option<String>,
    /// Category for grouping.
    pub category: ActionCategory,
    /// Optional keyboard shortcut hint (display only).
    pub shortcut: Option<String>,
}

impl Action {
    /// Create a new action.
    #[must_use]
    pub fn new(id: impl Into<String>, label: impl Into<String>, category: ActionCategory) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            description: None,
            category,
            shortcut: None,
        }
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the shortcut hint.
    #[must_use]
    pub fn with_shortcut(mut self, shortcut: impl Into<String>) -> Self {
        self.shortcut = Some(shortcut.into());
        self
    }
}

// ─── Palette State ───────────────────────────────────────────────────────────

/// State of the command palette overlay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaletteState {
    /// Palette is closed / hidden.
    Closed,
    /// Palette is open and accepting input.
    Open,
}

// ─── Command Palette ─────────────────────────────────────────────────────────

/// Command palette that provides fuzzy-filtered action search.
pub struct CommandPalette {
    /// All registered actions.
    actions: Vec<Action>,
    /// Current search query.
    query: String,
    /// Current selection index in filtered results.
    selected: usize,
    /// Current state.
    state: PaletteState,
}

impl CommandPalette {
    /// Create an empty command palette.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            actions: Vec::new(),
            query: String::new(),
            selected: 0,
            state: PaletteState::Closed,
        }
    }

    /// Register an action.
    pub fn register(&mut self, action: Action) {
        self.actions.push(action);
    }

    /// Open the palette.
    pub fn open(&mut self) {
        self.state = PaletteState::Open;
        self.query.clear();
        self.selected = 0;
    }

    /// Close the palette.
    pub fn close(&mut self) {
        self.state = PaletteState::Closed;
        self.query.clear();
        self.selected = 0;
    }

    /// Toggle the palette open/closed.
    pub fn toggle(&mut self) {
        match self.state {
            PaletteState::Closed => self.open(),
            PaletteState::Open => self.close(),
        }
    }

    /// Current palette state.
    #[must_use]
    pub const fn state(&self) -> &PaletteState {
        &self.state
    }

    /// Current search query.
    #[must_use]
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Append a character to the query.
    pub fn push_char(&mut self, ch: char) {
        self.query.push(ch);
        self.selected = 0;
    }

    /// Remove the last character from the query.
    pub fn pop_char(&mut self) {
        self.query.pop();
        self.selected = 0;
    }

    /// Get filtered actions matching the current query.
    #[must_use]
    pub fn filtered(&self) -> Vec<&Action> {
        if self.query.is_empty() {
            return self.actions.iter().collect();
        }

        let query_lower = self.query.to_lowercase();
        self.actions
            .iter()
            .filter(|a| {
                a.label.to_lowercase().contains(&query_lower)
                    || a.id.to_lowercase().contains(&query_lower)
                    || a.description
                        .as_ref()
                        .is_some_and(|d| d.to_lowercase().contains(&query_lower))
            })
            .collect()
    }

    /// Currently selected index.
    #[must_use]
    pub const fn selected(&self) -> usize {
        self.selected
    }

    /// Move selection up.
    pub fn select_prev(&mut self) {
        let count = self.filtered().len();
        if count > 0 {
            self.selected = if self.selected == 0 {
                count - 1
            } else {
                self.selected - 1
            };
        }
    }

    /// Move selection down.
    pub fn select_next(&mut self) {
        let count = self.filtered().len();
        if count > 0 {
            self.selected = (self.selected + 1) % count;
        }
    }

    /// Confirm the current selection. Returns the selected action's ID.
    #[must_use]
    pub fn confirm(&self) -> Option<String> {
        let filtered = self.filtered();
        filtered.get(self.selected).map(|a| a.id.clone())
    }

    /// Number of registered actions.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.actions.len()
    }

    /// Whether the palette has no actions.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}

impl Default for CommandPalette {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_palette() -> CommandPalette {
        let mut palette = CommandPalette::new();
        palette.register(
            Action::new("nav.search", "Go to Search", ActionCategory::Navigation)
                .with_shortcut("Ctrl+1"),
        );
        palette.register(
            Action::new("nav.index", "Go to Indexing", ActionCategory::Navigation)
                .with_shortcut("Ctrl+2"),
        );
        palette.register(
            Action::new("debug.logs", "Show Logs", ActionCategory::Debug)
                .with_description("Display structured log output"),
        );
        palette.register(Action::new(
            "settings.theme",
            "Change Theme",
            ActionCategory::Settings,
        ));
        palette
    }

    #[test]
    fn palette_starts_closed() {
        let palette = CommandPalette::new();
        assert_eq!(palette.state(), &PaletteState::Closed);
        assert!(palette.is_empty());
    }

    #[test]
    fn palette_register_and_count() {
        let palette = sample_palette();
        assert_eq!(palette.len(), 4);
    }

    #[test]
    fn palette_toggle() {
        let mut palette = sample_palette();
        assert_eq!(palette.state(), &PaletteState::Closed);
        palette.toggle();
        assert_eq!(palette.state(), &PaletteState::Open);
        palette.toggle();
        assert_eq!(palette.state(), &PaletteState::Closed);
    }

    #[test]
    fn palette_filter_empty_query() {
        let palette = sample_palette();
        assert_eq!(palette.filtered().len(), 4);
    }

    #[test]
    fn palette_filter_by_label() {
        let mut palette = sample_palette();
        palette.open();
        palette.push_char('s');
        palette.push_char('e');
        palette.push_char('a');
        // Should match "Go to Search"
        let results = palette.filtered();
        assert!(results.iter().any(|a| a.id == "nav.search"));
    }

    #[test]
    fn palette_filter_by_description() {
        let mut palette = sample_palette();
        palette.open();
        for ch in "structured".chars() {
            palette.push_char(ch);
        }
        let results = palette.filtered();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "debug.logs");
    }

    #[test]
    fn palette_navigation() {
        let mut palette = sample_palette();
        palette.open();
        assert_eq!(palette.selected(), 0);
        palette.select_next();
        assert_eq!(palette.selected(), 1);
        palette.select_prev();
        assert_eq!(palette.selected(), 0);
        // Wrap around.
        palette.select_prev();
        assert_eq!(palette.selected(), 3);
    }

    #[test]
    fn palette_confirm() {
        let mut palette = sample_palette();
        palette.open();
        let id = palette.confirm();
        assert_eq!(id, Some("nav.search".to_string()));
    }

    #[test]
    fn palette_pop_char() {
        let mut palette = sample_palette();
        palette.open();
        palette.push_char('x');
        palette.push_char('y');
        assert_eq!(palette.query(), "xy");
        palette.pop_char();
        assert_eq!(palette.query(), "x");
    }

    #[test]
    fn action_serde_roundtrip() {
        let action = Action::new("test.action", "Test Action", ActionCategory::Debug)
            .with_description("A test action")
            .with_shortcut("Ctrl+T");
        let json = serde_json::to_string(&action).unwrap();
        let decoded: Action = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, action.id);
        assert_eq!(decoded.label, action.label);
        assert_eq!(decoded.description, action.description);
        assert_eq!(decoded.shortcut, action.shortcut);
    }

    #[test]
    fn action_category_display() {
        assert_eq!(ActionCategory::Navigation.to_string(), "Navigation");
        assert_eq!(ActionCategory::Custom("Foo".to_string()).to_string(), "Foo");
    }
}
