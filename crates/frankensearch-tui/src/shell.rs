//! App shell: status bar, breadcrumbs, screen lifecycle.
//!
//! The [`AppShell`] owns the [`ScreenRegistry`], manages navigation between
//! screens, renders the chrome (status bar, breadcrumbs), and dispatches
//! input events to the active screen.

use std::cell::Cell;
use std::hash::{DefaultHasher, Hash, Hasher};

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Tabs};
use serde::{Deserialize, Serialize};

use crate::frame::{CachedLayout, CachedTabState};
use crate::input::{InputEvent, KeyAction, Keymap};
use crate::overlay::OverlayManager;
use crate::palette::{CommandPalette, PaletteState};
use crate::screen::{ScreenAction, ScreenContext, ScreenId, ScreenRegistry};
use crate::theme::Theme;

// ─── Shell Config ────────────────────────────────────────────────────────────

/// Configuration for the app shell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellConfig {
    /// Application title shown in the status bar.
    pub title: String,
    /// Theme preset to use.
    pub theme: Theme,
    /// Whether to show the status bar.
    pub show_status_bar: bool,
    /// Whether to show breadcrumbs (tab bar).
    pub show_breadcrumbs: bool,
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            title: "frankensearch".to_string(),
            theme: Theme::dark(),
            show_status_bar: true,
            show_breadcrumbs: true,
        }
    }
}

// ─── Status Line ─────────────────────────────────────────────────────────────

/// Status line content rendered at the bottom of the shell.
#[derive(Debug, Clone, Default)]
pub struct StatusLine {
    /// Left-aligned status text.
    pub left: String,
    /// Center status text.
    pub center: String,
    /// Right-aligned status text.
    pub right: String,
}

impl StatusLine {
    /// Create a new status line.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the left-aligned text.
    #[must_use]
    pub fn with_left(mut self, text: impl Into<String>) -> Self {
        self.left = text.into();
        self
    }

    /// Set the center text.
    #[must_use]
    pub fn with_center(mut self, text: impl Into<String>) -> Self {
        self.center = text.into();
        self
    }

    /// Set the right-aligned text.
    #[must_use]
    pub fn with_right(mut self, text: impl Into<String>) -> Self {
        self.right = text.into();
        self
    }
}

// ─── App Shell ───────────────────────────────────────────────────────────────

/// The main app shell that manages screens, chrome, and input dispatch.
pub struct AppShell {
    /// Shell configuration.
    pub config: ShellConfig,
    /// Screen registry.
    pub registry: ScreenRegistry,
    /// Currently active screen ID.
    pub active_screen: Option<ScreenId>,
    /// Keymap for input resolution.
    pub keymap: Keymap,
    /// Overlay manager.
    pub overlays: OverlayManager,
    /// Command palette.
    pub palette: CommandPalette,
    /// Status line content.
    pub status_line: StatusLine,
    /// Whether the app should quit.
    pub should_quit: bool,
    /// Last confirmed palette action ID (reset each `handle_input` call).
    last_palette_action: Option<String>,
    /// Terminal area captured from the most recent render pass.
    last_render_area: Cell<Rect>,
    /// Cached layout to avoid recomputing splits every frame.
    cached_layout: CachedLayout,
    /// Cached tab titles and selected index.
    cached_tabs: CachedTabState,
}

impl AppShell {
    /// Create a new app shell with the given config.
    #[must_use]
    pub fn new(config: ShellConfig) -> Self {
        Self {
            config,
            registry: ScreenRegistry::new(),
            active_screen: None,
            keymap: Keymap::default_bindings(),
            overlays: OverlayManager::new(),
            palette: CommandPalette::new(),
            status_line: StatusLine::new(),
            should_quit: false,
            last_palette_action: None,
            last_render_area: Cell::new(Rect::new(0, 0, 0, 0)),
            cached_layout: CachedLayout::new(),
            cached_tabs: CachedTabState::new(),
        }
    }

    /// Navigate to a screen by ID.
    pub fn navigate_to(&mut self, id: &ScreenId) {
        if self.registry.get(id).is_some() {
            // Blur the old screen.
            if let Some(old_id) = &self.active_screen {
                let old_id = old_id.clone();
                if let Some(screen) = self.registry.get_mut(&old_id) {
                    screen.on_blur();
                }
            }
            // Focus the new screen.
            self.active_screen = Some(id.clone());
            if let Some(screen) = self.registry.get_mut(id) {
                screen.on_focus();
            }
            // Invalidate tab cache since the active screen changed.
            self.cached_tabs.invalidate();
        }
    }

    /// Navigate to the next screen in tab order.
    pub fn next_screen(&mut self) {
        if let Some(current) = &self.active_screen {
            if let Some(next) = self.registry.next_screen(current).cloned() {
                self.navigate_to(&next);
            }
        }
    }

    /// Navigate to the previous screen in tab order.
    pub fn prev_screen(&mut self) {
        if let Some(current) = &self.active_screen {
            if let Some(prev) = self.registry.prev_screen(current).cloned() {
                self.navigate_to(&prev);
            }
        }
    }

    /// Build the screen context for the current state.
    #[must_use]
    pub fn screen_context(&self, area: Rect) -> ScreenContext {
        ScreenContext {
            active_screen: self
                .active_screen
                .clone()
                .unwrap_or_else(|| ScreenId::new("")),
            terminal_width: area.width,
            terminal_height: area.height,
            focused: true,
        }
    }

    /// Handle an input event. Returns `true` if the app should quit.
    ///
    /// Returns the confirmed palette action ID if the user selected a command.
    /// Callers should check `last_palette_action()` after calling this.
    #[allow(clippy::too_many_lines)]
    pub fn handle_input(&mut self, event: &InputEvent) -> bool {
        self.last_palette_action = None;

        if let InputEvent::Resize(width, height) = event {
            self.last_render_area.set(Rect::new(0, 0, *width, *height));
        }

        if self.last_render_area.get().width == 0 || self.last_render_area.get().height == 0 {
            if let Ok((width, height)) = crossterm::terminal::size() {
                self.last_render_area.set(Rect::new(0, 0, width, height));
            } else {
                // Fallback keeps context sane in non-interactive test harnesses.
                self.last_render_area.set(Rect::new(0, 0, 80, 24));
            }
        }

        // If the command palette is open, route input there first.
        if self.palette.state() == &PaletteState::Open {
            if let InputEvent::Key(key, mods) = event {
                if let Some(action) = self.keymap.resolve(*key, *mods) {
                    match action {
                        KeyAction::TogglePalette | KeyAction::Dismiss => {
                            self.palette.close();
                            return false;
                        }
                        KeyAction::Up => {
                            self.palette.select_prev();
                            return false;
                        }
                        KeyAction::Down => {
                            self.palette.select_next();
                            return false;
                        }
                        KeyAction::Confirm => {
                            if let Some(action_id) = self.palette.confirm() {
                                self.last_palette_action = Some(action_id);
                            }
                            self.palette.close();
                            return false;
                        }
                        _ => {}
                    }
                }
                // Handle text input for palette search.
                match key {
                    crossterm::event::KeyCode::Char(ch)
                        if !mods.intersects(
                            crossterm::event::KeyModifiers::CONTROL
                                | crossterm::event::KeyModifiers::ALT,
                        ) =>
                    {
                        self.palette.push_char(*ch);
                        return false;
                    }
                    crossterm::event::KeyCode::Backspace => {
                        self.palette.pop_char();
                        return false;
                    }
                    _ => {}
                }
            }
            return false;
        }

        // If an overlay is active, let it handle first.
        if self.overlays.has_active() {
            if let InputEvent::Key(key, mods) = event {
                if let Some(action) = self.keymap.resolve(*key, *mods) {
                    if action == &KeyAction::Dismiss {
                        self.overlays.dismiss();
                        return false;
                    }
                }
            }
            return false;
        }

        // Resolve key actions.
        if let InputEvent::Key(key, mods) = event {
            if let Some(action) = self.keymap.resolve(*key, *mods).cloned() {
                match action {
                    KeyAction::Quit => {
                        self.should_quit = true;
                        return true;
                    }
                    KeyAction::NextScreen => {
                        self.next_screen();
                        return false;
                    }
                    KeyAction::PrevScreen => {
                        self.prev_screen();
                        return false;
                    }
                    KeyAction::ToggleHelp => {
                        if self
                            .overlays
                            .top()
                            .is_some_and(|o| o.kind == crate::overlay::OverlayKind::Help)
                        {
                            self.overlays.dismiss();
                        } else {
                            self.overlays.push(crate::overlay::OverlayRequest::new(
                                crate::overlay::OverlayKind::Help,
                                "Keyboard Shortcuts",
                            ));
                        }
                        return false;
                    }
                    KeyAction::TogglePalette => {
                        self.palette.toggle();
                        return false;
                    }
                    _ => {}
                }
            }
        }

        // Forward to active screen.
        if let Some(screen_id) = &self.active_screen {
            let screen_id = screen_id.clone();
            let ctx = self.screen_context(self.last_render_area.get());
            if let Some(screen) = self.registry.get_mut(&screen_id) {
                match screen.handle_input(event, &ctx) {
                    ScreenAction::Quit => {
                        self.should_quit = true;
                        return true;
                    }
                    ScreenAction::Navigate(target) => {
                        self.navigate_to(&target);
                    }
                    ScreenAction::OpenOverlay(name) => {
                        self.overlays.push(crate::overlay::OverlayRequest::new(
                            crate::overlay::OverlayKind::Custom(name.clone()),
                            name,
                        ));
                    }
                    ScreenAction::Consumed | ScreenAction::Ignored => {}
                }
            }
        }

        false
    }

    /// Get the last palette action confirmed by the user.
    ///
    /// Returns `Some(action_id)` if the user selected an action from the
    /// command palette during the last `handle_input()` call.
    #[must_use]
    pub fn last_palette_action(&self) -> Option<&str> {
        self.last_palette_action.as_deref()
    }

    /// Render the shell chrome and active screen.
    ///
    /// Uses cached layout and tab state to avoid redundant allocations
    /// when the terminal dimensions and screen configuration haven't changed.
    #[allow(clippy::too_many_lines)]
    pub fn render(&mut self, frame: &mut Frame<'_>) {
        let area = frame.area();
        self.last_render_area.set(area);
        let ctx = self.screen_context(area);

        // Cached layout avoids recomputing splits when area hasn't changed.
        // Copy the Rect values directly (Rect is Copy, 8 bytes) instead of
        // allocating a Vec clone every frame.
        let show_bc = self.config.show_breadcrumbs;
        let show_sb = self.config.show_status_bar;
        let num_screens = self.registry.len();
        let layout_chunks = self
            .cached_layout
            .get_or_compute(area, show_bc, show_sb, num_screens);
        let bc_area = if show_bc && num_screens > 1 {
            Some(layout_chunks[0])
        } else {
            None
        };
        let content_idx = usize::from(bc_area.is_some());
        let content_area = layout_chunks[content_idx];
        let status_area = if show_sb {
            Some(layout_chunks[content_idx + 1])
        } else {
            None
        };

        // Breadcrumbs / tabs (using cached titles and selected index).
        if let Some(bc_rect) = bc_area {
            let screen_ids = self.registry.screen_ids();
            let mut id_hasher = DefaultHasher::new();
            let mut title_hasher = DefaultHasher::new();
            for id in screen_ids {
                id.0.hash(&mut id_hasher);
                id.0.hash(&mut title_hasher);
                if let Some(screen) = self.registry.get(id) {
                    screen.title().hash(&mut title_hasher);
                }
            }
            let screen_ids_hash = id_hasher.finish();
            let title_signature = title_hasher.finish();
            let active_str = self.active_screen.as_ref().map(|id| id.0.as_str());

            if !self
                .cached_tabs
                .is_valid(screen_ids_hash, title_signature, active_str)
            {
                let titles: Vec<String> = screen_ids
                    .iter()
                    .map(|id| {
                        self.registry
                            .get(id)
                            .map_or_else(|| id.0.clone(), |s| s.title().to_string())
                    })
                    .collect();

                let selected = self
                    .active_screen
                    .as_ref()
                    .and_then(|active| screen_ids.iter().position(|id| id == active))
                    .unwrap_or(0);

                self.cached_tabs.update(
                    titles,
                    selected,
                    screen_ids_hash,
                    title_signature,
                    active_str,
                );
            }

            let tab_titles: Vec<Line<'_>> = self
                .cached_tabs
                .titles
                .iter()
                .map(|t| Line::from(t.as_str()))
                .collect();

            let tabs = Tabs::new(tab_titles)
                .select(self.cached_tabs.selected)
                .highlight_style(
                    Style::default()
                        .fg(self.config.theme.highlight_fg.to_ratatui())
                        .bg(self.config.theme.highlight_bg.to_ratatui())
                        .add_modifier(Modifier::BOLD),
                )
                .style(
                    Style::default()
                        .fg(self.config.theme.muted.to_ratatui())
                        .bg(self.config.theme.bg.to_ratatui()),
                );

            frame.render_widget(tabs, bc_rect);
        }

        // Main content area.
        if let Some(screen_id) = &self.active_screen {
            if let Some(screen) = self.registry.get(screen_id) {
                screen.render(frame, &ctx);
            }
        } else {
            // No screen active — render placeholder.
            let block = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(self.config.theme.border.to_ratatui()))
                .style(
                    Style::default()
                        .bg(self.config.theme.bg.to_ratatui())
                        .fg(self.config.theme.fg.to_ratatui()),
                );
            let placeholder = Paragraph::new("No screens registered").block(block);
            frame.render_widget(placeholder, content_area);
        }

        // Status bar.
        if let Some(sb_rect) = status_area {
            let status_text = if self.status_line.center.is_empty() {
                format!(" {} ", self.config.title)
            } else {
                format!(" {} │ {} ", self.config.title, self.status_line.center)
            };

            let status_spans = vec![
                Span::styled(
                    &self.status_line.left,
                    Style::default().fg(self.config.theme.status_bar_fg.to_ratatui()),
                ),
                Span::styled(
                    status_text,
                    Style::default()
                        .fg(self.config.theme.status_bar_fg.to_ratatui())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    &self.status_line.right,
                    Style::default().fg(self.config.theme.status_bar_fg.to_ratatui()),
                ),
            ];

            let status = Paragraph::new(Line::from(status_spans))
                .style(Style::default().bg(self.config.theme.status_bar_bg.to_ratatui()));

            frame.render_widget(status, sb_rect);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::{Arc, Mutex};

    use ratatui::Frame;

    use crate::screen::{Screen, ScreenAction};

    use super::*;

    #[test]
    fn shell_config_default() {
        let config = ShellConfig::default();
        assert_eq!(config.title, "frankensearch");
        assert!(config.show_status_bar);
        assert!(config.show_breadcrumbs);
    }

    #[test]
    fn status_line_builder() {
        let status = StatusLine::new()
            .with_left("left")
            .with_center("center")
            .with_right("right");
        assert_eq!(status.left, "left");
        assert_eq!(status.center, "center");
        assert_eq!(status.right, "right");
    }

    #[test]
    fn shell_creation() {
        let shell = AppShell::new(ShellConfig::default());
        assert!(!shell.should_quit);
        assert!(shell.active_screen.is_none());
        assert!(shell.registry.is_empty());
    }

    #[test]
    fn shell_config_serde_roundtrip() {
        let config = ShellConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let decoded: ShellConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.title, config.title);
    }

    #[test]
    fn shell_quit_handling() {
        let mut shell = AppShell::new(ShellConfig::default());
        let event = InputEvent::Key(
            crossterm::event::KeyCode::Char('q'),
            crossterm::event::KeyModifiers::NONE,
        );
        let quit = shell.handle_input(&event);
        assert!(quit);
        assert!(shell.should_quit);
    }

    struct CaptureContextScreen {
        id: ScreenId,
        captured: Arc<Mutex<Option<(u16, u16)>>>,
    }

    impl CaptureContextScreen {
        fn new(id: &str, captured: Arc<Mutex<Option<(u16, u16)>>>) -> Self {
            Self {
                id: ScreenId::new(id),
                captured,
            }
        }
    }

    impl Screen for CaptureContextScreen {
        fn id(&self) -> &ScreenId {
            &self.id
        }

        fn title(&self) -> &'static str {
            "capture"
        }

        fn render(&self, _frame: &mut Frame<'_>, _ctx: &ScreenContext) {}

        fn handle_input(&mut self, _event: &InputEvent, ctx: &ScreenContext) -> ScreenAction {
            *self.captured.lock().expect("capture lock") =
                Some((ctx.terminal_width, ctx.terminal_height));
            ScreenAction::Consumed
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[test]
    fn handle_input_uses_last_render_area_for_context() {
        let mut shell = AppShell::new(ShellConfig::default());
        let captured = Arc::new(Mutex::new(None));
        let screen_id = ScreenId::new("capture");
        shell.registry.register(Box::new(CaptureContextScreen::new(
            "capture",
            captured.clone(),
        )));
        shell.navigate_to(&screen_id);

        shell.last_render_area.set(Rect::new(0, 0, 132, 47));
        let event = InputEvent::Key(
            crossterm::event::KeyCode::Char('x'),
            crossterm::event::KeyModifiers::NONE,
        );
        let _ = shell.handle_input(&event);

        let seen = captured
            .lock()
            .expect("capture lock")
            .expect("context captured");
        assert_eq!(seen, (132, 47));
    }

    #[test]
    fn palette_toggle_shortcut_closes_palette_when_open() {
        let mut shell = AppShell::new(ShellConfig::default());
        let toggle = InputEvent::Key(
            crossterm::event::KeyCode::Char('p'),
            crossterm::event::KeyModifiers::CONTROL,
        );

        let _ = shell.handle_input(&toggle);
        assert_eq!(shell.palette.state(), &PaletteState::Open);

        let _ = shell.handle_input(&toggle);
        assert_eq!(shell.palette.state(), &PaletteState::Closed);
    }

    #[test]
    fn palette_accepts_shift_modified_characters() {
        let mut shell = AppShell::new(ShellConfig::default());
        let open = InputEvent::Key(
            crossterm::event::KeyCode::Char('p'),
            crossterm::event::KeyModifiers::CONTROL,
        );
        let _ = shell.handle_input(&open);

        let shifted = InputEvent::Key(
            crossterm::event::KeyCode::Char('A'),
            crossterm::event::KeyModifiers::SHIFT,
        );
        let _ = shell.handle_input(&shifted);

        assert_eq!(shell.palette.query(), "A");
    }

    #[test]
    fn resize_event_refreshes_context_dimensions() {
        let mut shell = AppShell::new(ShellConfig::default());
        let captured = Arc::new(Mutex::new(None));
        let screen_id = ScreenId::new("capture");
        shell.registry.register(Box::new(CaptureContextScreen::new(
            "capture",
            captured.clone(),
        )));
        shell.navigate_to(&screen_id);

        let resize = InputEvent::Resize(111, 37);
        let _ = shell.handle_input(&resize);

        let key = InputEvent::Key(
            crossterm::event::KeyCode::Char('x'),
            crossterm::event::KeyModifiers::NONE,
        );
        let _ = shell.handle_input(&key);

        let seen = captured
            .lock()
            .expect("capture lock")
            .expect("context captured");
        assert_eq!(seen, (111, 37));
    }
}
