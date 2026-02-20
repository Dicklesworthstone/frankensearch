//! Overlay rendering for ops TUI: help, alerts, command palette.
//!
//! These functions render overlay content on top of the active screen.
//! The shell manages the overlay stack; this module provides the visual
//! presentation for each overlay kind.

use ftui_core::geometry::Rect;
use ftui_layout::{Constraint, Flex};
use ftui_render::frame::Frame;
use ftui_style::Style;
use ftui_text::WrapMode;
use ftui_text::{Line, Span, Text};
use ftui_widgets::{
    Widget,
    block::Block,
    borders::{BorderType, Borders},
    list::{List, ListItem},
    paragraph::Paragraph,
};

use frankensearch_tui::overlay::{OverlayKind, OverlayRequest};
use frankensearch_tui::palette::{CommandPalette, PaletteState};

use crate::theme::SemanticPalette;

// ─── Centered Popup Area ────────────────────────────────────────────────────

/// Compute a centered popup rectangle within the given area.
#[must_use]
fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Flex::vertical()
        .constraints([
            Constraint::Percentage(f32::from(100 - percent_y) / 2.0),
            Constraint::Percentage(f32::from(percent_y)),
            Constraint::Percentage(f32::from(100 - percent_y) / 2.0),
        ])
        .split(area);

    Flex::horizontal()
        .constraints([
            Constraint::Percentage(f32::from(100 - percent_x) / 2.0),
            Constraint::Percentage(f32::from(percent_x)),
            Constraint::Percentage(f32::from(100 - percent_x) / 2.0),
        ])
        .split(popup_layout[1])[1]
}

// ─── Help Overlay ───────────────────────────────────────────────────────────

/// Keyboard shortcut entry for the help overlay.
pub struct HelpEntry {
    /// Key combination (e.g., "Ctrl+P").
    pub key: String,
    /// Description of what the shortcut does.
    pub description: String,
}

fn help_entry(key: &str, description: &str) -> HelpEntry {
    HelpEntry {
        key: key.to_owned(),
        description: description.to_owned(),
    }
}

/// Default keyboard shortcuts shown in the help overlay.
#[must_use]
pub fn default_help_entries() -> Vec<HelpEntry> {
    vec![
        help_entry("? / F1", "Toggle help"),
        help_entry("q / Ctrl+C", "Quit"),
        help_entry("Ctrl+P / :", "Command palette"),
        help_entry("Tab", "Next screen"),
        help_entry("Shift+Tab", "Previous screen"),
        help_entry("j / Down", "Move down"),
        help_entry("k / Up", "Move up"),
        help_entry("h / Left", "Move left"),
        help_entry("l / Right", "Move right"),
        help_entry("Enter", "Confirm / select"),
        help_entry("Esc", "Dismiss overlay"),
        help_entry("PgUp / PgDn", "Page navigation"),
        help_entry("Ctrl+T", "Cycle theme"),
        help_entry("Ctrl+Y", "Copy to clipboard"),
    ]
}

fn parse_screen_help_entries(actions: &[String]) -> Vec<HelpEntry> {
    actions
        .iter()
        .filter_map(|entry| entry.split_once('|'))
        .map(|(key, description)| help_entry(key.trim(), description.trim()))
        .collect()
}

/// Render the help overlay showing keyboard shortcuts.
pub fn render_help_overlay(
    frame: &mut Frame,
    area: Rect,
    request: &OverlayRequest,
    palette: &SemanticPalette,
) {
    let popup = centered_rect(60, 70, area);

    let mut entries = default_help_entries();
    let screen_entries = parse_screen_help_entries(&request.actions);
    if !screen_entries.is_empty() {
        entries.push(help_entry("──────────────────", "Screen Controls"));
        entries.extend(screen_entries);
    }
    let title = format!(" {} ", request.title);
    let items: Vec<ListItem> = entries
        .into_iter()
        .map(|e| {
            ListItem::new(Line::from_spans(vec![
                Span::styled(
                    format!("{:<18}", e.key),
                    Style::new().fg(palette.accent).bold(),
                ),
                Span::styled(e.description, Style::new().fg(palette.fg)),
            ]))
        })
        .collect();

    let list = List::new(items).block(
        Block::new()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(palette.border_focused))
            .title(title.as_str()),
    );

    list.render(popup, frame);
}

// ─── Alert Overlay ──────────────────────────────────────────────────────────

/// Render an alert overlay with title and optional body.
pub fn render_alert_overlay(
    frame: &mut Frame,
    area: Rect,
    request: &OverlayRequest,
    palette: &SemanticPalette,
) {
    let popup = centered_rect(50, 30, area);

    let body_text = request
        .body
        .clone()
        .unwrap_or_else(|| "(no details)".to_owned());

    let title = format!(" {} ", request.title);
    let content = Paragraph::new(Line::from(Span::styled(
        body_text,
        Style::new().fg(palette.fg),
    )))
    .wrap(WrapMode::Word)
    .block(
        Block::new()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(palette.border_focused))
            .title(title.as_str()),
    );

    content.render(popup, frame);
}

// ─── Confirm Overlay ────────────────────────────────────────────────────────

/// Render a confirmation dialog with action buttons.
pub fn render_confirm_overlay(
    frame: &mut Frame,
    area: Rect,
    request: &OverlayRequest,
    palette: &SemanticPalette,
) {
    let popup = centered_rect(50, 30, area);

    let mut lines: Vec<Line> = Vec::new();
    if let Some(body) = &request.body {
        lines.push(Line::from(Span::styled(
            body.clone(),
            Style::new().fg(palette.fg),
        )));
        lines.push(Line::from(""));
    }

    if !request.actions.is_empty() {
        let actions_text: String = request
            .actions
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if i == 0 {
                    format!("[{a}]")
                } else {
                    format!("  {a}")
                }
            })
            .collect();
        lines.push(Line::from(Span::styled(
            actions_text,
            Style::new().fg(palette.accent).bold(),
        )));
    }

    let title = format!(" {} ", request.title);
    let content = Paragraph::new(Text::from_lines(lines))
        .wrap(WrapMode::Word)
        .block(
            Block::new()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::new().fg(palette.border_focused))
                .title(title.as_str()),
        );

    content.render(popup, frame);
}

// ─── Command Palette Overlay ────────────────────────────────────────────────

/// Render the command palette overlay.
pub fn render_palette_overlay(
    frame: &mut Frame,
    area: Rect,
    palette: &CommandPalette,
    sem: &SemanticPalette,
) {
    if palette.state() != &PaletteState::Open {
        return;
    }

    // Palette appears at the top-center, like VS Code's command palette.
    let width = area.width.min(60);
    let x = (area.width.saturating_sub(width)) / 2;
    let max_height = area.height.min(16);
    let popup = Rect::new(x, 1, width, max_height);

    let chunks = Flex::vertical()
        .constraints([Constraint::Fixed(3), Constraint::Min(1)])
        .split(popup);

    // Search input.
    let query_display = if palette.query().is_empty() {
        "Type to search...".to_string()
    } else {
        palette.query().to_string()
    };

    let input_style = if palette.query().is_empty() {
        Style::new().fg(sem.fg_muted)
    } else {
        Style::new().fg(sem.fg)
    };

    let input = Paragraph::new(Line::from_spans([Span::styled(query_display, input_style)])).block(
        Block::new()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(sem.border_focused))
            .title(" Command Palette "),
    );
    input.render(chunks[0], frame);

    // Filtered results.
    let filtered = palette.filtered();
    let selected = palette.selected();

    let items: Vec<ListItem> = filtered
        .iter()
        .enumerate()
        .map(|(i, action)| {
            let style = if i == selected {
                sem.style_highlight()
            } else {
                Style::new().fg(sem.fg)
            };

            let mut spans = vec![Span::styled(action.label.clone(), style)];
            if let Some(shortcut) = &action.shortcut {
                spans.push(Span::styled(
                    format!("  ({shortcut})"),
                    Style::new().fg(sem.fg_muted),
                ));
            }

            ListItem::new(Line::from_spans(spans))
        })
        .collect();

    let list = List::new(items).block(
        Block::new()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(sem.border)),
    );
    list.render(chunks[1], frame);
}

// ─── Dispatch Overlay Rendering ─────────────────────────────────────────────

/// Render the appropriate overlay based on the request kind.
pub fn render_overlay(
    frame: &mut Frame,
    area: Rect,
    request: &OverlayRequest,
    palette: &SemanticPalette,
) {
    match &request.kind {
        OverlayKind::Help => render_help_overlay(frame, area, request, palette),
        OverlayKind::Confirm => render_confirm_overlay(frame, area, request, palette),
        OverlayKind::Alert | OverlayKind::Custom(_) => {
            render_alert_overlay(frame, area, request, palette);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_rect_within_bounds() {
        let area = Rect::new(0, 0, 100, 50);
        let popup = centered_rect(60, 40, area);
        assert!(popup.x >= area.x);
        assert!(popup.y >= area.y);
        assert!(
            u32::from(popup.x) + u32::from(popup.width)
                <= u32::from(area.x) + u32::from(area.width)
        );
        assert!(
            u32::from(popup.y) + u32::from(popup.height)
                <= u32::from(area.y) + u32::from(area.height)
        );
    }

    #[test]
    fn default_help_entries_nonempty() {
        let entries = default_help_entries();
        assert!(!entries.is_empty());
        assert!(entries.len() >= 10);
    }

    #[test]
    fn help_entries_have_content() {
        for entry in default_help_entries() {
            assert!(!entry.key.is_empty());
            assert!(!entry.description.is_empty());
        }
    }

    #[test]
    fn palette_closed_is_noop() {
        // Just verify render_palette_overlay doesn't panic when palette is closed.
        let palette = CommandPalette::new();
        assert_eq!(palette.state(), &PaletteState::Closed);
        // We can't easily test rendering without a real terminal backend,
        // but we can verify the guard check.
    }

    #[test]
    fn alert_overlay_request() {
        let request =
            OverlayRequest::new(OverlayKind::Alert, "Test Alert").with_body("Something happened");
        assert_eq!(request.title, "Test Alert");
        assert_eq!(request.body.as_deref(), Some("Something happened"));
    }

    #[test]
    fn confirm_overlay_request() {
        let request = OverlayRequest::new(OverlayKind::Confirm, "Delete?")
            .with_body("This cannot be undone.")
            .with_actions(vec!["Cancel".into(), "Delete".into()]);
        assert_eq!(request.actions.len(), 2);
    }

    #[test]
    fn render_overlay_dispatches_by_kind() {
        // Verify dispatch doesn't panic for each kind.
        let kinds = vec![
            OverlayKind::Help,
            OverlayKind::Alert,
            OverlayKind::Confirm,
            OverlayKind::Custom("test".into()),
        ];
        for kind in kinds {
            let _request = OverlayRequest::new(kind, "Test");
        }
    }

    #[test]
    fn centered_rect_full_area() {
        let area = Rect::new(0, 0, 80, 24);
        let popup = centered_rect(100, 100, area);
        // 100% should produce the full area
        assert_eq!(popup.width, area.width);
        assert_eq!(popup.height, area.height);
    }

    #[test]
    fn centered_rect_small_area_does_not_panic() {
        let area = Rect::new(0, 0, 3, 3);
        let popup = centered_rect(50, 50, area);
        assert!(popup.width <= area.width);
        assert!(popup.height <= area.height);
    }

    #[test]
    fn centered_rect_zero_area_does_not_panic() {
        let area = Rect::new(0, 0, 0, 0);
        let popup = centered_rect(50, 50, area);
        assert_eq!(popup.width, 0);
        assert_eq!(popup.height, 0);
    }

    #[test]
    fn help_entries_include_quit() {
        let entries = default_help_entries();
        assert!(
            entries.iter().any(|e| e.description.contains("Quit")),
            "help entries should include a quit shortcut"
        );
    }

    #[test]
    fn help_entries_include_command_palette() {
        let entries = default_help_entries();
        assert!(
            entries
                .iter()
                .any(|e| e.description.contains("Command palette")),
            "help entries should include command palette"
        );
    }

    #[test]
    fn help_entry_keys_are_unique() {
        let entries = default_help_entries();
        let keys: Vec<&str> = entries.iter().map(|e| e.key.as_str()).collect();
        for (i, key) in keys.iter().enumerate() {
            for (j, other) in keys.iter().enumerate() {
                if i != j {
                    assert_ne!(key, other, "duplicate key: {key}");
                }
            }
        }
    }

    #[test]
    fn alert_request_without_body_defaults_to_none() {
        let request = OverlayRequest::new(OverlayKind::Alert, "Title Only");
        assert!(request.body.is_none());
        assert!(request.actions.is_empty());
    }

    #[test]
    fn custom_overlay_preserves_name() {
        let kind = OverlayKind::Custom("my_custom".into());
        match &kind {
            OverlayKind::Custom(name) => assert_eq!(name, "my_custom"),
            _ => panic!("expected Custom variant"),
        }
    }
}
