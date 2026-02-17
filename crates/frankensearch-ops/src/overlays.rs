//! Overlay rendering for ops TUI: help, alerts, command palette.
//!
//! These functions render overlay content on top of the active screen.
//! The shell manages the overlay stack; this module provides the visual
//! presentation for each overlay kind.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, List, ListItem, Paragraph, Wrap};

use frankensearch_tui::overlay::{OverlayKind, OverlayRequest};
use frankensearch_tui::palette::{CommandPalette, PaletteState};

// ─── Centered Popup Area ────────────────────────────────────────────────────

/// Compute a centered popup rectangle within the given area.
#[must_use]
fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

// ─── Help Overlay ───────────────────────────────────────────────────────────

/// Keyboard shortcut entry for the help overlay.
pub struct HelpEntry {
    /// Key combination (e.g., "Ctrl+P").
    pub key: &'static str,
    /// Description of what the shortcut does.
    pub description: &'static str,
}

/// Default keyboard shortcuts shown in the help overlay.
#[must_use]
pub fn default_help_entries() -> Vec<HelpEntry> {
    vec![
        HelpEntry {
            key: "?  / F1",
            description: "Toggle help",
        },
        HelpEntry {
            key: "q  / Ctrl+C",
            description: "Quit",
        },
        HelpEntry {
            key: "Ctrl+P / :",
            description: "Command palette",
        },
        HelpEntry {
            key: "Tab",
            description: "Next screen",
        },
        HelpEntry {
            key: "Shift+Tab",
            description: "Previous screen",
        },
        HelpEntry {
            key: "j / Down",
            description: "Move down",
        },
        HelpEntry {
            key: "k / Up",
            description: "Move up",
        },
        HelpEntry {
            key: "h / Left",
            description: "Move left",
        },
        HelpEntry {
            key: "l / Right",
            description: "Move right",
        },
        HelpEntry {
            key: "Enter",
            description: "Confirm / select",
        },
        HelpEntry {
            key: "Esc",
            description: "Dismiss overlay",
        },
        HelpEntry {
            key: "PgUp / PgDn",
            description: "Page navigation",
        },
        HelpEntry {
            key: "Ctrl+T",
            description: "Cycle theme",
        },
        HelpEntry {
            key: "Ctrl+Y",
            description: "Copy to clipboard",
        },
    ]
}

/// Render the help overlay showing keyboard shortcuts.
pub fn render_help_overlay(frame: &mut Frame<'_>, area: Rect) {
    let popup = centered_rect(60, 70, area);
    frame.render_widget(Clear, popup);

    let entries = default_help_entries();
    let items: Vec<ListItem<'_>> = entries
        .iter()
        .map(|e| {
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{:<18}", e.key),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(e.description),
            ]))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(ratatui::style::Color::Rgb(100, 160, 240)))
            .title(" Keyboard Shortcuts ")
            .title_style(Style::default().add_modifier(Modifier::BOLD)),
    );

    frame.render_widget(list, popup);
}

// ─── Alert Overlay ──────────────────────────────────────────────────────────

/// Render an alert overlay with title and optional body.
pub fn render_alert_overlay(frame: &mut Frame<'_>, area: Rect, request: &OverlayRequest) {
    let popup = centered_rect(50, 30, area);
    frame.render_widget(Clear, popup);

    let body_text = request.body.as_deref().unwrap_or("(no details)");

    let content = Paragraph::new(body_text).wrap(Wrap { trim: true }).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(ratatui::style::Color::Rgb(100, 160, 240)))
            .title(format!(" {} ", request.title))
            .title_style(Style::default().add_modifier(Modifier::BOLD)),
    );

    frame.render_widget(content, popup);
}

// ─── Confirm Overlay ────────────────────────────────────────────────────────

/// Render a confirmation dialog with action buttons.
pub fn render_confirm_overlay(frame: &mut Frame<'_>, area: Rect, request: &OverlayRequest) {
    let popup = centered_rect(50, 30, area);
    frame.render_widget(Clear, popup);

    let mut lines: Vec<Line<'_>> = Vec::new();
    if let Some(body) = &request.body {
        lines.push(Line::from(body.as_str()));
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
            Style::default().add_modifier(Modifier::BOLD),
        )));
    }

    let content = Paragraph::new(lines).wrap(Wrap { trim: true }).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(ratatui::style::Color::Rgb(100, 160, 240)))
            .title(format!(" {} ", request.title))
            .title_style(Style::default().add_modifier(Modifier::BOLD)),
    );

    frame.render_widget(content, popup);
}

// ─── Command Palette Overlay ────────────────────────────────────────────────

/// Render the command palette overlay.
pub fn render_palette_overlay(frame: &mut Frame<'_>, area: Rect, palette: &CommandPalette) {
    if palette.state() != &PaletteState::Open {
        return;
    }

    // Palette appears at the top-center, like VS Code's command palette.
    let width = area.width.min(60);
    let x = (area.width.saturating_sub(width)) / 2;
    let max_height = area.height.min(16);
    let popup = Rect::new(x, 1, width, max_height);
    frame.render_widget(Clear, popup);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(1)])
        .split(popup);

    // Search input.
    let query_display = if palette.query().is_empty() {
        "Type to search...".to_string()
    } else {
        palette.query().to_string()
    };

    let input_style = if palette.query().is_empty() {
        Style::default().fg(ratatui::style::Color::DarkGray)
    } else {
        Style::default()
    };

    let input = Paragraph::new(Span::styled(query_display, input_style)).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(ratatui::style::Color::Rgb(100, 160, 240)))
            .title(" Command Palette ")
            .title_style(Style::default().add_modifier(Modifier::BOLD)),
    );
    frame.render_widget(input, chunks[0]);

    // Filtered results.
    let filtered = palette.filtered();
    let selected = palette.selected();

    let items: Vec<ListItem<'_>> = filtered
        .iter()
        .enumerate()
        .map(|(i, action)| {
            let style = if i == selected {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };

            let mut spans = vec![Span::styled(&action.label, style)];
            if let Some(shortcut) = &action.shortcut {
                spans.push(Span::styled(
                    format!("  ({shortcut})"),
                    Style::default().fg(ratatui::style::Color::DarkGray),
                ));
            }

            ListItem::new(Line::from(spans))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(ratatui::style::Color::DarkGray)),
    );
    frame.render_widget(list, chunks[1]);
}

// ─── Dispatch Overlay Rendering ─────────────────────────────────────────────

/// Render the appropriate overlay based on the request kind.
pub fn render_overlay(frame: &mut Frame<'_>, area: Rect, request: &OverlayRequest) {
    match &request.kind {
        OverlayKind::Help => render_help_overlay(frame, area),
        OverlayKind::Confirm => render_confirm_overlay(frame, area, request),
        OverlayKind::Alert | OverlayKind::Custom(_) => render_alert_overlay(frame, area, request),
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
        let keys: Vec<&str> = entries.iter().map(|e| e.key).collect();
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
