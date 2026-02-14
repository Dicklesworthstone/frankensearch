//! Color schemes, theme presets, and theming infrastructure.
//!
//! Provides a [`Theme`] type that defines colors for all TUI surfaces.
//! Ships with dark and light presets. Product crates can define custom
//! themes by implementing [`ColorScheme`].

use ratatui::style::Color;
use serde::{Deserialize, Serialize};

// ─── Theme Presets ──────────────────────────────────────────────────────────

/// Built-in theme presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThemePreset {
    /// Dark background theme (default).
    Dark,
    /// Light background theme.
    Light,
}

impl std::fmt::Display for ThemePreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dark => write!(f, "dark"),
            Self::Light => write!(f, "light"),
        }
    }
}

// ─── Color Scheme ───────────────────────────────────────────────────────────

/// Semantic color assignments for TUI surfaces.
///
/// Product crates can implement custom schemes by providing values
/// for all fields. Use [`Theme::dark`] or [`Theme::light`] for presets.
pub trait ColorScheme {
    /// Primary background color.
    fn bg(&self) -> Color;
    /// Primary foreground (text) color.
    fn fg(&self) -> Color;
    /// Status bar background.
    fn status_bar_bg(&self) -> Color;
    /// Status bar text.
    fn status_bar_fg(&self) -> Color;
    /// Selected/focused item highlight.
    fn highlight_bg(&self) -> Color;
    /// Highlight text.
    fn highlight_fg(&self) -> Color;
    /// Border color for panels and widgets.
    fn border(&self) -> Color;
    /// Muted/secondary text.
    fn muted(&self) -> Color;
    /// Error/alert color.
    fn error(&self) -> Color;
    /// Warning color.
    fn warning(&self) -> Color;
    /// Success/ok color.
    fn success(&self) -> Color;
    /// Info/accent color.
    fn info(&self) -> Color;
}

// ─── Theme ──────────────────────────────────────────────────────────────────

/// Concrete theme with all color assignments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Theme {
    pub preset: ThemePreset,
    pub bg: SerColor,
    pub fg: SerColor,
    pub status_bar_bg: SerColor,
    pub status_bar_fg: SerColor,
    pub highlight_bg: SerColor,
    pub highlight_fg: SerColor,
    pub border: SerColor,
    pub muted: SerColor,
    pub error: SerColor,
    pub warning: SerColor,
    pub success: SerColor,
    pub info: SerColor,
}

/// Serializable wrapper around `ratatui::style::Color`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl SerColor {
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Convert to ratatui `Color`.
    #[must_use]
    pub const fn to_ratatui(self) -> Color {
        Color::Rgb(self.r, self.g, self.b)
    }
}

impl Theme {
    /// Dark theme preset.
    #[must_use]
    pub const fn dark() -> Self {
        Self {
            preset: ThemePreset::Dark,
            bg: SerColor::new(0x1a, 0x1b, 0x26), // dark navy
            fg: SerColor::new(0xc0, 0xca, 0xf5), // soft white-blue
            status_bar_bg: SerColor::new(0x24, 0x28, 0x3b), // slightly lighter
            status_bar_fg: SerColor::new(0x7a, 0xa2, 0xf7), // bright blue
            highlight_bg: SerColor::new(0x33, 0x46, 0x7c), // muted blue
            highlight_fg: SerColor::new(0xff, 0xff, 0xff), // white
            border: SerColor::new(0x3b, 0x40, 0x61), // dark border
            muted: SerColor::new(0x56, 0x5f, 0x89), // gray
            error: SerColor::new(0xf7, 0x76, 0x8e), // red
            warning: SerColor::new(0xe0, 0xaf, 0x68), // orange
            success: SerColor::new(0x9e, 0xce, 0x6a), // green
            info: SerColor::new(0x7d, 0xcf, 0xff), // cyan
        }
    }

    /// Light theme preset.
    #[must_use]
    pub const fn light() -> Self {
        Self {
            preset: ThemePreset::Light,
            bg: SerColor::new(0xf5, 0xf5, 0xf5),
            fg: SerColor::new(0x34, 0x35, 0x4e),
            status_bar_bg: SerColor::new(0xe1, 0xe2, 0xe7),
            status_bar_fg: SerColor::new(0x34, 0x54, 0x8a),
            highlight_bg: SerColor::new(0xb6, 0xd4, 0xf0),
            highlight_fg: SerColor::new(0x00, 0x00, 0x00),
            border: SerColor::new(0xc8, 0xc8, 0xd0),
            muted: SerColor::new(0x8c, 0x8c, 0xa0),
            error: SerColor::new(0xc0, 0x3c, 0x3c),
            warning: SerColor::new(0x96, 0x5f, 0x00),
            success: SerColor::new(0x40, 0x7f, 0x00),
            info: SerColor::new(0x00, 0x6f, 0xaf),
        }
    }

    /// Create a theme from a preset.
    #[must_use]
    pub const fn from_preset(preset: ThemePreset) -> Self {
        match preset {
            ThemePreset::Dark => Self::dark(),
            ThemePreset::Light => Self::light(),
        }
    }
}

impl ColorScheme for Theme {
    fn bg(&self) -> Color {
        self.bg.to_ratatui()
    }
    fn fg(&self) -> Color {
        self.fg.to_ratatui()
    }
    fn status_bar_bg(&self) -> Color {
        self.status_bar_bg.to_ratatui()
    }
    fn status_bar_fg(&self) -> Color {
        self.status_bar_fg.to_ratatui()
    }
    fn highlight_bg(&self) -> Color {
        self.highlight_bg.to_ratatui()
    }
    fn highlight_fg(&self) -> Color {
        self.highlight_fg.to_ratatui()
    }
    fn border(&self) -> Color {
        self.border.to_ratatui()
    }
    fn muted(&self) -> Color {
        self.muted.to_ratatui()
    }
    fn error(&self) -> Color {
        self.error.to_ratatui()
    }
    fn warning(&self) -> Color {
        self.warning.to_ratatui()
    }
    fn success(&self) -> Color {
        self.success.to_ratatui()
    }
    fn info(&self) -> Color {
        self.info.to_ratatui()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dark_theme_colors() {
        let theme = Theme::dark();
        assert_eq!(theme.preset, ThemePreset::Dark);
        assert_ne!(theme.bg, theme.fg);
    }

    #[test]
    fn light_theme_colors() {
        let theme = Theme::light();
        assert_eq!(theme.preset, ThemePreset::Light);
    }

    #[test]
    fn theme_serde_roundtrip() {
        let theme = Theme::dark();
        let json = serde_json::to_string(&theme).unwrap();
        let decoded: Theme = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, theme);
    }

    #[test]
    fn preset_serde_roundtrip() {
        for preset in [ThemePreset::Dark, ThemePreset::Light] {
            let json = serde_json::to_string(&preset).unwrap();
            let decoded: ThemePreset = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, preset);
        }
    }

    #[test]
    fn from_preset_matches_direct() {
        assert_eq!(Theme::from_preset(ThemePreset::Dark), Theme::dark());
        assert_eq!(Theme::from_preset(ThemePreset::Light), Theme::light());
    }

    #[test]
    fn ser_color_to_ratatui() {
        let c = SerColor::new(0xff, 0x00, 0x80);
        assert_eq!(c.to_ratatui(), Color::Rgb(0xff, 0x00, 0x80));
    }

    #[test]
    fn color_scheme_trait() {
        let theme = Theme::dark();
        // Verify trait methods work
        assert_ne!(theme.bg(), theme.error());
        assert_ne!(theme.success(), theme.warning());
    }
}
