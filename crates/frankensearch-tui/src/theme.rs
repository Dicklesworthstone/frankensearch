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
    /// Deep purple bg, neon cyan fg, hot pink accent.
    CyberpunkAurora,
    /// IntelliJ-style dark gray, soft blue fg, blue accent.
    Darcula,
    /// Nord-palette blue-gray bg, white fg, cool blue accent.
    NordicFrost,
    /// Warm white bg, dark text, amber accent.
    LumenLight,
}

impl ThemePreset {
    /// All presets in cycling order.
    pub const ALL: [Self; 6] = [
        Self::Dark,
        Self::Light,
        Self::CyberpunkAurora,
        Self::Darcula,
        Self::NordicFrost,
        Self::LumenLight,
    ];

    /// Whether this preset uses a light background.
    #[must_use]
    pub const fn is_light(self) -> bool {
        matches!(self, Self::Light | Self::LumenLight)
    }

    /// Advance to the next preset (wrapping).
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::Dark => Self::Light,
            Self::Light => Self::CyberpunkAurora,
            Self::CyberpunkAurora => Self::Darcula,
            Self::Darcula => Self::NordicFrost,
            Self::NordicFrost => Self::LumenLight,
            Self::LumenLight => Self::Dark,
        }
    }
}

impl std::fmt::Display for ThemePreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dark => write!(f, "dark"),
            Self::Light => write!(f, "light"),
            Self::CyberpunkAurora => write!(f, "cyberpunk_aurora"),
            Self::Darcula => write!(f, "darcula"),
            Self::NordicFrost => write!(f, "nordic_frost"),
            Self::LumenLight => write!(f, "lumen_light"),
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
    /// Accent color for active tabs and focused borders.
    fn accent(&self) -> Color {
        self.info()
    }
    /// Elevated surface background (cards, panels).
    fn surface(&self) -> Color {
        self.bg()
    }
    /// Dimmed/inactive surface background.
    fn surface_dim(&self) -> Color {
        self.bg()
    }
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
    pub accent: SerColor,
    pub surface: SerColor,
    pub surface_dim: SerColor,
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
            accent: SerColor::new(0x7d, 0xcf, 0xff), // cyan (same as info)
            surface: SerColor::new(0x24, 0x28, 0x3b), // elevated surface
            surface_dim: SerColor::new(0x1a, 0x1b, 0x26), // dimmed (same as bg)
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
            accent: SerColor::new(0x00, 0x6f, 0xaf), // blue (same as info)
            surface: SerColor::new(0xff, 0xff, 0xff), // white elevated
            surface_dim: SerColor::new(0xf5, 0xf5, 0xf5), // dimmed (same as bg)
        }
    }

    /// Cyberpunk Aurora theme preset.
    #[must_use]
    pub const fn cyberpunk_aurora() -> Self {
        Self {
            preset: ThemePreset::CyberpunkAurora,
            bg: SerColor::new(0x13, 0x05, 0x2a), // deep purple
            fg: SerColor::new(0x00, 0xf0, 0xd0), // neon cyan
            status_bar_bg: SerColor::new(0x1e, 0x0a, 0x3e), // slightly lighter purple
            status_bar_fg: SerColor::new(0xff, 0x6e, 0xb4), // hot pink
            highlight_bg: SerColor::new(0x3a, 0x15, 0x6a), // vivid purple
            highlight_fg: SerColor::new(0xff, 0xff, 0xff), // white
            border: SerColor::new(0x2d, 0x10, 0x52), // mid purple
            muted: SerColor::new(0x6a, 0x4c, 0x93), // lavender gray
            error: SerColor::new(0xff, 0x30, 0x60), // neon red
            warning: SerColor::new(0xff, 0xd7, 0x00), // gold
            success: SerColor::new(0x39, 0xff, 0x14), // neon green
            info: SerColor::new(0x00, 0xf0, 0xd0), // neon cyan
            accent: SerColor::new(0xff, 0x6e, 0xb4), // hot pink
            surface: SerColor::new(0x1e, 0x0a, 0x3e), // elevated purple
            surface_dim: SerColor::new(0x13, 0x05, 0x2a), // dimmed (same as bg)
        }
    }

    /// Darcula theme preset (IntelliJ-inspired).
    #[must_use]
    pub const fn darcula() -> Self {
        Self {
            preset: ThemePreset::Darcula,
            bg: SerColor::new(0x2b, 0x2b, 0x2b), // dark gray
            fg: SerColor::new(0xa9, 0xb7, 0xc6), // soft blue-gray
            status_bar_bg: SerColor::new(0x3c, 0x3f, 0x41), // lighter gray
            status_bar_fg: SerColor::new(0x68, 0x97, 0xbb), // soft blue
            highlight_bg: SerColor::new(0x21, 0x4a, 0x83), // selection blue
            highlight_fg: SerColor::new(0xff, 0xff, 0xff), // white
            border: SerColor::new(0x4b, 0x4b, 0x4b), // mid gray
            muted: SerColor::new(0x78, 0x78, 0x78), // dim gray
            error: SerColor::new(0xbc, 0x35, 0x51), // muted red
            warning: SerColor::new(0xbb, 0xb5, 0x29), // olive yellow
            success: SerColor::new(0x6a, 0x87, 0x59), // forest green
            info: SerColor::new(0x68, 0x97, 0xbb), // soft blue
            accent: SerColor::new(0x68, 0x97, 0xbb), // soft blue
            surface: SerColor::new(0x31, 0x31, 0x35), // elevated dark
            surface_dim: SerColor::new(0x2b, 0x2b, 0x2b), // dimmed (same as bg)
        }
    }

    /// Nordic Frost theme preset (Nord-inspired).
    #[must_use]
    pub const fn nordic_frost() -> Self {
        Self {
            preset: ThemePreset::NordicFrost,
            bg: SerColor::new(0x2e, 0x34, 0x40), // nord0 polar night
            fg: SerColor::new(0xec, 0xef, 0xf4), // nord6 snow storm
            status_bar_bg: SerColor::new(0x3b, 0x42, 0x52), // nord1
            status_bar_fg: SerColor::new(0x88, 0xc0, 0xd0), // nord8 frost
            highlight_bg: SerColor::new(0x43, 0x4c, 0x5e), // nord2
            highlight_fg: SerColor::new(0xec, 0xef, 0xf4), // nord6
            border: SerColor::new(0x4c, 0x56, 0x6a), // nord3
            muted: SerColor::new(0x61, 0x6e, 0x88), // dimmed nord3
            error: SerColor::new(0xbf, 0x61, 0x6a), // nord11 red
            warning: SerColor::new(0xeb, 0xcb, 0x8b), // nord13 yellow
            success: SerColor::new(0xa3, 0xbe, 0x8c), // nord14 green
            info: SerColor::new(0x88, 0xc0, 0xd0), // nord8 frost
            accent: SerColor::new(0x5e, 0x81, 0xac), // nord10 cool blue
            surface: SerColor::new(0x3b, 0x42, 0x52), // nord1 elevated
            surface_dim: SerColor::new(0x2e, 0x34, 0x40), // dimmed (same as bg)
        }
    }

    /// Lumen Light theme preset.
    #[must_use]
    pub const fn lumen_light() -> Self {
        Self {
            preset: ThemePreset::LumenLight,
            bg: SerColor::new(0xfd, 0xf6, 0xe3), // warm cream
            fg: SerColor::new(0x3b, 0x38, 0x30), // dark brown
            status_bar_bg: SerColor::new(0xee, 0xe8, 0xd5), // muted cream
            status_bar_fg: SerColor::new(0xcb, 0x76, 0x16), // amber
            highlight_bg: SerColor::new(0xf5, 0xdc, 0xa0), // soft amber highlight
            highlight_fg: SerColor::new(0x2a, 0x27, 0x20), // dark
            border: SerColor::new(0xd6, 0xd0, 0xc0), // warm gray
            muted: SerColor::new(0x93, 0xa1, 0xa1), // cool gray
            error: SerColor::new(0xdc, 0x32, 0x2f), // red
            warning: SerColor::new(0xcb, 0x76, 0x16), // amber
            success: SerColor::new(0x85, 0x99, 0x00), // olive green
            info: SerColor::new(0x26, 0x8b, 0xd2), // blue
            accent: SerColor::new(0xcb, 0x76, 0x16), // amber
            surface: SerColor::new(0xff, 0xff, 0xf0), // warm white
            surface_dim: SerColor::new(0xfd, 0xf6, 0xe3), // dimmed (same as bg)
        }
    }

    /// Create a theme from a preset.
    #[must_use]
    pub const fn from_preset(preset: ThemePreset) -> Self {
        match preset {
            ThemePreset::Dark => Self::dark(),
            ThemePreset::Light => Self::light(),
            ThemePreset::CyberpunkAurora => Self::cyberpunk_aurora(),
            ThemePreset::Darcula => Self::darcula(),
            ThemePreset::NordicFrost => Self::nordic_frost(),
            ThemePreset::LumenLight => Self::lumen_light(),
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
    fn accent(&self) -> Color {
        self.accent.to_ratatui()
    }
    fn surface(&self) -> Color {
        self.surface.to_ratatui()
    }
    fn surface_dim(&self) -> Color {
        self.surface_dim.to_ratatui()
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
        for preset in ThemePreset::ALL {
            let json = serde_json::to_string(&preset).unwrap();
            let decoded: ThemePreset = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, preset);
        }
    }

    #[test]
    fn from_preset_matches_direct() {
        assert_eq!(Theme::from_preset(ThemePreset::Dark), Theme::dark());
        assert_eq!(Theme::from_preset(ThemePreset::Light), Theme::light());
        assert_eq!(
            Theme::from_preset(ThemePreset::CyberpunkAurora),
            Theme::cyberpunk_aurora()
        );
        assert_eq!(Theme::from_preset(ThemePreset::Darcula), Theme::darcula());
        assert_eq!(
            Theme::from_preset(ThemePreset::NordicFrost),
            Theme::nordic_frost()
        );
        assert_eq!(
            Theme::from_preset(ThemePreset::LumenLight),
            Theme::lumen_light()
        );
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

    #[test]
    fn preset_cycling_wraps() {
        let mut preset = ThemePreset::Dark;
        for _ in 0..ThemePreset::ALL.len() {
            preset = preset.next();
        }
        assert_eq!(
            preset,
            ThemePreset::Dark,
            "cycling should wrap back to start"
        );
    }

    #[test]
    fn preset_cycling_visits_all() {
        let mut visited = std::collections::HashSet::new();
        let mut preset = ThemePreset::Dark;
        for _ in 0..ThemePreset::ALL.len() {
            visited.insert(preset);
            preset = preset.next();
        }
        assert_eq!(visited.len(), ThemePreset::ALL.len());
    }

    #[test]
    fn accent_colors_distinct_per_theme() {
        let accents: Vec<SerColor> = ThemePreset::ALL
            .iter()
            .map(|p| Theme::from_preset(*p).accent)
            .collect();
        // At least 4 distinct accents out of 6 themes
        let unique: std::collections::HashSet<_> =
            accents.iter().map(|c| (c.r, c.g, c.b)).collect();
        assert!(
            unique.len() >= 4,
            "expected at least 4 distinct accents, got {}",
            unique.len()
        );
    }

    #[test]
    fn all_presets_have_all_const() {
        assert_eq!(ThemePreset::ALL.len(), 6);
    }
}
