//! Ops TUI screen implementations.
//!
//! Each screen implements the [`frankensearch_tui::Screen`] trait and
//! is registered in the [`frankensearch_tui::ScreenRegistry`].

pub mod fleet;

pub use fleet::FleetOverviewScreen;
