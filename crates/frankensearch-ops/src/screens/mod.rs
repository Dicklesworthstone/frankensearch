//! Ops TUI screen implementations.
//!
//! Each screen implements the [`frankensearch_tui::Screen`] trait and
//! is registered in the [`frankensearch_tui::ScreenRegistry`].

pub mod fleet;
pub mod live_stream;
pub mod timeline;

pub use fleet::FleetOverviewScreen;
pub use live_stream::LiveSearchStreamScreen;
pub use timeline::ActionTimelineScreen;
