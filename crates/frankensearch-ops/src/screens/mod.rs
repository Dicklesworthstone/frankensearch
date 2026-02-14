//! Ops TUI screen implementations.
//!
//! Each screen implements the [`frankensearch_tui::Screen`] trait and
//! is registered in the [`frankensearch_tui::ScreenRegistry`].

pub mod alerts_slo;
pub mod fleet;
pub mod historical_analytics;
pub mod index_resources;
pub mod live_stream;
pub mod project_detail;
pub mod timeline;

pub use alerts_slo::AlertsSloScreen;
pub use fleet::FleetOverviewScreen;
pub use historical_analytics::HistoricalAnalyticsScreen;
pub use index_resources::IndexResourceScreen;
pub use live_stream::LiveSearchStreamScreen;
pub use project_detail::ProjectDetailScreen;
pub use timeline::ActionTimelineScreen;
