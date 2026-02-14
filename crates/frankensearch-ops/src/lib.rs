//! Operations control-plane TUI for frankensearch fleet monitoring.
//!
//! This crate builds on the shared [`frankensearch_tui`] framework to provide
//! an operations console that discovers running frankensearch instances,
//! displays real-time metrics, and provides fleet-wide observability.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │  frankensearch-ops (this crate)                 │
//! │  ├─ app: OpsApp entry point and event loop      │
//! │  ├─ data_source: DataSource trait + mock impl   │
//! │  ├─ storage: Ops telemetry DB bootstrap          │
//! │  ├─ state: Shared AppState for async bridge     │
//! │  ├─ screens: Fleet, Search, Index, Resource     │
//! │  └─ category: Screen groupings for navigation   │
//! ├─────────────────────────────────────────────────┤
//! │  frankensearch-tui (shared framework)           │
//! │  Screen, ScreenRegistry, AppShell, Keymap, ...  │
//! ├─────────────────────────────────────────────────┤
//! │  ratatui + crossterm                            │
//! └─────────────────────────────────────────────────┘
//! ```

#![forbid(unsafe_code)]

pub mod app;
pub mod category;
pub mod data_source;
pub mod overlays;
pub mod preferences;
pub mod presets;
pub mod screens;
pub mod state;
pub mod storage;

// ─── Re-exports ─────────────────────────────────────────────────────────────

pub use app::OpsApp;
pub use category::ScreenCategory;
pub use data_source::{DataSource, MockDataSource};
pub use overlays::{render_overlay, render_palette_overlay};
pub use preferences::{ContrastMode, DisplayPreferences, FocusVisibility, MotionPreference};
pub use presets::{Density, ViewPreset, ViewState};
pub use state::{AppState, ControlPlaneHealth, ControlPlaneMetrics};
pub use storage::{
    OPS_SCHEMA_VERSION, OpsStorage, OpsStorageConfig, bootstrap as bootstrap_ops_storage,
    current_version as current_ops_schema_version,
};
