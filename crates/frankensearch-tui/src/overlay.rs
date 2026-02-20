//! Overlay system: help panels, confirmation dialogs, alerts.
//!
//! Overlays render on top of the active screen. The [`OverlayManager`]
//! maintains a stack so multiple overlays can nest (e.g. help → confirm).
//! The shell dismisses the topmost overlay on `Esc`.

use serde::{Deserialize, Serialize};

// ─── Overlay Kind ────────────────────────────────────────────────────────────

/// The kind of overlay to display.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverlayKind {
    /// Help / keyboard shortcuts overlay.
    Help,
    /// Confirmation dialog (e.g. "Are you sure?").
    Confirm,
    /// Alert / notification.
    Alert,
    /// Product-specific overlay.
    Custom(String),
}

impl std::fmt::Display for OverlayKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Help => write!(f, "Help"),
            Self::Confirm => write!(f, "Confirm"),
            Self::Alert => write!(f, "Alert"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

// ─── Overlay Request ─────────────────────────────────────────────────────────

/// A request to show an overlay.
#[derive(Debug, Clone)]
pub struct OverlayRequest {
    /// The kind of overlay.
    pub kind: OverlayKind,
    /// Title for the overlay.
    pub title: String,
    /// Optional body content.
    pub body: Option<String>,
    /// Optional action labels for confirmation dialogs.
    pub actions: Vec<String>,
}

impl OverlayRequest {
    /// Create a new overlay request.
    #[must_use]
    pub fn new(kind: OverlayKind, title: impl Into<String>) -> Self {
        Self {
            kind,
            title: title.into(),
            body: None,
            actions: Vec::new(),
        }
    }

    /// Set the body content.
    #[must_use]
    pub fn with_body(mut self, body: impl Into<String>) -> Self {
        self.body = Some(body.into());
        self
    }

    /// Add action labels for confirm dialogs.
    #[must_use]
    pub fn with_actions(mut self, actions: Vec<String>) -> Self {
        self.actions = actions;
        self
    }
}

// ─── Overlay Manager ─────────────────────────────────────────────────────────

/// Manages a stack of overlay requests.
///
/// The topmost overlay receives input first. `Esc` dismisses the top.
pub struct OverlayManager {
    stack: Vec<OverlayRequest>,
}

impl OverlayManager {
    /// Create an empty overlay manager.
    #[must_use]
    pub const fn new() -> Self {
        Self { stack: Vec::new() }
    }

    /// Push an overlay onto the stack.
    pub fn push(&mut self, request: OverlayRequest) {
        self.stack.push(request);
    }

    /// Dismiss (pop) the topmost overlay.
    pub fn dismiss(&mut self) -> Option<OverlayRequest> {
        self.stack.pop()
    }

    /// Dismiss all overlays.
    pub fn dismiss_all(&mut self) {
        self.stack.clear();
    }

    /// Get a reference to the topmost overlay.
    #[must_use]
    pub fn top(&self) -> Option<&OverlayRequest> {
        self.stack.last()
    }

    /// Whether any overlay is active.
    #[must_use]
    pub const fn has_active(&self) -> bool {
        !self.stack.is_empty()
    }

    /// Number of active overlays.
    #[must_use]
    pub const fn depth(&self) -> usize {
        self.stack.len()
    }
}

impl Default for OverlayManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_manager_empty() {
        let mgr = OverlayManager::new();
        assert!(!mgr.has_active());
        assert_eq!(mgr.depth(), 0);
        assert!(mgr.top().is_none());
    }

    #[test]
    fn overlay_push_and_dismiss() {
        let mut mgr = OverlayManager::new();
        mgr.push(OverlayRequest::new(OverlayKind::Help, "Help"));
        assert!(mgr.has_active());
        assert_eq!(mgr.depth(), 1);
        assert_eq!(mgr.top().unwrap().title, "Help");

        let dismissed = mgr.dismiss().unwrap();
        assert_eq!(dismissed.title, "Help");
        assert!(!mgr.has_active());
    }

    #[test]
    fn overlay_stack_ordering() {
        let mut mgr = OverlayManager::new();
        mgr.push(OverlayRequest::new(OverlayKind::Help, "Help"));
        mgr.push(OverlayRequest::new(OverlayKind::Confirm, "Confirm"));
        mgr.push(OverlayRequest::new(OverlayKind::Alert, "Alert"));

        assert_eq!(mgr.depth(), 3);
        assert_eq!(mgr.top().unwrap().title, "Alert");

        mgr.dismiss();
        assert_eq!(mgr.top().unwrap().title, "Confirm");

        mgr.dismiss();
        assert_eq!(mgr.top().unwrap().title, "Help");
    }

    #[test]
    fn overlay_dismiss_all() {
        let mut mgr = OverlayManager::new();
        mgr.push(OverlayRequest::new(OverlayKind::Help, "H"));
        mgr.push(OverlayRequest::new(OverlayKind::Alert, "A"));
        mgr.dismiss_all();
        assert!(!mgr.has_active());
        assert_eq!(mgr.depth(), 0);
    }

    #[test]
    fn overlay_request_builder() {
        let req = OverlayRequest::new(OverlayKind::Confirm, "Delete?")
            .with_body("This action cannot be undone.")
            .with_actions(vec!["Cancel".to_string(), "Delete".to_string()]);
        assert_eq!(req.kind, OverlayKind::Confirm);
        assert_eq!(req.body.as_deref(), Some("This action cannot be undone."));
        assert_eq!(req.actions.len(), 2);
    }

    #[test]
    fn overlay_kind_display() {
        assert_eq!(OverlayKind::Help.to_string(), "Help");
        assert_eq!(OverlayKind::Custom("Foo".into()).to_string(), "Foo");
    }

    #[test]
    fn overlay_kind_serde_roundtrip() {
        for kind in [
            OverlayKind::Help,
            OverlayKind::Confirm,
            OverlayKind::Alert,
            OverlayKind::Custom("test".into()),
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let decoded: OverlayKind = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, kind);
        }
    }
}
