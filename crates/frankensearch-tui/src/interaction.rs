//! Canonical interaction primitives ported from ftui-demo showcase patterns.
//!
//! This module defines product-agnostic contracts for:
//! - card/layout grammar
//! - command-palette intent semantics
//! - deterministic state serialization checkpoints
//! - interaction latency budget hooks

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

/// Versioned schema for interaction primitive contracts.
pub const SHOWCASE_INTERACTION_SPEC_VERSION: u16 = 1;

/// High-level interaction surfaces shared across frankensearch TUIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionSurfaceKind {
    Search,
    Results,
    Operations,
    Explainability,
}

impl InteractionSurfaceKind {
    /// Stable identifier for serialization and diagnostics.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Search => "search",
            Self::Results => "results",
            Self::Operations => "operations",
            Self::Explainability => "explainability",
        }
    }

    /// Required canonical surfaces from the showcase contract.
    #[must_use]
    pub const fn all() -> [Self; 4] {
        [
            Self::Search,
            Self::Results,
            Self::Operations,
            Self::Explainability,
        ]
    }
}

/// Preferred layout axis for a card region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutAxis {
    Horizontal,
    Vertical,
}

/// Semantic role of a layout card.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CardRole {
    QueryInput,
    Filters,
    ResultList,
    ResultPreview,
    JobQueue,
    ResourcePressure,
    Timeline,
    ScoreBreakdown,
    Provenance,
    OperatorControls,
}

/// Canonical card/layout grammar rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CardLayoutRule {
    pub card_id: String,
    pub role: CardRole,
    pub axis: LayoutAxis,
    pub min_width_cols: u16,
    pub min_height_rows: u16,
    pub virtualized: bool,
    pub sticky_header: bool,
}

impl CardLayoutRule {
    #[must_use]
    pub fn new(
        card_id: impl Into<String>,
        role: CardRole,
        axis: LayoutAxis,
        min_width_cols: u16,
        min_height_rows: u16,
        virtualized: bool,
        sticky_header: bool,
    ) -> Self {
        Self {
            card_id: card_id.into(),
            role,
            axis,
            min_width_cols,
            min_height_rows,
            virtualized,
            sticky_header,
        }
    }
}

/// Intent-level command semantics used by the command palette.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PaletteIntent {
    NavigateSurface,
    FocusQuery,
    RepeatQuery,
    PauseIndexing,
    ResumeIndexing,
    ToggleExplainability,
    OpenTimeline,
    ReplayTrace,
}

/// Route from a palette action ID to canonical intent semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PaletteIntentRoute {
    pub intent: PaletteIntent,
    pub action_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_surface: Option<InteractionSurfaceKind>,
    pub cross_screen_semantics: bool,
}

impl PaletteIntentRoute {
    #[must_use]
    pub fn new(
        intent: PaletteIntent,
        action_id: impl Into<String>,
        target_surface: Option<InteractionSurfaceKind>,
        cross_screen_semantics: bool,
    ) -> Self {
        Self {
            intent,
            action_id: action_id.into(),
            target_surface,
            cross_screen_semantics,
        }
    }
}

/// Deterministic checkpoints used by replay/snapshot contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeterministicCheckpoint {
    BeforeInputDispatch,
    AfterInputDispatch,
    BeforeStateSerialize,
    AfterStateSerialize,
    BeforeFrameCommit,
    AfterFrameCommit,
}

/// Explicit state boundary serialized at a deterministic checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeterministicStateBoundary {
    pub checkpoint: DeterministicCheckpoint,
    pub state_keys: Vec<String>,
}

impl DeterministicStateBoundary {
    #[must_use]
    pub fn new(checkpoint: DeterministicCheckpoint, state_keys: Vec<&str>) -> Self {
        Self {
            checkpoint,
            state_keys: state_keys.into_iter().map(str::to_owned).collect(),
        }
    }
}

/// Latency budget hooks exposed at interaction/component boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InteractionLatencyHooks {
    pub input_to_route_ms: u16,
    pub route_to_state_ms: u16,
    pub state_to_render_ms: u16,
    pub frame_budget_ms: u16,
}

impl InteractionLatencyHooks {
    #[must_use]
    pub const fn new(
        input_to_route_ms: u16,
        route_to_state_ms: u16,
        state_to_render_ms: u16,
        frame_budget_ms: u16,
    ) -> Self {
        Self {
            input_to_route_ms,
            route_to_state_ms,
            state_to_render_ms,
            frame_budget_ms,
        }
    }

    /// Returns the total component budget in milliseconds.
    ///
    /// Uses `u32` to avoid overflow when summing three `u16` fields.
    #[must_use]
    pub const fn component_budget_ms(self) -> u32 {
        self.input_to_route_ms as u32
            + self.route_to_state_ms as u32
            + self.state_to_render_ms as u32
    }

    fn validate(self, surface: InteractionSurfaceKind) -> Result<(), ShowcaseInteractionSpecError> {
        if self.input_to_route_ms == 0
            || self.route_to_state_ms == 0
            || self.state_to_render_ms == 0
            || self.frame_budget_ms == 0
        {
            return Err(ShowcaseInteractionSpecError::InvalidLatencyBudget(
                surface,
                "latency hooks must all be > 0".to_owned(),
            ));
        }
        if self.component_budget_ms() > u32::from(self.frame_budget_ms) {
            return Err(ShowcaseInteractionSpecError::InvalidLatencyBudget(
                surface,
                format!(
                    "component budget {}ms exceeds frame budget {}ms",
                    self.component_budget_ms(),
                    self.frame_budget_ms
                ),
            ));
        }
        Ok(())
    }
}

/// Contract for one canonical interaction surface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InteractionSurfaceContract {
    pub surface: InteractionSurfaceKind,
    pub cards: Vec<CardLayoutRule>,
    pub palette_routes: Vec<PaletteIntentRoute>,
    pub deterministic_boundaries: Vec<DeterministicStateBoundary>,
    pub latency_hooks: InteractionLatencyHooks,
}

impl InteractionSurfaceContract {
    fn validate(&self) -> Result<(), ShowcaseInteractionSpecError> {
        if self.cards.is_empty() {
            return Err(ShowcaseInteractionSpecError::EmptyCardGrammar(self.surface));
        }
        if self.palette_routes.is_empty() {
            return Err(ShowcaseInteractionSpecError::EmptyPaletteRoutes(
                self.surface,
            ));
        }

        let mut card_ids = BTreeSet::new();
        for card in &self.cards {
            if !card_ids.insert(card.card_id.clone()) {
                return Err(ShowcaseInteractionSpecError::DuplicateCardId(
                    self.surface,
                    card.card_id.clone(),
                ));
            }
        }

        let mut route_ids = BTreeSet::new();
        for route in &self.palette_routes {
            if !route_ids.insert(route.action_id.clone()) {
                return Err(ShowcaseInteractionSpecError::DuplicatePaletteActionId(
                    self.surface,
                    route.action_id.clone(),
                ));
            }
        }

        let has_before_serialize = self
            .deterministic_boundaries
            .iter()
            .any(|b| b.checkpoint == DeterministicCheckpoint::BeforeStateSerialize);
        let has_after_serialize = self
            .deterministic_boundaries
            .iter()
            .any(|b| b.checkpoint == DeterministicCheckpoint::AfterStateSerialize);
        if !(has_before_serialize && has_after_serialize) {
            return Err(ShowcaseInteractionSpecError::MissingSerializationBoundary(
                self.surface,
            ));
        }

        for boundary in &self.deterministic_boundaries {
            if matches!(
                boundary.checkpoint,
                DeterministicCheckpoint::BeforeStateSerialize
                    | DeterministicCheckpoint::AfterStateSerialize
            ) && boundary.state_keys.is_empty()
            {
                return Err(ShowcaseInteractionSpecError::EmptySerializationStateKeys(
                    self.surface,
                ));
            }
        }

        self.latency_hooks.validate(self.surface)
    }
}

/// Canonical interaction contract snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShowcaseInteractionSpec {
    pub spec_version: u16,
    pub source_profile: String,
    pub surfaces: Vec<InteractionSurfaceContract>,
}

impl ShowcaseInteractionSpec {
    /// Canonical showcase-derived contract that downstream TUIs should map to.
    #[must_use]
    pub fn canonical() -> Self {
        Self {
            spec_version: SHOWCASE_INTERACTION_SPEC_VERSION,
            source_profile: "ftui-demo-showcase".to_owned(),
            surfaces: vec![
                search_surface_contract(),
                results_surface_contract(),
                operations_surface_contract(),
                explainability_surface_contract(),
            ],
        }
    }

    /// Look up a surface by kind.
    #[must_use]
    pub fn surface(&self, surface: InteractionSurfaceKind) -> Option<&InteractionSurfaceContract> {
        self.surfaces
            .iter()
            .find(|candidate| candidate.surface == surface)
    }

    /// Validate contract determinism and replay/snapshot suitability.
    ///
    /// # Errors
    ///
    /// Returns [`ShowcaseInteractionSpecError`] if the contract is incomplete
    /// or violates deterministic interaction constraints.
    pub fn validate(&self) -> Result<(), ShowcaseInteractionSpecError> {
        if self.spec_version != SHOWCASE_INTERACTION_SPEC_VERSION {
            return Err(ShowcaseInteractionSpecError::UnsupportedSpecVersion(
                self.spec_version,
            ));
        }

        let mut seen = BTreeSet::new();
        for surface in &self.surfaces {
            if !seen.insert(surface.surface) {
                return Err(ShowcaseInteractionSpecError::DuplicateSurface(
                    surface.surface,
                ));
            }
            surface.validate()?;
        }

        for required in InteractionSurfaceKind::all() {
            if !seen.contains(&required) {
                return Err(ShowcaseInteractionSpecError::MissingSurface(required));
            }
        }

        Ok(())
    }
}

/// Validation failure for showcase interaction contracts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShowcaseInteractionSpecError {
    UnsupportedSpecVersion(u16),
    DuplicateSurface(InteractionSurfaceKind),
    MissingSurface(InteractionSurfaceKind),
    EmptyCardGrammar(InteractionSurfaceKind),
    EmptyPaletteRoutes(InteractionSurfaceKind),
    DuplicateCardId(InteractionSurfaceKind, String),
    DuplicatePaletteActionId(InteractionSurfaceKind, String),
    MissingSerializationBoundary(InteractionSurfaceKind),
    EmptySerializationStateKeys(InteractionSurfaceKind),
    InvalidLatencyBudget(InteractionSurfaceKind, String),
}

impl Display for ShowcaseInteractionSpecError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSpecVersion(version) => {
                write!(
                    f,
                    "unsupported showcase interaction spec version: {version}"
                )
            }
            Self::DuplicateSurface(surface) => {
                write!(f, "duplicate showcase surface contract: {}", surface.id())
            }
            Self::MissingSurface(surface) => {
                write!(f, "missing required showcase surface: {}", surface.id())
            }
            Self::EmptyCardGrammar(surface) => {
                write!(f, "surface {} has empty card grammar", surface.id())
            }
            Self::EmptyPaletteRoutes(surface) => {
                write!(f, "surface {} has empty palette routes", surface.id())
            }
            Self::DuplicateCardId(surface, card_id) => write!(
                f,
                "surface {} defines duplicate card id: {card_id}",
                surface.id()
            ),
            Self::DuplicatePaletteActionId(surface, action_id) => write!(
                f,
                "surface {} defines duplicate palette action id: {action_id}",
                surface.id()
            ),
            Self::MissingSerializationBoundary(surface) => write!(
                f,
                "surface {} is missing before/after serialization checkpoints",
                surface.id()
            ),
            Self::EmptySerializationStateKeys(surface) => write!(
                f,
                "surface {} has serialization checkpoint with empty state keys",
                surface.id()
            ),
            Self::InvalidLatencyBudget(surface, detail) => write!(
                f,
                "surface {} has invalid latency budget: {detail}",
                surface.id()
            ),
        }
    }
}

impl Error for ShowcaseInteractionSpecError {}

fn search_surface_contract() -> InteractionSurfaceContract {
    InteractionSurfaceContract {
        surface: InteractionSurfaceKind::Search,
        cards: vec![
            CardLayoutRule::new(
                "search.query",
                CardRole::QueryInput,
                LayoutAxis::Horizontal,
                60,
                3,
                false,
                true,
            ),
            CardLayoutRule::new(
                "search.filters",
                CardRole::Filters,
                LayoutAxis::Horizontal,
                40,
                3,
                false,
                true,
            ),
        ],
        palette_routes: vec![
            PaletteIntentRoute::new(
                PaletteIntent::FocusQuery,
                "search.focus_query",
                Some(InteractionSurfaceKind::Search),
                false,
            ),
            PaletteIntentRoute::new(
                PaletteIntent::RepeatQuery,
                "search.repeat_last",
                Some(InteractionSurfaceKind::Search),
                false,
            ),
        ],
        deterministic_boundaries: vec![
            DeterministicStateBoundary::new(
                DeterministicCheckpoint::BeforeInputDispatch,
                vec!["active_screen", "palette.query", "search.query"],
            ),
            DeterministicStateBoundary::new(
                DeterministicCheckpoint::BeforeStateSerialize,
                vec!["search.query", "search.filters", "search.mode"],
            ),
            DeterministicStateBoundary::new(
                DeterministicCheckpoint::AfterStateSerialize,
                vec!["search.query", "search.filters", "search.cursor"],
            ),
            DeterministicStateBoundary::new(
                DeterministicCheckpoint::AfterFrameCommit,
                vec!["frame.seq", "search.focused"],
            ),
        ],
        latency_hooks: InteractionLatencyHooks::new(4, 4, 8, 16),
    }
}

fn results_surface_contract() -> InteractionSurfaceContract {
    InteractionSurfaceContract {
        surface: InteractionSurfaceKind::Results,
        cards: vec![
            CardLayoutRule::new(
                "results.list",
                CardRole::ResultList,
                LayoutAxis::Vertical,
                64,
                12,
                true,
                true,
            ),
            CardLayoutRule::new(
                "results.preview",
                CardRole::ResultPreview,
                LayoutAxis::Vertical,
                48,
                10,
                false,
                false,
            ),
        ],
        palette_routes: vec![
            PaletteIntentRoute::new(
                PaletteIntent::NavigateSurface,
                "nav.fsfs.search",
                Some(InteractionSurfaceKind::Results),
                false,
            ),
            PaletteIntentRoute::new(
                PaletteIntent::ToggleExplainability,
                "explain.toggle_panel",
                Some(InteractionSurfaceKind::Explainability),
                true,
            ),
        ],
        deterministic_boundaries: vec![
            DeterministicStateBoundary::new(
                DeterministicCheckpoint::AfterInputDispatch,
                vec!["results.selected_index", "results.scroll_offset"],
            ),
            DeterministicStateBoundary::new(
                DeterministicCheckpoint::BeforeStateSerialize,
                vec!["results.selected_doc_id", "results.visible_window"],
            ),
            DeterministicStateBoundary::new(
                DeterministicCheckpoint::AfterStateSerialize,
                vec!["results.selected_doc_id", "results.render_model_hash"],
            ),
            DeterministicStateBoundary::new(
                DeterministicCheckpoint::BeforeFrameCommit,
                vec!["frame.seq", "results.virtualized_window"],
            ),
        ],
        latency_hooks: InteractionLatencyHooks::new(3, 5, 8, 16),
    }
}

fn operations_surface_contract() -> InteractionSurfaceContract {
    InteractionSurfaceContract {
        surface: InteractionSurfaceKind::Operations,
        cards: vec![
            CardLayoutRule::new(
                "ops.jobs",
                CardRole::JobQueue,
                LayoutAxis::Vertical,
                48,
                8,
                false,
                true,
            ),
            CardLayoutRule::new(
                "ops.pressure",
                CardRole::ResourcePressure,
                LayoutAxis::Horizontal,
                48,
                6,
                false,
                true,
            ),
            CardLayoutRule::new(
                "ops.timeline",
                CardRole::Timeline,
                LayoutAxis::Vertical,
                64,
                10,
                true,
                true,
            ),
        ],
        palette_routes: vec![
            PaletteIntentRoute::new(
                PaletteIntent::PauseIndexing,
                "index.pause",
                Some(InteractionSurfaceKind::Operations),
                false,
            ),
            PaletteIntentRoute::new(
                PaletteIntent::ResumeIndexing,
                "index.resume",
                Some(InteractionSurfaceKind::Operations),
                false,
            ),
            PaletteIntentRoute::new(
                PaletteIntent::OpenTimeline,
                "ops.open_timeline",
                Some(InteractionSurfaceKind::Operations),
                false,
            ),
        ],
        deterministic_boundaries: vec![
            DeterministicStateBoundary::new(
                DeterministicCheckpoint::BeforeInputDispatch,
                vec!["ops.active_lane", "ops.pause_state"],
            ),
            DeterministicStateBoundary::new(
                DeterministicCheckpoint::BeforeStateSerialize,
                vec![
                    "ops.queue_depth",
                    "ops.disk_budget_stage",
                    "ops.pressure_state",
                ],
            ),
            DeterministicStateBoundary::new(
                DeterministicCheckpoint::AfterStateSerialize,
                vec!["ops.timeline_cursor", "ops.alert_counts"],
            ),
            DeterministicStateBoundary::new(
                DeterministicCheckpoint::AfterFrameCommit,
                vec!["frame.seq", "ops.timeline_window"],
            ),
        ],
        latency_hooks: InteractionLatencyHooks::new(5, 4, 9, 20),
    }
}

fn explainability_surface_contract() -> InteractionSurfaceContract {
    InteractionSurfaceContract {
        surface: InteractionSurfaceKind::Explainability,
        cards: vec![
            CardLayoutRule::new(
                "explain.scores",
                CardRole::ScoreBreakdown,
                LayoutAxis::Vertical,
                48,
                8,
                false,
                true,
            ),
            CardLayoutRule::new(
                "explain.provenance",
                CardRole::Provenance,
                LayoutAxis::Vertical,
                48,
                8,
                false,
                false,
            ),
            CardLayoutRule::new(
                "explain.controls",
                CardRole::OperatorControls,
                LayoutAxis::Horizontal,
                32,
                4,
                false,
                false,
            ),
        ],
        palette_routes: vec![
            PaletteIntentRoute::new(
                PaletteIntent::ToggleExplainability,
                "explain.toggle_panel",
                Some(InteractionSurfaceKind::Explainability),
                false,
            ),
            PaletteIntentRoute::new(
                PaletteIntent::ReplayTrace,
                "diag.replay_trace",
                Some(InteractionSurfaceKind::Explainability),
                true,
            ),
        ],
        deterministic_boundaries: vec![
            DeterministicStateBoundary::new(
                DeterministicCheckpoint::AfterInputDispatch,
                vec!["explain.active_panel", "explain.selected_component"],
            ),
            DeterministicStateBoundary::new(
                DeterministicCheckpoint::BeforeStateSerialize,
                vec!["explain.rank_components", "explain.prior_evidence"],
            ),
            DeterministicStateBoundary::new(
                DeterministicCheckpoint::AfterStateSerialize,
                vec!["explain.panel_state_hash", "explain.selection_hash"],
            ),
            DeterministicStateBoundary::new(
                DeterministicCheckpoint::BeforeFrameCommit,
                vec!["frame.seq", "explain.viewport"],
            ),
        ],
        latency_hooks: InteractionLatencyHooks::new(4, 6, 8, 20),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DeterministicCheckpoint, InteractionSurfaceKind, ShowcaseInteractionSpec,
        ShowcaseInteractionSpecError,
    };

    #[test]
    fn canonical_spec_contains_all_required_surfaces() {
        let spec = ShowcaseInteractionSpec::canonical();
        spec.validate().expect("canonical spec should validate");

        for required in InteractionSurfaceKind::all() {
            assert!(spec.surface(required).is_some());
        }
    }

    #[test]
    fn serialization_boundaries_are_present_for_each_surface() {
        let spec = ShowcaseInteractionSpec::canonical();
        for surface in &spec.surfaces {
            assert!(
                surface
                    .deterministic_boundaries
                    .iter()
                    .any(|b| b.checkpoint == DeterministicCheckpoint::BeforeStateSerialize)
            );
            assert!(
                surface
                    .deterministic_boundaries
                    .iter()
                    .any(|b| b.checkpoint == DeterministicCheckpoint::AfterStateSerialize)
            );
        }
    }

    #[test]
    fn latency_hooks_fit_inside_frame_budget() {
        let spec = ShowcaseInteractionSpec::canonical();
        for surface in &spec.surfaces {
            assert!(
                surface.latency_hooks.component_budget_ms()
                    <= u32::from(surface.latency_hooks.frame_budget_ms)
            );
        }
    }

    #[test]
    fn validate_rejects_missing_required_surface() {
        let mut spec = ShowcaseInteractionSpec::canonical();
        spec.surfaces
            .retain(|surface| surface.surface != InteractionSurfaceKind::Results);

        let err = spec
            .validate()
            .expect_err("missing required surface must fail");
        assert_eq!(
            err,
            ShowcaseInteractionSpecError::MissingSurface(InteractionSurfaceKind::Results)
        );
    }

    #[test]
    fn validate_rejects_duplicate_palette_routes() {
        let mut spec = ShowcaseInteractionSpec::canonical();
        let search_surface = spec
            .surfaces
            .iter_mut()
            .find(|surface| surface.surface == InteractionSurfaceKind::Search)
            .expect("search surface should exist");
        let duplicated_route = search_surface.palette_routes[0].clone();
        search_surface.palette_routes.push(duplicated_route.clone());

        let err = spec.validate().expect_err("duplicate routes must fail");
        assert_eq!(
            err,
            ShowcaseInteractionSpecError::DuplicatePaletteActionId(
                InteractionSurfaceKind::Search,
                duplicated_route.action_id
            )
        );
    }

    // ── InteractionSurfaceKind tests ─────────────────────────────────

    #[test]
    fn surface_kind_ids_are_unique() {
        let all = InteractionSurfaceKind::all();
        let ids: Vec<&str> = all.iter().map(|kind| kind.id()).collect();
        for (i, id) in ids.iter().enumerate() {
            for (j, other) in ids.iter().enumerate() {
                if i != j {
                    assert_ne!(id, other, "duplicate surface id: {id}");
                }
            }
        }
    }

    #[test]
    fn surface_kind_all_returns_four_variants() {
        assert_eq!(InteractionSurfaceKind::all().len(), 4);
    }

    #[test]
    fn surface_kind_ids_are_nonempty() {
        for kind in InteractionSurfaceKind::all() {
            assert_ne!(kind.id(), "");
        }
    }

    // ── InteractionLatencyHooks tests ────────────────────────────────

    #[test]
    fn component_budget_sums_phases() {
        let hooks = super::InteractionLatencyHooks::new(1, 2, 3, 10);
        assert_eq!(hooks.component_budget_ms(), 6);
    }

    #[test]
    fn validate_rejects_zero_latency_fields() {
        let hooks = super::InteractionLatencyHooks::new(0, 2, 3, 10);
        let err = hooks
            .validate(InteractionSurfaceKind::Search)
            .expect_err("zero field must fail");
        match err {
            ShowcaseInteractionSpecError::InvalidLatencyBudget(surface, _) => {
                assert_eq!(surface, InteractionSurfaceKind::Search);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_component_exceeding_frame() {
        let hooks = super::InteractionLatencyHooks::new(5, 5, 5, 10);
        let err = hooks
            .validate(InteractionSurfaceKind::Results)
            .expect_err("component > frame must fail");
        match err {
            ShowcaseInteractionSpecError::InvalidLatencyBudget(_, detail) => {
                assert!(detail.contains("exceeds"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn validate_accepts_component_equal_to_frame() {
        let hooks = super::InteractionLatencyHooks::new(3, 3, 4, 10);
        assert!(hooks.validate(InteractionSurfaceKind::Search).is_ok());
    }

    // ── Contract validation tests ────────────────────────────────────

    #[test]
    fn validate_rejects_empty_card_grammar() {
        let mut spec = ShowcaseInteractionSpec::canonical();
        let surface = spec
            .surfaces
            .iter_mut()
            .find(|s| s.surface == InteractionSurfaceKind::Search)
            .unwrap();
        surface.cards.clear();
        let err = spec.validate().expect_err("empty cards must fail");
        assert_eq!(
            err,
            ShowcaseInteractionSpecError::EmptyCardGrammar(InteractionSurfaceKind::Search)
        );
    }

    #[test]
    fn validate_rejects_empty_palette_routes() {
        let mut spec = ShowcaseInteractionSpec::canonical();
        let surface = spec
            .surfaces
            .iter_mut()
            .find(|s| s.surface == InteractionSurfaceKind::Search)
            .unwrap();
        surface.palette_routes.clear();
        let err = spec.validate().expect_err("empty routes must fail");
        assert_eq!(
            err,
            ShowcaseInteractionSpecError::EmptyPaletteRoutes(InteractionSurfaceKind::Search)
        );
    }

    #[test]
    fn validate_rejects_duplicate_card_ids() {
        let mut spec = ShowcaseInteractionSpec::canonical();
        let surface = spec
            .surfaces
            .iter_mut()
            .find(|s| s.surface == InteractionSurfaceKind::Search)
            .unwrap();
        let dup = surface.cards[0].clone();
        surface.cards.push(dup.clone());
        let err = spec.validate().expect_err("duplicate card ids must fail");
        assert_eq!(
            err,
            ShowcaseInteractionSpecError::DuplicateCardId(
                InteractionSurfaceKind::Search,
                dup.card_id
            )
        );
    }

    #[test]
    fn validate_rejects_missing_serialization_boundaries() {
        let mut spec = ShowcaseInteractionSpec::canonical();
        let surface = spec
            .surfaces
            .iter_mut()
            .find(|s| s.surface == InteractionSurfaceKind::Search)
            .unwrap();
        surface
            .deterministic_boundaries
            .retain(|b| b.checkpoint != DeterministicCheckpoint::BeforeStateSerialize);
        let err = spec
            .validate()
            .expect_err("missing serialization boundary must fail");
        assert_eq!(
            err,
            ShowcaseInteractionSpecError::MissingSerializationBoundary(
                InteractionSurfaceKind::Search
            )
        );
    }

    #[test]
    fn validate_rejects_empty_state_keys_at_serialization_checkpoint() {
        let mut spec = ShowcaseInteractionSpec::canonical();
        let surface = spec
            .surfaces
            .iter_mut()
            .find(|s| s.surface == InteractionSurfaceKind::Search)
            .unwrap();
        let boundary = surface
            .deterministic_boundaries
            .iter_mut()
            .find(|b| b.checkpoint == DeterministicCheckpoint::BeforeStateSerialize)
            .unwrap();
        boundary.state_keys.clear();
        let err = spec.validate().expect_err("empty state keys must fail");
        assert_eq!(
            err,
            ShowcaseInteractionSpecError::EmptySerializationStateKeys(
                InteractionSurfaceKind::Search
            )
        );
    }

    #[test]
    fn validate_rejects_wrong_spec_version() {
        let mut spec = ShowcaseInteractionSpec::canonical();
        spec.spec_version = 999;
        let err = spec.validate().expect_err("wrong version must fail");
        assert_eq!(
            err,
            ShowcaseInteractionSpecError::UnsupportedSpecVersion(999)
        );
    }

    #[test]
    fn validate_rejects_duplicate_surfaces() {
        let mut spec = ShowcaseInteractionSpec::canonical();
        let dup = spec.surfaces[0].clone();
        spec.surfaces.push(dup);
        let err = spec.validate().expect_err("duplicate surface must fail");
        assert!(matches!(
            err,
            ShowcaseInteractionSpecError::DuplicateSurface(_)
        ));
    }

    // ── surface() lookup ─────────────────────────────────────────────

    #[test]
    fn surface_lookup_returns_none_for_missing() {
        let mut spec = ShowcaseInteractionSpec::canonical();
        spec.surfaces
            .retain(|s| s.surface != InteractionSurfaceKind::Explainability);
        assert!(
            spec.surface(InteractionSurfaceKind::Explainability)
                .is_none()
        );
    }

    #[test]
    fn surface_lookup_returns_matching() {
        let spec = ShowcaseInteractionSpec::canonical();
        let search = spec.surface(InteractionSurfaceKind::Search);
        assert!(search.is_some());
        assert_eq!(search.unwrap().surface, InteractionSurfaceKind::Search);
    }

    // ── Error Display formatting ─────────────────────────────────────

    #[test]
    fn error_display_contains_surface_id() {
        let err = ShowcaseInteractionSpecError::EmptyCardGrammar(InteractionSurfaceKind::Search);
        let msg = format!("{err}");
        assert!(
            msg.contains("search"),
            "error should mention surface: {msg}"
        );
    }

    #[test]
    fn error_display_version() {
        let err = ShowcaseInteractionSpecError::UnsupportedSpecVersion(42);
        let msg = format!("{err}");
        assert!(msg.contains("42"));
    }

    // ── Serde roundtrip ──────────────────────────────────────────────

    #[test]
    fn canonical_spec_serde_roundtrip() {
        let spec = ShowcaseInteractionSpec::canonical();
        let json = serde_json::to_string(&spec).expect("serialize");
        let deser: ShowcaseInteractionSpec = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(spec, deser);
    }

    // ── Construction helpers ─────────────────────────────────────────

    #[test]
    fn card_layout_rule_construction() {
        let rule = super::CardLayoutRule::new(
            "test.card",
            super::CardRole::QueryInput,
            super::LayoutAxis::Horizontal,
            40,
            3,
            true,
            false,
        );
        assert_eq!(rule.card_id, "test.card");
        assert!(rule.virtualized);
        assert!(!rule.sticky_header);
    }

    #[test]
    fn palette_intent_route_construction() {
        let route = super::PaletteIntentRoute::new(
            super::PaletteIntent::FocusQuery,
            "test.action",
            None,
            true,
        );
        assert_eq!(route.action_id, "test.action");
        assert!(route.cross_screen_semantics);
        assert!(route.target_surface.is_none());
    }

    #[test]
    fn deterministic_state_boundary_converts_keys() {
        let boundary = super::DeterministicStateBoundary::new(
            DeterministicCheckpoint::BeforeFrameCommit,
            vec!["key1", "key2"],
        );
        assert_eq!(boundary.state_keys, vec!["key1", "key2"]);
    }
}
