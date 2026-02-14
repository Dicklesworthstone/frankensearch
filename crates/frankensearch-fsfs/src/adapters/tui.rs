use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use frankensearch_tui::{
    Action, ActionCategory, CommandPalette, InteractionSurfaceKind, PaletteIntent,
    PaletteIntentRoute, ScreenId, ShowcaseInteractionSpec,
};

use crate::config::{Density, FsfsConfig, TuiTheme};

const REPLAY_TRACE_ACTION_ID: &str = "diag.replay_trace";

/// TUI-facing settings derived from resolved fsfs config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiAdapterSettings {
    pub theme: TuiTheme,
    pub density: Density,
    pub show_explanations: bool,
}

impl From<&FsfsConfig> for TuiAdapterSettings {
    fn from(config: &FsfsConfig) -> Self {
        Self {
            theme: config.tui.theme,
            density: config.tui.density,
            show_explanations: config.tui.show_explanations,
        }
    }
}

/// Canonical fsfs deluxe-TUI screen identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FsfsScreen {
    Search,
    Indexing,
    Pressure,
    Explainability,
    Configuration,
    OpsTimeline,
}

impl FsfsScreen {
    /// Ordered screen list used by tab navigation.
    #[must_use]
    pub const fn all() -> [Self; 6] {
        [
            Self::Search,
            Self::Indexing,
            Self::Pressure,
            Self::Explainability,
            Self::Configuration,
            Self::OpsTimeline,
        ]
    }

    /// Stable screen ID used by shared TUI shell/registry APIs.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Search => "fsfs.search",
            Self::Indexing => "fsfs.indexing",
            Self::Pressure => "fsfs.pressure",
            Self::Explainability => "fsfs.explain",
            Self::Configuration => "fsfs.config",
            Self::OpsTimeline => "fsfs.timeline",
        }
    }

    /// Human label shown in navigation and command palette.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Search => "Search",
            Self::Indexing => "Indexing",
            Self::Pressure => "Pressure",
            Self::Explainability => "Explainability",
            Self::Configuration => "Configuration",
            Self::OpsTimeline => "Timeline",
        }
    }

    /// Shared TUI `ScreenId` conversion.
    #[must_use]
    pub fn screen_id(self) -> ScreenId {
        ScreenId::new(self.id())
    }
}

/// Retention policy for cross-screen navigation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextRetentionPolicy {
    PreserveAll,
    PreserveQueryAndFilters,
    ResetTransient,
}

/// Explicit transition policy for preserving search context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextRetentionRule {
    pub from: FsfsScreen,
    pub to: FsfsScreen,
    pub policy: ContextRetentionPolicy,
}

/// Shell navigation model reusable by both runtime bootstrap and tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiNavigationModel {
    pub default_screen: FsfsScreen,
    pub screen_order: Vec<FsfsScreen>,
    pub history_limit: usize,
    pub context_rules: Vec<ContextRetentionRule>,
}

impl Default for TuiNavigationModel {
    fn default() -> Self {
        Self {
            default_screen: FsfsScreen::Search,
            screen_order: FsfsScreen::all().to_vec(),
            history_limit: 64,
            context_rules: vec![
                ContextRetentionRule {
                    from: FsfsScreen::Search,
                    to: FsfsScreen::Explainability,
                    policy: ContextRetentionPolicy::PreserveAll,
                },
                ContextRetentionRule {
                    from: FsfsScreen::Search,
                    to: FsfsScreen::Pressure,
                    policy: ContextRetentionPolicy::PreserveQueryAndFilters,
                },
                ContextRetentionRule {
                    from: FsfsScreen::Explainability,
                    to: FsfsScreen::Search,
                    policy: ContextRetentionPolicy::PreserveAll,
                },
            ],
        }
    }
}

impl TuiNavigationModel {
    /// Convert ordered screens into shared shell IDs.
    #[must_use]
    pub fn screen_ids(&self) -> Vec<ScreenId> {
        self.screen_order
            .iter()
            .copied()
            .map(FsfsScreen::screen_id)
            .collect()
    }

    /// Next screen in tab order with wrap-around fallback.
    #[must_use]
    pub fn next_screen(&self, current: FsfsScreen) -> FsfsScreen {
        let Some(index) = self
            .screen_order
            .iter()
            .position(|screen| *screen == current)
        else {
            return self.default_screen;
        };

        let next = (index + 1) % self.screen_order.len();
        self.screen_order[next]
    }

    /// Previous screen in tab order with wrap-around fallback.
    #[must_use]
    pub fn previous_screen(&self, current: FsfsScreen) -> FsfsScreen {
        let Some(index) = self
            .screen_order
            .iter()
            .position(|screen| *screen == current)
        else {
            return self.default_screen;
        };

        let prev = if index == 0 {
            self.screen_order.len().saturating_sub(1)
        } else {
            index - 1
        };
        self.screen_order[prev]
    }

    /// Resolve context-retention policy for a navigation transition.
    #[must_use]
    pub fn context_policy(&self, from: FsfsScreen, to: FsfsScreen) -> ContextRetentionPolicy {
        self.context_rules
            .iter()
            .find(|rule| rule.from == from && rule.to == to)
            .map_or_else(|| Self::default_policy_for_target(to), |rule| rule.policy)
    }

    #[must_use]
    const fn default_policy_for_target(target: FsfsScreen) -> ContextRetentionPolicy {
        match target {
            FsfsScreen::Search | FsfsScreen::Explainability => ContextRetentionPolicy::PreserveAll,
            FsfsScreen::Pressure | FsfsScreen::OpsTimeline => {
                ContextRetentionPolicy::PreserveQueryAndFilters
            }
            FsfsScreen::Indexing | FsfsScreen::Configuration => {
                ContextRetentionPolicy::ResetTransient
            }
        }
    }

    /// Validate deterministic navigation invariants.
    ///
    /// # Errors
    ///
    /// Returns [`TuiModelValidationError`] when screen order is invalid,
    /// default screen is missing, or history settings are non-viable.
    pub fn validate(&self) -> Result<(), TuiModelValidationError> {
        if self.screen_order.is_empty() {
            return Err(TuiModelValidationError::EmptyScreenOrder);
        }
        if self.history_limit == 0 {
            return Err(TuiModelValidationError::InvalidHistoryLimit(0));
        }

        let mut seen = BTreeSet::new();
        for screen in &self.screen_order {
            if !seen.insert(*screen) {
                return Err(TuiModelValidationError::DuplicateScreenId(
                    screen.id().to_owned(),
                ));
            }
        }
        if !seen.contains(&self.default_screen) {
            return Err(TuiModelValidationError::MissingDefaultScreen(
                self.default_screen.id().to_owned(),
            ));
        }

        Ok(())
    }
}

/// Keybinding scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuiKeyBindingScope {
    Global,
    Palette,
}

impl TuiKeyBindingScope {
    #[must_use]
    const fn as_str(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Palette => "palette",
        }
    }
}

/// Normalized keybinding model exposed for docs/tests/automation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiKeyBindingSpec {
    pub chord: String,
    pub action_id: String,
    pub scope: TuiKeyBindingScope,
}

impl TuiKeyBindingSpec {
    #[must_use]
    pub fn new(
        chord: impl Into<String>,
        action_id: impl Into<String>,
        scope: TuiKeyBindingScope,
    ) -> Self {
        Self {
            chord: chord.into(),
            action_id: action_id.into(),
            scope,
        }
    }
}

/// Global + palette keymap contract for deluxe TUI shell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiKeymapModel {
    pub bindings: Vec<TuiKeyBindingSpec>,
}

impl Default for TuiKeymapModel {
    fn default() -> Self {
        Self {
            bindings: vec![
                TuiKeyBindingSpec::new(
                    "Ctrl+P",
                    "shell.palette.toggle",
                    TuiKeyBindingScope::Global,
                ),
                TuiKeyBindingSpec::new("Tab", "shell.screen.next", TuiKeyBindingScope::Global),
                TuiKeyBindingSpec::new(
                    "Shift+Tab",
                    "shell.screen.previous",
                    TuiKeyBindingScope::Global,
                ),
                TuiKeyBindingSpec::new("?", "shell.help.toggle", TuiKeyBindingScope::Global),
                TuiKeyBindingSpec::new("Esc", "shell.dismiss", TuiKeyBindingScope::Global),
                TuiKeyBindingSpec::new("Ctrl+L", "search.focus_query", TuiKeyBindingScope::Global),
                TuiKeyBindingSpec::new(
                    "Ctrl+Enter",
                    "search.submit_query",
                    TuiKeyBindingScope::Global,
                ),
                TuiKeyBindingSpec::new("Up", "search.select_up", TuiKeyBindingScope::Global),
                TuiKeyBindingSpec::new("Down", "search.select_down", TuiKeyBindingScope::Global),
                TuiKeyBindingSpec::new("PageUp", "search.page_up", TuiKeyBindingScope::Global),
                TuiKeyBindingSpec::new("PageDown", "search.page_down", TuiKeyBindingScope::Global),
                TuiKeyBindingSpec::new(
                    "Ctrl+E",
                    "search.toggle_explain",
                    TuiKeyBindingScope::Global,
                ),
                TuiKeyBindingSpec::new(
                    "Ctrl+O",
                    "search.open_selected",
                    TuiKeyBindingScope::Global,
                ),
                TuiKeyBindingSpec::new(
                    "Ctrl+J",
                    "search.jump_to_source",
                    TuiKeyBindingScope::Global,
                ),
                TuiKeyBindingSpec::new("Enter", "palette.confirm", TuiKeyBindingScope::Palette),
                TuiKeyBindingSpec::new("Up", "palette.up", TuiKeyBindingScope::Palette),
                TuiKeyBindingSpec::new("Down", "palette.down", TuiKeyBindingScope::Palette),
                TuiKeyBindingSpec::new("Esc", "palette.dismiss", TuiKeyBindingScope::Palette),
            ],
        }
    }
}

impl TuiKeymapModel {
    /// Validate duplicate keybindings inside a scope.
    ///
    /// # Errors
    ///
    /// Returns [`TuiModelValidationError::DuplicateKeyBinding`] when two
    /// bindings collide on the same `(scope, chord)` tuple.
    pub fn validate(&self) -> Result<(), TuiModelValidationError> {
        let mut seen = BTreeSet::new();
        for binding in &self.bindings {
            let key = format!("{}::{}", binding.scope.as_str(), binding.chord);
            if !seen.insert(key.clone()) {
                return Err(TuiModelValidationError::DuplicateKeyBinding(key));
            }
        }
        Ok(())
    }
}

/// Palette taxonomy used by fsfs command router.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuiPaletteCategory {
    Navigation,
    Search,
    Indexing,
    Explainability,
    Configuration,
    Operations,
    Diagnostics,
}

impl TuiPaletteCategory {
    #[must_use]
    fn to_action_category(self) -> ActionCategory {
        match self {
            Self::Navigation => ActionCategory::Navigation,
            Self::Search => ActionCategory::Search,
            Self::Indexing => ActionCategory::Custom("Indexing".to_owned()),
            Self::Explainability => ActionCategory::Custom("Explainability".to_owned()),
            Self::Configuration => ActionCategory::Settings,
            Self::Operations => ActionCategory::Custom("Operations".to_owned()),
            Self::Diagnostics => ActionCategory::Debug,
        }
    }
}

/// Action specification for palette rendering and routing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiPaletteActionSpec {
    pub id: String,
    pub label: String,
    pub description: String,
    pub category: TuiPaletteCategory,
    pub shortcut_hint: Option<String>,
    pub target_screen: Option<FsfsScreen>,
}

impl TuiPaletteActionSpec {
    #[must_use]
    pub fn navigation(target: FsfsScreen) -> Self {
        Self {
            id: format!("nav.{}", target.id()),
            label: format!("Go to {}", target.label()),
            description: format!("Navigate to the {} screen", target.label()),
            category: TuiPaletteCategory::Navigation,
            shortcut_hint: None,
            target_screen: Some(target),
        }
    }

    #[must_use]
    pub fn named(
        id: impl Into<String>,
        label: impl Into<String>,
        description: impl Into<String>,
        category: TuiPaletteCategory,
        shortcut_hint: Option<&str>,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            description: description.into(),
            category,
            shortcut_hint: shortcut_hint.map(str::to_owned),
            target_screen: None,
        }
    }

    #[must_use]
    pub fn to_palette_action(&self) -> Action {
        let mut action = Action::new(
            self.id.clone(),
            self.label.clone(),
            self.category.to_action_category(),
        )
        .with_description(self.description.clone());

        if let Some(shortcut) = &self.shortcut_hint {
            action = action.with_shortcut(shortcut.clone());
        }

        action
    }
}

/// Searchable palette model with taxonomy and route IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiPaletteModel {
    pub actions: Vec<TuiPaletteActionSpec>,
}

impl TuiPaletteModel {
    #[must_use]
    pub fn from_navigation(navigation: &TuiNavigationModel) -> Self {
        let mut actions = navigation
            .screen_order
            .iter()
            .copied()
            .map(TuiPaletteActionSpec::navigation)
            .collect::<Vec<_>>();

        actions.extend(Self::search_actions());
        actions.extend(Self::indexing_actions());
        actions.extend(Self::operations_actions());

        Self { actions }
    }

    fn search_actions() -> Vec<TuiPaletteActionSpec> {
        vec![
            TuiPaletteActionSpec::named(
                "search.focus_query",
                "Focus Query Input",
                "Place cursor in the search query editor",
                TuiPaletteCategory::Search,
                Some("Ctrl+L"),
            ),
            TuiPaletteActionSpec::named(
                "search.submit_query",
                "Submit Query",
                "Submit the current query text for staged retrieval",
                TuiPaletteCategory::Search,
                Some("Ctrl+Enter"),
            ),
            TuiPaletteActionSpec::named(
                "search.clear_query",
                "Clear Query",
                "Clear query input and pending incremental submission",
                TuiPaletteCategory::Search,
                None,
            ),
            TuiPaletteActionSpec::named(
                "search.repeat_last",
                "Repeat Last Search",
                "Re-run the most recent search with current mode settings",
                TuiPaletteCategory::Search,
                None,
            ),
            TuiPaletteActionSpec::named(
                "search.toggle_explain",
                "Toggle Inline Explainability",
                "Show or hide inline score/provenance details for selected result",
                TuiPaletteCategory::Search,
                Some("Ctrl+E"),
            ),
            TuiPaletteActionSpec::named(
                "search.open_selected",
                "Open Selected Result",
                "Open the currently selected result in the local editor/viewer",
                TuiPaletteCategory::Search,
                Some("Ctrl+O"),
            ),
            TuiPaletteActionSpec::named(
                "search.jump_to_source",
                "Jump To Source",
                "Jump directly to the selected result's file/source location",
                TuiPaletteCategory::Search,
                Some("Ctrl+J"),
            ),
        ]
    }

    fn indexing_actions() -> Vec<TuiPaletteActionSpec> {
        vec![
            TuiPaletteActionSpec::named(
                "index.pause",
                "Pause Indexing",
                "Pause background indexing jobs while preserving queue state",
                TuiPaletteCategory::Indexing,
                None,
            ),
            TuiPaletteActionSpec::named(
                "index.resume",
                "Resume Indexing",
                "Resume paused indexing jobs",
                TuiPaletteCategory::Indexing,
                None,
            ),
            TuiPaletteActionSpec::named(
                "index.throttle",
                "Throttle Indexing",
                "Apply constrained watcher/indexing throttle under resource pressure",
                TuiPaletteCategory::Indexing,
                None,
            ),
            TuiPaletteActionSpec::named(
                "index.recover",
                "Recover Indexing Mode",
                "Request staged recovery toward normal indexing/query operation",
                TuiPaletteCategory::Indexing,
                None,
            ),
            TuiPaletteActionSpec::named(
                "index.override.auto",
                "Override: Auto Policy",
                "Clear manual override and return to automatic degradation policy",
                TuiPaletteCategory::Indexing,
                None,
            ),
            TuiPaletteActionSpec::named(
                "index.override.full",
                "Override: Force Full",
                "Force full retrieval/indexing mode when guardrails permit",
                TuiPaletteCategory::Indexing,
                None,
            ),
            TuiPaletteActionSpec::named(
                "index.override.embed_deferred",
                "Override: Force Embed Deferred",
                "Force embed-deferred mode to protect interactive latency",
                TuiPaletteCategory::Indexing,
                None,
            ),
            TuiPaletteActionSpec::named(
                "index.override.lexical_only",
                "Override: Force Lexical Only",
                "Force lexical-only retrieval while preserving correctness",
                TuiPaletteCategory::Indexing,
                None,
            ),
            TuiPaletteActionSpec::named(
                "index.override.metadata_only",
                "Override: Force Metadata Only",
                "Force metadata-only fallback for emergency operation",
                TuiPaletteCategory::Indexing,
                None,
            ),
            TuiPaletteActionSpec::named(
                "index.override.paused",
                "Override: Force Pause",
                "Force paused state until operator clears override",
                TuiPaletteCategory::Indexing,
                None,
            ),
        ]
    }

    fn operations_actions() -> [TuiPaletteActionSpec; 4] {
        [
            TuiPaletteActionSpec::named(
                "explain.toggle_panel",
                "Toggle Explainability Panel",
                "Toggle score/provenance breakdown overlay",
                TuiPaletteCategory::Explainability,
                None,
            ),
            TuiPaletteActionSpec::named(
                "config.reload",
                "Reload Configuration",
                "Reload fsfs config and apply supported hot settings",
                TuiPaletteCategory::Configuration,
                Some("Ctrl+R"),
            ),
            TuiPaletteActionSpec::named(
                "ops.open_timeline",
                "Open Ops Timeline",
                "Jump to timeline-focused operations view",
                TuiPaletteCategory::Operations,
                None,
            ),
            TuiPaletteActionSpec::named(
                REPLAY_TRACE_ACTION_ID,
                "Replay Last Trace",
                "Replay last failing scenario using repro manifest metadata",
                TuiPaletteCategory::Diagnostics,
                None,
            ),
        ]
    }

    /// Build shared `CommandPalette` from fsfs action specs.
    #[must_use]
    pub fn build_palette(&self) -> CommandPalette {
        let mut palette = CommandPalette::new();
        for action in &self.actions {
            palette.register(action.to_palette_action());
        }
        palette
    }

    /// Validate action IDs and navigation routing targets.
    ///
    /// # Errors
    ///
    /// Returns [`TuiModelValidationError`] when action IDs are duplicated,
    /// or when navigation actions have missing/unknown target screens.
    pub fn validate(&self, navigation: &TuiNavigationModel) -> Result<(), TuiModelValidationError> {
        let mut ids = BTreeSet::new();
        for action in &self.actions {
            if !ids.insert(action.id.clone()) {
                return Err(TuiModelValidationError::DuplicatePaletteActionId(
                    action.id.clone(),
                ));
            }
        }

        let screen_set = navigation
            .screen_order
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        for action in &self.actions {
            if action.category != TuiPaletteCategory::Navigation {
                continue;
            }
            let Some(target) = action.target_screen else {
                return Err(TuiModelValidationError::NavigationActionMissingTarget(
                    action.id.clone(),
                ));
            };
            if !screen_set.contains(&target) {
                return Err(TuiModelValidationError::NavigationActionUnknownTarget(
                    action.id.clone(),
                ));
            }
        }

        Ok(())
    }
}

/// Canonical card-level interaction primitives ported from showcase UX grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FsfsCardPrimitive {
    QueryToolbar,
    ResultListVirtualized,
    ResultDetailPanel,
    IndexingQueuePanel,
    IndexingBacklogChart,
    IndexingThroughputChart,
    DegradationBannerStrip,
    IndexingControlPanel,
    PressureBudgetPanel,
    PressureIndicatorStrip,
    ExplainabilityEvidencePanel,
    OpsTimelineStream,
    ConfigInspectorPanel,
    StatusFooter,
}

/// Per-screen card/layout contract for the deluxe TUI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiScreenLayoutContract {
    pub screen: FsfsScreen,
    pub cards: Vec<FsfsCardPrimitive>,
    pub primary_focus_card: FsfsCardPrimitive,
}

impl TuiScreenLayoutContract {
    #[must_use]
    pub fn defaults() -> Vec<Self> {
        vec![
            Self {
                screen: FsfsScreen::Search,
                cards: vec![
                    FsfsCardPrimitive::QueryToolbar,
                    FsfsCardPrimitive::ResultListVirtualized,
                    FsfsCardPrimitive::ResultDetailPanel,
                    FsfsCardPrimitive::StatusFooter,
                ],
                primary_focus_card: FsfsCardPrimitive::QueryToolbar,
            },
            Self {
                screen: FsfsScreen::Indexing,
                cards: vec![
                    FsfsCardPrimitive::IndexingQueuePanel,
                    FsfsCardPrimitive::IndexingBacklogChart,
                    FsfsCardPrimitive::IndexingThroughputChart,
                    FsfsCardPrimitive::DegradationBannerStrip,
                    FsfsCardPrimitive::IndexingControlPanel,
                    FsfsCardPrimitive::StatusFooter,
                ],
                primary_focus_card: FsfsCardPrimitive::IndexingQueuePanel,
            },
            Self {
                screen: FsfsScreen::Pressure,
                cards: vec![
                    FsfsCardPrimitive::PressureIndicatorStrip,
                    FsfsCardPrimitive::PressureBudgetPanel,
                    FsfsCardPrimitive::DegradationBannerStrip,
                    FsfsCardPrimitive::IndexingControlPanel,
                    FsfsCardPrimitive::StatusFooter,
                ],
                primary_focus_card: FsfsCardPrimitive::PressureIndicatorStrip,
            },
            Self {
                screen: FsfsScreen::Explainability,
                cards: vec![
                    FsfsCardPrimitive::ResultListVirtualized,
                    FsfsCardPrimitive::ExplainabilityEvidencePanel,
                    FsfsCardPrimitive::StatusFooter,
                ],
                primary_focus_card: FsfsCardPrimitive::ExplainabilityEvidencePanel,
            },
            Self {
                screen: FsfsScreen::Configuration,
                cards: vec![
                    FsfsCardPrimitive::ConfigInspectorPanel,
                    FsfsCardPrimitive::StatusFooter,
                ],
                primary_focus_card: FsfsCardPrimitive::ConfigInspectorPanel,
            },
            Self {
                screen: FsfsScreen::OpsTimeline,
                cards: vec![
                    FsfsCardPrimitive::OpsTimelineStream,
                    FsfsCardPrimitive::StatusFooter,
                ],
                primary_focus_card: FsfsCardPrimitive::OpsTimelineStream,
            },
        ]
    }
}

/// Normalized action-intent taxonomy for command-palette semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuiPaletteIntent {
    NavigateScreen,
    FocusQueryInput,
    SubmitSearchQuery,
    ClearSearchQuery,
    RepeatLastSearch,
    ToggleInlineExplainability,
    OpenSelectedSearchResult,
    JumpToSelectedSource,
    PauseIndexing,
    ResumeIndexing,
    ThrottleIndexing,
    RecoverIndexing,
    SetOverrideAuto,
    ForceOverrideFull,
    ForceOverrideEmbedDeferred,
    ForceOverrideLexicalOnly,
    ForceOverrideMetadataOnly,
    ForceOverridePaused,
    ToggleExplainabilityPanel,
    ReloadConfiguration,
    OpenOpsTimeline,
    ReplayDiagnostics,
    Unknown,
}

/// Explicit semantic contract for one palette action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiPaletteIntentSpec {
    pub action_id: String,
    pub intent: TuiPaletteIntent,
    pub target_screen: Option<FsfsScreen>,
    pub preserves_context: bool,
}

/// Full palette-intent mapping model used for cross-screen semantics checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiPaletteIntentModel {
    pub intents: Vec<TuiPaletteIntentSpec>,
}

impl TuiPaletteIntentModel {
    #[must_use]
    pub fn from_palette(palette: &TuiPaletteModel) -> Self {
        let intents = palette
            .actions
            .iter()
            .map(|action| TuiPaletteIntentSpec {
                action_id: action.id.clone(),
                intent: Self::intent_for_action(action),
                target_screen: action.target_screen,
                preserves_context: !matches!(action.id.as_str(), "config.reload"),
            })
            .collect();
        Self { intents }
    }

    fn intent_for_action(action: &TuiPaletteActionSpec) -> TuiPaletteIntent {
        if action.category == TuiPaletteCategory::Navigation {
            return TuiPaletteIntent::NavigateScreen;
        }
        match action.id.as_str() {
            "search.focus_query" => TuiPaletteIntent::FocusQueryInput,
            "search.submit_query" => TuiPaletteIntent::SubmitSearchQuery,
            "search.clear_query" => TuiPaletteIntent::ClearSearchQuery,
            "search.repeat_last" => TuiPaletteIntent::RepeatLastSearch,
            "search.toggle_explain" => TuiPaletteIntent::ToggleInlineExplainability,
            "search.open_selected" => TuiPaletteIntent::OpenSelectedSearchResult,
            "search.jump_to_source" => TuiPaletteIntent::JumpToSelectedSource,
            "index.pause" => TuiPaletteIntent::PauseIndexing,
            "index.resume" => TuiPaletteIntent::ResumeIndexing,
            "index.throttle" => TuiPaletteIntent::ThrottleIndexing,
            "index.recover" => TuiPaletteIntent::RecoverIndexing,
            "index.override.auto" => TuiPaletteIntent::SetOverrideAuto,
            "index.override.full" => TuiPaletteIntent::ForceOverrideFull,
            "index.override.embed_deferred" => TuiPaletteIntent::ForceOverrideEmbedDeferred,
            "index.override.lexical_only" => TuiPaletteIntent::ForceOverrideLexicalOnly,
            "index.override.metadata_only" => TuiPaletteIntent::ForceOverrideMetadataOnly,
            "index.override.paused" => TuiPaletteIntent::ForceOverridePaused,
            "explain.toggle_panel" => TuiPaletteIntent::ToggleExplainabilityPanel,
            "config.reload" => TuiPaletteIntent::ReloadConfiguration,
            "ops.open_timeline" => TuiPaletteIntent::OpenOpsTimeline,
            REPLAY_TRACE_ACTION_ID => TuiPaletteIntent::ReplayDiagnostics,
            _ => TuiPaletteIntent::Unknown,
        }
    }

    /// Validate that every palette action has an explicit intent mapping.
    ///
    /// # Errors
    ///
    /// Returns [`TuiModelValidationError`] when action IDs are duplicated,
    /// missing from the mapping, or mapped to `Unknown`.
    pub fn validate(&self, palette: &TuiPaletteModel) -> Result<(), TuiModelValidationError> {
        let palette_ids = palette
            .actions
            .iter()
            .map(|action| action.id.as_str())
            .collect::<BTreeSet<_>>();

        let mut seen = BTreeSet::new();
        for intent in &self.intents {
            if !seen.insert(intent.action_id.clone()) {
                return Err(TuiModelValidationError::DuplicatePaletteIntent(
                    intent.action_id.clone(),
                ));
            }
            if !palette_ids.contains(intent.action_id.as_str()) {
                return Err(TuiModelValidationError::PaletteIntentUnknownAction(
                    intent.action_id.clone(),
                ));
            }
            if intent.intent == TuiPaletteIntent::Unknown {
                return Err(TuiModelValidationError::MissingPaletteIntent(
                    intent.action_id.clone(),
                ));
            }
            if intent.intent == TuiPaletteIntent::NavigateScreen && intent.target_screen.is_none() {
                return Err(TuiModelValidationError::NavigationActionMissingTarget(
                    intent.action_id.clone(),
                ));
            }
        }

        for action in &palette.actions {
            if !seen.contains(action.id.as_str()) {
                return Err(TuiModelValidationError::MissingPaletteIntent(
                    action.id.clone(),
                ));
            }
        }

        Ok(())
    }
}

/// Deterministic checkpoint trigger for replay/snapshot serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuiSerializationTrigger {
    FrameCommitted,
    PaletteActionApplied,
    SearchPhaseTransition,
    DegradationTransition,
}

/// Serializable state boundary used by replay/snapshot suites.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiStateSerializationPoint {
    pub id: String,
    pub screen: FsfsScreen,
    pub trigger: TuiSerializationTrigger,
    pub state_keys: Vec<String>,
}

impl TuiStateSerializationPoint {
    #[must_use]
    pub fn defaults() -> Vec<Self> {
        vec![
            Self {
                id: "search.frame.commit".to_owned(),
                screen: FsfsScreen::Search,
                trigger: TuiSerializationTrigger::FrameCommitted,
                state_keys: vec![
                    "query".to_owned(),
                    "filters".to_owned(),
                    "cursor".to_owned(),
                    "phase".to_owned(),
                    "interaction_id".to_owned(),
                    "visible_window".to_owned(),
                    "frame_budget_ms".to_owned(),
                    "latency_bucket".to_owned(),
                ],
            },
            Self {
                id: "indexing.degradation.transition".to_owned(),
                screen: FsfsScreen::Indexing,
                trigger: TuiSerializationTrigger::DegradationTransition,
                state_keys: vec![
                    "queue_state".to_owned(),
                    "degradation_stage".to_owned(),
                    "degradation_banner".to_owned(),
                    "transition_reason_code".to_owned(),
                    "override_mode".to_owned(),
                    "override_allowed".to_owned(),
                ],
            },
            Self {
                id: "pressure.palette.action".to_owned(),
                screen: FsfsScreen::Pressure,
                trigger: TuiSerializationTrigger::PaletteActionApplied,
                state_keys: vec!["active_profile".to_owned(), "budget_snapshot".to_owned()],
            },
            Self {
                id: "explainability.search.phase".to_owned(),
                screen: FsfsScreen::Explainability,
                trigger: TuiSerializationTrigger::SearchPhaseTransition,
                state_keys: vec![
                    "query_id".to_owned(),
                    "hit_focus".to_owned(),
                    "rank_movements".to_owned(),
                ],
            },
            Self {
                id: "configuration.frame.commit".to_owned(),
                screen: FsfsScreen::Configuration,
                trigger: TuiSerializationTrigger::FrameCommitted,
                state_keys: vec![
                    "effective_config_hash".to_owned(),
                    "theme_density".to_owned(),
                ],
            },
            Self {
                id: "timeline.frame.commit".to_owned(),
                screen: FsfsScreen::OpsTimeline,
                trigger: TuiSerializationTrigger::FrameCommitted,
                state_keys: vec![
                    "timeline_cursor".to_owned(),
                    "severity_filter".to_owned(),
                    "stream_lag_ms".to_owned(),
                ],
            },
        ]
    }

    /// Validate ID uniqueness and per-screen serialization coverage.
    ///
    /// # Errors
    ///
    /// Returns [`TuiModelValidationError`] when IDs collide, state keys are
    /// empty, or a screen has no serialization point.
    pub fn validate(
        points: &[Self],
        navigation: &TuiNavigationModel,
    ) -> Result<(), TuiModelValidationError> {
        let mut ids = BTreeSet::new();
        for point in points {
            if !ids.insert(point.id.clone()) {
                return Err(TuiModelValidationError::DuplicateSerializationPoint(
                    point.id.clone(),
                ));
            }
            if point.state_keys.is_empty() {
                return Err(TuiModelValidationError::SerializationPointMissingStateKeys(
                    point.id.clone(),
                ));
            }
        }

        for screen in &navigation.screen_order {
            if !points.iter().any(|point| point.screen == *screen) {
                return Err(TuiModelValidationError::MissingSerializationPoint(
                    screen.id().to_owned(),
                ));
            }
        }

        Ok(())
    }
}

/// Interaction latency boundary checkpoints for budget conformance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuiLatencyBoundary {
    InputToIntent,
    IntentToState,
    StateToFrame,
}

/// Budget hook emitted at component boundaries for latency accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiLatencyBudgetHook {
    pub id: String,
    pub boundary: TuiLatencyBoundary,
    pub screen: Option<FsfsScreen>,
    pub budget_ms: u16,
    pub metric_key: String,
}

impl TuiLatencyBudgetHook {
    #[must_use]
    pub fn defaults() -> Vec<Self> {
        vec![
            Self {
                id: "global.input_to_intent".to_owned(),
                boundary: TuiLatencyBoundary::InputToIntent,
                screen: None,
                budget_ms: 8,
                metric_key: "tui.input_to_intent_ms".to_owned(),
            },
            Self {
                id: "global.intent_to_state".to_owned(),
                boundary: TuiLatencyBoundary::IntentToState,
                screen: None,
                budget_ms: 12,
                metric_key: "tui.intent_to_state_ms".to_owned(),
            },
            Self {
                id: "global.state_to_frame".to_owned(),
                boundary: TuiLatencyBoundary::StateToFrame,
                screen: None,
                budget_ms: 16,
                metric_key: "tui.state_to_frame_ms".to_owned(),
            },
            Self {
                id: "search.result_virtualization".to_owned(),
                boundary: TuiLatencyBoundary::StateToFrame,
                screen: Some(FsfsScreen::Search),
                budget_ms: 10,
                metric_key: "tui.search.virtualized_render_ms".to_owned(),
            },
            Self {
                id: "search.navigation.intent_to_state".to_owned(),
                boundary: TuiLatencyBoundary::IntentToState,
                screen: Some(FsfsScreen::Search),
                budget_ms: 6,
                metric_key: "tui.search.navigation_intent_to_state_ms".to_owned(),
            },
            Self {
                id: "search.inline_explain.state_to_frame".to_owned(),
                boundary: TuiLatencyBoundary::StateToFrame,
                screen: Some(FsfsScreen::Search),
                budget_ms: 8,
                metric_key: "tui.search.inline_explain_render_ms".to_owned(),
            },
            Self {
                id: "timeline.stream_update".to_owned(),
                boundary: TuiLatencyBoundary::IntentToState,
                screen: Some(FsfsScreen::OpsTimeline),
                budget_ms: 14,
                metric_key: "tui.timeline.stream_update_ms".to_owned(),
            },
        ]
    }

    /// Validate latency hook IDs, budgets, and metric keys.
    ///
    /// # Errors
    ///
    /// Returns [`TuiModelValidationError`] when IDs collide, budgets are zero,
    /// or metric keys are empty.
    pub fn validate(hooks: &[Self]) -> Result<(), TuiModelValidationError> {
        let mut ids = BTreeSet::new();
        for hook in hooks {
            if !ids.insert(hook.id.clone()) {
                return Err(TuiModelValidationError::DuplicateLatencyBudgetHook(
                    hook.id.clone(),
                ));
            }
            if hook.budget_ms == 0 {
                return Err(TuiModelValidationError::InvalidLatencyBudget(
                    hook.id.clone(),
                ));
            }
            if hook.metric_key.trim().is_empty() {
                return Err(TuiModelValidationError::MissingLatencyMetricKey(
                    hook.id.clone(),
                ));
            }
        }
        Ok(())
    }
}

/// Mapping from shared showcase surfaces to fsfs screens and palette routes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsfsShowcaseSurfaceMapping {
    pub surface: InteractionSurfaceKind,
    pub screen: FsfsScreen,
    pub palette_routes: Vec<PaletteIntentRoute>,
}

impl FsfsShowcaseSurfaceMapping {
    #[must_use]
    pub const fn new(
        surface: InteractionSurfaceKind,
        screen: FsfsScreen,
        palette_routes: Vec<PaletteIntentRoute>,
    ) -> Self {
        Self {
            surface,
            screen,
            palette_routes,
        }
    }
}

/// fsfs binding contract for showcase interaction primitives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsfsShowcasePortingSpec {
    pub interaction_spec: ShowcaseInteractionSpec,
    pub mappings: Vec<FsfsShowcaseSurfaceMapping>,
}

impl FsfsShowcasePortingSpec {
    #[must_use]
    pub fn from_shell(_shell: &FsfsTuiShellModel) -> Self {
        Self {
            interaction_spec: ShowcaseInteractionSpec::canonical(),
            mappings: vec![
                FsfsShowcaseSurfaceMapping::new(
                    InteractionSurfaceKind::Search,
                    FsfsScreen::Search,
                    vec![
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
                        PaletteIntentRoute::new(
                            PaletteIntent::ToggleExplainability,
                            "search.toggle_explain",
                            Some(InteractionSurfaceKind::Search),
                            false,
                        ),
                    ],
                ),
                FsfsShowcaseSurfaceMapping::new(
                    InteractionSurfaceKind::Results,
                    FsfsScreen::Search,
                    vec![PaletteIntentRoute::new(
                        PaletteIntent::NavigateSurface,
                        "nav.fsfs.search",
                        Some(InteractionSurfaceKind::Results),
                        false,
                    )],
                ),
                FsfsShowcaseSurfaceMapping::new(
                    InteractionSurfaceKind::Operations,
                    FsfsScreen::OpsTimeline,
                    vec![
                        PaletteIntentRoute::new(
                            PaletteIntent::OpenTimeline,
                            "ops.open_timeline",
                            Some(InteractionSurfaceKind::Operations),
                            false,
                        ),
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
                    ],
                ),
                FsfsShowcaseSurfaceMapping::new(
                    InteractionSurfaceKind::Explainability,
                    FsfsScreen::Explainability,
                    vec![
                        PaletteIntentRoute::new(
                            PaletteIntent::ToggleExplainability,
                            "explain.toggle_panel",
                            Some(InteractionSurfaceKind::Explainability),
                            false,
                        ),
                        PaletteIntentRoute::new(
                            PaletteIntent::ReplayTrace,
                            REPLAY_TRACE_ACTION_ID,
                            Some(InteractionSurfaceKind::Explainability),
                            true,
                        ),
                    ],
                ),
            ],
        }
    }

    /// Validate mapping coverage, routes, and deterministic interaction spec.
    ///
    /// # Errors
    ///
    /// Returns [`TuiModelValidationError`] when mappings are incomplete,
    /// route IDs are missing from fsfs palette actions, or screens are unknown.
    pub fn validate(&self, shell: &FsfsTuiShellModel) -> Result<(), TuiModelValidationError> {
        self.interaction_spec.validate().map_err(|err| {
            TuiModelValidationError::InvalidShowcaseInteractionSpec(err.to_string())
        })?;

        let mapped_surfaces = self
            .mappings
            .iter()
            .map(|mapping| mapping.surface)
            .collect::<BTreeSet<_>>();
        for surface in InteractionSurfaceKind::all() {
            if !mapped_surfaces.contains(&surface) {
                return Err(TuiModelValidationError::MissingShowcaseSurfaceMapping(
                    surface.id().to_owned(),
                ));
            }
        }

        let screen_set = shell
            .navigation
            .screen_order
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        for mapping in &self.mappings {
            if !screen_set.contains(&mapping.screen) {
                return Err(TuiModelValidationError::ShowcaseMappingUnknownScreen(
                    mapping.screen.id().to_owned(),
                ));
            }
        }

        let palette_ids = shell
            .palette
            .actions
            .iter()
            .map(|action| action.id.as_str())
            .collect::<BTreeSet<_>>();
        for mapping in &self.mappings {
            for route in &mapping.palette_routes {
                if !palette_ids.contains(route.action_id.as_str()) {
                    return Err(TuiModelValidationError::ShowcaseMappingMissingAction(
                        route.action_id.clone(),
                    ));
                }
            }
        }

        Ok(())
    }
}

/// Reusable shell contract for fsfs deluxe-TUI startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsfsTuiShellModel {
    pub settings: TuiAdapterSettings,
    pub navigation: TuiNavigationModel,
    pub keymap: TuiKeymapModel,
    pub palette: TuiPaletteModel,
    pub layout_contracts: Vec<TuiScreenLayoutContract>,
    pub palette_intents: TuiPaletteIntentModel,
    pub serialization_points: Vec<TuiStateSerializationPoint>,
    pub latency_budget_hooks: Vec<TuiLatencyBudgetHook>,
}

impl FsfsTuiShellModel {
    #[must_use]
    pub fn from_config(config: &FsfsConfig) -> Self {
        let settings = TuiAdapterSettings::from(config);
        let navigation = TuiNavigationModel::default();
        let keymap = TuiKeymapModel::default();
        let palette = TuiPaletteModel::from_navigation(&navigation);
        let layout_contracts = TuiScreenLayoutContract::defaults();
        let palette_intents = TuiPaletteIntentModel::from_palette(&palette);
        let serialization_points = TuiStateSerializationPoint::defaults();
        let latency_budget_hooks = TuiLatencyBudgetHook::defaults();
        Self {
            settings,
            navigation,
            keymap,
            palette,
            layout_contracts,
            palette_intents,
            serialization_points,
            latency_budget_hooks,
        }
    }

    /// Validate shell, navigation, keymap, and palette routing invariants.
    ///
    /// # Errors
    ///
    /// Returns [`TuiModelValidationError`] when any submodel fails
    /// deterministic validation.
    pub fn validate(&self) -> Result<(), TuiModelValidationError> {
        self.navigation.validate()?;
        self.keymap.validate()?;
        self.palette.validate(&self.navigation)?;
        Self::validate_layout_contracts(&self.layout_contracts, &self.navigation)?;
        self.palette_intents.validate(&self.palette)?;
        TuiStateSerializationPoint::validate(&self.serialization_points, &self.navigation)?;
        TuiLatencyBudgetHook::validate(&self.latency_budget_hooks)?;
        self.showcase_porting_spec().validate(self)?;
        Ok(())
    }

    /// Utility helper for callers that need concrete registry IDs.
    #[must_use]
    pub fn screen_registry_ids(&self) -> Vec<ScreenId> {
        self.navigation.screen_ids()
    }

    /// Build the shared-showcase mapping contract for fsfs screens.
    #[must_use]
    pub fn showcase_porting_spec(&self) -> FsfsShowcasePortingSpec {
        FsfsShowcasePortingSpec::from_shell(self)
    }

    fn validate_layout_contracts(
        layout_contracts: &[TuiScreenLayoutContract],
        navigation: &TuiNavigationModel,
    ) -> Result<(), TuiModelValidationError> {
        let mut seen = BTreeSet::new();
        for contract in layout_contracts {
            let screen_id = contract.screen.id().to_owned();
            if !seen.insert(contract.screen) {
                return Err(TuiModelValidationError::DuplicateLayoutContract(screen_id));
            }
            if !contract.cards.contains(&contract.primary_focus_card) {
                return Err(TuiModelValidationError::MissingLayoutFocusCard(screen_id));
            }
        }

        for screen in &navigation.screen_order {
            if !seen.contains(screen) {
                return Err(TuiModelValidationError::MissingLayoutContract(
                    screen.id().to_owned(),
                ));
            }
        }

        Ok(())
    }
}

/// Deterministic validation failures for TUI shell bootstrap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuiModelValidationError {
    EmptyScreenOrder,
    InvalidHistoryLimit(usize),
    DuplicateScreenId(String),
    MissingDefaultScreen(String),
    DuplicateKeyBinding(String),
    DuplicatePaletteActionId(String),
    NavigationActionMissingTarget(String),
    NavigationActionUnknownTarget(String),
    DuplicateLayoutContract(String),
    MissingLayoutContract(String),
    MissingLayoutFocusCard(String),
    DuplicatePaletteIntent(String),
    PaletteIntentUnknownAction(String),
    MissingPaletteIntent(String),
    DuplicateSerializationPoint(String),
    MissingSerializationPoint(String),
    SerializationPointMissingStateKeys(String),
    DuplicateLatencyBudgetHook(String),
    InvalidLatencyBudget(String),
    MissingLatencyMetricKey(String),
    InvalidShowcaseInteractionSpec(String),
    MissingShowcaseSurfaceMapping(String),
    ShowcaseMappingUnknownScreen(String),
    ShowcaseMappingMissingAction(String),
}

impl Display for TuiModelValidationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyScreenOrder => write!(f, "navigation screen order cannot be empty"),
            Self::InvalidHistoryLimit(limit) => {
                write!(f, "navigation history_limit must be > 0 (got {limit})")
            }
            Self::DuplicateScreenId(id) => write!(f, "duplicate screen id in order: {id}"),
            Self::MissingDefaultScreen(id) => {
                write!(f, "default screen is not present in screen order: {id}")
            }
            Self::DuplicateKeyBinding(binding) => write!(f, "duplicate key binding: {binding}"),
            Self::DuplicatePaletteActionId(id) => {
                write!(f, "duplicate command palette action id: {id}")
            }
            Self::NavigationActionMissingTarget(id) => {
                write!(f, "navigation action missing target screen: {id}")
            }
            Self::NavigationActionUnknownTarget(id) => {
                write!(f, "navigation action points to unknown screen: {id}")
            }
            Self::DuplicateLayoutContract(id) => {
                write!(f, "duplicate layout contract for screen: {id}")
            }
            Self::MissingLayoutContract(id) => {
                write!(f, "missing layout contract for screen: {id}")
            }
            Self::MissingLayoutFocusCard(id) => {
                write!(
                    f,
                    "layout contract focus card is not present in card list: {id}"
                )
            }
            Self::DuplicatePaletteIntent(id) => {
                write!(f, "duplicate palette intent mapping for action: {id}")
            }
            Self::PaletteIntentUnknownAction(id) => {
                write!(f, "palette intent references unknown action: {id}")
            }
            Self::MissingPaletteIntent(id) => {
                write!(f, "palette action has no explicit intent mapping: {id}")
            }
            Self::DuplicateSerializationPoint(id) => {
                write!(f, "duplicate serialization point id: {id}")
            }
            Self::MissingSerializationPoint(id) => {
                write!(f, "missing serialization point for screen: {id}")
            }
            Self::SerializationPointMissingStateKeys(id) => {
                write!(f, "serialization point missing state keys: {id}")
            }
            Self::DuplicateLatencyBudgetHook(id) => {
                write!(f, "duplicate latency budget hook id: {id}")
            }
            Self::InvalidLatencyBudget(id) => {
                write!(f, "latency budget hook has invalid budget: {id}")
            }
            Self::MissingLatencyMetricKey(id) => {
                write!(f, "latency budget hook missing metric key: {id}")
            }
            Self::InvalidShowcaseInteractionSpec(detail) => {
                write!(f, "invalid showcase interaction spec: {detail}")
            }
            Self::MissingShowcaseSurfaceMapping(surface) => {
                write!(f, "missing fsfs showcase mapping for surface: {surface}")
            }
            Self::ShowcaseMappingUnknownScreen(screen) => {
                write!(f, "showcase mapping references unknown screen: {screen}")
            }
            Self::ShowcaseMappingMissingAction(action_id) => {
                write!(
                    f,
                    "showcase mapping references missing palette action: {action_id}"
                )
            }
        }
    }
}

impl Error for TuiModelValidationError {}

#[cfg(test)]
mod tests {
    use frankensearch_tui::InteractionSurfaceKind;
    use std::collections::BTreeSet;

    use super::{
        ContextRetentionPolicy, FsfsCardPrimitive, FsfsScreen, FsfsTuiShellModel,
        REPLAY_TRACE_ACTION_ID, TuiAdapterSettings, TuiKeyBindingScope, TuiLatencyBoundary,
        TuiLatencyBudgetHook, TuiModelValidationError, TuiNavigationModel, TuiPaletteActionSpec,
        TuiPaletteCategory, TuiPaletteIntent, TuiStateSerializationPoint,
    };
    use crate::config::{Density, FsfsConfig, TuiTheme};
    use crate::repro::{ReplayClientSurface, ReplayEntrypoint};

    #[test]
    fn converts_from_resolved_config() {
        let mut config = FsfsConfig::default();
        config.tui.theme = TuiTheme::Light;
        config.tui.density = Density::Compact;
        config.tui.show_explanations = false;

        let settings = TuiAdapterSettings::from(&config);
        assert_eq!(settings.theme, TuiTheme::Light);
        assert_eq!(settings.density, Density::Compact);
        assert!(!settings.show_explanations);
    }

    #[test]
    fn default_config_produces_default_settings() {
        let config = FsfsConfig::default();
        let settings = TuiAdapterSettings::from(&config);
        assert_eq!(settings.theme, config.tui.theme);
        assert_eq!(settings.density, config.tui.density);
        assert_eq!(settings.show_explanations, config.tui.show_explanations);
    }

    #[test]
    fn settings_clone_is_independent() {
        let config = FsfsConfig::default();
        let settings = TuiAdapterSettings::from(&config);
        let cloned = settings.clone();
        assert_eq!(settings, cloned);
    }

    #[test]
    fn settings_debug_format() {
        let config = FsfsConfig::default();
        let settings = TuiAdapterSettings::from(&config);
        let debug = format!("{settings:?}");
        assert!(debug.contains("TuiAdapterSettings"));
    }

    #[test]
    fn navigation_wraps_in_tab_order() {
        let navigation = TuiNavigationModel::default();
        assert_eq!(
            navigation.next_screen(FsfsScreen::OpsTimeline),
            FsfsScreen::Search
        );
        assert_eq!(
            navigation.previous_screen(FsfsScreen::Search),
            FsfsScreen::OpsTimeline
        );
    }

    #[test]
    fn navigation_context_policy_is_explicit() {
        let navigation = TuiNavigationModel::default();
        assert_eq!(
            navigation.context_policy(FsfsScreen::Search, FsfsScreen::Explainability),
            ContextRetentionPolicy::PreserveAll
        );
        assert_eq!(
            navigation.context_policy(FsfsScreen::Search, FsfsScreen::Configuration),
            ContextRetentionPolicy::ResetTransient
        );
    }

    #[test]
    fn shell_model_is_valid_and_builds_palette_registry() {
        let shell = FsfsTuiShellModel::from_config(&FsfsConfig::default());
        shell.validate().expect("shell model should validate");

        let palette = shell.palette.build_palette();
        assert!(palette.len() >= shell.navigation.screen_order.len());

        let ids = shell.screen_registry_ids();
        assert_eq!(ids.len(), shell.navigation.screen_order.len());
        assert_eq!(ids[0].0, "fsfs.search");
    }

    #[test]
    fn validation_rejects_duplicate_palette_ids() {
        let mut shell = FsfsTuiShellModel::from_config(&FsfsConfig::default());
        shell.palette.actions.push(TuiPaletteActionSpec::named(
            "search.focus_query",
            "Duplicate",
            "duplicate id for test",
            TuiPaletteCategory::Search,
            None,
        ));

        let err = shell.validate().expect_err("duplicate IDs must fail");
        assert!(matches!(
            err,
            TuiModelValidationError::DuplicatePaletteActionId(_)
        ));
    }

    #[test]
    fn layout_contracts_cover_each_screen_with_focus_card() {
        let shell = FsfsTuiShellModel::from_config(&FsfsConfig::default());
        let screens = shell
            .layout_contracts
            .iter()
            .map(|contract| contract.screen)
            .collect::<BTreeSet<_>>();
        assert_eq!(screens.len(), shell.navigation.screen_order.len());
        for contract in &shell.layout_contracts {
            assert!(contract.cards.contains(&contract.primary_focus_card));
            assert!(contract.cards.contains(&FsfsCardPrimitive::StatusFooter));
        }
    }

    #[test]
    fn indexing_layout_includes_backlog_throughput_and_controls_cards() {
        let shell = FsfsTuiShellModel::from_config(&FsfsConfig::default());
        let indexing_contract = shell
            .layout_contracts
            .iter()
            .find(|contract| contract.screen == FsfsScreen::Indexing)
            .expect("indexing contract should exist");

        assert!(
            indexing_contract
                .cards
                .contains(&FsfsCardPrimitive::IndexingBacklogChart)
        );
        assert!(
            indexing_contract
                .cards
                .contains(&FsfsCardPrimitive::IndexingThroughputChart)
        );
        assert!(
            indexing_contract
                .cards
                .contains(&FsfsCardPrimitive::IndexingControlPanel)
        );
        assert!(
            indexing_contract
                .cards
                .contains(&FsfsCardPrimitive::DegradationBannerStrip)
        );
    }

    #[test]
    fn palette_includes_throttle_and_recover_controls() {
        let shell = FsfsTuiShellModel::from_config(&FsfsConfig::default());
        let action_ids = shell
            .palette
            .actions
            .iter()
            .map(|action| action.id.as_str())
            .collect::<BTreeSet<_>>();

        assert!(action_ids.contains("index.throttle"));
        assert!(action_ids.contains("index.recover"));
        assert!(action_ids.contains("index.override.auto"));
        assert!(action_ids.contains("index.override.lexical_only"));
        assert!(action_ids.contains("index.override.metadata_only"));
        assert!(action_ids.contains("index.override.paused"));
    }

    #[test]
    fn palette_includes_search_virtualization_and_explainability_actions() {
        let shell = FsfsTuiShellModel::from_config(&FsfsConfig::default());
        let action_ids = shell
            .palette
            .actions
            .iter()
            .map(|action| action.id.as_str())
            .collect::<BTreeSet<_>>();

        assert!(action_ids.contains("search.submit_query"));
        assert!(action_ids.contains("search.clear_query"));
        assert!(action_ids.contains("search.toggle_explain"));
        assert!(action_ids.contains("search.open_selected"));
        assert!(action_ids.contains("search.jump_to_source"));
    }

    #[test]
    fn keymap_includes_explicit_search_submit_chord() {
        let shell = FsfsTuiShellModel::from_config(&FsfsConfig::default());
        assert!(shell.keymap.bindings.iter().any(|binding| {
            binding.scope == TuiKeyBindingScope::Global
                && binding.chord == "Ctrl+Enter"
                && binding.action_id == "search.submit_query"
        }));
    }

    #[test]
    fn keymap_includes_search_navigation_chords() {
        let shell = FsfsTuiShellModel::from_config(&FsfsConfig::default());
        assert!(shell.keymap.bindings.iter().any(|binding| {
            binding.scope == TuiKeyBindingScope::Global
                && binding.chord == "Up"
                && binding.action_id == "search.select_up"
        }));
        assert!(shell.keymap.bindings.iter().any(|binding| {
            binding.scope == TuiKeyBindingScope::Global
                && binding.chord == "Down"
                && binding.action_id == "search.select_down"
        }));
        assert!(shell.keymap.bindings.iter().any(|binding| {
            binding.scope == TuiKeyBindingScope::Global
                && binding.chord == "PageUp"
                && binding.action_id == "search.page_up"
        }));
        assert!(shell.keymap.bindings.iter().any(|binding| {
            binding.scope == TuiKeyBindingScope::Global
                && binding.chord == "PageDown"
                && binding.action_id == "search.page_down"
        }));
    }

    #[test]
    fn keymap_includes_search_drilldown_chords() {
        let shell = FsfsTuiShellModel::from_config(&FsfsConfig::default());
        assert!(shell.keymap.bindings.iter().any(|binding| {
            binding.scope == TuiKeyBindingScope::Global
                && binding.chord == "Ctrl+E"
                && binding.action_id == "search.toggle_explain"
        }));
        assert!(shell.keymap.bindings.iter().any(|binding| {
            binding.scope == TuiKeyBindingScope::Global
                && binding.chord == "Ctrl+O"
                && binding.action_id == "search.open_selected"
        }));
        assert!(shell.keymap.bindings.iter().any(|binding| {
            binding.scope == TuiKeyBindingScope::Global
                && binding.chord == "Ctrl+J"
                && binding.action_id == "search.jump_to_source"
        }));
    }

    #[test]
    fn palette_intents_cover_all_palette_actions() {
        let shell = FsfsTuiShellModel::from_config(&FsfsConfig::default());
        assert_eq!(
            shell.palette_intents.intents.len(),
            shell.palette.actions.len()
        );
        assert!(
            shell
                .palette_intents
                .intents
                .iter()
                .all(|intent| intent.intent != TuiPaletteIntent::Unknown)
        );
    }

    #[test]
    fn serialization_points_and_latency_hooks_validate_defaults() {
        let shell = FsfsTuiShellModel::from_config(&FsfsConfig::default());
        TuiStateSerializationPoint::validate(&shell.serialization_points, &shell.navigation)
            .expect("default serialization points should be valid");
        TuiLatencyBudgetHook::validate(&shell.latency_budget_hooks)
            .expect("default latency hooks should be valid");
    }

    #[test]
    fn latency_hooks_include_search_navigation_and_inline_explain_metrics() {
        let shell = FsfsTuiShellModel::from_config(&FsfsConfig::default());

        let navigation_hook = shell
            .latency_budget_hooks
            .iter()
            .find(|hook| hook.id == "search.navigation.intent_to_state")
            .expect("search navigation latency hook must exist");
        assert_eq!(navigation_hook.boundary, TuiLatencyBoundary::IntentToState);
        assert_eq!(navigation_hook.screen, Some(FsfsScreen::Search));
        assert_eq!(navigation_hook.budget_ms, 6);
        assert_eq!(
            navigation_hook.metric_key,
            "tui.search.navigation_intent_to_state_ms"
        );

        let explain_hook = shell
            .latency_budget_hooks
            .iter()
            .find(|hook| hook.id == "search.inline_explain.state_to_frame")
            .expect("search inline explain latency hook must exist");
        assert_eq!(explain_hook.boundary, TuiLatencyBoundary::StateToFrame);
        assert_eq!(explain_hook.screen, Some(FsfsScreen::Search));
        assert_eq!(explain_hook.budget_ms, 8);
        assert_eq!(
            explain_hook.metric_key,
            "tui.search.inline_explain_render_ms"
        );
    }

    #[test]
    fn degradation_serialization_point_includes_override_audit_keys() {
        let shell = FsfsTuiShellModel::from_config(&FsfsConfig::default());
        let point = shell
            .serialization_points
            .iter()
            .find(|point| point.id == "indexing.degradation.transition")
            .expect("indexing degradation serialization point must exist");

        let keys = point
            .state_keys
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        assert!(keys.contains("degradation_banner"));
        assert!(keys.contains("transition_reason_code"));
        assert!(keys.contains("override_mode"));
        assert!(keys.contains("override_allowed"));
    }

    #[test]
    fn search_serialization_point_includes_interaction_telemetry_keys() {
        let shell = FsfsTuiShellModel::from_config(&FsfsConfig::default());
        let point = shell
            .serialization_points
            .iter()
            .find(|point| point.id == "search.frame.commit")
            .expect("search frame serialization point must exist");

        let keys = point
            .state_keys
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        assert!(keys.contains("interaction_id"));
        assert!(keys.contains("visible_window"));
        assert!(keys.contains("frame_budget_ms"));
        assert!(keys.contains("latency_bucket"));
    }

    #[test]
    fn validation_rejects_missing_layout_contract() {
        let mut shell = FsfsTuiShellModel::from_config(&FsfsConfig::default());
        shell
            .layout_contracts
            .retain(|contract| contract.screen != FsfsScreen::Pressure);

        let err = shell
            .validate()
            .expect_err("missing screen layout contract must fail");
        assert!(matches!(
            err,
            TuiModelValidationError::MissingLayoutContract(id) if id == "fsfs.pressure"
        ));
    }

    #[test]
    fn validation_rejects_unknown_palette_intent_mapping() {
        let mut shell = FsfsTuiShellModel::from_config(&FsfsConfig::default());
        let first_action = shell.palette.actions[0].id.clone();
        let first_intent = shell
            .palette_intents
            .intents
            .iter_mut()
            .find(|intent| intent.action_id == first_action)
            .expect("intent mapping should exist");
        first_intent.intent = TuiPaletteIntent::Unknown;

        let err = shell
            .validate()
            .expect_err("unknown intent mapping must fail");
        assert!(matches!(
            err,
            TuiModelValidationError::MissingPaletteIntent(id) if id == first_action
        ));
    }

    #[test]
    fn showcase_porting_spec_maps_all_required_surfaces() {
        let shell = FsfsTuiShellModel::from_config(&FsfsConfig::default());
        let spec = shell.showcase_porting_spec();
        spec.validate(&shell)
            .expect("default showcase mapping should validate");

        let mapped_surfaces = spec
            .mappings
            .iter()
            .map(|mapping| mapping.surface)
            .collect::<BTreeSet<_>>();
        for required_surface in InteractionSurfaceKind::all() {
            assert!(mapped_surfaces.contains(&required_surface));
        }
    }

    #[test]
    fn replay_entrypoint_action_id_matches_tui_palette_contract() {
        let shell = FsfsTuiShellModel::from_config(&FsfsConfig::default());
        let entrypoint = ReplayEntrypoint {
            trace_id: "trace-1".to_owned(),
            client_surface: ReplayClientSurface::Tui,
            manifest_path: Some("manifest.json".to_owned()),
            artifact_root: None,
            start_frame_seq: None,
            end_frame_seq: None,
            strict_reason_codes: false,
        };
        let action_id = entrypoint.tui_action_id();

        assert_eq!(action_id, REPLAY_TRACE_ACTION_ID);
        assert!(
            shell
                .palette
                .actions
                .iter()
                .any(|action| action.id == action_id)
        );
        assert!(shell.palette_intents.intents.iter().any(|intent| {
            intent.action_id == action_id && intent.intent == TuiPaletteIntent::ReplayDiagnostics
        }));
    }

    #[test]
    fn showcase_mapping_validation_rejects_missing_required_action() {
        let mut shell = FsfsTuiShellModel::from_config(&FsfsConfig::default());
        shell
            .palette
            .actions
            .retain(|action| action.id != REPLAY_TRACE_ACTION_ID);

        let spec = shell.showcase_porting_spec();
        let err = spec
            .validate(&shell)
            .expect_err("missing required showcase action should fail");
        assert!(matches!(
            err,
            TuiModelValidationError::ShowcaseMappingMissingAction(id)
                if id == REPLAY_TRACE_ACTION_ID
        ));
    }
}
