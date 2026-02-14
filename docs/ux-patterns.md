# UX Pattern Extraction for frankensearch Ops TUI

Issue: `bd-2yu.1.1`  
Source audited: `/dp/frankentui` (`ftui-demo-showcase` + `ftui-widgets`)

## Scope

This document extracts reusable advanced UX patterns from `ftui-demo-showcase` for the frankensearch operations console.

Required coverage (all included below):

1. Dashboard tile composition + drilldown
2. Action timeline/event stream
3. Performance HUD
4. Explainability/evidence ledger presentation
5. Command palette ergonomics
6. Metadata-driven screen registry/navigation
7. Status bar chrome
8. Accessibility controls
9. Deterministic replay mode
10. Virtualized large-list handling
11. Sparkline/mini-chart widgets
12. Alert/notification toasts

## Pattern Matrix

| # | Pattern | Source File References | Screenshot Description | Reusability | frankensearch Ops Mapping | Deterministic Reproduction Notes | Anti-Pattern To Avoid |
|---|---|---|---|---|---|---|---|
| 1 | Dashboard tile composition + drilldown | `/dp/frankentui/crates/ftui-demo-showcase/src/screens/dashboard.rs:5496` `/dp/frankentui/crates/ftui-demo-showcase/src/app.rs:4150` `/dp/frankentui/crates/ftui-demo-showcase/src/app.rs:5722` | A KPI tile grid where each tile has a hover/click region and selecting a tile jumps to the relevant detail screen. | **Adapt** (hit registration + click dispatch can be reused directly; tile content is domain-specific). | Top tiles for query throughput, index freshness, quality-tier health, and error budget; click to open lexical/vector/explainability screens. | Run with `FTUI_DEMO_DETERMINISTIC=1` and fixed tick settings before snapshot capture so pane-hit ids and click behavior are stable. | Hard-coding pointer coordinates for navigation instead of registered hit regions. |
| 2 | Action timeline / event stream | `/dp/frankentui/crates/ftui-demo-showcase/src/screens/action_timeline.rs:1` | Scrollable timeline with severity/component tags, filter controls, follow mode, and bounded history. | **Copy/Adapt** (bounded queue + filter primitives copy; event schema adapts). | Stream query lifecycle, embedder choice, candidate fusion, rerank, cache hits/misses, and failure reasons. | Use deterministic mode and replay the same event fixture to assert ordering, filtering, and clipping behavior in snapshots. | Unbounded timeline growth or event rows without typed severity/kind metadata. |
| 3 | Performance HUD with degradation tiers | `/dp/frankentui/crates/ftui-demo-showcase/src/screens/performance_hud.rs:1` `/dp/frankentui/crates/ftui-demo-showcase/src/app.rs:2620` | Real-time FPS/latency sparkline with clear tier labels (normal/degraded/stressed) and stress-mode indicators. | **Adapt** (calculation/render pattern reusable; metrics names differ). | Overlay p50/p95/p99 latency, quality-phase budget burn, and system pressure to drive adaptive mode choices. | Enable deterministic mode and fixed tick interval so sampled windows and trend lines are reproducible. | Logging raw per-frame metrics without stable windowing/tier classification. |
| 4 | Explainability cockpit / evidence ledger | `/dp/frankentui/crates/ftui-demo-showcase/src/screens/explainability_cockpit.rs:1` `/dp/frankentui/crates/ftui-demo-showcase/src/app.rs:3240` | Side-panel cockpit showing evidence entries over time, reason kinds, and selected-event details. | **Adapt** (ledger model and panel layout reusable; evidence payloads are frankensearch-specific). | Show per-hit score decomposition (BM25/vector/RRF/blend/rerank), skip reasons, and ranking movement across phases. | Export deterministic evidence JSONL with seed/tick metadata; replay the same fixture and compare snapshots + log hashes. | Mixing explainability widgets into the primary dashboard layout instead of isolating as a focused cockpit. |
| 5 | Command palette (fuzzy, categorized actions) | `/dp/frankentui/crates/ftui-widgets/src/command_palette/mod.rs:1` `/dp/frankentui/crates/ftui-demo-showcase/src/app.rs:2990` `/dp/frankentui/crates/ftui-demo-showcase/src/app.rs:4670` | Overlay palette with typeahead, grouped actions, keyboard navigation, and execution telemetry. | **Copy/Adapt** (core widget reusable; action inventory adapts). | Jump to screens, trigger diagnostic captures, run index health checks, toggle fast-only mode, open replay sessions. | Capture deterministic keystroke fixtures and assert sorted action matches + selected command outcomes. | Per-screen ad-hoc command menus with divergent bindings and no centralized action metadata. |
| 6 | Metadata-driven screen registry and navigation | `/dp/frankentui/crates/ftui-demo-showcase/src/screens/mod.rs:100` `/dp/frankentui/crates/ftui-demo-showcase/src/app.rs:2990` `/dp/frankentui/crates/ftui-demo-showcase/src/chrome.rs:80` | Single registry defines screen id/title/category/tags used by tabs, palette, and hit-target routing. | **Copy** (architecture pattern is directly transferable). | Central registry for `overview`, `query-stream`, `index-health`, `fusion-diagnostics`, `evidence`, `alerts`, etc. | Keep registry order static and run deterministic startup snapshots to verify consistent tab/palette ordering. | Duplicating screen metadata across tabs, keymaps, and palette code paths. |
| 7 | Status bar chrome (left/center/right segments) | `/dp/frankentui/crates/ftui-demo-showcase/src/chrome.rs:560` `/dp/frankentui/crates/ftui-demo-showcase/src/app.rs:3984` | Dense status strip with screen identity, timing, toggle states, and contextual hints without crowding content panes. | **Adapt** (layout pattern reusable; indicator semantics adapt). | Show active index set, current query class, deterministic/replay flag, pressure profile, and quick-toggle hints. | Snapshot with fixed tick and static screen size to make segment truncation/spacing deterministic. | Free-form concatenated status text that shifts width unpredictably across screens. |
| 8 | Accessibility panel + telemetry hooks | `/dp/frankentui/crates/ftui-demo-showcase/src/screens/accessibility_panel.rs:1` `/dp/frankentui/crates/ftui-demo-showcase/src/app.rs:2870` | Overlay for contrast/motion/text controls plus live accessibility event log and score indicators. | **Adapt** (panel behavior reusable; accessible design tokens adapt). | Runtime toggles for high contrast, reduced motion, larger text density, and keyboard-only navigation hints. | Execute deterministic toggle scripts and assert both visual snapshots and emitted a11y telemetry events. | Theme overrides scattered across screens without a centralized accessibility control plane. |
| 9 | Deterministic replay mode contract | `/dp/frankentui/crates/ftui-demo-showcase/src/determinism.rs:24` `/dp/frankentui/crates/ftui-demo-showcase/src/app.rs:2539` `/dp/frankentui/crates/ftui-demo-showcase/src/screens/determinism_lab.rs:540` | A dedicated deterministic mode exposing seed/tick controls and replay metadata that make UI behavior reproducible. | **Copy/Adapt** (env contract and wiring can be mirrored with frankensearch naming). | Deterministic incident replay for ranking regressions and control-plane anomalies using recorded query/event fixtures. | Pin deterministic env vars (`*_DETERMINISTIC`, tick ms, seeds), replay fixture, and compare hash-keyed artifacts. | Allowing wall-clock timing or uncontrolled RNG to influence snapshots and replay outputs. |
| 10 | Virtualized search/list rendering | `/dp/frankentui/crates/ftui-demo-showcase/src/screens/virtualized_search.rs:610` `/dp/frankentui/crates/ftui-demo-showcase/src/screens/virtualized_search.rs:1866` `/dp/frankentui/crates/ftui-widgets/src/virtualized.rs:1` | Large result sets render smoothly with overscan, retained selection state, and diagnostics for scroll/filter behavior. | **Copy/Adapt** (core virtualized widget reusable; row rendering adapts). | Efficiently render large candidate/result lists (10k+) for document hits, query history, and event records. | Replay deterministic key-scroll scripts and assert row windows, selection state, and diagnostics counters. | Rendering full datasets eagerly or coupling scroll behavior to non-deterministic frame timing. |
| 11 | Sparkline + mini-bar microvisualizations | `/dp/frankentui/crates/ftui-demo-showcase/src/screens/dashboard.rs:4214` `/dp/frankentui/crates/ftui-widgets/src/sparkline.rs:127` `/dp/frankentui/crates/ftui-widgets/src/progress.rs:247` | Compact inline trend and threshold bars embedded in KPI cards for quick directional reads. | **Copy** (widget primitives are general-purpose and stable). | Show query rate trends, stale-index ratio, quality-pass rate, queue depth, and error budget burn in cards. | Feed fixed metric fixtures and compare rendered widget strings/snapshots in deterministic mode. | Hand-rolled chart glyphs per screen causing inconsistent scaling and style drift. |
| 12 | Alert/notification toast system | `/dp/frankentui/crates/ftui-demo-showcase/src/screens/notifications.rs:1` | Severity-colored toast queue with actions, queue caps, and dismiss behavior for actionable notifications. | **Copy/Adapt** (queue mechanics copy; alert taxonomy adapts). | Notify on model-load failures, stale-index rebuild triggers, degraded mode entry, and recovery completion. | Use deterministic event scripts to validate enqueue order, max-visible limits, and dismiss actions. | Unlimited toast stacking or blocking UI interaction while notifications are displayed. |

## Category-Level Anti-Patterns

At least one anti-pattern per major category:

| Category | Anti-Pattern |
|---|---|
| Dashboard/Drilldown | Coordinate-based navigation without hit registration metadata. |
| Timeline | Unbounded append-only logs with no severity/taxonomy tags. |
| Performance HUD | Raw counters without rolling windows/tier semantics. |
| Explainability | Mixing evidence details into main layout instead of a dedicated cockpit. |
| Command Palette | Multiple disconnected command systems per screen. |
| Navigation Registry | Duplicated screen metadata across tabs/palette/mouse routing. |
| Status Bar | Free-text, width-unstable status rendering with no structured fields. |
| Accessibility | Theme/accessibility state scattered and non-telemetered. |
| Determinism | Wall-clock and RNG variance left enabled in snapshot/replay flows. |
| Virtualization | Full-list rendering for large datasets. |
| Microcharts | Bespoke chart logic in each screen rather than shared widgets. |
| Notifications | Uncapped toast queues and missing dismiss/action semantics. |

## Suggested Reuse Policy

- **Copy verbatim:** screen registry pattern, virtualized widget foundation, sparkline/mini-bar primitives, notification queue policy.
- **Adapt carefully:** dashboard tile mappings, performance HUD metric set, explainability evidence schema, status bar fields, accessibility toggles.
- **Inspire only:** cosmetic theme specifics; keep frankensearch visual language domain-specific.

## Deterministic Audit Harness (Source Project)

Use the source showcase in deterministic mode when revalidating extracted patterns:

```bash
cd /dp/frankentui
FTUI_DEMO_DETERMINISTIC=1 \
FTUI_DEMO_TICK_MS=16 \
FTUI_DEMO_SEED=42 \
cargo run -p ftui-demo-showcase
```

Capture:

1. Snapshot images for each pattern lane.
2. Corresponding JSONL telemetry/event artifacts.
3. Seed/tick/env metadata in artifact headers.

This ensures downstream frankensearch pattern adoption can be snapshot-tested without ambiguity.
