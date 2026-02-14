# frankensearch Ops TUI IA + Navigation Contract

Issue: `bd-2yu.1.2`  
Depends on: `bd-2yu.1.1` (pattern extraction complete)

## Contract Goal

This document is the implementation contract for downstream screen/workflow beads.  
If a behavior is not defined here, it is out of scope for implementation and must be added to this spec first.

## Final Screen Registry

Screen registry fields are mandatory for every screen:

- `id` (stable machine key)
- `title` (operator label)
- `category` (`overview`, `operations`, `diagnostics`, `evidence`, `system`)
- `default_hotkey` (single-key mnemonic or chord)
- `purpose` (decision this screen supports)
- `primary_widgets` (must-have visual primitives)
- `input_contract` (required context args)
- `drilldowns` (allowed outbound navigation targets)

### Required Screens (Finalized)

| id | title | category | default_hotkey | purpose | primary_widgets | input_contract | drilldowns |
|---|---|---|---|---|---|---|---|
| `fleet_overview` | Fleet Overview | `overview` | `1` | Triage all detected instances and spot unhealthy projects quickly. | KPI tile grid, status sparkline strip, instance table. | none | `project_dashboard`, `alerts_timeline`, `resource_trends` |
| `project_dashboard` | Project Detail Dashboard | `overview` | `2` | Understand one project’s hybrid search health at a glance. | health tiles, phase latency bars, top anomaly cards. | `project_id` | `live_search_stream`, `index_embed_progress`, `explainability_cockpit`, `historical_analytics` |
| `live_search_stream` | Live Search Stream | `operations` | `3` | Observe current query traffic and phase transitions in real time. | virtualized event list, severity filter chips, rate sparkline. | `project_id` | `explainability_cockpit`, `alerts_timeline` |
| `index_embed_progress` | Index + Embedding Progress | `operations` | `4` | Monitor indexing/embedding queue throughput and staleness. | queue depth chart, batch latency bars, progress timeline. | `project_id` | `resource_trends`, `historical_analytics` |
| `resource_trends` | Resource Trends (CPU/Mem/IO) | `operations` | `5` | Determine whether host pressure is causing degraded search behavior. | multi-window charts (1m/15m/1h/6h/24h/3d/1w), threshold overlays. | `project_id` | `alerts_timeline`, `project_dashboard` |
| `historical_analytics` | Historical Analytics | `diagnostics` | `6` | Compare performance and quality over time windows for regressions. | window selector, percentile trend charts, anomaly timeline. | `project_id`, `time_window` | `explainability_cockpit`, `alerts_timeline` |
| `alerts_timeline` | Alerts + Timeline | `diagnostics` | `7` | Investigate incident chronology and active SLO/error-budget pressure. | alert queue, incident timeline, severity counters. | `project_id` optional | `explainability_cockpit`, `project_dashboard` |
| `explainability_cockpit` | Explainability Cockpit | `evidence` | `8` | Explain why rankings changed and which signals drove outcomes. | per-hit decomposition table, evidence ledger timeline, rank-movement panel. | `project_id`, `query_id` optional | `live_search_stream`, `historical_analytics` |
| `command_center` | Command Center (Palette + Actions) | `system` | `Ctrl+K` | Execute cross-screen actions and jump paths without leaving context. | command palette, favorites, action history. | current global context | any screen |
| `operator_settings` | Operator Settings + A11y | `system` | `9` | Configure density/theme/accessibility and deterministic replay controls. | toggles panel, keymap reference, replay controls. | none | returns to prior screen |

## Global Navigation and Focus Model

## Navigation Layers (priority order)

1. `palette` (if open)
2. `modal_overlay` (help/alerts/explainability deep panel)
3. `status_chrome` (toggle hit regions)
4. `tab/category chrome`
5. `active screen content`

Lower layers do not receive input until higher active layers are dismissed.

## Global Keybindings (fixed)

| Key | Action | Scope |
|---|---|---|
| `Ctrl+K` | Open/close command palette | global |
| `Esc` | Close topmost overlay; if none, clear screen-local focus | global |
| `Tab` / `Shift+Tab` | Cycle focus groups within active layer | global |
| `1..9` | Jump to default screen hotkeys | global |
| `[` / `]` | Previous/next screen in current category order | global |
| `?` | Open keybinding/help overlay | global |
| `Ctrl+P` | Toggle performance HUD overlay | global |
| `Ctrl+A` | Toggle accessibility/settings overlay | global |
| `Ctrl+M` | Toggle mouse capture mode | global |
| `Ctrl+R` | Trigger reconnect flow for current project stream | stream-backed screens |
| `Enter` | Activate focused item or drilldown | focused element |

## Mouse Hit Region Contract

All clickable UI elements must register stable region IDs:

- `status:<toggle>`
- `tab:<screen_id>`
- `category:<category_id>`
- `pane:<screen_id>:<pane_id>`
- `overlay:<overlay_id>:<control_id>`

Dispatch behavior:

1. MouseDown stores candidate hit region + origin layer.
2. MouseUp on same region activates action.
3. Drag threshold prevents accidental click activation.
4. Wheel routes to topmost scrollable layer.

## Command Palette Verb Taxonomy

Palette actions are structured as stable verb IDs:

- `screen.open:<screen_id>`
- `screen.back`
- `filter.apply:<name>`
- `filter.clear`
- `overlay.toggle:<overlay_id>`
- `ops.capture_snapshot`
- `ops.export_evidence`
- `ops.replay_last_incident`
- `ops.reconnect_stream`
- `ops.toggle_fast_only`

Each verb must define:

- required arguments
- preconditions
- emitted telemetry event
- success/failure result message

## Inline vs Alt-Screen + Reconnect Semantics

## Display Modes

- `alt_screen` (default interactive operations console)
- `inline` (embedded mode for constrained terminals/log contexts)

Mode contract:

- Context state persists across mode switches.
- Overlay stack is preserved where possible.
- Inline mode may collapse panels but cannot drop critical status/error signals.

## Reconnect States

Every stream-backed screen shows one of:

- `connected`
- `degraded` (partial feeds or lag)
- `reconnecting` (retry in progress)
- `offline` (manual intervention needed)

Reconnect policy:

1. Exponential backoff with jitter for automatic retries.
2. `Ctrl+R` forces immediate retry.
3. Last known good metrics remain visible with stale markers.
4. Transition events are logged to timeline + status chrome.

## Cross-Screen Drilldown and Context Preservation

## Context Envelope (must be carried across screens)

```text
project_id
instance_id (optional)
query_id (optional)
time_window
active_filters[]
sort_mode
cursor/selection anchor
```

## Required Drilldown Flows

1. `fleet_overview` tile -> `project_dashboard` with `project_id`.
2. `project_dashboard` latency anomaly card -> `live_search_stream` with `time_window` + `active_filters=latency_anomaly`.
3. `live_search_stream` query row -> `explainability_cockpit` with `query_id`.
4. `alerts_timeline` incident -> `historical_analytics` with incident time window preloaded.
5. `historical_analytics` regression point -> `explainability_cockpit` with nearest relevant `query_id` if available.

Back-navigation contract:

- Return restores prior screen selection, filters, and scroll anchor.
- No workflow may reset context unless user explicitly runs `filter.clear`.

## Implementation Validation Contract

Downstream beads must implement tests and diagnostics against this IA:

| Level | Required Validation |
|---|---|
| Unit | screen registry completeness, unique `id`s/hotkeys, valid drilldown targets, keybinding collision checks, hit-region ID format validation. |
| Integration | navigation layer priority, overlay dismissal order, context envelope propagation across drilldowns, reconnect state transitions. |
| E2E | operator workflows covering all five required drilldown flows, inline/alt-screen mode switching with state retention, command palette action execution. |

Structured diagnostics required in artifacts:

- `navigation_event` (from/to screen, cause, context hash)
- `palette_action_event` (verb, args, result)
- `overlay_state_event` (open/close, layer)
- `reconnect_state_event` (from/to state, retry count)
- `context_restore_event` (screen, restored fields)

Artifacts required for CI/replay:

1. deterministic run metadata (seed, tick, terminal size)
2. structured JSONL logs for navigation/palette/reconnect
3. snapshot captures for each required screen + overlay stack states

## E2E Failure Triage Playbook Link

Operator triage for failing unified v1 artifact bundles is standardized in:

- `docs/e2e-artifact-contract.md#replay-and-triage-playbook`

CI failure outputs in `.github/workflows/ci.yml` also publish this same runbook link so operators can pivot from failed jobs directly into the replay workflow.

## Operator Runbook (Production Use)

This runbook is the day-1/day-2 operational baseline for the control-plane TUI.

### A) Startup and Verification Checklist (Shift Start)

1. Confirm fsfs/control-plane health input is available:
   - `fsfs status --no-watch-mode --format json`
2. Open the TUI and verify primary navigation surfaces:
   - `1` Fleet Overview
   - `2` Project Detail
   - `7` Alerts + Timeline
   - `?` Help overlay renders with keybindings
3. Confirm stream health on one active project:
   - open `live_search_stream` (`3`),
   - verify state is `connected` or expected `degraded` with explicit reason.
4. Confirm trend windows are populated:
   - open `resource_trends` (`5`) and `historical_analytics` (`6`),
   - verify at least one non-empty historical window (`15m` or greater).
5. Confirm explainability path:
   - open `explainability_cockpit` (`8`) from a live query drilldown.

### B) Fast Incident Triage Workflow (5-10 minutes)

1. Detect:
   - start at `fleet_overview` (`1`) and identify impacted project(s).
2. Scope:
   - jump to `project_dashboard` (`2`) and compare latency/health cards.
3. Correlate:
   - inspect `alerts_timeline` (`7`) for first failure timestamp and reason codes.
4. Deep dive:
   - inspect `historical_analytics` (`6`) for percentile/regression windows.
   - pivot into `explainability_cockpit` (`8`) for query/ranking evidence.
5. Classify severity:
   - `Fatal`: hard outage/data unavailability.
   - `Degraded`: partial service with constrained quality/throughput.
   - `Transient`: short-lived recoverable blip.
6. Act:
   - apply operator action via palette (`Ctrl+K`) only if guardrails allow,
   - record reason code + action in incident thread.

### C) Deterministic Replay Procedure

Use this when triaging CI or production-captured artifact bundles.

1. Download artifacts:

```bash
gh run download <run-id> --dir /tmp/frankensearch-ci
```

2. Inspect manifest and key fields:

```bash
cd <bundle_dir>
jq '.body | {suite, exit_status, determinism_tier, seed, duration_ms}' manifest.json
```

3. Replay exactly from bundled command:

```bash
bash replay_command.txt
```

4. Record:
   - `run_id`, `suite`, top `reason_code`, replay result (`reproduced`/`not_reproduced`), next owner.

### D) Rollout Verification Checklist

Before promoting a rollout phase (`shadow -> canary -> default`), verify:

1. No unresolved fatal diagnostics in alerts timeline.
2. Latency/error-budget gates remain within declared phase envelope.
3. Degraded/fallback reason-code rate is under declared threshold.
4. Replay command for latest verification artifact succeeds.
5. Rollback path has been dry-run in the current release window.

### D.1) Host Rollout Matrix (Required)

Apply the same phase sequence for each priority host:

| Host | Shadow gate | Canary gate | Default gate |
|---|---|---|---|
| `coding_agent_session_search` | Query/result parity on representative automation corpus | Error/latency budget stable for canary cohort | 24h stable operation + replay validation pass |
| `xf` | High-volume query stream parity and reason-code stability | Canary SLO + degradation-rate thresholds met | Stable production window + no unresolved P0/P1 incidents |
| `mcp_agent_mail_rust` | Thread/message retrieval contract parity in shadow mode | Agent workflow contract checks pass in canary | Full rollout only after deterministic replay checks pass |
| `frankenterm` | Interactive workflow parity (latency + explainability surfaces) | Canary interactive latency and error budget within thresholds | Default only after rollback drill is validated |

### D.2) Post-Rollout Health Checks (Required)

After each host reaches `default`, verify:

1. SLO/error-budget metrics stay within target.
2. `p95/p99` latency remains within rollout envelope.
3. Fallback/degradation reason-code rates stay within threshold.
4. Deterministic replay of sampled incidents succeeds.

Rollback triggers:

1. Fatal incident attributable to rollout delta.
2. Contract-breaking output regression.
3. Persistent latency/error-budget breach beyond allowed window.
4. Reproducible failure remains unresolved after defined mitigation window.

### E) Rollback Procedure (Operator-Facing)

If rollout gate fails:

1. Freeze progression for affected host/project.
2. Re-pin to last known good deployment path.
3. Re-verify startup, stream connectivity, and project dashboard health.
4. Re-run deterministic replay on failing artifact pack.
5. Publish rollback summary with:
   - trigger reason code,
   - rollback version/path,
   - verification evidence links.

Rollback is complete only when the project returns to known-good telemetry and all blocking alerts are resolved or downgraded with explicit rationale.

## Downstream Implementation Boundaries

This spec is now authoritative for:

- `bd-2yu.6.1`, `bd-2yu.6.4`, `bd-2yu.6`
- `bd-2yu.2.1`, `bd-2yu.2.3`, `bd-2yu.2`
- `bd-2hz.7.1`, `bd-2hz.7.7`
- `bd-2hz.12`

Any mismatch between implementation and this contract must be resolved by updating this file and the bead thread first.
