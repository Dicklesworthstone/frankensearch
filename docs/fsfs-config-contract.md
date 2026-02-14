# fsfs Configuration Contract v1

Issue: `bd-2hz.13`  
Parent: `bd-2hz`

## Goal

Define the canonical fsfs configuration model, precedence rules, validation semantics, and diagnostics contract so downstream implementation beads can consume one unambiguous source of truth.

This contract is normative for both fsfs UX modes:

- agent-first CLI
- deluxe TUI

Artifacts:

- Schema: `schemas/fsfs-config-v1.schema.json`
- Valid fixtures: `schemas/fixtures/fsfs-config-*.json`
- Invalid fixtures: `schemas/fixtures-invalid/fsfs-config-invalid-*.json`
- Checker: `scripts/check_fsfs_config_contract.sh`

## Source Precedence (Normative)

Exact precedence order (highest to lowest):

1. CLI flags
2. environment variables
3. config file
4. compiled defaults

No component may override this order.

## Config File Location Policy

Primary path:

- `${XDG_CONFIG_HOME}/fsfs/config.toml`

Fallback path:

- `~/.config/fsfs/config.toml`

Path expansion rule:

- Any path value starting with `~` MUST expand to the current user home directory before validation.

## Configuration Sections

## `[discovery]`

- `roots: string[]` (default: `[$HOME]`)
- `exclude_patterns: string[]`
- `text_selection_mode: "blocklist" | "allowlist"` (default: `blocklist`)
- `binary_blocklist_extensions: string[]`
- `max_file_size_mb: int` (`1..1024`)
- `follow_symlinks: bool`

## `[indexing]`

- `fast_model: string`
- `quality_model: string`
- `model_dir: string`
- `embedding_batch_size: int` (`1..4096`)
- `reindex_on_change: bool`
- `watch_mode: bool`

## `[search]`

- `default_limit: int` (`1..200`)
- `quality_weight: number` (`0.0..1.0`)
- `rrf_k: number` (`>=1.0`)
- `quality_timeout_ms: int` (`>=50`)
- `fast_only: bool`
- `explain: bool`

## `[pressure]`

- `profile: "strict" | "performance" | "degraded"`
- `cpu_ceiling_pct: int` (`1..100`)
- `memory_ceiling_mb: int` (`>=128`)

## `[tui]`

- `theme: "auto" | "light" | "dark"`
- `frame_budget_ms: int` (`8..200`)
- `show_explanations: bool`
- `density: "compact" | "normal" | "expanded"`

## `[storage]`

- `db_path: string`
- `evidence_retention_days: int` (`1..3650`)
- `summary_retention_days: int` (`1..3650`)

## `[privacy]`

- `redact_file_contents_in_logs: bool` (default MUST be `true`)
- `redact_paths_in_telemetry: bool` (default MUST be `true`)

## Validation Rules (Normative)

- Unknown keys in config files MUST generate warnings, not hard errors.
- Validation failures MUST include stable reason codes and field paths.
- `summary_retention_days` MUST be greater than or equal to `evidence_retention_days`.
- If `search.fast_only = true` while `indexing.quality_model` is configured, a warning MUST be emitted:
  `config.search.fast_only_with_quality_model`.

## Environment Variable Mapping

Every key has an env var mapped with `FSFS_{SECTION}_{KEY}` in `SCREAMING_SNAKE_CASE`.

Examples:

- `search.quality_weight` -> `FSFS_SEARCH_QUALITY_WEIGHT`
- `pressure.profile` -> `FSFS_PRESSURE_PROFILE`
- `privacy.redact_paths_in_telemetry` -> `FSFS_PRIVACY_REDACT_PATHS_IN_TELEMETRY`

## CLI Flag Mapping (Required Surface)

- `--roots`
- `--exclude`
- `--limit`
- `--fast-only`
- `--explain`
- `--profile`
- `--theme`

## Diagnostics + Logging Contract

`config_loaded` event MUST include:

- `event`
- `source_precedence_applied`
- `cli_flags_used`
- `env_keys_used`
- `config_file_used`
- `resolved_values`
- `warnings`
- `reason_codes`

All diagnostics MUST be machine-readable and replay-safe.

## Policy Interpretation Guide (Operator)

Use this table when translating config changes into expected runtime behavior.

| Domain | Primary fields | Policy effect | Diagnostics anchor |
|---|---|---|---|
| Discovery scope | `discovery.roots`, `discovery.exclude_patterns`, `discovery.max_file_size_mb`, `discovery.binary_blocklist_extensions` | Controls corpus inclusion/exclusion before indexing cost is incurred. | `reason_codes` such as `discovery.root.accepted`, `discovery.root.rejected`, `discovery.file.excluded_pattern`, `discovery.file.too_large`, `discovery.file.binary_blocked` |
| Search quality/latency | `search.fast_only`, `search.quality_weight`, `search.quality_timeout_ms`, `search.rrf_k` | Governs fast-vs-quality behavior, blend semantics, and timeout-based fallback pressure. | `warnings[]`, `reason_codes`, and downstream `RefinementFailed`/fallback events |
| Pressure governance | `pressure.profile`, `pressure.degradation_override`, `pressure.hard_pause_requested` | Applies profile locks and safety clamps that may override lower-priority knobs. | `profile_reason_code`, `overrides[]`, `safety_clamps[]`, `override.rejected.locked_field` |
| Privacy/logging | `privacy.redact_file_contents_in_logs`, `privacy.redact_paths_in_telemetry` | Defines redaction guarantees for operator-visible traces and artifacts. | `resolved_values.privacy.*` plus contract checks in `docs/fsfs-scope-privacy-contract.md` |

## Scenario Playbooks (Troubleshooting + Recovery)

### Playbook 1: Startup fails due to invalid configuration

- Symptoms: process exits before serving, with `SearchError::InvalidConfig`.
- Diagnose:
  - inspect `field` + `value` from the error payload,
  - compare against limits in this contract (for example `search.default_limit`, `pressure.sample_interval_ms`, retention bounds),
  - verify expanded paths for `~` values.
- Recovery:
  - fix the offending field in file/env/CLI source,
  - rerun with machine output (`fsfs status --format json`) to confirm `config_loaded` emits expected `resolved_values`,
  - if unresolved, validate fixtures and schema with the commands below.
- Exit criteria: startup succeeds and no hard validation errors are emitted.

### Playbook 2: Effective config differs from operator expectation

- Symptoms: behavior suggests a different mode than configured (for example quality unexpectedly disabled).
- Diagnose:
  - inspect `source_precedence_applied`, `cli_flags_used`, `env_keys_used`, and `resolved_values`,
  - inspect `reason_codes` for `config.search.fast_only_with_quality_model`, `override.rejected.locked_field`, or `profile.resolution.conflict`.
- Recovery:
  - remove/adjust the highest-precedence override causing drift (CLI first, then env),
  - re-run and confirm `profile.resolution.ok` and expected `resolved_values`,
  - record the corrected precedence source in incident notes.
- Exit criteria: effective values match intended policy and conflict warnings are gone.

### Playbook 3: Discovery scope causes resource pressure

- Symptoms: indexing load spikes, long queues, or excessive churn immediately after scope changes.
- Diagnose:
  - inspect discovery decisions and count `discovery.file.excluded_pattern` vs `discovery.file.included`,
  - inspect large-file and binary filters via `discovery.file.too_large` and `discovery.file.binary_blocked`,
  - verify root acceptance/rejection (`discovery.root.accepted`, `discovery.root.rejected`).
- Recovery:
  - tighten `exclude_patterns`, reduce `max_file_size_mb`, and confirm binary extension blocking,
  - if needed, switch to a stricter pressure profile before retrying ingestion,
  - rerun indexing on a bounded root subset before re-expanding scope.
- Exit criteria: sustained indexing within resource ceilings and stable reason-code mix.

## Validation Commands

```bash
scripts/check_fsfs_config_contract.sh --mode unit
scripts/check_fsfs_config_contract.sh --mode integration
scripts/check_fsfs_config_contract.sh --mode e2e
scripts/check_fsfs_config_contract.sh --mode all
```

## Integration Mapping

- `bd-2hz.3.1`: consumes section/type model for config loader and CLI surface.
- `bd-2hz.4.5`: consumes `[pressure]` profile contract.
- `bd-2hz.1.3`: consumes `[privacy]` defaults and telemetry redaction semantics.
- `bd-2hz.3.8`: consumes file policy and restart/reload expectations.
