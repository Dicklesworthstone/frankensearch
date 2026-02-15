# Baseline Comparator and Budgeted-Mode Planning Policy (bd-2l7y)

## Goal
Make high-risk control/performance beads auditable before implementation by requiring explicit rollout/evaluation fields.

## Required Fields
Applicable beads must include these four fields (in description or an explicit policy comment):
1. `BASELINE_COMPARATOR`
2. `BUDGETED_MODE_DEFAULTS`
3. `ON_EXHAUSTION`
4. `SUCCESS_THRESHOLDS_AND_STOP_CONDITIONS`

## Standard Template
```text
[bd-2l7y baseline-budget]
BASELINE_COMPARATOR: <current behavior or prior strategy being beaten>
BUDGETED_MODE_DEFAULTS: <time/memory/depth/retry defaults>
ON_EXHAUSTION: <fallback/degraded behavior and reason code>
SUCCESS_THRESHOLDS_AND_STOP_CONDITIONS: <explicit success bars + stop triggers>
```

## Field Guidance
- `BASELINE_COMPARATOR`: Name a concrete incumbent behavior (for example, static config, no prefaulting, naive eviction).
- `BUDGETED_MODE_DEFAULTS`: Use deterministic defaults and units (`ms`, `MB`, counts, retries).
- `ON_EXHAUSTION`: Define what happens when budget is exhausted (fallback mode, disablement, or early termination).
- `SUCCESS_THRESHOLDS_AND_STOP_CONDITIONS`: Include measurable threshold(s) and a clear abort condition.

## Alignment With Composition Matrix Gate (`bd-1pkl`)
For advanced ranking/control beads, baseline-budget planning and composition linkage are coupled:
1. `ON_EXHAUSTION` text must align with the composition annotation `FALLBACK_SEMANTICS`.
2. `SUCCESS_THRESHOLDS_AND_STOP_CONDITIONS` must be compatible with the interaction-test plan used for composition validation.
3. Reviewers should reject beads where baseline-budget fields and composition fallback semantics disagree.

## Migration Notes (Retrofit Wave)
Backfilled with explicit `bd-2l7y` planning fields:
- Adaptive/control beads: `bd-21g`, `bd-22k`, `bd-2ps`, `bd-2yj`, `bd-1do`, `bd-2tv`
- Performance-heavy beads: `bd-i37`, `bd-l7v`, `bd-1co`, `bd-2rq`, `bd-2u4`, `bd-6sj`

Notes for maintainers:
- Existing beads can use comments for retrofit instead of rewriting long historical descriptions.
- New beads in this class should include the fields at creation time, not post-hoc.
- Keep field values specific enough to be testable and reviewable.

## Example
```text
[bd-2l7y baseline-budget]
BASELINE_COMPARATOR: always run quality-tier refinement with static blend.
BUDGETED_MODE_DEFAULTS: observation_window_queries=200, cooldown_ms=30000, max_memory_mb=16, retry_budget=0.
ON_EXHAUSTION: switch to fast_only for cooldown window with reason_code=circuit_exhausted.
SUCCESS_THRESHOLDS_AND_STOP_CONDITIONS: p95 latency improves >= 10% while failure rate <= 2%; stop if oscillation > 3 trips/hour.
```

## Reviewer Alignment Checklist
1. Baseline names a real incumbent behavior.
2. Budget defaults have explicit units and are deterministic.
3. Exhaustion behavior has a concrete fallback and reason semantics.
4. Success and stop conditions are measurable, not aspirational.
5. Values align with the bead's stated scope and acceptance criteria.

## Checker and Replay Commands
Checker script:
- `scripts/check_bead_baseline_budget.sh`

Modes:
- `--mode unit`: enforces required fields on bd-2l7y retrofit target beads.
- `--mode integration`: validates all beads carrying the bd-2l7y marker and surfaces additional adoption candidates.
- `--mode e2e`: governance self-check proving actionable failure diagnostics for malformed templates.
- `--mode all`: runs unit + integration + e2e.

Replay:
```bash
scripts/check_bead_baseline_budget.sh --mode all
```

## Statistical Regression Gate Policy (bd-2hz.9.6)

This section defines the canonical CI-facing statistical gate for performance-sensitive lanes.

### Required Metrics and Budgets

Every performance gate evaluation MUST include:

- latency: `p50`, `p95`, `p99` (milliseconds)
- memory: peak resident memory (`MB`)

Default fail thresholds (unless bead-specific tighter budgets are declared):

- `p50` regression > `5%`
- `p95` regression > `8%`
- `p99` regression > `12%`
- peak memory regression > `10%`

### Confidence Policy

- Minimum independent run count: `5`.
- Preferred confidence target: `95%` for regression decision.
- If confidence cannot be established (insufficient samples or unstable variance), gate result MUST be `inconclusive` with deterministic reason code.

### Flake Mitigation Policy

- Detect flake candidate when metric spread exceeds configured variance envelope.
- Allow up to `2` bounded re-runs before final gate decision.
- Emit triage metadata identifying:
  - rerun count,
  - variance signal,
  - final decision basis.

### CI Artifact Requirements

Performance gate lanes MUST publish:

- `perf_regression_gate_policy.json`
- `perf_regression_gate_result.json`
- baseline and current metric manifests (when available)
- replay command for reproduction

### Deterministic Reason Codes

- `perf.gate.pass`
- `perf.gate.missing_inputs`
- `perf.gate.inconclusive_confidence`
- `perf.gate.flake_suspected`
- `perf.gate.regression_detected`
