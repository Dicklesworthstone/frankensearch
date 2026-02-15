# Self-Documentation Lint Policy v1 (`bd-3qwe.7.1`)

## Purpose
Define a stable lint rule catalog and severity policy for backlog self-documentation quality.
This policy is consumed by `scripts/check_bead_test_matrix.sh` and future lint stages in `bd-3qwe.7.*`.
For the broader rationale/evidence/exception rubric, see `docs/bead-self-documentation-rubric.md`.

## Bead Classes
Beads are classified before rule evaluation:

1. `implementation`
   Applies to `task`, `feature`, `bug` by default.
2. `program`
   Applies to `epic` and other coordination/aggregation workstreams.
3. `gate`
   Any bead with `gate` or `release-gate` label (overrides issue type).
4. `exploratory`
   Applies to `question` and `docs`.

## Rule Catalog
Stable rule IDs and default severities:

1. `SDOC-MATRIX-001` (default: `error`)
   Missing explicit `[bd-264r test-matrix] TEST_MATRIX` annotation on implementation beads that require matrix coverage.
2. `SDOC-MATRIX-002` (default: `error`)
   Missing required matrix sections:
   `Unit tests`, `Integration tests`, `E2E tests`, `Performance/bench`, `Logs/artifacts`.
3. `SDOC-MATRIX-003` (default: `error`)
   Missing explicit `[bd-264r test-matrix] EXCEPTION` annotation for approved workstream exceptions.
4. `SDOC-POLICY-000` (default: `error`)
   Internal policy self-test failure (classification/severity mapping drift in the checker).
5. `SDOC-MATRIX-004` (default: `error`)
   Advanced ranking/control beads are missing the required composition-matrix linkage annotation.
6. `SDOC-MATRIX-005` (default: `error`)
   Composition-matrix annotation exists but is missing required linkage/fallback/interaction-test fields.
7. `SDOC-MATRIX-006` (default: `error`)
   Composition-matrix exception exists but is missing required waiver metadata.

Severity remapping by bead class for missing-matrix findings:

1. `implementation` -> `error`
2. `gate` -> `error`
3. `program` -> `warning`
4. `exploratory` -> `info`

## CI Fail Thresholds And Rollout
Phased rollout knobs:

1. Phase 0 (`audit`): report only, no fail.
2. Phase 1 (`default`): fail on `error`.
3. Phase 2 (`strict`): fail on `warning` and above for selected tracks.

Recommended environment knobs:

1. `SELFDOC_LINT_PHASE=audit|default|strict`
2. `SELFDOC_LINT_MIN_FAIL_SEVERITY=error|warning|info`
3. `SELFDOC_LINT_SCOPE=<label|prefix|all>`

## Exception Contract
Any waiver must be explicit and machine-readable in bead comments:

1. Marker: `[bd-264r test-matrix] EXCEPTION`
2. Required metadata:
   `rule_id`, `owner`, `justification`, `expires_on`, `follow_up_bead`
3. Waivers without `expires_on` are invalid for strict CI.
4. Composition-matrix waivers must use `rule_id: SDOC-MATRIX-004`.

## Reporter Payload Contract
Each rule outcome must include these structured fields:

1. `rule_id`
2. `severity`
3. `bead_id`
4. `message`
5. `fix_hint`

`scripts/check_bead_test_matrix.sh` emits this payload as JSON lines for findings.

## Composition Matrix Gate (`bd-1pkl`)
Advanced ranking/control beads must carry a machine-readable annotation that links to the
composition matrix lane (`bd-3un.52` or explicit successor), defines fallback semantics, and
names interaction-test coverage.

Required marker:
1. `[bd-1pkl composition-matrix] COMPOSITION_MATRIX`

Required fields:
1. `MATRIX_LINK`
2. `FALLBACK_SEMANTICS`
3. `INTERACTION_TEST_PLAN`

Validation expectations:
1. `MATRIX_LINK` points to `bd-3un.52` or an explicit successor reference.
2. `FALLBACK_SEMANTICS` includes an `ON_EXHAUSTION` behavior.
3. `INTERACTION_TEST_PLAN` names deterministic interaction coverage (for example `interaction_unit`, `interaction_integration`).

## Normalized Scanner Model v1 (`bd-3qwe.7.2`)
The lint scanner normalizes `.beads/issues.jsonl` into a deterministic in-memory model before rule evaluation.

Normalized per-bead fields:

1. `id`
2. `issue_type`
3. `status`
4. `priority`
5. `labels` (sorted, unique)
6. `title`
7. `description`
8. `acceptance_criteria` (scalarized string)
9. `notes` (scalarized string)
10. `comments` (array of `{author,text,created_at}`)
11. `comment_text` (joined comment text)
12. `dependencies` (array of `{depends_on_id,dep_type}`)
13. `source_line`

Determinism contract:

1. Output is sorted by `(id, source_line)`.
2. Mixed scalar/array/object shapes are normalized to stable string/object forms.
3. Scanner emits deterministic diagnostics for parse and normalization errors:
   `SDOC-SCAN-001` (parse) and `SDOC-SCAN-002` (normalization).

Scanner validation coverage:

1. Unit checks for mixed-type comments, acceptance criteria, and dependencies.
2. Unit checks for malformed-record normalization behavior (missing IDs/dependency targets).
3. Integration run confirms normalized model is consumable by matrix rule checks.

## Matrix Template
```text
[bd-264r test-matrix] TEST_MATRIX
Unit tests:
- ...
Integration tests:
- ...
E2E tests:
- ... (or N/A: <reason>)
Performance/bench:
- ... (or N/A: <reason>)
Logs/artifacts:
- ...
```

## Validation Command
```bash
scripts/check_bead_test_matrix.sh --mode all
```

## Composition Matrix Template
```text
[bd-1pkl composition-matrix] COMPOSITION_MATRIX
MATRIX_LINK: bd-3un.52
FALLBACK_SEMANTICS: ON_EXHAUSTION -> lexical_only (reason_code=composition.matrix.exhausted)
INTERACTION_TEST_PLAN: interaction_unit + interaction_integration + lane ownership artifact check
```
