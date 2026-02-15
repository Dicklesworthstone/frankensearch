# Dependency Semantics Policy (bd-17dv)

## Purpose
Prevent dependency-graph churn by enforcing explicit semantics for every relationship:
- `HARD_DEP`: correctness blocker.
- `SOFT_DEP`: coordination/interaction note, not a blocker edge.
- `INFO_REF`: context/reference only.

## Contract

### `HARD_DEP`
Use a dependency edge (`type=blocks`) only when implementation correctness cannot proceed without the upstream bead's artifact.

Required rationale template (must appear in bead body or comment):
- `HARD_DEP <upstream-id>: required because <concrete contract/artifact>`

### `SOFT_DEP`
Use prose comments (not dependency edges) for sequencing preference, optional integrations, or cross-feature awareness.

Required rationale template:
- `SOFT_DEP <upstream-id>: interaction only; does not block standalone correctness`

### `INFO_REF`
Use prose-only references for papers, prior art, design inspirations, or optional future coupling.

Required rationale template:
- `INFO_REF <subject>: context/reference only`

### Epic Relationship Rule
If a task is part of an epic, prefer `parent-child` over `blocks` for the epic link. Epic links are grouping semantics, not runtime/correctness blockers.

## Reviewer Checklist
For each new/edited dependency:
1. Verify dependency type matches semantics (`blocks` vs `parent-child` vs prose-only).
2. Verify rationale text uses `HARD_DEP` / `SOFT_DEP` / `INFO_REF` and names specific contract/artifact.
3. Reject vague rationale (`"needed for integration"`, `"related"`) without concrete contract.
4. Ensure soft interactions are in comments/body, not blocker edges.
5. Ensure epic links are not encoded as `blocks` unless explicitly justified.

## Concrete Examples (>=10)
1. `HARD_DEP bd-z3j -> bd-3un.20`: MMR needs fused relevance prior from RRF.
2. `HARD_DEP bd-z3j -> bd-3un.15`: MMR requires top-k semantic candidates.
3. `HARD_DEP bd-z3j -> bd-3un.13`: MMR needs embedding/index retrieval access.
4. `HARD_DEP bd-2rq -> bd-3un.24`: federation fans out to `TwoTierSearcher`.
5. `HARD_DEP bd-2rq -> bd-26e`: typed filters must propagate to sub-indices.
6. `HARD_DEP bd-2u4 -> bd-3un.18`: prefix mode needs lexical prefix execution.
7. `HARD_DEP bd-2tv -> bd-3un.20`: feedback boost is applied post-RRF.
8. `HARD_DEP bd-6sj -> bd-3un.39`: OPE consumes structured evidence logs.
9. `HARD_DEP bd-sot -> bd-3un.13`: tombstones require FSVI flags semantics.
10. `SOFT_DEP bd-z3j <-> bd-11n`: explanation population when both features enabled.
11. `SOFT_DEP bd-1co <-> bd-l7v`: cache/prefault are complementary, independent.
12. `INFO_REF bd-z3j`: Carbonell & Goldberg (1998) MMR formulation.

## Retrofit Normalization Report (High-Churn Set)
Scope: `bd-z3j`, `bd-i37`, `bd-2rq`, `bd-2u4`, `bd-2tv`, `bd-6sj`, `bd-1co`, `bd-sot`.

Edge updates applied:
- `bd-z3j -> bd-3un`: `blocks` -> `parent-child`
- `bd-i37 -> bd-3un`: `blocks` -> `parent-child`
- `bd-2rq -> bd-3un`: `blocks` -> `parent-child`
- `bd-2u4 -> bd-3un`: `blocks` -> `parent-child`
- `bd-2tv -> bd-3un`: `blocks` -> `parent-child`
- `bd-1co -> bd-3un`: `blocks` -> `parent-child`
- `bd-sot -> bd-3un`: `blocks` -> `parent-child`
- `bd-6sj -> bd-3un`: already `parent-child` (no change)

Documentation retrofit:
- Added `[bd-17dv retrofit] DEP_SEMANTICS` comments to all eight scope beads, with explicit `HARD_DEP` and `SOFT_DEP`/`INFO_REF` annotations.

## Lint and CI Checks
Checker script:
- `scripts/check_dependency_semantics.sh`

Composition-matrix policy extension (`bd-1pkl`):
- Advanced ranking/control governance beads must declare explicit dependency linkage to `bd-3un.52` (or documented successor) and include a composition semantics annotation:
  - marker: `[bd-1pkl composition-matrix] DEP_SEMANTICS`
  - required fields: `MATRIX_LINK`, `FALLBACK_SEMANTICS`, `INTERACTION_TEST_PLAN`
- The lint script now fails if required composition linkage/annotation is missing.

Modes:
- Unit checks:
  - targeted beads use `parent-child` for `bd-3un`.
  - targeted beads contain retrofit `DEP_SEMANTICS` annotations.
  - annotations include at least one `HARD_DEP` example.
- Integration checks:
  - targeted beads do not retain ambiguous `bd-3un` blocker edges.
  - global candidate ambiguous epic blocker list is surfaced for triage.

Replay command:
```bash
scripts/check_dependency_semantics.sh --mode all
```

CI recommendation:
```bash
scripts/check_dependency_semantics.sh --mode all
```

Workflow:
- `.github/workflows/dependency-semantics-lint.yml`

Maintenance playbook:
- `docs/dependency-semantics-lint-playbook.md`
