# fsfs Query-Plan Metamorphic Contract

`bd-pkl0.6` adds a corpus-independent query-plan contract for the fsfs search path. The suite does not depend on indexed documents; it exercises the planner directly with deterministic query families and records the expected replay command for each case.

The contract covers:

- canonicalization idempotence for raw and whitespace-mutated queries
- stable classification, fallback, execution mode, and budget profile across whitespace variants
- safe candidate budgets for lexical, semantic, quality, and rerank stages
- empty-query no-op behavior
- lexical-biased identifier planning
- semantic-biased natural-language planning
- dash and `NOT "phrase"` negation parsing
- low-signal, malformed, and missing-capability fallback paths
- deterministic RRF tie-break policy order

Artifacts:

- `schemas/fsfs-query-plan-metamorphic-v1.schema.json`
- `schemas/fixtures/fsfs-query-plan-metamorphic-contract-v1.json`
- `schemas/fixtures/fsfs-query-plan-metamorphic-report-v1.json`
- `schemas/fixtures/fsfs-query-plan-metamorphic-minimized-failure-v1.json`
- `schemas/fixtures-invalid/fsfs-query-plan-metamorphic-invalid-missing-replay-v1.json`
- `schemas/fixtures-invalid/fsfs-query-plan-metamorphic-invalid-unreported-failure-v1.json`

Replay one generated case:

```bash
FSFS_QUERY_PLAN_CASE=qp-negation-0005 cargo test -p frankensearch-fsfs query_plan_metamorphic_contract_suite -- --nocapture
```

Validate schema fixtures and the focused Rust suite:

```bash
scripts/check_fsfs_query_plan_metamorphic.sh --mode all
```
