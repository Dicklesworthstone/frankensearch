# fsfs Alien Recommendation Contracts v1

Issue: `bd-2hz.1.4`  
Parent: `bd-2hz.1`

## Goal

Define reusable recommendation-contract cards for top adaptive fsfs controllers:

- ingestion policy
- degradation scheduler
- ranking policy

Each card is machine-readable and directly consumable by implementation and test-planning workstreams.

## Required Card Fields

Every recommendation card MUST include:

- `ev_score`
- `priority_tier`
- `adoption_wedge`
- budgeted mode + fallback trigger
- baseline comparator
- isomorphism proof plan
- reproducibility artifact requirements
- rollback plan

## Card Catalog

## Ingestion Policy Card

Focus:
- high-cost include/skip/index-later tradeoffs
- utility-sensitive ingest under bounded compute

## Degradation Scheduler Card

Focus:
- pressure-state transitions
- safe fallback ladders and recovery gates

## Ranking Policy Card

Focus:
- quality/latency balancing under constrained resources
- stable tie-break and regression-safe rollout criteria

## Crawl/Ingest Optimization Track (`bd-2hz.9.3`)

Prioritized hotspot candidates and target gains:

1. `ingest.catalog.batch_upsert`
   - stage: `catalog_mutation`
   - target: p50 -16%, p95 -24%, throughput +20%
2. `crawl.classification.policy_batching`
   - stage: `classification`
   - target: p50 -10%, p95 -16%, throughput +12%
3. `ingest.queue.lane_budget_admission`
   - stage: `queue_admission`
   - target: p50 -9%, p95 -14%, throughput +11%
4. `crawl.discovery.path_metadata_cache`
   - stage: `discovery_walk`
   - target: p50 -8%, p95 -13%, throughput +10%
5. `ingest.embed_gate.early_skip`
   - stage: `embedding_gate`
   - target: p50 -7%, p95 -11%, throughput +9%

Isomorphism proof checklist requirements (per lever):

- baseline comparator explicitly names incumbent behavior (discovery/classification/catalog/queue/embed gate)
- replay command: `fsfs profile replay --lane ingest --lever-id <id> --compare baseline`
- invariants include:
  - deterministic scope/classification outcomes
  - monotonic catalog/changelog sequencing
  - bounded queue semantics and stable backpressure reason codes
  - explainability-preserving ingest/degrade reason codes

Rollback guardrails (per optimization class):

- rollback command: `fsfs profile rollback --lever-id <id> --restore baseline`
- abort triggers include class-specific reason codes (scope regressions, idempotency violations, queue starvation/unbounded growth, embed/degrade policy regressions)
- required recovery reason code: `opt.rollback.completed`

## Contract Semantics

- `ev_score` is numeric expected value (impact-confidence-reuse-effort normalization)
- `priority_tier` uses `A|B|C`
- `adoption_wedge` states where rollout starts first and why
- budgeted mode includes explicit defaults and exhaustion behavior
- fallback trigger includes `condition`, `fallback_action`, and `reason_code`
- baseline comparator names incumbent behavior being outperformed
- isomorphism proof plan defines invariants and replay checks
- reproducibility fields define required artifacts and replay command
- rollback plan defines deterministic rollback command and abort conditions

## Validation Artifacts

- `schemas/fsfs-alien-recommendations-v1.schema.json`
- `schemas/fixtures/fsfs-alien-recommendation-card-ingestion-v1.json`
- `schemas/fixtures/fsfs-alien-recommendation-bundle-v1.json`
- `schemas/fixtures-invalid/fsfs-alien-recommendation-invalid-*.json`
- `scripts/check_fsfs_alien_recommendations.sh`

## Validation Command

```bash
scripts/check_fsfs_alien_recommendations.sh --mode all
```
