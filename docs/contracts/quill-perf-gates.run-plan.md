# Quill Performance Gates — Generated Operator Run Plan

**GENERATED FILE — do not edit.** Rendered from `quill-perf-gates.toml` and the
compiled `PerfMatrixSpec` by `render_perf_run_plan_markdown`; the gauntlet test
`perf_run_plan_document_matches_the_manifest` fails closed on any drift. Regenerate
deliberately with `QUILL_PERF_RUN_PLAN_UPDATE=1`.

- manifest contract SHA-256: `d4a751d3dcbae22f282744a927971ccacb16df107784fc35346a316fb0f6993e`
- canonical matrix cells: 120

## Registered machines

| hardware-class | execution-profile | registry status |
|---|---|---|
| trj-zen3-5995wx | physical-64 | registered, required |
| trj-zen3-5995wx | smt2-128 | registered, required |
| m4-macos | scheduler-10 | registered, required, promotion held pending supported actual-loaded-image attestation |
| m5-macos | scheduler-14 | unavailable but required |
| x86-vps-ovh | x86-diagnostic | registered diagnostic-only |

## Gates

### QG-1 — bulk indexing, multi-core

- activated: `false`
- fixture: tiny + small + medium + xlarge; positions ON/OFF; threads = 1,2,4,8,16,32,64,96,128; commit included in timed window
- target: docs_per_sec >= 3.0x oracle AND >= 0.60x measured tokenize-only bandwidth ceiling (honesty denominator lane: same corpus through the tokenizer alone)
- primary_target_cell_width: 8

**Canonical cells (74):**

| fixture | metric | corpus | threads | positions | extras |
|---|---|---|---|---|---|
| bulk/tiny/1/positions_on | docs_per_second | tiny | 1 | positions_on |  |
| bulk/tiny/1/positions_off | docs_per_second | tiny | 1 | positions_off |  |
| bulk/tiny/2/positions_on | docs_per_second | tiny | 2 | positions_on |  |
| bulk/tiny/2/positions_off | docs_per_second | tiny | 2 | positions_off |  |
| bulk/tiny/4/positions_on | docs_per_second | tiny | 4 | positions_on |  |
| bulk/tiny/4/positions_off | docs_per_second | tiny | 4 | positions_off |  |
| bulk/tiny/8/positions_on | docs_per_second | tiny | 8 | positions_on |  |
| bulk/tiny/8/positions_off | docs_per_second | tiny | 8 | positions_off |  |
| bulk/tiny/16/positions_on | docs_per_second | tiny | 16 | positions_on |  |
| bulk/tiny/16/positions_off | docs_per_second | tiny | 16 | positions_off |  |
| bulk/tiny/32/positions_on | docs_per_second | tiny | 32 | positions_on |  |
| bulk/tiny/32/positions_off | docs_per_second | tiny | 32 | positions_off |  |
| bulk/tiny/64/positions_on | docs_per_second | tiny | 64 | positions_on |  |
| bulk/tiny/64/positions_off | docs_per_second | tiny | 64 | positions_off |  |
| bulk/tiny/96/positions_on | docs_per_second | tiny | 96 | positions_on |  |
| bulk/tiny/96/positions_off | docs_per_second | tiny | 96 | positions_off |  |
| bulk/tiny/128/positions_on | docs_per_second | tiny | 128 | positions_on |  |
| bulk/tiny/128/positions_off | docs_per_second | tiny | 128 | positions_off |  |
| bulk/small/1/positions_on | docs_per_second | small | 1 | positions_on |  |
| bulk/small/1/positions_off | docs_per_second | small | 1 | positions_off |  |
| bulk/small/2/positions_on | docs_per_second | small | 2 | positions_on |  |
| bulk/small/2/positions_off | docs_per_second | small | 2 | positions_off |  |
| bulk/small/4/positions_on | docs_per_second | small | 4 | positions_on |  |
| bulk/small/4/positions_off | docs_per_second | small | 4 | positions_off |  |
| bulk/small/8/positions_on | docs_per_second | small | 8 | positions_on |  |
| bulk/small/8/positions_off | docs_per_second | small | 8 | positions_off |  |
| bulk/small/16/positions_on | docs_per_second | small | 16 | positions_on |  |
| bulk/small/16/positions_off | docs_per_second | small | 16 | positions_off |  |
| bulk/small/32/positions_on | docs_per_second | small | 32 | positions_on |  |
| bulk/small/32/positions_off | docs_per_second | small | 32 | positions_off |  |
| bulk/small/64/positions_on | docs_per_second | small | 64 | positions_on |  |
| bulk/small/64/positions_off | docs_per_second | small | 64 | positions_off |  |
| bulk/small/96/positions_on | docs_per_second | small | 96 | positions_on |  |
| bulk/small/96/positions_off | docs_per_second | small | 96 | positions_off |  |
| bulk/small/128/positions_on | docs_per_second | small | 128 | positions_on |  |
| bulk/small/128/positions_off | docs_per_second | small | 128 | positions_off |  |
| bulk/medium/1/positions_on | docs_per_second | medium | 1 | positions_on |  |
| bulk/medium/1/positions_off | docs_per_second | medium | 1 | positions_off |  |
| bulk/medium/2/positions_on | docs_per_second | medium | 2 | positions_on |  |
| bulk/medium/2/positions_off | docs_per_second | medium | 2 | positions_off |  |
| bulk/medium/4/positions_on | docs_per_second | medium | 4 | positions_on |  |
| bulk/medium/4/positions_off | docs_per_second | medium | 4 | positions_off |  |
| bulk/medium/8/positions_on | docs_per_second | medium | 8 | positions_on |  |
| bulk/medium/8/positions_off | docs_per_second | medium | 8 | positions_off |  |
| bulk/medium/16/positions_on | docs_per_second | medium | 16 | positions_on |  |
| bulk/medium/16/positions_off | docs_per_second | medium | 16 | positions_off |  |
| bulk/medium/32/positions_on | docs_per_second | medium | 32 | positions_on |  |
| bulk/medium/32/positions_off | docs_per_second | medium | 32 | positions_off |  |
| bulk/medium/64/positions_on | docs_per_second | medium | 64 | positions_on |  |
| bulk/medium/64/positions_off | docs_per_second | medium | 64 | positions_off |  |
| bulk/medium/96/positions_on | docs_per_second | medium | 96 | positions_on |  |
| bulk/medium/96/positions_off | docs_per_second | medium | 96 | positions_off |  |
| bulk/medium/128/positions_on | docs_per_second | medium | 128 | positions_on |  |
| bulk/medium/128/positions_off | docs_per_second | medium | 128 | positions_off |  |
| bulk/xlarge/1/positions_on | docs_per_second | xlarge | 1 | positions_on |  |
| bulk/xlarge/1/positions_off | docs_per_second | xlarge | 1 | positions_off |  |
| bulk/xlarge/2/positions_on | docs_per_second | xlarge | 2 | positions_on |  |
| bulk/xlarge/2/positions_off | docs_per_second | xlarge | 2 | positions_off |  |
| bulk/xlarge/4/positions_on | docs_per_second | xlarge | 4 | positions_on |  |
| bulk/xlarge/4/positions_off | docs_per_second | xlarge | 4 | positions_off |  |
| bulk/xlarge/8/positions_on | docs_per_second | xlarge | 8 | positions_on |  |
| bulk/xlarge/8/positions_off | docs_per_second | xlarge | 8 | positions_off |  |
| bulk/xlarge/16/positions_on | docs_per_second | xlarge | 16 | positions_on |  |
| bulk/xlarge/16/positions_off | docs_per_second | xlarge | 16 | positions_off |  |
| bulk/xlarge/32/positions_on | docs_per_second | xlarge | 32 | positions_on |  |
| bulk/xlarge/32/positions_off | docs_per_second | xlarge | 32 | positions_off |  |
| bulk/xlarge/64/positions_on | docs_per_second | xlarge | 64 | positions_on |  |
| bulk/xlarge/64/positions_off | docs_per_second | xlarge | 64 | positions_off |  |
| bulk/xlarge/96/positions_on | docs_per_second | xlarge | 96 | positions_on |  |
| bulk/xlarge/96/positions_off | docs_per_second | xlarge | 96 | positions_off |  |
| bulk/xlarge/128/positions_on | docs_per_second | xlarge | 128 | positions_on |  |
| bulk/xlarge/128/positions_off | docs_per_second | xlarge | 128 | positions_off |  |
| tokenize_only/medium | tokenize_docs_per_second | medium | 1 | - |  |
| tokenize_only/xlarge | tokenize_docs_per_second | xlarge | 1 | - |  |

**Registered-host commands:**

```text
scripts/perf-runner.sh --gate QG-1 --hardware-class trj-zen3-5995wx --execution-profile physical-64 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-1 --hardware-class trj-zen3-5995wx --execution-profile smt2-128 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-1 --hardware-class m4-macos --execution-profile scheduler-10 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-1 --hardware-class m5-macos --execution-profile scheduler-14 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-1 --hardware-class x86-vps-ovh --execution-profile x86-diagnostic --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
```

### QG-2 — bulk indexing, single-thread

- activated: `false`
- fixture: medium; positions ON; threads = 1; commit included
- target: docs_per_sec >= 1.5x oracle

**Canonical cells (1):**

| fixture | metric | corpus | threads | positions | extras |
|---|---|---|---|---|---|
| bulk/medium/1/positions_on | docs_per_second | medium | 1 | positions_on |  |

**Registered-host commands:**

```text
scripts/perf-runner.sh --gate QG-2 --hardware-class trj-zen3-5995wx --execution-profile physical-64 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-2 --hardware-class trj-zen3-5995wx --execution-profile smt2-128 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-2 --hardware-class m4-macos --execution-profile scheduler-10 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-2 --hardware-class m5-macos --execution-profile scheduler-14 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-2 --hardware-class x86-vps-ovh --execution-profile x86-diagnostic --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
```

### QG-3 — watch-mode incremental

- activated: `false`
- fixture: 5k-update upsert-heavy batches over a warm medium corpus; measured in BOTH topologies (bd-quill-duel-visibility-contract): (a) in-process reader, (b) fresh-process reader
- target: floor: >=20k docs/s initial, >=5k updates/s, p95 <= 25ms (lexical_pipeline.rs contract, both topologies where applicable); headroom: update->searchable >= 4x oracle commit+reload path — claim (a) IN-PROCESS ONLY, publish both numbers with topology labels

**Canonical cells (5):**

| fixture | metric | corpus | threads | positions | extras |
|---|---|---|---|---|---|
| watch/medium/initial | docs_per_second | medium | 1 | positions_on |  |
| watch/medium/5000/inprocess | updates_per_second | medium | 1 | positions_on | topology=InProcess |
| watch/medium/5000/inprocess | update_to_searchable_ms | medium | 1 | positions_on | topology=InProcess |
| watch/medium/5000/freshprocess | updates_per_second | medium | 1 | positions_on | topology=FreshProcess |
| watch/medium/5000/freshprocess | update_to_searchable_ms | medium | 1 | positions_on | topology=FreshProcess |

**Registered-host commands:**

```text
scripts/perf-runner.sh --gate QG-3 --hardware-class trj-zen3-5995wx --execution-profile physical-64 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-3 --hardware-class trj-zen3-5995wx --execution-profile smt2-128 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-3 --hardware-class m4-macos --execution-profile scheduler-10 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-3 --hardware-class m5-macos --execution-profile scheduler-14 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-3 --hardware-class x86-vps-ovh --execution-profile x86-diagnostic --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
```

### QG-4 — commit latency

- activated: `false`
- fixture: 100k-doc index, warm; sealed-commit distribution over 100 commits
- target: p99 <= 50ms sealed commit; visibility lead (in-process searchable-before-commit) demonstrated once delta (e5.x) lands — until then this clause is N/A (G1a has no delta)

**Canonical cells (1):**

| fixture | metric | corpus | threads | positions | extras |
|---|---|---|---|---|---|
| commit/100000/warm | commit_latency_ms | - | 1 | positions_on |  |

**Registered-host commands:**

```text
scripts/perf-runner.sh --gate QG-4 --hardware-class trj-zen3-5995wx --execution-profile physical-64 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-4 --hardware-class trj-zen3-5995wx --execution-profile smt2-128 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-4 --hardware-class m4-macos --execution-profile scheduler-10 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-4 --hardware-class m5-macos --execution-profile scheduler-14 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-4 --hardware-class x86-vps-ovh --execution-profile x86-diagnostic --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
```

### QG-5 — full compaction

- activated: `false`
- fixture: 1M docs (xlarge) at tombstone densities {5, 20, 50}% — NEVER 0% (bd-6m8p lesson). The LANDED corpus.xlarge e6.1 generator re-baselines QG-5 on xlarge: perf.rs emits `compaction/xlarge/{density}pct` and perf_ratchet pins that same label, so the ratchet cannot bind no emitted cell and read green.
- target: wall-clock >= 5x faster than oracle force-merge at 20%; report all densities

**Canonical cells (3):**

| fixture | metric | corpus | threads | positions | extras |
|---|---|---|---|---|---|
| compaction/xlarge/5pct | wall_clock_ms | xlarge | 1 | positions_on | tombstones=5% |
| compaction/xlarge/20pct | wall_clock_ms | xlarge | 1 | positions_on | tombstones=20% |
| compaction/xlarge/50pct | wall_clock_ms | xlarge | 1 | positions_on | tombstones=50% |

**Registered-host commands:**

```text
scripts/perf-runner.sh --gate QG-5 --hardware-class trj-zen3-5995wx --execution-profile physical-64 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-5 --hardware-class trj-zen3-5995wx --execution-profile smt2-128 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-5 --hardware-class m4-macos --execution-profile scheduler-10 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-5 --hardware-class m5-macos --execution-profile scheduler-14 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-5 --hardware-class x86-vps-ovh --execution-profile x86-diagnostic --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
```

### QG-6 — query latency

- activated: `false`
- fixture: query-class mix from search_quality_harness slices (identifier/short_keyword/natural_language/phrase/boolean) x k in {10,100} x corpus {100k, 1M}
- target: p50 parity (+-10%) per class; p99 <= oracle per class; NO class regresses >10%; fuel-metering checkpoints (bd-quill-duel-fuel-metering) must show no measurable delta here before merging
- queries_per_class: 16

**Canonical cells (20):**

| fixture | metric | corpus | threads | positions | extras |
|---|---|---|---|---|---|
| query/identifier/k10/100k | latency_ms | - | 1 | positions_on | class=identifier, k=10 |
| query/identifier/k10/1m | latency_ms | - | 1 | positions_on | class=identifier, k=10 |
| query/identifier/k100/100k | latency_ms | - | 1 | positions_on | class=identifier, k=100 |
| query/identifier/k100/1m | latency_ms | - | 1 | positions_on | class=identifier, k=100 |
| query/short_keyword/k10/100k | latency_ms | - | 1 | positions_on | class=short_keyword, k=10 |
| query/short_keyword/k10/1m | latency_ms | - | 1 | positions_on | class=short_keyword, k=10 |
| query/short_keyword/k100/100k | latency_ms | - | 1 | positions_on | class=short_keyword, k=100 |
| query/short_keyword/k100/1m | latency_ms | - | 1 | positions_on | class=short_keyword, k=100 |
| query/natural_language/k10/100k | latency_ms | - | 1 | positions_on | class=natural_language, k=10 |
| query/natural_language/k10/1m | latency_ms | - | 1 | positions_on | class=natural_language, k=10 |
| query/natural_language/k100/100k | latency_ms | - | 1 | positions_on | class=natural_language, k=100 |
| query/natural_language/k100/1m | latency_ms | - | 1 | positions_on | class=natural_language, k=100 |
| query/phrase/k10/100k | latency_ms | - | 1 | positions_on | class=phrase, k=10 |
| query/phrase/k10/1m | latency_ms | - | 1 | positions_on | class=phrase, k=10 |
| query/phrase/k100/100k | latency_ms | - | 1 | positions_on | class=phrase, k=100 |
| query/phrase/k100/1m | latency_ms | - | 1 | positions_on | class=phrase, k=100 |
| query/boolean/k10/100k | latency_ms | - | 1 | positions_on | class=boolean, k=10 |
| query/boolean/k10/1m | latency_ms | - | 1 | positions_on | class=boolean, k=10 |
| query/boolean/k100/100k | latency_ms | - | 1 | positions_on | class=boolean, k=100 |
| query/boolean/k100/1m | latency_ms | - | 1 | positions_on | class=boolean, k=100 |

**Registered-host commands:**

```text
scripts/perf-runner.sh --gate QG-6 --hardware-class trj-zen3-5995wx --execution-profile physical-64 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-6 --hardware-class trj-zen3-5995wx --execution-profile smt2-128 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-6 --hardware-class m4-macos --execution-profile scheduler-10 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-6 --hardware-class m5-macos --execution-profile scheduler-14 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-6 --hardware-class x86-vps-ovh --execution-profile x86-diagnostic --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
```

### QG-7 — memory

- activated: `false`
- fixture: peak ingest RSS at equal budget config (medium, 8 threads); index bytes/doc on medium+xlarge
- target: peak RSS <= oracle; bytes/doc <= 1.15x oracle (positions ON) and <= 0.8x (positions OFF vs oracle positions-on default); budget DECLARES the 8-bytes/doc IDMAP content_hash (registry 1.0.2) and tombstone/blockmax/idmap overheads itemized

**Canonical cells (8):**

| fixture | metric | corpus | threads | positions | extras |
|---|---|---|---|---|---|
| memory/medium/positions_on | peak_rss_bytes | medium | 8 | positions_on |  |
| size/medium/positions_on | index_bytes_per_document | medium | 8 | positions_on |  |
| memory/medium/positions_off | peak_rss_bytes | medium | 8 | positions_off |  |
| size/medium/positions_off | index_bytes_per_document | medium | 8 | positions_off |  |
| memory/xlarge/positions_on | peak_rss_bytes | xlarge | 8 | positions_on |  |
| size/xlarge/positions_on | index_bytes_per_document | xlarge | 8 | positions_on |  |
| memory/xlarge/positions_off | peak_rss_bytes | xlarge | 8 | positions_off |  |
| size/xlarge/positions_off | index_bytes_per_document | xlarge | 8 | positions_off |  |

**Registered-host commands:**

```text
scripts/perf-runner.sh --gate QG-7 --hardware-class trj-zen3-5995wx --execution-profile physical-64 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-7 --hardware-class trj-zen3-5995wx --execution-profile smt2-128 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-7 --hardware-class m4-macos --execution-profile scheduler-10 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-7 --hardware-class m5-macos --execution-profile scheduler-14 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-7 --hardware-class x86-vps-ovh --execution-profile x86-diagnostic --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
```

### QG-8 — scaling curve (contention honesty)

- activated: `false`
- fixture: bulk indexing threads {1,2,4,8,16,(32 where applicable)} projected through each registered hardware/execution profile's typed applicability plan; oracle curve measured alongside
- target: 16-thread >= 1.8x own 4-thread on x86 reference; any plateau ATTRIBUTED (allocator contention axis mandatory — e8.4 notes; a bandwidth ceiling is honest, a lock plateau is a bug). Apple Silicon QG-8 uses the unchanged canonical literal widths and the typed execution-profile applicability plan: M4 scheduler-10 has capacity 10 but its widest runnable canonical cell is 8, so no width-10 cell or P/E residency claim may be invented; M4 promotion remains unavailable until actual scheduler-state and loaded-image witnesses bind the run and a reviewed profile-aware 8-vs-4 target/evaluator replaces the x86-only 16-thread requirement. M5 scheduler-14 remains unavailable with no capacity or executable cells until a real fingerprint and scheduler receipt land; no M4/x86 substitute or all-N/A release plan may activate an Apple claim

**Canonical cells (6):**

| fixture | metric | corpus | threads | positions | extras |
|---|---|---|---|---|---|
| scaling/xlarge/1/positions_on | docs_per_second | xlarge | 1 | positions_on |  |
| scaling/xlarge/2/positions_on | docs_per_second | xlarge | 2 | positions_on |  |
| scaling/xlarge/4/positions_on | docs_per_second | xlarge | 4 | positions_on |  |
| scaling/xlarge/8/positions_on | docs_per_second | xlarge | 8 | positions_on |  |
| scaling/xlarge/16/positions_on | docs_per_second | xlarge | 16 | positions_on |  |
| scaling/xlarge/32/positions_on | docs_per_second | xlarge | 32 | positions_on |  |

**Registered-host commands:**

```text
scripts/perf-runner.sh --gate QG-8 --hardware-class trj-zen3-5995wx --execution-profile physical-64 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-8 --hardware-class trj-zen3-5995wx --execution-profile smt2-128 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-8 --hardware-class m4-macos --execution-profile scheduler-10 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-8 --hardware-class m5-macos --execution-profile scheduler-14 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-8 --hardware-class x86-vps-ovh --execution-profile x86-diagnostic --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
```

### QG-9 — cold open

- activated: `false`
- fixture: 1M-doc index (xlarge), cold page cache (documented eviction method per OS); default config (durable verify OFF)
- target: open() <= 50ms (manifest + lazy sections) vs oracle reader open

**Canonical cells (1):**

| fixture | metric | corpus | threads | positions | extras |
|---|---|---|---|---|---|
| cold_open/xlarge/default | open_latency_ms | xlarge | 1 | positions_on |  |

**Registered-host commands:**

```text
scripts/perf-runner.sh --gate QG-9 --hardware-class trj-zen3-5995wx --execution-profile physical-64 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-9 --hardware-class trj-zen3-5995wx --execution-profile smt2-128 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-9 --hardware-class m4-macos --execution-profile scheduler-10 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-9 --hardware-class m5-macos --execution-profile scheduler-14 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-9 --hardware-class x86-vps-ovh --execution-profile x86-diagnostic --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
```

### QG-10 — dependency surface

- activated: `false`
- fixture: cargo tree -p frankensearch --features lexical, default features, post-flip
- target: tantivy + its transitive tree ABSENT from default graph; delta recorded in the flip evidence bundle (quill-e7.6)

**Canonical cells (1):**

| fixture | metric | corpus | threads | positions | extras |
|---|---|---|---|---|---|
| dependency_surface/default_lexical | tantivy_nodes | - | 1 | - |  |

**Registered-host commands:**

```text
scripts/perf-runner.sh --gate QG-10 --hardware-class trj-zen3-5995wx --execution-profile physical-64 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-10 --hardware-class trj-zen3-5995wx --execution-profile smt2-128 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-10 --hardware-class m4-macos --execution-profile scheduler-10 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-10 --hardware-class m5-macos --execution-profile scheduler-14 --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
scripts/perf-runner.sh --gate QG-10 --hardware-class x86-vps-ovh --execution-profile x86-diagnostic --run-id <unique-pass-id> --run-window <shared-window> [platform-required bounded options]
```
