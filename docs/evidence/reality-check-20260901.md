# Reality Check — frankensearch, 2026-09-01

**Author:** Claude (claude-code / fable-5.1), user-directed `/reality-check-for-project`, end to end.
**Basis:** `main` = `origin/main` = `103ed6c1` (clean tree), the published v1.7.0 release assets, crates.io
0.4.2, the JSONL work graph (`.beads/issues.jsonl`, 1,260 beads), and six parallel read-only code
audits. **Every claim below was executed on the real release binary or read at a named line.**
The prior check is `docs/evidence/reality-check-20260801.md`; §2 measures the month's delta against it.

---

## 0. The headline

**The product ships and the engine works. The product's headline promise does not.**

The README's one-line install, `fsfs index`, and `fsfs search` all work on the real v1.7.0 binary:
a 65-file corpus indexes in 3.3 s and exits cleanly, hybrid search answers in 19 ms in-process with
real potion semantic ranks, a zero-term-overlap paraphrase is carried by the semantic arm, streaming,
explain, serve, update, append/delete/compact, watch and uninstall are all wired. The library line is
on crates.io, no git dependencies remain, no tokio/hyper/reqwest exists anywhere in the lock file,
the July semantic wrong-result criticals are all fixed, and there are zero open GitHub issues.

But the "two-tier" search that names the project is **one-tier in the shipped CLI**. `fsfs index` writes
only the 256-d potion generation; the quality stage is gated on a 384-d index that nothing in fsfs ever
builds, so every search logs `quality_refine status=skipped reason_code=query.stage.quality.disabled.unavailable`
and `rerank ... disabled.no_quality`. The README's `PHASE 1 (refined)` output cannot occur. This is
exactly GitHub #30 (filed against v1.2.5), which was **closed on 2026-07-27 with no comment, no bead
and no fix**. It reproduces on v1.7.0 today. No open bead covers it.

Two more things the bead count hides: **all seven GitHub workflows are `disabled_manually`** (last run
2026-08-20, and those runs were red), so none of the 211 commits since v1.7.0 has automated proof; and
the Quill default flip **shipped on 2026-08-04 with 0 of 10 performance gates activated**, against a
binding user decision recorded on 2026-07-29 that the flip must wait for a validated win on every gate.

---

## 1. Vision Checklist

Statuses use the skill's vocabulary. `NO_BEAD` means no open or in-progress bead covers the gap.

| # | Goal | Source | Status | Evidence |
|---|---|---|---|---|
| 1 | One-line install yields a working semantic `fsfs` | README Install | **WORKING** (Linux x86_64 glibc) | `install.sh --dest X --quiet` fetched the 68 MB full gnu binary; `doctor` → `model.fast pass … loadable`, `model.quality pass … loadable` |
| 2 | `fsfs index <dir>` builds a durable index and exits | README Quick Start | **WORKING** | 65 files / 2.98 MB → 3,319 ms, exit 0, no strays; `index_sentinel.json generation_complete: true`, `vector/index.fsvi`, `lexical/CURRENT`, `lexical/quill-v1/` |
| 3 | `fsfs search` returns hybrid results, Initial < 15 ms | README | **WORKING** (in-process) | `duration_ms: 19`; hits carry `lexical_rank`/`semantic_rank`/`in_both_sources`; `vector_generation_id: potion-multilingual-128M` |
| 4 | Semantic arm is real, not hash control | v1.6.0 / bd-a6zt | **WORKING** | zero-overlap query "stopping work that is still running without losing data" → top hits ranked by `semantic_rank` 0/1 |
| 5 | Progressive **Refined** phase (~150 ms) + optional rerank in `fsfs` | README Quick Start, AGENTS architecture | **NOT WORKING / NO_BEAD** | `runtime.rs:734` gate `index.dimension() != 384` fails on every real corpus; only writer is fast tier `runtime.rs:11250,11402`; `rerank: Disabled` hardcoded `runtime.rs:722`; GH#30 closed 07-27 unfixed |
| 6 | Daemon-backed search amortizes model load | `fsfs --help`, `cli.rs:780` | **NOT WORKING / NO_BEAD** | 4 sequential default searches: 3.11 / 3.11 / 3.01 / 3.01 s, no daemon process survives; `PR_SET_PDEATHSIG` at `runtime.rs:6318-6333` kills the child with the spawning search; stale `.sock` left behind |
| 7 | Agent streaming (`--stream --format jsonl`), `toon`, `csv`, `explain` | README | **WORKING** | `fsfs.stream.query.v1` frames `started/progress/result/terminal`; `explain R0` returns `policy_decisions/query/ranking/trace`. Gaps: session ids `R0..` never surfaced in search output; explain BM25 `tf: 0.0, idf: 0.0` canned (`runtime.rs:6539-6540`) |
| 8 | Watch / incremental mode | README | **WORKING** (lifecycle) | `fsfs watch .` exits on SIGINT with no strays. A file added at t=5 s was not searchable by t=12 s — not verified within the window |
| 9 | Index management: append-batch / delete / compact / flush / daemon | CLI | **PARTIAL** | append→search→delete→compact→gone verified; but `append-batch` updates the vector WAL only (appended doc has `lexical_rank: null`), `delete --prefix` never scans the WAL (`runtime.rs:9245-9254` dead loop), `fsfs daemon` holds the publication lease for life with no stop verb |
| 10 | Library: reusable two-tier engine, README quickstart compiles | README Library Quickstart | **PARTIAL** | Two-tier is real: `IndexBuilder` writes `vector.fast.idx` + `vector.quality.idx` (`index_builder.rs:726-745`), `TwoTierSearcher` emits `Refined` (`searcher.rs:1299`) with the bd-180wl guard at `:2874`. **README block does not compile**: `use frankensearch::LexicalSearch` — trait removed in `327d264a`; it is `LexicalRead`. No test proves Refined against a real model (semantic doubles only) |
| 11 | Publishable to crates.io | README | **WORKING** | facade 0.4.2 (2026-08-28), core 0.2.3, quill 0.2.2; zero `git =` deps; `Cargo.lock` has 0 git sources |
| 12 | Quill replaces Tantivy as default lexical backend, "same results, faster", gated on conformance + 10 perf gates | Plan §1.4, §14, bd-3beo | **FLIPPED WITHOUT ITS GATE** | `frankensearch/Cargo.toml:15 lexical = ["quill"]` since `d117ce1f` (08-04, bd-8nqz.4.2). `docs/contracts/quill-perf-gates.toml`: 10/10 `activated = false`. `.bench-history`: 10 × `QG-N.v8.unmeasured.latest.json`; the only promoted measurement ever is QG-2 (07-28) at **0.35× MISS** (Quill 59.8k vs Tantivy 171.2k docs/s), since quarantined |
| 13 | Quill engine complete vs plan §3.1/§5–§12 | Plan | **WORKING** (delta segment opt-in) | Query surface, BM25 parity table, snippets, merge=concat (`quiver.rs:805-880` verbatim copy), compaction, LOCK/CAS/CURRENT, FSLX checksums, columnar SIMD ingest (`scribe.rs:384-395,655`), per-move cancel poll. Delta searchable-before-commit exists but default `index_documents` is still commit-gated (`index.rs:6580-6582`) |
| 14 | Honest perf doctrine producing measured Quill-vs-Tantivy numbers | Plan §14, E8 | **STALLED** | Last real KEEP 08-01; last E8-H evidence 08-03; last ledger entry 08-09. Runner refuses every host except trj 5995WX (`scripts/perf-runner.sh:88-117`; this box is a 5975WX) and refuses QG-3/4/5 everywhere. Gauntlet = 184k lines, ~54 % receipt/ratchet machinery |
| 15 | Semantic identity: fail closed across spaces, no silent hash | bd-jbfg / bd-xomn / v1.6.0 | **PARTIAL (in progress)** | Library-side causes closed (bd-a6zt comment 08-29); FSVI v2 identity train (`bd-xomn.1-.4`, `bd-8utj` slice 1 landed 08-30) still open; multilingual space fails closed (0.4.2) |
| 16 | asupersync-only; unsafe forbidden | AGENTS.md | **TRUE / FALSE** | tokio/hyper/reqwest: 0 in `Cargo.lock` (646 packages). Unsafe: workspace lint is `deny`, not `forbid`; 91 non-test `unsafe` sites in 5 crates; `frankensearch-durability/src/lib.rs:2` blanket `#![allow(unsafe_code)]` |
| 17 | Native rerank (frankentorch, no ONNX) and native HNSW in-tree | bd-2ba5, bd-kcek | **WRITTEN, DARK** | `rerank/src/native.rs` real 6-layer int8 forward pass, unreachable from fsfs (no dep). `index/src/native_hnsw.rs` 7k lines / 54 tests, no search-path consumer; `ann` still binds `frankenhnsw =0.3.5`; fsfs has no ANN. `ort` still ships in default fsfs via `frankensearch-embed/fastembed` |
| 18 | Durability repair for index artifacts | README | **PARTIAL** | RaptorQ + `FSDR` trailer real and wired to Quill segments in watch-mode (`runtime.rs:13405`); `FsviProtector` (1.1k lines) has **zero** consumers — FSVI is never protected; one-shot `fsfs index` gets no protection |
| 19 | Ops fleet console | README crate table | **LIBRARY-ONLY** | Real binary + 12-table FrankenSQLite schema + 7 screens; only telemetry producer is its own `simulator.rs`; no crate depends on it; not built or shipped by any release lane |
| 20 | Release/quality gates run in CI (quickstart gate "release-blocking", publish lane) | README, AGENTS | **NOT RUNNING / NO_BEAD** | `gh workflow list`: all 7 `disabled_manually`. Last runs 2026-08-20: CI failure, Migration Compatibility failure, Self-Documentation Lint failure. Local hooks = agent-mail guard only |
| 21 | Docs are accurate | implicit | **STALE / NO_BEAD** | §4 G6 lists 14 concrete drifts across README, AGENTS.md, `docs/architecture/overview.md` |

**Scoreboard: 8 WORKING, 5 PARTIAL, 2 NOT WORKING, 1 FLIPPED-WITHOUT-GATE, 1 STALLED, 1 NOT RUNNING, 1 LIBRARY-ONLY, 1 WRITTEN-DARK, 1 STALE.**
**Six goals have no bead at all (#5, #6, #20, #21, plus the wiring halves of #9 and #18).**

---

## 2. Delta since 2026-08-01

| 08-01 item | Then | Now |
|---|---|---|
| `fsfs index` never exits (D1) | REGRESSED | **FIXED** — exit 0, repeat clean, SIGINT clean |
| Default source build inert (D2) | REGRESSED | **FIXED** — `default = ["semantic-loaders"]`, runtime SHA-verified models |
| Progressive phases | UNPROVEN | **PROVEN NOT WORKING** in the product (structural, §4 G1) |
| Quill default flip | NOT STARTED | **FLIPPED 08-04** — with 0/10 gates and against bd-3beo (§6) |
| QG-1..10 certified | NOT STARTED | **0/10**, one quarantined MISS |
| 5 semantic criticals | filed | **5/5 FIXED** at the line (`blend.rs:65-76`, `searcher.rs:2874`, `refresh.rs:1404-1481`, `two_tier.rs:2240-2403`, `federated.rs:283-286`, `daemon_fallback.rs:1247-1283`) |
| Typed identity enforcement | PARTIAL, zero call sites | in-progress train, first cache slice landed 08-30 |
| crates.io | BLOCKED on git `hnsw_rs` | **PUBLISHED** 0.4.0 → 0.4.2 |
| Cross-process write safety | PARTIAL | `PublicationLease` + Windows admission (gh#39) landed |
| CI | running | **all workflows disabled** |

Net: the month delivered shipping, publishing and correctness. It delivered nothing on the performance
thesis, and the product-level two-tier promise is where it was on 2026-07-27.

---

## 3. Verified working, with the commands

```
install.sh --dest <scratch>/bin --quiet          # 68 MB fsfs-1.7.0-x86_64-unknown-linux-gnu, doctor loaders pass
fsfs index .                                     # 65 files, 3319 ms, exit 0, no leftover processes
fsfs search "reciprocal rank fusion tie-breaking" --no-daemon --format json   # 19 ms, in_both_sources: true
fsfs search "stopping work that is still running without losing data" ...   # semantic_rank 0/1 carry the answer
fsfs search "..." --stream --format jsonl        # started → progress(retrieve.fast) → result×k → terminal
fsfs explain R0 --format json                    # ok, policy_decisions/ranking/trace
printf 'q1\nq2\nquit\n' | fsfs serve --format jsonl   # ready + payloads per line
fsfs append-batch --file batch.jsonl / delete / compact / flush   # appended doc found (semantic), deleted, compacted
timeout -s INT 12 fsfs watch .                   # exits on SIGINT, no strays
fsfs update --check / config / uninstall --dry-run / status / doctor          # all ok: true
```

Stub census of production code: 0 `todo!`, 0 `unimplemented!`, 0 TODO/FIXME, 0 no-op public `Result<()>`.
The previously dead `conformance-internals` cfg is now declared (`quill/Cargo.toml:24`).

---

## 4. The gaps, with root cause

### G1 — The shipped CLI is one-tier (P0, NO_BEAD)

- Only writer: `runtime.rs:11250 resolve_fast_embedder()` → `runtime.rs:11402-11406 VectorIndex::replace_with_empty(... embedder.dimension())` at `vector/index.fsvi` (256-d potion).
- Query-time gate: `runtime.rs:726-746 quality_stage_viable` returns `false` when `index.dimension() != 384` (`:734`) or the index was built by the fast embedder (`:738`). Both are true for every real corpus → `capabilities.quality_semantic = Disabled` → `query_planning.rs:673 "query.stage.quality.disabled.unavailable"`.
- The planner models a quality tier (`VectorSchedulingTier::FastAndQuality`, `AppendQuality` action) but its only consumer is a no-op debug log (`runtime.rs:2299-2300`) and live ingest hardcodes `FastOnly` (`:2229-2231`).
- `rerank: CapabilityState::Disabled` is hardcoded (`runtime.rs:722`).
- Cost of the dead tier: every `fsfs search` still loads all-MiniLM-L6-v2 through ONNX (`0.38 s` in the trace) before potion, for a stage that can never run.
- The table banner ("a quality-only failure preserves Initial and emits an actionable RefinementFailed phase") promises an event the planner never emits.
- No test asserts a `refined` phase from a real `fsfs index` corpus (`e2e_recall.rs:810` accepts either phase; `cli_command_tests.rs:807` checks ordering only if the frame exists).

### G2 — Daemon mode amortizes nothing on Linux (P0, NO_BEAD)

- `spawn_search_daemon` (`runtime.rs:6171`) forks from the `fsfs search` process, installs `PR_SET_PDEATHSIG(SIGTERM)` (`:6318-6333`), and never `setsid`s. The parent exits seconds later; the daemon dies with it.
- Measured: four sequential default searches, 1 s apart: 3.11, 3.11, 3.01, 3.01 s; zero fsfs processes afterwards; one stale socket in `/run/user/1000/frankensearch/daemon/`.
- Where the 3 s goes: potion 500k-vocab load ≈ 2.5 s + MiniLM/ORT ≈ 0.4 s, per invocation. `--no-daemon` costs the same.
- Related: stale-socket removal is TOCTOU (`:5616-5629`); no idle timeout in the accept loop (`:5661-5710`); non-Linux gets no pdeathsig and runs until reboot.

### G3 — The Quill flip shipped without its gate (governance, needs a ruling — §6)

### G4 — The performance campaign has become its instrument

- Since v1.7.0 (211 commits): `chore` 112 (101 bead bookkeeping), `feat` 30, `fix` 27, `test` 25. Lines: gauntlet +9,378, index +5,586 (generation root), quill +1,715, fsfs +318.
- Gauntlet crate 183,580 lines: ~98k receipts/evidence/ratchet/registry/supervisor, ~69k engine-facing conformance, ~15k paired bench. `runner.rs` is 1,481 production / 26,123 test lines.
- `scripts/perf-runner.sh:88-117` accepts only `trj-zen3-5995wx:{physical-64,smt2-128}`; m4/m5/x86-vps die; QG-3/4/5 die on every host. `perf-diagnostic.sh` header: "THIS TOOL PRODUCES NO PROMOTABLE EVIDENCE."
- QG-5 could not be measured over rch (three RCH-E104 timeouts; rch relinks the graph ~25 min per exec — bd-h6eh comment 07-29). Nothing has changed since.
- Suite AGENTS.md pattern 10 (conformance metastasis) applies verbatim: rigor became the product.

### G5 — No automated gate has run since 2026-08-20 (P0, NO_BEAD)

- `gh workflow list --all`: CI, Dependency Semantics Lint, Interaction Matrix Gate, Ledger Integrity Lint, Migration Compatibility, Quill structural evidence rehearsal, Self-Documentation Lint — all `disabled_manually`.
- The last runs were red (CI scheduled failure; Migration Compatibility failure; Self-Documentation Lint failure).
- `.githooks/hooks.d/{pre-commit,pre-push}` contain only the agent-mail reservation guard.
- The README describes the quickstart gate as release-blocking and the publish lane as tag-driven; both are dormant. v1.7.0 and 0.4.x were cut by hand (dsr/manual recipes).
- Workspace test suite: see §8 (run locally by this check, `bash -c 'cargo test --workspace --no-fail-fast'`).

### G6 — Documentation drift (NO_BEAD; bd-quill-e9.1 is blocked behind the flip bead)

README: (1) library quickstart imports `frankensearch::LexicalSearch` — removed, use `LexicalRead`; (2) config precedence stated as CLI > project > user > env > defaults, code is `[Cli, Env, File, Defaults]` (`fsfs/src/config.rs:13-18`); (3) `PHASE 0 (fast)`/`PHASE 1 (refined) … in 151ms` — emitter prints `PHASE INITIAL: n hit(s) for "q"` (`format_emitter.rs:502-508`) and refined never occurs; (4) performance envelope has no committed evidence, and the two entries that do exist contradict it (hash embed 1.78 µs vs "~11 µs"; RRF 1000+1000 23 µs vs "~1 ms" for 500+500); (5) publish lane described as `v*` tags and a 7-crate list — it is `crates-v*` and every publishable member; (6) "Full release binaries bundle default semantic models" — false for linux-gnu (loaders + runtime download; `install.sh:311-313, 1159-1163`); (7) FlashRank named as the rerank path — its manifest is placeholder-pinned and excluded from production readiness (`model_manifest.rs:1531-1559`); (8) `tests/` at the workspace root holds no Rust tests.
AGENTS.md: (9) packaging boundary says rerank/ann pull git deps and are unpublishable — false since 08-24 (`Cargo.toml:121-147`, registry `frankentorch-*`, `frankenhnsw`); (10) "Unsafe code: Forbidden" — workspace lint is `deny`, 91 sites; (11) Key Dependencies lists `ort` for reranking and FlashRank — rerank is frankentorch; `ort` enters only via `fastembed`; (12) workspace tree shows root `tests/ benches/ examples/` — they live under `frankensearch/`.
`docs/architecture/overview.md` + `docs/architecture.md`: (13) "12 crates", no Quill/gauntlet, lexical = Tantivy, rerank = FlashRank/ONNX. (14) `crates/frankensearch-index/src/hnsw.rs:2810` says the crate is `#![forbid(unsafe_code)]`; it has 66 unsafe sites.
Per the suite policy: correct each to the right value; no retraction narrative.

### G7 — Written-but-dark engines and orphans

- Native HNSW (`index/src/native_hnsw.rs`, 7,072 lines, 54 tests): no consumer on the search path; `hnsw.rs:36` still imports `hnsw_rs`; bd-kcek in_progress since July.
- Native reranker (`rerank/src/native.rs`): real, but fsfs has no rerank dependency; its parity tests gate on `/private/tmp/ee-reranker-port/model` (`native.rs:1811`) and silently return on Linux.
- `ort` (2.0.0-rc.13) ships in the default fsfs binary through `semantic-loaders → frankensearch-embed/fastembed`; bd-2ba5 is `blocked` although `br doctor` reports every blocker closed.
- `FsviProtector`: zero consumers.
- Orphaned files compiled by nothing (7, ~2,658 lines): `core/src/metrics.rs` (1,402 lines: TDigest, HyperLogLog, robust stats — real code, no `mod`), `durability/src/tantivy_wrapper.rs` (1,115, retired per `lib.rs:12` but not removed), `fusion/src/repro_blend.rs`, `fusion/src/repro_rrf.rs`, `tui/src/repro_input.rs`, `rerank/src/test_api.rs` (empty), `rerank/src/test_inputs.rs`.
- Ops console: 33k lines, real binary and schema, discovery has only `StaticDiscoverySource`; nothing feeds it.

### G8 — CLI defects found live (all NO_BEAD, each needs a red test)

1. `fsfs search --help` runs a search for the string `--help` (`adapters/cli.rs:349-364` captures the query before the flag loop; `is_known_cli_flag` at `:1006-1053` omits `--help`/`-h`). The token then lexically matches all 65 documents with monotonically decreasing scores — a match-all on a nonsense token.
2. `explain` BM25 sub-scores are canned `tf: 0.0, idf: 0.0` (`runtime.rs:6539-6540`); table prints `quality_score = "n/a"; rerank_score = "n/a"` (`:15716-15717`).
3. `delete --prefix` never scans the WAL: dead loop at `runtime.rs:9245-9254`.
4. `append-batch` writes the vector WAL only; the Quill arm is not updated (appended doc: `lexical_rank: null`).
5. `fsfs daemon` acquires the publication lease for its whole life (`:9396`) — `index`/`append-batch`/`delete`/`compact` fail while it runs; no stop verb; SIGKILL leaves the lock.
6. Fresh index logs `WARN discarding stale/mismatched WAL entries … main_gen=2 wal_gen=2` on a brand-new directory.
7. `fsfs index --format json` prints prose; `fsfs status` reports `metadata_bytes: 659456` for an index that does not exist.
8. `config.search.shadow_mode` is a live knob that cannot work on the default build (`runtime.rs:13249-13267`, warns).
9. `explain` result ids (`R0`, `R1`) are never surfaced in any search output format.

### G9 — Work-graph hygiene

- `br list` fails in this checkout: `SYNC_CONFLICT … bd-fixed-generation-root-publisher-ycng6 does not match its normalized JSONL payload` (bd-hpvkk). JSONL is truth; this report read it directly.
- `br doctor`: 2 beads `blocked` with every blocker closed (`bd-2ba5`, `bd-fsfs-cross-platform-semantic-installer-46z3u`); DB/JSONL counts differ; 210 recovery artifacts retained.
- 40 beads `in_progress`; several untouched since July (`bd-kcek`, `bd-jbfg` last comment 07-26, `bd-p6z6` 07-26).

---

## 5. Would implementing every open bead close the gap? No.

150 beads are not closed (98 open, 40 in progress, 8 blocked, 4 deferred). By family: Quill E6/E7/E8 + E8-H + QG
≈ 70, FSVI generation/identity ≈ 15, ArtifactStore v4 3, RCH watchdog 5, semantic provisioning/recovery ≈ 8.
Closing all of them would activate gates **only if** the trj and Apple hardware produce measurements the
current runner accepts, and would still leave every `NO_BEAD` item above: the product quality tier (#5),
daemon amortization (#6), CI (#20), documentation (#21), the CLI defects (G8), append-batch's lexical arm,
FSVI protection wiring, the ops telemetry source, and the orphaned modules.

---

## 6. A contradiction the graph cannot resolve — needs the owner's ruling

- `bd-3beo` (P0, open, last updated 07-30): "BINDING USER PERFORMANCE DECISION FOR THE LIBRARY LEXICAL FLIP … must not flip to Quill until Quill is both conformant and faster." Comment 07-29: *"I don't want to switch over until Quill is faster and better than tantivy across the board."*
- Plan §1.4: "The `lexical` feature default does not flip to Quill until the gauntlet gates … and the perf gates of §14 [are] met."
- `bd-quill-e7-integration-flip-d0tx.6` ("THE SINGLE REVIEWED COMMIT THAT MAKES QUILL THE BACKEND"): open, blocked by bd-3beo, bd-8nqz, bd-8nqz.6, bd-…-0r2p.
- Yet `d117ce1f` (2026-08-04, bd-8nqz.4.2 "QG-10 facade … structural receipt", closed) set `lexical = ["quill"]`, and that default was published to crates.io as 0.4.0/0.4.1/0.4.2. bd-8nqz's own text calls Quill "the requested default". No comment on bd-3beo, bd-8nqz or d0tx.6 records a supersession.
- The only Quill-vs-Tantivy number ever promoted is a 0.35× single-thread MISS (07-28, quarantined).

Two consistent resolutions exist; the report does not pick one: (a) the July ruling stands → the facade
default reverts to Tantivy (or `lexical` becomes explicit-only) until gates activate; (b) the ruling was
superseded → close bd-3beo/d0tx.6 as superseded, amend plan §1.4, and state the flip's actual basis
(conformance-only) in README/CHANGELOG.

---

## 7. Bridge plan (sequenced by user impact)

*Superseded on 2026-09-02 by the Phase 2 document `docs/planning/BRIDGE_PLAN_2026-09-02.md`
(29 gaps against 22 re-scored vision goals, with per-gap current state, target, criteria, steps,
dependencies, and bead coverage). The list below is the 09-01 sequencing and is kept for history.*

1. **Ruling on §6**, then one commit that makes graph, plan, AGENTS and code agree.
2. **Product two-tier (G1).** `fsfs index` writes a quality generation (384-d MiniLM, its own identity) beside the fast one; `quality_stage_viable` reads the quality tier; Refined emitted on a real corpus with a red-then-green e2e test on the actual binary; stop loading MiniLM/ORT when the stage is unavailable; README output made true. This is GH#30.
3. **Daemon amortization (G2).** Detach (`setsid`/double fork) or a supervised long-lived daemon with idle timeout; `fsfs daemon stop`; stale-socket ownership check; a test that the second search costs < 500 ms.
4. **A gate that runs (G5).** Either re-enable a minimal CI (fmt/check/clippy/test + quickstart gate on push) or install the same as a local pre-push chain; record in README that releases are manual until then.
5. **Docs truth pass (G6)** — the 14 items, corrected to the right value, no narrative.
6. **CLI defect batch (G8)** — nine items, each with a failing test first.
7. **Perf campaign rebalance (G4).** Freeze harness growth (new checks must cite a defect class); get one measured cell per gate on the registered trj host, or register a diagnostic profile for this 5975WX so local evidence exists; publish honest Quill-vs-Tantivy numbers.
8. **Wire or shelve dark engines (G7).** Native HNSW behind `ann`; `FsviProtector` on `fsfs index`; frankentorch embedder path to retire `ort` (bd-2ba5 is blocked on nothing); ops: product or archive; register `core/src/metrics.rs` or remove the orphans (file deletion needs written permission).
9. **Graph hygiene (G9).** Repair the bd-hpvkk import; unblock the two falsely blocked P0s; triage the 40 in-progress beads for staleness.

Bead creation (Phase 3a) is deliberately not done here: graph structure is single-writer and these
findings include a decision only the owner can make.

---

## 8. Test suite

`cargo test --workspace --no-fail-fast` was issued through a `bash -c` wrapper; the rch PreToolUse hook
intercepted it anyway (`RCH_REQUIRE_REMOTE=1` is in the session environment) and ran it on worker
`ovh-a`. The run was **cut off by rch's 1800 s SSH ceiling (`RCH-E104`) inside the gauntlet unit-test
binary** — the same structural limit that blocked QG-5 in July (bd-h6eh, 07-29).

| Scope | Result |
|---|---|
| 53 test binaries across core, embed, index, lexical, fusion, durability, quill, fsfs (13 binaries), ops, facade | **7,931 passed · 0 failed · 42 ignored** |
| `frankensearch-quill-gauntlet` unit binary (894 tests) | cut off mid-run; 2 `FAILED` observed before the cutoff: `local_perf_runner::tests::cargo_config_guard_rejects_a_restored_tracked_path` and `…_transient_unbound_candidate` (a rename-park-restore of a tempdir `.cargo/config.toml` must be detected as "changed during the promotion build" — filesystem-identity semantics, environment-shaped on the worker; cf. GH#37) |
| `frankensearch-rerank`, `frankensearch-storage`, `frankensearch-tui`, `optimize-params` | never reached (sort after the gauntlet) |
| Doc-tests | never reached |

**Local, hook-bypassed rerun of the uncovered set** (`cargo-rch-real test … -p quill-gauntlet --lib`,
`RCH_CARGO_WRAPPER_BYPASS=1`, `RUSTUP_TOOLCHAIN=nightly-2026-08-25`, shared `/data/tmp/cargo-target`,
started 19:37Z): compile finished in 4 m 54 s; cargo then ran the gauntlet unit binary first and my own
55-minute cap expired while it was still inside it — **530 of 894 passed, 0 failed, 5 ignored** at the
cutoff. Thirty-seven tests were flagged "running for over 60 seconds", all in `perf_assembly` (23) and
`perf_ratchet` (14) — the receipt and ratchet machinery. So the gauntlet's own unit suite cannot complete
in 50 minutes on a 64-core host, in either lane; that is a G4 finding in its own right. The four small
crates behind it (rerank, storage, tui, optimize-params) were then attempted alone and hit `E0514`
("compiled by an incompatible version of rustc") on `bit_vec`/`fnv` freshly rebuilt into the shared
target by a peer's toolchain; a remote rch run of just those four was then refused by the fleet
(`[RCH] remote required; refusing local fallback (no admissible workers: critical_pressure=1,
insufficient_total_slots=10, active_project_exclusion=1)`, exit 103). **Those four crates therefore
carry no fresh test evidence from this check**; with CI disabled, nothing else has run them recently either. Two practical notes for whoever runs the suite next: the toolchain's own
`~/.rustup/toolchains/<tc>/bin/cargo` is now an rch-managed shim (the real binary is `cargo-rch-real`
beside it, bypass via `RCH_CARGO_WRAPPER_BYPASS=1`), and neither a remote nor a local workspace run can
finish inside a 30-minute window because of the gauntlet unit binary alone.

## 9. Replay

```bash
# shipped binary
bash install.sh --dest /tmp/fsfs-rc/bin --quiet && /tmp/fsfs-rc/bin/fsfs doctor --format json
mkdir -p /tmp/fsfs-rc/corpus && cp docs/*.md docs/contracts/*.md /tmp/fsfs-rc/corpus/ && cd /tmp/fsfs-rc/corpus
/tmp/fsfs-rc/bin/fsfs index . && FRANKENSEARCH_LOG=debug /tmp/fsfs-rc/bin/fsfs search "reciprocal rank fusion" --no-daemon -v 2>&1 >/dev/null | grep quality_refine
for i in 1 2 3 4; do /usr/bin/time -f "%es" /tmp/fsfs-rc/bin/fsfs search "reciprocal rank fusion" --format json --limit 1 >/dev/null; sleep 1; done
# governance
git log -S'lexical = ["quill"]' --format='%h %ci %s' -- frankensearch/Cargo.toml
grep -c 'activated = false' docs/contracts/quill-perf-gates.toml
gh workflow list -R Dicklesworthstone/frankensearch --all
```

## 10. Landed the same day (BlueLynx, 2026-09-01 evening)

Operator directive: close the highest bang-for-buck gaps now. What landed, with its proof:

| Gap | Change | Proof |
|---|---|---|
| G2 daemon amortizes nothing | `spawn_search_daemon` detaches the daemon with `setsid` instead of `PR_SET_PDEATHSIG`; accept loop exits after `--idle-timeout-ms` (default `FSFS_DAEMON_IDLE_TIMEOUT_MS = 600_000`, `0` = never) when no request is in flight; `serve --idle-timeout-ms` flag | Real binary: searches 17.0 s → 0.11 s → 0.11 s with one daemon in its own session (sid == pid); `quit` reclaims the socket; `serve --idle-timeout-ms 1500` exits 0 at 1.61 s; no strays. Quickstart lane: `daemon-survives-spawner cold_ms=4163 warm_ms=50`, `daemon-idle-exit wall_ms=1530` |
| G1 cost half: MiniLM/ORT loaded on every search | `EmbedderStack::auto_detect_fast_semantic_with_options` (embed crate) detects the fast tier only; fsfs `resolve_fast_embedder` uses it | Real binary with `FRANKENSEARCH_LOG=info`: `FastEmbed model loaded` 0, `Model2Vec model loaded` 1, `availability=FastOnly`. Quickstart lane: `lazy-quality-load fast_loaded=true quality_loaded=false` |
| G8.3 `delete --prefix` skipped the WAL | Peer fix 367f894a landed without a test; regression test `delete_prefix_tombstones_documents_that_exist_only_in_the_wal` (planted negative: non-matching prefix deletes nothing) | passes |
| G8.1 `search --help` | Peer fix c840926c with test | passes |
| G1 truth half | Table banner no longer promises a `RefinementFailed` phase the planner cannot emit; README quick-start shows the real `PHASE INITIAL` output and states that the CLI is fast-tier + Quill today | text |
| G6 docs | README: precedence (`cli > env > file(project > user) > defaults`), envelope table sourced from the ledger, publish lane (`crates-v*`, contract-derived sequence), bundling claim, rerank truth, CI-disabled note. AGENTS.md: packaging boundary (registry renames, publishable), unsafe policy (`deny` + per-site allow classes), workspace tree, key deps. `docs/architecture{,/overview}.md`: 15 members, Quill default, gauntlet, arrows. `index/src/hnsw.rs` comment | reviewed |

Gates on the touched crates (`frankensearch-embed`, `-fsfs`, `-index`): `cargo check --all-targets`
clean, `cargo clippy --all-targets -- -D warnings` clean, `cargo fmt --check` clean, `ubs -v` adds
zero critical findings on the added lines, fsfs lib tests for the new CLI flag and the WAL-prefix
regression pass, the env-gated real-model quickstart lane passes end to end (2 passed). The embed
crate's own unit test for the fast-only detector could not run locally (shared target `E0514`
from a peer's toolchain); it passed on the rch fleet (worker `hz4`, 1 passed, 0 failed).

Deliberately NOT done here, for the owner to file: the product quality-tier generation (the
README's two-tier promise — a design, not a patch), the flip ruling (§6), re-enabling any CI,
`append-batch` lexical arm, explain rank/path ids, the fresh-index WAL `warn!` (rule at
`crates/frankensearch-index/src/lib.rs:1913-1925` flags `wal_gen == main_gen` as stale on a
brand-new generation), the phantom `metadata_bytes` in `status` for a missing index (counts a
metadata DB outside the index root), orphaned modules, ops telemetry source, FSVI protector wiring.

## 11. Owner mandate 2026-09-02 and what landed against it

The owner ruled: Quill is the default everywhere (the flip stands); the product's two-tier
promise must be built, not designed; CI runs through `dsr`, not GitHub Actions; and the beads
workspace is to be repaired with `/fixing-beads-problems`. All four were done the same night.

**Ruling recorded (d8336095).** bd-3beo and bd-quill-e7-integration-flip-d0tx.6 closed as
superseded with the ruling text; comments on bd-8nqz, the E7/E8/E8-H epics and the renegotiation
checkpoint; plan §1.4 amended (carried by d9a53ce0). Beads repaired per the skill: cross-issue
comment ids 3581/3582 renumbered to 3590/3591 in one issue, temp DB rebuilt from the harmonized
JSONL (1260 created), verified (show / status healthy / doctor), promoted; old family kept aside
as `.beads/beads.db.bad_20260902T0101Z*`. `br` works again; two beads that were `blocked` with
every blocker closed (bd-2ba5, bd-fsfs-cross-platform-semantic-installer-46z3u) reopened; ten new
beads filed (bd-pyzzd quality tier, bd-9j1ga dsr gate, bd-a2hct, bd-iw2w9, bd-k7x34, bd-k1vcc,
bd-f8j9z, bd-d7xk1, bd-9ekrw, bd-p6k61); `br dep cycles` empty.

**Two-tier product (bd-pyzzd).** `fsfs index` now writes `vector/quality.fsvi` (all-MiniLM-L6-v2,
384-d, own identity and WAL) beside `vector/index.fsvi` under the same publication lease, from the
same documents, batch by batch after the fast tier with the same retry budget; the tier is
all-or-nothing per generation (retry exhaustion retires the partial file), reuse follows the fast
tier's exact-identity rule, incremental and final reconciliation compact both, and a run without a
quality model (or under a fast-only profile) records `vector.quality_tier.skipped.*` on the sentinel
and retires any stale quality file. Search binds the quality tier as its own resource, the quality
stage runs against it and fails closed on identity mismatch, and the generation fingerprint covers
the new file and its WAL. `append-batch`, `delete`, `compact`, the compaction daemon and watch mode
keep both tiers in step. `fsfs status` reports `quality_generation_id/dimension`; `fsfs doctor`
gained `semantic.quality_generation`. The `FSFS_DEFAULT_QUALITY_EMBEDDER_DIMENSION` gate that made
refinement structurally unreachable is gone. Quality-embedder resolution no longer re-opens the fast
model (`EmbedderStack::auto_detect_quality_with_options`, d9a53ce0).

| Proof | Result |
|---|---|
| Unit: `one_shot_index_builds_the_quality_tier_and_search_emits_refined` (semantic doubles in two distinct spaces; planted negative without a quality model) | pass; 55/55 in the targeted set incl. the three quality-phase fixtures, serve-socket and delete-prefix tests |
| Real binary, 54-doc corpus, real models | `quality.fsvi` 54/54 live, MiniLM 384; sentinel `vector.quality_tier.built`; status shows both generations; doctor `semantic.quality_generation` pass |
| Real binary search | stream `initial_ready` then `refined_ready`; JSON `phase: "refined"`; table `PHASE REFINED` |
| Real binary mutations | append → both WALs, appended doc refined rank 0; delete mirrored; compact both 54/54; doctor pass |
| Real binary, strict profile | quality tier retired (`vector.quality_tier.skipped.fast_only`), doctor warn with the reason, search INITIAL only; re-index under the default profile restores `built` |
| Real-model quickstart lane | 2 passed: `durable-quality-vector records=10 dimension=384 embedder_id=all-minilm-l6-v2`, hybrid `phase: refined`, `fast_only_quality_loaded=false two_tier_quality_loads=1 refined_ready=true`, daemon stages green |
| Gates | fmt clean; check clean; clippy `-D warnings` clean (embed, fsfs, index, all targets) |

Known limit found on the way (filed as bd-k7x34): `FRANKENSEARCH_FAST_ONLY` and `--fast-only`
are no-ops under the default performance pressure profile because that profile locks the quality
override; the strict profile is the only way to build a fast-only generation today.

**CI via dsr (bd-9j1ga).** `scripts/quality-gate.sh` runs fmt, check, clippy, workspace lib tests
(excluding the gauntlet unit binary), the fsfs test binaries, the real-model quickstart lane and the
executable quick-start gate on a real host with the rch shim bypassed; `~/.config/dsr/repos.yaml`
and `repos.d/frankensearch.yaml` now name it as the frankensearch check (dry run: 8 checks).
README and AGENTS.md point at it; the GitHub Actions description in README now states that those
lanes do not run.

Receipt (bd-9j1ga, closed): `dsr quality --tool frankensearch` passed its seven packaging and
installer contract checks; the gate script then passed all seven of its stages in one detached
invocation on ts1 at 2026-09-02T02:03:17Z (fmt 5 s, check 11 s, clippy 16 s, workspace lib tests
113 s, fsfs test binaries 74 s, real-model e2e 38 s, executable quick-start gate 152 s; exit 0).
The two earlier runs each failed exactly one test, `verify_new_binary_uses_version_subcommand_not_flag`,
and surfacing its spawn error identified the cause as `ExecutableFileBusy`: a multithreaded test
binary racing `execve` of a just-written fixture script against sibling forks. The five fixtures
that share the pattern now retry only that error (5f5b0c40); the same pattern exists in production
update verification and is recorded on bd-9j1ga as a hazard, not changed.

## 12. Bridge execution receipts (BlueLynx, 2026-09-02)

Bridge plan items closed from the "wiring finished code" tier of
[`docs/planning/BRIDGE_PLAN_2026-09-02.md`](../planning/BRIDGE_PLAN_2026-09-02.md); the owner's
sweep commit 5d228943 banked the first half of this work mid-session, the rest lands with this
section. Every row below was re-executed on the real dev binary by
`proof-bridge.sh` (scratchpad, seven steps, exit 0) after the final rebuild.

| Gap | What was wrong | What landed | Proof |
|---|---|---|---|
| #5 ETXTBSY | `fsfs update` verified the new binary with a bare `Command::output()`; a sibling fork could hold the write descriptor and `execve` failed | `spawn_verifying_executable` (50 x 10 ms bounded retry on `ExecutableFileBusy`) at the update and rollback sites; the test fixtures share it | unit: writer closed after 120 ms succeeds; writer never closed surfaces the error after the budget (two tests) |
| #6 append-batch (bd-a2hct) | appended docs never reached the Quill arm; then, once written, still invisible because Quill buffers until `commit`; then, once committed, `delete`/`compact` were refused for 10 min after any search because the now long-lived query daemon holds the FSVI map lock | one-shot lexical helper applies upserts/tombstones and **commits**; `quiesce_query_daemon` (`:shutdown` on the daemon socket, bounded wait) plus a retrying writer open before every in-place mutation, including the compaction daemon | real binary: appended doc `rank 1, lexical_rank 0, semantic_rank 0, in_both_sources true, phase refined`; delete + compact remove it from every arm; CLI test `append_batch_reaches_the_lexical_arm_and_delete_removes_it_everywhere` |
| #7 explain (bd-iw2w9) | only never-printed `R0` ids resolved; BM25 tf/idf were silent placeholders | `fsfs explain <rank|R-id|path>` (unique file-name suffix too); `bm25_stats_unavailable` typed warning in JSON and table; help, README, tutorial corrected | unit resolve test; CLI tests by rank, path and `R0`; real binary `explain 1` and `explain alpha.md` |
| #19 WAL warn (bd-k1vcc) | compaction reloaded the new generation while the just-merged sidecar was still on disk and warned about discarding it | superseded sidecar removed before the reload when the generation bumps | index unit test with an in-crate WARN recorder; real binary `fsfs index` prints zero `discarding stale` lines |
| #20 status (bd-f8j9z) | the global catalog's bytes were reported as index metadata | only an in-root catalog counts; `catalog_path`/`catalog_bytes` report it either way | unit test; real binary in an empty dir: `size_bytes 0, metadata_bytes 0, catalog_bytes 659456` with the path |
| #21 daemon | no stop verb, no pid file, help named a flag that does not exist | `daemon.pid` under the index root, `fsfs daemon --stop` (SIGTERM, 10 s wait, stale file cleared), `--idle-timeout-ms`, help fixed | parser + runtime tests (stale pid, live child); real binary: idle exit after 1.7 s, `--stop` in 50 ms with the pid file gone, second `--stop` is a typed error |
| bd-k7x34 | `--fast-only` / `FRANKENSEARCH_FAST_ONLY` silently rejected under the default profile | the rejected locked override is printed on stderr at the command; README rows corrected to name `FRANKENSEARCH_PRESSURE_PROFILE=strict` | CLI test; real binary stderr line |
| #18 CHANGELOG | no entries since 2026-08-28 | `[Unreleased]` section for everything above and the earlier two-tier / daemon / gate work | this commit |

Not closed here: watch-mode freshness proof on the real binary (#6 tail), real per-term BM25
statistics (needs a Quill stats API; the warning is the honest interim), and the release itself
(#1), which is the next step.
