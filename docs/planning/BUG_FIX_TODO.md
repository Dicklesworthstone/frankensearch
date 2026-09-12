# Open bug repair TODO

Requested by the owner on 2026-09-12. Maintainer: PurpleBay.
This is a working checklist, not a substitute for Beads or evidence of closure.
Inventory baseline: `062f5b8f`; GitHub issues and JSONL read live on September 12.
Checked items mean only the specific stated step is complete. Parent issues stay
open until their acceptance criteria, including consumer execution, are met.

## Execution rules and immediate queue

- [x] Read governing documentation and inspect current source/history.
- [x] Inventory all 22 nonclosed bug beads and GitHub issues 41, 43, 46, 47, 48.
- [x] Ask existing owners for current progress and bounded handoffs (Mail 41382).
- [ ] Reconcile owner replies; do not interpret silence or null assignee as permission to overwrite work.
  - ChartreuseCarp resumed release/dependency ownership (Mail 41394), owns oracle/profile/fuzz and full Quill qualification. Per Mail 41407, PurpleBay will execute the genuine-model doctor regression on the existing ovh fsfs target after the active gate; peer's old fsfs target was reaped. No independent watcher subpiece is ready.
- [x] Restore tracker mutation capability with its graph owner: migrate schema 17 to 19.
  - [x] Inspect explicit migration plan read-only and its source preconditions: eligible 17→19, integrity OK, 1298 issues/2257 dependencies/3779 comments. Prestate-bound receipt reported in Agent Mail; no apply performed.
  - [x] Coordinate a single migration writer before any apply; preserve JSONL and DB family. ChartreuseCarp delegated PurpleBay in Mail 41399; peers notified in 41401. Snapshot `.beads/recovery_20260912T194346Z`; supported migration run `20260912T194351.279401Z-476704-0` retains raw prestate under `.beads/.br_recovery/schema-migrations/` and supports undo.
  - [x] Verify ready/show, import and dependency-cycle checks after recovery: reads work, sync healthy with 1298 DB/JSONL issues and no drift, import zero changes, zero active cycles (two archived closed cycles). Doctor remains degraded for retained historical recovery artifacts; no deletion or blanket repair performed.
  - [x] Record claim bd-6kafg as PurpleBay/in_progress and doctor follow-up under bd-pz8va after recovery; flush succeeded.
- [x] Land doctor producer-admission fix `062f5b8f` (main and compatibility mirror).
- [x] Run remote formatting and 16 doctor library tests (semantic-support/no-default; all pass).
- [x] Validate doctor fix with a genuine model and stale index through the production CLI.
  - [x] Add isolated production-CLI regression `loader_only::doctor_rejects_stale_real_model_producer_revisions` in the existing quickstart suite; real index vectors, both tiers, stale-revision refusal, exact-byte restoration acceptance, observer-only assertions.
  - [x] Compile and execute the ignored genuine-model regression with both pinned caches: corrected fixture passed, one test/zero failures, 18.07s, terminal exit 0 at 20:49:10 UTC on ovh-a.
  - [x] Remote rustfmt passed for this new regression on hz4 at 19:37:30 UTC.
  - [x] Source review caught and corrected the new test's diagnostic expectation: doctor embeds the safe `SearchError` display text, not its internal variant code. The production error envelope remains `subsystem_error`; refusal/recovery/observer-only assertions are retained. Corrected-source remote formatting passed on hz4 at 19:53:34 UTC; subsequent execution results are below.
  - [x] Further source review found the raw revision-byte mutation invalidates FSVI's header CRC. Replaced it with the production writer using the genuine vectors, original shape/quantization and nonce, plus a read-only structural reopen and bitwise vector checks. The f17e0a73 run used the earlier invalid fixture; its result is retained as fixture-development evidence, not product acceptance.
  - [x] Retained old-fixture terminal result: RCH 30017802786046230 exit 101 at 20:48 UTC, zero passed/one failed in 11.01s; pinned models verified and ten-document fast+quality indexing succeeded, then raw revision mutation produced the predicted CRC refusal. This is fixture-development evidence, not a product-regression verdict. Source/executable hashes and full output retrieved under `.rch-results/doctor-producer-20260912-f17e0a73`.
  - [x] Checksum-valid fixture `041950ce` passed on ovh-a at 20:49:10 UTC; receipt directory `.rch-results/doctor-producer-20260912-041950ce` includes transferred producer manifest hash, full source checks, terminal output and executable hashes (test `75bfafd1...`, fsfs `d5f13039...`). Fast and quality stale identity refusals, exact-original restoration acceptance and observer-only vector assertions all executed. Formatting passed at 20:30:52 UTC.
- [ ] Validate default, native, lite and full workspace gates for the doctor fix.
- [ ] Reproduce ARM native refusal with the new digest diagnostic; locate first numerical divergence.
- [ ] Implement the supported numerical repair or explicitly distinct qualified producer only after evidence.
- [ ] Continue code-bearing bug work in dependency order; retain each new discovered task below.
  - Ordinary `br ready --json` now works: bd-6kafg was the only unowned ready bug and is now claimed; other ready entries are the owned installer, publication reconciliation and bd-d7xk1.

## GH #47 / bd-6kafg: native MiniLM ARM64 exact producer rejection

Owner: PurpleBay (Beads claim recorded); native migration parent bd-2ba5 remains peer-owned.

- [x] Reproduce native Int8 refusal on ARM and success on x86 with matched source/model (existing receipts).
- [x] Preserve exact certificate gate; no tolerance or reporter-hash substitution.
- [x] Add bounded expected/observed digest, target and precision diagnostics (`1698b288`).
- [x] Verify F32 positive/negative diagnostic regression remotely (1 pass; executable hash was not retained).
- [x] Refresh private Mac RCH controller health/capacity: healthy, zero jobs, eight available slots. Preserve original dirty release clone; use a separate clean current-source clone.
- [ ] Run one native certificate probe with the diagnostic patch; retain terminal output and executable hash.
  - Attempt 2026-09-12 19:32 UTC: strict private RCH refused before execution (exit 103, memory_pressure_critical). No model/compile verdict; retry only after fresh host-pressure admission. Own clean clone exists at `/tmp/fsfs-release-root-20260912/frankensearch-purplebay`; peer release clone is untouched.
  - Fresh 19:39 UTC status confirms critical memory pressure 95.081%, high confidence, telemetry age 59s. Host-owner relief or another admitted ARM worker is required; no blind retries or threshold changes.
  - Read-only refresh at 20:07 UTC: memory pressure 93.798%, still critical/high confidence with 30-second telemetry. No ARM job submitted. User asked asynchronously for host-owner relief or another admitted ARM64 worker.
- [ ] Bind compiler target features, model hashes and source to both platforms.
- [ ] Compare tokenizer IDs and positions before numerical kernels.
- [ ] Compare embedding layer-normalization outputs.
- [ ] Compare Q/K/V quantization bytes, scales, integer accumulations and dequantized bits.
- [ ] Compare attention softmax, GELU and first divergent encoder layer.
- [ ] Prove which arithmetic/dispatch difference causes the first divergence.
- [ ] Implement deterministic arithmetic or an explicitly separate qualified producer; never silently relabel old vectors.
- [ ] Test repeated and batched genuine-model execution on both platforms.
- [ ] Test negative cross-producer index admission and required rebuild behavior.
- [ ] Run a populated CASS semantic query using the qualified artifacts.
- [ ] Complete linked acceptance and update GitHub/Beads with exact evidence before closure.

## GH #43 / bd-pz8va: shutdown, pressure and retained generation

Owner: ChartreuseCarp; doctor follow-up implemented by PurpleBay.

- [x] Existing one-shot cancellation/RSS sampling repair is on main (90e8fb14 lineage).
- [x] Existing bounded SIGTERM, second-SIGINT and strict/performance pressure probes are recorded.
- [x] Keep cooperative pressure threshold distinct from an OS hard memory limit.
- [ ] Preserve the previous committed generation across interruption of replacement indexing.
  - [ ] Finish composite-generation prerequisites with bd-xomn.1 owner.
  - [ ] Stage lexical, fast and quality artifacts under a new generation identity.
  - [ ] Seal/verify complete generation before publication.
  - [ ] Publish once atomically, retaining the old readable generation on cancellation/error.
  - [ ] Test interruption during discovery, loading, batches, sealing and pointer publication.
  - [ ] Test concurrent search while indexing is interrupted and then retried.
- [ ] Repeat bounded original-workload acceptance without recreating the unsafe 12.9 GiB incident.
- [x] Investigate new doctor false-healthy report (comment 5645302291).
- [x] Reuse loaded producer and unchanged search admission in doctor; no second model construction.
- [x] Test current/missing/historical revision, foreign ID, width mismatch, absent tier and read-only inspection for both tiers.
- [x] Post scoped implementation/test update on GitHub (comment 5648153122); do not close parent.
- [x] Genuine-model CLI stale-index refusal and exact-original restoration acceptance, both tiers, with read-only vector assertions (041950ce; remote one test passed). Actual full rebuild/release matrix remains under the surrounding open gates.
- [ ] Complete full-feature, native-only and explicit-lite validation.
- [ ] Complete unchanged real-model deadlines, quickstart and full quality gate.
- [ ] Publish only if separately authorized; verify reporter acceptance before closing the complete incident.

## GH #46 / bd-u8cof: Model2Vec startup allocations and tokenizer rebuild

Owner: ChartreuseCarp.

- [x] Existing streamed F32 matrix decoding avoids redundant whole-file allocation.
- [x] Existing same-process shared loader and cache-mutation rejection tests are recorded.
- [x] Correct identity/reindex guidance: equal vector certificate does not imply equal complete producer revision.
- [ ] Preserve exact bytes, artifact verification and external mutation/truncation safety in further work.
- [ ] Identify a supported persistent compiled Unigram representation (JSON serialization rebuilds the trie).
- [ ] Design cache binding to exact tokenizer/model/implementation identity, with bounded corrupt-cache refusal.
- [ ] Implement cross-process reuse without unsafe borrowing from mutable/truncatable files.
- [ ] Test cold creation, warm reopen, corrupt/truncated cache, tokenizer swap and concurrent builders.
- [ ] Run real Potion token/vector parity against the current implementation.
- [ ] Measure cold-process RSS, faults and wall time against a live baseline in the same invocation.
- [ ] Retain honest loss/no-verdict results and negative cache tests; no speed claim from correctness alone.
- [ ] Validate with the reporting ee workload and complete required gates before closure.

## GH #41 / bd-458gu: fragmented writer sessions, merge fuel, visibility

Owner: ChartreuseCarp; coordinate scorer work with GoldenMink.

- [x] Existing continuous-writer lease repair preserves safe live leases (9e3117df lineage).
- [x] Existing unchanged 192-document control proves 3 segments, IDs 0–191.
- [x] Keep hole-ratio guard: recorded naive concat expands sparse 10,672 bytes to 16,345,526 bytes.
- [ ] Implement sparse-safe merge/reencoding for repeated writer reopen sessions.
- [ ] Preserve stable document identities, tombstones, positions and exact query results.
- [ ] Repair dictionary/merge fuel accounting without granting unbounded work.
- [ ] Avoid sealing every shard solely for the one-second visibility checkpoint.
- [ ] Test 24-session reopen fragmentation, cross-shard overlaps and crash/reopen behavior.
- [ ] Validate bounded fuel errors and successful search after publication/compaction.
- [ ] Run full Quill default/all-feature conformance and unchanged negative controls.
- [ ] Retain real incumbent-relative evidence for any performance claim and close only on full acceptance.

## Other nonclosed bug beads: dependency-ordered work inventory

Each row needs source reconciliation, ownership/reservation confirmation, a minimal
reproduction, implementation where missing, focused negative/control tests, strict
RCH validation, and evidence-backed Beads closure. None is silently waived.

- [ ] bd-z2nfa — watcher-held writer excludes search. Implement through sealed composite generations, not relaxed locking; test independent-process search during watch and teardown. Depends in practice on generation publisher and bd-fsfs-identity-bound-watch-staging-t9m9m.
- [ ] bd-fsfs-identity-bound-watch-staging-t9m9m — identity-bound watch staging/publication; prerequisites bd-9xuj, bd-xomn.1. Test late writers, interrupted seals, old readers and tier identity agreement.
- [ ] bd-raw-vector-api-retirement-8sc8a — remove public raw-vector/infallible cross-tier bypasses; prerequisites bd-9xuj, bd-xomn.3. Inventory callers, migrate to authenticated typed inputs, test wrong-space rejection.
- [ ] bd-8utj — AzureCove: validated generation-aware caches and same-ID revision invalidation; prerequisites bd-07os, bd-r65a, bd-xomn. Test generation changes during active queries and failed replacement.
- [ ] bd-jbfg — AzureCove: terminal embedding-space enforcement across all entry points; prerequisites bd-4v5n, bd-8utj, bd-fsvi-readonly-semantic-inspection-qxo6, bd-lk1g, bd-r65a, bd-raw-vector-api-retirement-8sc8a, bd-remote-api-space-attestation-fcfj. Inventory every bypass and prove typed terminal refusal.
- [ ] bd-a6zt — semantic search silently degrades to hash; prerequisite bd-3fy9. Reconcile already-landed fail-closed work, test real consumer behavior and remove any remaining implicit fallback.
- [ ] bd-quill-union-horizon-exactness-salej — GoldenMink: exact TopDocs across UNION_HORIZON refills/ties. Preserve live Tantivy differential and adversarial boundary fixtures.
- [ ] bd-r1-exact-repair-or-residual-1i4j4 — scorer first-divergence repair or controlled residual; prerequisites bd-5o5z8, bd-r1-preconstruction-scorer-trace-rtnwu. Establish owner before editing; no generic tolerance substitution.
- [ ] bd-qg6-exact-case-v8-migration-vyun6 — migrate exact/case-bound V8 evidence; prerequisite bd-quill-flip-real-prose-lexical-ghhh. Remove generic V7 tolerance only with valid current witness and rejection controls.
- [ ] bd-qg2-lifecycle-quarantine-w1-rerun-l4ra4.1 — AzureCove: summed gauge bound to continuous terminal lifecycle. Require fresh live producer and missing-terminal negatives.
- [ ] bd-6xhh9 — AzureCove: prohibit watchdog daemon restart under active evidence jobs; test live leases and interruption preservation across restart routes.
- [ ] bd-6xhh9.5 — harden remaining restart surfaces; prerequisites bd-6xhh9, bd-6xhh9.3. Inventory each restart caller and classify terminal interruptions without fake passes.
- [ ] bd-quill-e8-perf-doctrine-x4e4.6 — QG-1 true indexing/tokenizer denominators; prerequisites bd-qg1-class-applicability-repair-89o4o, bd-qg1-final-multiclass-adjudication-07xik, bd-quill-e8-perf-doctrine-x4e4.15.1, bd-tqi3, bd-uh2f, bd-vmpb.
- [ ] bd-quill-e8-perf-doctrine-x4e4.7 — QG-3 real update-to-searchable visibility, not completion time; prerequisites bd-quill-e8-perf-doctrine-x4e4.15.1, bd-tqi3, bd-uh2f.
- [ ] bd-quill-e8-perf-doctrine-x4e4.8 — QG-4 durable steady-state commit latency excluding rebuild contamination; same prerequisites as QG-3.
- [ ] bd-quill-e8-perf-doctrine-x4e4.9 — QG-6 realistic queries, tail samples and absolute per-class latency; prerequisites bd-live-total-lexical-contract-gxwy, bd-qg6-exact-case-v8-migration-vyun6, bd-quill-e8-perf-doctrine-x4e4.15, bd-quill-e8-perf-doctrine-x4e4.15.1, bd-tqi3, bd-uh2f.
- [ ] bd-quill-e8-perf-doctrine-x4e4.10 — QG-9 prove cold-open and distinguish warm/page-fault costs; prerequisites bd-quill-e8-perf-doctrine-x4e4.15.1, bd-tqi3, bd-uh2f.
- [ ] bd-quill-e8-perf-doctrine-x4e4.11 — QG-10 deterministic dependency/build-footprint facts outside timing A/A; prerequisites bd-quill-e8-perf-doctrine-x4e4.15.1, bd-uh2f.

## GH #48: optional allocator enhancement (not a confirmed bug)

- [x] Classify separately from the four bug reports; do not silently add a dependency/default allocator.
- [ ] Evaluate safety/API/maintenance and actual workload evidence if owner wants this enhancement after bugs.
- [ ] If accepted, implement opt-in only, real cross-platform tests and live baseline comparison; otherwise explain disposition.

## Validation, handoff and discovered-work ledger

- [ ] Preserve exact command, source, selected test count, terminal exit and executed artifact identity where required.
- [ ] Complete `scripts/quality-gate.sh` remotely with registered models; retain failures instead of weakening gates.
  - Strict RCH job 30017802786046197 on ovh-a from transferred 062f5b8f source, started 19:33 UTC, explicitly cancelled at 19:54 UTC (terminal exit 143) to prioritize current-source doctor execution after stale contracts were confirmed. No workspace test verdict. Remote Git metadata reports dd093fb2+dirty, so this invocation is not release-source provenance. The new genuine-model regression was not part of that transferred tree.
  - Formatting, workspace/all-target check (282s), Clippy plus hybrid Clippy, and Windows index cross-check (92s) passed before cancellation during library-test compilation. External fast_cmaes emits seven deprecated-constant warnings and Windows index check emits nine dead-code warnings; no unrelated dependency edits or lint suppressions made.
  - Earlier doctor fixture f17e0a73: two-slot queue timed out without execution (exit 103 at 20:00:29 UTC); reduced Cargo parallelism to one to fit actual spare capacity, job 30017802786046230 admitted on ovh-a at 20:01:31 UTC. Same retained target (dependencies nevertheless rebuilt), finite 7200-second transport budget, unchanged model-test deadline; source/executable/output retained under `.rch-results/doctor-producer-20260912-f17e0a73`. It ended with the CRC fixture failure recorded above, superseded by the passing checksum-valid fixture. Initial receipt wrapper was blocked before execution for dynamic redirect targets; explicit new paths replaced them without disabling the guard.
  - Read-only worker checks confirm both model directories exist and all five recorded source hashes match local: runtime `38cad088...`, corrected regression `090622ee...`, root/fsfs manifests and lockfile. Actual pinned model-byte verification remains part of the test, not inferred from directory presence.
- [ ] Complete additional Quill full/probe lanes for affected changes.
  - ChartreuseCarp reports stale oracle v8/engine profile v6 contracts against new 0.3.0 adapters in the already-transferred gate source (Mail 41406). Their isolated five-file repair is under RCH validation; old identities remain historical. Expected contract refusal is not a doctor regression or permission to weaken conformance.
  - [x] Peer-reported focused repair validation (Mail 41421): RCH 30017802786046208 terminal exit 0 at 20:16:52 UTC; 12 oracle/version plus four profile tests pass, formatter and standalone locked fuzz-bin check pass. Peer retained executable/source hashes. This does not satisfy full Quill/full workspace gates.
- [ ] Accepted bounded handoff from ChartreuseCarp (Mail 41441): after corrected doctor execution, run current storage/durability library consumers sequentially on the retained ovh target: `cargo test --locked -p frankensearch-storage -p frankensearch-durability --lib`. Preserve source, executable identities and exact terminal counts; no duplicate cold lane and no publish-pin advance based on compilation alone.
- [ ] Reconcile actual shipped fixes with every GH/bead acceptance criterion; do not close a broad issue for one sub-fix.
- [ ] Commit only intended paths; preserve the unrelated golden `.actual.json` file.
- [ ] Keep main/mirror synchronization and coordinate concurrent remote histories without force pushes.
- [ ] Update this checklist on every new finding or completed validation step.
- [ ] Newly discovered: RCH source-content receipt refuses a symlink in sibling fast_cmaes `.claude/worktrees`; use a supported isolated-source route, do not delete peer files or weaken proof.
- [ ] Newly discovered: retain executable provenance before remote cleanup; old F32 diagnostic run has source/log evidence but no executable digest.
- [x] Newly discovered genuine-model doctor regression gap after 16 library passes: closed by checksum-valid production CLI regression 041950ce and the retained 20:49 UTC pass. Full workspace/release gaps remain separately open.
- [ ] Newly discovered: ChartreuseCarp reports today's registry publication of facade 0.6.0 / component 0.3.0 / rerank 0.4.0 from dd093fb2 (Mail 41404/41406); reconcile publication-owner receipts before release qualification. These immutable versions must not be reused, and they do not contain the later doctor repair. This is peer-reported registry evidence, not an independent publication receipt collected by PurpleBay.
- [ ] Newly discovered: historical producer-fingerprint reconstruction test in `frankensearch-embed/src/model_manifest.rs` assumes current `0.2.7` implementation prefix despite adapter `0.3.0` (Mail 41418). SwiftWillow owns the correction under bd-dsbym; preserve historical fingerprints/vector certificates and execute exact manifest regressions after the fix. No duplicate edit or blanket golden regeneration.
- [ ] Newly discovered: publication audit and three current-execution assertions in `scoped_logging_contract.rs` remain pinned to asupersync 0.4.10 / fsqlite 0.3.18 despite current 0.5.0 / 0.4.0 (Mail 41435). ChartreuseCarp owns forward contract reconciliation after consumer validation, retaining old-version/source/duplicate/mixed-family rejection controls. Registry publication alone does not satisfy these checks.
- [ ] Newly discovered: Potion production `protocol_revision` named SafeTensors 0.7.0 although the actual loader dependency is 0.8.0 (Mail 41436). SwiftWillow owns the identity correction plus historical-preservation/same-adapter-old-protocol rejection tests; require real Potion certificate/decode parity. The earlier f17 doctor fixture predates this fix; the passing 041950ce run includes current manifest SHA `68d1d185...` and real model execution, but is not the full producer matrix. Peer reports 404 + 8 + 3 producer-related tests passed with Clippy still active (Mail 41446); retain their terminal receipts before final reconciliation.
- [ ] Newly reproduced: anti-rollback floor publisher race, `bd-generation-antirollback-floor-ynwdi` (Mail 41444): core suite 1128 passed/one failed; a reader can observe a newly created but incomplete floor file before the winning publisher writes/fsyncs it. AzureCove/ChartreuseCarp are coordinating ownership. Preserve genuine torn/tampered-head `UnresolvedAttempt` refusal; add deterministic paused-writer/competitor coverage and crash-releasing cross-process serialization before rerunning core. Do not widen the test or blindly remap all invalid heads.
- [ ] bd-d7xk1 (ready problem task, not bug-typed): inspect seven unreferenced source files (`core/metrics.rs`, `durability/tantivy_wrapper.rs`, `fusion/repro_blend.rs`, `fusion/repro_rrf.rs`, `tui/repro_input.rs`, `rerank/test_api.rs`, `rerank/test_inputs.rs`). Register only real supported functionality or move valid repros into actual tests; deletion requires explicit written permission. Add a meaningful source-reachability check without compiling dead code solely to satisfy a census.
