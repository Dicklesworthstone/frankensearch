# Handoff — harmonizing local `main` with `origin/main` (2026-08-02)

**Status: INCOMPLETE. 29 of 38 conflicts resolved, 9 remain. Nothing committed,
nothing pushed.** The merge is live and must be finished, not restarted.

- Worktree: `/data/projects/frankensearch-harmonize-20260802`
- Branch: `harmonize/main-origin-20260802`, based on `020812e2`
- `MERGE_HEAD`: `845624d7` (origin/main at merge time)
- `main`: **`020812e2`, untouched**
- Backup ref: `refs/backup/premerge-local-20260802` = `36b5ba7c` (do not delete)
- Resolution snapshot (belt and braces):
  `<session-scratch>/harmonize-resolved-so-far.patch`, sha256 `400264360202efeb…`

Do **not** `git merge --abort`, reset, stash, or clean. Resolutions live in the
worktree index. `git commit` is refused while paths are unmerged — that is
expected, not a problem to fix.

## Why the merge runs in a worktree

The same merge was destroyed twice in the shared checkout
`/data/projects/frankensearch` by an external `reset: moving to HEAD` (index
unstaged; `HEAD` never moved, so nothing was lost). A linked worktree has its
own `HEAD` and index and is immune. `MERGE_HEAD` has survived here for hours.
Finish the merge here, then land it on `main`. Do not investigate or stop the
host/swarm infrastructure — that is the owner's, not the agent's.

## The tool that made this tractable — keep it

`superset_check.py` decides whether taking one side of a conflicted file loses
anything, by diffing *declared symbols* rather than lines. It builds the
ours-resolved and theirs-resolved texts, then lists identifiers
(`fn`/`struct`/`enum`/`trait`/`type`/`const`/`static`/`mod`/`macro_rules!`)
declared on the losing side that appear nowhere in the winning text.

- Empty report ⇒ the winning side is a symbol-level superset ⇒ taking it drops
  no declared surface.
- Non-empty report ⇒ exactly the list that must be re-applied by hand.

It proved 11 files safe and pinpointed the 5 risky ones. Copy in this commit:
`scripts/merge-tools/superset_check.py` (also `conflict_stats.py`,
`resolve_side.py`, `resolve_both.py`, `union_beads.py`).

```bash
cd /data/projects/frankensearch-harmonize-20260802
python3 scripts/merge-tools/superset_check.py theirs $(git diff --name-only --diff-filter=U | grep '\.rs$')
python3 scripts/merge-tools/conflict_stats.py <file>          # per-hunk triage
python3 scripts/merge-tools/resolve_side.py theirs <file>...  # ONLY after verifying superset
python3 scripts/merge-tools/resolve_both.py theirs-first <file> [hunk#...]  # append-only ledgers
```

**`resolve_side.py` is not a shortcut.** Run `superset_check.py` first and read
the report. It checks declared symbols only — it cannot see behavioural
differences inside a shared function body.

## The one genuinely irreconcilable file — OWNER DECISION, LEFT CONFLICTED

### `scripts/perf-runner.sh` (add/add, 1 whole-file hunk) — DO NOT RESOLVE

Both sides created this path independently. They are incompatible *designs*,
not different code, so no per-hunk merge yields a coherent script.

**Local (205 lines) — "machine-class provenance runner."** Shell-centric
*diagnostic* tool. The shell itself captures host provenance (CPU/NUMA
topology, governor + SMT, macOS thermal pressure + page size, git dirty-state
hash, toolchain versions). Runs an **arbitrary benchmark command** (default
`perf_matrix`). Flags: `--class {x86-vps-ovh,trj-zen-128c,m4-macos,m5-macos}`,
`--label`, `--calibrate-aa`, `--allow-rch` (opt-in; defaults to
`RCH_DISABLE=1`), `--foreground`, `--out`.

**Origin (214 lines) — "registered-host launcher."** Thin wrapper over a **typed
Rust producer** that owns the exclusive lease, start/end host probes, benchmark
child, run log, artifact-manifest binding, receipt sealing, and self-admission.
Flags: `--gate`, `--hardware-class`, `--execution-profile`, `--run-id`,
`--run-window`, `--cpu-list`, `--runs`. States: *"This script never compiles
during a measurement invocation, manufactures JSON, or writes promotion
history"* and *"Timed runs are always local; there is no RCH override."*

**Why they cannot both hold:** different argument surfaces, different division
of labour (shell-owned vs producer-owned provenance), and a contradictory trust
model — local permits `--allow-rch` and arbitrary commands; origin forbids both
by construction.

**The substantive stake is M4.** Origin **fails closed** on M4 promotion
(*"every current M4 promotion invocation fails closed until the producer can
attest the actual executing image"*) and relocates the diagnostic role:
*"Diagnostic Apple profiling happens outside this promotion producer."* Local
has a working `--class m4-macos` diagnostic path. The owner is explicitly
targeting Apple Silicon, so this is a real product decision.

**Suggestion only, not applied:** take origin's runner (stronger promotion
pipeline: typed producer, real lease, fail-closed, consistent with the bd-yo5by
hardening) and preserve local's diagnostic script at a distinct path such as
`scripts/perf-diag-runner.sh` — which is where origin's own header says
diagnostic Apple profiling belongs. That keeps both intents instead of deleting
one. Creating a new file is a design act, so it was left for the owner.

**Blocks two files:** `e8h-hypothesis-ledger.md` hunk 5 is this exact fork
(local `Invocation: scripts/perf-runner.sh --class m4-macos` vs origin
`Invocation: PENDING a diagnostic-only M4 profiler`), and
`quill-hyperopt-campaign.md` carries 5 `perf-runner` references inside the
`machine-class` → `hardware-class`+`execution-profile` migration.

## `0d755bc2` — superseded by origin `d890f0a7`, effect fully retained

This was flagged as must-survive, so the reasoning is recorded in full.

`0d755bc2` ("fold identity preimage into running batch digest at accumulation")
deleted per-document work in the ingest path: two `Vec<u8>` allocations per
document, retention of every document's canonical preimage until seal, and a
second full walk over all retained preimage bytes at seal to build the
segment-id batch digest. It preserved the legacy witness bytes exactly.

Origin **independently made the same change, further**, in
`d890f0a7 perf(quill): hash content without canonical JSON buffers`:

```rust
) -> Result<u64, QuillIndexError> {
    let mut hasher = Xxh3::new();
    hasher.update(CONTENT_HASH_DOMAIN);
    for field in [id, content, title, metadata] { len-prefix; update }
    Ok(hasher.digest())
}
```

Effect-by-effect, against `0d755bc2`'s three goals:

| `0d755bc2` deletes | origin `d890f0a7` | verdict |
|---|---|---|
| per-doc preimage `Vec<u8>` allocation | never builds a buffer at all — hashes fields directly | **superset** (local still serialized JSON into a reused scratch buffer) |
| retention of preimages until seal | `PendingIdentity` stores `content_hash: u64`; no bytes retained | **equal** |
| seal-time re-walk of all preimage bytes | `derive_segment_id` folds `content_hash.to_le_bytes()` — 8 bytes/doc | **equal in effect** (local folded incrementally; origin's residual is O(docs)×8B, negligible) |
| — | adds `CONTENT_HASH_DOMAIN` domain separation | **origin-only gain** |

So this is best-of-both, not a drop: **every behaviour `0d755bc2` produced is
present in what is kept, and origin adds domain separation on top.**

**The one real difference, stated plainly:** the persisted witness *value*
changes. `0d755bc2` deliberately preserved the legacy `xxh3_64(JSON preimage)`;
origin's is a domain-separated field-wise hash, a different number. They cannot
coexist — one hash definition must win. Origin's is the format already on
`main` behind 474 commits of review; `0d755bc2` was only ever compiled, never
benchmarked (`cargo check -p frankensearch-quill --lib` and `--all-targets`
both exit 0; no A/B was ever run).

**Do not "restore" the streaming variant.** Origin already tried and rejected
one: `0059b511 perf(quill): stream document identity hashes` →
`c774eaf1 revert(quill): drop slower streaming identity hash` → `d890f0a7`.
There is measurement history behind the shape that landed.

## Remaining 9 files

| file | hunks | disposition | why |
|---|---:|---|---|
| `crates/frankensearch-quill/src/index.rs` | 28 | **Mixed — hand surgery.** Take origin on h18/h25 (the `d890f0a7` hash, above). **Keep local h17** (P17 within-batch fan-out) and port origin's edits into `index_batch_serial`. **Union h26–h28** (both sides added tests). Keep local-only h4, h16. | Origin's h17 side is the *old* serial `route_batch()` path, so the fan-out is genuinely local-only and is on the must-survive list. But origin edited that serial body in place while P17 had extracted it verbatim into `index_batch_serial`, so origin's edits must be re-applied there by hand. Origin also added its own parallel machinery (`ParallelIngest` trigger, "shared-nothing ingest wave", `parallel_worker_panic_is_a_typed_precommit_failure`) — reconcile, don't pick. Must re-apply: `DOCUMENTS`, `fanout_corpus`, `within_batch_fanout_agrees_with_the_retained_single_shard_path`, `within_batch_fanout_rejects_a_duplicate_id_inside_one_batch`. |
| `crates/frankensearch-quill-gauntlet/src/runner.rs` | 69 | Take origin, then re-apply 3 local symbols | `superset_check` says origin drops `CAMPAIGN_REPORT_HASH_DOMAIN`, `MAX_CAMPAIGN_REASON_BYTES`, `production_default_rank_envelope_is_rejected_before_ingest_with_valid_provenance` (local `b24c8893`). Verify none is an origin rename before re-adding. |
| `crates/frankensearch-core/src/recovery_plan.rs` | 42 | Almost certainly take origin wholesale — **verify first** | Origin rewrote the module API (`plan()`→`planned()`, `degraded_response`→`response_contract`, new `acquisition_authorization`/`required_authorization`, `consent_required`). 0 ours-only hunks. The 4 flagged symbols look like renames: `ALL_REPRESENTATIVE`→`representative_states()`, `ALL_MODES`, `ResponseDegradation`→response-contract type, and `offline_policy_blocks_network_actions_with_prerequisite`→`offline_policy_reports_missing_import_capability_without…`. Confirm each is genuinely superseded; local's old-signature call sites cannot compile against origin's API. |
| `crates/frankensearch-core/src/types.rs` | 19 | Take origin + re-apply 2 | `supported_topology`, `tiered_constructors_report_supported_topology` |
| `crates/frankensearch-embed/src/auto_detect.rs` | 19 | Take origin + re-apply 2 | `detect_remote_intent`, `materialize_bundled_default_models` — local's offline-capable model cache (`feab6151`); check origin has no equivalent under a new name |
| `docs/contracts/quill-hyperopt-campaign.md` | 27 | Take origin's vocabulary | Systematic `machine-class` → `hardware-class` + `execution-profile` migration. Depends on the `perf-runner.sh` ruling (5 references). |
| `docs/evidence/e8h-hypothesis-ledger.md` | 12 | Take origin's vocabulary, **except hunk 5** | Same migration. **Hunk 5 is the M4 fork** — blocked on `perf-runner.sh`. |
| `docs/contracts/quill-divergence-register.md` | 2 | Take origin | h1 origin adds a 121-line "Machine contract and review workflow" section; h2 is a pure origin addition. Confirm local's "Entry schema" heading survives renumbering. |
| `scripts/perf-runner.sh` | 1 | **LEAVE CONFLICTED — owner decision** | See above. |

## Already resolved (29) — audit, don't redo

**Three modify/deletes, decided deliberately:**

- `schemas/quill-divergence-register-v1.schema.json` — **kept origin's.** The
  local deletion was collateral damage in `3bbfe8c8`, not intent; origin
  actively maintains the register (`d3b5b303` created it, `d7f05acb` recent).
- `.beads/beads.base.jsonl` — **accepted origin's deletion.** Origin's
  `.beads/.gitignore` explicitly ignores it, and all 1104 of its ids are
  subsumed by the issues union (**0 unique**), so no bead content is lost.
- `.beads/beads.db-fsqlite-ns-use` — **accepted origin's deletion.** Origin's
  `.beads/.gitignore` ignores `*.db-fsqlite-ns-*`. A 40-byte machine-local
  sqlite namespace sidecar; tracking it causes the known
  "unable to open beads.db-fsqlite-ns-gate" stale-sidecar failure.

**Beads union:** `.beads/issues.jsonl` = **1145 beads** — 5 local-only + 36
origin-only preserved, **0 dropped**; on-both collisions resolved by later
`updated_at`, ties to origin. Reproduce with `union_beads.py`; it asserts no id
is lost.

**Verified symbol-level supersets (took origin, `superset_check.py` clean):**
`fsfs/runtime.rs` (2), `fusion/daemon_fallback.rs` (4), `fusion/sync_searcher.rs`
(3), `index/hnsw.rs` (6), `gauntlet/artifact.rs` (19), `gauntlet/comparator.rs`
(15), `gauntlet/lib.rs` (4), `quill/argus.rs` (3), `quill/scribe.rs` (8),
`frankensearch/index_builder.rs` (3), `frankensearch/lib.rs` (2).

**Re-export lists (origin is a strict superset; every local name present):**
`core/lib.rs`, `quill/lib.rs`, `fusion/lib.rs`.

**Judgement calls worth auditing:**

- `crates/frankensearch-index/Cargo.toml` — **both intents.** Origin's
  non-optional `sha2` (it is used outside the feature: `wal.rs`, `lib.rs`,
  `build.rs`, so local's `optional = true` would break them) **plus** local's
  `same-file` (genuinely used by `mapped_file.rs`). Feature line drops
  `dep:sha2` because `dep:` is invalid on a non-optional dependency.
- `gauntlet/perf_ratchet.rs` — **both intents, and a caught fail-open.** Origin's
  `median_ci95` CI estimator (bd-yo5by) applied to **local's `compaction/medium/20pct`**
  label. Origin's `xlarge` label matches no cell (`perf.rs` emits
  `compaction/medium/{density}pct`), so `if let Some(..)` would bind nothing and
  **QG-5 would score nothing while still reading green.** `perf.rs` says so:
  *"an xlarge-pinned cell can never produce a decision — it can only ever report
  NoDecision, which is indistinguishable from 'not run'."* Local's side also
  would not compile (binds `ratio`; the retained body uses `ci_low`/`ci_high`).
  Re-pin to xlarge together with `perf.rs` when the e6.1 generator lands.
- `.github/workflows/ci.yml` — took origin both hunks: the 228-line
  `salej-conformance` job is a pure addition, and origin's lockfile loop is
  null-delimited **and macOS-aware** (`sed -i ''` vs GNU `sed -i`), preserving
  local's path-rewrite intent while fixing it for the Apple target.
- `docs/NEGATIVE_EVIDENCE.md` — **unioned**, origin's 2026-07-29 entry then
  local's 2026-07-31. The ledger runs *ascending* in that region (07-28 → 07-29
  → conflict), so both append in order. Append-only ledger: never pick a side.
- `CHANGELOG.md` — origin's post-publication scope correction (local side empty).
- `fusion/tests/interaction_{unit,integration}.rs` — origin correctly dropped an
  **unused** `IndexableDocument` import (1 occurrence each = the import itself).
- `fsfs/adapters/cli.rs` — origin's arg order; the CLI parser is origin's domain.
- `index/build.rs` — origin adds `#[allow(clippy::unreadable_literal)]` to
  generated code, needed for the clippy gate.
- `index/src/lib.rs` — origin adds generation imports and `HnswLoadDisposition`.

## Context the successor needs

**`3bbfe8c8` is a destructive local commit.** That local perf-evidence commit
net-deleted **10,024 lines** of origin's correctness code: `daemon.rs` (929,
deleted outright), and gutted `api_embedder.rs` (1893), `model_download.rs`
(2014), `daemon_fallback.rs` (1584), `api_provider.rs` (601), `lifecycle.rs`
(352), plus the divergence-register schema and fixture. Origin's versions are
alive and larger (1161 / 2229 / 3385 lines). **The merge already auto-restored
them.** Consequence for judgement: a large share of "local changes" in the
correctness files are accretion damage, not intent — the same failure
`5116b352` had to repair once for `boolean_topdocs`. This is why "keep both
intents" usually resolves toward origin in the correctness crates, and toward
local in the perf/evidence ones.

**Pre-existing, not caused by this merge:**

- `conformance-internals` gates 35 `cfg` blocks in `quill/index.rs` but is
  declared in **no Cargo.toml in the workspace** — that code is compiled by
  nothing and its assertions are vacuous. rustc confirms it (38
  `unexpected cfg condition value` warnings). Worth a separate bead.
- Three `variable does not need to be mutable` warnings in `index.rs` test code
  come from P17's tests in `1a66c44a` (`index_documents` takes `&self`). Clear
  them in the gate pass — but **do not let warning cleanup rewrite test intent**;
  if clearing one would change what a test asserts, leave it and flag it.

## Remaining gate before anything is pushed

All required, none yet run on the merged tree: `cargo check --all-targets`,
`cargo clippy --all-targets`, `cargo fmt --check`, `cargo test`, and `ubs` on
the changed files. Then review `git diff HEAD` against **both** parents and
confirm no accretive hunk from either side vanished. Then report both
must-survive lists item by item **to the owner for checking before pushing**.
Push is `main` and also `git push origin main:master`.

**Must survive from local:** O(1) `bytes_reserved` (2.76x KEEP); shard fan-out
KEEP 9.27x; `boolean_topdocs` lowering restore; `index_builder` depth +
lexical/storage adapters; `parallel_shard_ingest_ab` harness + its
seal-site/rounds-floor fix.

**Must survive from origin:** three fail-open bypasses closed in the
null-control gate (bd-z4lqq); parent-directory durability sync after
`write_durable`'s rename (bd-xx286); publish/`install_replacement` crash windows
(bd-zhjv8); cross-tier publication nonce (bd-miio8); exact FSVI owner retention
in native HNSW handles (bd-kcek); the gauntlet A/A-null-interleaved-with-A/B
change gating effect drift and order-effect (bd-yo5by); CI workspace lockfile
refresh.

Standing contract: merge not rebase; per hunk never per file; never a side
wholesale; union the beads; the three modify/deletes reasoned in the merge
commit message; zero-deletion diff review against both parents; nothing pushed
until the owner has checked both lists.

---

# Completion record (2026-08-02, successor session)

All 38 conflicts are resolved and the gate is green. Nothing pushed.

## `scripts/perf-runner.sh` — OWNER RULING APPLIED

Not irreconcilable: a **filename collision between two complementary tools**.
The decisive evidence is in origin's own tree — `machine_class_registry.rs`
`include_bytes!`s `docs/evidence/e8h/fingerprints/{trj-zen-128c,m4-macos}-*/`
`provenance.json`, which are *the local runner's output format*
(`"schema": "frankensearch.perf-runner.v1"`). Origin consumes what local
produces; they are not rival designs.

1. `scripts/perf-runner.sh` = **origin's version, byte-identical, unmodified**
   (verified by `diff` against `:3:`). Typed-producer trust model.
2. `scripts/perf-diagnostic.sh` = **local's 205-line profiler, kept**, with a
   header stating it produces **no promotable evidence**. It fills origin's own
   *"Diagnostic Apple profiling happens outside this promotion producer."* slot.
   Only two executable lines changed from local's original, both deliberate:
   the output root is now `$PERF_DIAGNOSTIC_OUT`
   (default `~/.frankensearch-perf-diagnostics`) so diagnostic directories can
   never be mistaken for sealed receipts, and `usage()` is now robust to header
   edits. The emitted `"schema"` string is deliberately **unchanged** —
   it names the artifact format and the registered fingerprints are hashed as
   bytes; renaming it would silently fork the format.

### Machine-class taxonomy mapping (no class value vanished)

| `perf-diagnostic.sh --class` | `perf-runner.sh --hardware-class` | `--execution-profile` | promotion status |
|---|---|---|---|
| `x86-vps-ovh`  | `x86-vps-ovh`     | `x86-diagnostic` | diagnostic-only (producer refuses) |
| `trj-zen-128c` | `trj-zen3-5995wx` | `smt2-128`       | available |
| `trj-zen3-<N>c`| `trj-zen3-5995wx` | `physical-64`    | available |
| `m4-macos`     | `m4-macos`        | `scheduler-10`   | fails closed pending executing-image attestation |
| `m5-macos`     | `m5-macos`        | `scheduler-14`   | registered, unavailable (no reachable host) |

`trj-zen-128c` and width-encoded `trj-zen3-<N>c` are **legacy execution
labels, not hardware classes**; `parse_hardware_class_id()` rejects both with
`ObsoleteClassId`, and this table is that rejection's migration path. The table
also lives in `scripts/perf-diagnostic.sh`'s header and is referenced from
`docs/contracts/quill-hyperopt-campaign.md`.

**macOS probes survive on both sides.** Origin's producer already carries them
(`local_perf_runner.rs`: `pmset -g therm`, `sysctl hw.pagesize`, plus Linux
`getconf PAGESIZE` and k10temp/zenpower thermal sensors). No gap to fill. The
diagnostic script keeps its own copies because it must run on hosts the
producer will not yet admit.

## The handoff's own claim that was WRONG — read this first

> *"`3bbfe8c8` … net-deleted 10,024 lines of origin's correctness code … **The
> merge already auto-restored them.**"*

**It had not.** Three-way merge only conflicts where *both* sides touched a
region. Where local deleted code origin never touched, git applied local's
deletion **silently and without a conflict**, so those deletions survived into
the resolved tree. `superset_check.py` cannot see this either: it matches a
symbol *name anywhere* in the winning text, so a type that is still **used** and
still **re-exported** but whose **definition** was deleted reads as "present".
That is exactly how `DaemonTrustLevelV1` passed a "clean superset" check while
its `enum` had been deleted — 114 compile errors.

Files repaired by restoring origin's version (each traced to `3bbfe8c8`, each
with **no** later local commit, so nothing local was lost):

`fusion/daemon_fallback.rs` (1914→2619), `fsfs/lifecycle.rs` (3327→3631),
`gauntlet/runner.rs` (18905→18906), `gauntlet/version_contract.rs`,
`embed/tests/model_download_tests.rs`, `gauntlet/fixtures/q1-obligations.json`,
`gauntlet/fixtures/divergence-register-v1.json` (was **deleted outright**),
`docs/contracts/quill-divergence-register.md`,
`schemas/fsfs-index-footprint-advisor-v1.schema.json`, and the six
footprint-advisor fixture/golden files under `schemas/fixtures{,-invalid}/` and
`fsfs/tests/golden/`.

Also corrected: `crates/frankensearch-index/Cargo.toml`. The handoff kept
local's `same-file = { workspace = true }` believing `mapped_file.rs` used it.
It does not — `mapped_file.rs::ensure_same_file` is a **private local fn**, and
origin deliberately replaced the crate with in-tree
`frankensearch-index/src/file_identity.rs` ("*Filesystem object identity
comparison without the `same-file` crate*"). Origin also removed `same-file`
from `[workspace.dependencies]`, so the inherited dep did not even resolve —
the workspace would not load. Dependency dropped, matching origin's intent.

## Two real bugs the merge surfaced in local's O(1) `bytes_reserved` (P16 KEEP)

The counter is preserved and is still O(1) on the hot path, but it was wrong in
two ways once combined with origin's lazy arena. Both are **fixed**, and both
were caught by origin's tests, not local's:

1. **Lazy first chunk did not update the counter.** Origin's `push` allocates
   the first chunk on demand; local's counter never saw it, so `bytes_reserved`
   reported `0` for a non-empty arena. 32 fsfs tests panicked on the
   `debug_assert`.
2. **`ByteArena` derived `Clone`.** `Vec::clone` allocates exactly `len`, so a
   cloned chunk's capacity differs from its source's while the derived clone
   copied `running_bytes_reserved` verbatim — permanent drift. Replaced with a
   hand-written `Clone` that recomputes. **Local had no test for this; origin's
   `collision_bucket_reserved_bytes_match_full_rescan` did**, and it is the test
   that caught it.
3. `reset()` also lost its recompute-after-`retain` line in the merge (origin's
   `reset` had no counter to maintain). Restored, with the invariant now
   documented on the field: four sites change capacity, and the
   `debug_assert_eq!` in `bytes_reserved` is the backstop.

## `quill/index.rs` — the mixed-disposition file

Local's P17 within-batch fan-out is **kept** (`index_batch_fanout`,
`index_batch_serial`, `accumulate_shard_run`, `FANOUT_MIN_SHARD_DOCUMENTS`), and
origin's `d890f0a7` identity-hash change was **ported into it by hand**: both
accumulate paths now call `canonical_document_content_hash` (domain-separated,
field-wise, no preimage buffer) and push `PendingIdentity` directly.
`retain_identity`, `write_canonical_document_preimage` and the per-shard
`batch_hasher`/`scratch_preimage` are gone — origin's `derive_segment_id`
folds `content_hash` at seal, and it already takes `&ScribeShardState`, so it
works unchanged for both the serial and fan-out paths.

Tests from both sides were **unioned by transplanting whole items** (a naive
hunk union would have nested the two fixture builders and broken the file):
local's `fanout_corpus` + 2 fan-out tests alongside origin's
`parallel_budget_fixture_documents`, `assert_parallel_budget_bound_for_schema`
and `parallel_worker_panic_is_a_typed_precommit_failure`. A duplicated 150-line
`conformance_*` helper block introduced by the resolution was removed (origin
newly **declares** `conformance-internals` in `quill/Cargo.toml`, so that code
compiles now — the memory note that it "is declared in no Cargo.toml" is stale).

## QG-5 `medium` vs `xlarge` — a contradiction to resolve

The previous session's choice of `compaction/medium` over origin's
`compaction/xlarge` is **kept** and is self-consistent across `perf.rs`,
`perf_ratchet.rs`, `quill-perf-gates.toml` and the plan: an xlarge-pinned
ratchet cell would match no emitted cell and QG-5 would score nothing while
reading green. But its stated *reason* ("xlarge is still PENDING its e6.1
generator") is now **contradicted by origin's own `[corpus.xlarge].status`,
which says the generator LANDED**. The fixture note is rewritten to say so, and
re-pinning is now unblocked. **Owner decision outstanding:** re-pin QG-5 to
xlarge as one change across those three files plus a re-baseline. Both gates are
`activated = false`, so nothing is scored today either way.

## Gate results

- `cargo check --workspace --all-targets` — **green**
- `cargo clippy --workspace --all-targets` — **no errors.** 3 P17 `unused_mut`
  cleared (binding-mode only, no assertion touched); 9 pre-existing warnings in
  local's `fsvi_4bit_vs_incumbent` bench, untouched by this merge
- `cargo fmt --all --check` — **clean**
- `cargo test --workspace` — **3731 passed, 3 failed**. The 3 are
  `fsfs::runtime::tests::{expanded_query_variants_reject_cross_generation_fusion,
  runtime_download_models_verify_reports_mismatch,
  search_resources_rebind_on_vector_wal_append_and_tombstone}` and they were
  **reproduced failing identically at clean `origin/main` (845624d7)** in a
  throwaway worktree — same panics, same line numbers. Pre-existing, not
  merge-caused. The whole `frankensearch-fsfs` crate is byte-identical to
  origin's, so it is running origin's code. Two further `serve_socket_*`
  failures were a **long `TMPDIR`** artifact (unix socket path limit); they pass
  under `TMPDIR=/tmp/fsq`.
- `ubs` on all 107 changed `.rs` files — merged **1673** critical vs
  origin baseline **1672** on the same file list. The entire delta localizes to
  `index.rs` and is one more `rust.panic-macro` instance (an `.expect` in
  local's transplanted tests); the SARIF finding *sets* are identical, no new
  finding class.

## Zero-loss review against BOTH parents

- **Origin→merged:** scripted line-level audit; every surviving difference is
  accounted for (README lines superseded by local's richer versions, the
  deliberate O(1) counter, the documented medium/xlarge choice, and index.rs's
  64 re-indented serial-body lines whose semantics were each confirmed present
  inside `index_batch_serial`).
- **Local→merged:** symbol-level audit over every `.rs` local touched. Exactly
  20 local-introduced symbols are absent, each individually adjudicated as a
  rename or strict supersession (full list in the session report), e.g.
  `detect_remote_intent`→`resolve_remote_intent`,
  `ALL_REPRESENTATIVE`→`representative_states()`,
  `ResponseDegradation`→`SemanticResponseContract`,
  `lifecycle_success_path_reaches_ready`→`..._requires_reindex_after_acquisition`
  (local's `mark_ready()` no longer exists).
- **Beads:** union verified **by id**: local 1109 ∪ origin 1140 = **1145**,
  which is exactly the merged count. Zero dropped in either direction.
