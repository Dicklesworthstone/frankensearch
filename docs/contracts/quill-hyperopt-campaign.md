# Quill Hyperopt Campaign (E8-H) — Multi-Microarchitecture Performance Offensive

**Status:** Normative campaign contract. Companion to `quill-perf-gates.md` /
`quill-perf-gates.toml` (which remain the authority on gate targets, activation
discipline, and the paired-estimator/evidence-artifact contracts — nothing in
this file overrides them).
**Owning bead:** `bd-quill-e8-hyperopt-*` epic (see § Bead map).
**Scope:** closing the measured Quill-vs-Tantivy performance deficit on three
microarchitecture classes — the existing x86 VPS fleet, a 128-core AMD
Threadripper (`trj`), and Apple Silicon M4/M5 — under the repository's
existing honesty machinery.

---

## 0. Premise, stated honestly

The campaign premise ("Quill ≥ 3.0x Tantivy at bulk indexing") is currently
**falsified by the only admissible data produced so far**:

- **QG-1 (bulk indexing), provisional, gate inactive:** Quill at
  0.1066–0.1725x Tantivy docs/s on `bulk/medium/*` — 5.8–9.4x behind, a ~23x
  miss against the ≥3.0x target (`docs/NEGATIVE_EVIDENCE.md`, 2026-07-27,
  `bd-h6eh`).
- **QG-6 (query latency), smoke scale only:** ~25x slower at 500 docs, with
  Round-0 profiling attributing >83% to immutable TERMDICT reparse/validation
  (`bd-quill-e8-perf-doctrine-x4e4.5`, `bd-quill-gauntlet-qg6-cache-termdict-gwd4`).

This campaign therefore does not start from "optimize Quill." It starts from
"close a 6–25x deficit, one attributed lever at a time, on three
microarchitectures, with a built-in renegotiation checkpoint if the 3x target
proves unreachable." A miss honestly measured and ledgered is a campaign
success; a win that cannot survive a hostile reading of its own artifacts is a
campaign failure.

## 1. Standing laws

Laws 1–5 are inherited verbatim from `quill-perf-gates.md` (no benchmark-only
semantics; distributions not averages; never hide maintenance; memory is
first-class; one lever per change). This campaign adds:

6. **Machine-class scoping.** Every claim is `(gate, fixture, machine-class)`
   scoped. Ratios are NEVER compared, averaged, or quoted across machine
   classes. Every machine class carries its own A/A null tolerances, derived
   from measured dispersion on that machine, not copied from another class. A
   lever may KEEP on one class and NoClaim on another; the ledger row must say
   so explicitly.
7. **Platform-symmetric durability.** On macOS, any commit/durability-adjacent
   number is admissible only with `F_FULLFSYNC` treatment attested symmetric
   for both engines (macOS `fsync(2)` does not guarantee media durability
   without it). This is Law 1 applied to Apple Silicon; the attestation lives
   in the run's provenance JSON.

## 2. Machine-class matrix

`.bench-history/QG-<n>.<machine-class>.latest.json` already keys baselines by
machine class; this campaign registers three new classes beside the existing
fleet:

| Class ID | Hardware | Campaign role | Mandatory provenance (beyond the evidence-v1 contract) |
|---|---|---|---|
| `x86-vps-ovh` | existing rch workers | continuity baseline; CI ratchet | existing (ELF SHA-256, worker id, governor) |
| `trj-zen-128c` | 128-core AMD Threadripper ("trj") | scale-out truth: QG-1 high-thread cells, QG-8 thread scaling, allocator/NUMA attribution | governor=performance pinned; SMT state; NUMA topology dump (`lscpu -e`, `numactl -H`); isolated `CARGO_TARGET_DIR`; exclusive build-slot during timed windows |
| `m4-macos` | Apple M4 (ARMv9, NEON, 16 KiB pages, P+E cores) | ARM64 latency truth: QG-6/QG-9; P-vs-E scaling curves | `sysctl hw`/`machdep.cpu` dump; thermal pressure sampled around runs; `F_FULLFSYNC` symmetry attestation; page size recorded |
| `m5-macos` | Apple M5 | same as `m4-macos` + generational-delta lane | same |

Rules:

- Candidate and rerun on the same machine, same run window (existing
  promotion contract).
- `trj` and the Macs are NOT rch workers. They run via
  `scripts/perf-runner.sh` (this campaign's one piece of new infrastructure),
  which captures the fingerprint, runs detached, and emits the same sealed
  artifact layout so the ratchet can consume all four classes.
- QG-8's manifest already demands "Apple Silicon P-only vs P+E curves
  published (graceful E-core join, no cliff)" and "any plateau ATTRIBUTED
  (allocator contention axis mandatory)" — this matrix finally gives those
  clauses hardware.

## 3. Phase plan

```
Phase 0  INSTRUMENT INTEGRITY + MACHINE-CLASS ONBOARDING   (blocking)
Phase 1  ROUND-0 PROFILING TRUTH                           (per gate x class; local, never rch — rch cannot symbolize, bd-e41k)
Phase 2  HYPOTHESIS LEDGER SEEDING + PRIOR MINING          (mandatory pre-flight)
Phase 3  OPTIMIZATION ROUNDS                               (workstreams W1–W5; >=10 rounds)
Phase 4  MATH-FAMILY ARTIFACTS                             (bounded to 3 families)
Phase 5  CONVERGENCE + GATE ACTIVATION + RENEGOTIATION     (2 consecutive clean rounds; Round-6 checkpoint)
```

### Phase 0 — Instrument integrity + onboarding (nothing counts until green)

1. `bd-h6eh` validator repairs land (force_no_claim self-rejection, QG-6
   corpus-digest scope, ignored hierarchical estimator). Every measurement
   sealed to the defective validator SHA stays diagnostic-only.
2. The QG-3/5/6 validity gaps from the 2026-07-28 read-only audit close:
   version-unique visibility probes, on-disk merge arms with reopen parity,
   prepared four-arm reuse, terminal per-cell artifacts + shard assembly so a
   timeout leaves receipts.
3. QG-1 retry predicate executed as written: `QUILL_PERF_FIXTURE` narrowed,
   `xlarge` + `tokenize_only` captured, complete `QG-1.json` emitted. The
   global rch timeout is not raised.
4. Machine-class onboarding: `perf-runner.sh` verified on trj/M4/M5 with
   fingerprint artifacts committed, then one **A/A-only calibration run per
   class** (trj at 1/16/64/max threads; M4/M5 P-only and P+E) to fix each
   class's null tolerances from measured dispersion before any A/B runs.
   Manifest (`quill-perf-gates.toml`) additions for the new classes are
   coordinated with the manifest's reservation holder — not edited
   unilaterally.

**Exit:** QG-1 + QG-6 activation-eligible artifacts on `x86-vps-ovh`;
admissible A/A nulls on all four classes; ratchet consumes four class
baselines.

### Phase 1 — Round-0 profiling truth

Local lanes (flamegraph + samply + dhat allocation census + `strace -c` /
`fs_usage` syscall+fsync census) under `release-perf` with frame pointers.
Deliverable per (gate x class): a committed profile card with the top-10
self-time frames ≥0.1%, triangulated (a lever is actionable only when two
profilers agree on the frame). QG-1 cards additionally record the
**tokenize-only honesty denominator** — the measured ceiling that converts
"optimize indexing" into "close X% of a bounded gap." If a class's ceiling is
itself below Tantivy's measured throughput, that is renegotiation evidence,
not an optimization target. `x86-vps-ovh` cards are owned by the in-flight
pre-admission beads (`bd-6oiq` for QG-1, `x4e4.5.4` for QG-6); the campaign
adds trj/M4/M5 cards only.

### Phase 2 — Hypothesis ledger + prior mining

Every candidate lever gets a hypothesis row BEFORE implementation:
`hypothesis / minimal repro / expected signal (as % of ceiling gap) /
falsifiability / one-line invocation / machine-class scope / results-inline /
retry-condition predicate on reject`. Rows live in the campaign evidence
directory, NOT in `docs/NEGATIVE_EVIDENCE.md` (terminal rejects still go
there, via its null-control commit gate, once an experiment concludes).

Banked priors that constrain digging (do not re-test without satisfying the
recorded retry predicate):

- Grouped MaxScore activation — REJECTED with evidence (`46a475ac`).
- `core::simd` + `#[target_feature]` — REJECTED on packaging (nightly leaks to
  crates.io consumers, bd-7zjk), not on perf.
- SIMD posting-unpack dispatch is BANDED (wins widths 4–28, regresses narrow
  and full-u32); bands must be re-derived per microarchitecture.
- Count-free WAND gate extension — dead end; fixture artifact. Retrieval
  fixtures must include saturating + mid-IDF + rare term classes.
- SWAR tokenizer — length-dependent win; bench long and short corpora.
- Tombstone bitmap at ~1% density — wash inside the A/A null.

### Phase 3 — Optimization rounds (see § Workstreams)

One lever per commit. Every lever ships: a recommendation-contract card (see
§ Keep-gates), the paired A/B + same-invocation A/A evidence per claimed
class, and a **green differential parity campaign at the same SHA** — the
gauntlet is this campaign's isomorphism oracle. Rank-exactness (the vendored
BM25 fieldnorm table and f32 op order in `contract.rs`) is the invariant most
levers can silently break; "Floating-point: identical" is a load-bearing line
of every proof.

### Phase 4 — Math families (complexity budget: 3)

Compiled artifacts only, never prose:

1. **Information-theoretic ceiling** — the tokenize-only lane per class plus a
   derived `ceiling.json` (max docs/s if inversion+seal were free). Every
   W2/W3 lever states its expected signal as % of remaining ceiling gap.
2. **Queueing/pipeline stage budget for trj ingest** — measured per-stage
   service rates (tokenize → invert → seal → merge) at 1/16/64/128 workers,
   compiled to a **static** worker-allocation table with budgeted mode and
   deterministic fallback = current single-pool rayon. An adaptive controller
   is justified only if the static table demonstrably leaves >15% on the
   floor.
3. **Anytime-valid e-process side-car on `.bench-history`** — per gate per
   class, a sequential-evidence alarm over ratchet history that stays valid
   under continuous monitoring (no p-hacking across reruns). Pure
   post-processing; no runtime cost.

Explicitly NOT selected (diminishing-returns rule): optimal transport, TDA,
control-theoretic compaction scheduling — no measured failure signature.
Each gets a hypothesis-ledger row with the predicate that would revive it.

### Phase 5 — Convergence, activation, renegotiation

- A round = every active workstream lands or ledgers ≥1 lever with receipts.
- Convergence = ≥10 rounds AND 2 consecutive rounds with <3 new genuine
  findings AND zero unresolved hypothesis rows.
- Gates activate per gate per class exactly per the existing activation
  contract.
- **Renegotiation checkpoint (non-optional), end of Round 6:** if the QG-1
  admissible best cell is still <1.0x on every class, the campaign formally
  presents the evidence for retargeting the flip criterion from "≥3.0x" to a
  bounded envelope (e.g., ≥0.8x bulk + p50 query parity per class + memory ≤
  oracle + Quill's structural wins). `bd-3beo` already frames the flip as
  needing a bounded safety envelope, not the full leapfrog; the checkpoint
  makes that decision explicit and user-owned instead of drifting. The
  decision belongs to the user, not the fleet.

## 4. Workstreams

### W1 — Query fixed-cost elimination (QG-6, QG-9; first truth machine: m4/m5)

| Lever | Basis | Notes |
|---|---|---|
| TERMDICT decoded-metadata cache | `gwd4` (P0, >83% attribution) | The named lever. Snapshot-scoped; budgeted (bounded bytes); deterministic full-reparse fallback on exhaustion. |
| Verify-once checksum memoization | reparse/validation attribution | Section checksums validated at first touch per snapshot, sealed thereafter; invalidated on generation change. |
| Collector allocation churn | `collect_id_hits` lazy-Vec sibling win | Audit Argus collectors for per-query allocs; apply the proven pattern. |

### W2 — Bulk-index single-thread cost (QG-1; thread=1 cell is 8.9x behind, so per-doc work is the core deficit)

| Lever | Hypothesis | Caution |
|---|---|---|
| Term-interner hashing + arena | interner re-hash per field + SipHash-class cost | aHash/FxHash family is a proven repo-wide win; measure, don't assume. |
| Postings accumulation growth | per-term Vec realloc churn vs arena-backed exponential chunks (the incumbent's `expull` trick) | dhat census first. |
| Seal-time section checksum cost | if sealing hashes every byte with a heavy hash, it is a prime constant-factor suspect | Algorithm change = FSLX format-registry bump; EV-scored; registry owners coordinate. |
| Commit-path fsync count | batch directory syncs; two-slot manifest bounds the floor | `strace -c` / `fs_usage` census first; Law 7 on macOS. |

### W3 — Parallel scale-out (QG-1 high-thread, QG-8; truth machine: trj-zen-128c)

Shard-per-worker indexing into independent segments, then
`KeeperWriter::concat_merge` (already exists) — N independent single-thread
problems plus one cheap concat. NUMA/CCD-aware sharding; per-thread arenas;
sharded interners merged at seal (no global interner Mutex). Deliverables
include the allocator-contention axis; "a bandwidth ceiling is honest, a lock
plateau is a bug." On M4/M5: rayon pool P-only vs P+E, published as the two
required curves.

### W4 — SIMD/µarch kernels (all classes; safe-code constraint binding)

`#![forbid(unsafe_code)]` holds and bd-7zjk stands: the lane is `wide`
(current; lowers to 2x NEON on ARM, AVX2 on x86) by default, with a mandatory
**ecosystem scan** (safe runtime-dispatch abstractions, e.g. pulp- or
multiversion-class crates — internal unsafe in a vetted dependency is not a
workspace violation, but adoption is a dependency-policy decision) BEFORE any
kernel is written. AVX-512 on trj is **C-tier/blocked** until the packaging
predicate changes; recorded as a ledger row so nobody burns a week on it.
Targets: FOR unpack/pack (re-derive dispatch bands on NEON), BM25 scoring
loops, block-max screening. Every kernel measured across the full width
domain on every claimed class; a kernel that wins on trj and regresses on M4
ships behind per-class dispatch or not at all.

### W5 — Memory + I/O behavior (QG-7, QG-9)

Bytes/doc itemization per the manifest (postings/positions/dict/blockmax/
idmap/tombstones); 16 KiB page-alignment audit for FSLX hot sections on Apple
Silicon (the format's 64-byte section alignment is not page alignment);
madvise policy for cold-open vs scan; QG-9's verified cache-state proof gets
`purge` on macOS and `drop_caches` on Linux, closing the doctrine's recorded
"reopens in-process" NoDecision.

## 5. Keep-gates (merged: repo contract + gauntlet discipline)

A lever is KEEP only if ALL hold:

1. Profile-first: hotspot evidence ≥0.1% self-time exists BEFORE the source
   touch, committed as a profile card.
2. Paired estimator (`quill-paired-estimator-v1`) A/B with same-invocation A/A
   null whose CI contains 1.0, per claimed machine class.
3. `release-perf` profile; never a bare `--release` with different codegen.
4. Focused and broad gates moved in the same run window (same git state, same
   target dir, same machine, same window).
5. cv_pct reported (provenance, never a gate per the repo contract; >5%
   flags the cell for rerun).
6. Differential parity campaign green at the same SHA (isomorphism oracle);
   rank-exactness explicitly attested.
7. Ratchet `Allow` on every claimed class; per-class KEEP/NoClaim split stated
   in the ledger row.
8. Ledger entry written before merge (`PERF_LEDGER.md` wins;
   `NEGATIVE_EVIDENCE.md` rejects with comparison class INCUMBENT vs SELF
   tagged, per the 2026-07-27 ledger gates).

Every lever bead body carries the recommendation contract:

```
Change:
Hotspot evidence (frame, % self-time, class):
EV score (Impact x Confidence x Reuse / Effort x Friction, >=2.0):
Machine-class scope:
Adoption wedge:
Budgeted mode (caps + on-exhaust behavior):
Isomorphism proof plan (parity campaign + rank-exactness):
p50/p95/p99 before/after target (as % of ceiling gap):
Primary failure risk + countermeasure:
Fallback trigger:
Baseline comparator:
Rollback:
```

## 6. Campaign anti-patterns

- Quoting any cross-machine-class ratio, ever.
- Running trj timed windows while the agent swarm builds on the same box —
  isolated target dir + exclusive build slot, or the A/A null vetoes the run.
- Treating M4 results as M5 results.
- Calling QG-6 done at smoke scale; the gate decision runs at 100K/1M with
  hierarchical per-query resampling.
- macOS commit-latency numbers without the Law-7 attestation.
- Touching `perf_matrix.rs`, `quill-perf-gates.toml`, `PERF_LEDGER.md`, or
  `NEGATIVE_EVIDENCE.md` without coordinating with their reservation holders.
- Optimizing anything without its hypothesis row and prior-mining pre-flight.

## 7. Bead map + lanes

Epic `bd-quill-e8-hyperopt-*` under the E8 doctrine (not a parallel taxonomy).
Children: `p0-*` (onboarding/calibration), `p1-*` (per-class profile cards),
`p2-*` (hypothesis ledger), `w1-*`/`w2-*` (specified levers), `w3-*`/`w4-*`/
`w5-*`/`math-*` (stubs behind Phase 0/1), `convergence`, `renegotiation`.
External anchors: `bd-h6eh` (instrument), `gwd4` (W1 base lever), `bd-6oiq` +
`x4e4.5.4` (x86 profiling cards), `bd-3beo` (consumes the renegotiation
evidence).

Lanes: **I** instrument (Phase 0, currently FoggySquirrel/CobaltWillow's
surface), **P** profiling cards, **O** one agent per workstream with
exclusive per-module reservations, **V** standing read-only auditor +
ledger/ratchet curation (the role that caught two instrument defects the week
this contract was written).

## 8. Cross-references

- Gate targets, activation, estimator, evidence artifacts:
  `docs/contracts/quill-perf-gates.md`, `quill-perf-gates.toml`.
- Ledgers: `docs/PERF_LEDGER.md`, `docs/NEGATIVE_EVIDENCE.md` (+
  `scripts/check_ledger_null_control.sh` commit gate).
- Runner: `scripts/perf-runner.sh` (machine-class provenance capture).
- QG-1 provisional miss row: `docs/NEGATIVE_EVIDENCE.md` § 2026-07-27
  (bd-h6eh). QG-6 attribution: `gwd4` bead body.
- Flip decision consumer: `bd-3beo`; conformance authority:
  `bd-quill-flip-conformance-release-gate-0r2p`.
