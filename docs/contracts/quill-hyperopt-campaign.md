# Quill Hyperopt Campaign (E8-H) — Multi-Microarchitecture Performance Offensive

**Status:** Normative campaign contract. Companion to `quill-perf-gates.md` /
`quill-perf-gates.toml` (which remain the authority on gate targets, activation
discipline, and the paired-estimator/evidence-artifact contracts — nothing in
this file overrides them).
**Owning bead:** `bd-quill-e8-hyperopt-*` epic (see § Bead map).
**Scope:** closing the measured Quill-vs-Tantivy performance deficit across
the existing x86 VPS fleet, one 64-core/128-thread AMD Threadripper PRO 5995WX
(`trj`), and independently keyed Apple Silicon M4/M5 hardware/execution
profiles under the repository's existing honesty machinery.

---

## 0. Premise, stated honestly

The campaign premise ("Quill ≥ 3.0x Tantivy at bulk indexing") is currently
**falsified by the admissible and diagnostic data produced so far**:

- **QG-1 (bulk indexing), provisional, gate inactive:** Quill at
  0.1066–0.1725x Tantivy docs/s on `bulk/medium/*` — 5.8–9.4x behind, a ~23x
  miss against the ≥3.0x target (`docs/NEGATIVE_EVIDENCE.md`, 2026-07-27,
  `bd-h6eh`).
- **QG-2 (single-worker indexing), gate inactive / no current claim:** commit
  `ebd91757` removed the post-join replacement-writer lifecycle defect and the
  corrected campaign passed the six-arm fairness audit. It still produced no
  admissible magnitude. Five terminal-lifecycle attempts ended in either
  invalid A/A dispersion or a candidate/rerun log-ratio delta of `0.036853`
  against the fixed `0.019803` reproduction law. Their diagnostic ratios
  cluster near 0.34–0.36x, far below the 1.5x target, but **are not a certified
  QG-2 number**. The terminal disposition is
  `INVALID-NULL + INVALID-REPRODUCTION / NO CLAIM`; QG-2 remains inactive and
  no corrected baseline replaces the quarantined pre-fix file. The five
  immutable attempt bundles and the exact retry predicate landed in
  `07718353`.
- **QG-6 (query latency), smoke scale only:** ~25x slower at 500 docs, with
  Round-0 profiling attributing >83% to immutable TERMDICT reparse/validation
  (`bd-quill-e8-perf-doctrine-x4e4.5`, `bd-quill-gauntlet-qg6-cache-termdict-gwd4`).

This campaign therefore does not start from "optimize Quill." It starts from
"measure every runnable lane and close each attributed deficit one lever at a
time." A miss honestly measured and ledgered is useful evidence, but it does
not lower a target and it never authorizes the Quill flip. The user has fixed
the terminal policy: Quill replaces Tantivy only after Quill wins the written
QG targets across the board. A win that cannot survive a hostile reading of
its own artifacts is a campaign failure. An unavailable hardware/profile key
limits claim scope; it never suppresses useful measurements on ready hardware.

## 1. Standing laws

Laws 1–5 are inherited verbatim from `quill-perf-gates.md` (no benchmark-only
semantics; distributions not averages; never hide maintenance; memory is
first-class; one lever per change). This campaign adds:

6. **Hardware/profile scoping.** Every claim is
   `(gate, fixture, hardware-class, execution-profile)` scoped. Ratios are
   NEVER compared, averaged, or quoted across canonical profile keys. Every
   profile carries its own A/A null tolerances, derived from measured
   dispersion on that profile, not copied from another profile. A lever may
   KEEP on one profile and NoClaim on another; the ledger row must say so
   explicitly.
7. **Platform-symmetric durability.** On macOS, any commit/durability-adjacent
   number is admissible only with `F_FULLFSYNC` treatment attested symmetric
   for both engines (macOS `fsync(2)` does not guarantee media durability
   without it). This is Law 1 applied to Apple Silicon; the attestation lives
   in the run's provenance JSON.

## 2. Hardware/profile matrix

`.bench-history/QG-<n>.<hardware-class>.<execution-profile>.latest.json` keys
baselines by one closed `MachineProfileKey` through an atomic pointer to
immutable run-ID-qualified threshold and evidence objects. Requested widths,
CPU counts, affinity, SMT, or scheduler settings never create or relabel a
key. The sole normative grammar, predicates, applicability plan, receipt
fields, reason codes, and shared conformance fixtures are in
[`quill-machine-classes.json`](quill-machine-classes.json); prose in this
document cannot weaken or override that contract.

| Canonical hardware/profile key | Hardware and capacity semantics | Campaign role | Mandatory provenance (beyond evidence-v5) |
|---|---|---|---|
| `x86-vps-ovh/x86-diagnostic` | Existing heterogeneous RCH workers; runtime-derived diagnostic capacity only | Diagnostic continuity; never supplies default-flip authority | Exact worker facts and receipt are retained. Every gate is `Diagnostic`, has no frozen maximum width, and cannot promote history |
| `trj-zen3-5995wx/physical-64` | AMD Threadripper PRO 5995WX; 64 physical cores, one admitted worker per core | Required default-flip profile; physical-core QG-1 scale-out plus QG-8/allocator/NUMA truth | Exact 64-core capacity; one-thread-per-core cpuset/sibling map, SMT state, NUMA policy, governor, observable worker activity, local execution, exclusive lease, and pre/post fingerprints. QG-1 max width 64; QG-8 max width 32 |
| `trj-zen3-5995wx/smt2-128` | Same host; 128 admitted logical threads with two siblings per physical core | Required default-flip profile; independent SMT scale-out/null-band truth | Exact 128-thread capacity and two-threads-per-core evidence. Lower literal widths remain profile-specific and never reuse `physical-64` baselines, null bands, or destinations. QG-1 max width 128; QG-8 max width 32 |
| `m4-macos/scheduler-10` | Apple M4 Pro, 10P+4E, 14 logical CPUs, 64 GiB, 16 KiB pages; scheduler-managed capacity 10 | Required default-flip profile, but currently promotion-unavailable until executing-image attestation lands; QG-8 additionally awaits a reviewed profile-aware 8-vs-4 target/evaluator | Requested pool/QoS, observable worker activity, thermal state, page size, local execution, exclusive lease, loaded-image attestation, symmetric durability where applicable, and sealed completion. The widest canonical QG-1/QG-8 cell is 8; no width-10 cell or P/E residency claim is invented |
| `m5-macos/scheduler-14` | No reachable, fingerprinted M5 host; capacity intentionally absent | Required future default-flip profile; no runnable cells while unavailable | `Unavailable` remains an explicit release blocker, not `NotApplicable`. No M5 fact, capacity, number, or applicability decision is inferred from M4 |

The registry is a specification plus executable conformance corpus:

| Surface | Normative coverage | Implementation bead |
|---|---|---|
| Identity and resolution | Closed hardware IDs plus closed execution-profile IDs, obsolete/unknown/cross-profile rejection, immutable fingerprint provenance | `bd-e8h-p0-machine-classes-26os` |
| Applicability plan | Exact registry-v2 plus matrix-v2 plus gate hash reconstruction; every canonical cell is `Required`, `Diagnostic`, or `NotApplicable` for one profile | `bd-e8h-p0-machine-classes-26os` |
| Receipt admission | Hardware identity plus explicit request/start/end execution snapshots, registry-derived capacity/max width, observed CPU assignment, SMT/NUMA/scheduler facts, canonical hashes, durability, and sealed completion-v6 | `bd-e8h-p0-machine-classes-26os` |
| Rust evidence and ratchet | Evidence-v5 reconstructs the plan; threshold-v7, finalizer, ratchet, and history pointer enforce exact profile/plan/destination identity and zero history writes on rejection | `bd-e8h-p0-machine-classes-26os` |
| Shell runner | Accepts only `--hardware-class` plus `--execution-profile`; callers cannot supply widths, Apple modes, lease paths, or legacy class aliases | `bd-e8h-p0-machine-classes-26os` |

Rules:

- Candidate and rerun on the same machine, same run window (existing
  promotion contract).
- Admission is per `(gate, fixture, hardware-class, execution-profile, source
  SHA, executable SHA, execution identity, applicability-plan hash)`.
  Onboarding or calibration for one profile never blocks diagnostic or
  activation-eligible work on another profile.
- `trj` and the Macs are NOT rch workers. Promotion-capable TRJ runs use
  `scripts/perf-runner.sh` (this campaign's one piece of new infrastructure).
  Its typed producer builds and resolves the exact benchmark ELF from the clean
  source snapshot and owns one continuous lease/probe/child/log/manifest/receipt
  lifecycle. Detached and foreground modes emit the same finalized layout;
  neither path writes history. M4 is a registered hardware/profile key, but
  Apple promotion remains fail-closed until a supported executing-image
  attestation design lands.
- Every M4 gate is deliberately promotion-unavailable today because a
  read-only `/dev/fd` alias cannot attest or execute the loaded image under the
  strict contract. A future profile-scoped contract must solve that boundary,
  retain the frozen `scheduler-10` applicability plan and a real
  scheduler-state witness, and retain the durability laws. QG-8 also needs a
  reviewed profile-aware 8-vs-4 target/evaluator because its current ratchet
  threshold is the x86-only 16-vs-4 requirement. The widest current canonical
  M4 scale cell is 8; neither a width-10 cell nor any M5 `scheduler-14`
  capacity may be invented. Trimming the x86 matrix ad hoc can never activate
  an Apple claim.

## 3. Phase plan

```
Phase 0  LANE-LOCAL INSTRUMENT INTEGRITY + ONBOARDING
Phase 1  PROFILE TRUTH                                     (per gate x profile; local, never rch — rch cannot symbolize, bd-e41k)
Phase 2  HYPOTHESIS LEDGER SEEDING + PRIOR MINING          (mandatory before a production lever)
Phase 3  INDEPENDENT OPTIMIZATION LOOPS                    (workstreams W1–W5)
Phase 4  MATH-FAMILY ARTIFACTS                             (bounded to 3 families)
Phase 5  DISCOVERY REVIEW + ACTIVATION + TERMINAL GATE
```

### Phase 0 — Lane-local instrument integrity + onboarding

1. `bd-h6eh` validator repairs land for any lane seeking activation
   (force_no_claim self-rejection, QG-6 corpus-digest scope, ignored
   hierarchical estimator). Every measurement sealed to a defective validator
   SHA stays diagnostic-only, retained with its exact defect; it is not hidden.
2. The QG-3/5/6 validity gaps from the 2026-07-28 read-only audit close:
   version-unique visibility probes, on-disk merge arms with reopen parity,
   prepared four-arm reuse, terminal per-cell artifacts + shard assembly so a
   timeout leaves receipts.
3. QG-1 retry predicate executed as written: `QUILL_PERF_FIXTURE` narrowed,
   `xlarge` + `tokenize_only` captured, complete `QG-1.json` emitted. The
   global rch timeout is not raised.
4. Hardware/profile onboarding: harden the shared runner, then validate it
   independently on `x86-vps-ovh/x86-diagnostic` (`bd-9f03f`),
   `trj-zen3-5995wx/{physical-64,smt2-128}` (`bd-7ihq9`),
   `m4-macos/scheduler-10` (`bd-0v8uz`), and
   `m5-macos/scheduler-14` only when a real host becomes reachable
   (`bd-swfyn`). Derive A/A-only bands independently for x86 (`bd-jjs6q`),
   each trj profile (`bd-cpqjb`), M4 (`bd-w7zxm`), and M5 (`bd-6wnws`).
   Threadripper calibration uses the exact frozen runnable cell set for each
   physical/SMT profile; lower literal widths never cross profiles. Current
   Apple calibration is keyed by scheduler-managed capacity 10 but exercises
   only the registry's canonical runnable widths (maximum 8). Report inferred
   P/E residency only as a diagnostic: a worker-count request is not evidence
   that macOS assigned those workers to performance cores. P-only calibration
   remains outside the registered producer until a real scheduler-assignment
   witness exists.
   At least two independent live calibrations are required for a band.
   Diagnostic A/B runs may execute earlier only when labeled non-claim.
   Manifest (`quill-perf-gates.toml`) additions for the new classes are
   coordinated with the manifest's reservation holder — not edited
   unilaterally.

**Lane exit:** the exact lane has a valid instrument/schema, matched workload,
required A/A calibration, complete immutable provenance, and a ratchet that
consumes that profile's artifact. QG-1, QG-2, and QG-6 diagnostics on
`x86-vps-ovh/x86-diagnostic` may advance while trj or Apple onboarding remains
incomplete. Conversely, Apple or trj lanes advance when their own
prerequisites pass. There is no fleet-wide measurement barrier, but every
`required_for_default_flip` profile remains a release requirement even when
its hardware is unavailable.

### Phase 1 — Profile truth

Local lanes (flamegraph + samply + dhat allocation census + `strace -c` /
`fs_usage` syscall+fsync census) under `release-perf` with frame pointers.
Deliverable per (gate x profile): a committed profile card with the top-10
self-time frames ≥0.1%, triangulated (a lever is actionable only when two
profilers agree on the frame). QG-1 cards additionally record the
**tokenize-only honesty denominator** — the measured ceiling that converts
"optimize indexing" into "close X% of the measured current-architecture gap."
If a class's ceiling is itself below Tantivy's measured throughput, that
evidence forces a new attack plan; it does not make the target negotiable.
`x86-vps-ovh` cards are owned by the in-flight pre-admission beads (`bd-6oiq`
for QG-1, `x4e4.5.4` for QG-6); the campaign adds trj/M4/M5 cards only.

### Phase 2 — Hypothesis ledger + prior mining

Every candidate lever gets a hypothesis row BEFORE implementation:
`hypothesis / minimal repro / expected signal (as % of ceiling gap) /
falsifiability / one-line invocation / hardware/profile scope / results-inline /
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

### Phase 3 — Independent optimization loops (see § Workstreams)

One lever per commit. Every lever ships: a recommendation-contract card (see
§ Keep-gates), same-window old-Quill/new-Quill causal evidence, a genuine
Tantivy incumbent arm, a same-invocation A/A control per claimed profile, and a
**green differential parity campaign at the same SHA** — the gauntlet is this
campaign's isomorphism oracle. Rank-exactness (the vendored BM25 fieldnorm table
and f32 op order in `contract.rs`) is the invariant most levers can silently
break; "Floating-point: identical" is a load-bearing line of every proof.

Each `(gate, fixture, hardware-class, execution-profile)` lane advances independently through hypothesis,
implementation, verdict, and re-profile. Chronological batches may still be
called rounds in status reports, but no round is a synchronization barrier and
no round count is an acceptance criterion. After three rejects in one
candidate family, switch veins.

### Phase 4 — Math families (complexity budget: 3)

Compiled artifacts only, never prose:

1. **Measured current-architecture tokenize-only ceiling** — the tokenize-only
   lane per class plus a derived `ceiling.json` (diagnostic max docs/s if the
   current architecture's inversion and seal work were free). This is not an
   information-theoretic or cross-architecture bound. Every W2/W3 lever states
   its expected signal as % of the remaining measured ceiling gap.
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

### Phase 5 — Discovery review, activation, and terminal gate

- A lane's current **discovery phase** reaches an auditable review boundary
  after one of three outcomes:
  1. it meets its predeclared target and passes independent confirmation;
  2. a measured current-architecture tokenize-only ceiling plus confidence
     bounds shows that the current architecture cannot meet the target; or
  3. two consecutive clean profile/discovery passes leave no unresolved
     high-impact hypothesis above the documented EV threshold, and every
     currently valid high-EV row is KEEP or REJECT with a retry predicate.
- Outcomes 2 and 3 end only the exhausted discovery phase. They leave the lane,
  its QG, and the campaign open as a MISS or NoDecision and require an
  evidence-review handoff to a materially different architecture, algorithm,
  data layout, parallelization strategy, or platform-enablement attack.
  Repeating a rejected family is forbidden unless its recorded retry predicate
  has become true.
- The performance campaign has one terminal convergence condition: every
  complete written target frozen in `quill-perf-gates.toml` is a validated WIN
  on every required profile and its `Required` cells. Infeasibility evidence, exhausted
  hypotheses, unavailable hardware, missing execution attestation, or a
  bounded non-inferiority result cannot close a gate, satisfy a dependency,
  authorize a waiver, or retarget the target.
- Gates activate per gate per profile exactly per the existing activation
  contract. A complete eligible lane need not wait for unrelated gates or
  hardware. Diagnostic runs remain visible but cannot activate a claim.
- **Round-6 evidence-review checkpoint (non-optional when triggered):** open
  the review as soon as discovery outcome 2 or 3 occurs while the lane remains
  below its target. Present absolute and relative results, ceilings, old/new
  Quill causal artifacts, the real Tantivy incumbent, null validity, parity,
  memory and durability effects, retained rejects, retry predicates, and
  unmeasured scopes. The output is the next materially different optimization
  attack plan, with an owner and executable falsification test. It must not
  recommend or enact a bounded-envelope target reduction.
- Performance and conformance proceed in parallel, but both block the library
  flip. Conformance is necessary and the complete written QG target surface is
  also necessary: no `AllowWithOptimizationDebt` or slower-but-bounded path is
  a valid flip trigger.

## 4. Workstreams

### W1 — Query fixed-cost elimination (QG-6, QG-9)

Discovery starts on the first calibrated machine with symbolized profiles;
M4/M5 diagnostic validation guides ARM64 optimization but cannot support a
promotion claim until its executing-image boundary is solved; it does not delay
an x86 profile or portable implementation.

| Lever | Basis | Notes |
|---|---|---|
| TERMDICT decoded-metadata cache | `gwd4` (P0, >83% attribution) | The named lever. Snapshot-scoped; budgeted (bounded bytes); deterministic full-reparse fallback on exhaustion. |
| Verify-once checksum memoization | reparse/validation attribution | Section checksums validated at first touch per snapshot, sealed thereafter; invalidated on generation change. |
| Collector allocation churn | `collect_id_hits` lazy-Vec sibling win | Audit Argus collectors for per-query allocs; apply the proven pattern. |

### W2 — Bulk-index single-thread cost (QG-1 and QG-2)

The provisional QG-1 thread=1 cell is 8.9x behind. Corrected QG-2 diagnostics
cluster near a roughly 2.8–2.9x deficit, but invalid null/reproduction evidence
is not a baseline and must not size a performance claim. The two contracts are
not interchangeable; both nevertheless justify profiling single-worker
per-document work. A usable QG-2 magnitude exists only after the §0 retry
predicate passes.

| Lever | Hypothesis | Caution |
|---|---|---|
| Term-interner hashing + arena | interner re-hash per field + SipHash-class cost | aHash/FxHash family is a proven repo-wide win; measure, don't assume. |
| Postings accumulation growth | per-term Vec realloc churn vs arena-backed exponential chunks (the incumbent's `expull` trick) | dhat census first. |
| Seal-time section checksum cost | if sealing hashes every byte with a heavy hash, it is a prime constant-factor suspect | Algorithm change = FSLX format-registry bump; EV-scored; registry owners coordinate. |
| Commit-path fsync count | batch directory syncs; two-slot manifest bounds the floor | `strace -c` / `fs_usage` census first; Law 7 on macOS. |

### W3 — Parallel scale-out (QG-1 high-thread, QG-8; truth profiles: `trj-zen3-5995wx/{physical-64,smt2-128}`)

Shard-per-worker indexing into independent segments, then
`KeeperWriter::concat_merge` (already exists) — N independent single-thread
problems plus one cheap concat. NUMA/CCD-aware sharding; per-thread arenas;
sharded interners merged at seal (no global interner Mutex). Deliverables
include the allocator-contention axis; "a bandwidth ceiling is honest, a lock
plateau is a bug." On M4, promotion must first attest the actual executing
image and scheduler state for `m4-macos/scheduler-10`; QG-1 and QG-8 exercise
the frozen canonical maximum width 8, not an invented width 10. P/E residency
remains diagnostic unless a witness proves it; scheduler capacity alone must
never be relabeled as `10P`. `m5-macos/scheduler-14` remains unavailable with
no inferred capacity or runnable cells. Until these boundaries are satisfied
there is no promotion-grade Apple scaling curve.

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
domain on every claimed profile; a kernel that wins on trj and regresses on M4
ships behind per-profile dispatch or not at all.

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
2. The claimed causal speedup compares old Quill and new Quill in one
   source-state-controlled window. The same evidence window also runs the
   genuine Tantivy incumbent and a same-invocation A/A null whose CI contains
   1.0, per claimed profile. A cross-commit Quill/Tantivy ratio or a
   Quill self-speedup alone is not an incumbent KEEP.
3. `release-perf` profile; never a bare `--release` with different codegen.
4. Focused and broad gates moved in the same run window (same git state, same
   target dir, same machine, same window).
5. cv_pct reported (provenance, never a gate per the repo contract; >5%
   flags the cell for rerun).
6. Differential parity campaign green at the same SHA (isomorphism oracle);
   rank-exactness explicitly attested.
7. Ratchet `Allow` on every claimed profile; per-profile KEEP/NoClaim split
   stated in the ledger row.
8. Both-engine absolute metrics, p50/p95/p99 or throughput distributions,
   RSS/bytes, disk bytes, and durability/maintenance work are reported beside
   ratios. An unavailable metric is explicit and limits scope.
9. Ledger entry written before merge (`PERF_LEDGER.md` wins;
   `NEGATIVE_EVIDENCE.md` rejects with comparison class INCUMBENT vs SELF
   tagged, per the 2026-07-27 ledger gates).

Every lever bead body carries the recommendation contract:

```
Change:
Hotspot evidence (frame, % self-time, profile):
EV score (Impact x Confidence x Reuse / Effort x Friction, >=2.0):
Expected recoverable fraction of measured gap:
Hardware/profile scope:
Adoption wedge:
Budgeted mode (caps + on-exhaust behavior):
Isomorphism proof plan (parity campaign + rank-exactness):
p50/p95/p99 before/after target (as % of ceiling gap):
Old Quill / new Quill / Tantivy / A-A evidence plan:
Source, executable, corpus, and worker identity:
Primary failure risk + countermeasure:
Fallback trigger:
Baseline comparator:
Rollback:
Reject retry predicate:
```

## 6. Campaign anti-patterns

- Quoting any cross-profile ratio, ever.
- Running trj timed windows while the agent swarm builds on the same box —
  isolated target dir + exclusive build slot, or the A/A null vetoes the run.
- Treating M4 results as M5 results.
- Waiting for every profile, every gate, or a fleet-wide phase before
  recording or acting on a locally valid diagnostic.
- Requiring an arbitrary number of rounds, or requiring all workstreams to
  produce a lever in lockstep.
- Calling QG-6 done at smoke scale; the gate decision runs at 100K/1M with
  hierarchical per-query resampling.
- macOS QG-3/QG-4/QG-5 numbers without a non-declarative witness that both
  benchmark arms used symmetric `F_FULLFSYNC`.
- Treating a Quill self-speedup, or ratios collected from different source
  commits, as proof against the genuine Tantivy incumbent.
- Hiding a failed, truncated, or validator-defective run instead of retaining
  it with a precise non-claim reason and retry predicate.
- Touching `perf_matrix.rs`, `quill-perf-gates.toml`, `PERF_LEDGER.md`, or
  `NEGATIVE_EVIDENCE.md` without coordinating with their reservation holders.
- Optimizing anything without its hypothesis row and prior-mining pre-flight.

## 7. Bead map + lanes

Epic `bd-quill-e8-hyperopt-*` under the E8 doctrine (not a parallel taxonomy).
Children: `p0-*` (onboarding/calibration), `p1-*` (per-profile cards),
`p2-*` (hypothesis ledger), `w1-*`/`w2-*` (specified levers), `w3-*`/`w4-*`/
`w5-*`/`math-*` (stubs behind their lane-local profile prerequisites),
`convergence`, `renegotiation`. The historically named `renegotiation` bead is
now an evidence-review and next-attack-plan checkpoint; it cannot lower a
target. `bd-quill-e8-hyperopt-nyps.1` owns the
progressive-admission contract repair and must close before convergence or
the evidence-review checkpoint, but it does not block measurements or
optimization.
External anchors: `bd-h6eh` (instrument), `gwd4` (W1 base lever), `bd-6oiq` +
`x4e4.5.4` (x86 profiling cards), `bd-3beo` (consumes the evidence-review and
next-attack record).

Lanes: **I** lane-local instrument and calibration (currently
FoggySquirrel/CobaltWillow's surface), **P** profiling cards, **O** one agent
per workstream with
exclusive per-module reservations, **V** standing read-only auditor +
ledger/ratchet curation (the role that caught two instrument defects the week
this contract was written).

## 8. Cross-references

- Gate targets, activation, estimator, evidence artifacts:
  `docs/contracts/quill-perf-gates.md`, `quill-perf-gates.toml`.
- Ledgers: `docs/PERF_LEDGER.md`, `docs/NEGATIVE_EVIDENCE.md` (+
  `scripts/check_ledger_null_control.sh` commit gate).
- Runner: `scripts/perf-runner.sh` plus `quill-perf-finalize` (registered-host
  lease, probes, exact child/log binding, receipt admission, and transactional
  finalization).
- QG-1 provisional miss row: `docs/NEGATIVE_EVIDENCE.md` § 2026-07-27
  (bd-h6eh). QG-2 pre-fix quarantined diagnostic: `73444b59` from timed
  source `5bb74e76`; corrected terminal-lifecycle attempts ended in an explicit
  no-claim closeout at `07718353`, so QG-2 remains inactive. QG-6 attribution:
  `gwd4` bead body.
- Flip decision consumer: `bd-3beo`; conformance authority:
  `bd-quill-flip-conformance-release-gate-0r2p`.
