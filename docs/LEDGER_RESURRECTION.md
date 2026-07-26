# Ledger Resurrection — frankensearch

Campaign §1 (`PERF_CAMPAIGN_2026-07-25`), cc / STRUCTURAL lane, agent `SageCardinal`.

Audits every entry in `docs/NEGATIVE_EVIDENCE.md` against the campaign's VOID
criteria: a row is **VOID** not because "the lever didn't work" but because
**the measurement could not have detected the lever**.

Source of truth for the audit: `docs/NEGATIVE_EVIDENCE.md` @ `da149fd9`
(16,151 lines, 398 `###` entries). Per campaign §4, **no row is ever deleted** —
this file annotates, it does not rewrite history.

## 0. Six-class taxonomy correction (authoritative)

The fleet-wide frankenfs taxonomy supersedes the v1 C1–C5 counts and ranking
below. These are the six classes, verbatim in meaning:

| Class | Hand-adjudication rule |
|---|---|
| `VALID-PROFILE` | Rejected before any source edit on a named frame with non-zero self-time plus a computed Amdahl ceiling. |
| `VALID-MECHANISM` | No A/A null, but refuted on a **counted** mechanism—unchanged instructions, cycles, syscalls, allocations, or faults—because a null control cannot change the fact that no work was removed. |
| `VALID-AB` | A/B with a recorded A/A null and the effect sits inside it. |
| `VOID-CV` | Killed **only** by a `cv < 5%` gate. |
| `VOID-ZEROSELF` | The target frame had approximately zero percent self-time in the profile the benchmark actually ran. |
| `VOID-NONULL` | A near-1.0 A/B, no A/A null, and no counted mechanism. |

The mechanical screen found 399 sections: 54 surveys and 345
verdict-shaped sections. Hand adjudication then removed four sections that
never timed from the REJECT denominator: an untimed candidate is a
`BLOCKED`/`UNTIMED` quarantine, not a seventh resurrection class. The corrected
six-class census is therefore:

| Class | Rows |
|---|---:|
| `VALID-PROFILE` | 1 |
| `VALID-MECHANISM` | 68 |
| `VALID-AB` | 56 |
| `VOID-CV` | **0** |
| `VOID-ZEROSELF` | 11 |
| `VOID-NONULL` | **205** |
| **Six-class total** | **341** |
| **VOID total** | **216 / 341 (63.3%)** |

The four untimed quarantines explain the earlier mechanical
`220 / 345 (63.8%)` figure; they are still measurement work, but they are not
REJECT evidence. The raw screen found only 9/345 verdict-shaped sections with
an executing-ELF SHA-256 and only two with explicit target-frame self-time.
Those provenance counts are reported on the raw denominator so this correction
does not silently change what was counted.

The important correction is mechanical: a row that contains an A/A null is not
`VOID-CV`, even when an obsolete CV rule participated in its prose verdict.
Accordingly, `bd-l5x3` is `VALID-AB`, not VOID: its A/B median 0.8223 lies
inside its own extremely broad A/A interval `[0.7606, 1.2099]`. That evidence
does not refute the lever, but it does correctly say the run is undecidable.
The old ranked queue in §4 is retained as v1 audit history, not as the current
rerun order.

`bd-b5wl` (`vpmaddubs` pass-1 scan wiring) is a high-EV future retry, but not a
VOID row: its recorded A/A makes it `VALID-AB`. It has 44.54% attributed
self-time, full correctness proof, and two same-binary runs that disagreed on a
shared worker. Its concrete retry predicate is an authorized Lane M window on
a quiet pinned `x86-64-v3` worker, with the same ELF SHA-256,
same-invocation A/A and A/B, and a median-CI/null-floor decision. No corrected
top-five VOID rerun was performed in Lane B.

### Ranking limit and resurrection queue

The requested “top five VOID rows by target-frame self-time” cannot be produced
honestly from this ledger. Only two verdict-shaped rows record target-frame
self-time at all: `bd-b5wl` records 44.54% but is `VALID-AB`, while `bd-i40y`
records approximately 0% and is `VOID-ZEROSELF`. Inventing four missing
self-time values would repeat the provenance failure this audit is meant to
remove.

The hand-read fallback queue below is therefore **not** a self-time ranking. It
keeps the one high-attribution retry first, then preserves the four untimed
measurement quarantines with concrete predicates. A Lane M owner may execute
these; Lane B did not benchmark them.

| Order | Entry | Audit status | Concrete retry predicate |
|---:|---|---|---|
| 1 | `bd-b5wl` — `vpmaddubs` pass-1 scan | `VALID-AB`, 44.54% self-time | Quiet pinned `x86-64-v3`; one executing ELF SHA-256; full parity; same-invocation A/A and A/B; decide only on median CI versus the null floor. |
| 2 | `bd-3srq` — direct gated ANN probe | `BLOCKED/UNTIMED` quarantine | Use the repaired directly linked probe; preserve 32/32 ordering and recall 1.0; record the executing ELF SHA-256; collect same-invocation A/A and A/B; gate only on median CI. |
| 3 | `bd-btgh` — mmap int8 SIMD byte store | `BLOCKED/UNTIMED` quarantine | Warm release artifact on one worker; exact 10k-by-384 byte parity; one executing ELF SHA-256; same-invocation A/A and A/B; median-CI decision. |
| 4 | `bd-x99j` — adaptive NQC single shift | `BLOCKED/UNTIMED` quarantine | Reopen only with the exact single-shift candidate and bit-parity fixture; warm one release ELF; same-invocation A/A and A/B; median-CI decision. |
| 5 | `bd-q9u4` — hyphen decomposition allocation | `BLOCKED/UNTIMED` quarantine | Exact lexical benchmark artifact; full token-text, offset, and position parity; warm one release ELF; same-invocation A/A and A/B; median-CI decision. |

---

## 1. Superseded v1 headline yield

| Metric | Count | % |
|---|---:|---:|
| Entries audited | 398 | 100% |
| **VOID** (any of C1–C4) | **295** | **74.1%** |
| — of which *strong* void (C1/C3/C4) | 71 | 17.8% |
| STANDS (rejection survives the audit) | 103 | 25.9% |
| Entries carrying a **binary-provenance sha256** | **10** | **2.5%** |
| Entries carrying **any A/A null control** | 67 | 16.8% |
| Entries carrying **self-time attribution** | **17** | **4.3%** |

The provenance numbers are the story. **388 of 398 entries (97.5%) record no
binary-provenance sha256** — and 8 of the 10 that do come from a single
two-day burst by one agent (`cc_fse`, 07-09…07-11). This is the frankenlibc
finding (0 of 93) reproduced almost exactly. In a repo where concurrent agents edit crates continuously — AGENTS.md
explicitly documents this as happening "multiple times PER MINUTE" — an entry
with no binary identity cannot prove which ELF produced its numbers.

### VOID criteria and their incidence

| Code | Criterion | Hits |
|---|---|---:|
| C1 | Claimed ratio lies **inside** the A/A null floor (the harness was rejected, not the lever) | 43 |
| C2 | **No A/A null control recorded at all** | 331 |
| C3 | Target never reached the timed path / target frame ~0% self-time | 30 |
| C4 | Gate applied was **`cv < 5%` on a shared, unpinnable rch worker** | 9 |
| C5 | No binary sha256 (recorded as provenance weakness; not sufficient alone) | 387 |

**Scoring discipline.** C2 alone voids a row *only* when the measured effect is
small enough to plausibly sit inside a null floor (any reported ratio inside
`[0.85, 1.18]`). A lever measured at 2.62× **slower** is decidable without a
null control, so those rejections STAND. This is why the honest figure is 74.1%
and not the 93.0% that a naive "no A/A ⇒ void" rule produces.

---

## 2. The dominant failure mode here is C3, and it is an *instrument* failure

frankenmermaid's void class was "the bench never routed through the code under
test". **frankensearch's is different and more mechanical: the benchmark never
linked at all.** 30 entries are `INVALID/HOLD` or `BLOCKED/UNTIMED` rows whose
text says, in the authors' own words:

- "never reached timing" (`bd-btgh`, `bd-x99j`, `bd-5hz0`, `bd-g8f1`, `bd-x6pa`)
- "hit a cold remote cache" / "cold RCH release target" (`bd-79bn`, `bd-d2a8`, `bd-q9u4`, `bd-bfuc`)
- "lost its retained warm binary to an RCH pool rewrite" (`bd-9urb`)
- "cannot reach its timed path" (`bd-3srq`, twice)
- "remains inadmissible" (`bd-l5x3`)

These are not perf findings. They are **rch scheduling and target-warmth
failures wearing a perf finding's clothes**, and campaign §3c names this repo as
INSTRUMENT-blocked for exactly this reason. Every one of these levers is
*unmeasured*, not *rejected* — and several authors said so explicitly and were
still filed into the negative-evidence ledger, where later agents (correctly
following the "grep the ledger before proposing a lever" hard gate) then treat
them as closed.

**That is the buried-win mechanism in this repo.**

### 2b. C4 is real here and it is self-inflicted

Nine entries were gated on `cv < 5%` per Criterion arm. Campaign §2.3 establishes
that this gate is **unreachable on this hardware** (floor ~12%) and, worse, that
`cv` does not track decidability. The most recent example is `bd-l5x3`
(2026-07-23), whose own table shows the candidate arm at CV 15.99% while the
*direction* was consistently favorable across three independent runs. The row
rejects the harness and records it as a lever rejection.

---

### 2c. The same failure mode reproduced live during this audit (2026-07-25)

The audit's thesis is that this repo files claims that were never verified in
the state they shipped in. That happened again, in the same session, and was
caught only because the gate change forced a full-suite run:

`argus::tests::grouped_max_score_matches_exhaustive_and_prunes` — landed in
`1b5a1018`, whose commit subject is literally *"prove grouped MaxScore prunes
without perturbing the top-k"* — **fails on `main`** with
`max_score_windows: 0`. The bit-parity half of the test passes; the *pruning*
half, which is the entire point of the commit, does not.

Cause: the test landed Jul 24 17:10. Deterministic query-fuel metering
(`ae5baa0d`, Jul 24 10:56) was developed on a parallel branch and only reached
this line through merge `afb7800d`. **The test was green on a tree that never
contained the change that breaks it.** Both sides merged cleanly as text; no one
re-ran it.

This is the C3 class with a new mechanism: not "the bench never linked" and not
"the workload never routed through the code", but **"the proof was never
re-executed in the merged state it now certifies."** In a repo where AGENTS.md
documents concurrent agents landing work "multiple times PER MINUTE", a green
test at authoring time is not evidence about `main`.

Ledgered as `bd-bt2t` (P1) and in `docs/NEGATIVE_EVIDENCE.md`. It is the reason
`GROUPED_MAX_SCORE_ENABLED` ships `false`: activating a pruning lever that
provably prunes nothing would have paid BLOCKMAX-open cost on the most common
query shape in the engine for zero benefit — and, had the gate been flipped
without running the suite, would have shipped as a "win".

**Audit rule this adds:** a row asserting a *behavioural* property (prunes,
skips, caches, short-circuits) must record the commit its proof was last
**executed** against, not the commit that authored it.

## 3. This repo has already completed one successful resurrection — and did not name it

`bd-r3rd` is the proof case, and it validates the campaign method end to end:

| Stage | Evidence |
|---|---|
| **2026-07-14 — INVALID/HOLD** | "fused generic analyzer never reached the retained lexical benchmark". Sync took 34s, target pool was cold, Cargo never emitted a compilation — exit 124. Filed as negative evidence. |
| **2026-07-18 — KEEP** (`375e4237`, CopperOrchid) | Re-run on **pinned worker `ovh-a`**, one release binary with **ELF SHA-256 `6778f06d…bbf58a`**, exact `Token` parity across 9 Unicode fixtures first, **A/A control `0.9960 [0.9790, 1.0173]`**, fused/original median **`0.9634 [0.9514, 0.9808]`** — below the null p5, therefore **decidable ~1.038× win**. |

That is precisely the §2 contract (pinned worker + self-reported ELF sha + A/A
null in the same invocation + **median-CI gate, not `cv`**) applied to a VOID
row, yielding a shipped win. The repo executed the method once, by hand, and
filed it as a routine closeout rather than recognising it as a repeatable
procedure. **It is the template for everything below.**

Note the contrast with `bd-l5x3`, which was re-run three times under *ordinary
fleet scheduling* and killed each time by the `cv` gate. The difference between
the two outcomes is **the worker and the gate, not the lever.**

---

## 4. Resurrection queue (ranked)

§1 asks for a ranking by profile self-time of the target frame. Only 17 of 398
entries record self-time at all, so self-time is used where present and
otherwise the rank is: **(a) does the target sit on the live Quill engine path**
(the thing the repo is actually shipping and measuring against Tantivy),
**(b) how large and how reproducible was the directional signal**, **(c) is the
blocker now removable**.

| # | Entry | Void code | Directional signal | Why it ranks here | Retry gate |
|---|---|---|---|---|---|
| 1 | `bd-l5x3` short-token ASCII boundary selection mask (3 rows: 07-22, 07-23 ×2) | C1+C3+C4 | **0.8223 median** short corpus (≈1.22× faster), favorable in **3 independent runs** | Live Quill scribe/analyzer tokenizer path. Named in campaign §3c. Direction never once reversed; killed only by the `cv` gate and a ±20% A/A band | Pinned quiet worker; `min_sample=2ms, min_of=3`; median-CI gate; ELF sha |
| 2 | `bd-3srq` top-k A/A CV revalidation (2 rows: 07-22, 07-23) | C3+C4 | n/a — **never linked** | It *is* the harness bead. Resurrecting it adopts §2 parts 1–3 and unblocks every other row | Warm worker-scoped release target for `int8_vs_f16_fast_ab` |
| 3 | `bd-quill-e1-scribe-bejd.7` Delta hash chains / columnar bulk seal | C3 | 1.29 / 1.46 / 1.14 | Live Quill ingest path; three separate favorable numbers, whole-path claim held back as UNTIMED | Same-binary paired A/B, setup-inclusive |
| 4 | `bd-5hz0` fused ASCII query classification | C3 (07-14) then **STANDS** (07-16) | 2.62× **slower** on re-measure | Included only to mark it **correctly closed** — the 07-14 UNTIMED row was later properly measured and is a genuine REJECT. Do not re-dig | — (closed, correctly) |
| 5 | `bd-9urb` score-only bulk sidecar traversal | C1+C3 | inside stored-baseline floor | Lost its warm binary to an RCH pool rewrite, then re-measured against a *stored* baseline rather than a same-invocation A/A | Same-invocation paired A/A + A/B |

Rows 1–3 are live resurrection candidates. Row 4 is listed to record that the
audit **confirmed a rejection** — the queue is not a list of assumed wins.

---

## 5. Re-run status

| Rank | Entry | Action taken this round | Outcome |
|---|---|---|---|
| 1 | `bd-l5x3` | **Audited → VOID.** Bead had been closed REJECT at the top of this session on the pre-campaign `cv` gate; that closure is now recorded as resting on a void measurement. Re-run requires a pinned quiet worker (§3c blocker, not yet cleared). | **Queued, not re-won.** Blocker: no pinned/idle worker available; every rch slot this session was shared. |
| 2 | `bd-3srq` | Audited → VOID (C3, never linked). | Queued. Enabler for the rest. |
| 3 | `bd-quill-e1-scribe-bejd.7` | Audited → VOID (C3). | Queued. |

**Honest yield this round: 398 audited / 295 void / 0 re-run / 0 re-won.**

### 5a. Lane B admission follow-up (2026-07-25)

The allocation addendum reassigned frankensearch to BUILD/FIX and explicitly
forbade benchmarking or taking a worker. Commit `bf982ae4` therefore fixes the
instrument without pretending to complete a resurrection:

- former v1 rank 1 (`bd-l5x3`) now has a retained, bench-only boundary-mask
  candidate and a direct `cargo run --profile release-perf` probe. It performs exact
  candidate/shipping/scalar parity before a same-invocation shipping/shipping
  A/A and candidate/shipping A/B; the production tokenizer is unchanged.
- former v1 rank 2 (`bd-3srq`) is likewise a direct probe rather than a
  Criterion bench, preserving the 32-query recall/order oracle plus A/A and A/B.
- both probes self-report the executing ELF SHA-256 and decide only on
  bootstrap median CIs/null-floor separation. `cv_pct` is not a gate.
- thin LTO, opt-in Quill/Tantivy oracle dependencies, and valid job-scope
  workflow concurrency remove the three observed pre-timing admission costs.

The corrected taxonomy makes `bd-l5x3` a valid but undecidable A/B rather than
a VOID row; `bd-3srq` is an untimed quarantine outside the REJECT denominator.
The former v1 queue is not the current top-five rerun order, and no
resurrection result changed: honest yield remains **0 re-run / 0 re-won**.
Retry only in an authorized measurement window using the repaired same-binary
contract. Three REJECTs still require a vein switch; BLOCKED/UNTIMED and
inactive-gate quarantines do not count as lever REJECTs.

The original audit stopped here because `rch queue` showed every frankensearch
slot sharing workers with other campaign jobs. Re-running rank 1 then would
have reproduced the ±20% A/A band that voided it three times already.
**Reporting a re-run under those conditions would have manufactured a fourth
void row, not a resurrection.**

The follow-up clears the source-level admission defects, not the measurement
allocation. The remaining prerequisite is an authorized Lane M window on an
isolated/pinned worker. That request belongs on the campaign thread instead of
being burned as another invalid attempt.

---

## 6. Standing rules this audit adds

1. **An `INVALID`/`UNTIMED` row is not negative evidence.** It belongs in a
   blocker log, not in `NEGATIVE_EVIDENCE.md`, because the "grep the ledger
   before proposing a lever" hard gate then suppresses an unmeasured lever
   forever. The v1 screen over-counted this shape; the corrected verdict screen
   quarantines four such sections outside the REJECT denominator.
2. **Never gate on `cv`** (§2.3). Gate on: claimed ratio outside the arm's A/A
   null 95% CI with a 2× margin. Report `cv` as provenance only.
3. **Every timed row records the ELF sha256 of the binary that produced it**,
   self-reported by the bench binary from `env::current_exe()` (§2.1). The
   corrected raw screen found 9/345.
4. **A/A null control in the same invocation, always** (§2.2).
5. The `bd-r3rd` closeout (§3 above) is the worked example. Copy it verbatim.

## 7. Institutionalized preflight

Commit-time enforcement lives in `scripts/check_ledger_null_control.sh`.
Before proposing a lever, provide both its name and target surface:

```bash
scripts/check_ledger_null_control.sh \
  --candidate 'proposed lever' \
  --surface 'crate::module::target_function'
```

Exit 0 means no prior negative-evidence section matched. Exit 2 prints the
matching section and its retry predicate, and blocks the candidate until the
predicate is satisfied or the lane switches veins.

The staged-row mode reads each complete section from the staged blob rather
than grepping detached added lines. It returns exit 2 when:

- a new REJECT has neither numeric same-invocation A/A evidence nor an
  explicitly unchanged counted mechanism; or
- a new KEEP has no executing ELF/binary SHA-256.

The tracked `.githooks/pre-commit` invokes that mode, and this checkout uses
`core.hooksPath=.githooks`. The same check runs in
`.github/workflows/ledger-integrity-lint.yml`, so bypassing a local hook does
not bypass the repository gate. `--all` is deliberately labeled a mechanical
screen and never treated as hand adjudication.
