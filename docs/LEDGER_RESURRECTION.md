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

## 8. Model-integrity re-audit: 2026-07-25 20:40 through 2026-07-26 00:35 EDT

The provider silently substituted a lower-capability model during this window.
The 21-commit census below therefore re-read every first-parent commit plus the
parallel formatting commit merged by `3407e060`. Existing ELF hashes, A/A
controls, and byte-identity artifacts were not rerun. The review instead
checked whether the workload reached the claimed code, whether each
behavior-preservation argument was complete, whether each performance
conclusion followed from its own numbers, and whether ungated code remained
maintainable.

`CORRECTED` means the original commit contained a false or incomplete claim or
proof, but the named later change (or this re-audit) repaired it. `SOUND` means
the commit's behavior and claims survive all four checks. No commit required
retraction.

| Commit | Verdict and fresh-eyes basis |
|---|---|
| `b9b3e969ec062455e07a6de33bb5127a8aa8ed89` | **CORRECTED** — the Rust formatting is behavior-neutral, but the accompanying `bd-l5x3` `REJECT/INVALID-CV` closure and byte-copy description were wrong; `c487a221`/§1 reclassify the measurement as `VALID-AB` and `dab52390` records that the stray files were stale copies. |
| `85d0cc9725c0856d05accd2d8302261008224ccd` | **SOUND** — one `rustfmt`-only grouped-MaxScore test-signature change; no executable or evidence semantics changed. |
| `75f4ccd978a482491103bd4ddf5cf07b70b351dc` | **SOUND** — the parallel copy of the same `rustfmt` change is redundant but behavior-neutral. |
| `3407e0601136e49282a60494783e3c0b5cd6897d` | **SOUND** — clean merge of the two identical formatting changes, with no conflict resolution or semantic delta. |
| `bb25935b92a540de2816cfcbacf9404071ac7900` | **SOUND** — the CASS oracle producer executes the real CASS query builder, preserves native `DocAddress` and score bits, expands cutoff ties before truncation, and explicitly covers prefix matching, filter-only browse, and structured filtering. |
| `5cc07b79ab49c4f606f945636bc5523960f298e4` | **CORRECTED** — useful tracker evidence survived, but its speculative `CheckpointPostingCursor` explanation for zero grouped pruning was false; `afcb1478`/`dab52390` replace it with the drained-buffer root cause later fixed by `73c61d74`. |
| `fcd4a4851adc5dbc470afc54ea5a82dfb17b7b88` | **CORRECTED** — the `295/398 VOID` census used a non-authoritative taxonomy and contradictory screen; `c487a221`/§1 replace it with the hand-adjudicated six-class `216/341` result and quarantine four untimed rows. |
| `c435328e09358488a7e97059e86e70436819038b` | **CORRECTED** — provenance plumbing fails closed, but the original CASS v1/v2 semantic preimages misstated lowercase, edge-ngram, field, blank-query, term, and ranking semantics; `ace575cb` replaces them with the executable v2/v4 contracts and live adapters. |
| `8e7f8fe178537883e506e1d0d21dd854ae1a0310` | **CORRECTED** — the production gate stayed safely closed, but the root-cause prose was wrong and the A/B fixture confused ingest batches with physical leaves and never built its claimed 1M shape; `73c61d74` fixes pruning correctness and `974a6453` repairs the evidence workload. |
| `afcb1478399bfe0af2828225ca13302a7026abde` | **SOUND** — blocks activation, demonstrates that the closed gate preserves the prior path, corrects the root cause, and explicitly rejects the tempting ceiling-only fix that would drop buffered documents. |
| `09f8aca819a6594c3cc79a3f18baa3c5fab4aa50` | **CORRECTED** — self-hash, same-invocation A/A, deterministic median CI, and the no-CV decision rule are sound, but ratchet reproduction initially accepted a different executing ELF; `308f038d` adds fail-closed SHA equality and `c487a221` completes admission parity checks. |
| `3070c936e8631d61c4585b394d6bce34e3ff6483` | **SOUND** — forwards the required provenance selectors and prints the already-sealed artifact between unambiguous retrieval markers; it changes transport, not results or gate judgment. |
| `3e135e2e27d0a12603b518b8485af4095b9ff667` | **SOUND** — folds only an empty JSON object to absent at the shared read boundary, retains non-empty metadata, and proves both hydrated search and deferred-fusion hydration without changing stored bytes or ordering. |
| `e6ecdbe32907fe3069aa9753b0f3a5ae20a7af45` | **CORRECTED** — the cache logic and pinned revision check are valid, but the hard-coded `/data/projects/frankensearch` manifest is not RCH 1.0.52's content-addressed worker path, so normal workers skipped the warm; this re-audit discovers the newest matching `/data/tmp/rch/frankensearch/<hash>/Cargo.toml` while retaining an explicit override. |
| `1b42dc9f7bd2dddc734d8e3d278f4b766a165359` | **SOUND** — removes two Cargo-unreachable, unreferenced stale source copies under the authorization recorded on `bd-kld2`; their differing blob IDs confirm deletion removes misleading phantom code rather than production targets. |
| `bf982ae40ca01466971b02f235f5f1dde42074d6` | **CORRECTED** — ordinary-bin/thin-LTO admission and the closed shipping gate are sound, but the first parity checks omitted token position metadata and exact top-k order; `c487a221` adds complete token equality/order proof and `4d6316a9` records build-only link admission without inventing a performance result. |
| `dab52390ad270c1360cc683a67953f852831852d` | **SOUND** — accurately corrects the stale-copy characterization, records recoverability and authorization, preserves the open E4.10 cross-engine obligation, and states the grouped-pruning correctness trap before implementation. |
| `f2c0694f0b934f85615f3081752b4401faeda94c` | **CORRECTED** — it blocks one common `VOID-NONULL` write shape, but did not implement candidate lookup, KEEP SHA enforcement, or the mandatory local hook and used detached added-line heuristics; `c487a221` replaces it with the complete staged-blob three-part preflight plus CI. |
| `73c61d749d7552b83276b52e0b2a47d0c299e60d` | **SOUND** — candidate discovery enumerates the current document, untaken buffered residue, and live child cursors; the construction-time ceiling is conservative, scoring order is unchanged, and the new k=3 fixture makes dropped residue observable before the gate can open. |
| `79cce4b4ba36d38634443e633b4b80712f0c4071` | **SOUND** — closes only the now-proven pruning and feature-gating blockers, keeps grouped activation disabled, and retains the independent A/B admission requirement. |
| `3ba2802b6fe3b7447bf5b4c63dc8cf1c4c4c0604` | **SOUND** — tracker-only handoff correctly places empty-index visibility at open/consumer boundaries, avoids per-query warning state, and does not claim an implementation or measurement. |

The audit therefore yields **12 SOUND, 9 CORRECTED, 0 RETRACTED**. These are
judgment verdicts, not new benchmark results. The allocation remains Lane B:
no measurement was run and no QG baseline was fabricated.

## 9. Model-integrity remediation closure: 2026-07-27

Every `CORRECTED` verdict above now maps to a concrete fix that is an ancestor
of `origin/main`. There are no fix-required verdicts whose repair exists only
in a worktree, unmerged branch, tracker comment, or prose promise.

| Corrected commit | Landed remediation |
|---|---|
| `b9b3e969` | `c487a221` installs the authoritative taxonomy/admission contract; `dab52390` corrects the stale-copy characterization. |
| `5cc07b79` | `afcb1478` and `dab52390` replace the speculative cursor diagnosis; `73c61d74` fixes the real drained-buffer candidate enumeration. |
| `fcd4a485` | `c487a221` replaces the v1 census with the hand-adjudicated six-class `216/341` result and four untimed quarantines. |
| `c435328e` | `ace575cb` replaces the inaccurate CASS preimages with executable v2/v4 contracts and real Quill/Tantivy adapters. |
| `8e7f8fe1` | `73c61d74` repairs grouped pruning correctness; `974a6453` repairs the evidence workload and document-count assertion. |
| `09f8aca8` | `308f038d` requires rerun/executing ELF SHA equality; `c487a221` completes the harness parity and ledger gates. |
| `e6ecdbe3` | `93534e8f` replaces the nonexistent fixed worker path with content-addressed RCH project-root discovery. |
| `bf982ae4` | `c487a221` adds full token metadata and exact top-k order parity; `4d6316a9` records build/link admission without inventing timing. |
| `f2c0694f` | `c487a221` replaces the partial added-line heuristic with candidate lookup, staged-blob reject/KEEP enforcement, pre-commit wiring, and CI. |

The downstream-citation sweep covered `NEGATIVE_EVIDENCE`, `PERF_LEDGER`,
all repository READMEs, scorecard-named files/content, and every bead body.
This incident window contains **zero `RETRACTED` verdicts**, so it has no
retraction-dependent claims to unwind. It did expose stale dependents of a
`CORRECTED` verdict: five mirrored `bd-l5x3`/`bd-3srq` sections in each ledger
and the two bead contracts. All ten ledger sections now carry adjacent
model-integrity corrections, `bd-l5x3` is closed as `VALID-AB / UNDECIDABLE`,
and `bd-3srq` is closed as `BLOCKED/UNTIMED` under the median-CI contract.
README and scorecard scans found no incident-window dependent claim.

The gate itself was not complete at the start of this closure pass. A synthetic
`REJECT/INVALID-CV` row with a numeric same-invocation A/A null still passed:
the guard enforced null presence but not the decision rule. The remediation
adds a fail-closed CV-verdict check and an executable six-case self-check:

1. `VOID-NONULL` rejection is blocked.
2. CV-only rejection is blocked even when an A/A null exists.
3. A median-CI effect inside its same-invocation null is admitted as a
   no-ship/null-contained row.
4. A counted unchanged mechanism is admitted.
5. A KEEP without an executing ELF SHA-256 is blocked.
6. A KEEP with a 64-hex executing ELF SHA-256 is admitted.

The tracked pre-commit hook and ledger-integrity CI both execute that
self-check. A historical replay beginning before `f36ea6ce` now exits 2 and
identifies the original `bd-l5x3` row specifically as `BLOCKED CV-VERDICT`.
The candidate preflight also exits 2 for the boundary-selection-mask/tokenizer
surface and prints the existing retry context.

Closure counters: `verdicts_total=21`, `fixes_landed=9/9`,
`downstream_citations_corrected=12`, `gate_selfcheck_pass=true`.
