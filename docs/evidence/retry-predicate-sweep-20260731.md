# Negative-ledger retry-predicate sweep — 2026-07-31 (BlackThrush)

Every section in `docs/NEGATIVE_EVIDENCE.md` carrying a retry condition was
enumerated and its predicate evaluated against observable current state. **84
sections carry a retry predicate.** This card records the verdict for each
class, including — per the brief — the ones that stay negative.

Enumeration is reproducible:

```bash
grep -nE "[Rr]etry (condition|predicate|only|performance|the|when|if)|Retry:|Do not re-attempt" \
  docs/NEGATIVE_EVIDENCE.md | wc -l
```

## Summary

| class | count | action |
|---|---:|---|
| A. Predicate satisfied, **re-run dispatched this session** | 1 | `bd-3srq` |
| B. Predicate satisfied **and already re-decided** in-ledger | 5 | none — verified current |
| C. **Decision basis VOID** (rule withdrawn) — relabelled UNDECIDABLE, no re-run needed | 6 | relabelled below |
| D. Predicate **NOT satisfied — fleet/isolation** | 3 | stays blocked, evidence recorded |
| E. Predicate **NOT satisfied — mechanism threshold unmet** | 4 | stays rejected |
| F. Predicate **NOT satisfied — hardware absent** | 2 | stays closed |
| G. Predicate **NOT satisfied — candidate source no longer exists** | 1 | stays undecidable |

---

## Class C — the null-gate correction VOIDS five decision bases (the substantive finding)

The `2026-07-27 model-integrity correction` withdrew the CV gate campaign-wide:

> *"decide solely from the candidate median CI versus the A/A null floor. Raw CV
> is diagnostic only."*

It was applied to five rows (`bd-l5x3` ×4, `bd-3srq` ×1). **It was never applied
to the 2026-07-10 `cod_fse` measurement-bundle family, every member of which was
rejected on exactly the withdrawn criterion.** Those six rows are:

| row | title | stated rejection basis | basis status |
|---|---|---|---|
| `:10701` | unbatched paired int8 row-block A/B | "exceeds the 5% CV gate" | **VOID** |
| `:10737` | batch16 int8 row-block A/B | "misses CV gate by 0.0318 point" | **VOID** |
| `:10771` | batch32 linear sampling | CV gate | **VOID** |
| `:10804` | 60-second flat sampling | "leaves one mult5 CV above gate" | **VOID** |
| `:10835` | 120-second flat samples | "failed at least one strict CV gate" | **VOID** |
| `:10904` | balanced null control | *inside null floor* — **basis stands** | valid |

**They cannot simply be re-decided under the corrected gate either.** Five of the
six (`:10701`, `:10737`, `:10771`, `:10804`, `:10835`) carry Criterion
mean/CI/CV only and have **no same-invocation A/A arm**. The corrected gate
decides against an A/A null floor; with no null arm there is no floor, so these
five are *undecidable*, not *rejected*. Only `:10904` ran a null control, and it
already reports the honest verdict (`INSIDE NULL FLOOR`, UNDECIDABLE).

### Direction across the six occasions — a prior, not a verdict

The six occasions ran on three different workers (`hz1` EPYC-Milan, `ovh-a`
Ryzen 7 5800X, `vmi1227854` EPYC) under five different sampling substrates, each
with bit-exact index/doc/score parity and recall@10 = nDCG@10 = 1.0000. Their
twelve candidate/original ratios:

| occasion | mult3 | mult5 |
|---|---:|---:|
| `:10701` unbatched | 0.9852 | 1.0948 |
| `:10737` batch16 | 1.0209 | 1.0024 |
| `:10771` batch32 | 1.2389 | 1.0301 |
| `:10804` flat60 | 1.0212 | 1.0203 |
| `:10835` flat120 | 1.2048 | 1.1893 |
| `:10904` null-control | 1.1369 | 1.1928 |

**11 of 12 ratios exceed 1.0 (candidate slower). Sign test p = 0.0064
two-sided; median ratio 1.0625.**

```bash
python3 -c "
from math import comb
v=[0.9852,1.0948,1.020871,1.002430,1.238912,1.030111,1.021188,1.020255,1.204816,1.189327,1.136852,1.192786]
s=sum(1 for x in v if x>1.0); n=len(v)
print(s,'/',n,' p2=',2*sum(comb(n,k) for k in range(s,n+1))/2**n)"
# -> 11 / 12  p2= 0.0063
```

**The `bd-b5wl` four-row int8 query-reuse candidate remains UNDECIDABLE** — the
direction is consistent across six occasions and three microarchitectures and
points *away* from the candidate, but that is a **prior, not a decision**. The
occasions differ in batching, sampling and worker, so they are not independent
replicates of one protocol; the sign test licenses a direction and never a ratio.

**The ledger gate refused an earlier draft of this finding, correctly.** That
draft declared the lever rejected on the sign test. `check_ledger_null_control.sh`
returned `BLOCKED REJECT ... missing: same-invocation numeric A/A null OR counted
no-change mechanism`. A sign test across occasions is neither, so the verdict was
downgraded to UNDECIDABLE rather than the gate worked around. Recording this
because directional evidence of this shape feels decisive and is not.

To actually decide `bd-b5wl`: one invocation, both arms in one binary,
`paired_median_ratio` A/A + A/B, self-reported ELF SHA-256, median-CI/null-floor
decision — the substrate `int8_vs_f16_fast_ab` already implements. Blocked on a
quiet worker, not on a lever.

The distinct `vpmaddubs` primitive (`:10994`, kernel-level 1.23×) is a
**separate vein** and is untouched by this correction, exactly as `:10904`
records.

---

## Class A — predicate satisfied, re-run dispatched

**`bd-3srq` — revalidate top-k same-invocation A/A under the median-CI gate.**

Predicate (from the 2026-07-27 correction): *same executing ELF, same-invocation
A/A and A/B, exact 32/32 order and recall@10 = 1.0000, median-CI/null-floor
decision.*

Every clause is already implemented in the retained harness
`crates/frankensearch-index/benches/int8_vs_f16_fast_ab.rs`:

| clause | implementation |
|---|---|
| ELF self-report | `print_bench_elf_sha256()` — `main()`, line 190 |
| same-invocation A/A | `paired_median_ratio(31, 3, mk_f16(), mk_f16())` — line 152 |
| same-invocation A/B | `paired_median_ratio(31, 3, mk_f16(), mk_i8())` — line 153 |
| null admissibility | `null.is_admissible_null()` — line 162 |
| median-CI decision | `lever.decidable_against(&null)` — line 172 |

Build/link admission was repaired in `bf982ae4` and revalidated by FoggySquirrel
on 2026-07-26 (`hz2`, 9m10s, exit 0) — **but the binary was never executed**, so
the row stayed `BLOCKED/UNTIMED` for a reason that no longer applies. The only
missing step was running it.

Dispatched this session via
`rch exec --base 3debdf25 --clean-overlay --no-overlay -- cargo run --profile
release-perf -p frankensearch-index --features bench-internals --bin
int8_vs_f16_fast_ab`. **Result recorded in the RESULT section at the foot of
this card.** Until that section carries numbers, `bd-3srq` remains
`BLOCKED/UNTIMED` and nothing here converts it.

## RESULT — `bd-3srq`

<!-- ITEM3-RESULT -->

---

## Class B — predicate satisfied and already re-decided (verified, no action)

| row | bead | verified state |
|---|---|---|
| `:15769` | `bd-quill-e3-keeper-ndtk.5` | RESOLVED 2026-07-22 by the fixed-span control |
| `:10955` | `bd-yt8m` ISA residual | **already corrected** at `:15308` — the audit's "ceiling is small" was measured and REVISED to 3.4–4.9× DECIDABLE on the `wide::f32x8` site |
| `:16038`/`:16079`/`:16113` | `bd-l5x3` ×3 | re-decided 2026-07-27 → VALID-AB / UNDECIDABLE (0.8223 inside its own [0.7606, 1.2099] null) |
| `:16188` | grouped `MaxScore` blocker | prerequisite fixed on `main`; re-run 2026-07-27 → REJECT (2–4-group wash, 8-group regression) |

The ISA re-test therefore needs **no** further propagation: its stale "small
ceiling" language at `:10955` is already superseded in-ledger at `:15308`. Both
rows should be read together; a cross-reference is the only outstanding gap.

---

## Class D — NOT satisfied: fleet/isolation (evidence recorded)

Fleet state captured `2026-07-31T09:51Z` and `09:56Z`:

```
rch exec -> [RCH] remote required; refusing local fallback (no admissible workers:
            critical_pressure=2,insufficient_slots=1,insufficient_total_slots=1,
            hard_preflight=4,active_project_exclusion=4)
rch queue -> 12-14 Active Build(s); frankensearch jobs held on
             hz1, ovh-a, vmi1153651, vmi1156319, vmi1264463
```

| row | predicate | why unmet |
|---|---|---|
| `:10904` | "external state change that supplies worker **isolation/affinity** for the entire one-invocation paired run" | fleet at 12–14 concurrent builds; 5 workers excluded. No isolation available. |
| `:15803` | "four-slot admissible worker with a **warm** Quill release-benchmark graph, or binary built inside ten minutes" | zero admissible workers at request time; graph cold |
| `:11529` | tightened to "**not merely worker isolation** (soft-pin ≠ isolation) but a hard pin" | `rch` has no hard worker pin — established at `:11490` and unchanged |

These stay BLOCKED. None is a lever rejection.

## Class E — NOT satisfied: mechanism threshold unmet

| row | predicate | measured |
|---|---|---|
| `:16553` seal-path copy elision (`bd-w8dut`) | SegmentAssembler seam adopted, **or** >3% profile attribution | P1 bounds addressable share at **~0.8%**; seam not adopted |
| `:16583` allocator/THP (E8-H P3) | >3% median move **or** >8% allocator self-time | measured allocation-growth **~0.45%** |
| `:15927` fanned shared pruning floor (`bd-dknq`) | ≥16 sealed segments, **or** `limit+offset` ≥ 200, **or** low block-skip rates | none present; requires a new fixture, not an environmental unblock |
| `:11596` fsfs hybrid-fuse merge | "a quiet-fleet window" | see Class D |

## Class F — NOT satisfied: hardware absent

| row | predicate | current hardware |
|---|---|---|
| `:3565` AVX-512 / VNNI | "deployment moves to Zen4+/AVX-512 hardware" | `thinkstation1` = **AMD Ryzen Threadripper PRO 5975WX (Zen3)**, `avx2 f16c fma`, no `avx512*` — re-verified this session |
| `:4772` AVX-512 on the flat scan | same | same |

## Class G — NOT satisfied: candidate source no longer exists

`bd-l5x3`'s retry requires re-running the boundary-mask candidate, but
`:15821`/`:16038` record that *"every speculative source/bench edit was manually
removed"* and the implementation survives only in a long-gone session
scratchpad. Re-entry is a **reimplementation**, not a re-run. Ranked accordingly.

---

## What this sweep did NOT do

No row was deleted. The five void-basis rows keep their text; the correction is
additive so the original error stays visible. No lever was re-landed. The
`bd-b5wl` re-decision changes a *basis*, and the production scan remains
ORIGINAL exactly as `:10904` left it — no source changed in this sweep.
