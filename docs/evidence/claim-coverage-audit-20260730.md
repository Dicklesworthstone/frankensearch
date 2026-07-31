# Claim-coverage audit — how much of frankensearch's KEEP base rests on a vs-incumbent ratio (2026-07-30)

**Prompted by the fleet.** frankenfs audited itself against the policy that a
perf KEEP requires a vs-incumbent ratio and found 67 of 186 KEEPs without one.
This is the same audit, run on frankensearch. Nobody asked for it; it is run
because the priority order says to.

## THE NUMBER

> **147 KEEP claims total. 10 carry a vs-incumbent ratio measured with the real
> incumbent live in the same invocation. 137 do not. Zero satisfy this repo's
> own written `INCUMBENT` contract in full.**

That is **6.8% substantive coverage, 0% formal coverage** — worse than
frankenfs's 64%, and the honest number is the one that matters.

**Two levels, because they differ and both are true:**

| standard | passing | failing |
|---|---:|---:|
| **substantive** — a real incumbent arm ran side-by-side in the same invocation | **10** | **137** |
| **formal** — the `PERF_LEDGER.md` "Comparison-class contract": names the actual legacy incumbent, numeric incumbent ratio, numeric A/A null *from that same invocation*, and the executing ELF SHA-256 | **0** | **147** |

`grep -c "Actual legacy incumbent" docs/PERF_LEDGER.md` → **0**. Not one row in
the file carries the field its own contract requires.

## Claim base — how 147 was counted (reproducible)

`docs/PERF_LEDGER.md` records claims in three structural forms. All three were
counted; none double-counts another (verified: the BOLD-VERIFY sections at
`:693` and `:724` are not in the narrative set).

| form | count | selector |
|---|---:|---|
| (a) date-prefixed table rows with old/new/ratio columns | **83** | `^\| 20\d\d-\d\d-\d\d \|` |
| (b) narrative `##`/`###` sections asserting KEEP / WIN / LANDED | **54** | heading match, minus REJECT/WASH/VOID/HOLD/REFUTED/MEASUREMENT/ARTIFACT/CORRECTION/BLOCKER/FRONTIER |
| (c) BOLD-VERIFY workload rows with `Status = KEEP` | **10** | `^\| \`bold_verify` … `KEEP` |
| **total** | **147** | |

A stricter heading selector (`KEEP|WIN:|LANDED` only) yields 44 rather than 54
for form (b), because 10 older-format headings state the lever in the title
without a verdict keyword (e.g. ``2026-07-16 — `vector_at_f32` decodes f16 via
SIMD widen``). Those are claims and are counted. **Range on the total: 137–147.
The supported count is 10 either way.**

## The 10 that ARE supported — and they are stronger than their own label

All ten are the BOLD-VERIFY rows (`docs/PERF_LEDGER.md:686, 716-717, 751-755`
and two more), reported as "**Ratio vs Tantivy-class**". The ledger contract
disqualifies "Tantivy-class" as a proxy — so on a formalistic read these score
zero. **Reading the bench source instead of the label reverses that:**

- `frankensearch/benches/search_bench.rs:88` — `lexical: Arc<TantivyIndex>`
- `:202` — `TantivyIndex::in_memory().expect("create tantivy comparator index")`
- `:312-318` — `tantivy_only_search()` calls `fixture.lexical.search_doc_ids(...)`, `.expect("tantivy comparator search")`
- `crates/frankensearch-lexical/Cargo.toml:19` — `tantivy = { workspace = true }`

Both arms run **in one Criterion binary, over one fixture, in one invocation**.
That is real Tantivy 0.26.1, not a proxy. The label understates the evidence.

**What they still lack**, and why formal coverage is 0 rather than 10: no
`Actual legacy incumbent:` field, no A/A null recorded from that invocation,
and no executing ELF SHA-256. Converting these ten is a **documentation** task
against an existing measurement, not a re-measurement — the cheapest item in
the queue.

**Scope that must travel with them:** the ledger already says it —
*"a zero-hit incumbent win for the non-semantic hash/no-quality lane, not a
claim that hybrid search dominates Tantivy/Lucene/Meilisearch-class BM25
overall."* These compare frankensearch **hybrid** against Tantivy
**lexical-only**: different capability, not like-for-like. Three of the ten sit
at 0.96 or worse, i.e. near-ties.

## Ranked conversion queue

Ranked by how load-bearing each claim is where a user could act on it, per the
brief. **The first finding is that this ranking's top tier is empty.**

### Tier 0 — public/user-facing unsupported claims: **0**

Verified by grep across every published surface:

| surface | perf claims found |
|---|---|
| `README.md` | **0** — mentions benchmark *lanes* and artifact contracts; no speed claim, no "Nx", no "vs Tantivy" |
| `CHANGELOG.md` | **0** |
| all `crates/*/Cargo.toml` `description` | **0** |
| all `crates/*/src/lib.rs` rustdoc `//!` | **0** (one `2×` in `frankensearch-lexical:17` is a *scoring boost factor*, not a perf claim) |

**No user can act on any of the 137 unsupported claims.** Every one is
internal-ledger-only. That is a materially safer position than an unsupported
README number, and it is why nothing below is urgent in the
user-harm sense — only in the campaign-integrity sense.

### Tier 1 — convertible NOW; the incumbent arm and the harness already exist (~24)

| group | ~count | why convertible |
|---|---:|---|
| the 10 BOLD-VERIFY rows | 10 | already measured vs real Tantivy; needs the three missing contract fields, nothing re-run |
| Quill engine levers (`TermInterner`, vint decode, bitmap decode, posting-unpack dispatcher, `TopDocsCollector`, SWAR tokenizer, `consume_position_run`) | ~10 | Tantivy 0.26.1 is in-tree behind `tantivy-oracle`; `perf_matrix`/`frankensearch-quill-gauntlet` **already** runs both arms interleaved with A/A nulls in one invocation. Conversion is a re-run, not a build. |
| lexical-facade levers touching `search_doc_ids` / `collect_id_hits` | ~4 | see caveat below — likely reclassify, not convert |

**Expect conversion to produce losses, not wins.** P13/P14 measure Quill at
**0.484** of Tantivy on QG-2's own fixture. Converting a Quill KEEP from
"−16.2% instructions vs our previous self" to "vs incumbent" will mostly
produce ratios below 1.0. That is the point of the exercise; a coverage number
bought by only converting the claims expected to win is worse than no number.

**Also load-bearing on the campaign, and currently zero:** `.bench-history/`
holds artifacts for QG-1..QG-10 and **not one QG gate is activated** (all
`*.unmeasured.latest.json`; the two measured QG-1/QG-2 files record a deficit).
So the one harness in this repo capable of emitting a contract-satisfying
`INCUMBENT` row has emitted **zero** KEEPs — not because it is broken, but
because Quill currently loses. That is the true state and it should not be
softened.

### Tier 2 — convertible only by BUILDING an incumbent arm (~41)

| group | ~count | incumbent that would have to be wired |
|---|---:|---|
| `frankensearch-index` vector/ANN/SIMD (f16 & int8 dot, slab packing, FSVI gather, HNSW) | 33 | faiss / hnswlib / usearch — **none in-tree** |
| `frankensearch-embed` (hash embedder, JL projection, `CachedEmbedder`) | 6 | fastembed is already a dependency; model2vec is Python-side |
| `frankensearch-rerank` (native reranker CLS attention) | 2 | sentence-transformers cross-encoder — Python-side |

Real work, not paperwork. The ledger is already honest here in places —
several rows self-declare *"Ratio vs Tantivy N/A (vector tier)"*. **48
statements across the file explicitly record their comparator ratio as N/A**,
so the repo has been recording the gap rather than hiding it.

### Tier 3 — CANNOT be converted: no incumbent arm exists for the surface (~68)

Per the brief, this is called out as a different problem from "nobody got
around to measuring it". These are not deferred work; they are permanently
`SELF-SPEEDUP / MAINTENANCE` and the correct action is **relabelling**, not
measurement.

| group | ~count | why no incumbent exists |
|---|---:|---|
| `frankensearch-core` text canonicalization (`strip_markdown_line`, `nfc_normalize`, `QueryClass::classify`, `DocumentFingerprint`, `filter_low_signal`, `ParsedQuery::parse`) | ~40 | frankensearch's own preprocessing contract. No external product performs this exact transform, so there is nothing to run side-by-side. |
| ops/TUI (keymaps, cass preview, prefix source) | ~9 | control-plane TUI; no competitor product |
| fusion internals (RRF merge, NQC, hubness, smoothing, federated fuse) | ~19 | RRF/NQC/hubness are frankensearch's own algorithms. **Note:** the *quality* side of fusion **does** already carry genuine incumbent comparisons — BEIR + real Tantivy BM25 + `rank_bm25` rows in `NEGATIVE_EVIDENCE.md:5768-6401` — but those are nDCG/recall, not latency ratios, and they are measurements rather than KEEP claims. |
| lexical-facade levers | ~4 | these sit *on top of* Tantivy, so the only available comparator is Tantivy-without-our-optimization — a before/after by construction. Can never be `INCUMBENT`-class. |

## What is NOT being done in this turn

No claim was deleted, weakened, retracted, or relabelled. This is inventory
only, per the brief. The ledger gate (`scripts/check_ledger_null_control.sh`)
already enforces the contract on **new** rows — it passed on this session's
commit with `checked_new_rows=0` — so the exposure is entirely historical and
is not growing.

---

# CORRECTION 2026-07-31 (BlackThrush) — Tier 0 is NOT empty. `CHANGELOG.md` carries 10 public perf claims.

**The 2026-07-30 audit above got its own headline finding wrong.** It recorded
`CHANGELOG.md | 0` perf claims and concluded "Tier 0 — public/user-facing
unsupported claims: **0**". That is false, and it is false against the audit's
*own* reproduce command, which returns five matching lines, not zero:

```
$ grep -cE "[0-9]+(\.[0-9]+)?\s*[x×]|faster|speedup|beats" README.md CHANGELOG.md
README.md:0
CHANGELOG.md:5
```

The `README.md → 0` half is confirmed and stands. The `CHANGELOG.md → 0` half
was a misread of a non-empty result. `CHANGELOG.md` is rendered on the GitHub
repository front page; it is as public as the README.

## The 10 public claims (CHANGELOG.md:65-68)

| # | claim | line | comparator actually used | incumbent-live ratio? |
|---|---|---:|---|---|
| 1 | f16 dot products **3.6–4.0x** | 65 | our scalar/`wide` path | no |
| 2 | 4-bit slab pack **10.3–13.6x** | 65 | our prior scalar pack | no |
| 3 | FSVI slab write **6.4–7.3x** | 65 | our prior byte path | no |
| 4 | **"the fastest lossless vector-search primitive"** | 66 | *nothing* — unbounded superlative | no |
| 5 | parallelized MRL truncated scan **8.64x** | 66 | our serial scan | no |
| 6 | selective-filter gather **6.9–50x** | 66 | our prior filter path | no |
| 7 | fuse-step `doc_id` moves **7.8–21.5x** | 67 | our prior clone path | no |
| 8 | merge-structured `rrf_fuse` **1.31–1.46x** | 67 | our prior sort | no |
| 9 | ASCII NFC analyzer hot path **~45–368x** | 68 | our prior NFC path | no |
| 10 | Tantivy fast-field id materialization **up to 6.32x** | 68 | our own prior materialization, *not* Tantivy-as-incumbent | no |

**0 of 10 carry a vs-incumbent ratio.** Every one is a self-vs-self
before/after promoted to a public surface. Claim 10 is the trap worth naming:
it contains the word "Tantivy", so it reads as an incumbent comparison, but
Tantivy is the substrate being optimized, not the arm being beaten.

(`CHANGELOG.md:101` additionally carries "dense tier ~4.4x smaller
contribution" — a retrieval-quality contribution ratio, not a speed claim, so
it is excluded from the 10 and is separately supported by the BEIR harness.)

## Claim 4 is the load-bearing one

Claims 1-3 and 5-10 are bounded self-improvements: an over-claimed ratio
misleads about *how much we improved*, which is a campaign-integrity problem
but not a user-actionable one. **Claim 4 is different in kind.** "The fastest
lossless vector-search primitive" is an unbounded superlative about the world.
A user reading it could reasonably choose frankensearch over faiss, usearch, or
hnswlib on its strength. Its entire evidentiary base is
`docs/PERF_LEDGER.md:825` and `:827`, and both compare the 4-bit two-pass
against **our own int8 two-pass and our own flat f16 scan**:

> `| 888.2 µs | 831.4 µs | 0.936 (1.07× vs int8; 2.56× vs flat) | KEEP` — `:825`
> `... 3.09× vs flat — the fastest lossless in-memory vector-search primitive` — `:827`

There is no third-party arm anywhere in its provenance. It is therefore ranked
**#1 in the conversion queue** and converted in
`fsvi-4bit-vs-incumbent-20260731.md`.

## Effect on the audit's numbers

The three headline numbers are **unchanged** — 147 / 10 / 137 — because the 10
CHANGELOG claims are restatements of ledger rows already counted in the 137.
What changes is the *ranking*: the queue is no longer "Tier 0 empty, start with
the cheapest paperwork". It starts with a public superlative.

The audit's other counts were re-verified independently on 2026-07-31 and hold:
`Actual legacy incumbent` → 0, form (a) → 83, form (c) → 10, N/A rows → 48.
Form (b)'s **44** is mechanically reproducible; the **54** is 44 plus 10
hand-identified older-format headings, so the defensible total is a range,
**137–147, with 137 as the mechanically reproducible floor**.

## Method note

The audit ran the right command and misreported its output. A grep whose result
contradicts the conclusion drawn from it is the cheapest possible error to
catch and the most expensive to leave standing, because every downstream
ranking inherits it. Reproduce commands in this directory should be run and
their output pasted, not summarised.

## Reproduce

```bash
grep -c "Actual legacy incumbent" docs/PERF_LEDGER.md          # -> 0
grep -cE "^\| 20[0-9]{2}-[0-9]{2}-[0-9]{2} \|" docs/PERF_LEDGER.md   # -> 83  (form a)
grep -cE "^\| \`bold_verify" docs/PERF_LEDGER.md               # -> 10  (form c)
grep -cE "(original-comparator|Ratio vs Tantivy|comparator ratio).{0,40}N/A" docs/PERF_LEDGER.md  # -> 48
grep -nE "[0-9]+(\.[0-9]+)?\s*[x×]|faster|speedup|beats" README.md CHANGELOG.md  # -> no perf claim
sed -n '88p;202p;312,318p' frankensearch/benches/search_bench.rs  # the real TantivyIndex comparator arm
ls .bench-history/                                             # QG-1..QG-10, all unmeasured/deficit
```
Classification scripts: `classify.sh`, `inventory.sh`, `final_audit.sh` under
this session's scratchpad `audit/`.
