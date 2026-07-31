# E8-H P13 — the Q2 deferral question, answered: the QG-2 indexing gap is a CONSTANT FACTOR, and "8.7x" does not reproduce (2026-07-30)

**Task:** answer the campaign's longest-standing open question with a
measurement rather than an argument — *is Quill doing per-token/per-document
work EAGERLY that Tantivy DEFERS, making the QG-2 deficit structural rather
than a constant factor?*

**Answer — conditional, and the condition is the fixture size.** The
eager/deferred asymmetry is **real in the code**, and its cost has **two
regimes**:

- **N ≤ 128,000 — constant factor.** Quill's per-document cost is flat-to-
  improving (19.12 → 17.26 µs/doc across 256x) while Tantivy's falls 2.8x and
  plateaus, so the paired ratio converges to a constant **0.456–0.482 from
  N=8,000 upward**. **QG-2's fixture is N=50,000, inside this regime**, so
  *"structural difference, not a constant factor"* is **FALSIFIED for QG-2**.
- **N > 128,000 — superlinearity switches on.** By N=512,000 Quill's cost rises
  to **21.01 µs/doc (+21.7%)** against Tantivy's +6.7%, and the Quill-specific
  excess grows **+36%** (8.86 → 12.05 µs/doc). The structural term is real; it
  simply lives **above** the gate that is being chased, in `Xlarge` territory.

**Second finding, larger than the first:** on the *identical* 50,000-document
QG-2 fixture — warm, generation-excluded, interleaved-paired, admissible A/A
null — the deficit is **2.07x pinned to one core** and **2.96x unpinned**, not
8.7x. The published 8.7x is the noisiest of the three existing measurements of
that cell and is not reproduced by either of the other two.

## Disposition

`VALID-AB / DIAGNOSTIC-CLASS`. Host `thinkstation1` is the local diagnostic
class `local-5975wx-32c`, **not** a registered campaign class: nothing here may
activate a gate, move a ratchet, or be quoted as a competitive ratio. Every
number is a within-host, one-ELF, one-session comparison — which is exactly
what a scale-sensitivity question requires and is why the class limitation does
not weaken the conclusion. **No Quill engine source was changed.**

## Provenance

| axis | value |
|---|---|
| Source | `git archive origin/main` = **`0355d4ad43c464c7a0478c2cb29dd052c4d5eed6`**, exported read-only to session scratchpad; `fast_cmaes` sibling satisfied by symlink |
| Why not the working tree | this checkout's `main` is **254 commits behind `origin/main`** and dirty with peer WIP. It predates `bd16d35d` (TermInterner identity-hasher swap, P6) and `504fa185` (interner span inline, P8) — **both are in the measured ELF**, so this card supersedes the pre-P6 attribution lineage |
| Toolchain | rustc 1.99.0-nightly (9f36de775 2026-07-19), pinned by `rust-toolchain.toml` `nightly-2026-07-20` |
| Cargo.lock | fresh resolve, sha256 `fd99abcdb07b164f123f4acd6522e748c3b56763672652b6f166116b3d1de98b`. **Not** the worktree's lock (`fe8fea8f…`): that one is 254 commits stale and `--locked` fails against `origin/main`'s manifests. Recorded rather than hidden. |
| Bench ELF (baseline, no overlay) | sha256 `c887c73bae0fa3318565fe8adeb1abe884e90a4f2e9a713e1ed40f421b91b348`, 81,683,464 B |
| Bench ELF (+ measurement overlay) | sha256 `c0ff90b9809987f06d75362263c051ec8975da8c6b0c07e7ec4f15c7b0595e9e` |
| Build | `cargo bench -p frankensearch-quill-gauntlet --features perf-harness --profile release-perf --bench perf_matrix --no-run`, `RUSTFLAGS="-C force-frame-pointers=yes"`, isolated `CARGO_TARGET_DIR=/data/tmp/cargo-target-h3-scaling`; `Finished release-perf in 6m 34s`, rc=0, **zero `[RCH] remote` lines** (verified local) |
| Build route | LOCAL under the Route-2 allowance. **`rch exec --base … --clean-overlay` cannot serve this task**: it has no artifact-retrieval mechanism, so it cannot return a locally-executable `release-perf` bench ELF. It stays the correct route for compile/clippy/test gates; this card lands no code, so no such gate applies. |
| CPU / kernel | AMD Ryzen Threadripper PRO 5975WX (Zen 3, 32c/64t, no AVX-512); Linux 6.17.0-35-generic; `amd-pstate-epp`, governor **powersave**; THP `madvise` |
| Affinity | harness-reported: `Cpus_allowed_list=8 (1 of 64 host logical threads)`, `process_available_threads=1`, ISA `aes avx2 bmi2 fma vaes` |
| Host load | shared with active peer `codex`/`rustc` sessions; 1-min load 5.5–15.4 across the window. **Relative paired ratios only; no absolute-throughput claims.** Every A/A null is published beside its A/B. |

## The instrument, and the one line of overlay it needed

The blessed QG-2 path (`bulk_metric_unpooled`) is the right instrument:

- corpus generation is materialised **before** `Instant::now()`, per batch
  (`index_batches_observed`, `perf_matrix.rs:877-878`) — the timed region is
  engine-only;
- arms are interleaved-paired with a tantivy-vs-tantivy A/A null and bootstrap
  median CIs;
- the Tantivy arm's terminal writer join is inside the measured region
  (`ebd91757` fairness fix);
- Quill runs its shipping default (`bulk_load_mode` NOT enabled, standing law 1).

But it exposes exactly **two** values of `N`: `MatrixScale::Smoke` clamps the
`medium` cell to 500, `Full` gives 50,000 — and switching between them *also*
switches `batch_documents` 250 → 5,000, confounding `N` with publish cadence.

**Overlay (measurement-only, diagnostic, deliberately not landed):** an explicit
`QUILL_PERF_DOC_COUNT_OVERRIDE` consulted by `MatrixScale::document_count`.
Unset ⇒ byte-identical upstream behavior. No engine code touched.

**The overlay self-verifies.** `corpus_manifest_hash` folds in the *effective*
document count, so every row below carries a corpus identity recomputable
offline:

```
sha256( b"bulk/medium/1/positions_on" ‖ u64le(N) ‖ u64le(CORPUS_SEED)
        ‖ u32le(VOCABULARY_SIZE) ‖ u32le(MAX_DOCUMENT_BYTES) )
```

All rows verified against their declared `N` before the numbers were read.

## Corollary established en route: any QG artifact's fixture size is recoverable from its corpus hash

The `MatrixScale::Smoke` clamp made "the 8.7x is really a 500-document
measurement" a live hypothesis. It is **false**, and the digest above settles it
without a rerun:

| corpus hash | document count |
|---|---|
| `d4dfe3e94929343a6eb6c11cf29d51098ff1915d82b040e4747e3373c5bee922` | 500 |
| `31272ba338d2a07389ce66677440ed763964207afb76448fec02c5524f4d0be8` | **50,000** |

Both committed QG-2 baselines (`QG-2.linux-x86_64-zen3.2026-07-28.json` and the
trj candidates) carry `31272ba3…`, so both ran at 50,000 documents. **This
technique generalises to every committed QG artifact** — fixture size is a
property of the recorded hash, not something that has to be trusted from a
commit message.

## THE TABLE — QG-2 scaling curve, one ELF, one host, batch size FIXED at 250

`bulk/medium/1/positions_on`, positions ON, writer heap 50 MB, threads=1,
`taskset -c 8`, warmup 5 rounds, 20 interleaved paired runs per row.

| N | Quill docs/s | **Quill µs/doc** | Tantivy docs/s | **Tantivy µs/doc** | excess µs/doc | paired A/B | A/B CI95 | A/A null | null CI95 |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| 500 | 52,303 | **19.12** | 46,281 | **21.61** | −2.49 | 1.1324 | [1.073, 1.181] | 1.0281 | [0.924, 1.325] |
| 2,000 | 52,524 | **19.04** | 82,401 | **12.14** | +6.90 | 0.6220 | [0.578, 0.765] | 1.0145 | [0.859, 1.149] |
| 8,000 | 53,307 | **18.76** | 113,729 | **8.79** | +9.97 | 0.4762 | [0.446, 0.497] | 0.9859 | [0.917, 1.048] |
| 32,000 | 57,744 | **17.32** | 128,144 | **7.80** | +9.51 | 0.4556 | [0.428, 0.478] | 1.0191 | [0.978, 1.071] |
| 128,000 | 57,950 | **17.26** | 119,063 | **8.40** | +8.86 | 0.4820 | [0.468, 0.540] | 0.9732 | [0.858, 1.061] |
| 512,000 | 47,597 | **21.01** | 111,594 | **8.96** | **+12.05** | 0.4246 | [0.406, 0.451] | 0.9744 | [0.917, 1.017] |

**Every A/A null contains 1.0.** Peak process RSS at N=512,000 was 1.92 GiB with
113 GiB free — the degradation in that row is not swap.

Batch-size control (rules out publish cadence as the driver): the same cell at
`N=50,000` with the harness's own **batch 5,000** (`QUILL_PERF_SCALE=full`, no
override) gives A/B **0.4840** [0.4648, 0.5008], null **0.9899** [0.968, 1.019]
— indistinguishable from the batch-250 rows at 32,000 and 128,000.

### Reading — there are two regimes, and the QG-2 fixture sits in the first

**Regime 1, N ≤ 128,000: constant factor.**

1. **Quill is flat-to-improving**: 19.12 → 17.26 µs/doc across 256x. If eager
   per-document work were superlinear here, this column would rise. It falls.
2. **Tantivy amortises a large fixed per-run cost**: 21.61 → 7.80 µs/doc, fully
   amortised by N≈8,000 (that fixed cost is ~6.8 ms). At N=500 Tantivy is
   *slower than Quill* (A/B 1.13).
3. **The ratio converges to a constant** 0.456–0.482 over 8,000–128,000 (16x).
   Essentially all of its movement between N=500 and N=8,000 is *Tantivy getting
   better*, not Quill getting worse. **QG-2's own fixture is N=50,000 — inside
   this regime — so the QG-2 deficit is a constant per-document factor ≈2.1x.**

**Regime 2, N > 128,000: superlinearity switches on.**

4. Between 128,000 and 512,000 Quill's per-document cost rises **17.26 → 21.01
   µs/doc (+21.7%)** while Tantivy's rises only 8.40 → 8.96 (+6.7%). The
   Quill-specific excess grows **8.86 → 12.05 µs/doc (+36%)**. Both A/A nulls
   are clean and both arms' cv is ≤11.8%, so this is not noise.
5. So the honest verdict is **conditional, not flat**: *"structural difference,
   not a constant factor"* is **FALSIFIED for the QG-2 fixture and everything
   below ~10⁵ documents**, and **SUPPORTED above it** — the onset is between
   128k and 512k, i.e. inside `PerfCorpus::Xlarge` (1,000,000) territory, not
   `Medium`.
6. Direction-of-fit for the mechanism: at the pinned 50 MB shard budget a flush
   lands roughly every ~25k documents, so live segments go from ~5 at 128k to
   ~20 at 512k. `resolve_document_id_in` is O(live segments) *per document*, so
   a 4x segment count predicts growth in the excess — observed +36%, i.e.
   directionally consistent but far from proportional. **The O(S) term is
   present and is at most a minority of the regime-2 growth**; the rest is
   unattributed and is the first thing a successor should profile.

## What Quill actually does eagerly that Tantivy defers (code inventory at `0355d4ad`)

The asymmetry is real. It is simply worth a constant, not a slope. Both engines
read at the same revision.

| granularity | Quill | Tantivy 0.26.1 via `frankensearch-lexical` |
|---|---|---|
| per **batch** | `snapshot().next_manifest()` — full `Manifest` clone (`index.rs:3110`) | — |
| per **document** | `snapshot().delete_document(&mut manifest, id)` (`index.rs:3112-3119`) → `validate_staged_manifest` (**whole-manifest validate + successor check, once per document**) then `resolve_document_id_in` (**IDMAP hash probe against EVERY live segment**, `keeper.rs:3193-3232`) | `writer.delete_term(term)` — **one opstamped enqueue; resolution deferred to merge/reader** (`lexical/src/lib.rs:2210-2212`) |
| per **document** | `canonical_metadata` + `canonical_document_preimage` (serde_json over the whole document), retained per doc in `PendingIdentity.canonical_content` until flush | nothing equivalent |
| per **document** | `document.id.clone()` x2 — into `PendingIdentity` and into `uncommitted_ids: BTreeSet<String>`, which grows until commit | `ord_table.push(DocId)` |
| per **token** | `TermInterner::intern` — hash + `HashMap<u64,Bucket>` probe + arena byte compare, **then** `append_token` writing **12 B** into three parallel `Vec<u32>` | hash + `TermHashMap` probe in the stacker arena, append VInt deltas — **the arena bytes *are* the posting stream** |
| at **flush** | materialise `Vec<FlushTokenRow>` (12 B/token, AoS) → `stable_digit_scatter` radix passes → build POSTINGS/POSITIONS/BLOCKMAX/TERMDICT | serialize: stream the arena lists, bitpack, build FST |

Two consequences, which must travel together:

- **The Q2 headline claim is still unimplemented**, independently confirming the
  audit in `p1-qg2-cross-engine-memmove-counters-20260729.md`. Q2 promises
  "cache-shaped sequential passes replace per-token hashmap random access"; the
  implementation performs the per-token hashmap probe **and then** materialises
  a 12-byte triple Tantivy never materialises. Quill pays **both** costs.
- **`resolve_document_id_in` is O(live segments) per document** — genuinely
  eager where Tantivy is genuinely deferred, and the structural term in the
  ingest path. Below ~10⁵ documents the live-segment count stays in the single
  digits (50 MB shard budget ⇒ a flush roughly every ~25k docs; publication only
  at commit or visibility lag) and the coefficient is inactive. Above it, the
  segment count grows and the measured excess grows with it — regime 2. Note
  that on a bulk load with all-new ids **every one of these probes misses**: the
  work is pure overhead by construction.

## Why "8.7x" is the wrong number to be chasing

Three measurements of the *same* cell at the *same* 50,000 documents:

| source | class | pinning | runs | Quill docs/s | Tantivy docs/s | A/B | A/B CI95 | Tantivy cv | A/A null |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| `351f5c6d` committed baseline | `linux-x86_64-zen3` (Ryzen 7 5800X) | none | 10 | 20,415 | 171,680 | **0.1181** | **[0.113, 0.269]** | **44.8%** | 1.0097 |
| trj committed candidate | `trj-zen3-64c` (5995WX) | none | 30 | 59,818 | 171,223 | **0.3498** | tight | — | 1.0146 |
| **this card, unpinned** | `local-5975wx-32c` | none | 20 | 57,583 | 169,436 | **0.3379** | [0.3226, 0.3677] | 10.0% | 0.9754 |
| **this card, one core** | `local-5975wx-32c` | `taskset -c 8` | 20 | 51,322 | 108,494 | **0.4840** | [0.4648, 0.5008] | 9.9% | **0.9899** |

- The published **8.7x is the noisiest of the four**: its own A/B CI spans
  [0.113, 0.269] — a factor of 2.4 — with a Tantivy arm at cv 44.8% whose
  `median_ci95_low` (49,718) is 3.5x below its own p50 (171,680). Its Quill arm
  (20,415 docs/s) is also 2.5–2.9x slower than the same arm on two other hosts,
  which no 5800X-versus-Threadripper single-thread difference explains.
- **Two independent hosts now agree unpinned: 0.3379 here vs 0.3498 on trj**,
  with Tantivy arms within 1.3% of each other (169,436 vs 171,223 docs/s). That
  is the reproducible number.
- **The pinned-vs-unpinned pair isolates the `threads=1` asymmetry on one host,
  one ELF, one session.** Going from one core to all cores at the *same nominal*
  `threads=1`, Tantivy gains **1.56x** (108,494 → 169,436) while Quill gains
  **1.12x** (51,322 → 57,583) — Tantivy still runs a docstore compressor and a
  segment updater at "threads=1". So of the 2.96x unpinned deficit, **~1.4x is
  thread count and ~2.07x is genuine per-document work.**

## Consequences for the campaign

1. **Restate the target arithmetic.** QG-2's bar is ≥1.5x. From 0.484 that is a
   **3.1x improvement**, not the ~13x the 8.7x baseline implies. The budget is
   now concrete: **Quill must go from 17.3 µs/doc to ≤5.6 µs/doc.**
2. **Split the hypothesis by regime instead of retiring it.** For QG-2 (50,000
   docs) the excess is *flat* at ~9 µs/doc, so levers there must be justified by
   their measured self-time share of that 9 µs — no growth term will rescue
   them. The structural hypothesis survives only above ~10⁵ documents, which is
   `Xlarge`/QG-7/QG-9 scope, and should be pursued **there**, not against QG-2.
3. **Stop validating levers at smoke scale.** At N=500 Quill *beats* Tantivy
   (A/B 1.13) purely because Tantivy has not yet amortised ~6.8 ms of fixed
   cost. Any lever A/B'd on a 500-document fixture is measured against a
   handicapped incumbent.
4. **The QG-2 fixture structurally cannot test the term-dictionary suspects.**
   `synthetic_spec` pins `vocabulary_size = 8,192` at *every* N
   (`perf_matrix.rs:552-560`), so the distinct-term count — and therefore the
   term dictionary, the interner, and any FST/MPH/ART construction cost — is
   **constant regardless of document count**. FST-per-segment, minimal perfect
   hashing, and ART term dictionaries are **unmeasurable on this corpus**. A
   vocabulary-growing corpus (Heaps'-law term growth) must exist *before* any of
   those levers can be scored. Filing that fixture is a prerequisite, not a
   lever.
5. **No combination of the current prime-list levers reaches the bar.** The
   named families sum to well under half of Quill's per-document cost, and the
   two largest have already been adjudicated: memmove REJECTED (incumbent pays
   the same toll), canonical-encode REJECTED (1.0175 < 1.03 hard bar),
   allocator family DEAD on this class, seal-path copy elision WASH. Reaching
   5.6 µs/doc requires removing *classes* of per-document work, not shaving
   frames.

## Next suspects, re-ranked by this card

1. **Per-document eager identity resolution on the append-only path.**
   `upsert_documents` calls `delete_document` for every document, paying a
   whole-manifest validate plus an all-segments IDMAP probe — for a bulk load
   whose ids are all new, so **every probe misses**. `validate_staged_manifest`
   has **never appeared in any profile card's frame list**: it is unmeasured,
   not measured-small. This is the single candidate that pays in *both* regimes
   — a flat component (per-document manifest validate) and the O(live segments)
   component that drives regime 2. Profile it first.
2. **Bisect the regime-2 onset** between 128,000 and 512,000 with arm-scoped
   profiles at both ends, and attribute the +36% excess growth to named frames
   before assuming it is `resolve_document_id_in`. Cheap: the override in this
   card's overlay makes it one env var.
3. **Actually implement Q2** (deferred token→term resolution), which the peer
   audit explicitly parked as "a structurally distinct follow-up". It is the
   only remaining lever that removes a whole per-token class rather than
   shaving it, and the flat regime is where class-removals are the only thing
   that can move a 3.1x bar.
4. **Vocabulary-growing corpus** as a *fixture* deliverable, unblocking the
   term-dictionary family (see consequence 4 above). Prerequisite, not a lever.

## Retry predicate for the deferral hypothesis

The hypothesis is **settled false for QG-2** and needs no retry there: a rerun
at N ≤ 128,000 cannot revive it, and re-litigating it against the `medium`
fixture is a banned re-dig.

It is **open above ~10⁵ documents**, where this card measured the onset. Pursue
it only with a fixture that actually drives the live-segment count up — an
`Xlarge` cell, a `scribe_shard_budget_bytes` small enough to force tens of
flushes, or an on-disk commit-bearing cell — and bisect the onset between
128,000 and 512,000 before spending a lever, so the growth is attributed to a
named frame rather than assumed to be `resolve_document_id_in`.

## Repro

```bash
git archive origin/main | tar -x -C <scratch>/src-main
cp <repo>/Cargo.lock <scratch>/src-main/    # then resolve fresh; record the sha256
RUSTFLAGS="-C force-frame-pointers=yes" CARGO_TARGET_DIR=/data/tmp/cargo-target-h3-scaling \
  cargo bench -p frankensearch-quill-gauntlet --features perf-harness \
  --profile release-perf --bench perf_matrix --no-run
# one row of the curve
QUILL_PERF_GATE=QG-2 QUILL_PERF_FIXTURE=bulk/medium/1/positions_on \
QUILL_PERF_DOC_COUNT_OVERRIDE=<N> QUILL_PERF_RUNS=20 QUILL_PERF_WARMUP_ROUNDS=5 \
QUILL_PERF_BUILD_PROFILE=release-perf QUILL_PERF_ENVIRONMENT_SHA256=<64hex> \
QUILL_PERF_OUTPUT_DIR=<out> taskset -c 8 <elf> --bench
# recover any committed artifact's fixture size, no rerun:
python3 - <<'PY'
import hashlib, struct
h = hashlib.sha256()
h.update(b'bulk/medium/1/positions_on')
h.update(struct.pack('<Q', 50000))                 # candidate N
h.update(struct.pack('<Q', 0x5155494C4C504552))    # CORPUS_SEED
h.update(struct.pack('<I', 8192))                  # VOCABULARY_SIZE
h.update(struct.pack('<I', 4096))                  # MAX_DOCUMENT_BYTES
print(h.hexdigest())
PY
```

Artifacts (session scratchpad, machine-local):
`nsweep/n{500,2000,8000,32000,128000,512000}/QG-2.json`,
`nsweep/n50000-unpinned/QG-2.json`, `qg2-full/QG-2.json`, `qg2-smoke2/QG-2.json`.
