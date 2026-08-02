# E8-H P9 — GENERATOR-CORRECTED arm-scoped attribution, quill QG-2 ingest, local-5975wx-32c (2026-07-29/30)

**Publication provenance:** integrated after `a61456ec` in evidence-set
commit `5b91f680`; measurements are diagnostic/NoClaim per card scope.

**Task:** the sibling card `docs/evidence/e8h/p1-local-qg2-incumbent-copy-comparison-20260729.md`
(commit `5b19f4f2`) proved the QG-2 200k in-memory profiles are contaminated:
the harness corpus generator runs inside the measured child and owns ~20-22%
of BOTH arms, and the Round-1 "canonicalization ~12-15% family" figure was
suspected of being inflated by generator frame-name collisions (fmt/serde).
This card partitions every sample of a fresh quill-child profile by CALL
CHAIN — generator subtree vs engine subtree — and re-derives the engine-only
family table plus the lever ranking. The measurement pass itself made zero
shared-tree edits, commits, or tracker calls; this document is its published
diagnostic record.

**Headline: the deflation feared by 5b19f4f2 does NOT kill the canonical
lever. Net of ALL generator cost, canonicalization/identity is 16.2% of the
engine side (11.8% of the child) as a subtree, 20.1% / 14.6% with ingest-side
xxh3 — because subtree accounting returns the serde/alloc/copy children that
self-time tables undercounted. Canonical-encode survives as the only >=2.0
lever.**

## Machine-class fingerprint (Law 6: not comparable to trj-zen3-* or m4-macos)

| axis | value |
|---|---|
| Host | `thinkstation1` — class `local-5975wx-32c` (dev host; diagnostic only, NOT ratchet-admissible) |
| CPU | AMD Ryzen Threadripper PRO 5975WX, Zen 3, 32c/64t, SMT on |
| Kernel / governor | Linux 6.17.0-35-generic; `amd-pstate-epp`, governor **powersave** (relative self-% only) |
| Load during runs | ~11/64 threads background; quill runs pinned `taskset -c 8`, tantivy runs unpinned (see asymmetry note) |
| Toolchain of ELF | rustc 1.99.0-nightly (9f36de775 2026-07-19) |
| perf | 6.17.13, `perf_event_paranoid=1`, `perf_event_max_sample_rate=7000` |

## Provenance

- ELF: the pass-2 stashed baseline **`scratchpad/elfs/perf_matrix_base_iso`**,
  SHA-256 **re-verified before use** =
  `9c3cacf0fa0ab66b46b9fb9482c1b8e858985a02b4e7775ef47dec574f22078b`
  (pristine `git archive` of continuity base **`3684b1477`**, the P1/P2/P6/P8
  lineage; `perf_matrix.rs` and `generator.rs` byte-identical to `origin/main`
  per 5b19f4f2). **No rebuild performed** — a valid stashed ELF existed.
  Pre-P6/P8: contains NEITHER banked lever (deliberate — measures the
  published baseline; P6/P8 corrections applied narratively below).
- Method choice — **(a) call-chain split**, not (b) pre-generated corpus:
  `run_memory_child` (perf_matrix.rs:3119) calls `index_batches` →
  `index_batches_observed` (perf_matrix.rs:526), which **interleaves**
  `generated_batch` (corpus materialization) with `index_documents` per
  250-doc batch. There is **no seam** that generates the corpus before the
  measured window, and the gauntlet crate is read-only for this pass, so (b)
  is unavailable. The split is sound because the two subtrees are disjoint
  by construction: no engine frame ever appears under `generated_batch`, and
  no generator frame under `index_documents` (measured cross-contamination
  0.76-1.05%, unwind-glitch noise, counted into GEN by precedence = the
  conservative direction for engine claims).

## Method (commands verbatim)

```bash
# untraced timed runs (one per arm)
env QUILL_PERF_CHILD_MODE=memory QUILL_PERF_CHILD_ENGINE=quill \
  QUILL_PERF_CHILD_COUNT=200000 QUILL_PERF_CHILD_HEAP=50000000 \
  QUILL_PERF_CHILD_THREADS=1 QUILL_PERF_CHILD_POSITIONS=true QUILL_PERF_SCALE=smoke \
  /usr/bin/time -v taskset -c 8 scratchpad/elfs/perf_matrix_base_iso   # quill
# tantivy: same env with ENGINE=tantivy, NO taskset (its "threads=1" still runs
# a docstore-compressor + segment-updater thread; pinning 4 threads to one core
# would fabricate a different fixture)

# traced runs
... perf record -F 1997 -g --call-graph dwarf,32768 -o quill-200k-dwarf.perf.data \
  -- taskset -c 8 scratchpad/elfs/perf_matrix_base_iso        # 11,447 samples, 352 MB
... perf record -F 1997 -g --call-graph dwarf,32768 -o tantivy-200k-dwarf.perf.data \
  -- scratchpad/elfs/perf_matrix_base_iso                     # 15,258 samples, 457 MB

# classification (p9/classify.awk): perf script -i <data> | gawk -f classify.awk
```

Both sample counts are far above the >=8k floor. Symbolization is
self-consistent by construction: the traced binary runs from its stashed
path, so `perf script` resolves against exactly that ELF; no manual
addr2line was needed (the 5b19f4f2 per-run-mmap-base rule applies only to
manual address arithmetic, which this method avoids).

### Classifier design — three traps that silently corrupt a naive split

1. **Generic-type-parameter pollution (partition).** The harness carrier
   frame `perf_matrix::index_batches_observed::<frankensearch_quill::index::QuillIndex,…>`
   contains the engine crate name in its GENERICS, so a naive
   `frankensearch_quill::` stack-match classifies ~100% of generator samples
   as engine (first attempt: overlap 20.3% of child). Fix: strip all
   `<…>` groups iteratively before namespace matching; match engine/generator
   on raw-symbol PREFIX (`^<?frankensearch_quill::`, `^<?frankensearch_quill_gauntlet::generator::`)
   plus stripped function names (`index_documents`, `generated_batch`,
   `document_at`, …).
2. **Same trap, family level.** `TermInterner` etc. appear in generic params
   of scribe ingest frames; matching families on raw symbols put the
   tokenizer at 0.4% (truth: 20% of ENG). Fix: family markers match only the
   qualified-self type head (`<TYPE as TRAIT>::fn` → TYPE) plus the stripped
   function path.
3. **Whole-stack priority vs innermost-first.** `FrankensearchTokenizer::analyze`
   takes a per-token sink closure — interning happens INSIDE analyze's
   dynamic extent (`analyze` → `add_document_with_values::{closure#6}` →
   `intern_accounted` → `find_in_bucket`), and `add_document_with_values`
   sits in the caller chain of every tokenizer sample. Whole-stack priority
   matching misroutes entire families. Correct semantics: **innermost
   matching structural frame wins** (leaf→root), which also preserves
   subtree accounting: generic work (serde, memmove, malloc, hashing) under
   `canonical_document_preimage` is charged to canonical — exactly what a
   lever would remove.

**Replicate jitter (new method finding, binding on successors):** `perf
script` inline expansion is NONDETERMINISTIC run-to-run over the same
perf.data (external addr2line under load): identical TOTAL cycles, but
family assignment jitters. Class-level partition is stable (ENG 72.62-72.70%
across 3 replicates); families jitter up to 0.75% of ENG. All quill family
numbers below are **means of 3 classifier replicates** with the max-min
spread column. Anyone reproducing family tables from perf script text must
replicate or pin addr2line behavior.

## Run receipts (asymmetry rule: CPU/wall per arm for every timed context)

| arm | wall | user | sys | CPU | **CPU/wall** | peak RSS | pinning | samples (dwarf) |
|---|---|---|---|---|---|---|---|---|
| Quill | 6.22 s | 5.74 s | 0.47 s | 6.21 s | **1.00x** | 505 MB | `taskset -c 8` | 11,447 |
| Tantivy 0.26.1 | 3.33 s | 5.18 s | 0.70 s | 5.88 s | **1.77x** | 501 MB | unpinned (4 active threads at nominal threads=1) | 15,258 |

Reproduces 5b19f4f2 (5.58/5.57/1.00 and 3.23/5.92/1.83) within ~11% wall on
a loaded host with a different-lineage ELF. Wall ratio 1.87x ≈ CPU ratio
1.06x × thread-parallelism 1.77x — the `threads=1` pin asymmetry, again:
**never quote a wall ratio from this fixture without the CPU/wall column.**

## THE SPLIT — whole-child partition (cycles-weighted)

| partition | Quill arm | Tantivy arm | meaning |
|---|---|---|---|
| **ENGINE subtree** | **72.67%** (72.62-72.70) | **62.34%** | stacks through `index_documents`/engine namespaces/commit |
| **GENERATOR subtree** | **20.34%** (20.32-20.37) | **23.80%** | stacks through `generated_batch`/`document_at` |
| loop residue (batch Vec collect/drop, runtime hop) | 1.42% | 2.97% | harness toll, generator-adjacent |
| unclassified (truncated/kernel/startup) | 5.57% | 10.89% | see note |
| **total harness toll (GEN+LOOP)** | **21.8%** | **26.8%** | |

**Validation against 5b19f4f2:** the sibling estimated the generator at
20.02% (quill) / 21.59% (tantivy) by namespace attribution; this card's
independent call-chain method lands at **20.34% / 23.80%** — inside the
20-22% band for the quill arm and confirming the contamination finding
outright. Absolute harness cost: quill 21.8% × 6.21 s = **1.35 s CPU**;
tantivy 26.8% × 5.88 s = **1.58 s CPU** — same-work-within-15% (identical
corpus by construction; residual gap = DVFS on unpinned threads + the
tantivy arm's worse unwind truncation pushing generator fmt fragments into
UNCLASS). UNCLASS caveat: its composition (u32 `Display::fmt`, `pad_integral`,
memcmp fragments) is morphologically generator/interner work with truncated
stacks, so the generator share is a **floor**, not a ceiling.

### Generator-side table (quill arm; % of GEN partition / % of child)

| generator family | % of GEN | % of child | note |
|---|---|---|---|
| core::fmt materialization (pad_integral, `write_fmt`, u32/u64 Display) | 49.98% | 10.17% | `document_at` → `push_ranked_term_bounded` → `write_fmt<String>` |
| generator compute (Zipf sampling, term choice, content assembly) | 28.31% | 5.76% | |
| HashMap<String,String> metadata churn (SipHash on String keys) | 10.03% | 2.04% | the `hashbrown::insert` 1.08% frame from Round-0 lives HERE, not in quill |
| allocator (String/Vec growth + frees) | 9.75% | 1.98% | |
| BTree ops | 1.67% | 0.34% | |
| memmove | 0.13% | 0.03% | matches 5b19f4f2's 0.68%-of-arm scale |
| serde | 0.09% | 0.02% | **the generator is fmt-heavy, serde-FREE** |

**Correction to 5b19f4f2's reading:** the generator's serde share is ~zero
in BOTH arms (tantivy arm GEN serde 0.02% of child). The `serialize_str` /
`write_char` frames in the tantivy arm sit ENGINE-side (447M cycles = 1.9%
of child, under tantivy's own doc/metadata handling), and quill's
`serialize_str` sits under `canonical_document_preimage` (engine). What IS
generator in both arms is the `core::fmt` block (~10% of child) — Round-1's
deflation of *fmt* frames stands; its deflation of *serde* frames
over-corrected.

## THE ENGINE-ONLY FAMILY TABLE (quill arm; mean of 3 replicates)

Innermost-structural subtree families. Every row: `local-5975wx-32c`,
200k-doc seed-pinned corpus (Zipf S11), heap 50 MB, threads=1, positions ON,
batch 250, ELF `9c3cacf0` (pre-P6/P8 baseline), 11,447 samples.

| # | engine family | % of ENG | % of child | spread (3 reps, %ENG) | contents (self-time leaves inside, % of child) |
|---|---|---|---|---|---|
| 1 | **tokenizer subtree** | **20.11%** | 14.62% | 0.09 | `analyze` self **12.12%** (sibling: 12.15% — reproduced), `analyze_admitted` 0.68+0.37, unwind-glitch fragments |
| 2 | **seal/flush encode** | **19.70%** | 14.31% | 0.23 | `append_canonical_term` 1.87, `collect_flush_rows` 1.64, `EncodedPositionList::encode_with_limits` 1.44, `build_term_rows` 1.11, `encode_ordered_term_streams` 1.10, `stable_radix_partition_serial` 0.55, `EncodedStoredMetaSection::encode_accumulator` 0.40, vint/termdict + memmove/alloc children |
| 3 | **canonicalization core** | **16.23%** | **11.80%** | 0.75 | `canonical_document_preimage` self 2.78, `stable_digit_scatter` 2.41, span-flattened poll-body share 1.43, `canonical_metadata` 0.57, serde `serialize_str` 0.42 (rest of its 0.98 counted at seal-meta), plus malloc/memset/realloc/memcmp children |
| 4 | **interner+maphash @ingest** | **12.50%** | 9.08% | 0.30 | `find_in_bucket` 2.34, `hash_one::<&u64>` 2.25, `intern_accounted` 1.43, sip `write` 1.37, `matches` 1.00, `hash_parts` 0.77, memcmp share |
| 5 | **ingest poll body / per-doc admin** | **8.71%** | 6.33% | 0.10 | span-flattened `index_documents_with_replacements` poll self 2.89, closure body 0.90; explicit tracing frames only ~0.4% ENG (StageTimer+Span::record) — NOT a tracing lever |
| 6 | columnar/stored-field append | 5.19% | 3.77% | 0.67 | `add_document_with_values::{closure#6}` 1.38 + `Vec::append_elements`/copy children |
| 7 | **xxh3 identity @ingest** | 3.85% | 2.80% | 0.14 | `hash_long_internal_loop` 1.54, `xxh3_stateful_consume_stripes` 1.40 (content_hash of preimage) |
| 8 | dual-ID resolve/replace | **2.88%** | **2.09%** | 0.10 | `resolve_document_id_in` self 1.69, `delete_document` self 0.69 (+~0.4% inlined id-hash routed to xxh3/canonical) |
| 9 | keeper maintenance (tier policy, concat, snapshot) | 2.87% | 2.08% | 0.50 | `RecoveredSegmentBacking::section` 0.54 etc. |
| 10 | posting re-decode at seal (structural re-parse) | 2.45% | 1.78% | 0.17 | `decode_block_at` 0.46 + vint payload decode |
| 11 | interner @seal (term-row resolve reads) | 2.03% | 1.47% | 0.13 | `field_and_term` 0.62 |
| 12 | seal publish / segment-bytes clone (bd-s1rc1 site) | 1.86% | 1.35% | 0.05 | `publish_pending_segments::{closure#0}` 0.76 |
| 13 | xxh3 identity @seal (segment ids, idmap) | 1.26% | 0.91% | 0.04 | |
| — | btree/other/alloc residue | 0.47% | 0.34% | — | below floor |

(Rows >=0.5% of ENG all shown; columns sum to ~100% of ENG.)

**BlackThrush dual-ID recheck:** the dual-ID/replacement family is **2.88%
of the engine side = 2.09% of the child** (subtree), with self-time leaves
resolve 1.69% + delete 0.69% of child. The "2.18%-children" figure is
confirmed at child scope within noise; it was never generator-inflated.

**Interner residual post-P6/P8 (narrative correction — this ELF is pre-both):**
total interner+maphash on this baseline = 14.53% of ENG (10.55% of child;
P6 card measured the same family at 10.87% self-based — consistent). The
banked P6 identity-hasher swap deletes the `hash_one::<&u64>` + sip-u64
rows (≈5.0% of ENG here; measured +3.3% throughput) and P8's span-inline
arena trims `find_in_bucket`/`matches` further (measured +2.85%,
sub-threshold vs the 3% band, A/A-clean). Post-P6+P8 the expected residual
is ≈ **7.5-8.5% of ENG (~5.5-6% of child)**, dominated by `intern_accounted`
body, arena memcmp, `hash_parts` (must stay), and the 2.03% seal-side reads.

## Cross-arm engine comparison (context for ranking)

- Engine-only CPU: quill 72.67% × 6.21 s = **4.51 s** vs tantivy 62.34% ×
  5.88 s = **3.67 s** → **engine-only CPU ratio 1.23x** — coheres with
  P10's same-binary counter card (1.27x instructions at higher IPC): the
  gap is WORK COUNT, not execution quality.
- Tokenizer parity rule: both arms run the frankensearch tokenizer family.
  Absolute subtree cycles quill 3.31B vs tantivy 2.64B = **1.25x excess**
  (5b19f4f2 self-normalized: 1.36x). Parity work — only the ~3%-of-child
  excess is lever-addressable, not the 14.6% family.
- Incumbent-absent work (what tantivy does not do at all): canonical 16.23 +
  xxh3 5.11 + dual-ID 2.88 ≈ **24% of quill's engine side ≈ 17.6% of the
  child** — the instruction-count-reduction target list, exactly P10's
  strategy frame.

## DECISION NUMBER — canonicalization/identity

| definition | % of ENG | % of child |
|---|---|---|
| canonical core subtree (preimage + scatter + metadata + their serde/alloc/copy children) | **16.23%** | **11.80%** |
| + xxh3 @ingest (content_hash) | **20.08%** | **14.60%** |
| + xxh3 @seal (segment ids/idmap) | 21.34% | 15.51% |

Round-1 claimed "~12-15% family" from generator-polluted self-time; 5b19f4f2
correctly flagged the pollution; this card's generator-NET subtree lands at
**11.8-14.6% of the child anyway** — the self-time table undercounted the
family's serde/alloc/copy children by roughly what the generator pollution
overcounted its fmt frames. The estimate survives correction; the mechanism
of the estimate did not.

**VERDICT: SPEND A LEVER on canonical-encode.** Realistic recovery: the
non-hash, non-irreducible portion — serializer indirection (`serialize_str`
0.98% child), digit emission via core::fmt (`stable_digit_scatter` 2.41%
child is itoa-shaped), preimage assembly alloc/copy children (~2-3% child),
part of `canonical_metadata` — plausible **5-8% of child CPU**, xxh3 (2.8%)
untouchable. CONSTRAINT unchanged from Round-1: content_hash preimage bytes
are registry-pinned (IDMAP 8B/doc, registry 1.0.2) — the lever must prove
**hash-identical bytes**, not rank-identical results (P6's byte-identity
probe pattern is the template).

## RE-RANKED LEVER TABLE (Impact x Confidence / Effort; >=2.0 eligible)

Impact = plausible child-CPU % recovered (midpoint estimate from the
engine-side table). Effort in P6-equivalent pass units. Basis rows cite the
table above.

| rank | lever | basis (%ENG / %child) | Impact | Conf | Effort | **score** | eligible |
|---|---|---|---|---|---|---|---|
| 1 | **canonical-encode fast path** (bytes-direct preimage, itoa-style digit scatter, drop serde_json Serializer indirection; hash-identity proof gate) | 16.23-20.08 / 11.80-14.60 | 6.0 | 0.70 | 2.0 | **2.10** | **YES — the only one** |
| 2 | tokenizer excess vs incumbent (1.25x on parity work; SWAR-history caution: length-dependent washes) | excess ≈3.0 child | 3.0 | 0.50 | 2.0 | 0.75 | no |
| 3 | dual-ID fresh-ingest fast path (skip resolve/delete when id provably absent) | 2.88 / 2.09 | 1.8 | 0.60 | 1.5 | 0.72 | no |
| 4 | ingest poll-body admin thinning (per-doc span/admin work; explicit tracing only ~0.4% ENG) | 8.71 / 6.33 gross, addressable slice unclear | 1.0 | 0.40 | 1.0 | 0.40 | no |
| 5 | interner next lever beyond P6+P8 (residual ~7.5-8.5 ENG post-both, hash_parts must stay) | see residual note | 1.5 | 0.50 | 2.0 | 0.38 | no |
| 6 | seal-path posting re-decode removal (structural: stream already-encoded blocks) | 2.45 / 1.78 + seal share | 2.5 | 0.40 | 3.0 | 0.33 | no |
| — | seal publish byte-clone (bd-s1rc1) | 1.86 / 1.35 | — | — | — | — | IN FLIGHT (P5, banked) |
| — | allocator substitution / codegen tuning | — | — | — | — | — | CLOSED (P3 null, P7 null, P10 counters) |

Selection guidance for the next 3 passes: (1) canonical-encode
implementation + byte-identity gate + A/B (the one eligible lever); (2) if a
second slot exists, the highest-information CHEAP diagnostic is a
span-off/admin probe of family 5 (poll body) to firm its addressable slice;
(3) dual-ID fresh-path and tokenizer-excess are the best sub-threshold
candidates but neither clears 2.0 on current evidence — re-score after (1)
lands, since removing 5-8% of child CPU raises every survivor's percentage
basis.

## Repro

```bash
# verify the ELF before ANY use
sha256sum scratchpad/elfs/perf_matrix_base_iso   # must be 9c3cacf0fa0a...
# runs: see Method above (env verbatim; quill pinned core 8, tantivy unpinned)
# classify: perf script -i <perf.data> | gawk -f scratchpad/p9/classify.awk > split.txt
#   -> run >=3 replicates (perf script inline expansion is nondeterministic)
# tables: python3 scratchpad/p9/tables.py
```

Artifacts (session scratchpad, machine-local), all under `scratchpad/p9/`:
`quill-200k-dwarf.perf.data` (352 MB, 11,447 samples),
`tantivy-200k-dwarf.perf.data` (457 MB, 15,258 samples),
`{quill,tantivy}-timed.{out,time}`, `classify.awk`, `tables.py`,
`quill-split{,-rep2,-rep3}.txt`, `tantivy-split.txt`, `tables.out`.
