# E8-H P1 — QG-2 memmove-family attribution, LOCAL machine class (2026-07-29)

**Task:** close the Round-1 open item — "memmove 8.32% — largest single libc
frame, attribution OPEN" (`p1-trj-qg2-round0.md`, Round-1 section) — by
attributing the memmove/memcpy family to specific Quill call sites via an
ARM-SCOPED quill-only child profile. Profiling-only pass: no source edits, no
gauntlet-crate modifications.

## Machine-class fingerprint (Law 6: nothing here is comparable to trj-zen3-* or m4-macos)

| axis | value |
|---|---|
| Class label | `local-5975wx-32c` (this dev host; NOT a registered campaign class — diagnostic evidence only, not ratchet-admissible) |
| CPU | AMD Ryzen Threadripper PRO 5975WX, Zen 3, 32c/64t, SMT on, L3 128 MiB (4 instances), max 4.56 GHz |
| NUMA | single node (NPS1), 220 GB RAM |
| Kernel / governor | Linux 6.17.0-35-generic; `amd-pstate-epp`, governor **powersave** (boosts under load; acceptable for relative self-% attribution, NOT for absolute-throughput claims) |
| Toolchain | rustc 1.99.0-nightly (9f36de775 2026-07-19) |
| perf | 6.17.13, `perf_event_paranoid=1` |

Contrast for Law 6 context: trj = 5995WX 64c/128t (different part, different
class). Every % below is `(QG-2 in-memory bulk shape, 200k-doc child,
local-5975wx-32c)`-scoped.

## Provenance (read before citing)

- **The shared working tree was UNBUILDABLE** at run time: peer WIP in
  `crates/frankensearch-quill` fails with
  `error[E0599]: no associated function or constant named 'boolean_topdocs' found for struct 'ReferenceScorer<'a>'`
  (`index.rs:6759` vs `argus.rs:2305`). This pass therefore built from a
  read-only `git archive origin/main` export in the session scratchpad.
- Source: **origin/main = `3684b1477`** (contains `b8c1465b`, the exact
  source lineage of the Round-1 trj profile; local branch HEAD `8cf57819`
  had diverged and does NOT contain `b8c1465b` — ancestry checked against
  origin/main per campaign-audit protocol). `../fast_cmaes` sibling
  satisfied by symlink to `/data/projects/fast_cmaes`.
- Build: `cargo bench -p frankensearch-quill-gauntlet --features perf-harness
  --profile release-perf --bench perf_matrix --no-run` with
  `RUSTFLAGS="-C force-frame-pointers=yes"`, isolated
  `CARGO_TARGET_DIR=/data/tmp/cargo-target-sandygrove-p1`, `RCH_DISABLE=1`
  via wrapper script (build log contains zero `[RCH] remote` lines — fully
  local).
- Bench ELF: `perf_matrix-61bfe5c149a7f626`, SHA-256
  `03308f4b4b74140cc2a9cbcf926cbe29b9c9118e34512af947e70538d88114e0`.
- Child mode never constructs the Tantivy arm (`run_memory_child`,
  `QUILL_PERF_CHILD_ENGINE=quill` — arm-scoped by construction). The
  parent-path `assert_incumbent_is_genuine_tantivy` (since `098e99d4`) is
  untouched and does not apply to a quill-only child.

## Method (commands verbatim)

Documented Round-1 seam (`QUILL_PERF_CHILD_MODE=memory` in
`crates/frankensearch-quill-gauntlet/benches/perf_matrix.rs`), pinned config
heap 50 MB / threads 1 / positions on / smoke batch size 250:

```bash
env QUILL_PERF_CHILD_MODE=memory QUILL_PERF_CHILD_ENGINE=quill \
  QUILL_PERF_CHILD_COUNT=200000 QUILL_PERF_CHILD_HEAP=50000000 \
  QUILL_PERF_CHILD_THREADS=1 QUILL_PERF_CHILD_POSITIONS=true \
  QUILL_PERF_SCALE=smoke \
  perf record -F 1997 -g --call-graph dwarf,32768 -o quill-200k-dwarf.perf.data -- \
  $CARGO_TARGET_DIR/release-perf/deps/perf_matrix-61bfe5c149a7f626
# and identically with --call-graph fp for the cross-check run
```

- Smoke sanity (COUNT=20000, fp): 1230 samples → escalated to 200k per the
  ≥2000-sample rule.
- **Primary dwarf run: 10,572 samples** (trj Round-1: 10,351 — near-identical
  sampling depth). **fp cross-check run: 10,731 samples.** Event `cycles:P`,
  F=1997.
- Untraced timed run (same env, `/usr/bin/time -v`): wall 5.78 s, user
  5.39 s, sys 0.38 s, peak RSS 509 MB. Conversion basis: **1% of run ≈ 58 ms
  per 200k docs ≈ 0.29 µs/doc.** (All rows below are TIME-derived perf
  samples, not allocation counts, so the count×ns/op rule for count-derived
  rows is satisfied by construction; conversions are direct.)
- Caller attribution: stack collapse of `perf script` filtered to
  memmove/memcpy-family leaves (both runs), plus `addr2line -f -C -i` on the
  callers' return addresses (addr−1, ELF bias from the recorded MMAP2 event)
  for the inline-flattened fp chains. Residual harness presence (corpus
  generator) separated by namespace, as in Round-1.

## Family totals (self-time, cycles-weighted)

| symbol | dwarf | fp |
|---|---|---|
| `__memmove_avx_unaligned_erms` | **7.74%** | 7.25% |
| `__memset_avx2_unaligned_erms` | 0.34% | 0.35% |
| `realloc` + `_int_realloc` (bookkeeping self-time) | — | 0.41% |

Local family ≈ 7.7% of the arm vs trj Round-1's 8.32% — same order, same
profile shape (tokenizer, canonicalization, SipHash, interner, xxh3 families
all within ~1% of the trj card). Sample-count share of memmove-leaf samples:
652/10,572 (dwarf), 684/10,731 (fp).

## Attribution table (dwarf primary; % of run = 7.74% × family share)

Site groups over the 652 dwarf memmove-leaf samples. fp corroboration counts
in brackets. Line numbers carry a return-address/inlining caveat: function
identity is certain, exact line is approximate where noted (†).

| # | call-site (innermost attributable frame chain) | file:line | samples (dwarf) [fp] | % of family | % of run | wall / 200k docs | class |
|---|---|---|---|---|---|---|---|
| 1 | ingest-body columnar+stored-field append — `ColumnarAccumulator::add_document_with_values` → `StoredFieldColumns::append_document` → `Vec::extend_from_slice` (tracing-span-flattened poll body) | `scribe.rs:2382` region† | 116 [~208] | 17.8% | 1.38% | ~80 ms | **data-copy** |
| 2 | seal section assembly — `EncodedSegment::encode_with_limits` → `Vec::append_elements` (`copy_section`/`write_section` payload copies) 64 + `EncodedStoredMetaSection::encode_accumulator` 44 + `EncodedIdMapSection::encode_with_limits` 5 + termdict bitpack 2 | `segment.rs:511/521` impl, `segment.rs:347-393` mechanism; `quiver.rs` STOREDMETA/IDMAP encoders | 115 [~100] | 17.6% | 1.37% | ~79 ms | **data-copy** |
| 3 | segment publication byte clone — `publish_pending_segments::{closure}` → `<[u8]>::to_vec` → `EncodedSegment` bytes clone; keeper `publish_segment`/`reconcile_published_segment` | `keeper.rs:3536` region†; = **bd-s1rc1's site** | 68 [29 + commit-path 66+50+16] | 10.4% | 0.81% | ~47 ms | **data-copy** |
| 4 | canonical preimage assembly — `canonical_document_preimage` → `extend_from_slice` 57 + serde_json `serialize_str` into `Vec<u8>` 15 | `index.rs` preimage path | 72 [~42] | 11.0% | 0.85% | ~49 ms | **data-copy** |
| 5 | seal-path posting **re-decode/re-encode** — `append_canonical_term` → `encode_with_block_max` → `PostingList::parse_with_limits` → `decode_block_at`/`decode_vint_payload` | `scribe.rs` term streams / `quiver.rs` codecs | 47 [~60] | 7.2% | 0.56% | ~33 ms | **data-copy** (structural: scribe seals by decoding already-encoded blocks and re-encoding) |
| 6 | BTree node shifts — `BTreeMap<String, SetValZST>::insert` (a `BTreeSet<String>`) + `find_key_index` chains; plus `BTreeMap<u16,(u64,u32)>::or_insert` under posting parse [fp 34] | alloc btree internals; owner in ingest path | 33 [34] | 5.1% | 0.39% | ~23 ms | data-structure (node memmove) |
| 7 | allocator growth + allocator-internal — realloc family (`_int_realloc` memcpy, `finish_grow`) 20; malloc/dealloc-adjacent fragments 18 | libc / `RawVecInner::finish_grow` (under preimage + serialize_str) | 38 [~30] | 5.8% | 0.45% | ~26 ms | **allocation-growth** (the ONLY growth-driven slice) |
| 8 | interner storage — `TermInterner::intern_accounted` / `find_in_bucket` copies | `scribe.rs` interner | 18 [4] | 2.8% | 0.21% | ~12 ms | data-copy (arena append) |
| 9 | tokenizer `analyze_admitted` copies | `scribe.rs` | 8 [29] | 1.2% | 0.09% | ~5 ms | below 0.1% floor (dwarf) |
| 10 | harness generator (`SyntheticCorpus::document_at`, `generated_batch`, core::fmt/bignum float+int formatting) — **not Quill** | gauntlet `generator.rs` | ~71 [~62] | 10.9% | 0.84% | — | excluded from quill-owned |
| 11 | residue: fragmented chains <8 samples each (String::fmt writes, misc keeper/quiver small sites, `[unknown]`-truncated) | mixed | ~66 | 10.1% | 0.78% | ~45 ms | unattributed |

fp-only corroborating sites resolved by addr2line (same story, extra names):
concat-merge under `apply_tier_policy` (`index.rs:2987` region†, 66 samples —
segment concatenation copies inside `concat_merge_owned`), IDHASH build
`quiver.rs:7652 → document_id_at_ordinal quiver.rs:6439` (29),
`build_term_rows scribe.rs:4324` (32), concat representatives
`keeper.rs:5608/5510` (25), `derive_segment_id` xxh3 input copies
`index.rs:3479` (10).

## Resolution statement

- On the LOCAL class the memmove family is **7.74%** of the quill arm. Of it:
  **77.9% attributed to named Quill sites** (rows 1–8, each ≥0.1% of run
  except row 9), **10.9% is harness-generator** (excluded from lever math),
  **~11.3% residue** (row 11 + row 9) — i.e. ≈0.87% of the run remains
  unattributed, down from 8.32 points wholly unattributed at Round-1.
- The trj 8.32% number itself remains formally unattributed ON TRJ (Law 6
  forbids transferring these shares across classes), but the profile shapes
  match closely at every named family, and this method (arm-scoped child +
  fp build + return-address addr2line) reproduces on trj verbatim.

## Allocator-vs-layout split (machine-class-divergence hypothesis input)

Requested classification given the QG-2 class split (certified 0.1113
x86/Linux vs 0.528 m4-macos attempt):

- The memmove family is **overwhelmingly data-copy-driven**: rows 1–5+8
  (structural byte movement: column append, section assembly, publication
  clone, preimage assembly, seal re-encode) ≈ **5.2% of the run**, vs
  **allocation-growth-driven ≈ 0.45%** (row 7). `Vec`-growth realloc traffic
  is a rounding error here — the postings-accumulation WASH
  (`bd-e8h-w2-postings-accumulation-3onsu`) predicted exactly this.
- Separate glibc allocator SELF-time (`_int_malloc` 2.12 + `_int_free_chunk`
  0.84 + `malloc_consolidate` 0.79 + `unlink_chunk` 0.63 + realloc 0.41)
  ≈ **4.8%** of the arm.
- Consequence: on this class, allocator-substitution levers (glibc-vs-
  libmalloc/jemalloc, THP) are bounded by roughly 4.8% + 0.45% ≈ **5.3% of
  the quill arm** — they cannot explain a 0.11-vs-0.53 cross-class ratio gap
  from the quill side alone. The layout/copy levers (rows 1–5) and the
  compute families (canonicalization ~13-15%, interner ~6-8%, SipHash ~4.3%,
  tokenizer ~11%) are where the arm's time actually is. Any allocator-shaped
  explanation of the m4 split must therefore implicate the INCUMBENT arm's
  behavior on macOS or microarchitectural effects on the compute families —
  not quill's memmove profile. (This card profiles only the quill arm and
  cannot bound the incumbent side.)

## Repro

```bash
# build (wrapper exports RCH_DISABLE=1, isolated CARGO_TARGET_DIR, TMPDIR, frame pointers)
cargo bench -p frankensearch-quill-gauntlet --features perf-harness \
  --profile release-perf --bench perf_matrix --no-run
# profile (see Method above for full env), then:
perf report -i quill-200k-dwarf.perf.data --stdio --no-children -g none --percent-limit 0.1
# caller collapse: perf script | awk leaf-filter (memmove|memcpy) -> sort | uniq -c
# fp-chain call sites: addr2line -e <ELF> -f -C -i <ip - mmap_base + 0x230000 - 1>
```

Artifacts (session scratchpad, machine-local):
`quill-200k-dwarf.perf.data` (325 MB), `quill-200k-fp.perf.data`,
`memmove-chains-{fp,dwarf}.txt`, `memmove-caller-addrs.txt` under
`/data/tmp/claude-1000/-data-projects-frankensearch/6ba18fed-10c5-4855-b287-a7067faf4133/scratchpad/perf/`.
