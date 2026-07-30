# E8-H P1 — does the INCUMBENT pay the same copy toll? Tantivy vs Quill, one host, one binary, one corpus (2026-07-29)

**Task:** the Round-1 memmove row (`p1-trj-qg2-round0.md` rank 3) and the local
attribution card (`p1-local-qg2-memmove-attribution-20260729.md`) measured
Quill's memmove family at 7.74% / 7.25% with memset 0.34% and realloc 0.41%,
and concluded "data-copy-shaped, not allocator-shaped". That number was
**meaningless on its own** because the incumbent's equivalent was never
measured. This card measures it.

**Method delta that makes this card stronger than a cross-citation:** both arms
are profiled from the **same ELF, in the same session, on the same host, over
the same corpus**. The earlier Quill numbers came from a *different* binary
(see Provenance), so this card re-measures Quill rather than citing it.

## Machine-class fingerprint (Law 6: not comparable to trj-zen3-* or m4-macos)

| axis | value |
|---|---|
| Host identity | `thinkstation1` — class label `local-5975wx-32c` (dev host; NOT a registered campaign class — diagnostic only, not ratchet-admissible) |
| CPU | AMD Ryzen Threadripper PRO 5975WX, Zen 3, 32c/64t, SMT on |
| Kernel / governor | Linux 6.17.0-35-generic; `amd-pstate-epp`, governor **powersave** (relative self-% only; NOT for absolute-throughput claims) |
| THP | `madvise` |
| Toolchain | rustc 1.99.0-nightly (9f36de775 2026-07-19) |
| perf | 6.17.13, `perf_event_paranoid=1`, `perf_event_max_sample_rate=7000` |

## Provenance (read before citing)

- Source: **clean `git archive origin/main` = `ae8acd031259b130c96869674f863c3d89962908`**,
  exported read-only to session scratchpad; `fast_cmaes` sibling satisfied by
  symlink (`tools/optimize_params` needs `../../../fast_cmaes`).
- **Why not the working tree:** this checkout's `main` is **141 commits behind
  `origin/main`** (and 10 ahead with this lineage's own unpushed work) and dirty
  with peer WIP. Its `perf_matrix.rs` differs from `origin/main` by 1655 lines.
  `perf_matrix.rs` and `generator.rs` are **byte-identical** between
  `origin/main` and `3684b1477` (the source the prior Quill card profiled), so
  the corpus recipe and child seam match that card exactly.
- Bench ELF: `perf_matrix-61bfe5c149a7f626`, SHA-256
  `71e4bd1d084761eec9dde7719a1b433b350f328e609f3a98b309ed7bda1d4a69`, 77,550,936 bytes.
  **Builder identity: LOCAL (`thinkstation1`), not an rch worker.**
  The prior card's ELF was `03308f4b4b71...` — the metadata hash in the filename
  is identical (same package/feature/profile pins) but the **content differs**,
  because that card built `3684b1477` and this one builds `ae8acd03` (same bench
  source, different linked Quill/lexical crates). **This is exactly why both arms
  are re-measured here rather than one being cited.**
- Build: `cargo bench -p frankensearch-quill-gauntlet --features perf-harness
  --profile release-perf --bench perf_matrix --no-run`,
  `RUSTFLAGS="-C force-frame-pointers=yes"`,
  `CARGO_TARGET_DIR=/data/projects/frankensearch/target` (the repo's ONE
  canonical target dir, reused — not minted per task), `Finished release-perf`
  in 7m20s, rc=0, **zero `[RCH] remote` lines** (fully local).
  Build-route note: this was built locally under the Route-2 allowance
  (`POLICY_local_perf_binaries.md` PART B) with a 353G-free precheck against the
  150G floor. `rch exec` has **no artifact-retrieval mechanism**, so the remote
  route cannot yield a locally-executable `release-perf` binary at all.
  `force_local` is NOT set in rch config.

## Method (commands verbatim; identical for both arms except the engine label)

```bash
env QUILL_PERF_CHILD_MODE=memory QUILL_PERF_CHILD_ENGINE=<tantivy|quill> \
  QUILL_PERF_CHILD_COUNT=200000 QUILL_PERF_CHILD_HEAP=50000000 \
  QUILL_PERF_CHILD_THREADS=1 QUILL_PERF_CHILD_POSITIONS=true \
  QUILL_PERF_SCALE=smoke \
  perf record -F 1997 -g --call-graph dwarf,32768 \
  -o <arm>-200k-dwarf.perf.data -- \
  target/release-perf/deps/perf_matrix-61bfe5c149a7f626
```

**Corpus identity is structural, not asserted.** `index_batches` is generic over
`LexicalWrite`; both arms are driven by the same `SyntheticCorpus`
(`synthetic_spec`: fixed `CORPUS_SEED`, `VOCABULARY_SIZE`, `ZipfExponent::S11`,
`MAX_DOCUMENT_BYTES`) at the same document count and the same
`SMOKE_BATCH_DOCUMENTS` batch size. Independent confirmation below: the harness
generator costs 21.59% of the Tantivy arm and 20.02% of the Quill arm — the same
work, as it must be.

Caller attribution: `perf script` leaf-filtered to memmove/memcpy/memset/realloc
leaves, classified engine-vs-harness by callchain markers, plus
`addr2line -e <ELF> -f -C -i` on caller return addresses (addr−1, minus the
per-process ELF bias taken from that run's own `PERF_RECORD_MMAP2` — **the two
runs have different ASLR bases; reusing one bias for the other silently
resolves to `??`**).

## Run receipts (untraced, `/usr/bin/time`; same host, same corpus, 200k docs)

| arm | wall | user | sys | CPU (user+sys) | CPU/wall | peak RSS | index bytes | writer threads pinned | threads OBSERVED with samples |
|---|---|---|---|---|---|---|---|---|---|
| Tantivy 0.26.1 | 3.23 s | 5.17 s | 0.75 s | **5.92 s** | **1.83×** | 501,580 kB | 120,949,716 | 1 | **4** (`thrd-tantivy-in` 47.48%, main 41.23%, `docstore-compre` 9.87%, `segment_updater` 1.41%; 4 idle `merge_thread_*`) |
| Quill | 5.58 s | 5.19 s | 0.38 s | **5.57 s** | **1.00×** | 519,844 kB | 0 (in-memory) | 1 | **1** (main 99.99%) |

Prior local Quill card: wall 5.78 s, user 5.39 s, sys 0.38 s, peak RSS 509 MB.
This run: 5.58 / 5.19 / 0.38 / 508 MB — **reproduces within 3.5%**, so the
anchor holds.

Traced runs: Tantivy **14,645 samples** (438.8 MB), Quill **12,881 samples**
(395.1 MB), event `cycles:P`, F=1997. Prior card: 10,572. Both far above the
≥2000-sample escalation rule; no pooling required.

## THE TABLE — copy/alloc family self-time, both arms, one host, one binary, one corpus

Cycles-weighted self-time, whole process. Every row: host `thinkstation1`
(`local-5975wx-32c`), 200k-doc `SyntheticCorpus` (seed-pinned, Zipf S11),
heap 50 MB, positions ON, `QUILL_PERF_CHILD_THREADS=1`, batch 250, smoke scale.

| family (self-time) | **Tantivy 0.26.1** (4 threads observed, 1 pinned) | **Quill** (1 thread observed, 1 pinned) | Quill ÷ Tantivy |
|---|---|---|---|
| `__memmove_avx_unaligned_erms` (whole process) | **5.79%** (3.36 `thrd-tantivy-in` + 1.80 main + 0.63 `docstore-compre`) | **7.11%** (all main) | 1.23× |
| `__memset_avx2_unaligned_erms` | **0.08%** | **0.22%** | 2.75× |
| `realloc` + `_int_realloc` | **0.30%** | **0.37%** | 1.23× |
| copy family subtotal (memmove+memset+realloc) | **6.17%** | **7.70%** | 1.25× |
| — engine-owned portion of that memmove | 4.96% of arm | 6.38% of arm | 1.29× |
| — harness-generator portion of that memmove | 0.70% of arm | 0.68% of arm | 0.97× (identical, as required) |
| **memmove normalized to each engine's OWN CPU** | **6.33%** (thread-scoped cross-check: 3.99/58.76 = 6.79%) | **7.97%** | **1.26×** |
| `__memcmp_avx2_movbe` (context) | 0.86% | 1.96% | 2.28× |
| glibc allocator self-time excl. realloc (context) | 8.95% | 4.97% | 0.56× |
| harness generator, whole arm (validity check) | 21.59% | 20.02% | 0.93× |

Engine-owned/harness split from dwarf leaf classification: Tantivy 712
memmove+memcpy leaves (328 engine / 102 harness / 282 unclassified-but-engine-side);
Quill 844 leaves (614 / 87 / 143). Quill's 10.3% harness share reproduces the
prior card's 10.9% independently.

### Absolute conversion (untraced CPU basis)

| arm | engine-owned memmove | × CPU | per 200k docs | per doc |
|---|---|---|---|---|
| Tantivy | 4.96% | 5.92 s | 0.294 s | 1.47 µs |
| Quill | 6.38% | 5.57 s | 0.355 s | 1.78 µs |
| **excess** | — | — | **0.061 s** | **0.31 µs** |

**Deleting Quill's ENTIRE memmove excess over the incumbent recovers 0.061 s of
5.57 s CPU = 1.1%.**

## Where each arm's copies actually are (addr2line -f -C -i, same binary)

**Tantivy** — it makes per-document and per-section copies of exactly the same
structural shape as Quill:

| samples | site |
|---|---|
| 53 | `tantivy::directory::ram_directory::InnerDirectory::write` → `copy_to_nonoverlapping<u8>` |
| 28 | `tantivy::store::writer::StoreWriter::send_current_block_to_compressor` → `copy_to_nonoverlapping<u8>` |
| 25 | `tantivy::directory::ram_directory::VecWriter::write` → `copy_from<u8>` |
| 18 | `tantivy::schema::document::default_document::write_bytes_into` → `Vec::append_elements_unreserved<u8>` — **the per-document copy into `CompactDoc`** |
| 6 | `crossbeam_channel::IntoIter<SmallVec<[AddOperation<CompactDoc>;4]>>::next` → `read<..>` — **per-document copy at the writer queue hop** |
| 6 | `DedicatedThreadBlockCompressorImpl::new::{closure#0}` → doc-store block message read |

**Quill** (same binary, reproducing the prior card's site list):

| samples | site |
|---|---|
| 78 | `QuillWriterState::publish_pending_segments::{closure#0}` (= `bd-s1rc1`'s site) |
| 56 | `index::canonical_document_preimage` |
| 50 | `Vec<u8>::append_elements` |
| 46 | `quiver::EncodedStoredMetaSection::encode_accumulator` |
| 18 | `index_documents_with_replacements` poll body (slice cmp) |
| 16 | `BTreeMap<String, SetValZST>::insert` |
| 9 | `quiver::EncodedIdMapSection::encode_with_limits` |

## CONCLUSION (one sentence)

**Tantivy pays the same copy toll — 5.79% memmove / 0.08% memset / 0.30% realloc
against Quill's 7.11% / 0.22% / 0.37%, or 6.3–6.8% versus 8.0% once each is
normalized to its own engine's CPU — so the per-document data copy is NOT where
the gap lives, and eliminating Quill's entire copy excess would return 1.1% of
CPU.**

## Scope correction that must travel with this card

The brief asked which suspect explains "the 8.7x". **This fixture does not
contain an 8.7x gap, so it cannot localize one.** On this shape (200k docs,
in-memory, `threads=1`, positions ON, batch 250) the measured facts are:

- Wall-clock: Quill 5.58 s vs Tantivy 3.23 s = **1.73× slower**.
- Total CPU: Quill **5.57 s** vs Tantivy **5.92 s** — Quill burns **less** CPU.
- CPU/wall: Tantivy **1.83×**, Quill **1.00×**.
- `1.83 × (5.57 / 5.92) = 1.72` ≈ the observed 1.73× wall ratio.

The wall-clock gap on this fixture is **entirely** accounted for by thread
parallelism, not by per-document work. The `QUILL_PERF_CHILD_THREADS=1` pin is
**not symmetric between the arms**: it sets Tantivy's indexing-*worker* count
while Tantivy still runs a dedicated docstore compressor thread and a segment
updater, so the incumbent gets 1.83× CPU parallelism at "threads=1" while Quill
gets 1.00×. The QG-2 8.7x figure (`351f5c6d`) is a different fixture (medium
corpus, on-disk, commit inside the timed window) and this card makes no claim
about it.

## Next ranked suspect

Round-1's lever queue ranked: (1) canonical-encode fast path ~12-15%,
(2) u64-map hasher swap ~4.8%, (3) memmove attribution 8.32%, (4) interner.
Row 3 is now **closed as a REJECT** by the table above. The re-ranking this
card forces:

1. **`threads=1` pin asymmetry — promote to rank 1, and it is a HARNESS
   fairness bug before it is a Quill lever.** Tantivy gets 1.83× CPU
   parallelism at the same nominal pin. Either the pin must equalize observed
   thread counts, or every QG-2-family ratio must publish CPU/wall per arm
   alongside the wall ratio. Until then a "Quill is Nx slower single-thread"
   claim is partly measuring thread count.
2. **Canonicalization family — rank 2, but the Round-1 estimate needs
   deflating.** Round-1 credited Quill with serde_json `serialize_str` 3.10 +
   `core::fmt` share. This card's Tantivy arm shows `serialize_str` **1.45%**,
   `pad_integral` **3.61%**, `fmt::write` **2.41%**, `write_char` **1.96%**
   with **zero Quill in the process** — those frames are substantially
   *harness generator*, not Quill canonicalization. Quill's own
   `canonical_document_preimage` (2.80%) + `stable_digit_scatter` (2.20%) +
   `append_canonical_term` (1.65%) are real and incumbent-absent, but the
   "~12-15% family" figure is inflated by generator cost and must be
   re-derived arm-scoped before a lever is spent on it.
3. **Tokenizer — the largest single Quill-owned frame, and now a measured
   asymmetry rather than a dismissal.** `FrankensearchTokenizer::analyze`
   **12.15%** (15.19% engine-normalized) vs the incumbent's
   `FrankensearchTokenStream::advance` **8.74%** (11.15% engine-normalized) for
   the *same* frankensearch tokenizer family = **1.36×**. Round-1 said
   "proportionally comparable … NOT the differentiator"; on this host/binary it
   is the biggest owned frame and carries a real 1.36× excess.
4. `hash_one::<&u64>` 2.81% + `resolve_document_id_in` 2.45% (u64-map hasher
   swap) — unchanged rank, still a clean proven-family win.
5. glibc allocator self-time is **4.97% for Quill vs 8.95% for Tantivy** — the
   incumbent pays *more* allocator time than we do. Confirms `bd-e8h-w2-postings-
   accumulation-3onsu` (WASH) and the P3 allocator survey (DEAD): allocator
   substitution is not a Quill lever on this class.

## Repro

```bash
# build (local, Route 2; reuses the repo's single target dir)
RUSTFLAGS="-C force-frame-pointers=yes" CARGO_TARGET_DIR=/data/projects/frankensearch/target \
  cargo bench -p frankensearch-quill-gauntlet --features perf-harness \
  --profile release-perf --bench perf_matrix --no-run
# profile: see Method above, once per arm
perf report -i <arm>-200k-dwarf.perf.data --stdio --no-children -g none --percent-limit 0.25
perf report -i <arm>-200k-dwarf.perf.data --stdio --no-children -g none --sort comm   # thread split
# caller sites: perf script | awk leaf-filter -> uniq -c, then
#   addr2line -e <ELF> -f -C -i $((caller_addr - THIS_RUN_mmap_base - 1))
```

Artifacts (session scratchpad, machine-local):
`{tantivy,quill}-200k-dwarf.perf.data`, `{tantivy,quill}-mm-callers.txt` under
`/data/tmp/claude-1000/-data-projects-frankensearch/0d945743-bb96-4fe7-8b2e-12de1b0fc46f/scratchpad/perf/`.
