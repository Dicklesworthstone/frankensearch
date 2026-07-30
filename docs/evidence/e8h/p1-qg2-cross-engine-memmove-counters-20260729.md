# E8-H P1 — QG-2 cross-engine copy-volume diagnosis (2026-07-29)

## Disposition

**VALID-MECHANISM / DIAGNOSTIC ONLY.** On the identical 200,000-document
in-memory child fixture, Quill enters the resolved glibc AVX copy routine
**1.9585x** as often as Tantivy and moves **8.1587x** as many bytes:
**202.329 KiB/document** versus **24.799 KiB/document**. The copy-volume
hypothesis is supported. Allocation was already bounded separately by the
matching Quill profile (`realloc` bookkeeping 0.41% self-time).

This is not a QG result, current-incumbent ratio, or timing claim:

- the profiled ELF was built from source revision `3684b147`, before the
  current integration train;
- the two counter arms ran as separate sequential child invocations, not
  side-by-side in one invocation;
- there was no A/A null control around these profiler runs;
- uprobes materially slow the instrumented process, so their elapsed time is
  unusable;
- the child is a profiling seam, not the complete normative QG-2 matrix.

The taxonomy is therefore `VALID-MECHANISM`, not `VALID-AB`. The counted
mechanism can route the next lever, but cannot activate a gate or publish a
competitive number.

No Quill source changed in this pass.

## Provenance and fairness boundary

| Axis | Receipt |
|---|---|
| Host | `thinkstation1`, local diagnostic class `local-5975wx-32c` |
| CPU | AMD Ryzen Threadripper PRO 5975WX, 32 physical cores / 64 logical threads, one socket, one NUMA node |
| Affinity / cpuset | process affinity `0-63`; `Cpus_allowed_list=0-63`; no narrower cpuset |
| RAM | 215 GiB visible |
| ISA | x86-64; AVX2/FMA/BMI2/AES/VAES/ERMS present; no AVX-512 |
| Kernel | Linux `6.17.0-35-generic` |
| CPU policy | `amd-pstate-epp`, status `active`, governor `powersave`, EPP `balance_performance`, boost enabled |
| Toolchain | `rustc 1.99.0-nightly (9f36de775 2026-07-19)` |
| Profiler | `perf 6.17.13`; `bpftrace 0.23.5` |
| libc | glibc 2.42; `/lib/x86_64-linux-gnu/libc.so.6` SHA-256 `6791cc9bdc08295aafcfae01a7d66d788ee5577cbe94db00ace5f1ee04ef2b09` |
| Source | `3684b147797c5babdad4a5568e993db40ed90da5` |
| ELF | `perf_matrix-61bfe5c149a7f626`, build ID `a3d5a28c3b19e5af2e5b2580ba22fb31d9276c2b` |
| ELF SHA-256 | `03308f4b4b74140cc2a9cbcf926cbe29b9c9118e34512af947e70538d88114e0` |
| Build profile | `release-perf`, `RUSTFLAGS="-C force-frame-pointers=yes"`; existing ELF reused, no build in this pass |

Both engines received the same child inputs:

```text
QUILL_PERF_CHILD_MODE=memory
QUILL_PERF_CHILD_COUNT=200000
QUILL_PERF_CHILD_HEAP=50000000
QUILL_PERF_CHILD_THREADS=1
QUILL_PERF_CHILD_POSITIONS=true
QUILL_PERF_SCALE=smoke
```

Only `QUILL_PERF_CHILD_ENGINE={quill,tantivy}` changed. In
`run_memory_child`, both arms consume the same `corpus_for(count)`, the same
document order and batch path, the same 50 MB requested heap, positions-on
schema, and one terminal `commit`. The child uses
`pinned_quill_config(...)` on Quill and
`TantivyIndex::in_memory_with_benchmark_config(...)` on the pinned Tantivy
0.26.1 arm.

This is fixture parity, not invocation parity. The separate-process boundary
is why the result is mechanism evidence only.

## Matching Tantivy CPU profile

The prior Quill card at commit `6cb219d9` recorded
`__memmove_avx_unaligned_erms` at **7.74%** self-time in the DWARF profile and
**7.25%** in its frame-pointer replicate. This pass profiled the Tantivy arm
with the same ELF, environment, 200,000-document count, 1,997 Hz sampling
frequency, and unwind modes.

```bash
perf record -F 1997 -g --call-graph dwarf,32768 \
  -o tantivy-200k-dwarf.perf.data -- \
  env QUILL_PERF_CHILD_MODE=memory \
      QUILL_PERF_CHILD_ENGINE=tantivy \
      QUILL_PERF_CHILD_COUNT=200000 \
      QUILL_PERF_CHILD_HEAP=50000000 \
      QUILL_PERF_CHILD_THREADS=1 \
      QUILL_PERF_CHILD_POSITIONS=true \
      QUILL_PERF_SCALE=smoke \
      /data/tmp/cargo-target-sandygrove-p1/release-perf/deps/perf_matrix-61bfe5c149a7f626
```

The DWARF run captured 13,117 samples with zero lost samples. The
frame-pointer replicate captured 12,354 samples. Raw top-ten self-time from
the DWARF run:

| Rank | Self-time | Frame | Attribution |
|---:|---:|---|---|
| 1 | 9.06% | `FrankensearchTokenStream::advance` | shared analyzer used by Tantivy arm |
| 2 | 5.81% | `SpecializedPostingsWriter<TfAndPositionRecorder>::subscribe::{closure}` | Tantivy postings construction |
| 3 | 5.78% | `SpecializedPostingsWriter<TfAndPositionRecorder>::subscribe` | Tantivy postings construction |
| 4 | **5.75%** | `__memmove_avx_unaligned_erms` | unified libc copy family |
| 5 | 5.67% | `lz4_flex::compress_internal` | Tantivy stored-field compression |
| 6 | 4.72% | `SyntheticCorpus::document_at` | shared harness generator, not Tantivy |
| 7 | 4.26% | `_int_malloc` | allocator self-time, mixed ownership |
| 8 | 3.83% | `Formatter::pad_integral` | mostly shared corpus formatting |
| 9 | 2.37% | `core::fmt::write` | mostly shared corpus formatting |
| 10 | 2.13% | `String as fmt::Write::write_char` | mostly shared corpus formatting |

The Tantivy copy frame reproduced at **5.29%** in the independent
frame-pointer profile. Thus copy is hot in both engines, but self-time alone
does not distinguish their total copy traffic.

## Exact copy-family counters

`bpftrace -l` exposes both `__memmove_avx_unaligned_erms` and
`__memcpy_avx_unaligned_erms` in this glibc. A two-probe smoke run produced
identical maps for both names because they resolve to the same implementation
entry on this host. The final runs therefore attached one uprobe to that
resolved entry, preventing double-counting:

```bash
sudo -n bpftrace -q \
  -c 'env QUILL_PERF_CHILD_MODE=memory
          QUILL_PERF_CHILD_ENGINE=<quill|tantivy>
          QUILL_PERF_CHILD_COUNT=200000
          QUILL_PERF_CHILD_HEAP=50000000
          QUILL_PERF_CHILD_THREADS=1
          QUILL_PERF_CHILD_POSITIONS=true
          QUILL_PERF_SCALE=smoke
          /data/tmp/cargo-target-sandygrove-p1/release-perf/deps/perf_matrix-61bfe5c149a7f626' \
  -e 'uprobe:/lib/x86_64-linux-gnu/libc.so.6:__memmove_avx_unaligned_erms {
         @copy_calls = count();
         @copy_bytes = sum(arg2);
         @copy_min = min(arg2);
         @copy_max = max(arg2);
       }'
```

At this SysV x86-64 entry, `arg2` is the byte-count argument. The counter
includes every call by the child process that reaches the resolved glibc
implementation, including shared corpus-generation traffic. Because the
input generator and executable are identical, the cross-arm excess is the
useful mechanism signal; no attempt is made to relabel the total as
engine-exclusive bytes.

| Engine | Calls | Bytes moved | Calls/document | Bytes/document | KiB/document | Largest call |
|---|---:|---:|---:|---:|---:|---:|
| Quill | 203,720,623 | 41,437,077,594 | 1,018.603 | 207,185.388 | 202.329 | 11,275,748 B |
| Tantivy 0.26.1 | 104,016,287 | 5,078,894,209 | 520.081 | 25,394.471 | 24.799 | 36,112,937 B |
| Quill / Tantivy | **1.958545x** | **8.158681x** | **1.958545x** | **8.158681x** | **8.158681x** | — |

Quill performs an extra **498.522 copy calls/document** and moves an extra
**181,790.917 bytes/document**.

### Scale check: the excess appears at Quill flush/seal scale

The 20,000-document smoke counts were retained only as a scale diagnostic:

| Engine | Fixture | Calls/document | Bytes/document |
|---|---:|---:|---:|
| Quill | 20,000 docs | 795.868 | 67,790.020 |
| Quill | 200,000 docs | 1,018.603 | 207,185.388 |
| Tantivy | 20,000 docs | 527.266 | 25,748.656 |
| Tantivy | 200,000 docs | 520.081 | 25,394.471 |

From 20,000 to 200,000 documents, Quill's calls/document rise **27.99%** and
bytes/document rise **205.63%**. Tantivy stays essentially flat
(`-1.36%` calls/document, `-1.38%` bytes/document). This is direct evidence
that the Quill excess is not merely a constant shared corpus-generator toll:
large flush, seal, publication, or merge copies amplify it. That shape agrees
with the Quill caller attribution already mapped to stored-field append,
section assembly, publication clone, canonical preimage assembly, and
seal-path posting re-decode/re-encode.

## Q2 deferral audit: only the flush half is implemented

**Explicit verdict: the full Q2 deferral does not happen in the profiled hot
path.** Quill defers radix grouping, posting construction, term sorting, and
durable TERMDICT encoding until flush. It does **not** defer token-to-local-term
resolution. Every admitted token synchronously hashes and probes the shard's
local term interner before Quill can append its columnar triple.

The distinction matters because “term-dictionary lookup” can refer to two
different operations:

- Quill does not probe an on-disk FSLX TERMDICT for each token.
- Quill does synchronously probe its in-memory `TermInterner` dictionary for
  each token.

The exact profiled path at source `3684b147` is:

```text
perf_matrix::run_memory_child
  -> index_batches
  -> LexicalSearch::index_documents
  -> QuillIndex::upsert_documents
  -> QuillWriter::index_documents
  -> ColumnarAccumulator::add_document_with_values
  -> analyze_admitted callback, once per admitted token
       term_id = terms.intern(field_ord, token_bytes)
       column.append_token(term_id, doc_ord, position)
```

`TermInterner::intern_accounted` then performs the work synchronously:

1. hash `(field_ord, term bytes)`;
2. probe `HashMap<u64, Bucket>`;
3. resolve the candidate arena span and compare the term bytes;
4. for a new term, copy the composite key into the byte arena and insert the
   bucket immediately.

Only after accumulation does `flush_accumulator` materialize
`FlushTokenRow { term_id, doc_ord, position }`, radix-scatter rows by
`term_id`, call `TermInterner::sorted_ids`, and build the ordered
POSTINGS/POSITIONS/BLOCKMAX/TERMDICT streams.

This is not a source-lineage ambiguity. At audit time, the profiled revision
`3684b147` and current `origin/main` (`ae8acd03`) have the identical
`crates/frankensearch-quill/src/scribe.rs` Git blob
`0d66bba20f4fd988c4b0f4a13cc757b274c63af2`. The same per-token call is at
`scribe.rs:2342-2345`; the hash/probe/compare path is at
`scribe.rs:1148-1188`; flush-time resolution starts at
`scribe.rs:4014-4033`.

The design document itself contains the unresolved tension. Its Q2 headline
says cache-shaped sequential passes replace per-token hashmap random access,
while §6.1 also specifies a shard-local `ahash` interner and triples keyed by
`local_term_id`. The implementation follows the latter: it emits a compact
triple only after a per-token map lookup has already produced that ID.

The matching Quill DWARF profile exposes the direct self-time:

| Phase | Frame | Self-time |
|---|---|---:|
| per-token local resolution | `TermInterner::find_in_bucket` | 1.79% |
| per-token local resolution | `TermInterner::matches` | 1.34% |
| per-token local resolution | `TermInterner::intern_accounted` | 0.83% |
| per-token local resolution | `TermInterner::hash_parts` | 0.78% |
| **explicit per-token subtotal** |  | **4.74%** |
| flush resolution | `TermInterner::field_and_term` | 1.04% |
| flush sort | `sort_unstable_by` over `TermInterner::sorted_ids` | at least 0.58% |

These are self-time leaves, not an inclusive cost bound: generic hash-table,
comparison, allocation, cache-miss, and copy work may land in other symbols.
They prove that the supposedly deferred lookup is live; they do not prove
that deleting the four named frames alone yields a 5-10x wall-speedup.

**Hypothesis disposition:** “Q2 already removes synchronous per-token
dictionary lookup” is **REJECTED**. A true deferred-resolution experiment is
a structurally distinct follow-up, but it is not yet a measured explanation
of the 8.16x copy counter: the existing caller attribution places most named
copy bytes in stored-field append, section assembly, publication, canonical
preimage, and posting re-encode rather than interner storage. Evaluate it as
one representation change with byte-identical index output, exact
token-stream parity, and a 200,000-document mechanism rerun. Only if the
relevant interner cost and/or copy bytes/document falls should it advance to a
same-invocation QG-2 A/A+A/B. Until that experiment exists, “5-10x from Q2”
remains a hypothesis rather than a measured speedup.

## `powersave` / EPP asymmetry audit

No governor, EPP, affinity, boost, or sysctl setting was changed.
`ref-cycles` is not supported by this Zen 3 PMU, so the read-only audit used
per-process core `cycles / task-clock`, the observed GHz statistic printed by
`perf stat`. Five 200,000-document runs per arm were interleaved in balanced
order `Q T T Q / Q T T Q / Q T`.

```bash
perf stat -x, -e duration_time,task-clock,cycles,instructions -- \
  env <the identical child variables above> \
      QUILL_PERF_CHILD_ENGINE=<quill|tantivy> \
      /data/tmp/cargo-target-sandygrove-p1/release-perf/deps/perf_matrix-61bfe5c149a7f626
```

| Arm | Observed GHz samples | Median GHz | Range | Median CPU-equivalents (`task-clock / wall`) |
|---|---|---:|---:|---:|
| Quill | 4.215984, 4.274650, 4.268922, 4.250573, 4.274585 | **4.268922** | 4.216–4.275 | **0.996** |
| Tantivy | 4.029451, 4.029461, 4.044005, 4.004012, 3.971354 | **4.029451** | 3.971–4.044 | **1.815** |

The ranges do not overlap. Quill receives **5.943% more median observed
frequency** than Tantivy under the current `powersave` /
`balance_performance` policy. The policy is therefore not arm-neutral, but
the skew favors Quill and cannot explain Quill's deficit. If anything, these
local numbers understate the deficit relative to equal frequency.

The CPU-equivalent count also proves that this profiling child is not a
strict one-hardware-thread comparison: Tantivy uses about 1.82 CPUs while
Quill uses about one, despite the common requested value `threads=1`. That
is another reason not to promote the child timing to QG-2 evidence.

## Resolution and retry predicate

The alternative “both engines move similar bytes and Quill's remaining
92% is only diffuse scalar work” is rejected by a counted mechanism:
Quill moves **8.16x** the bytes per document, and its byte cost grows sharply
at the 200,000-document flush/seal scale while Tantivy's stays flat.

The profile's 7.74% self-time remains the direct Amdahl ceiling for removing
only libc execution time. An 8.16x traffic ratio does not by itself promise
an 8.16x wall-speedup: a large win requires the mapped call-site change to
also remove caller work and cache/memory pressure rather than merely replace
one copy primitive.

**Concrete retry predicate:** after the immutable integration train lands,
take one mapped copy site at a time. Require:

1. byte-identical index output (or the repo's stronger canonical identity
   proof);
2. the exact-current ELF SHA-256 self-report;
3. a same-fixture 200,000-document mechanism rerun showing fewer Quill
   copy-family bytes/document, not merely a different profile percentage;
4. only after condition 3, a same-invocation QG-2 run with the pinned Tantivy
   0.26.1 incumbent, both A/A nulls, and bootstrap median-CI gating;
5. for a QG-1 rerun, more than one observed CPU-active Quill ingest worker in
   addition to the reduced copy count.

If a proposed copy-elision does not reduce the counted bytes/document, reject
it as a mechanism miss without timing. Do not resample QG-1/QG-2 merely
because this diagnosis exists, infer a competitive ratio from these
separate profiler invocations, or gate on CV.

## Machine-local artifacts

Raw profiles are intentionally not committed because they total hundreds of
megabytes. They remain under:

```text
/data/tmp/claude-1000/-data-projects-frankensearch/
  6ba18fed-10c5-4855-b287-a7067faf4133/scratchpad/perf/
```

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `quill-200k-dwarf.perf.data` | 341,253,072 | `233180d62b081aee6e08fc46f5a089a5ac0312b1fed831bec5f0e77df457a929` |
| `quill-200k-fp.perf.data` | 1,992,480 | `86992db203a148a5230cd62d302cc43a4ebe9c94e4c3f649b271a37e0c59a312` |
| `foggysquirrel-followup/tantivy-200k-dwarf.perf.data` | 411,327,048 | `24640d5110fd4b048080f261c83f657b1bbe4078fb0b8fcdcd607f15d28f0c57` |
| `foggysquirrel-followup/tantivy-200k-fp.perf.data` | 1,902,456 | `debd439a031cfe95930c77a17a3e6299f8a98a34e3d2002bcb5ee0cb909d054c` |

The ten `foggysquirrel-followup/freq-{q,t}{1..5}.csv` receipts have a
sorted-SHA256-manifest digest of
`b4879e2715fa9e532239bc6d78865e12a16a0fa11b625a2391e05feb3d56910b`.
