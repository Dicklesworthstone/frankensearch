# P10 — Same-binary two-arm perf-counter card (local-5975wx-32c, seam-diagnostic)

**Landing provenance:** landed at 928a16ba-successor; measurements are diagnostic/NoClaim per card scope.

Date: 2026-07-30T~02:00Z. Author: SandyGrove (orchestrator-run). Status: BANKED
DRAFT (publication freeze; lands with the batch). This is the local-class half
of the approved P7 routing ("same-binary cross-class counters"); the 5800X/trj/
m4 halves difference against THIS card later. Seam = QG-2 smoke memory child
(200k docs, thread=1, positions on, core 8) — NEVER gate evidence.

ELF: perf_matrix_G (P7 generic build) sha256 ae96a2ac... — SAME binary, both
arms. perf stat, 3 interleaved runs per arm, medians.

| counter | quill | tantivy | q/t |
|---|---:|---:|---|
| cycles | 22,591,884,682 | 20,556,806,457 | 1.10 |
| instructions | 52,554,082,261 | 41,314,442,589 | **1.27** |
| branches | 9,042,880,021 | 7,641,046,139 | 1.18 |
| branch-misses | 91,720,261 | 113,391,492 | 0.81 |
| cache-references | 801,635,610 | 633,131,773 | 1.27 |
| cache-misses | 125,485,638 | 142,261,782 | 0.88 |
| L1-dcache-loads | 15,648,036,107 | 13,243,294,112 | 1.18 |
| L1-dcache-load-misses | 450,338,519 | 293,410,707 | 1.53 |
| l2_accesses_from_dc_misses | 472,572,607 | 326,834,241 | 1.45 |
| stalled-cycles-frontend | 977,371,433 | 1,168,057,321 | 0.84 |

Derived: quill IPC 2.33 / tantivy 2.01; branch-miss 1.01% vs 1.48%;
L1d-miss 2.88% vs 2.22%; LLC-miss-of-refs 15.7% vs 22.5%.

## Reading (the load-bearing paragraph)

On this seam the quill arm retires **1.27x the instructions** of the tantivy
arm for identical input, and converts that to only a 1.10x cycle deficit
because its IPC is HIGHER (2.33 vs 2.01) with fewer branch misses. Quill is
not microarchitecturally inferior here — it is doing MORE WORK, executed
WELL. This coheres the P3 allocator null and the P7 codegen null: neither
allocator choice nor instruction selection can remove instructions that
algorithmically shouldn't execute. The elevated L1d-miss count (1.53x) is
consistent with the interner/copy families the ladder is already draining
(P6 KEEP, P8 sub-threshold real effect, s1rc1 in flight).

Strategy implication: the QG-2 lane is an INSTRUCTION-COUNT-REDUCTION
campaign — remove work Tantivy doesn't do (canonicalization/identity ~13-15%,
dual sealed-ID resolution, copy/clone families, seal validation) rather than
tune execution. "The answer to any MISS is make Quill faster" == make Quill
do less.

Follow-ups routed: (a) same battery on gate-cell-shaped runs per class (the
seam understates the gate gap; the instruction ratio there is the number that
matters); (b) difference this card against trj/m4/5800X captures on the same
binary per the P7 routing.

Repro: env (QG-2 smoke memory child env) perf stat -x, -e cycles,instructions,
branches,branch-misses,cache-references,cache-misses,L1-dcache-loads,
L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,stalled-cycles-frontend,
stalled-cycles-backend -- taskset -c 8 <G-ELF>; raw CSVs in this dir.
