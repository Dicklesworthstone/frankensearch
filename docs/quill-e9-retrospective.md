# Quill E9.3 — Post-Flip Retirement Sweep & Retrospective

Prepared 2026-09-04 by PeachCliff (bd-quill-e9-docs-retirement-3gul.3). The
flip soak condition is satisfied: the facade flip d117ce1f landed 2026-08-04;
this sweep runs a full nightly cycle later, on 2026-09-04. Method: read-only
sweeps over `docs/PERF_LEDGER.md` (8,846 lines), `docs/NEGATIVE_EVIDENCE.md`
(19,123 lines), `docs/contracts/quill-divergence-register.md`, the bead export,
and the gauntlet crate's own certification docs; live `cargo tree` probes via
rch. Every number cites its source; approximate counts say so.

## 1. QG-10 re-check: tantivy is out of the default tree

Live `cargo tree` probes (rch `--job` lane, 2026-09-04): `tantivy` appears in
**zero** of `--no-default-features`, `lexical` (post-flip default = Quill),
`hybrid` (670-line tree, `frankensearch-quill v0.2.2` present), `full`,
`full-fts5`, and the `frankensearch-fsfs` default binary features. The control
lane `--features lexical-tantivy` contains `tantivy v0.26.1`, proving the probe
detects what it excludes. Facade consumers are feature-gated re-exports
(`frankensearch/src/lib.rs`: 3 gates; `index_builder.rs`: 2). No default-path
Tantivy plumbing remains.

## 2. Removal proposal delivered (your disposal pending)

`docs/quill-e9-removal-proposal.md` lists the exact files: the retired
`durability/src/tantivy_wrapper.rs` (1115 lines, no `mod`, crate README
documents it as retained only for this sweep's approved removal), the bd-d7xk1
set (core `metrics.rs`, repro scratch, empty rerank stubs), and the census
corrections — `ord_table` is alive in the oracle lane (the e9.3 bead's original
census was stale on this), and the lexical crate itself is the retained oracle,
not dead code.

## 3. Gauntlet catch census (what the differential suites actually caught)

Full census with file:line citations in the sweep working notes; the register
of record is `docs/contracts/quill-divergence-register.md` (DIV-001..DIV-010,
machine-enforced for DIV-008/009/010 via
`crates/frankensearch-quill-gauntlet/fixtures/divergence-register-v2-live.json`).

Tally of documented catches: **query-lowering/membership 5** (DIV-001, 003,
009, 010 + bd-8a2a8), **score/rank 3** (DIV-006, 007, 008), **error-contract 2**
(DIV-002, 005), **admission semantics 1** (DIV-004), **pruning-conformance 1**
(bd-669hb UNION_HORIZON), **meta/register 1** (bd-iiidv — the gauntlet
falsifying its own register and appending a retraction), **instrument reds 6**
(bd-916qm family), **ratchet/flip blocks 5** (QG-2 reproduction hold, QG-2
MISS at 0.1087x, INFRA/QG HOLD, bd-aei6b provenance refusal, bd-3beo standing
gate). Dispositions: 6 fixed, 4 accepted-as-divergence, 2 oracle bugs with
shipping-path repairs, 2 still open (bd-669hb, bd-8a2a8).

Findings that feed the testing doctrine:

1. **Differential testing catches bugs in BOTH engines.** DIV-009 and DIV-010
   are oracle (Tantivy 0.26.1) defects — boosted negation flipping membership,
   `A AND NOT B` emptying on both engines — found because a second engine
   disagreed loudly. The shipping path was repaired while the oracle stayed
   bit-faithful; the register records the divergence class instead of hiding it.
2. **Machine-witnessed ingestion beats prose records.** DIV-008 was the first
   register entry ingested via `DivergenceRegisterLedger::observation_from_artifact`
   and it immediately falsified DIV-007's hand-written "≥8 leaves" boundary at
   3 clauses. The register's own enforcement table marks DIV-001..007 as
   prose-only records — the sweep's recommendation is to finish typing them.
3. **The metamorphic maintenance laws' catches so far are harness-level**
   (classification corrections, matrix fixes: bd-quill-e6-gauntlet-scale-rm3q.3/.8),
   not live engine bugs. The laws' value is the closed divergence-class set
   that "rejects classes that do not exist yet"; do not overstate their catch
   record.
4. **Negative space is recorded as such** — bd-y8ozo registered an over-cap
   truncation measurement as NOT-A-DIVERGENCE, the gauntlet refusing to log a
   non-catch.

## 4. Perf ledger summary (levers kept/rejected with ratios)

Exact `##`-heading counts in `docs/PERF_LEDGER.md` (~162 dated entries):
KEEP 15, WIN 20 (legacy label), LANDED 15, ~58 token-less (≈53 landed keeps by
body, ≈5 diagnostics — approximate), REJECT 12, HOLD 4, INFRA 1, BLOCKED 5,
INVALID-* 15, EXECUTION-FAILURE 3, MISS 2, CORRECTION 7, SURVEY 2, AUDIT 1,
METHODOLOGY 1; REVERT/WASH/VOID appear only in prose. Comparison classes: 13
explicit declarations — 8 INCUMBENT, 5 SELF-SPEEDUP. **Verdict: the ledger
contains zero competitive keeps.** Every INCUMBENT row is a MISS
(QG-2 0.1087x single-thread indexing), a no-claim SURVEY, an INVALID/HOLD, or
retracted post-hoc (the 4.639→9.132 concurrency claim is VOID for inference;
the QG-1 exclusive-sweep "4.0x–5.3x slower" reject was reclassified diagnostic
by the same-week CORRECTION). Per the ledger's own contract, all ~149
pre-contract rows are SELF-SPEEDUP/maintenance regardless of their original
headings.

Largest kept ratios (all SELF-SPEEDUP, with printed scope caveats): empty-NQC
sketch skip ~17–556x in the affected region (`:626`), `cass_prefix_source`
O(n)→O(1) ~3333x on the prefix-truncation path (`:5299`), TUI palette cache
~300x (`:6164`), phrase-position decode-once 21.85x at 100k docs (`:8537`),
`truncate_to_chars` ASCII fast path ~13.8x (`:5411`). Largest clean rejections:
token-start SWAR mask +20.3% short / +13.1% long (`:6839`), repair-log line
buffer ~1.50x (`:5884`), demand-gated BLOCKMAX +6.32% (`:7156`), deferred
grouped MaxScore grouped8 1.0261 (`:7196`).

## 5. Negative-evidence sweep: coverage was 15/19, now 19/19

Audit of every Quill-program rejected/WASH experiment (19 lever-level items +
14 QG gate rows): 15 levers had NEGATIVE_EVIDENCE entries (5 with prose-only
predicates), all 14 QG rows covered with labeled predicates. Four gaps found
and **closed in this commit** by backfilled entries at the tail of
`docs/NEGATIVE_EVIDENCE.md` (each citing its original source-of-record
verbatim):

- E7.2 debounce retune wash — ledger-only, no predicate anywhere → entry with
  predicate tied to the bd-z2nfa watch-lock redesign.
- E8.5.2 demand-gated BLOCKMAX reject — ledger-only → entry restating the
  ledger's prose predicate.
- E8-H W2.2 postings-accumulation WASH — hypothesis-ledger-only → entry with
  the source's terminal "none (settled)" predicate preserved verbatim.
- E5.6 Range/Glob memoization reject (−1.30%) — beads-only (its author had
  declared the bead comment the ledger-of-record) → entry with the verbatim
  retry predicate.

Standing caveat recorded in the audit: the published QG-2 ratios (0.108698 /
0.349775 / 0.345546) are quarantined as nonpromotable by the 2026-07-30 binding
state correction (bead bd-quill-e8-perf-doctrine-x4e4.12) — cite them only as
audit history. A minor provenance gap is open with the owner of
bd-quill-e1-scribe-bejd.7 (its KEEP entry lacks ELF SHA-256/comparison class;
`docs/PERF_LEDGER.md:8805-8810`).

## 6. Lessons to carry forward

1. A second engine is a bug finder for the first: budget oracle bugs, not just
   Quill bugs, in differential lanes.
2. Type the register: machine-witnessed divergence ingestion caught in one
   try what prose records got wrong (DIV-008 vs DIV-007's boundary).
3. Comparison-class discipline is doing its job — the ledger's honest answer
   is "zero competitive keeps," and the flip's legitimacy rests on conformance
   plus the owner ruling, not on fabricated speed. QG targets remain the E8
   program's open work.
4. Negative results need a home the next agent reads: beads-only ledgers are
   invisible to doc-ledger readers (the E5.6 gap). NEGATIVE_EVIDENCE.md is
   that home; backfill, don't fork.
5. Terminal "settled" sites may carry a predicate of "none" — but say so
   explicitly, as the W2.2 backfill does, so the rule's exception is visible.
