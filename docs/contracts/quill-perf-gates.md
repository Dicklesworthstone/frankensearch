# Quill Performance Gates — Activation Rules & Standing Laws

**Status:** Normative companion to `quill-perf-gates.toml` (the machine-readable manifests).
**Owning bead:** `bd-quill-e0-contracts-j53p.6`. **Design of record:** plan §14. **Harness owner:** `quill-e8.1` (bench matrix), `quill-e8.2` (ratchet).

## Activation discipline

Every gate ships `activated = false` until ALL of its pins are real: fixture committed (or generator landed for xlarge lanes), oracle config verified byte-identical in the harness (same analyzer semantics, same heap budget, commits inside timed windows for both engines), build profile applied, and the statistical rule wired. Activating a gate = a PR flipping the flag with the evidence linked. **No number from a non-activated gate may be quoted anywhere** (README, docs, commit messages) except marked "provisional, gate inactive".

## The five standing laws (bind every published number)

1. **No benchmark-only semantics.** Durability settings, commits, and result consumption match shipped defaults; no marker-only "commit latency"; no positions-off numbers marketed against positions-on defaults without saying so.
2. **Distributions, not averages.** p50/p95/p99, a deterministic bootstrap
   95% confidence interval on the median, MAD, and cv_pct provenance always;
   extreme quantiles only with sufficient observations. `cv_pct` is never an
   admission gate.
3. **Never hide maintenance.** Merge/compaction/GC time inside the bulk-index window; foreground latency during background work is part of QG-6.
4. **Memory is first-class.** Bytes/doc itemizes postings/positions/dict/blockmax/idmap(+content_hash)/tombstones; RSS probes are per-OS (see toml `defaults.rss_probe`).
5. **One lever per change.** Every optimization lands alone with ≥0.1% frame attribution (local flamegraph lanes — RCH cannot symbolize, bd-e41k), keep-gated by the ratchet, and ledgered (`docs/PERF_LEDGER.md` wins, `docs/NEGATIVE_EVIDENCE.md` rejects with the Ratio convention; pre-flight ledger grep mandatory).

## `.bench-history` layout (decided here)

```
.bench-history/
  QG-<n>.<machine-class>.latest.json     # committed; the ratchet baseline
  QG-<n>.<machine-class>.<date>.json     # retained run window
  QG-<n>.unmeasured.latest.json          # explicit bootstrap quarantine
```
Schema per file: `{schema_version, gate, bench_elf_sha256,
machine_fingerprint, git_rev,
run_window, run_id, corpus_manifest_hash, manifest_sha256, cells: [{fixture,
metric, engine, unit, value, p50, median_ci95_low, median_ci95_high, p95, p99,
mad, cv_pct, runs}],
laws_attested}`. Candidate and rerun share `run_window` but must have distinct
`run_id` values. `bench_elf_sha256` is computed and printed by the executing
benchmark process itself before Criterion emits any output. The ratchet script
(quill-e8.2) refuses a missing/malformed ELF identity, fewer than 10 samples,
an invalid median CI, or comparisons across differing `corpus_manifest_hash`.

The bootstrap files contain no measurements. They make the absence of a real
machine-class baseline visible without fabricating a number and force PR alarms
to `Quarantine`. After a gate is activated, an otherwise-allowable full
candidate/rerun pair may establish its first measured baseline.
`quill-perf-ratchet` evaluates later candidates against that committed baseline
with a directional 5% pass-over-pass threshold. A movement beyond 5% is
`Block` only when the baseline and candidate bootstrap median CIs prove the
directional regression; a median movement whose CIs do not decide it is
`Quarantine`, never a silent keep. MAD and robust z remain diagnostic
provenance only. Promotion also
requires a second artifact from the same git revision, machine fingerprint,
corpus hash, and run window, with a distinct run ID and medians reproducing
within 5%.

## Paired estimator contract

`quill-paired-estimator-v1` replaces positional vectors and fixed alternation
with complete, seeded, balanced-random paired blocks. Every decision sample
carries a stable block/sample ID, arm and within-block order, measurement phase,
monotonic start/end timestamps, versioned operation scope, matched work/byte
denominators, and executable/corpus/worker/profile provenance. Warmup is
kept outside the estimator input and is rejected if it leaks into the decision
set. Missing, duplicate, overlapping, cross-scope, or cross-provenance pairs
fail closed.

The primary effect is the robust center of paired
`log(treatment/control)` values with a deterministic seeded paired-bootstrap
95% confidence interval. Each arm's absolute p50/p95/p99 and throughput remain
first-class. The ratio of arm medians is persisted as a diagnostic only; it is
not interchangeable with the median paired ratio. The estimator checks their
directions and the exact mean-log identity so contradictory summaries are
structurally visible.

The A/A lane executes the identical operation twice under the same randomized
block schedule. Before a live run, each metric/scale fixes its admissible null
center, confidence-width, robust log-dispersion, order-balance, order-effect,
drift, and reproduction tolerance. A null failure persists as
`InvalidNull`/`NoDecision`, with bounded raw samples and deterministic reason
codes, and cannot emit an `Allow` or `Block` claim. `cv_pct` remains provenance
and never gates. Criterion results may coexist only under an explicitly
different operation scope, or when they are derived from the exact same raw
blocks; a separate Criterion measurement stream cannot be reconciled as if it
were the paired evidence.

The current v3 QG writer remains provisional until `bd-uh2f.1` integrates this
estimator into atomic, verified evidence artifacts and the metric-specific
throughput, hierarchical-latency, RSS, and cold-open estimands. Until then its
legacy `paired_ab`/`paired_null` rows are diagnostics, every gate stays
inactive, and no live performance claim is decision-grade.

Every decision JSON names and SHA-256 hashes the manifest, baseline, candidate,
and rerun. `Allow` may update the machine-class `latest.json` and write a dated
sibling; `Block` and `Quarantine` never change history. Older evidence is
retained rather than automatically deleted under repository Rule 1.

CI uses the same binary in two lanes:

- PRs run the QG-2 smoke slice twice and raise a regression alarm against the
  committed Ubuntu baseline.
- The scheduled/workflow-dispatch matrix runs each QG gate in full
  `release-perf`, measures candidate and rerun from one checkout/host/target,
  evaluates promotion, and uploads measurements, decision, and any history
  candidate. Since CI has read-only repository permission, an allowed history
  update remains a reviewable artifact until committed deliberately.

## Topology honesty (QG-3/QG-4)

Update→searchable and visibility claims carry topology labels per the cross-process visibility contract (`bd-quill-duel-visibility-contract`): **in-process** (delta-visible once e5.x lands) vs **fresh-process** (published-generation freshness). QG-3 also records the required initial-index throughput row; omitting that row makes the gate incomplete. G1a (scalar checkpoint) has no delta: QG-4's visibility-lead clause is N/A until bet Q3 lands as a lever — the manifests encode this so nobody quotes a visibility number the architecture doesn't yet earn.

## Cross-references

Gate manifests: `quill-perf-gates.toml`. Oracle pinning: gauntlet version contract (e0.5). Fixture corpora: fsfs golden profiles + xlarge generator (e6.1). Scaling/attribution method: e8.3/e8.4 notes. Flip evidence: QG-10 delta in `quill-e7.6`'s bundle.
