# Quill Performance Gates — Activation Rules & Standing Laws

**Status:** Normative companion to `quill-perf-gates.toml` (the machine-readable manifests).
**Owning bead:** `bd-quill-e0-contracts-j53p.6`. **Design of record:** plan §14. **Harness owner:** `quill-e8.1` (bench matrix), `quill-e8.2` (ratchet).

## Activation discipline

Every gate ships `activated = false` until ALL of its pins are real: fixture committed (or generator landed for xlarge lanes), oracle config verified byte-identical in the harness (same analyzer semantics, same heap budget, commits inside timed windows for both engines), build profile applied, and the statistical rule wired. Activating a gate = a PR flipping the flag with a complete, admissible candidate and same-revision reproduction linked. Activation certifies the measurement, not the target: the target evaluator records PASS or MISS separately, and an active MISS never supports a competitive claim. **No number from a non-activated gate may be quoted anywhere** (README, docs, commit messages) except marked "provisional, gate inactive".

The manifest-contract SHA-256 canonicalizes only administrative `activated`
assignments to `false`. This makes the evidence used for a false→true review
stable across the flip itself; changing a fixture, target, estimator, or any
other manifest byte still invalidates prior evidence.

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
  QG-<n>.<machine-class>.latest.json
      # one atomic pointer to the complete immutable baseline generation
  QG-<n>.<machine-class>.<date>.<run-id>.json
      # immutable threshold object
  QG-<n>.<machine-class>.<date>.<run-id>.evidence.json
      # immutable receipt-bound evidence object
  QG-<n>.unmeasured.latest.json          # explicit bootstrap quarantine
```
Measured latest pointers use
`{schema_version, gate, machine_class, run_id, threshold_file,
threshold_sha256, evidence_file, evidence_sha256}` and resolve both halves of
one generation from the same directory. Threshold schema:
`{schema_version, gate, bench_elf_sha256,
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
to `Quarantine`. A full, evidence-admissible candidate/rerun pair may establish
the first measured baseline and activate the gate with either a PASS or MISS
target verdict. The exemption is deliberately narrow: only the exact
current-schema sentinel for the evaluated gate and manifest may omit baseline
evidence and a baseline runner receipt. Supplying either for that sentinel is
rejected as fabrication; near-sentinels and measured legacy artifacts receive
no exemption. Candidate and rerun still require two independent, verified,
post-exit receipts. That first MISS baseline is a reference point, not a speed
claim.
`quill-perf-ratchet` evaluates later candidates against that committed baseline
with a directional 5% pass-over-pass threshold. A movement beyond 5% is
`Block` only when the baseline and candidate bootstrap median CIs prove the
directional regression; a median movement whose CIs do not decide it is
`Quarantine`, never a silent keep. MAD and robust z remain diagnostic
provenance only. Promotion also
requires a second artifact from the same git revision, registry-derived machine
class and execution identity, corpus hash, and run window, with a distinct run
ID and independently sealed runner receipt. Ordinary measured-baseline
promotions require three distinct run IDs and receipts across baseline,
candidate, and rerun. Exact-bootstrap promotion requires two across candidate
and rerun and never fabricates a third. All measured roles share one exact
NUL-delimited argv identity and one rustc/target/profile/feature context.
Candidate and rerun medians must reproduce within 5%.

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

The v5 QG writer's `paired_ab`, `paired_null` (Tantivy/Tantivy), and
`paired_null_quill` rows are diagnostics only.
Decision-grade output is the `quill-perf-evidence-v3` artifact (`bd-uh2f` /
`bd-uh2f.1`), which the harness emits beside every v5 artifact from the
exact same raw paired blocks.

## Evidence artifacts (`quill-perf-evidence-v3`)

One `<gate>.evidence.json` (plus a derived `<gate>.evidence.md` table) per
gate, sealed with an embedded SHA-256 over its own canonical JSON. Every cell
carries: both engines' absolute distributions from the same paired blocks, the
paired log-ratio effect with a seeded bootstrap CI, same-invocation
Tantivy/Tantivy and Quill/Quill A/A results for QG-1, bounded raw samples that
every summary recomputes from on load, and a same-scope
absolute-versus-paired reconciliation. Run provenance records the benchmark
ELF SHA-256, git revision plus dirty-state hash, `Cargo.lock` hash,
the exact NUL-separated and NUL-terminated argv SHA-256,
rustc/target/profile/features, host identity, host-wide physical cores and
logical threads, producer OS, process-available concurrency, configured engine
thread widths, runtime-detected ISA, effective affinity/cpuset cap, governor and
load, peak RSS with its method (`unsupported` is reported honestly, never a
zero), and corpus/query-set hashes with generator coordinates. QG-1 and QG-8
add per-cell Quill and Tantivy concurrency witnesses: each observation has a
positive count and proves `min == max ==` the normative configured width. A
configuration parameter alone is never described as observed concurrency.

Version 3 additionally persists the producer OS and binds the exact frozen
`quill-machine-classes.json` registry identity and canonicalization contract,
the canonical class derived from strict runner facts, immutable hardware facts,
the explicit execution request and start/end snapshots, every recomputed
hardware/cpuset/snapshot/execution hash, and the SHA-256 plus exact bytes of one
verified sealed runner-completion receipt and its exact artifact manifest. That
manifest hashes the actual run log, canonical v5 threshold artifact, and exact
pre-binding v3 evidence bytes and names their gate, run ID, and run window.
The strict v5 completion receipt also binds the typed finalizer's contract
version, build-time Git revision and dirty posture, build-time `Cargo.lock`
SHA-256, and the SHA-256 independently computed from the finalizer executable
handle held by the running process. It also binds a canonical digest of the
cleared-and-rebuilt Cargo and measurement environments: the typed gate, run
count, warmup count, bootstrap seed, thread budget, fixed fixture scope,
resolved Cargo/rustc/Git paths and executable digests, compiler flags, and
toolchain/cache roots are identity; run IDs and held artifact paths use typed
placeholders so an immediate reproduction can retain the same policy identity.
Unsupported ambient compiler, allocator, loader, Rayon, Cargo, Rustup, and
`QUILL_PERF_*` overrides reject before compilation. Producer revision and lock must equal the
benchmark build, and a fresh candidate plus its immediate rerun must use the
same producer identity and byte-identical benchmark ELF; a historical baseline
may retain its original producer and benchmark executable.
Manifest, registry, receipt, threshold, and evidence parsers reject duplicate
and unknown fields and require their typed canonical encodings. Loading
re-admits the embedded receipt and manifest against the frozen registry;
resealing an outer artifact cannot legitimize a stale, drifted, mixed,
incomplete, or tampered identity. An explicit `unverified` binding remains
durable for diagnosis but is never ratchet-admissible.
The v5 threshold artifact's execution block is only a compatibility projection:
promotion requires it to equal the sealed current-evidence execution block and
to agree with the verified receipt's producer OS, physical/logical topology,
configured thread budget, exact runtime ISA, and effective CPU-set bounds. A
caller-supplied execution block is never an independent identity authority.
The typed producer derives the exact maximum thread width from the selected
gate's complete frozen matrix; the receipt thread budget and
`RAYON_NUM_THREADS` must equal that width, never merely exceed it.

Receipt binding is deliberately two-phase on the currently promotion-capable
Linux lane. The shell launcher performs only bounded argument and
immutable-root shape checks, opens the prebuilt typed finalizer on inherited
descriptor 9, and executes that held descriptor. The
launched Rust producer derives and opens the canonical shared-namespace
host-global lease before Git inspection, hashing, run-directory creation, or
local benchmark compilation. Registered performance hosts must provide one
shared mount namespace and must not unlink or rename the effective-user-owned
lease inode while a campaign is active; receipts retain the platform-family
logical lease identity. The lock descriptor is inherited by compiler and
benchmark children so a killed producer cannot leave an unleased noisy orphan.
Under the lease, the producer rejects redirecting `GIT_*` and process-injection state,
assume-unchanged/skip-worktree entries, dirty source, and noncanonical external
roots; it clears both child environments and repopulates only the bound typed
allowlist. It rejects both Cargo configuration filenames in `CARGO_HOME`, the
repository root, and every ancestor through the filesystem root before
compilation and at every later held-build verification through receipt commit;
the complete absent-path set is part of the controlled-environment preimage.
The exact hashed rustup-selected `rustc` is forced into Cargo's build
environment. It pins output and target directory
descriptors, hashes and validates the launcher-held finalizer against its
embedded build identity, and creates the unique run directory relative to the
held output parent and synchronizes that parent entry. Linux benchmarks address
artifact descendants through the held `/proc/self/fd` directory. A future
macOS producer may use a canonical pinned artifact path because XNU's
`/dev/fd` directory aliases are not traversable, but no current M4 run is
promotion-admissible: the read-only descriptor design cannot execute and
attest the actual loaded image there. Cargo must report
one benchmark executable beneath the pinned target root. The producer opens
that executable without following links, requires a single-link owned
executable file, hashes the held descriptor, and executes through the descriptor
while setting canonical `argv[0]`. After an observed thermal/start capture, the
frozen registry admits the exact hardware/execution/durability envelope before
the log is opened or the benchmark child is spawned. The live benchmark writes
the current v3 artifact with an explicit `unverified` binding and exact
NUL-delimited process-argv hash while the child runs. The producer keeps the
lease and held roots/images across the exact child, log synchronization, end
probes, manifest construction, receipt sealing, terminal registry admission,
and an in-memory bind-and-reverify preview. A nonzero or signaled child writes a
separately sealed `frankensearch.perf-runner-attempt.v1` diagnostic receipt and
can never emit or be parsed as a promotion completion. After every promotion
check passes, the producer writes the manifest and diagnostic-only
`frankensearch.perf-run-precommit.v3` `PRECOMMIT.json`, syncs them, rechecks the
lease, roots, source, held benchmark, and held producer, and writes the
ratchet-required v5 receipt last as the sole finalization commit boundary.
The canonical `environment-policy.json` preimage is written under the held run
directory before the child starts, re-read by descriptor, and hashed directly
by both completion and attempt receipts; `PRECOMMIT.json` inventories that same
digest but is never an admission authority. Promotion receives a measured
baseline as already-bound
committed evidence; only candidate and rerun receive fresh external receipt,
manifest, actual run-log, threshold, and pre-binding evidence inputs. The
ratchet binds those two roles in memory and validates all three measured roles
before opening any history destination. Only receipt-bound bytes may be written
on `Allow`; original unverified producer evidence remains diagnostic provenance
and is never copied into history.

Estimands are metric-specific: flat paired log ratios (QG-1/2/3/4/5/8),
two-stage hierarchical per-query resampling (QG-6, sixteen query groups per
class), process RSS (QG-7), cold open requiring verified cache-state proof
(QG-9 currently persists `NoDecision` because the harness reopens in-process
without dropping the OS page cache), and dependency facts outside timing A/A
(QG-10, diagnostic).

Decision statuses fold deterministically with severity precedence
`Fatal > Block > Quarantine > NoClaim > Allow`. Invalid runs persist durably
as `InvalidNull` or `NoDecision` and are never ratchet-admissible;
`ratchet_admissible()` is the single predicate downstream validators
(`bd-quill-e8-perf-doctrine-x4e4.15.1`) must consult before applying
`Allow`/`Block`/`Quarantine` via `apply_gate_decision`. Persistence is atomic
(temp file, `fsync`, rename, directory sync); loading verifies the hash seal,
re-runs every estimator from the retained raw samples, and rejects truncated
files and stale schema versions. Legacy v3 artifacts load only through the
explicit read-only `load_legacy_gate_artifact_v3`.

Every decision JSON names and SHA-256 hashes the contract manifest, baseline,
candidate, rerun, the already-bound baseline evidence, each candidate/rerun
source and bound evidence form, each fresh receipt and exact artifact manifest,
and each actual candidate/rerun log applicable to promotion. Ordinary promotion
has three measured roles, but the baseline identity is self-contained in its
committed evidence; candidate and rerun carry fresh external finalization
inputs. Exact-bootstrap promotion carries candidate and rerun only; the
sentinel has neither evidence nor identity. Promotion requires one
registry-verified execution identity across every measured role before any
mutable history path is opened. The candidate and immediate rerun additionally
require byte-equal nested producer identities; this parity law deliberately
does not relabel or invalidate an older committed baseline's producer.
`--machine-class` is an expected value only: it must equal the class derived
from every receipt and cannot relabel evidence or select a different latest
key. On `Allow`, the complete decision JSON first records and hashes the
publication plan and reaches stable storage outside history. The ratchet then
uses create-new/idempotent-exact writes for the run-ID-qualified threshold and
evidence objects and atomically advances the one measured latest pointer last.
The pointer binds both immutable hashes, so no crash can publish a mixed
threshold/evidence generation. `Block`, `Quarantine`, receipt rejection, destination
mismatch, and legacy/current mixtures leave every history byte unchanged.
Legacy threshold artifacts remain readable only in explicitly nonpromotable
regression-alarm mode. Older evidence is retained rather than automatically
deleted under repository Rule 1.

CI and registered-host production are deliberately separate lanes:

- PRs run the QG-2 smoke slice twice to exercise the harness, reproduction
  checks, canonical artifacts, and ratchet denial paths. There is no stable
  hosted-Ubuntu performance baseline.
- The scheduled/workflow-dispatch matrix runs each QG gate in full
  `release-perf`, measures candidate and rerun from one checkout/host/target,
  evaluates a structural reproduction diagnostic, and uploads its artifacts.
  Exact ephemeral host identity makes the unmeasured-sentinel result an
  expected `Quarantine`; hosted `ubuntu-latest` is never promotion-eligible and
  this lane is not a functional pass-over-pass performance alarm.
- Promotion runs currently occur only on a registered TRJ class through
  `scripts/perf-runner.sh` and the typed producer. Those finalized candidate and
  rerun bundles are supplied deliberately to `quill-perf-ratchet`; only its
  `Allow` path may write a reviewable history candidate.

Every M4 gate remains promotion-unavailable until a supported `O_EXEC` or
loaded-image mechanism attests the actual executing image. All current Apple
measurements are diagnostic-only. A future M4 contract must also freeze
class-specific scheduler-managed 10- and 14-worker endpoints, bind actual
scheduler-state observations, and retain the durability requirements. Worker
pool width alone never proves P/E residency; inferred P/E attribution is
diagnostic unless the scheduler-state witness proves it. QG-3/QG-4/QG-5 remain
unavailable on every machine class until both benchmark arms emit a
non-declarative witness of the required symmetric durability treatment
(`F_FULLFSYNC` on macOS and the registered equivalent on Linux).

## Topology honesty (QG-3/QG-4)

Update→searchable and visibility claims carry topology labels per the cross-process visibility contract (`bd-quill-duel-visibility-contract`): **in-process** (delta-visible once e5.x lands) vs **fresh-process** (published-generation freshness). QG-3 also records the required initial-index throughput row; omitting that row makes the gate incomplete. G1a (scalar checkpoint) has no delta: QG-4's visibility-lead clause is N/A until bet Q3 lands as a lever — the manifests encode this so nobody quotes a visibility number the architecture doesn't yet earn.

## Cross-references

Gate manifests: `quill-perf-gates.toml`. Oracle pinning: gauntlet version contract (e0.5). Fixture corpora: fsfs golden profiles + xlarge generator (e6.1). Scaling/attribution method: e8.3/e8.4 notes. Flip evidence: QG-10 delta in `quill-e7.6`'s bundle.
