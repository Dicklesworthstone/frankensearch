# Quill Performance Gates — Activation Rules & Standing Laws

**Status:** Normative companion to `quill-perf-gates.toml` (the machine-readable manifests).
**Owning bead:** `bd-quill-e0-contracts-j53p.6`. **Design of record:** plan §14. **Harness owner:** `quill-e8.1` (bench matrix), `quill-e8.2` (ratchet).

## Activation discipline

Every gate ships `activated = false` until ALL of its pins are real: fixture committed (or generator landed for xlarge lanes), oracle config verified byte-identical in the harness (same analyzer semantics, same heap budget, commits inside timed windows for both engines), build profile applied, and the statistical rule wired. Activating a gate = a PR flipping the flag with a complete, admissible candidate and same-revision reproduction linked. Activation certifies the measurement, not the target: the target evaluator records PASS or MISS separately, and an active MISS never supports a competitive claim. **No number from a non-activated gate may be quoted anywhere** (README, docs, commit messages) except marked "provisional, gate inactive".

The manifest-contract SHA-256 canonicalizes only administrative `activated`
assignments to `false`. This makes the evidence used for a false→true review
stable across the flip itself; changing a fixture, target, estimator, or any
other manifest byte still invalidates prior evidence.

Applicability plan schema
`frankensearch.quill-perf-applicability-plan.v2` binds that independently
normalized manifest SHA-256 into every plan hash. QG-1 additionally declares
`primary_target_cell_width = 8`; every registered profile required for the
default flip must retain ordinary width-8 cells as `Required`. A wider canonical
cell classified `NotApplicable` carries the exact typed hardware/profile,
capacity semantics, execution capacity, required cell width, and admitted
maximum that justify the classification. Missing or mutated facts fail
reconstruction rather than falling back to explanatory text.

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
  QG-<n>.<hardware-class>.<execution-profile>.latest.json
      # v2 atomic pointer to the complete immutable baseline generation
  QG-<n>.<hardware-class>.<execution-profile>.<date>.<run-id>.json
      # immutable threshold object
  QG-<n>.<hardware-class>.<execution-profile>.<date>.<run-id>.evidence.json
      # immutable receipt-bound evidence object
  QG-<n>.unmeasured.latest.json          # explicit bootstrap quarantine
```
Measured latest pointers use schema
`frankensearch.perf-history-pointer.v2` and fields
`{schema_version, gate, profile: {hardware_class_id, execution_profile_id},
run_id, threshold_file,
threshold_sha256, evidence_file, evidence_sha256}` and resolve both halves of
one generation from the same directory. Neither half, its A/A band, nor its
destination may cross the immutable hardware/profile key. Current threshold
schema `quill-perf-artifact-v6` uses
`{schema_version, gate, applicability_plan, bench_elf_sha256,
machine_fingerprint, execution, git_rev,
run_window, run_id, corpus_manifest_hash, manifest_sha256, cells: [{fixture,
metric, engine, unit, value, p50, median_ci95_low, median_ci95_high, p95, p99,
mad, cv_pct, runs}],
laws_attested}`. Candidate and rerun share `run_window` but must have distinct
`run_id` values. `bench_elf_sha256` is computed and printed by the executing
benchmark process itself before Criterion emits any output. The ratchet script
(quill-e8.2) refuses a missing/malformed ELF identity, fewer than 10 samples,
an invalid median CI, or comparisons across differing `corpus_manifest_hash`.

The bootstrap files contain no measurements. They make the absence of a real
hardware/profile baseline visible without fabricating a number and force hosted
replay to fail closed as `Block` or `Quarantine`, never `Allow`. A full,
evidence-admissible candidate/rerun pair may establish
the first measured baseline and activate the gate with either a PASS or MISS
target verdict. The exemption is deliberately narrow: only the exact
`quill-perf-artifact-v6` sentinel for the evaluated gate and manifest may omit
`applicability_plan`, execution, baseline evidence, and a baseline runner
receipt. Supplying evidence or identity for that sentinel is rejected as
fabrication; near-sentinels and measured legacy artifacts receive no exemption.
Candidate and rerun still require two independent, verified, post-exit
`frankensearch.perf-runner-completion.v6` receipts. That first MISS baseline is
a reference point, not a speed claim.
`quill-perf-ratchet` evaluates later candidates against that committed baseline
with a directional 5% pass-over-pass threshold. A movement beyond 5% is
`Block` only when the baseline and candidate bootstrap median CIs prove the
directional regression; a median movement whose CIs do not decide it is
`Quarantine`, never a silent keep. MAD and robust z remain diagnostic
provenance only. Promotion also
requires a second artifact from the same git revision, registry-derived machine
hardware/profile and execution identity, corpus hash, and run window, with a distinct run
ID and independently sealed runner receipt. Ordinary measured-baseline
promotions require three distinct run IDs and receipts across baseline,
candidate, and rerun. Exact-bootstrap promotion requires two across candidate
and rerun and never fabricates a third. All measured roles share one exact
NUL-delimited argv identity and one rustc/target/profile/feature context.
Candidate and rerun medians must reproduce within 5%.

## Frozen hardware/profile applicability

Hardware and execution identity is one closed, immutable
`MachineProfileKey`; requested widths, CPU counts, affinity, SMT, or scheduler
settings never create or relabel a key. Registry schema
`frankensearch.quill-machine-class-registry.v2` currently defines:

| Hardware class | Execution profile | Availability / release role | Frozen capacity and per-gate maximum canonical width |
|---|---|---|---|
| `trj-zen3-5995wx` | `physical-64` | registered; required for default flip | 64 physical cores; QG-1 64, QG-2–6 1, QG-7 8, QG-8 32, QG-9–10 1 |
| `trj-zen3-5995wx` | `smt2-128` | registered; required for default flip | 128 logical threads; QG-1 128, QG-2–6 1, QG-7 8, QG-8 32, QG-9–10 1 |
| `m4-macos` | `scheduler-10` | registered and required, but promotion held on actual loaded-image attestation | 10 scheduler workers; QG-1 8, QG-2–6 1, QG-7 8, QG-8 8, QG-9–10 1 |
| `m5-macos` | `scheduler-14` | unavailable but still required for default flip | Capacity and maxima remain absent until a real fingerprint and scheduler receipt land; an all-N/A substitute is forbidden |
| `x86-vps-ovh` | `x86-diagnostic` | registered; diagnostic-only for every gate | No default-flip capacity or maximum; bounded execution facts come from one exact local diagnostic receipt and never acquire release authority |

The typed producer and evidence loader reconstruct the complete canonical gate
matrix and its profile-specific applicability plan; callers supply neither an
ad hoc capacity nor an ad hoc maximum width. A registry-only diagnostic profile
with either bound absent produces a typed no-claim planning error: `None` never
means unlimited. A future runtime-bound diagnostic plan must first bind a
verified capacity and maximum into its admitted identity. In unchanged matrix
order, a cell above that bound is `NotApplicable` with typed
hardware/profile/capacity and width facts, a runnable cell on a diagnostic
profile or a canonically diagnostic cell is `Diagnostic`, and every other
runnable cell is `Required`. A diagnostic-only plan therefore has no `Required`
cells; the evidence fold must emit
`evidence.gate_without_required_cells`/`NoDecision` and can never produce an
`Allow` claim. The v2 plan hash also binds the normalized manifest SHA-256 and
the gate's declared primary target width. Evidence v4 must contain exactly the
union of `Required` and `Diagnostic` cells with matching roles. Missing runnable
cells, extra cells, role changes, and any measured `NotApplicable` cell fail
closed.

Runtime availability is independent of release requirement. M4 remains held
until the producer can attest the actual loaded image; M5 remains unavailable
and required; x86 VPS evidence remains diagnostic-only. QG-3, QG-4, and QG-5
remain promotion-unavailable on every profile until both arms emit a
non-declarative witness of identical durability treatment. None of those holds
changes the committed matrix, target, or release disposition.

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

The v6 QG writer's `paired_ab`, `paired_null` (Tantivy/Tantivy), and
`paired_null_quill` rows are diagnostics only.
Decision-grade output is the `quill-perf-evidence-v4` artifact (`bd-uh2f` /
`bd-uh2f.1`), which the harness emits beside every v6 artifact from the
exact same raw paired blocks.

## Evidence artifacts (`quill-perf-evidence-v4`)

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

Version 4 additionally binds the exact
`frankensearch.quill-machine-class-registry.v2` registry and the complete
profile-qualified applicability-plan identity. It persists the producer OS,
immutable hardware/profile key, registry-derived capacity semantics, execution
capacity and per-gate maximum width, explicit execution request and start/end
snapshots, every recomputed hardware/cpuset/snapshot/execution hash, and the
SHA-256 plus exact bytes of one verified
`frankensearch.perf-runner-completion.v6` receipt and its exact
`frankensearch.perf-runner-artifact-manifest.v2` manifest. The manifest hashes
the actual run log, canonical v6 threshold artifact, and exact pre-binding v4
evidence bytes and names their profile, gate, run ID, and run window.
The v6 completion receipt also binds producer contract
`frankensearch.quill-local-perf-producer.v4`, build-time Git revision and dirty
posture, build-time `Cargo.lock` SHA-256, and the SHA-256 independently computed
from the finalizer executable handle held by the running process. It also binds
a canonical digest of the cleared-and-rebuilt Cargo and measurement
environments: the typed gate and profile, registry-derived execution capacity
and maximum width, run count, warmup count, bootstrap seed, fixed fixture scope,
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
The v6 threshold artifact's applicability and execution blocks are only
compatibility projections:
promotion requires it to equal the sealed current-evidence execution block and
to agree with the verified receipt and reconstructed plan's hardware/profile
key, capacity semantics, execution capacity, per-gate maximum width, producer
OS, physical/logical topology, exact runtime ISA, and effective CPU-set bounds.
A caller-supplied execution or applicability block is never an independent
identity authority. The typed producer obtains execution capacity and maximum
width only from the frozen registry and complete canonical plan;
`RAYON_NUM_THREADS` equals the profile execution capacity, while measured cells
stop at the plan maximum.

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
the current v4 evidence artifact with an explicit `unverified` binding and exact
NUL-delimited process-argv hash while the child runs. The producer keeps the
lease and held roots/images across the exact child, log synchronization, end
probes, manifest construction, receipt sealing, terminal registry admission,
and an in-memory bind-and-reverify preview. A nonzero or signaled child writes a
separately sealed `frankensearch.perf-runner-attempt.v2` diagnostic receipt and
can never emit or be parsed as a promotion completion. After every promotion
check passes, the producer writes the manifest and diagnostic-only
`frankensearch.perf-run-precommit.v4` `PRECOMMIT.json`, syncs them, rechecks the
lease, roots, source, held benchmark, and held producer, and writes the
ratchet-required v6 receipt last as the sole finalization commit boundary.
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
sentinel has neither evidence nor identity. Promotion takes an exclusive
advisory lock on the canonical history directory before resolving any baseline
and holds that same directory-inode lock through immutable-generation
publication and latest-pointer replacement. Under the lock, a profile with no
measured latest pointer accepts only the canonical
`QG-<n>.unmeasured.latest.json` bootstrap in that promotion directory. Once a
profile-qualified latest pointer exists, the baseline path must canonicalize to
that exact pointer and parse as its measured v2 generation; copied, direct,
stale, and bootstrap-replay baselines are rejected without a history write.
Promotion also requires one registry-verified execution identity across every
measured role before admitting any candidate generation. The candidate and
immediate rerun additionally require byte-equal nested producer identities;
this parity law deliberately does not relabel or invalidate an older committed
baseline's producer.
`--hardware-class` and `--execution-profile` form one expected immutable key:
both must equal the profile derived from every receipt and cannot relabel
evidence or select a different latest key. The launcher and typed producer
reject legacy `--class`, `--thread-budget`, and `--apple-mode`; the ratchet
rejects the former class-only `--machine-class` surface. A registered-host run
therefore starts with
`scripts/perf-runner.sh --gate <QG-N> --hardware-class <hardware> --execution-profile <profile> --run-id <id> --run-window <window>`
plus the platform-required bounded options. On `Allow`, the complete decision
JSON first records and hashes the publication plan and reaches stable storage
outside history. The ratchet then uses create-new/idempotent-exact writes for
the run-ID-qualified threshold and evidence objects and atomically advances the
one profile-qualified measured latest pointer last.
The pointer binds both immutable hashes, so no crash can publish a mixed
threshold/evidence generation. `Block`, `Quarantine`, receipt rejection, destination
mismatch, and legacy/current mixtures leave every history byte unchanged.
Historical v3 threshold artifacts remain audit-readable only through the
library's explicit `load_legacy_gate_artifact_v3` API; the CLI ratchet rejects
them in both modes. Regression-alarm mode may consume a direct current-schema
threshold plus its bound evidence but can never publish it. Older evidence is
retained rather than automatically deleted under repository Rule 1.

CI and registered-host production are deliberately separate lanes:

- Hosted CI executes no performance cells because it has no registered
  hardware/profile identity. It runs the typed contract tests, then replays the
  exact unmeasured sentinel as both baseline and candidate to exercise the
  ratchet denial path. Only fail-closed `Block` or `Quarantine` is accepted;
  `Allow` fails CI. This lane produces no timing, activation, or pass-over-pass
  performance claim.
- Promotion runs currently occur only on registered
  `trj-zen3-5995wx/physical-64` or
  `trj-zen3-5995wx/smt2-128` profiles through `scripts/perf-runner.sh` and the
  typed producer. Those finalized candidate and rerun bundles are supplied
  deliberately to `quill-perf-ratchet`; only its `Allow` path may write a
  reviewable history candidate.

Every M4 gate remains promotion-unavailable until a supported `O_EXEC` or
loaded-image mechanism attests the actual executing image. The frozen M4
profile is `m4-macos/scheduler-10`; its current work remains diagnostic and its
10-worker scheduler capacity does not invent a width-10 canonical cell. Worker
pool width alone never proves P/E residency; inferred P/E attribution is
forbidden. M4 QG-8 additionally remains promotion-unavailable until a reviewed
profile-aware 8-vs-4 target/evaluator replaces the current x86-only 16-vs-4
ratchet requirement. `m5-macos/scheduler-14` remains unavailable but required
for the default flip, with no fabricated fingerprint, capacity, or all-N/A
applicability plan. QG-3/QG-4/QG-5 remain promotion-unavailable on every
profile until both benchmark arms emit a
non-declarative witness of the required symmetric durability treatment
(`F_FULLFSYNC` on macOS and the registered equivalent on Linux).

## Topology honesty (QG-3/QG-4)

Update→searchable and visibility claims carry topology labels per the cross-process visibility contract (`bd-quill-duel-visibility-contract`): **in-process** (delta-visible once e5.x lands) vs **fresh-process** (published-generation freshness). QG-3 also records the required initial-index throughput row; omitting that row makes the gate incomplete. G1a (scalar checkpoint) has no delta: QG-4's visibility-lead clause is N/A until bet Q3 lands as a lever — the manifests encode this so nobody quotes a visibility number the architecture doesn't yet earn.

## Cross-references

Gate manifests: `quill-perf-gates.toml`. Oracle pinning: gauntlet version contract (e0.5). Fixture corpora: fsfs golden profiles + xlarge generator (e6.1). Scaling/attribution method: e8.3/e8.4 notes. Flip evidence: QG-10 delta in `quill-e7.6`'s bundle.
