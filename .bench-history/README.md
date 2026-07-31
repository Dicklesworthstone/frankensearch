# Quill performance history

For current measured history,
`QG-<n>.<hardware-class>.<execution-profile>.latest.json` is one canonical
`frankensearch.perf-history-pointer.v2` pointer to an immutable
`QG-<n>.<hardware-class>.<execution-profile>.<date>.<run-id>.json` threshold
object and its matching `.evidence.json` object. Its `profile` object carries
both `hardware_class_id` and `execution_profile_id`, and the pointer hashes both
artifacts, so the ratchet never reads a mixed generation or crosses an
immutable profile key. Initial activation and target attainment are separate
facts: a complete, admissible candidate plus a same-revision reproduction
establishes the first measured baseline whether the target verdict is PASS or
MISS. A MISS baseline records the current implementation; it does not support
a competitive claim. After that initial activation baseline, history advances
only when `quill-perf-ratchet` emits `Allow`; `Block` and `Quarantine` never
overwrite it. History files are retained—automation does not delete older
evidence under the repository's Rule 1.

A promotion Allow first persists a complete publication plan outside this
directory. Before resolving a promotion baseline, `quill-perf-ratchet`
acquires an exclusive lock on this canonical history directory and holds it
through the entire publication. With no profile-qualified latest pointer, the
only valid promotion baseline is that gate's exact canonical unmeasured
sentinel inside this directory; once a latest pointer exists, only that exact
canonical pointer is authoritative for promotion. Copied, directly supplied,
stale, and bootstrap-replay baselines fail promotion before a decision or
history write. Under the held lock, a promotion Allow creates the
run-ID-qualified threshold and evidence objects with no-clobber, byte-exact
retry semantics and atomically replaces the one latest pointer last. A crash
therefore leaves either the old complete pointer or the new complete pointer
authoritative. A colliding immutable filename with different bytes is a hard
error.

Full, sealed runs that do not earn promotion live under
`attempts/<date>/QG-<n>/<run-id>/`. These attempts are durable measurement
evidence, but they are not baselines and never replace either a promoted
`*.latest.json` file or an unmeasured bootstrap placeholder.

The committed `*.unmeasured.latest.json` files are explicit bootstrap
placeholders, not performance evidence. The current bootstrap is the exact
canonical pretty-JSON serialization of the `quill-perf-artifact-v6` sentinel,
with no trailing newline. It contains no cells, has `laws_attested=false`, and
cannot produce `Allow` when replayed as evidence. Only that exact
sentinel bound to the evaluated gate and final manifest may omit
`applicability_plan`, execution, baseline evidence, and identity; supplying
evidence or identity is rejected as fabrication. Candidate and rerun still
require two independent typed-producer finalizations, including their actual
logs, exact v2 manifests, and verified post-exit v6 receipts. Once that full
pair passes evidence admission, the gate may be activated and the first real
hardware/profile baseline committed with its separate PASS/MISS target verdict.
A target MISS remains a MISS in every claim surface even though the measurement
gate is active.

Threshold schema `quill-perf-artifact-v6` records the profile applicability
binding, SHA-256 self-reported by the executing benchmark ELF, auditable
execution topology, exact runtime ISA, configured engine widths, per-cell
observed concurrency where required, and a deterministic bootstrap 95%
confidence interval on every median. Current decision evidence is
`quill-perf-evidence-v4`; it reconstructs the frozen registry-v2 applicability
plan, requires exactly every `Required` plus `Diagnostic` runnable cell with
its exact role, and rejects any measured `NotApplicable` cell. It binds
producer OS, typed producer v4, exact v2 run-log/threshold/pre-binding-evidence
manifest, and v6 completion receipt. QG-1 carries same-invocation
Tantivy/Tantivy and Quill/Quill A/A nulls. The ratchet uses those median
intervals and the required 2x null-floor margin; `cv_pct` remains provenance
and never decides admission. Historical measured v3 threshold artifacts remain
read-only and cannot silently become v6 claims.

The retained `QG-2.trj-zen3-16c.latest.json` and dated 2026-07-28 siblings are
legacy pre-fix diagnostic evidence, not an activated baseline and not a
competitive number. Their old direct-threshold/latest-evidence layout is
read-only; every new promotion uses the atomic pointer layout above.

Evidence pins a manifest-contract SHA-256 that canonicalizes only
administrative `activated` assignments to `false`. Flipping a fully measured
gate to active therefore cannot invalidate the evidence used for that flip;
changing a fixture, target, estimator, or any other manifest byte still changes
the contract hash.
