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

The committed `QG-<n>.unmeasured.latest.json` files are explicit historical
bootstrap placeholders, not performance evidence. Their exact bytes remain the
canonical pretty-JSON serialization of the old `quill-perf-artifact-v6`
sentinel, with no trailing newline. They contain no cells, have
`laws_attested=false`, and cannot produce `Allow` when replayed as evidence.
The current v8 loader rejects them as stale-schema inputs. The separately
versioned `QG-<n>.v8.unmeasured.latest.json` files are the authoritative current
sentinels; each is exact canonical pretty JSON with one terminal newline and is
bound to the current normalized manifest hash. Candidate and rerun require two independent typed-producer
finalizations, including their actual logs, exact v3 manifests, and verified
post-exit v6 receipts. Once a current bootstrap and that full pair pass evidence
admission, the gate may be activated and the first real hardware/profile
baseline committed with its separate PASS/MISS target verdict. A target MISS
remains a MISS in every claim surface even though the measurement gate is
active.

Current threshold schema `quill-perf-artifact-v8` records the profile applicability
binding, SHA-256 self-reported by the executing benchmark ELF, auditable
execution topology, exact runtime ISA, configured engine widths, per-cell
observed concurrency where required, and a deterministic bootstrap 95%
confidence interval on every median. Current decision evidence is
`quill-perf-evidence-v7`; it reconstructs the frozen registry-v2 applicability
plan and rejects any measured `NotApplicable` cell. Bounded partial selections
remain durable, non-adjudicable diagnostic evidence; their stored fold may be
`NoDecision` or `MeasuredProvisional`, but they are never ratchet-admissible.
Ratchet admission requires exactly every `Required` plus `Diagnostic` runnable
cell with its exact role. It binds
producer OS, typed producer v4, exact v3 run-log/threshold/pre-binding-evidence
manifest, and v6 completion receipt. QG-1 carries same-invocation
Tantivy/Tantivy and Quill/Quill A/A nulls. The ratchet uses those median
intervals and the required 2x null-floor margin; `cv_pct` remains provenance
and never decides admission. Historical measured v3 threshold artifacts remain
read-only and cannot silently become v8 claims.

The retained `QG-2.trj-zen3-16c.latest.json` and dated 2026-07-28 siblings are
legacy pre-fix diagnostic evidence, not an activated baseline and not a
competitive number. Their old direct-threshold/latest-evidence layout is
read-only; every new promotion uses the atomic pointer layout above.

Evidence pins a manifest-contract SHA-256 that canonicalizes only
administrative `activated` assignments to `false`. Flipping a fully measured
gate to active therefore cannot invalidate the evidence used for that flip;
changing a fixture, target, estimator, or any other manifest byte still changes
the contract hash.
