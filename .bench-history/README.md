# Quill performance history

For current measured history, `QG-<n>.<machine-class>.latest.json` is one
atomic pointer to an immutable
`QG-<n>.<machine-class>.<date>.<run-id>.json` threshold object and its matching
`.evidence.json` object. The pointer hashes both, so the ratchet never reads a
mixed generation. Initial activation and target attainment are separate facts:
a complete, admissible candidate plus a same-revision reproduction establishes
the first measured baseline whether the target verdict is PASS or MISS. A MISS
baseline records the current implementation; it does not support a competitive
claim. After that initial activation baseline, history advances only when
`quill-perf-ratchet` emits `Allow`; `Block` and `Quarantine` never overwrite it.
History files are retained—automation does not delete older evidence under the
repository's Rule 1.

An Allow decision first persists a complete publication plan outside this
directory. It then creates the run-ID-qualified threshold and evidence objects
with no-clobber, byte-exact retry semantics and atomically replaces the one
latest pointer last. A crash therefore leaves either the old complete pointer
or the new complete pointer authoritative. A colliding immutable filename with
different bytes is a hard error.

Full, sealed runs that do not earn promotion live under
`attempts/<date>/QG-<n>/<run-id>/`. These attempts are durable measurement
evidence, but they are not baselines and never replace either a promoted
`*.latest.json` file or an unmeasured bootstrap placeholder.

The committed `*.unmeasured.latest.json` files are explicit bootstrap
placeholders, not performance evidence. They contain no cells, omit the
measured-only `execution` member, have `laws_attested=false`, and force hosted
structural diagnostics to `Quarantine`. Each file is the exact canonical pretty-JSON
serialization of the v5 sentinel, with no trailing newline. Only that exact
sentinel bound to the evaluated gate and final manifest may omit baseline
evidence and identity; supplying either is rejected as fabrication. Candidate
and rerun still require two independent typed-producer finalizations, including
their actual logs, exact manifests, and verified post-exit receipts. Once that
full pair passes evidence admission, the gate may be activated and the first
real machine-class baseline committed with its separate PASS/MISS target
verdict. A target MISS remains a MISS in every claim surface even though the
measurement gate is active.

Artifact schema v5 records the SHA-256 self-reported by the executing benchmark
ELF, auditable execution topology, exact runtime ISA, configured engine widths,
per-cell observed concurrency where required, and a deterministic bootstrap
95% confidence interval on every median. Current decision evidence is
`quill-perf-evidence-v3` and binds producer OS plus the exact run-log/threshold/
pre-binding-evidence manifest and receipt. QG-1 carries same-invocation
Tantivy/Tantivy and Quill/Quill A/A nulls. The ratchet uses those median
intervals and the required 2x null-floor margin; `cv_pct` remains provenance
and never decides admission. Historical measured v3 threshold artifacts remain
read-only and cannot silently become v5 claims.

The retained `QG-2.trj-zen3-16c.latest.json` and dated 2026-07-28 siblings are
legacy pre-fix diagnostic evidence, not an activated baseline and not a
competitive number. Their old direct-threshold/latest-evidence layout is
read-only; every new promotion uses the atomic pointer layout above.

Evidence pins a manifest-contract SHA-256 that canonicalizes only
administrative `activated` assignments to `false`. Flipping a fully measured
gate to active therefore cannot invalidate the evidence used for that flip;
changing a fixture, target, estimator, or any other manifest byte still changes
the contract hash.
