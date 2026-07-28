# Quill performance history

`QG-<n>.<machine-class>.latest.json` is the committed pass-over-pass baseline for
one normative Quill performance gate. Initial activation and target attainment
are separate facts: a complete, admissible candidate plus a same-revision
reproduction establishes the first measured baseline whether the target verdict
is PASS or MISS. A MISS baseline records the current implementation; it does not
support a competitive claim. After that initial activation baseline, history
advances only when `quill-perf-ratchet` emits `Allow`; `Block` and `Quarantine`
never overwrite it. Every Allow also writes a dated sibling. History files are
retained—automation does not delete older evidence under the repository's Rule
1.

Full, sealed runs that do not earn promotion live under
`attempts/<date>/QG-<n>/<run-id>/`. These attempts are durable measurement
evidence, but they are not baselines and never replace either a promoted
`*.latest.json` file or an unmeasured bootstrap placeholder.

The committed `*.unmeasured.latest.json` files are explicit bootstrap
placeholders, not performance evidence. They contain no cells, have
`laws_attested=false`, and force PR regression alarms to `Quarantine`. Once a
full candidate/rerun pair passes evidence admission, the gate may be activated
and the first real machine-class baseline committed with its separate PASS/MISS
target verdict. A target MISS remains a MISS in every claim surface even though
the measurement gate is active.

Artifact schema v3 records the SHA-256 self-reported by the executing benchmark
ELF and a deterministic bootstrap 95% confidence interval on every median.
Each paired claim carries an A/A null from the same invocation. The ratchet
uses those median intervals and the required 2x null-floor margin; `cv_pct`
remains provenance and never decides admission.

Evidence pins a manifest-contract SHA-256 that canonicalizes only
administrative `activated` assignments to `false`. Flipping a fully measured
gate to active therefore cannot invalidate the evidence used for that flip;
changing a fixture, target, estimator, or any other manifest byte still changes
the contract hash.
