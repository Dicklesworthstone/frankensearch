# Quill performance history

`QG-<n>.<machine-class>.latest.json` is the committed pass-over-pass baseline for
one normative Quill performance gate. A baseline may advance only when
`quill-perf-ratchet` emits `Allow`; `Block` and `Quarantine` never overwrite it.
Every Allow also writes a dated sibling. History files are retained—automation
does not delete older evidence under the repository's Rule 1.

Full, sealed runs that do not earn promotion live under
`attempts/<date>/QG-<n>/<run-id>/`. These attempts are durable measurement
evidence, but they are not baselines and never replace either a promoted
`*.latest.json` file or an unmeasured bootstrap placeholder.

The committed `*.unmeasured.latest.json` files are explicit bootstrap
placeholders, not performance evidence. They contain no cells, have
`laws_attested=false`, and force PR regression alarms to `Quarantine`. Once a
gate is activated, a full, stable candidate/rerun from distinct passes in one
measurement window may establish the first real machine-class baseline. No QG
number may be cited as kept before that first `Allow`.

Artifact schema v3 records the SHA-256 self-reported by the executing benchmark
ELF and a deterministic bootstrap 95% confidence interval on every median.
Each paired claim carries an A/A null from the same invocation. The ratchet
uses those median intervals and the required 2x null-floor margin; `cv_pct`
remains provenance and never decides admission.
