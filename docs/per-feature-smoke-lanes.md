# Per-Feature Smoke Lanes

The minimal feature smoke lanes are owned by `scripts/check_feature_matrix.sh`.
The script validates that every required lane has a compile target, a behavior
test, and a deterministic artifact file before it runs any cargo command. A
lane is rejected unless its exact behavior filter executes one test; Cargo's
otherwise-successful zero-test result is not accepted.

Run every lane locally:

```bash
scripts/check_feature_matrix.sh
```

Run one lane through `rch` with an isolated target directory:

```bash
FRANKENSEARCH_FEATURE_MATRIX_USE_RCH=1 \
CARGO_TARGET_DIR=/tmp/rch_target_frankensearch_${AGENT_NAME:-agent}_feature_matrix \
scripts/check_feature_matrix.sh --lane hybrid
```

Validate the lane contract without compiling:

```bash
scripts/check_feature_matrix.sh --mode validate
```

| Lane | Compile command | Behavior test command | Artifact |
|---|---|---|---|
| `default` | `cargo check -p frankensearch --all-targets` | `cargo test -p frankensearch --lib feature_matrix_smoke::default_lane_behavior -- --exact --nocapture` | `feature-smoke-default.json` |
| `quill` | `cargo check -p frankensearch --lib --no-default-features --features quill` | `cargo test -p frankensearch --lib --no-default-features --features quill feature_matrix_smoke::quill_lane_behavior -- --exact --nocapture` | `feature-smoke-quill.json` |
| `lexical-tantivy` | `cargo check -p frankensearch --lib --no-default-features --features lexical-tantivy` | `cargo test -p frankensearch --lib --no-default-features --features lexical-tantivy feature_matrix_smoke::lexical_tantivy_lane_behavior -- --exact --nocapture` | `feature-smoke-lexical-tantivy.json` |
| `cass-compat` | `cargo check -p frankensearch --lib --no-default-features --features cass-compat` | `cargo test -p frankensearch --lib --no-default-features --features cass-compat feature_matrix_smoke::cass_compat_lane_behavior -- --exact --nocapture` | `feature-smoke-cass-compat.json` |
| `semantic` | `cargo check -p frankensearch --lib --no-default-features --features semantic` | `cargo test -p frankensearch --lib --no-default-features --features semantic feature_matrix_smoke::semantic_lane_behavior -- --exact --nocapture` | `feature-smoke-semantic.json` |
| `hybrid` | `cargo check -p frankensearch --lib --no-default-features --features hybrid` | `cargo test -p frankensearch --lib --no-default-features --features hybrid feature_matrix_smoke::hybrid_lane_behavior -- --exact --nocapture` | `feature-smoke-hybrid.json` |
| `persistent` | `cargo check -p frankensearch --lib --no-default-features --features persistent` | `cargo test -p frankensearch --lib --no-default-features --features persistent feature_matrix_smoke::persistent_lane_behavior -- --exact --nocapture` | `feature-smoke-persistent.json` |
| `durable` | `cargo check -p frankensearch --lib --no-default-features --features durable` | `cargo test -p frankensearch --lib --no-default-features --features durable feature_matrix_smoke::durable_lane_behavior -- --exact --nocapture` | `feature-smoke-durable.json` |
| `full` | `cargo check -p frankensearch --lib --no-default-features --features full` | `cargo test -p frankensearch --lib --no-default-features --features full feature_matrix_smoke::full_lane_behavior -- --exact --nocapture` | `feature-smoke-full.json` |
| `full-fts5` | `cargo check -p frankensearch --lib --no-default-features --features full-fts5` | `cargo test -p frankensearch --lib --no-default-features --features full-fts5 feature_matrix_smoke::full_fts5_lane_behavior -- --exact --nocapture` | `feature-smoke-full-fts5.json` |

## Prospective facade-flip overlay gate

The direct smoke script above proves the tree that is checked out. The QG-10
overlay gate has a narrower release job: prove the exact prospective
Quill-default patch before that patch is merged. It is implemented by
`scripts/check_feature_matrix_overlay.sh` and the reviewed source contract in
`docs/contracts/quill-facade-source-contract-v1.json`.

The gate identity is the ordered pair:

```text
{base_git_sha, canonical_flip_patch_sha256}
```

Neither component is inferred. The base must be a full 40-character commit
equal to the clean trusted checkout's `HEAD`, and the expected 64-character
patch digest must equal the supplied patch bytes. The validator applies those
bytes to a temporary Git index, writes a deterministic synthetic commit, and
checks out a detached overlay worktree. It never applies the patch to the
trusted checkout. Both the artifact directory and the overlay worktree are
retained for audit.

The canonical patch may not change its trusted contract. This prevents a
candidate from changing the expected feature, target, dependency, public API,
schema, or source-order inventories that judge it. Contract review therefore
uses two passes:

1. Run `audit` against the frozen patch. Audit executes the proofs and emits
   `observed-contract-values.json`, but its receipt is always non-admissible.
2. Review those values, commit them to the source contract, regenerate the
   canonical patch against that new base, and run the full `gate`. Gate
   requires byte-for-byte agreement with every reviewed value.

An illustrative audit invocation is:

```bash
base_git_sha="$(git rev-parse HEAD)"
patch_sha256="$(sha256sum /absolute/path/to/canonical-flip.patch | awk '{print $1}')"
FRANKENSEARCH_QG10_USE_RCH=1 \
RCH_REQUIRE_REMOTE=1 \
scripts/check_feature_matrix_overlay.sh \
  --mode audit \
  --base-git-sha "$base_git_sha" \
  --canonical-flip-patch /absolute/path/to/canonical-flip.patch \
  --canonical-flip-patch-sha256 "$patch_sha256" \
  --artifact-dir /tmp/frankensearch-qg10-audit
```

Use a new, empty artifact directory for every run. A full gate uses the same
arguments with `--mode gate`. A single lane can be explored with `--lane`, but
only an all-lane gate can set `claim.release_admissible` to `true`.

The overlay matrix is exhaustive and ordered:

```text
no-default, default, hash, quill, lexical, lexical-bench,
lexical-tantivy, cass-compat, both-backends, semantic, hybrid,
persistent, durable, full, full-fts5, all-features
```

Every lane records and checks all of the following:

- Cargo-resolved facade features, not only requested command-line features;
- the resolved dependency package census, including positive backend
  requirements and forbidden Tantivy/Quill masking checks;
- `cargo check -p frankensearch --all-targets`, so examples, benches, tests,
  binaries, and the library are covered under their declared feature gates;
- facade doctests;
- exactly one runtime behavior probe whose v2 observation names both
  `lexical_backend` and `selected_backend`;
- a compiler-derived rustdoc JSON census of the facade's public API.

The global proof additionally freezes `Cargo.lock`, both relevant manifests,
the Rust toolchain file, every facade feature and Cargo target, the schema
inventory, and the source-workspace publish order. The order is derived by the
trusted package-contract planner owned by `bd-8nqz.6`; QG-10 only asserts that
the graph is topological and Quill precedes its fusion/facade consumers. It
does not make a registry or package-publish claim.

Compilation, doctests, runtime probes, and rustdoc generation run through
strict RCH when `FRANKENSEARCH_QG10_USE_RCH=1`. Cargo metadata and dependency
tree inspection remain local because they do not compile code. Missing remote
execution or a missing returned rustdoc artifact is a gate failure, never a
local-compilation fallback.

The prospective patch must include the reviewed `Cargo.lock`; an ignored or
host-generated lock is rejected. It must also provide every named runtime
probe in the contract. The gate deliberately does not provide fallback probe
names, hand-written backend claims, or a unified-feature substitute for a
missing lane.

Logs are redacted while they are captured and bounded per file. Exceeding the
bound fails the lane instead of truncating a nominally successful proof.
Stable lane and aggregate receipts are paired with SHA-256-addressed copies.
The aggregate receipt binds every lane receipt, inventory artifact, and log by
digest. Verify a retained receipt without compiling:

```bash
scripts/check_feature_matrix_overlay.sh \
  --mode verify \
  --artifact-dir /tmp/frankensearch-qg10-gate
```

The verifier rejects stable/content-addressed drift, missing or modified
inventories, unsafe receipt paths, lane-receipt drift, and log tampering. The
validator's own fast adversarial suite is:

```bash
scripts/check_feature_matrix_overlay.sh --mode self-test
```

It covers missing lanes, unified-feature dependency masking, late Quill source
order, real temporary-index overlay materialization, patch and base drift,
capture-time log limits and path redaction, and artifact/receipt tampering.

The resulting claim is intentionally limited to source-workspace conformance.
It is not a Quill performance claim, a registry publication claim, or the
facade flip itself. Any change to the base commit or canonical patch bytes
requires a fresh receipt.

## CASS compatibility retirement register

`cass-compat` is a foreign-format interop lane, not an incomplete Quill
migration. The external CASS tool owns schema-v8 Tantivy indexes under
`<base>/index/v8/`; `frankensearch-lexical::cass_compat` must continue reading
and writing that format while the integration exists. The facade dependency
chain is explicit:

```text
cass-compat -> lexical-tantivy -> frankensearch-lexical
```

The generic `lexical` feature is deliberately absent from this chain. Native
Tantivy-format consumers import `frankensearch::lexical_tantivy`, so a later
change to the facade-selected lexical backend cannot silently redirect foreign
schema-v8 reads. The default facade feature set is only `hash`, so the
compatibility chain must never enter a default build. CI protects both sides of
the boundary: the dedicated `cass-compat` lane above compile-checks and
executes its exact behavior test, while the all-features facade check prevents
the cfg-gated adapter from silently rotting.

Delete the lane only after coordination with the CASS project confirms one of
these external events:

1. CASS migrates the integration from its schema-v8 Tantivy index to FSLX; or
2. CASS drops the frankensearch index integration.

At that point, remove the facade feature and re-export, the dedicated smoke
lane and behavior test, and the CASS interop half of
`frankensearch-lexical` together. Native Quill reaching feature completeness
is not, by itself, a deletion signal.

CI runs the same script once per lane and uploads the generated artifact files
with deterministic names. Each per-lane artifact includes the lane name,
feature set, compile command, behavior test command, and status. The companion
`feature-smoke-matrix.json` records the complete required lane set for audit
and replay.
