# Reality-Check Verification & Closure — 2026-08-01

**Author:** CreamCoast (claude-code / fable-5), user-directed follow-through on
`docs/evidence/reality-check-20260801.md` (landed 4a6b4e6a).
**Scope:** independent verification of the D1 fix, permanent regression gate for both
blockers, and the D2 resolution decision — every command below was executed against
real binaries built from clean `origin/main` (`cbf95a22`), never simulated.

---

## 1. D1 — `fsfs index` termination (bd-fsfs-index-command-quiescence-v53qo)

**Fix under review:** `cbf95a22` — `config.rs` assigned the pressure-profile
*permission* `allow_background_indexing` to the user *intent* `watch_mode`, silently
promoting every one-shot `fsfs index` on a capable host into a watch run that awaited
a shutdown signal no non-interactive caller sends. The fix clamps
(`intent && permission`) instead of assigning.

**Independent verification** (binary `sha256 ca25b7aa11705eec…`, debug profile,
`--features embedded-models`, models SHA-verified at
`~/.local/share/frankensearch/models`, 3-file/196-byte corpus):

| Leg | Command shape | Result |
|---|---|---|
| Bounded one-shot | `timeout 150 fsfs index <corpus>` | **exit 0** (reality check measured exit 124 pre-fix) |
| Repeat run | same, second invocation | **exit 0** |
| SIGINT mid-run | 120-doc corpus, `kill -INT` at t=4s | **exited code 0 in 17 s**, no leftover process |
| Durability | inspect index root after run 1 | `index_sentinel.json` `generation_complete: true`, `indexed_files: 3`; `vector/index.fsvi`; `lexical/CURRENT`; `lexical/quill-v1/MANIFEST` + FSLX segment |
| Search | `fsfs search "how does retry backoff work" --format json` | `ok: true`, `retry.md` rank 1, `in_both_sources: true`, `degraded: false` |
| Status (flagless) | `fsfs status --format json` | **exit 0**, `ok: true` — the README's `--no-watch-mode` workaround is now unnecessary and has been dropped |

**Verdict: CONFIRMED.** The lifecycle contract holds on the real binary across
success, repeat, and interrupt paths.

## 2. Permanent regression gate (bd-fsfs-executable-quickstart-ci-ve3ul)

`scripts/check_fsfs_executable_quickstart.sh` + CI job `fsfs-executable-quickstart`
(release-blocking via `release-publish.needs`). The gate builds the binary with the
README's documented source-install feature set, indexes a deterministic corpus under
a hard deadline, asserts process quiescence, durable artifacts, ranked hybrid
results, and non-hash embedder attestation.

**Green proof:** full run (own build step) on `cbf95a22` → `Result: PASS`.

**Red proofs (real injection, not mocks):**

| Regression class | Injection | Gate verdict |
|---|---|---|
| `NONTERMINATION` | rebuilt the binary with the pre-fix assignment restored (`watch_mode = allow_background_indexing`) | **FAIL [NONTERMINATION]**, gate exit 1, at the 60 s deadline |
| `MODEL-UNAVAILABLE` | built the model-free default (`cargo build -p frankensearch-fsfs`, `default = []`) | **FAIL [MODEL-UNAVAILABLE]** on exit 78 |

Both injected binaries were then discarded and the fixed build re-verified green.

## 3. D2 — default-build usability (bd-fsfs-default-build-usable-6mtid)

The bead sanctions two resolutions: flip the default, **or** revise the
product/configuration boundary so a plain documented build has an explicit supported
real-semantic path. Three architectural facts force the second:

1. `scripts/check_fsfs_packaging_release_install_contract.sh:126` **hard-requires
   `default = []`** and pins the CI ordering "quality must compile model-free
   defaults before provisioning embedded inputs" — the model-free default is a
   release-gated contract, not an accident.
2. `build.rs` requires SHA-pinned model files at build time when
   `bundled-default-models` is on; defaulting it makes every workspace build
   model-dependent (the bd-cf80 regression that motivated `default = []`).
3. crates.io can never carry the bundled model inputs, so a registry
   `cargo install frankensearch-fsfs` cannot use an embedded default.

The boundary revision is already largely landed: `f51148fd` rewrote the README
developer path to state the default is model-free (with typed fail-closed behavior
for semantic requests) and to document the full build explicitly:
`scripts/rch-ensure-deps.sh --models-only` →
`cargo +nightly install --path crates/frankensearch-fsfs --no-default-features
--features embedded-models`. The quick-start gate now executes that documented path
verbatim in CI, which is the enforcement the reality check found missing. The curl
installer path ships prebuilt embedded binaries (release-build provisions models
before building).

**Verdict: RESOLVED via documented-explicit-feature boundary + executable
enforcement.** Exit 78 with an actionable message remains the correct typed behavior
for a model-free binary asked to do semantic indexing.

## 4. Replay

```bash
scripts/rch-ensure-deps.sh --models-only
scripts/check_fsfs_executable_quickstart.sh        # builds + runs the whole gate
```
