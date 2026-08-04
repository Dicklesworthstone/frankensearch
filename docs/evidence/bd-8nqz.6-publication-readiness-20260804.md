# bd-8nqz.6 — publication-readiness receipt (2026-08-04)

**Verdict: NOT READY TO PUBLISH.** 9 of 13 publishable crates are blocked on version
reuse before any dependency-ordered publication can begin.

- **Agent:** LavenderElk (claude-code / claude-opus-5), adopted from a stale YellowSparrow claim.
- **Source SHA at measurement:** `bae865c03848abddbfd26aa39130c4092ff2cc5f`.
- **Worktree:** dirty (8 paths) — a shared checkout with peer agents mid-edit. This is why
  `DIRTY_TRACKED_WORKTREE` and `UNTRACKED_PACKAGE_FILES` appear as blockers; they are
  environmental, not release defects. Every finding below was re-derived from causes that
  do not depend on worktree cleanliness.
- **Nothing was published.** No `cargo publish` without `--dry-run`, no tag created or pushed.
  Real publication is owner-gated.

Raw artifacts: [`publish-plan-live-census.json`](bd-8nqz.6-publication-readiness-20260804/publish-plan-live-census.json),
[`dry-run-chain-summary.txt`](bd-8nqz.6-publication-readiness-20260804/dry-run-chain-summary.txt).

---

## A. Topological plan — PASS

`scripts/check_crates_publish_contract.sh --mode audit --scope workspace` resolves 13
publishable packages. `frankensearch-quill` is ordered **before** `frankensearch-fusion`
and before the `frankensearch` facade, which is the ordering AC this bead was filed for.

The two non-published workspace members (`frankensearch-quill-gauntlet`,
`tools/optimize_params`) both carry `publish = false` and are correctly excluded.

Internal version coherence re-derived independently of the gate: all 12 internal
`[workspace.dependencies]` version requirements equal the crates' actual versions.
**0 mismatches.**

## B. Live registry census — 9 crates blocked on version reuse

`--registry-census live`:

| Action | Count | Crates |
|---|---|---|
| `publish` | 4 | quill, tui, fsfs, ops |
| `blocked_version_reuse` | 9 | core, durability, embed, index, lexical, rerank, fusion, storage, frankensearch |

Blockers: `PUBLISHED_VERSION_SOURCE_MISMATCH` ×9, `DEPENDENCY_GIT_VERSION_REQUIRED` ×4,
plus the two environmental ones.

This is the same "nine reused published versions from older source SHAs" YellowSparrow
recorded on 2026-07-28. **It has not moved in 7 days, and it is the binding constraint** —
strictly more blocking than the git-dependency problem this bead's blocker tracks.

### The version reuse is behavioural, not cosmetic

Published `frankensearch-core 0.2.1` (crates.io, 2026-05-20) exposes **zero features**.
The candidate `frankensearch-core 0.2.1` in-tree exposes `bench-internals`. Same version
number, different API surface. Because dependents require `^0.2.1`, cargo resolves the
*published* 0.2.1 during packaging and the build fails outright:

```
package `frankensearch-embed` depends on `frankensearch-core` with feature
`bench-internals` but `frankensearch-core` does not have that feature.
```

So version reuse is not merely a provenance annoyance — it makes four dependent crates
unpackageable today.

## C. Dependency-ordered `cargo publish --dry-run` chain — 3 pass / 10 fail

Run in the plan's topological order, `--locked --dry-run`, one crate at a time.

| # | Crate | Dry-run | Cause of failure |
|---|---|---|---|
| 1 | frankensearch-core | **PASS** | — (but census: `blocked_version_reuse`) |
| 2 | frankensearch-durability | **PASS** | — (but census: `blocked_version_reuse`) |
| 3 | frankensearch-embed | FAIL | registry core 0.2.1 lacks `bench-internals` |
| 4 | frankensearch-index | FAIL | git dep `hnsw_rs` specifies no version |
| 5 | frankensearch-lexical | FAIL | registry core 0.2.1 lacks `bench-internals` |
| 6 | frankensearch-quill | FAIL | registry core 0.2.1 lacks `bench-internals` |
| 7 | frankensearch-rerank | FAIL | git dep `ft-api` specifies no version |
| 8 | frankensearch-fusion | FAIL | `frankensearch-quill` not on crates.io |
| 9 | frankensearch-storage | FAIL | registry core 0.2.1 lacks `bench-internals` |
| 10 | frankensearch | FAIL | `frankensearch-quill` not on crates.io |
| 11 | frankensearch-tui | **PASS** | — (census: `publish`) |
| 12 | frankensearch-fsfs | FAIL | `frankensearch-quill` not on crates.io |
| 13 | frankensearch-ops | FAIL | published `frankensearch-tui` is 0.1.0, needs `^0.2.0` |

Package contents verified for the three that packaged: each produced a complete archive
(`LICENSE`, `README.md`, `Cargo.lock`, `.cargo_vcs_info.json`, sources) and cargo compiled
the *extracted packaged bytes*, not the workspace copy.

### A green dry-run does not mean publishable

`frankensearch-core` and `frankensearch-durability` pass `--dry-run` while the live census
marks both `blocked_version_reuse`. `cargo publish --dry-run` warns about an existing
version but does not fail on it. **Read the census, not the dry-run exit code**, when
deciding whether a crate can ship.

### A per-crate dry-run chain cannot validate a first-time topological publish

Each dry-run resolves against the registry *as it exists now*, not against the
hypothetical post-publish registry. Rows 8, 10, 12, 13 fail purely because their
prerequisites are unpublished, and rows 3/5/6/9 because the prerequisite is published at
the *wrong source*. Those failures are expected and would clear as the chain proceeds —
they are **not** independent defects. Only rows 4 and 7 (git dependencies) are defects
that no publication order can fix. This is the structural reason the bead demands a clean
external consumer rather than a dry-run chain.

## D. Packaged-consumer smoke — BLOCKED, and not by the obvious cause

Not achievable today, and it fails earlier than expected. `cargo vendor --locked` — the
only route to a registry-shaped consumer without publishing — **fails outright**:

```
found duplicate version of package `hnsw_rs v0.3.4` vendored from two sources:
  source 1: registry `crates-io`
  source 2: https://github.com/Dicklesworthstone/hnswlib-rs?rev=18a5a1a9...
```

`Cargo.lock` carries **two** `hnsw_rs v0.3.4` entries — the crates.io crate and the git
fork — and both resolve into `frankensearch-index`. This is *deliberate and correctly
gated*: `ann = ["dep:hnsw_rs"]` takes the fork, while `hnsw-patch-ab` pulls the registry
crate under the alias `hnsw_rs_034` as the bd-u3wt A/B baseline, never enabled by a
shipping aggregate. It is not a shipping defect.

It is, however, a hard blocker for two things this bead requires:
1. **Any vendored/clean-room consumer**, because optional deps still occupy the lock — so
   vendoring fails regardless of which features the consumer selects.
2. **The `ann_distribution_exit = native_in_tree` branch**, whose own addendum requires
   that "hnsw_rs and every alias/baseline package" be absent from every `Cargo.lock`
   package entry. The `hnsw_rs_034` alias is present today.

The remaining routes to a clean consumer are all closed: registry resolution needs the 9
version bumps plus a first-ever `frankensearch-quill` publication; path and `[patch]`
overrides are explicitly disallowed as proof by this bead.

## E. Version / tag alignment — PASS

- Facade `frankensearch` version: **0.3.2** → CI expects tag **`crates-v0.3.2`**
  (`RELEASE_TAG_MISMATCH` / `RELEASE_TAG_REQUIRED` enforce this in gate mode).
- Existing `crates-v*` tags: **none**. The crate-bundle tag namespace is unused and clean,
  consistent with "no crates published from this workspace".
- Existing `v*` tags (`v1.4.3` …) track the `frankensearch-fsfs` CLI product and are a
  separate namespace, as designed.

## F. Defects found in the CI `publish-crates` lane

The lane's **"Prepare path dependencies"** step is substantially dead code:

- It rewrites `/data/projects/` inside every `Cargo.toml`. **Zero manifests contain that
  string** — the `sed` is a no-op.
- It clones `asupersync`, which is now a plain registry dependency
  (`version = ">=0.3.10, <0.4"`), and `frankensqlite`, which is not referenced by any
  manifest. Both clones are unnecessary.
- Only `fast_cmaes` is still genuinely required: `tools/optimize_params` holds
  `path = "../../../fast_cmaes"`, escaping the repo, so `--locked` resolution needs the
  sibling checkout. (`crates/frankensearch-quill-gauntlet/fuzz` similarly path-escapes to
  `../../../../asupersync`.) Neither crate is in the publish plan.

Left unchanged — editing the release lane is out of scope for a receipt-only pass and
belongs in a reviewed slice.

## G. Unblock sequence

1. **Bump the 9 reused versions** (core, durability, embed, index, lexical, rerank,
   fusion, storage, facade). This is the binding constraint and is owner-gated. Note that
   bumping `frankensearch-core` is what clears the four `bench-internals` failures.
2. Resolve the ANN distribution XOR (bd-kcek native in-tree, or bd-mczj authorized fork).
   For the `native_in_tree` branch this must also remove the `hnsw_rs_034` alias from the
   lock, which is what currently makes the workspace unvendorable.
3. Resolve the frankentorch registry graph (bd-8nqz-6-ft-registry-4hca — currently blocked
   on owner authorization to publish six crates).
4. Only then: publish bottom-up in the plan order, waiting for index availability between
   layers, and run the clean registry-only consumer across the required feature lanes.

Steps 1–3 all require owner authorization. No further Linux-side receipt can advance this
bead until at least step 1 lands.
