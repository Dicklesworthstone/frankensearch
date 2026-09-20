# Quill publication validation reuse: staged, not integrated

This directory preserves the complete implementation candidate from the owner's
conversation attachment. It is NOT compiled into Quill and does not activate the
new publication method. No production source, existing test, dependency, format,
release gate, or publication policy is changed by retaining these artifacts.

## Exact candidate identity

The two source-bearing pieces are byte-identical to the attached candidate:

- `keeper.patch`: integration edits to
  `crates/frankensearch-quill/src/keeper.rs`; Git blob
  `ce6c2906bfc9b70054b9224297bbe0523bcab6a2` (8,528 bytes).
- `publication_reuse.rs`: complete proposed child module, including six new
  regression tests; Git blob `43faa88fd395083a243f11362d13e6c51dc60b79`
  (18,138 bytes).
- Required original Keeper blob:
  `19b4b3b6ae02001eabfa40e3221ffc76d3ad270f`, also observed at parent main
  `6f2ad6096da6304c999c131c50770891e8f048ae`.

The original combined attachment's SHA-256 is
`8ec238f34c35623f236cf246a9b2ac57c7de6e834429203d806457d05d4aa16b`.
The integration-only patch's SHA-256 is
`0b2d1b37bfd3c713441c4b299e7fe1b8cd4d378779a018d849a5fb510223bd25`.

## What the candidate attempts

An opt-in raw-image budget retains independently validated immutable preflight
copies. After publication, the normal fresh MANIFEST/recovery-claim read and
same-descriptor file authentication still run. Only exact byte equality permits
reusing the original owned allocation and its pointer-bound metadata. Missing,
changed, corrupt, or truncated selected files cannot be served from cached
bytes. Ordinary publication uses a zero budget. The candidate is not wired into
QuillConfig or CASS's ordinary commit path, and no performance improvement is
claimed. The image budget is not an RSS limit; pinned readers can retain several
budgets. Existing publication, reconciliation, sidecar, deletion-intent, and
cancellation checks must remain intact during integration and qualification.

## Incomplete integration and validation

Direct GitHub writes succeeded, but this session did not safely assemble and
upload a complete patched replacement for the 1,050,198-byte Keeper file. The
preparation-only workflow added in `6181cc4b` produced no workflow run in the
observed GitHub responses. Its exact bytes are archived here with a `.disabled`
extension, outside `.github/workflows`; there is no active preparation workflow.
It never produced a modified Keeper blob or a Rust test result.

The earlier patch-application check used only a labelled excerpt fixture, not a
complete checkout. Cargo is absent from the execution environment. Compilation,
Rust tests, rustfmt, Clippy, benchmark results, and full-checkout application are
all UNVERIFIED. These artifacts are not evidence that issue #51 or any release
gate has been cleared.

## Remaining integration on a complete checkout

First verify the exact original Keeper blob and that the target child module
does not already exist. Reconcile concurrent changes rather than overwriting them.
The following commands change a working tree; they do not commit or push it:

```sh
candidate=docs/evidence/quill-publication-reuse-candidate
parent=crates/frankensearch-quill/src/keeper.rs
child=crates/frankensearch-quill/src/keeper/publication_reuse.rs

test "$(git hash-object "$parent")" = 19b4b3b6ae02001eabfa40e3221ffc76d3ad270f
test ! -e "$child"
git apply --check "$candidate/keeper.patch"
git apply "$candidate/keeper.patch"
mkdir -p "$(dirname "$child")"
cp "$candidate/publication_reuse.rs" "$child"
```

Before a production integration is considered complete, inspect all affected
exhaustive matches, compile and execute the six regressions and the existing
publication/reconciliation/durability tests, and run the repository's normal
quality gates through its supported execution lane. Do not weaken assertions or
interpret a zero-test executable as a passing test suite. Keep the eventual
source-integration commit separate from this candidate-preservation commit.
