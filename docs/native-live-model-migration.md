# Upgrade models in a warm native search session

`frankensearch-native serve` can install a **sealed replacement-model cohort**
without restarting its live selector. This connects the native library's model
migration API to the executable's existing JSONL request loop. It is separate
from ordinary `activate`, which continues to require the original models and
quality-tier topology.

## Scope and authority

Start serving with `--allow-model-migration` only when the stdin controller is
trusted to choose local models and original selection receipts. Neither
`--allow-activation` nor `--allow-updates` implies this grant. The new grant does
not enable source updates, ordinary activation, downloads, generation deletion,
or writes to index artifacts or selection receipts. It cannot be combined with
`--lexical-only`.

The control currently supports **single-layout** CLI-built F32 cohorts, using
exact retrieval or the CLI's native HNSW parameters and seed. It explicitly
refuses sharded migration before reading the requested receipt or loading models;
there is no flattening or partial-inventory fallback. The ready frame reports
`model_migration_enabled` and `model_migration_supported` separately.

Model loading and selected reopening run on the command's owning lane between
requests. Replacement native models retain the existing caller-owned blocking
pool. No request worker or runtime is spawned. This is not a promise of concurrent
request processing or zero migration latency. Query filters, query timeouts, and
the configured reranker remain unchanged. Query timeouts do not apply to this
control operation, and `timeout_ms` is refused in its input.

## Prepare a replacement

Use the existing `rebuild` command with the old trusted receipt and the desired
replacement models. Keep the candidate and new receipt outside the predecessor;
all required model features/assets must already be available locally.

For example, for an explicitly fast-only, exact replacement:

```sh
frankensearch-native rebuild \
  --receipt /srv/search/old.json \
  --index-dir /srv/search/new-index \
  --new-receipt /srv/search/new.json \
  --model-dir /srv/models/new \
  --fast-only --exact
```

This authenticates the retained sources, re-embeds every document, and builds and
seals a complete new source/vector/Quill cohort. It does not select that cohort
for the warm process. The old process can continue serving its existing handle.
A separate publisher remains responsible for any durable restart selection.

For a quality-equipped target, omit `--fast-only` and supply the normal quality
backend/model-directory options. The control must request the same local models
and the same exact-versus-HNSW policy used by that rebuild. Quality addition or
removal is explicit in the selected cohort; a failed quality load never drops
that tier as a fallback.

## Activate through the existing session

Start the old selection with the extra startup grant:

```sh
frankensearch-native serve \
  --receipt /srv/search/old.json \
  --model-dir /srv/models/old \
  --allow-model-migration
```

Send one JSON object per line. A migration request has these fields:

| Field | Contract |
|---|---|
| `op` | Exactly `activate_model_migration`. |
| `id` | Optional normal request correlation ID. |
| `expected_generation` | Copy the **complete** generation object from the ready or latest status frame, including its nonce. |
| `receipt` | Absolute path to the original successful replacement selection receipt. |
| `model_dir` | Absolute replacement local model root; no ambient default is selected by the request. |
| `exact` | Required boolean, matching the replacement's build policy. `false` means the CLI's HNSW defaults and seed 42. Both tiers use F32 storage. |
| `quality_backend` | Optional `onnx`, `native-int8`, `native-f32`, or `native-multilingual`; the normal loader's stable default is ONNX. |
| `quality_model_dir` | Required absolute verified directory for an explicitly selected native quality backend; not an ONNX override. |

All paths must be UTF-8, NUL-free, and at most 4096 bytes. Unknown fields are
refused rather than ignored or interpreted as an unrestricted search. Quality
options on a fast-only target are errors. The regular 1 MiB request and 64 KiB
selection-receipt bounds still apply, followed by native selected-reopen limits.

A controller should fill `expected_generation` from an actual response, never
invent a nonce or infer authority from the largest sequence number. For example:

```python
import json

# session is an existing subprocess.Popen with text-mode stdin/stdout pipes.
# ready is the JSON-decoded ready frame (or a successful current status frame).
request = {
    "op": "activate_model_migration",
    "id": "upgrade-1",
    "expected_generation": ready["generation"],
    "receipt": "/srv/search/new.json",
    "model_dir": "/srv/models/new",
    "exact": True,
}
session.stdin.write(json.dumps(request) + "\n")
session.stdin.flush()
reply = json.loads(session.stdout.readline())
```

In this serial example the migration produces one terminal frame. Successful
acknowledgement includes the new generation,
previous generation, document count, quality presence, and both complete producer
fingerprints (`quality_producer` is null for fast-only). Subsequent searches use
the installed providers without reloading them per query.

## Refusal and recovery semantics

Before any requested path is read, the command verifies the independent startup
grant, layout, and exact expected generation. The existing offline verified model
loader then binds the requested models to the selected receipt. Native migration
admission verifies exact source IDs, content, titles and metadata; complete model
and prepared-input identities; generation including nonce; required tiers; vector
precision; graph parameters/seed; and the sealed Quill/vector artifacts.

A valid receipt for different source documents is not enough to replace the live
cohort, even if document count and model dimensions match. Changed prepared-input
contracts require freshly prepared input rather than relabeling retained text.
All these refusals leave this request without an installed candidate.

Installation uses the **same original predecessor pin** captured before model
loading. A source update or another migration that installs first makes this
candidate stale, even if its sequence is higher. Do not retry by changing only
`expected_generation`: rebuild from the current source cohort when necessary.
The refusal describes this request's effects, not a claim that another writer's
successful installation was undone.

A write or flush error while delivering a success acknowledgement stops the
session. Installation is not rolled back and no contradictory failure frame is
appended. A surviving owner can use `status` and the named trusted receipt to
resolve an unknown acknowledgement outcome. This remains process-local selection;
there is no automatic durable CURRENT update, rollback floor, or artifact GC.

Old externally retained snapshots keep their original models, source text,
lexical reader and vector owners. Keep their mapped artifacts immutable and
present for their entire lifetimes.

## Verification boundary

The implementation includes protocol tests using real v2 writers, sealed Quill
and native-HNSW artifacts, with synthetic identity-aware model providers. Those
fixtures do not establish real-model quality or latency. Run them on a supported
host with the repository's pinned dependencies and toolchain:

```sh
rch exec -- cargo test -p frankensearch --features quill \
  --bin frankensearch-native model_migration
```

Compilation, tests, rustfmt, Clippy, real-model execution and the unchanged full
quality gate remain required before release qualification. Their execution was
unavailable in the environment that authored this change.
