# Native warm update transactions

The existing native stdio server can apply source edits using its already-loaded
models and complete retained snapshot. This is distinct from activating a receipt
built elsewhere. Enable writing explicitly at process startup:

```sh
frankensearch-native serve --receipt /absolute/receipts/g1.json \
  --model-dir /absolute/models --allow-updates
```

Existing native-quality, reranker, filter and query-deadline options still apply.
`--allow-activation` alone does **not** enable updates. `--allow-updates` does not
enable externally supplied activation receipts; enable each permission needed.
The ready record reports `updates_enabled` and `update_max_mutations` separately
from `activation_enabled`.

## Request and success

Send one JSON object per line, using the exact generation object (sequence AND
nonce) returned by `ready`, `status`, or the preceding successful update:

```json
{"op":"update","id":"edit-17","expected_generation":{"sequence":1,"nonce":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]},"index_dir":"/absolute/indexes/g2","new_receipt":"/absolute/receipts/g2.json","changes":[{"op":"upsert","id":"src/client.rs","content":"retry network requests with exponential backoff","title":"Network client","metadata":{"language":"rust"}},{"op":"delete","id":"obsolete.md"}]}
```

The generation above is illustrative; copy the real complete object rather than
inventing its nonce. Every request needs both NEW absolute destinations. Their
parents must exist. Destinations may not overlap each other, the predecessor, or
a sealed generation, including via parent aliases. Existing files are never
replaced. Changes use the one-shot update command's same strict upsert/delete
schema and last-edit-wins semantics. Every record must be valid, even one later
overwritten. Unknown fields are refused, not ignored.

A request contains 1–1000 mutations and fits the server's existing 1 MiB encoded
line bound. The final source cohort, not only this request, remains subject to
the CLI's source-document and encoded-byte limits. Supplied bodies are prepared
model/lexical input; no filesystem source reread or hidden canonicalization is
performed. Updates run in batches of at most 16 changed documents, under the
existing per-batch input-byte bound.

The server takes one predecessor pin, builds through `NativeLiveHybridUpdate`,
and retains both original producers and the existing vector precision and graph
policy. Eligible unchanged bodies reuse vectors independently in each tier;
metadata/title-only changes update source and Quill without embedding inference.
Deletes disappear from every successor component. Inference is incremental, but
source/vector images, native graphs and Quill are still complete-cohort builds.
This is not an O(changed-bytes) or measured performance claim.

Only a successful complete build is sealed. Its new selection receipt is written
and synced before the existing exact-predecessor live installation. Success is a
terminal `operation: "update"` record with `stage: "installed"`,
`receipt_state: "durable"`, `selection_changed: true`, `previous_generation`,
`generation`, `edited_ids` (distinct IDs, not changed bodies), and the new receipt
and selection. Subsequent requests see the new snapshot. Independently held old
snapshots retain their old source, row ownership, models and Quill view.

## Failure boundaries

Stale expected generations are refused before path inspection, directory creation
or inference. A late competing installer is rejected by the exact predecessor Arc
comparison, not merely by generation sequence. It does not overwrite a winner or
relabel an old transaction as a new one.

Failure responses report the reached `stage` and `receipt_state`:

* `not_started`: no receipt write was attempted; an incomplete unselected build
  directory may remain when a build or seal failed.
* `uncertain`: receipt creation/write/sync was attempted and may have left bytes;
  live installation was not attempted. This includes conservative reporting when
  an exclusive-create collision prevented opening the destination.
* `durable` with `saved_not_installed: true`: the complete candidate and receipt
  were saved, but installation failed or was cancelled. They are retained. The
  request did not change the live selection; another concurrent owner may have.

An output error after installation ends the session. It never rolls back the
visible selection or appends a contradictory failure record after partial output.
Reconcile a lost acknowledgement using `status` and the explicit named receipt;
blindly replaying the old expected generation is refused before more build work.
A process restart still requires the explicitly selected receipt; no durable
CURRENT pointer, persistent antirollback floor, conflict merge or garbage
collector is introduced.

Source update work is not wrapped in the read-only per-query timeout. Cancellation
is checked at existing build/seal/install boundaries; synchronous filesystem,
graph and model work cannot be preempted. The stdio session is sequential: a build
holds up later stdin requests, though it holds no selection lock during inference
or construction. No concurrent-request or hard latency guarantee is made.

The stdin controller with `--allow-updates` is trusted to create new files and
edit the entire corpus. A search filter is **not** a mutation authorization policy;
upserts may change metadata and subsequent filter membership. Do not grant this
flag to an untrusted filtered-search client.

Focused regression command (no real embedding-model files needed):

```sh
cargo test -p frankensearch --features hybrid --bin frankensearch-native serve::warm_update
```

The fixtures use real native/exact storage and Quill with explicitly identified
deterministic providers. They exercise reuse, complete reopen, predecessor
preservation, refusal, failures, pending-work drop and lost acknowledgements;
they are not semantic quality or model-performance evidence.
