# Native hybrid search executable

`frankensearch-native` exposes the existing native builder and selected-reopen
APIs as a local command: prepared JSONL documents become identity-bound FSVI v2
fast/quality tiers, native HNSW graphs, a Quill keyword index, and retained source
text. A later process opens that exact cohort using its saved receipt.

This is an opt-in executable, not a change to `fsfs`'s default layout or a claim
that the complete generation-authority migration is finished. It introduces no
new indexing engine, runtime library, or model download service. The retrieval
engines are native; the standard quality embedder still uses ONNX Runtime.

## Build and index

```sh
cargo build -p frankensearch --features hybrid --bin frankensearch-native --release
mkdir -p ./native-indexes ./native-receipts
cat > ./documents.jsonl <<'JSONL'
{"id":"retry.rs","content":"Retry failed network requests with exponential backoff.","title":"Network retries","metadata":{"language":"rust"}}
{"id":"garden.md","content":"Plant flowers in the spring garden."}
JSONL
./target/release/frankensearch-native index \
  --index-dir ./native-indexes/g1 \
  --receipt ./native-receipts/g1.json \
  --input ./documents.jsonl \
  --model-dir /absolute/path/to/installed/models
```

Both models must already be installed in the normal model cache layout. Model
resolution is explicitly offline, refuses hash fallback, and does not download
anything. Missing quality is an error unless `index --fast-only` explicitly
chooses a generation without that tier. Native HNSW is built and saved for each
present tier by default; `--exact` creates exact-scan tiers without graphs.
Neither choice changes model identity. Storage is F32; inference batches default
to 16 documents and may be selected with `--batch-size 1..256`.

`--input` may be omitted to consume standard input. Each record has required
string `id` and `content`, optional string/null `title`, and optional string-valued
`metadata`. Unknown fields, duplicate IDs, malformed records and invalid UTF-8
reject the batch before model loading or destination creation. Content is passed
to the existing builder without hidden canonicalization or chunking; prepare
long documents into deliberate source units for the selected models.

The input limits are 16 MiB per encoded line, 256 MiB for the complete stream,
and 100,000 documents. These bound input, not model scratch space or total RSS.

## Reopen and search

```sh
./target/release/frankensearch-native search \
  --receipt ./native-receipts/g1.json \
  --model-dir /absolute/path/to/installed/models \
  --query 'retry failed network requests' --limit 10
```

`--mode full` (default) requires configured quality refinement; a declared
fast-only generation returns Initial. `--mode fast` does not run quality
inference. `--mode quality` independently retrieves quality plus keyword
candidates without running fast inference, and refuses a generation without
quality. The complete cohort and both configured models are admitted on reopen,
even when the requested query uses only one tier. `--limit` is 1..1000, default 10.

Without `--stream`, stdout is newline-terminated JSON containing the actual generation identity,
phase and ranked results. Errors return a nonzero exit code and JSON on stderr.
The result format reuses `ScoredResult`, including its scores and hydrated
metadata. JSON is completely encoded under a 16 MiB limit before output; an
encoding refusal does not expose a partial record. OS output failures can still
leave partial bytes and must be treated as failed delivery.

## Progressive output and warm serving

### Source filters in every query mode

`search --filter JSON` scopes candidates before vector normalization, lexical
fusion, and hydration. It works with full, fast, quality-primary and streamed
queries. For example:

```sh
./target/release/frankensearch-native search \
  --receipt ./native-receipts/g1.json --query 'retry network' \
  --filter '{"id_prefix":"src/","metadata":{"language":"rust"}}' --stream
```

A filter has optional `ids` (an exact document-ID set), `id_prefix` (a literal
prefix, not a glob or filesystem lookup), and string-valued `metadata`.
Comparisons are case-sensitive. Every present condition must match; missing
metadata does not match an expected value. `ids: []` selects no documents,
whereas `{}` adds no restriction. Unknown fields and invalid filters fail rather
than silently running an unrestricted search.

Warm requests accept the same `filter` object. `serve --filter JSON` establishes
a fixed server scope; a request's scope is INTERSECTED with it, never substituted.
An absent/null/empty request filter cannot remove the server scope. Activation
recomputes membership from the new retained source documents on the next query;
it does not reuse the old generation's eligible-ID set.

```jsonl
{"id":"scoped","query":"retry network","filter":{"ids":["retry.rs"]}}
{"id":"none","query":"retry network","filter":{"ids":[]}}
```

The implementation uses `NativeBuiltHybridIndex::scope`, which widens retrieval
within one pinned cohort instead of filtering an already-truncated top-k page.
Scoped result frames include `scope.filtered: true` and `scope.eligible_documents`
(the eligible source count, not query matches or a recall certificate). Empty
scopes skip inference but preserve required identity admission. Selective ANN
still has its ordinary approximate-recall contract; filtering does not make it
exact. Preparing a scope scans the retained source cohort once per query, and
selective lexical retrieval may scan all matches. Unfiltered queries avoid this
source scan and keep their previous result shape.

Filters are bounded to 4096 distinct IDs, 64 metadata fields and 64 KiB of
decoded filter text. Metadata keys/values are limited to 4096 bytes each; IDs
and prefixes are nonblank and at most 65535 bytes. Text is NUL-free. CLI filter
JSON itself is limited to 64 KiB; the existing server request bound still applies.
These are query restrictions, not authentication or revocation: status exposes
whole-cohort facts, and a controller permitted to activate cohorts remains
trusted to choose authorized receipts.

### Stream and serve

```sh
./target/release/frankensearch-native search \
  --receipt ./native-receipts/g1.json \
  --query 'retry failed network requests' --stream

./target/release/frankensearch-native serve \
  --receipt ./native-receipts/g1.json \
  --model-dir /absolute/path/to/installed/models
```

The server opens the selected cohort and configured models once, emits a `ready`
JSON line, and reads one JSON request per line from stdin:

```jsonl
{"id":"q1","query":"retry failed network requests","limit":10,"mode":"full"}
{"id":"q2","query":"spring garden","mode":"quality"}
```

Each admitted request emits `started`, one or more `results`, and a `terminal` frame.
Frames carry the client `id`, server `request` ordinal, per-request `seq` starting
at zero, and actual retained `generation`. Full mode flushes Initial before
polling independent quality refinement. Initial's candidate counts report zero
quality contribution. Refined replaces the displayed page. A quality failure
emits `refinement_failed` with the unchanged Initial results, followed by a
`degraded` terminal with `ok: false` and `partial_results: true`.

A request validation or retrieval error returns a failed terminal; the server
can handle the next request. Cancellation terminates serving at the next
checkpoint. An output failure terminates immediately, without attempting
another frame after possibly partial delivery. An oversized input frame also
stops the session without draining an unbounded line. Malformed complete JSON
lines are rejected individually. The request ceiling is 1 MiB; query text is
limited to 64 KiB, IDs to 256 bytes, and result limits to 1..1000.

`serve --mode` and `--limit` set defaults overridden by individual requests.
EOF, or a literal `quit` or `exit` line, ends the session. Requests run serially;
there is no result cache, detached worker, or model initialization per query.
Every query pins one complete cohort through all phases. By default the server
stays on its initial cohort. Explicit activation below can install a successor
between requests without loading models again; changing a receipt on disk alone
never retargets the process.

`search --stream` uses the same frame protocol for one request and returns a
nonzero exit code after reporting a failed/degraded terminal. It does not emit
the server's `ready` frame. Fast and quality-primary modes emit only their
requested result phase. The native phase engine adds no implicit deadlines:
checkpoint cancellation does not preempt synchronous graph/model work or an
idle blocking stdin read. This is not the `fsfs` stream-protocol schema.

## Incremental upsert and delete

```sh
cat > ./changes.jsonl <<'JSONL'
{"op":"upsert","id":"retry.rs","content":"Retry failed network requests with exponential backoff.","title":"Updated network notes","metadata":{"language":"rust","version":"two"}}
{"op":"delete","id":"garden.md"}
{"op":"upsert","id":"queues.rs","content":"A bounded queue applies backpressure to producers."}
JSONL
./target/release/frankensearch-native update \
  --receipt ./native-receipts/g1.json \
  --index-dir ./native-indexes/g2 \
  --new-receipt ./native-receipts/g2.json \
  --input ./changes.jsonl \
  --model-dir /absolute/path/to/installed/models
./target/release/frankensearch-native serve --receipt ./native-receipts/g2.json
```

Updates reopen the trusted predecessor and use its `NativeIndexUpdate` builder.
The last operation for an ID wins; deleting an absent ID is a no-op. An upsert
replaces the complete source document, including title and metadata. A rename
is an explicit delete plus upsert. All records, even superseded records, must
be valid. Unknown operations/fields, invalid IDs, malformed input, or an
oversized final source cohort reject the update.

Unchanged content reuses the admitted vector in each eligible tier. A title or
metadata-only change therefore updates Quill and retained text without running
embedding inference again. New or changed content is embedded by the original
providers. Tier presence, precision, native graph parameters and seed are
inherited; update cannot silently replace a model or remove quality.

Only inference is incremental: source serialization, FSVI images, native graphs
and Quill are rebuilt as a complete successor. This is not an O(delta) indexing
or memory claim. The final source cohort is bounded to 100,000 documents,
16 MiB per serialized document line, and 256 MiB in total, so repeated bounded
deltas cannot grow it beyond the executable's limits. `--batch-size` remains
available. Empty/no-op batches still build a new receipted generation.

The old generation and receipt are never overwritten. The successor receives a
strictly greater sequence and a fresh nonce, and is sealed before its NEW receipt
is saved. The command reports the predecessor, distinct edited-ID count and
successor selection; the edit count is not an inference count. An existing
server continues using its old complete cohort until explicitly activated or
restarted with the new receipt. There is no implicit live activation or automatic
choice between concurrently built successors.

## Activate a successor without restarting the warm server

Start `serve --allow-activation` only when stdin belongs to a trusted controller.
Ordinary serving refuses activation before reading its receipt path. Read-only
status is always available:

```jsonl
{"op":"status","id":"current"}
```

After `update` has saved a successful successor receipt, send an activation
record. `expected_generation` must be copied exactly from the current `ready`,
query, or status frame, including its nonce; the values below are illustrative.
The receipt path must be absolute, NUL-free and at most 4096 UTF-8 bytes.

```json
{"op":"activate","id":"install-g2","receipt":"/absolute/native-receipts/g2.json","expected_generation":{"schema_version":1,"sequence":1,"nonce":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]}}
```

The existing `NativeLiveHybridIndex` prepares the selected successor with the
retained model instances and admits source, fast/quality vectors, native graphs
and Quill before installation. The caller's full predecessor identity must match,
the successor sequence must be strictly greater, and both producer identities
and quality presence must agree. Installation also compares the exact retained
predecessor under the native live lock. Wrong receipts, corruption, cancellation,
stale requests, same-generation replay and rollback leave this request's selection
unchanged and do not trigger fallback, inference, or model discovery.

Status and activation each emit one `terminal` frame with `operation`, request
ordinal, client `id`, `seq: 0`, and the observed generation. Successful activation
adds `previous_generation`, `selection_changed: true`, and
`activation_scope: "process_local"`. Subsequent queries see the new cohort;
already-held snapshots still serve the previous cohort. The server is sequential,
so successor admission delays later requests rather than running on a detached
worker. Both generations can be resident during admission; this is not a
zero-pause or bounded-peak-RSS claim.

An acknowledgement write failure stops the server and does not roll back an
installed generation. Activation writes no receipt, CURRENT pointer or durable
authority: after process restart, `--receipt` still explicitly chooses the cohort.
The controller remains responsible for trusted receipt selection and persistence,
including choosing between concurrent update branches. This is not a filesystem
publication lineage, garbage collector, or persistent antirollback floor.

## Selection and lifetime contract

Both the generation directory and receipt must be NEW paths under existing
parents. Neither is overwritten. The receipt is outside the immutable generation
and new destinations beneath a known native or fsfs sealed ancestor are refused,
including through parent aliases. The receipt stores the exact hybrid-snapshot
byte receipt, generation identity, document
count and producing-model fingerprints. Reopening verifies the selected source,
vector, graph and Quill cohort through `NativeBuiltHybridIndex::open_selected`;
missing, changed or wrong-producer artifacts are errors, not fallback triggers.

Keep the successful receipt in trusted storage. Do not manufacture a digest from
an unknown or damaged directory to make it pass admission. Keep the generation
files immutable and present for every reader's lifetime. A receipt is explicit
selection, not a CURRENT pointer, writer-fenced live publisher, garbage collector,
antirollback floor or defense against hostile filesystem ancestors. Linux and
macOS are supported by the underlying snapshot-sealing contract.

Failure can leave an unselected directory or partial receipt for diagnosis;
retry with fresh destinations. A post-write sync/output failure does not imply
that no bytes reached disk. No cleanup, automatic rollback, model substitution,
source-file mutation or `fsfs` store conversion is performed.

## Tests

```sh
cargo test -p frankensearch --features hybrid --bin frankensearch-native
```

The executable's tests use explicitly identified deterministic providers with the
real Quill, FSVI v2 and native-HNSW implementation. They cover native/exact
round trips, independent tier execution, source/metadata preservation, unchanged
receipt reuse, corruption refusal, cancellation, input admission and bounded
output. Additional warm-server tests observe the actual Initial output flush
before any quality inference, exercise multiple requests on retained models,
preserve Initial on an injected quality failure, and stop after a broken Initial
delivery without starting quality. Update tests check last-edit-wins admission,
per-tier inference reuse, metadata replacement, deletion from Quill, persisted
successor reopening, old-reader preservation, cancellation and quality failure.
They are not real-model relevance or performance measurements.

Activation regressions cover both exact and native graph cohorts, multi-request
old/new membership, no inference during activation/status, malformed and damaged
receipt refusal, stale identity and rollback refusal, retained progressive phases,
cancellation before dispatch, and failed acknowledgement without rollback.

Scope regressions cover selective pages in all query modes, empty scopes without
inference, conjunctive metadata/ID restrictions, invalid-filter refusal, immutable
server restrictions, and membership changes after live activation.
