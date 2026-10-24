# Native hybrid search executable

`frankensearch-native` exposes the existing native builder and selected-reopen
APIs as a local command: prepared JSONL documents become identity-bound FSVI v2
fast/quality tiers, native HNSW graphs, a Quill keyword index, and retained source
text. A later process opens that exact cohort using its saved receipt.

This is an opt-in executable, not a change to `fsfs`'s default layout or a claim
that the complete generation-authority migration is finished. It introduces no
new indexing engine, runtime library, or model download service. The retrieval
engines are native; the standard quality embedder uses ONNX Runtime. The explicit
`hybrid-native` profile below uses the existing native transformer instead.

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

## Fixed-authority generations with retained search

`authority-index` builds and publishes a complete generation through the fixed
`LOCK` and `AUTHORITY` protocol. Its immutable components are one FSVI v2 vector
tier, a sealed Quill image, and the generation metadata. `authority-search`
reopens the exact authority named by the saved receipt and searches the retained
vector and lexical images together, including deferred result metadata.

Choose either `--tier fast` or `--tier quality`. Both use exact vector retrieval.
A quality generation loads its quality producer directly; it does not require
a fast model. The source checkpoint is explicit: `--source-low` and
`--source-high` are the inclusive source sequence boundaries that produced this
input cohort. Use the actual source boundaries even when the number of input
documents differs from the number of source changes.

These commands require the qualified Linux root provider and supported local
ext4/btrfs storage. Put the root under a trusted private parent. The command
qualifies the new root before reading documents or loading models; an unsupported
filesystem or insecure ancestor fails at that boundary. The root, work directory,
and receipt must be new, separate destinations outside existing sealed indexes.
The examples use three siblings under one private parent:

```sh
mkdir -m 700 "$HOME/frankensearch-authority"
./target/release/frankensearch-native authority-index \
  --root "$HOME/frankensearch-authority/g1" \
  --work-dir "$HOME/frankensearch-authority/g1-work" \
  --receipt "$HOME/frankensearch-authority/g1.json" \
  --tier fast --source-low 1 --source-high 42 --cooperative-local \
  --input ./documents.jsonl --model-dir /absolute/path/to/installed/models
./target/release/frankensearch-native authority-search \
  --receipt "$HOME/frankensearch-authority/g1.json" \
  --model-dir /absolute/path/to/installed/models \
  --query 'retry network requests' --limit 10
```

`--cooperative-local` is required when publishing. It declares that the caller
owns the local root and that this command supplies no external anti-rollback
service. This choice is persisted in the receipt. Search cannot change the
receipt's security profile, generation, authority, producer identity, or tier.
A changed current authority is a refusal, even when its components are otherwise
valid. Already opened readers retain their original component bytes.

For quality-only native inference, build with `hybrid-native` as described below
and select the exact local native model:

```sh
./target/release/frankensearch-native authority-index \
  --root "$HOME/frankensearch-authority/q1" \
  --work-dir "$HOME/frankensearch-authority/q1-work" \
  --receipt "$HOME/frankensearch-authority/q1.json" \
  --tier quality --source-low 1 --source-high 42 --cooperative-local \
  --input ./documents.jsonl \
  --quality-backend native-int8 --quality-model-dir /absolute/path/to/native-minilm
./target/release/frankensearch-native authority-search \
  --receipt "$HOME/frankensearch-authority/q1.json" \
  --quality-backend native-int8 --quality-model-dir /absolute/path/to/native-minilm \
  --query 'retry network requests' --limit 10
```

The `onnx`, `native-int8`, `native-f32`, and `native-multilingual` quality choices
have the same artifact verification and explicit-selection rules as the other
native commands. All loading is offline. A missing or different producer fails;
the selected generation is never relabelled to match an available model.

The encoded-image budget defaults to 256 MiB. Set `--max-image-bytes N` on either
authority command to choose a different positive byte limit. It covers the
activation manifest and every declared component image. Search checks the
manifest's own length before reading it, then checks the combined declared
lengths before reading component objects. The effective search limit also cannot
exceed the image size recorded by the trusted receipt. This bounds encoded
images, not model memory, query scratch space, page cache, or other retained
readers. Indexing inference batches default to 64 documents on this route and
can be selected with `--batch-size`.

New receipts use schema `frankensearch.native.authority-selection.v2` and are
written durably before publication. In addition to the selected authority and
producer, each receipt preserves the original writer, exact attempt identity,
complete expected slot pair, floor requirement, and floor idempotency key. The
publisher uses that exact prepared attempt. A process crash before any success
or error output therefore leaves the inputs needed for reconciliation.

Receipt existence identifies a proposed authority; it does not acknowledge a
commit. The publication result keeps commit knowledge separate from serving
activation:

| Publication result | Meaning |
|---|---|
| `not_committed` | The publication call refused before authority mutation. |
| `unknown` | An authority write may have occurred; the error includes the exact bounded attempt for reconciliation. |
| `committed` | The authority switch succeeded. A later activation or output error does not undo that commit. |

Successful indexing reports `publication: "committed"` and
`serving: "activated"`. After an interrupted or uncertain publication, run:

```sh
./target/release/frankensearch-native authority-recover \
  --receipt "$HOME/frankensearch-authority/g1.json" \
  --max-image-bytes 268435456
```

Recovery reconciles only the exact saved attempt. It does not republish the
candidate or load embedding models. The exclusive publisher also refuses to
overwrite an unresolved persisted attempt, including from a different process.
Recovery reports one of the following verdicts:

| Reconciliation | Publication knowledge and serving behavior |
|---|---|
| `committed` | Commit is confirmed; the command separately attempts bounded admission of the exact selected generation. |
| `not_committed` | The complete prior authority pair proves an abort. Only a matching abandoned attempt frame may be cleared. |
| `still_unknown` | The outcome remains unresolved; the attempt must be retained and reconciled again. |
| `fork` | A conflicting authority prevents a commit/abort decision; the command fails without replacing it. |
| `repair_required` | Inconsistent or damaged authority evidence prevents reconciliation; the command fails without guessing. |
| `committed_then_superseded` | Commit is confirmed but a newer authority is selected. The old receipt is reported as superseded, without activating another generation through it. |

Unresolved, fork, and repair outcomes return a failing exit status. A known
commit remains `publication: "committed"` when bounded reader admission is
refused, cancellation defers activation, or output delivery fails. Those cases
also return a failing exit status. Repeating recovery is safe; it does not create
another publication. The receipt and staged objects remain available.

Selection-only v1 receipts still support `authority-search`; they cannot
authorize `authority-recover` because they do not contain an exact prepared
attempt. Recovery never invents missing fields or drops a saved floor
requirement.

Library callers can use `GenerationPublisherV1::prepare_attempt`, durably save
its permit and security/floor requirements, and then call `publish_prepared`.
A fresh serving session can restore that permit with `restore_pending_attempt`
and reuse `reconcile_pending` while retaining its existing reader.

The authority commands provide initial publication, exact selected search, and
publication recovery. The existing `update`, `rebuild`, and `serve` commands
continue to use their own selected-snapshot receipts. Work files and staged
candidates remain available after success or failure.

## Native quality without the ONNX feature bundle

The executable is also available with Quill, Model2Vec, and the existing native
transformer backend, without enabling the `hybrid`/`fastembed` feature bundle:

```sh
cargo build -p frankensearch --no-default-features --features hybrid-native \
  --bin frankensearch-native --release
./target/release/frankensearch-native index \
  --index-dir ./native-indexes/native-g1 --receipt ./native-receipts/native-g1.json \
  --input ./documents.jsonl --model-dir /absolute/path/to/installed/models \
  --quality-backend native-int8 --quality-model-dir /absolute/path/to/native-minilm
./target/release/frankensearch-native search \
  --receipt ./native-receipts/native-g1.json --query 'retry network requests' \
  --model-dir /absolute/path/to/installed/models \
  --quality-backend native-int8 --quality-model-dir /absolute/path/to/native-minilm \
  --stream --timeout-ms 5000
```

`--model-dir` retains its ordinary fast-model cache-root meaning. A native
`--quality-model-dir` names the **exact** verified weights/tokenizer directory,
not another cache root. Choose one of these explicit `--quality-backend` values:

| Backend | Selected quality producer |
|---|---|
| `onnx` | Existing local ONNX quality detection under `--model-dir`; default. |
| `native-int8` | Registered English MiniLM L6 with native INT8 linear execution. |
| `native-f32` | Registered English MiniLM L6 with native F32 linear execution. |
| `native-multilingual` | Registered multilingual MiniLM L12 native producer. |

All native choices require `--quality-model-dir` and the `native` or `rerank`
feature. `hybrid-native` enables `model2vec`, `quill`, and `native`. The existing
native loader verifies the registered model artifacts and executing producer
certificate before returning a model. Native inference uses the command-owned
asupersync blocking pool; no new runtime or inference engine is introduced.
The INT8/F32 choice concerns transformer execution, not FSVI storage precision.

The default backend remains `onnx` in **every** feature combination. A native-only
binary therefore requires explicit native selection for a quality-enabled
command. Adding a compiled backend cannot silently change the same command's
producer. An unavailable feature, missing native directory, failed verification,
or absent blocking pool is an error, never permission to use ONNX, another native
profile, or hash control. `--quality-model-dir` is invalid for ONNX; explicit
quality options are invalid on a fast-only build or selected fast-only cohort.

The same options apply to `index`, `search`, `serve`, `update`, and `rebuild`.
Use the original producer for ordinary reopen and incremental update. Native
INT8, native F32, multilingual, and ONNX vectors are not interchangeable merely
because their dimensions agree. The full producer fingerprint in the trusted
selection and snapshot is still checked. A deliberate model/backend migration
uses `rebuild` to re-embed all authenticated source documents into new files and
a new receipt. Old receipts and live readers are not rewritten or relabelled.

Warm serving retains both loaded models, including the native quality pool,
through query phases and live activation. Activation cannot select a different
backend or load a model from a request-supplied path. Native cross-encoder
reranking can also be enabled with `--reranker-dir` in this profile; its directory
is separate from the sentence embedder's directory. Startup model verification
and loading remain synchronous and outside the per-query deadline.

Cargo features are additive. Building with `hybrid`, `full`, `fastembed`, or an
ONNX-enabled workspace member as well can still link ONNX. Inspect the isolated
normal/build dependency graph with:

```sh
cargo tree -p frankensearch --no-default-features --features hybrid-native -e normal,build
cargo test -p frankensearch --no-default-features --features hybrid-native \
  --bin frankensearch-native models::tests
```

Explicit real-model command tests are marked ignored, not reported as passes
without installed fixtures. Set `FRANKENSEARCH_NATIVE_CLI_MODEL_ROOT` to the
Potion cache root and `MINILM_FIXTURE_DIR` to the verified native English model
directory. Run the actual INT8 path with:

```sh
cargo test -p frankensearch --no-default-features --features hybrid-native \
  --bin frankensearch-native models::real_model_tests::native_int8 -- --ignored --nocapture
```

Replace the test filter with `models::real_model_tests::native_f32` for F32 or
`models::real_model_tests::native_multilingual` for multilingual. The latter uses
`MULTILINGUAL_MINILM_FIXTURE_DIR` instead. These tests exercise actual native
indexing, all three query modes, scoped progressive output, incremental update,
rebuild, and retained predecessor reads. The INT8 test also refuses an F32
producer on the same-dimensional original selection. Missing fixtures fail an
explicitly requested test; there is no download or successful-skip fallback.
This profile is not a numerical-equivalence, relevance, performance, complete
platform-qualification, or no-C-toolchain claim.

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

## End-to-end query deadlines

`search --timeout-ms N` and `serve --timeout-ms N` set a total query budget,
from 1 to 600000 milliseconds. No option preserves unbounded retrieval. A
server request may supply `timeout_ms`, but its effective budget is the minimum
of that value and the server ceiling. Omitting or nulling it cannot disable the
server ceiling; zero and out-of-range values are rejected.

The same absolute deadline covers scope preparation, fast/quality retrieval,
hydration, and all requested progressive phases. It starts after startup model
and selected-index admission, not while loading a cold process or waiting on
stdin. Time spent delivering an intermediate page consumes the remaining query
budget. Expired work is not restarted on a new budget or a new generation.

A timeout is reported as `status: "timed_out"`, `code: "search_timeout"`, with
`elapsed_ms` and `budget_ms`. A stream that already delivered a page reports
`partial_results: true`; that page remains valid and no late replacement page
is emitted. The server can process its next query without reloading models or
cancelling the shared context. Buffered search returns an error rather than
claiming a partial page satisfied the requested query.

Deadlines use the caller-owned asupersync timer. They drop pending cooperative
work and reject late synchronous completion, but cannot preempt a synchronous
model/graph call, a blocked stdin read, or output. Output/activation are never
wrapped in a timeout that could falsely claim visible effects were undone.

## Native cross-encoder reranking

Build with the existing optional reranker feature and explicitly select a local
cross-encoder directory (not the embedding-model directory):

```sh
cargo build -p frankensearch --features hybrid,rerank --bin frankensearch-native --release
./target/release/frankensearch-native search --receipt ./native-receipts/g1.json \
  --query 'retry network requests' --reranker-dir /absolute/local/cross-encoder \
  --rerank-window 50 --timeout-ms 5000 --stream
./target/release/frankensearch-native serve --receipt ./native-receipts/g1.json \
  --reranker-dir /absolute/local/cross-encoder --rerank-window 50 --timeout-ms 5000
```

This loads the existing pure-Rust `NativeReranker` once and attaches the command's
asupersync blocking pool. There is no download, ONNX reranker substitution, or
model loading per query/activation. A missing feature, model or blocking pool
fails startup rather than silently disabling requested reranking. Model loading
is startup work outside the per-query budget. The ordinary quality *embedder*
is unchanged and may still use ONNX Runtime.

Full-mode requests run Initial, any configured quality refinement, then native
reranking of the retained retrieval window. A fast-only built cohort can also
use full mode: Initial is followed directly by reranking. The first two pages
are flushed before cross-encoder inference starts. The window defaults to 50
and accepts 1..1000; it is independent of the displayed result limit. Native
candidate budgeting grows to include the requested window. With a larger result
limit, the existing native policy reranks the requested head and retains its
unscored tail; no evaluated-count or whole-corpus relevance claim is made.

Fast and quality-primary server requests deliberately skip the cross-encoder.
One-shot `search --reranker-dir` therefore requires full mode instead of quietly
ignoring an explicitly supplied reranker. Server readiness names the configured
full-mode reranker/window. Individual requests cannot select arbitrary model
paths, reload the provider, or enlarge the rerank window.

The source text is the exact stored body from the query's pinned cohort, including
after live activation. Server/request scopes apply before retrieval and also
restrict the rerank inputs. No current filesystem body or unrelated text resolver
is used. Native response validation retains document IDs, source evidence and row
ownership; a malformed score envelope is a failure, not a partial reorder.

A successful final phase is `reranked` with actual `evaluated` count. Empty pools
skip the scorer; buffered output then retains the actual retrieval phase and
reports `rerank_applied: false`. Buffered full-mode search fails when any required
phase fails. Streaming emits `rerank_failed` with the exact preceding page and a
`degraded` terminal, and the next request remains usable. The original total
deadline includes reranking: expiry after Refined starts no scorer; late scoring
never replaces the delivered page. Synchronous native work is not preemptible.

## Recover a damaged native index by rebuilding its selected sources

Ordinary search, activation and incremental update still require every selected
search artifact to be healthy. `rebuild` is an explicit alternative when an FSVI
image, native graph or Quill artifact is missing/corrupt, or when replacing the
embedding models requires regenerating all vectors:

```sh
./target/release/frankensearch-native rebuild \
  --receipt ./native-receipts/g1.json \
  --index-dir ./native-indexes/rebuilt-g2 \
  --new-receipt ./native-receipts/rebuilt-g2.json \
  --model-dir /absolute/path/to/installed/models
./target/release/frankensearch-native search \
  --receipt ./native-receipts/rebuilt-g2.json \
  --model-dir /absolute/path/to/installed/models \
  --query 'retry failed network requests'
```

The old receipt must be the original trusted selection. Recovery verifies the
complete hash chain from that receipt through `native.hybrid.json` and
`native.snapshot.json` to `native.source.jsonl`. Both descriptors and the entire
source stream must survive intact: lengths, hash, ordered unique IDs, generation
and document count are checked before model loading or candidate creation. A
corrupt source or missing descriptor is refused, never salvaged as a valid prefix
or replaced by another discovered generation. An external backup is required
when this authentication chain is lost. Do not manufacture a replacement digest
from a damaged directory to bypass this check.

Rebuild needs no original model and opens none of the old FSVI, graph or Quill
files. It uses the exact retained prepared bodies, titles and metadata, not the
current filesystem or stdin. `--input` is deliberately invalid. All documents
are re-embedded into a new FSVI v2 generation; native graphs and Quill are freshly
built through the same complete builder and sealer as `index`. Nothing is copied
from a damaged vector owner, and no old vector is relabelled with a new identity.
The returned receipt records the actual new producing models.

Like `index`, rebuild requires both local models by default. `--fast-only`
explicitly chooses a successor without quality; this is NOT inferred from a
missing model or the old topology. Storage is fresh F32 with default native-HNSW
parameters/seed, or exact scan with `--exact`; unlike incremental `update`,
rebuild does not inherit the old graph configuration. `--batch-size 1..256` is
available. No model download, fallback or hidden source normalization occurs.

The complete persisted source stream, including its header, must fit 256 MiB,
with at most 100,000 documents and 16 MiB per encoded source record. The CLI's
existing nonblank/NUL-free/length-bounded ID contract also applies. These bounds
are checked before inference, not a claim about peak model or process memory.
Rebuilding uses full-cohort inference, storage and graph work, not incremental
reuse or a measured performance improvement.

The new generation sequence is the authenticated source generation plus one;
exhaustion is an error, never wraparound. Both output destinations must be new,
outside the predecessor, and non-overlapping, including through parent aliases.
The command seals the complete successor before creating its new receipt. It
writes neither an old receipt nor a CURRENT pointer. Old readers keep their old
cohort; existing live activation accepts a rebuilt successor only when its
ordinary producer/topology/predecessor checks pass. Changing models or topology
therefore requires reopening the server with the explicitly chosen new receipt.
A higher sequence here is not a persistent antirollback floor or a decision
between competing update branches.

Success emits `event: "rebuilt"`, `recovery: "authenticated_retained_source"`,
`vectors_reused: false`, the source `predecessor` identity, and the new selection
and receipt path. Cancellation or a build failure leaves the original untouched
and can leave an inert partial candidate, but creates no success receipt. A
receipt-write/sync or output-delivery failure can leave new visible bytes; it is
not an automatic rollback or a claim that the new snapshot was never written.
Use fresh destinations for a failed attempt. No destructive cleanup is performed.

Library consumers can recover the same owned source cohort without models via
`NativeBuiltIndex::recover_selected_source` (vector/source receipt) or
`NativeBuiltHybridIndex::recover_selected_source` (hybrid receipt), passing the
existing explicit reopen limits. These return a source generation and documents,
not a partially validated searchable index, and retain the trusted immutable-
directory contract. The ordinary strict `open_selected` APIs are unchanged.
