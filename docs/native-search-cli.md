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

Vector sealing, FSVI admission, native graph construction or reuse, and graph
persistence run on the command-owned blocking pool. The same placement applies
to incremental updates, rebuilds, sharded cohorts, and authority builds. The
command joins the complete worker result before it can return or publish that
generation. A stopped pool refuses the operation; it cannot move graph work
back onto the async executor. Cancellation can leave private candidate files,
but those files do not select a generation or replace an existing receipt.

Library callers can select a pool with `NativeIndexBuilder::with_blocking_pool`
or `NativeIndexUpdate::with_blocking_pool`. Otherwise, construction inherits the
pool attached to the supplied `Cx`. The caller owns the pool's lifetime and
shutdown. With neither source of a pool, the existing synchronous execution
contract applies. Provider inference retains its existing async API.

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
tier, a sealed Quill image, the generation metadata, and an optional native HNSW
graph. `authority-search`
reopens the exact authority named by the saved receipt and searches the retained
vector and lexical images together, including deferred result metadata.

Choose either `--tier fast` or `--tier quality`. `--retrieval exact|hnsw` selects
the vector search policy independently of the tier; new indexes default to
`exact`. HNSW uses the native engine's default parameters and deterministic seed
42. The graph is published as a fourth immutable component. Its complete native
receipt is authenticated inside the metadata and binds the graph to the exact
FSVI image, artifact generation, producer, storage identity, and live document
set. Admission also checks its structure, checksums, topology and parameters.
Missing or corrupt declared graphs cause refusal; they never silently change
the retrieval mode.
A quality generation loads its quality producer directly; it does not require
a fast model. The source checkpoint is explicit: `--source-low` and
`--source-high` are the inclusive source sequence boundaries that produced this
input cohort. Use the actual source boundaries even when the number of input
documents differs from the number of source changes.

These commands require the qualified Linux root provider and supported local
ext4/btrfs storage. Put the root under a trusted private parent. The command
qualifies the new root before reading documents or loading models; an unsupported
filesystem or insecure ancestor fails at that boundary. For `authority-index`,
the root, work directory, and receipt must be new, separate destinations outside
existing sealed indexes.
The examples use three siblings under one private parent:

```sh
mkdir -m 700 "$HOME/frankensearch-authority"
./target/release/frankensearch-native authority-index \
  --root "$HOME/frankensearch-authority/g1" \
  --work-dir "$HOME/frankensearch-authority/g1-work" \
  --receipt "$HOME/frankensearch-authority/g1.json" \
  --tier fast --retrieval hnsw --source-low 1 --source-high 42 --cooperative-local \
  --input ./documents.jsonl --model-dir /absolute/path/to/installed/models
./target/release/frankensearch-native authority-search \
  --receipt "$HOME/frankensearch-authority/g1.json" \
  --model-dir /absolute/path/to/installed/models \
  --query 'retry network requests' --limit 10
```

`--cooperative-local` is required by `authority-index` when creating the root.
Rebuilds and updates inherit the recorded profile. It declares that the caller
owns the local root and that this command supplies no external anti-rollback
service. This choice is persisted in the receipt. Search cannot change the
receipt's security profile, generation, authority, producer identity, or tier.
A changed current authority is a refusal, even when its components are otherwise
valid. Already opened readers retain their original component bytes.
Search and recovery derive the retrieval mode from the admitted generation, not
from a command-line override or a separately discovered graph path. Successful
responses report `retrieval: "exact"` or `retrieval: "hnsw"` for the actual reader.

### Rebuild an existing authority generation

`authority-rebuild` replaces the complete source cohort in the existing root.
It reads the selected generation from `--from-receipt`, retains that admitted
reader, and captures its exact authority pair before reading input or loading a
model. By default, supply the complete replacement JSONL corpus: omitted documents disappear
from the new generation, while readers already holding the old generation keep
its original documents and metadata.

```sh
./target/release/frankensearch-native authority-rebuild \
  --from-receipt "$HOME/frankensearch-authority/g1.json" \
  --work-dir "$HOME/frankensearch-authority/g2-work" \
  --receipt "$HOME/frankensearch-authority/g2.json" \
  --source-low 1 --source-high 57 \
  --input ./replacement-documents.jsonl \
  --model-dir /absolute/path/to/installed/models
./target/release/frankensearch-native authority-search \
  --receipt "$HOME/frankensearch-authority/g2.json" \
  --model-dir /absolute/path/to/installed/models \
  --query 'retry network requests' --limit 10
```

The selected receipt fixes the root, tier, and cooperative-local security
profile. The work directory and output receipt must be new, separate paths
outside the root. Both source boundaries must be at least their previous values;
an unchanged source cut is allowed. Empty corpora remain unsupported by this
route's metadata contract. Rebuild advances both the artifact generation and
authority sequence, refusing exhausted counters before inference.

Rebuild preserves the selected generation's retrieval policy, including the
full HNSW parameter set and seed authenticated by its native receipt. Supply
`--retrieval exact` to deliberately replace it with an exact-scan generation,
or `--retrieval hnsw` to select the default HNSW configuration explicitly.
The replacement graph is built from the new complete cohort; the retained
previous reader keeps its own original graph and vectors.

The loaded provider must match the selected generation's complete identity
before embedding begins. To intentionally rebuild with another producer, add
`--migrate-model` and select its local model using the usual model options. This
re-embeds the whole cohort and records the new producer; it does not relabel or
reuse the old vectors. The tier stays fixed, including direct quality-only
inference. Provider identity must remain stable throughout the build.

To rebuild the selected generation's existing documents, supply
`--retained-source`. This reads the complete stored cohort from its already
admitted vector and Quill images, preserves the exact source checkpoint, and
re-embeds every live document. Original JSONL files and build work directories
are unnecessary. For example, this rebuild retains the documents while changing
retrieval to HNSW:

```sh
./target/release/frankensearch-native authority-rebuild \
  --from-receipt "$HOME/frankensearch-authority/g1.json" \
  --work-dir "$HOME/frankensearch-authority/g2-work" \
  --receipt "$HOME/frankensearch-authority/g2.json" \
  --retained-source --retrieval hnsw \
  --model-dir /absolute/path/to/installed/models
```

Combine `--retained-source` with `--migrate-model` and the new model's local
options for a full-cohort model migration without resupplying the corpus.
`--input`, `--source-low`, and `--source-high` are excluded in retained-source
mode; it neither reads standard input nor assigns a new source checkpoint.
An ordinary rebuild with replacement JSONL still requires explicit source
boundaries.

Retained export requires the shipping Quill schema, matching live vector,
lexical, and metadata counts, and each document's exact stored identifier, full
content, title, and canonical string-map metadata. Tombstones are omitted. Empty
stored titles export as absent titles because Quill stores both forms as empty
bytes; nonempty titles and content are preserved exactly. Embeddings consume
document content. Missing or malformed fields refuse the whole rebuild before
model loading, inference, or creation of the new work directory.

`--max-source-bytes N` on rebuild is available only with `--retained-source` and defaults to
256 MiB. It bounds the sum of raw stored `id`, `content`, `title`, and
`metadata_json` bytes, including JSON punctuation and escaping in the metadata
field, checked before copying or decoding each field. The budget does not bound process
RSS: retained images, field materialization, and decoded documents also use
memory. The separate encoded-image budget still applies during admission, and
the CLI still requires at most 100,000 documents and a 16 MiB JSON-encoded record
including its newline for each exported document. Library callers can obtain
the same complete cohort through
`NativeGenerationSnapshot::source_documents(&cx, max_source_bytes)`.

Staging creates absent objects relative to the retained root descriptor. An
existing content-addressed image is reused only after qualified admission
verifies its security metadata, length, hash, and exact bytes. A partial,
corrupt, or linked collision is refused and left intact. Existing objects and
the old receipt are never overwritten. The new receipt durably records the
exact successor attempt before publication, and `authority-recover` accepts it
after an interrupted build publication.

Publication compares the original authority pair under the exclusive lock. If
another publisher has advanced the root, this rebuild reports `not_committed`
without changing that winner's authority or owner evidence. It does not refresh
its expectation and overwrite the winner. Rebuild explicitly from the winning
receipt to try a new replacement. Successful output has `event: "rebuilt"`,
`publication: "committed"`, `serving: "activated"`, the replaced authority, and
whether the model identity changed. The `source` field is `"retained_generation"`
for retained-source rebuilding and `"input"` for replacement JSONL. A later
output failure still reports the known commit.

### Incremental authority upserts and deletes

`authority-update` applies a JSONL delta to the exact generation selected by
`--from-receipt`. It recovers the complete source from the retained Quill image
and reuses the stored vectors for unchanged document content. Title and metadata
changes update the retained source and lexical image without document inference.
Only the final new or changed content for each ID is embedded.

An input file can contain:

```jsonl
{"op":"upsert","id":"retry-guide","content":"Retry transient failures with bounded exponential backoff.","title":"Retry guide","metadata":{"version":"2"}}
{"op":"delete","id":"obsolete-guide"}
{"op":"upsert","id":"new-guide","content":"Use jitter to spread retries across clients."}
```

An upsert replaces the complete document, including its title and metadata.
Omitted optional fields use their empty defaults. The last operation for each
ID wins, but every input record must still be valid, including records later
overwritten by another edit. Deleting an absent ID has no effect.

```sh
./target/release/frankensearch-native authority-update \
  --from-receipt "$HOME/frankensearch-authority/g2.json" \
  --work-dir "$HOME/frankensearch-authority/g3-work" \
  --receipt "$HOME/frankensearch-authority/g3.json" \
  --source-low 1 --source-high 63 \
  --input ./edits.jsonl \
  --model-dir /absolute/path/to/installed/models
./target/release/frankensearch-native authority-search \
  --receipt "$HOME/frankensearch-authority/g3.json" \
  --model-dir /absolute/path/to/installed/models \
  --query 'retry transient failures' --limit 10
```

Omit `--input` to read bounded JSONL from standard input. Both source checkpoint
boundaries are required and must be at least their selected predecessor values.
An empty delta can advance the checkpoint with zero document inference. A delta
that deletes the entire cohort is refused before model loading or creation of
the new work directory because this generation metadata contract requires a
nonempty cohort.

The selected generation fixes the root, security profile, tier, complete
producer identity, F16/F32 vector encoding, and retrieval policy, including all
HNSW parameters and the seed. The matching local model is required even for
metadata-only updates; quality-only generations load their quality model
directly. Use `authority-rebuild` for an intentional producer migration or
retrieval-policy change. Original input files and original build work
directories are unnecessary.

The input delta is limited to 256 MiB, with 16 MiB per encoded record and
100,000 distinct edited IDs. The complete merged source is independently
limited to 100,000 documents and 256 MiB of encoded JSONL, so repeated small
updates cannot exceed the cohort limit. `--max-source-bytes` separately bounds
raw source export from the predecessor, using the same 256 MiB default and
field accounting described for retained-source rebuilds. Final source bounds,
malformed input, stale receipt selection and backward checkpoints are checked
before loading the model. `--batch-size` controls inference batches;
`--max-image-bytes` bounds component admission.

Vector reuse saves document inference. Every successor still writes and admits
complete FSVI, Quill, metadata and optional HNSW images. The new work directory
and receipt must be separate from each other and from the selected root.
Existing snapshots retain their original source, vectors and graph throughout
the update and after publication.

Publication uses the exact authority pair captured before reading input or
loading the model. A competing publication produces `not_committed`; the
command retains the losing attempt's recovery receipt and preserves the
winner. Retry from the winning receipt explicitly. A successful response has
`event: "updated"`, `publication: "committed"`, `serving: "activated"`,
the new `selection`, the replaced authority, the final `documents` count,
and `edited_ids` after last-operation resolution. A result-output failure
preserves known publication and activation status.

Library callers can prepare the same transaction through
`NativeGenerationSnapshot::begin_update(&cx, work_dir, generation, provider, max_source_bytes)`,
apply `upsert_document` / `delete_document`, and call
`NativeIndexUpdate::build_exact_generation(&cx, tier, source, max_image_bytes)`.
The returned complete generation is ready for the ordinary expected-authority
publisher; building it does not itself select a new authority.

### Roll back the most recent authority publication

`authority-rollback` restores the immediate predecessor still held by the
authenticated authority slots. It works after a process restart and uses the
existing component images without loading a model or re-embedding documents:

```sh
./target/release/frankensearch-native authority-rollback \
  --from-receipt "$HOME/frankensearch-authority/g2.json" \
  --target-receipt "$HOME/frankensearch-authority/g1.json" \
  --receipt "$HOME/frankensearch-authority/rollback-g3.json"
./target/release/frankensearch-native authority-search \
  --receipt "$HOME/frankensearch-authority/rollback-g3.json" \
  --model-dir /absolute/path/to/g1/models \
  --query 'retry network requests' --limit 10
```

The current and target receipts must identify the same root, cooperative-local
security profile, and semantic tier. By default, both complete generations must
pass bounded admission. The target must be the current manifest's exact
predecessor and must still occupy the other authenticated physical slot. A
receipt for an arbitrary older generation is insufficient. Ordinary rollback
requires a healthy current closure; it does not repair corrupt component files
or discover historical targets.

Rollback publishes a new `Rollback` activation at the current authority sequence
plus one. The target's artifact generation, source checkpoint, producer identity,
document set, and vector/lexical/ANN/metadata images remain unchanged. Its
retrieval mode is restored with those images. This can reverse a model migration;
subsequent search requires the target producer's local model. No model, source,
input, retrieval, or work-directory options are accepted by the rollback command.

Predecessor admission is pinned to the captured current authority pair. A
concurrent publication refuses admission or loses the normal exclusive
publication fence. It never selects the winner as a replacement expectation.
Only a fresh activation manifest is staged, and the new receipt must be outside
the root. Existing components and receipts are preserved. The receipt records
the complete prepared attempt before publication, so an interrupted result can
be reconciled with:

```sh
./target/release/frankensearch-native authority-recover \
  --receipt "$HOME/frankensearch-authority/rollback-g3.json"
```

Successful output reports `event: "rolled_back"`, `publication: "committed"`,
`serving: "activated"`, the restored retrieval mode, `replaces` for the old
current authority, and `restores` for the selected predecessor. The new receipt
selects the higher authority sequence. Output failure after publication still
reports the known committed state. An already running authority server keeps
its retained generation until explicit activation; a producer-changing rollback
requires restarting that server with the restored model.

#### Recover when the current component images are damaged

If the current vector, lexical, metadata, or ANN image is missing or corrupt,
normal search and ordinary rollback refuse it. Explicit
`--allow-damaged-current` restores the healthy immediate predecessor without
opening or serving the current component images:

```sh
./target/release/frankensearch-native authority-rollback \
  --from-receipt "$HOME/frankensearch-authority/g2.json" \
  --target-receipt "$HOME/frankensearch-authority/g1.json" \
  --receipt "$HOME/frankensearch-authority/recovered-g3.json" \
  --allow-damaged-current
```

This mode still authenticates the exact current `LOCK`/`AUTHORITY` state and
activation manifest. It checks both predecessor links, requires the target's
actual other slot, and fully admits the target's vector, lexical, metadata, and
optional ANN images. Two shared-lock resolutions must agree on the captured
current pair. It never follows a new winner or creates a searchable handle for
the damaged current generation.

The mode does not bypass corrupt control anchors or a corrupt current activation
manifest. Pending publication still requires reconciliation; a corrupt target,
unretained ancestor, unsafe alias, WAL, or other invalid root entry still refuses
recovery. If a damaged object is also used by the target, that target cannot pass
complete admission. The command remains limited to the cooperative-local
profile recorded in both receipts.

The configured `--max-image-bytes` budget covers the current activation manifest
plus the predecessor's manifest and component images. The target receipt's own
image limit also remains enforced. No model loads, source scans, or document
inference run. Only the new rollback activation manifest and external receipt are
created; damaged objects remain untouched. Publication uses the captured pair
and the existing durable attempt and `authority-recover` reconciliation path.

Successful output includes `current_admission: "control_manifest_only"` for
this explicit mode and `"complete_generation"` for ordinary rollback. This
reports what was validated; it does not claim that skipped current components
were examined or healthy. The restored generation always passes complete
admission before the command reports `serving: "activated"`.

### Keep an authority generation warm across requests

`authority-serve` opens one selected generation and loads its local producer once,
then serves bounded JSONL requests from standard input:

```sh
./target/release/frankensearch-native authority-serve \
  --receipt "$HOME/frankensearch-authority/g1.json" \
  --model-dir /absolute/path/to/installed/models --limit 10 --allow-activation
```

After the `ready` event, send requests such as:

```json
{"id":"first","query":"retry network requests","limit":10}
{"id":"second","query":"connection pooling","limit":5}
```

Each query searches the retained vector, lexical and optional HNSW images with
the warm producer. It does not reload models, rediscover component paths or follow
the root's current authority implicitly. Publishing a newer generation elsewhere
does not redirect an already running session. Results identify the authority,
artifact generation, tier and actual retrieval mode used for that request.

With `--allow-activation`, the stdin controller can explicitly install a newer
receipt between requests. Send `op: "activate"`, `receipt` naming the new receipt,
and `expected_authority` copied exactly from the session's last successful
selection event. The new receipt must select a higher authority sequence in the
same root, security profile and tier, with the same producer, embedding space and
input contract. Intentional rollback can therefore install older artifact bytes
under its new, higher authority sequence. A producer migration requires restarting
the server with that generation's local model configuration.

Activation admits the complete candidate through the root descriptor retained at
startup and swaps readers only after all checks pass. A malformed, foreign,
corrupt or stale activation reports a failed request and leaves the previous
reader installed. Search requests cannot change activation permission, the image
budget or the producer. With both activation and updates disabled, the session
stays pinned to its startup generation.

Input records are limited to 1 MiB. An oversized record stops the session without
draining it; an ordinary invalid bounded request produces one failure response.
Responses are completely encoded under the existing native CLI output bound
before emission; source-fetch responses have a stricter 1 MiB bound. Output
failure stops the session. Send `quit`, `exit`, or EOF
to finish. As with the existing native stdio server, input reads block on the
owning lane, and cancellation is observed between reads and requests.

### Publish edits through the warm authority server

Add `--allow-updates` to let the stdin controller apply durable incremental
updates using the already loaded model:

```sh
./target/release/frankensearch-native authority-serve \
  --receipt "$HOME/frankensearch-authority/g1.json" \
  --model-dir /absolute/path/to/installed/models \
  --allow-updates --batch-size 64 --max-source-bytes 268435456
```

The `ready` response advertises `updates_enabled`, the mutation count limit,
batch size and source-export budget. Each update is one JSON object with these
fields:

| Field | Value |
|---|---|
| `op` | `"update"` |
| `id` | Optional request ID |
| `expected_authority` | The exact byte array from the session's latest `ready`, `results`, `activated`, or `updated` response |
| `work_dir` | A new absolute work-directory path |
| `new_receipt` | A new absolute receipt-file path |
| `source_low`, `source_high` | The explicit inclusive source checkpoint, with neither boundary moving backward |
| `changes` | An array of the upsert/delete objects accepted by `authority-update` |

An update contains at most 1,000 mutation records, counting overwritten edits,
and must fit within the existing 1 MiB request limit. Every record is validated;
the final operation for each ID wins. Empty `changes` may advance the checkpoint.
The complete resulting cohort must remain nonempty and satisfy the same
100,000-document, 16 MiB record and 256 MiB encoded-source limits as the one-shot
command. Destinations must be UTF-8 paths of at most 4,096 bytes, outside the
selected root and separate from each other. `--batch-size` on the server
requires `--allow-updates`; `--max-source-bytes` also bounds filtered queries
and selected source retrieval when writes are disabled.

The server verifies that its retained authority pair is still current before
source export or inference. It preserves that exact pair throughout the build.
The original model, tier, F16/F32 encoding, retrieval policy and graph parameters
remain fixed. Unchanged content, including metadata-only edits, reuses its stored
vector. Complete successor images are built and admitted through the same path
as `authority-update`.

The new recovery receipt is synced before publication. A successful update
installs the exact admitted reader returned by the publisher, then acknowledges
`event: "updated"`, `publication: "committed"`, and `serving: "activated"`.
The response includes the new `authority`, `selection`, `receipt`, final
`documents` count, `edited_ids` count and `previous_authority`. Subsequent
queries use that selection without reloading the model. Update requests are
serialized with queries, so a long build occupies the request loop.

When the session has a fixed filter, the complete successor source and its
eligible membership are prepared before publication under the configured query
source budget. The acknowledged update installs the new receipt, reader and
scope together. Metadata-only changes can therefore move a document into or out
of scope while reusing its vector. Empty membership stays restricted, and later
updates can populate it again. A source-budget or predicate refusal preserves
the old authority and scope; unfiltered sessions skip this scope preparation.

Update permission includes installation of its own successful successor.
`--allow-activation` independently permits selecting a receipt published
elsewhere. A stale client, an externally advanced root, invalid edits or a
losing publication preserves the current warm reader. After an external advance,
explicitly activate the winning receipt before updating again.

Failure responses keep the serving `authority` separate from any
`candidate_authority`. They report `publication` as `not_committed`,
`unknown`, or `committed`, and `serving` as `not_activated` when the
candidate was not installed. A `receipt` is included only once its saved
recovery attempt is known to be durable. If publication is uncertain, run
`authority-recover` with that receipt before continuing. An unresolved attempt
or stale base refuses further writes. Existing retained readers remain available.

Cancellation after acknowledged publication does not erase its outcome, and a
broken output stream ends the session while retaining the known publication and
activation status in the terminal error. The server never retries an uncertain
write automatically.

### Filter fixed-authority search and warm requests

`authority-search --filter JSON` and `authority-serve --filter JSON` use the
same source predicate as ordinary native search: an optional exact `ids` set,
a literal `id_prefix`, and exact string-valued `metadata`. All present
constraints must match. Comparisons are case-sensitive; no glob, expression,
or path interpretation is applied.

```sh
./target/release/frankensearch-native authority-search \
  --receipt "$HOME/frankensearch-authority/g3.json" \
  --model-dir /absolute/path/to/installed/models \
  --query 'retry transient failures' --limit 10 \
  --filter '{"id_prefix":"src/","metadata":{"language":"rust"}}'
./target/release/frankensearch-native authority-serve \
  --receipt "$HOME/frankensearch-authority/g3.json" \
  --model-dir /absolute/path/to/installed/models --allow-activation \
  --filter '{"metadata":{"project":"client"}}' --max-source-bytes 134217728
```

The server's fixed filter is evaluated once against its retained source. Each
search request may add a `filter` object. This narrows the fixed membership;
it cannot widen or replace it:

```jsonl
{"id":"rust","query":"retry requests","filter":{"id_prefix":"src/","metadata":{"language":"rust"}}}
{"id":"all-allowed","query":"retry requests","filter":{}}
{"id":"none","query":"retry requests","filter":{"ids":[]}}
```

An absent or null request filter preserves the fixed scope. An empty object
adds no restriction; `ids: []` selects no documents. Filtered result frames
report `scope.filtered: true` and `scope.eligible_documents`, the number of
eligible source documents before query matching. Empty scopes return no hits
and perform no query inference, while still requiring the selected producer
identity and local model.

Both vector and lexical candidates are restricted before normalization,
fusion, and winner hydration. Retrieval widens within the same immutable
generation to find eligible candidates instead of discarding a completed
top-k page. HNSW remains approximate, and lexical scoring retains whole-corpus
statistics. A selective lexical query can examine the full matching cohort.

`--max-source-bytes` bounds complete stored-source export using the raw-field
accounting described for retained-source rebuilds, with a 256 MiB default. It
does not bound total process memory. A fixed server scope retains that source
once and reuses it for narrowed requests. Without a fixed scope, a filtered
request exports its own bounded source; unfiltered searches export none.
Requests cannot increase the configured budget. Filters retain the existing
limits of 4096 IDs, 64 metadata fields, and 64 KiB of decoded filter text.
Invalid filters fail the request without producing an unrestricted result.

Explicit activation evaluates the fixed filter against the newly admitted
generation before replacing any session state. Source validation, budget, or
policy failure preserves the previous reader and its membership. Previously
retained library scopes likewise keep their original source and authority
across publication and rollback; they are not live revocation mechanisms.
Library callers can retain an `Arc<NativeGenerationSnapshot>`, call
`snapshot.scope(&cx, max_source_bytes, predicate)`, and use `narrow` to derive
an intersected scope without exporting the source again.

### Fetch document text from the selected authority

Use `authority-fetch` to retrieve one document window in a fresh process without
loading an embedding model or starting a warm server:

```sh
./target/release/frankensearch-native authority-fetch \
  --receipt "$HOME/frankensearch-authority/g1.json" \
  --doc-id 'src/client.rs' --offset-bytes 0 --max-bytes 8192 \
  --include-metadata --filter '{"metadata":{"project":"client"}}' \
  --max-source-bytes 1048576
```

The receipt must still select the exact current authority. A newer publication
refuses this fresh open; use the matching retained server if you need text from
an earlier selection. The command reads the admitted vector and Quill images,
never the original document path. It rejects model, quality-backend, and writer
options and leaves authority controls and receipts untouched.

The same window, filter, metadata, and response contracts described below apply.
Only the requested live document is materialized. `--max-source-bytes` bounds
its complete stored ID, content, title, and metadata fields before copying, even
when only a small excerpt is requested. Unrelated documents consume no source
budget; a missing or deleted ID consumes none. The default source and encoded
image budgets are both 256 MiB and can be lowered independently. A successful
one-shot response has `request: 1` and a null request `id`. Validation, source,
or encoding failure emits no partial document frame; a write or flush failure
does not trigger another stdout response.

Every `authority-serve` session accepts read-only `fetch` requests, including
sessions without activation or update permission. This retrieves the stored text
behind a search hit from the exact retained authority. It uses no inference,
model reload, current-file read, or implicit authority refresh. The `ready` frame
advertises `fetch_enabled` and the document, content-window and response limits.

Each request has these fields:

| Field | Value |
|---|---|
| `op` | `"fetch"` |
| `id` | Optional request ID |
| `expected_authority` | The complete byte array from the search result whose source is being requested |
| `documents` | Between 1 and 32 document-window objects |
| `filter` | Optional source predicate that further restricts the fixed server scope |

For example, the `documents` array can request two windows of the same document
and the beginning of another:

```json
[
  {"doc_id":"src/client.rs","offset_bytes":0,"max_bytes":8192,"include_metadata":true},
  {"doc_id":"src/client.rs","offset_bytes":8192,"max_bytes":8192},
  {"doc_id":"docs/retries.md","max_bytes":4096}
]
```

`offset_bytes` defaults to zero and must be a UTF-8 boundary within the stored
body. `max_bytes` defaults to 8192 and accepts 4 through 65536 bytes. The end of
a window rounds down to a UTF-8 boundary. Use the returned `next_offset_bytes`
for continuation; it is null at the end. IDs remain literal document IDs, never
filesystem paths. They must be nonblank and NUL-free, at most 65535 bytes each,
and at most 64 KiB in total across the requested windows. Repeated windows count
toward the 32-window and ID limits and retain their original order.

Success is one `event: "terminal"` frame with `operation: "fetch"`,
`status: "complete"`, `partial_results: false`, `source: "retained_snapshot"`,
and the authority, artifact generation, tier, profile and retrieval mode. Each
found document supplies `doc_id`, `content`, `offset_bytes`, `end_offset_bytes`,
`total_bytes` and `next_offset_bytes`. Title and metadata are included only when
`include_metadata` is true. Absent, deleted and excluded IDs all return exactly
`{"status":"unavailable","doc_id":"the-requested-id"}` with no source details.
Filtered fetches report `scope.filtered: true`; they do not scan the corpus to
compute a global eligible-document count.

The full expected authority is checked before source access. A request carrying
an older authority fails after activation or update, even when a rollback has
restored the same artifact generation and component bytes. A server that has
not activated an externally published successor continues fetching from its
original retained reader. Clients must keep the authority from the matching
search result, rather than attach a newer authority to an old hit.

With a fixed filter, fetch borrows its admitted source inventory and only resolves
eligible IDs. Without one, it loads just the distinct requested live documents
from the retained vector and Quill readers. `--max-source-bytes` bounds their
combined raw stored ID, content, title and canonical metadata fields, using the
same accounting as retained-source rebuilds. Repeated windows share one source
materialization. This bound covers complete selected document fields even when
only a small content window is requested or metadata is omitted. It is not a
process-memory limit or an assertion that the entire corpus was exported.

The complete encoded frame, including JSON escaping, selection fields and final
newline, must fit 1 MiB. Invalid offsets, source-budget exhaustion or an oversized
encoded reply produce one bounded failure with no document prefix, and the next
request remains usable. A failed write or flush ends serving without a second
frame. Cancellation follows the existing authority-session contract.

Library clients can use `snapshot.source_document(&cx, id, max_source_bytes)` or
`snapshot.source_documents_by_ids(&cx, &ids, max_source_bytes)` without a model.
The batch API preserves order and duplicates, returns `None` for absent/deleted
IDs, and charges the raw source budget for each returned document entry. Callers
can deduplicate IDs when multiple windows share the same document, as the CLI
does. Both APIs retain the selected immutable reader and return no partial batch
on source validation, budget, allocation or cancellation failure.

### Quality models, budgets, and publication recovery

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

The encoded-image budget defaults to 256 MiB. Set `--max-image-bytes N` on any
authority command to choose a different positive byte limit. It covers the
activation manifest and every declared component image. Search checks the
manifest's own length before reading it, then checks the combined declared
lengths before reading component objects. The effective search limit also cannot
exceed the image size recorded by the trusted receipt. This bounds encoded
images, not model memory, query scratch space, page cache, or other retained
readers. Indexing inference batches default to 64 documents on this route and
can be selected with `--batch-size`.
HNSW graph bytes count toward this same aggregate budget during building,
staging, selected-reader admission, and recovery.

The publication budget also applies when activating a reader after the commit.
If another publisher selects a larger generation in that interval, admission
refuses it before component reads and keeps the prior installed reader. The
original publication remains a known commit; a reader-admission refusal does
not authorize another publication attempt.

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
and call `reconcile_pending_with_budget` while retaining its existing reader.
`GenerationPublisherV1::refresh_with_budget` provides the same encoded-image
limit for explicit refreshes. The existing `reconcile_pending` and `refresh`
methods remain available without an aggregate image cap.

### Library rollback from a retained generation

An application that retains a `NativeGenerationSnapshot` can propose a rollback
with `target.rollback_candidate(cx, &current, object_id, writer_fence)`. The
target and independently admitted current snapshot must belong to the same
logical root and the same physical root, `LOCK`, and `AUTHORITY` inodes. Their
security profiles must agree and allow mutation. A pending uncertain successor,
an exhausted authority sequence, or a target that is not older than the current
authority is refused.

The candidate is a new `Rollback` activation at `current.sequence + 1`. It keeps
the target's original artifact identity, producer, source checkpoint, document
set, and validated component receipts. Proposal construction performs no model
inference or filesystem operation. Stage only its fresh activation manifest
using `QualifiedGenerationRoot::stage_immutable_file`, prepare an attempt against
`current.retained_snapshot().authority_pair()`, save that permit durably, and
publish it through `GenerationPublisherV1::publish_prepared`. The normal publisher
re-admits the retained objects and applies the configured anti-rollback floor.
Use the target's original vector binding when admitting the resulting reader.
For an HNSW target, those receipts retain its graph and native owner binding;
rollback preserves its retrieval mode, parameters and seed as well as its vectors.

An intervening publication causes the stale rollback attempt to lose its exact
comparison. Existing readers keep their original bytes throughout successful
rollback or failure. This is a retained-snapshot library API; it does not add
historical discovery through paths or a rollback CLI command.

The authority commands provide initial publication, full-cohort successor
rebuilds, authority-selected exact or HNSW search, warm stdio serving with explicit
activation, and publication recovery. The existing
`update`, `rebuild`, and `serve` commands
continue to use their own selected-snapshot receipts. Work files and staged
candidates remain available after success or failure.

The library entry points retain their existing names:
`NativeIndexBuilder::build_exact_generation` and
`NativeGenerationSnapshot::admit_exact`. Here, "exact" describes the complete
component authority contract. Select `NativeBuildRetrieval::Hnsw` on the builder
to include a native graph, and use `retrieval_configuration()` on a built or
admitted generation to preserve its validated policy. This route still admits
one vector tier per generation; combined fast/quality and sharded authority
descriptors remain outside its supported shape.

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
