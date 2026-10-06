# Partitioned native indexes in the executable

`frankensearch-native index --shard-size N` builds a complete partitioned
source/vector inventory and one global Quill index. No alternate search engine
or shard-local BM25 merger is introduced. Without that option indexing retains
the ordinary single-cohort layout and existing receipt format.

```sh
cargo build -p frankensearch --features hybrid \
  --bin frankensearch-native --release

# Parent directories and the local embedding models must already exist.
frankensearch-native index \
  --index-dir /absolute/indexes/g1 \
  --receipt /absolute/receipts/g1.json \
  --input /absolute/documents.jsonl \
  --model-dir /absolute/models \
  --shard-size 5000

frankensearch-native search \
  --receipt /absolute/receipts/g1.json \
  --model-dir /absolute/models \
  --query 'retry network requests' --stream --timeout-ms 5000 \
  --filter '{"metadata":{"project":"alpha"}}'
```

The ordinary JSONL input language, unique-ID requirement, source limits and
model options apply: at most 100,000 documents, 16 MiB per record and 256 MiB
per input stream. Partition size is 1–100,000; more than 1,024 partitions is
refused before model loading or candidate creation. Input is validated before
any partition starts. An empty source still produces one identity-bearing
partition. Both tiers are required unless `--fast-only` is explicit; native
HNSW remains the default unless `--exact` is supplied. Native quality profiles
and `--quality-model-dir` work as documented for the ordinary command.

The selection schema `frankensearch.native.sharded-selection.v1` explicitly
chooses the partitioned loader. It binds the complete hybrid receipt, not a
vector-only or child receipt. Search follows that schema automatically; it does
not need a shard-size option. No filename scan, original-producer replacement,
missing-shard omission, graph-to-exact fallback or failed-open layout fallback
is performed. Aggregate native reopen limits apply before vector allocation,
with the command's document ceiling, plus the existing per-artifact and global
lexical limits. A receipt must be stored outside every sealed ancestor.

## Search contract and output

Fast, independently retrieved quality, full refinement and optional native
cross-encoder reranking all use the existing native library APIs. Each tier
embeds once per query, not once per partition. Source filters constrain every
partition and the global Quill candidate population before fusion and
hydration; restrictive scopes refill candidate windows instead of dropping an
already-truncated output page. Empty scopes skip inference but not identity or
required-tier admission. Reranking uses only the eligible retained source text.

Buffered queries require all requested phases to complete successfully.
Streaming reuses the ordinary command framing and deadline code: Initial is
flushed before quality inference, and Refined before reranking. One absolute
query deadline covers scoping, all retrieval stages and reranking; delivery
between phases consumes the budget. A timeout retains previously delivered
results and marks the terminal frame accordingly. Output failure stops
execution rather than appending a contradictory frame after partial output.

Partitioned result payloads include `layout: "sharded"` and `partitions`. Each
result preserves the ordinary score/metadata fields and adds separate nullable
`fast_row` and `quality_row` objects, each containing `shard` and `physical_row`.
The ordinary bare `index` is absent/null. Row coordinates refer to that response's
complete generation; a rank number is never a physical row or partition number.
No coordinate is synthesized for a lane that did not retrieve the document.
Unpartitioned queries retain their existing output fields and shapes.

Partitioned indexing, buffered/streamed search and warm serving with explicit
external activation and one-shot source updates are supported. Source recovery
for partitioned receipts is not yet enabled here. The corresponding library
lifecycle APIs remain available. This does not change default fsfs behavior or its generation
authority, create a durable CURRENT selector, establish persistent antirollback,
or authorize removing predecessors.

All admitted source/vector partitions remain resident and Quill retains mapped
files. Partition size is not a peak-RSS guarantee. Graph/model/filesystem work
and blocked input/output are not preemptible; deadlines reject late completion,
not promise hard interruption. This integration claims no measured speedup,
real-model quality, ANN recall or platform qualification.

Focused deterministic regressions (no model files required):

```sh
cargo test -p frankensearch --features hybrid --bin frankensearch-native \
  sharded::tests
```

## Warm serving and activation

`serve --receipt SHARDED_JSON` opens every selected partition and the global
Quill reader once, then reuses those owners and the loaded models for subsequent
JSONL requests. It uses the same bounded input loop, query parser, filter
intersection, progressive framing, total deadlines and output-failure policy as
the ordinary server. There is no separate sharded network service or protocol.

```sh
frankensearch-native serve \
  --receipt /absolute/receipts/g1.json \
  --model-dir /absolute/models \
  --allow-activation --timeout-ms 5000
```

The existing `op: "status"` control reports the current generation, document
count and required quality presence, plus `layout: "sharded"` and `partitions`.
The native-HNSW booleans are true only when every partition of that tier has a
graph; they do not mean that a mixed native/exact tier is wholly native.

An `op: "activate"` request requires the complete exact `expected_generation`
from ready/status (including its actual nonce) and the absolute path of a trusted
successor receipt. Activation is refused unless `--allow-activation` was given.
The selected receipt must use the same layout and original per-tier producers.
Prepare a higher-generation successor with `update` below or the sharded library
update/seal APIs, retaining its complete hybrid receipt; ordinary `index` starts
a fresh generation sequence and does not imply successor lineage.

Admission reopens all selected sources, vectors, native graphs and global
Quill through the original loaded models and resource limits. It does not run
inference, discover directories, change models, repair missing shards or fall
back to an ordinary index. The exact-predecessor live API installs the complete
candidate only after admission. Failed admission leaves the old head available;
a concurrently superseded predecessor is refused at installation.

Each query pins once through final delivery. Its scope, reranker text and both
row maps remain on that generation after activation. Subsequent queries obtain
the successor and recompute requested source scopes. Serving is sequential, so
admission delays later stdin requests; this is not a concurrent request server.
Output failure after activation stops input consumption without undoing the
installed generation or adding a contradictory failure frame. Reconcile a lost
acknowledgement through status rather than replaying a stale predecessor.

Activation grants no source-writing permission. `--allow-updates` is currently
refused for a sharded server before emitting ready, and update controls without
that grant perform no path access or inference. The ordinary server's source
transaction protocol is unchanged. All activation remains process-local: no
durable CURRENT switch, restart selector or persistent antirollback is added.

## Source updates and deliberate repartitioning

`update` accepts the same JSONL upsert/delete language for an explicitly selected
sharded receipt. Supply `--shard-size` for each sharded update; the capacity
originally requested at index time is not recorded in occupied partition lengths,
especially after deleting everything. No capacity is guessed and no layout change
is inferred. Ordinary updates continue to omit `--shard-size` and retain their
existing single-cohort implementation.

```sh
frankensearch-native update \
  --receipt /absolute/receipts/g1.json \
  --index-dir /absolute/indexes/g2 \
  --new-receipt /absolute/receipts/g2.json \
  --input /absolute/changes.jsonl \
  --model-dir /absolute/models \
  --shard-size 2500 --batch-size 16
```

The old receipt must admit every source/vector/graph/Quill artifact and the original
per-tier producers before update inference begins. The complete final source set,
including unchanged survivors in all partitions, must fit the command's document,
record and encoded-byte limits and the 1,024-partition ceiling before candidate
creation. The last edit per ID wins, but overwritten malformed operations are
still errors. No model substitution, failed-shard omission or quality fallback
is permitted. Use the original explicit quality-backend options when applicable.

The existing native update engine reuses vectors by source ID, exact prepared
content, producer identity and precision, not old shard position. Unchanged bodies
can move between partitions without inference; title/metadata-only edits update
source and global Quill without re-embedding. Inserts and changed bodies are
embedded once per required tier. Precision and graph policy are inherited from
the admitted predecessor. Deleting everything retains one empty identity-bearing
partition and can be followed by an ordinary sharded update after restart.

Every source/vector image, graph and the global Quill index is rebuilt in the new
directory. Only inference is incremental; this is not O(changed-bytes) writing or
a measured speedup. The new complete hybrid receipt is saved and synced before a
successful `updated` response. Old files, receipts and readers remain untouched.
Cancellation or a required late-stage failure can leave inert candidate files,
but never a partial-index success receipt. Failed output after receipt persistence
does not undo that saved generation. Use fresh destinations for retries.

This command does not activate a running server. Its receipt can be supplied to
the existing opt-in activation control, or used for a fresh search/server process.
Neither action establishes persistent CURRENT authority or authorizes collection.

```sh
cargo test -p frankensearch --features hybrid --bin frankensearch-native \
  update::sharded_tests
```
