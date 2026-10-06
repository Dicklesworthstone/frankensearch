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

This increment supports partitioned indexing and buffered/streamed search.
Warm serving, command-level mutation and source recovery for partitioned
receipts are not yet enabled here. The corresponding library lifecycle APIs
remain available. This does not change default fsfs behavior or its generation
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
