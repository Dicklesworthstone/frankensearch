# Explicit model-free keyword search

`frankensearch-native search --lexical-only` searches the authenticated source and
Quill components of an ordinary or sharded native receipt without loading an
embedding model, vector image, native graph, or cross-encoder. It is useful when
models are not installed or derived semantic artifacts are damaged, but it is an
explicit keyword query, not a fallback from a failed semantic request.

```sh
frankensearch-native search \
  --receipt /absolute/receipts/g1.json \
  --query 'retry network requests' \
  --lexical-only --limit 20 \
  --filter '{"metadata":{"project":"alpha"}}' \
  --stream --timeout-ms 5000
```

Omit all model options, `--mode`, and reranking options. Supplying them alongside
`--lexical-only` is an error, not permission to ignore them. Ordinary searches,
serving, activation and source updates retain their existing complete semantic
admission. This increment exposes keyword-only one-shot and streamed search;
`serve --lexical-only` is not supported. No default changes or downloads occur.

## What remains authenticated

The original caller-held receipt selects the complete hybrid descriptor. Every
source descriptor/stream and every selected lexical artifact must pass its exact
hash, length and structure checks. For sharded sources, all child inventories and
aggregate budgets are admitted before source decoding. The global Quill reader
is checked against the entire original source census, including IDs, content,
title and canonical metadata of empty or non-tokenized documents. Matching counts
and independently valid files are insufficient. Missing or corrupt source or
Quill data is an error; no successful prefix or neighboring generation is used.

This verifies the components used by the keyword query, not semantic health.
Vector, graph and model files are neither opened nor repaired. The source and
keyword files must remain trusted and immutable through the reader's lifetime,
including Quill's mapped files. This is not hostile-writer filesystem admission,
a durable CURRENT selector, a backup, or permission to delete old generations.

## Query and output contract

The command uses Quill's existing query-text preparation, candidate collection
and pinned hydration. Raw BM25 score bits, order and metadata are preserved;
there is no vector normalization, RRF, cross-encoder scoring, or shard-local BM25
merge. Only IDs belonging to the selected source may become results. Physical
vector `index` values are null because no vector image was admitted.

The existing exact-ID, ID-prefix and metadata filter syntax is supported. Scope
membership is evaluated against retained source records before candidate cutoff
and hydration. To avoid post-filter underfill, a selective scope requests the
complete matching keyword population once, then keeps its first eligible results.
This can use more candidate memory/work than an unfiltered query; it is bounded
by the admitted document count, not advertised as a filtered-query optimization.
An empty scope does not run keyword retrieval. No excluded body is hydrated for
output. Filtering is query scoping, not authentication or retroactive revocation.

Results carry `phase: "lexical"`, `retrieval: "lexical"`, the selected generation,
`source_layout`, and `semantic_components_verified: false`. They do not claim an
Initial/Refined semantic phase or successful validation of missing semantic files.
Streaming emits started, one results frame and terminal, with sequence numbers.
A failed delivery stops output; no contradictory frame follows partial delivery.

One existing native deadline covers scope construction, candidate work and
hydration, including elapsed time delivering started. Receipt/source/Quill
admission is startup work outside the query budget, as with other native queries.
Filesystem calls and blocked output are not preemptible; a deadline rejects late
completion, not guarantees hard interruption. Cancellation/timeout remains typed.

## Library entry points

Both `NativeBuiltHybridIndex::open_selected_lexical` and
`NativeBuiltShardedHybridIndex::open_selected_lexical` take the original complete
receipt and explicit source/lexical limits. They return the originating generation,
an immutable `QuillSearchIndex`, and owned source documents. The shared admission
uses the existing lexical seal and full census, not an alternative artifact format.
Ordinary full `open_selected` stays strict and never calls this reduced-mode API.

The CLI retains its 100,000-document, 16 MiB-record and 256 MiB-source ceilings.
For sharded input the aggregate 256 MiB also includes all source descriptors and
headers, as with recovery. Existing lexical artifact limits apply independently.
These are input bounds, not peak-RSS, inference-quality, ANN-recall or performance
claims. No model-migration, live-selection, durable-authority or retention change
is implied by successful keyword search.

```sh
cargo test -p frankensearch --features quill --lib \
  native_ann::builder::hybrid::snapshot::lexical_tests
cargo test -p frankensearch --features hybrid --bin frankensearch-native lexical::tests
```
