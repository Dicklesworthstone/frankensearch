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

## Search fixed-authority generations with fsfs

The ordinary `fsfs search` command accepts an explicit trusted native selection
receipt through `--authority-receipt`. It uses the same receipt validation and
retained-generation admission as `frankensearch-native`.

```sh
fsfs search "retry timeout" \
  --authority-receipt /absolute/path/to/native-receipts/g1.json \
  --authority-mode lexical --format json

fsfs search "retry failed network requests" \
  --authority-receipt /absolute/path/to/native-receipts/g1.json \
  --authority-mode full --format json
```

The receipt supplies the root, expected authority, immutable generation,
and independent producer identities. By default it must identify the current
selection when the request opens. A query already admitted retains its original
generation through retrieval and source hydration.

For historical search, supply the target receipt and a separately trusted
current receipt for the same root. Both generations must fit the image budget.
The response identifies the target generation and the head that authenticated it
in `selected_history_of`. A stale head or unavailable ancestor fails admission
before fsfs loads an embedding model.

```sh
fsfs search "retry timeout" --authority-mode lexical --format json \
  --authority-receipt /absolute/path/to/native-receipts/g1.json \
  --authority-current-receipt /absolute/path/to/native-receipts/g3.json
```

| Authority mode | Required topology and work |
|---|---|
| `lexical` | Any admitted topology; searches retained Quill data without loading an embedding model. |
| `fast` | A fast primary tier; invokes only the fast provider. |
| `quality` | A standalone quality tier or complete progressive generation; invokes only the quality provider. |
| `full` | Complete progressive coverage; retrieves both independently bound tiers. |
| `partial` | Explicit partial-quality topology; refines the complete fast corpus with its stored quality subset. |

Without `--authority-mode`, the selected topology determines the mode:
fast-only and partial-quality generations use fast retrieval, quality-only uses
quality retrieval, and a full progressive generation uses both tiers. Partial
coverage remains explicit in the result evidence and cannot satisfy a full or
quality-only request. Use `--authority-mode` to select a tier on this route;
the ordinary `--fast-only` and `--quality-only` aliases are refused. If a
configuration file enables `search.fast_only`, `--no-fast-only` allows an
explicit quality, full, or partial request to override that policy.

Semantic requests use the independently configured fsfs fast and quality models.
Each selected model must match the generation's frozen producer identity;
matching dimensions alone are insufficient. Missing or different producers
return an error without switching tiers. Lexical mode still admits the complete
stored generation, but loads neither model.

Native documents are identified by their retained document IDs. Result metadata
and source hydration come from the same retained generation as the vector and
lexical readers. The output includes the selected authority, generation and
topology so consumers can associate results with the receipt they supplied.
The ordinary `--limit` controls the number of returned results.

Opening a retained generation has a default 256 MiB image budget; source
hydration has a separate 16 MiB budget. `--authority-max-image-bytes` and
`--authority-max-source-bytes` accept positive byte counts to change these
bounds. They require `--authority-receipt`, as does `--authority-mode`.

`search.quality_timeout_ms` supplies one quality budget covering model loading
and every required query phase. Full and partial native queries admit both
producers before Initial, so their Initial work also consumes this budget. The
default is 500 ms; increase it in the configuration when cold quality loading
needs more time. Expiration returns an error before any success payload.

Native publication, updates and recovery remain available through
`frankensearch-native`. Unsupported ordinary fsfs options are refused before
model work, including `--stream`, daemon transport, filesystem filters, query
expansion and reranking. JSON, JSONL and TOON return the native result payload
inside the normal fsfs envelope; table and CSV use the generic envelope
rendering. The existing `fsfs explain` command is not a native rank-explanation
adapter: the native fused result does not yet expose the original per-arm rank
evidence needed by that command. The native CLI remains available for streaming,
source scopes and retained-source reranking.

## Query exclusions

Native search supports `-term`, `-"exact phrase"`, and `NOT "exact phrase"`.
Phrase exclusions also accept single quotes and case-insensitive `NOT`.
Quoted positive text and escaped literal dashes retain their original lexical
syntax. For example, `"alpha -beta" -deprecated` searches for the positive
phrase while excluding documents that match `deprecated`.

Recognized exclusions constrain the whole query, including every Boolean
branch: `alpha OR beta AND -deprecated` cannot return a document matching
`deprecated` through either its lexical or vector score. Quill still receives
the original query, with recognized phrase-negation keywords normalized, so
its positive clauses, fields, phrases, boosts and relative lexical scores keep
their usual meaning. Exclusion membership uses that same reader's term and
phrase matching rules, including its indexed fields; it is not a substring
check against a newly opened source file.

Before the first retrieval phase, the query captures all matching excluded IDs
from the retained immutable cohort, bounded by its document count. Explicit
ID, prefix and metadata scopes remain conjunctive. Both vector tiers use the
existing filtered candidate widening; lexical retrieval widens the original
ranked window until enough eligible candidates remain or the matching corpus
is exhausted. Initial, Refined and Reranked retain the same excluded IDs and
source generation. Query exclusions do not change the persistent scope count
reported as `scope.eligible_documents`.

Embedding and cross-encoder inference receive only the positive query text.
A query containing only exclusions, such as `-deprecated`, returns the lexical
complement without invoking either embedding model or the reranker. Semantic
entry points still validate every required tier and producer identity; this
does not grant a fallback from failed semantic admission. Unpublished candidate
previews reject exclusion-only queries because lexical results cannot prove
that a replacement producer executed successfully.

These rules apply to ordinary and sharded native search, retained-authority
queries, streaming, serving, and explicit model-free keyword modes. The library
entry points on `NativeBuiltHybridIndex`, `NativeBuiltShardedHybridIndex`,
`NativeGenerationSnapshot`, their source scopes, and the fenced
`NativeQuillSnapshot` share this preparation. The lower-level
`NativeAnnIndex::search_hybrid_*` and `NativeShardSet::search_hybrid_*` operations
accept independently supplied readers and text. They reject recognized
exclusions with `native_ann.query.exclusions: immutable-cohort-required` before
embedding or lexical retrieval, including zero-result requests. For progressive
queries the refusal occurs on the first requested phase and finishes the stream.
An arbitrary reader cannot guarantee that its exclusion census and candidate
batch share one population; applying the constraint only to lexical results
would still allow excluded vector matches. Use one of the complete cohort or
snapshot entry points above for exclusion queries. Quoted literals and escaped
dashes keep their ordinary meaning, and direct `LexicalRead` access continues to
expose the underlying lexical query grammar.

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
`LOCK` and `AUTHORITY` protocol. Its immutable components contain one or two
independently identified FSVI v2 vector tiers, a sealed Quill image, the generation
metadata, and optional native HNSW graphs. `authority-search`
reopens the exact authority named by the saved receipt and searches the retained
vector and lexical images together, including deferred result metadata.

Choose `--tier fast`, `--tier quality`, `--tier full`, or explicit `--tier partial`
with `--quality-ids`. Full generations contain both complete tiers; partial
generations retain the complete fast corpus and the selected quality subset.
`--retrieval exact|hnsw` selects the vector search policy for
every stored tier; new indexes default to `exact`. HNSW uses the native engine's
default parameters and deterministic seed 42. Graphs are published through the
optional ANN component. For a full generation, a versioned image contains each
selected tier's graph. Each graph's complete native receipt is authenticated
inside the metadata and binds the graph to its exact
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
receipt's security profile, generation, authority, producer identities, or topology.
A changed current authority is a refusal by default, even when its components are otherwise
valid. Already opened readers retain their original component bytes.
Semantic search and recovery derive the stored vector retrieval mode from the
admitted generation, not a query-time override or a separately discovered graph path. Successful
responses report `retrieval: "exact"` or `retrieval: "hnsw"` for a uniform policy,
or `retrieval: "mixed"` when a full generation's tiers differ. The
`retrieval_by_tier` object names each stored tier's actual policy. A selective
fast or quality query reports that tier's policy in `retrieval`; a full query
reports the complete pair. These labels describe the admitted implementation,
not latency or recall guarantees.

### Complete two-tier generations and progressive queries

Use `--tier full` to bind the fast and quality providers, their complete vector
images, one lexical image, and the shared source checkpoint to one authority:

```sh
./target/release/frankensearch-native authority-index \
  --root "$HOME/frankensearch-authority/full-g1" \
  --work-dir "$HOME/frankensearch-authority/full-g1-work" \
  --receipt "$HOME/frankensearch-authority/full-g1.json" \
  --tier full --retrieval hnsw --source-low 1 --source-high 42 --cooperative-local \
  --input ./documents.jsonl --model-dir /absolute/path/to/installed/models
./target/release/frankensearch-native authority-search \
  --receipt "$HOME/frankensearch-authority/full-g1.json" \
  --model-dir /absolute/path/to/installed/models \
  --query 'retry network requests' --limit 10 --stream
```

The two producers may use different dimensions. Each is independently admitted
against its own complete identity; sharing a dimension or display name does not
establish compatibility. Both images must contain the same ordered live document
set and artifact generation. Incomplete quality coverage, cross-generation
images, and mixed semantic/control provenance are refused.

Both exact scans and native HNSW are supported in each full tier. The CLI's
explicit retrieval choice applies to both, while library builders can select
each tier's policy, HNSW parameters, and seed independently. Admission validates
every declared graph against its own retained vector owner before model loading
or search. A missing, damaged, or swapped graph refuses the generation, including
queries selecting the other tier; there is no automatic exact-scan fallback.

Full search independently retrieves quality candidates and merges the candidate
sets, so refinement can discover documents absent from the fast candidate window.
The default full query returns the refined result. `--stream` emits and flushes an
`initial` frame before beginning quality inference, then a `refined` frame. Both
frames name the same retained authority and artifact generation. A refinement
failure remains a failing request after the initial frame; it is not reported as
successful full retrieval. Failure to deliver the first frame stops refinement.

`--mode fast` and `--mode quality` query the chosen tier plus Quill. One-shot
search loads only that tier's model. `--mode full` requires both. The default for
a single-tier receipt is its sole tier; requesting a missing tier is refused
before loading models. Results include `mode`, `phase`, and `topology` alongside
the selected authority. Topology describes stored tiers, not a latency guarantee
or proof of semantic model availability.

Library callers build with
`NativeIndexBuilder::build_progressive_generation(&cx, source, max_image_bytes)`
and admit with `NativeGenerationSnapshot::admit_progressive`, supplying both
bindings independently. `search_refined_text`, `search_quality_text`, and
`search_progressive_with_quality` retain those owners throughout retrieval and
hydration. The `frankensearch_index::vector_cohort` module documents the canonical
FSVC v1 envelope containing the two original FSVI v2 images. Its aggregate byte
receipt occupies the existing Vector component; metadata records each inner
image's identity, dimension, size, hash, and physical row count. Live documents
are counted once. No new authority pointer or publication protocol is introduced.
The optional full ANN image similarly retains each tier's original native graph
with an independent metadata receipt. A graph for one tier cannot be reinterpreted
as the other tier's graph, even when the dimensions agree.

### Publish and search an explicit quality subset

A quality tier can cover a selected subset while the fast and lexical tiers
retain the complete source. This is useful when only part of a corpus has been
selected for the more expensive quality model. The subset is an explicit build
input; inference errors still fail the build instead of silently dropping IDs.

```sh
./target/release/frankensearch-native authority-index \
  --root /absolute/private/search/partial-root \
  --work-dir /absolute/private/search/partial-work \
  --receipt /absolute/private/search/partial.json \
  --tier partial --quality-ids '["document-a","document-b"]' \
  --source-low 1 --source-high 42 --cooperative-local \
  --input ./documents.jsonl --model-dir /absolute/path/to/installed/models
./target/release/frankensearch-native authority-search \
  --receipt /absolute/private/search/partial.json \
  --model-dir /absolute/path/to/installed/models \
  --mode partial --query 'retry network requests' --stream
```

`--quality-ids` must contain unique, existing document IDs and select a nonempty
proper subset of the input. Empty quality belongs to `--tier fast`; full quality
belongs to `--tier full`. Invalid selections fail before document inference or
creation of the work directory. Both providers retain complete independent
identities and can have different dimensions. `--retrieval hnsw` builds each graph
against its actual vector owner, including the smaller quality document set.

The selection receipt uses `frankensearch.native.authority-selection.v4` and
records `quality_documents` separately from total `documents`. Admission derives
both live counts from the retained images and rejects a forged receipt count.
FSVC v2 and FSAC v2 distinguish partial coverage from the strict full-coverage v1
formats. The full fast document set still anchors the lexical and metadata join;
no source documents or placeholder vectors are invented to claim full quality.

Partial receipts default to fast search. `--mode partial` explicitly retrieves
and fuses quality candidates, with `initial` and `refined` streaming phases over
one retained generation. Full and direct quality modes refuse this topology.
Filters, total query deadlines, optional reranking, and warm query caching retain
their existing contracts. Responses identify `topology: "partial_quality"` and
report actual quality and total document counts plus coverage in parts per million.
Full-query tuning remains restricted to complete full generations.

`authority-serve` retains both producers and accepts `"mode":"partial"` explicitly;
its default remains fast. Source fetch, lexical search, publication-attempt
recovery, and retained rollback use the complete admitted source. Partial sources
support subset-preserving `authority-update` and granted warm updates, plus graph
repair with `authority-repair --reuse-vectors` when both vector owners still
authenticate. `authority-rebuild --retained-source` can change retrieval policy
or, with `--migrate-model`, either producer while preserving the source and exact
quality subset. A granted warm `activate_model` request can adopt that partial
successor without restarting. Adding `--query --mode partial` and a new
`--rollback-receipt` to a retained-source model migration validates and retries
that exact partial query, as described below. Fresh-input rebuild still refuses
partial sources. After vector damage, `authority-repair` can preserve the exact
quality subset recorded in authenticated metadata by newer partial generations.
For historical metadata without that membership record, healthy vectors with
`--reuse-vectors` or explicit `--complete-quality` remain necessary. The latter
re-embeds both tiers from authenticated source and publishes full quality coverage.
`authority-complete-quality` explicitly fills the missing quality coverage and
publishes a full generation before those ordinary maintenance operations resume.
Its `--quality-ids` option instead publishes a larger partial subset for bounded
quality backfill while preserving every already covered ID.
A warm partial server can adopt that completed generation through the explicit
`activate_quality_completion` request described below.
Sharded authority generations remain unsupported.

Library callers use `build_partial_quality_generation(&cx, source, quality_ids,
max_image_bytes)`, `NativeGenerationSnapshot::admit_partial_quality`, and
`search_progressive_with_partial_quality` (including pool and scoped variants).
The existing full-coverage methods continue to require complete quality coverage.
For edits, `begin_partial_quality_update` returns a restricted
`NativePartialQualityUpdate` plan with upsert/delete and batch/storage settings;
`build(&cx, source, max_image_bytes)` preserves the surviving covered IDs.
`partial_quality_document_ids(&cx)` exposes those authenticated IDs without
exporting document content or loading models.
`begin_partial_quality_model_migration` returns the same restricted plan with
explicit permission to replace either independently identified producer. The
unchanged tier reuses its vectors; changed quality inference covers only the
retained subset. Each producer must preserve its own prepared-input contract.

### Rerank an individual authority tier

`authority-search` and `authority-serve` can apply an explicitly selected local
cross-encoder to fast-only or quality-only retrieval, including an individual
tier selected from a full generation:

```sh
./target/release/frankensearch-native authority-search \
  --receipt "$HOME/frankensearch-authority/g1.json" \
  --mode quality --query 'retry network requests' \
  --reranker-dir /absolute/local/cross-encoder --rerank-window 50 \
  --timeout-ms 5000 --stream
```

The binary requires the existing `native` or `rerank` feature and the local
embedding model for the selected tier. One-shot search loads no unused embedding
producer. Streaming delivers `initial` for fast retrieval or `quality` for
quality retrieval, then `reranked` after successful cross-encoder inference.
Buffered requests deliver only the final successful page. Quality scores and
quality row ownership remain intact after reordering; no fast or refinement
phase is fabricated. A warm server applies its configured reranker to every
semantic mode and advertises eligible modes in `rerank_modes`.

The rerank window is independent of the displayed limit and can promote an
eligible candidate outside the first page. Both fixed and request filters apply
before candidate selection. Source comes only from the same retained authority,
under `--max-source-bytes`; original source files are never reopened. Empty and
exclusion-only queries skip the scorer. Source-budget, provider-response and
inference failures fail the request; streaming leaves the already delivered
page intact and reports one failure. The original total query deadline covers
retrieval and reranking. Failure to deliver the first page starts no reranking,
and slow delivery of an already completed final page does not create a timeout.

`--full-query` fusion and graph tuning still require full mode. Explicit lexical
queries skip a warm server's reranker; a lexical-only startup rejects its model
options. Ordinary and sharded receipt-based `search`/`serve` commands also support
selected-tier reranking, with the retained source policy described below.

### Complete or extend quality for an existing fast-only or partial generation

`authority-complete-quality` extends the independently bound quality tier of the
selected fast-only or partial-quality generation. It reads the authenticated
retained source, reuses the fast vectors and exact lexical image, and publishes
one successor at the same source checkpoint. Without a backfill option it
completes all missing quality rows and produces a full generation. For partial
quality, existing quality vectors are reused. Original JSONL files are
unnecessary, and the fast model performs no document inference.

```sh
./target/release/frankensearch-native authority-complete-quality \
  --from-receipt "$HOME/frankensearch-authority/fast.json" \
  --work-dir "$HOME/frankensearch-authority/full-work" \
  --receipt "$HOME/frankensearch-authority/full.json" \
  --model-dir /absolute/path/to/installed/models \
  --quality-backend native-int8 \
  --quality-model-dir /absolute/path/to/verified/native-quality-model \
  --retrieval hnsw
```

For bounded backfill, add `--quality-ids` with the **complete successor quality
membership**, including every previously covered ID. The list must add at least
one uncovered source ID while leaving at least one source document uncovered:

```sh
./target/release/frankensearch-native authority-complete-quality \
  --from-receipt "$HOME/frankensearch-authority/partial.json" \
  --work-dir "$HOME/frankensearch-authority/partial-backfill-work" \
  --receipt "$HOME/frankensearch-authority/partial-next.json" \
  --model-dir /absolute/path/to/installed/models \
  --quality-ids '["document-a","document-b","document-c"]'
```

Use the predecessor's matching quality backend and model-directory options when
it has a native quality tier. Only newly selected rows require quality inference.
Empty, duplicate, unknown, unchanged or shrinking selections refuse before model
loading or staging, as does a selection that drops a previously covered ID even
when its count increases. A list covering the entire source also refuses; omit
`--quality-ids` to request full completion explicitly. Repeated bounded backfills
can resume from each newly published receipt until the remaining rows are ready
for full completion. This also supports starting with a fast-only predecessor,
whose existing quality membership is empty.

To advance by a bounded number of documents without maintaining an ID list, use
`--max-new-quality-documents N` instead of `--quality-ids`:

```sh
./target/release/frankensearch-native authority-complete-quality \
  --from-receipt "$HOME/frankensearch-authority/partial.json" \
  --work-dir "$HOME/frankensearch-authority/next-batch-work" \
  --receipt "$HOME/frankensearch-authority/next-batch.json" \
  --model-dir /absolute/path/to/installed/models \
  --max-new-quality-documents 100 --batch-size 16
```

The command selects the first missing source IDs in lexicographic order, using
the admitted quality owner to distinguish covered rows. It retains every
existing quality member and infers at most `N` newly covered documents across
the entire build. `--batch-size` separately limits each inference request; it
does not change the total backfill limit. `N` must be between 1 and 100,000, and
the two backfill options cannot be combined.

Resume with the newly published receipt and fresh work/receipt paths. No cursor
or external coverage list is needed. If missing rows remain, the receipt records
the actual larger partial subset. When the remaining rows fit within `N`, the
same command publishes a full generation, enabling full and direct quality
search. Zero limits, empty source and already complete selections refuse before
model loading. Failed or cancelled work advances no coverage; retrying from the
same still-current receipt selects the same missing IDs. A stale receipt refuses
instead of following another publisher's selection.

This bounds new quality inference, while complete successor images and graphs
still require source-wide memory and I/O. It is not a process-memory or elapsed
time limit.

The configured fast provider must still match its retained identity. Choose the
quality provider with the usual local model-directory and quality-backend flags.
For a partial predecessor, that quality provider must match its independently
retained identity too; completion cannot change the model. Existing quality
precision and retrieval policy are inherited, including F16 storage.
`--retrieval` can explicitly change the quality retrieval policy without changing
its vectors' precision. A newly added quality tier defaults to F32 and exact
retrieval. The existing fast precision and retrieval configuration stay bound to
their retained artifacts. The command
accepts the ordinary batch, source-byte and image-byte limits, but refuses
replacement source input, source-cut overrides and model-migration flags.

Publication requires the entire requested quality membership to be built and admitted, with the
same document set, source checkpoint, fast identity, existing quality identity
and lexical receipt. A cancelled or failed build leaves the incumbent selected. A competing publication
refuses the original expected-authority pair instead of rebasing. Work files
remain available after failure. Bounded backfill retains truthful partial
coverage and explicit `--mode partial`; full and direct quality search still
require complete coverage. A failed inference cannot publish a shorter list.

Library callers use `NativeGenerationSnapshot::begin_quality_completion`, then
`NativeQualityCompletion::build_partial(&cx, complete_quality_ids, max_image_bytes)`
for explicit membership, `build_next_batch(&cx, max_new_quality_documents,
max_image_bytes)` for automatic bounded backfill, or `build(&cx, max_image_bytes)`
for full completion. The automatic method returns a partial or full candidate
according to the remaining rows; publication remains the caller's responsibility.
Bounded backfill rejects a change to existing quality precision, including the
automatic final batch, so it can reuse every retained quality vector. It still
permits a different quality graph policy.

A running fast-only or partial-quality server can adopt either successor receipt
without restarting. Start it with `--allow-model-activation`; its ready event
then reports `quality_completion_activation_enabled: true`. Publish the
successor separately with `authority-complete-quality`, then send a JSONL request
with these fields:

| Field | Value |
| --- | --- |
| `op` | `"activate_quality_completion"` |
| `id` | Your request identifier |
| `receipt` | Absolute path to the newly published full or expanded partial receipt |
| `expected_authority` | Exact byte array from the warm session's current selection event |
| `model_dir` | Explicit local model root for ONNX; optional when native quality uses its own directory |
| `quality_backend`, `quality_model_dir` | The usual explicit native-quality options, when used |

The operation retains the fast provider and admits the quality provider against
the completed receipt. For a partial predecessor, the existing quality identity
must remain identical, including its own prepared-input contract. The fast and
quality tiers may retain independently different input contracts; completion
does not reinterpret either one. Original source scope and query policy are
preserved. Fast query embeddings remain reusable, while the quality query cache
is recreated for the admitted provider. A partial successor requires a strictly
larger quality count, and admission proves that its actual authenticated quality
IDs include every previous member before loading the quality model.
Equal-count substitutions and larger sets that drop old members refuse.
References to the old snapshot keep their original coverage. An expanded partial
session still defaults to fast search, accepts explicit partial refinement, and
remains eligible for later backfill activation; a full successor enables full
and direct quality search.

This request only adopts an already published successor. It does not build
vectors, publish another generation or download models. Missing permission,
changed producers, changed source or lexical images, stale authority, loading
failure and cancellation leave the previous serving selection intact. Ordinary
`activate_model` preserves exact partial membership and cannot expand coverage.

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

The selected receipt fixes the root, topology, and cooperative-local security
profile. The work directory and output receipt must be new, separate paths
outside the root. Both source boundaries must be at least their previous values;
an unchanged source cut is allowed. Empty corpora remain unsupported by this
route's metadata contract. Rebuild advances both the artifact generation and
authority sequence, refusing exhausted counters before inference.

Rebuild preserves each stored tier's retrieval policy independently, including
its full HNSW parameter set and seed authenticated by its native receipt. A full
generation with exact fast retrieval and HNSW quality retrieval keeps that pair
when no override is supplied. Supply `--retrieval exact` to deliberately select
exact scans for every tier, or `--retrieval hnsw` to select the default HNSW
configuration for every tier. Replacement graphs are built from the new complete
cohort; retained previous readers keep their original graphs and vectors.

Every loaded provider must match its selected complete identity before embedding
begins. Full generations check both producers before either tier embeds source
documents. To intentionally rebuild with another producer in either tier, add
`--migrate-model` and select its local model using the usual model options. This
records the new producer and re-embeds every document in each changed tier.
Replacement JSONL rebuilds both tiers from that input. A migration using retained
source can reuse an unchanged tier as described below. The topology stays fixed,
including direct quality-only inference. Both producer identities must remain
stable throughout the build; a changed tier never adopts its predecessor's vectors.

To rebuild the selected generation's existing documents, supply
`--retained-source`. This reads the complete stored cohort from its already
admitted vector and Quill images, preserves the exact source checkpoint, and
builds a complete successor. Ordinary single-tier and full retained-source
rebuilds re-embed every live document; explicitly authorized model migrations can
reuse unchanged tiers. Partial retained-source rebuilds always reuse eligible
unchanged vectors and preserve the exact quality subset.
Original JSONL files and build work directories are unnecessary. For example,
this rebuild retains the documents while changing retrieval to HNSW:

```sh
./target/release/frankensearch-native authority-rebuild \
  --from-receipt "$HOME/frankensearch-authority/g1.json" \
  --work-dir "$HOME/frankensearch-authority/g2-work" \
  --receipt "$HOME/frankensearch-authority/g2.json" \
  --retained-source --retrieval hnsw \
  --model-dir /absolute/path/to/installed/models
```

Combine `--retained-source` with `--migrate-model` and the new model's local
options for a model migration without resupplying the corpus. Each tier is
evaluated independently against its authenticated predecessor identity, including
the producer, vector space, prepared-input contract, and persisted storage.
Changing only the quality model reuses the exact retained fast vectors while
re-embedding every quality document; changing only the fast model does the
reverse. If both identities match, neither tier performs document inference.
An eligible unchanged native graph can retain its topology; changed vectors or
graph construction parameters require a new graph. Every successor still has
complete vector, lexical, metadata, and configured ANN images with fresh
generation receipts. This avoids redundant inference without weakening complete
generation admission or claiming that migration cost is independent of corpus size.

Library callers use `NativeGenerationSnapshot::begin_model_migration` to prepare
the same explicit operation, then finish through `NativeIndexUpdate`'s ordinary
generation builders and expected-authority publication. Ordinary `begin_update`
continues to reject changed producers. The predecessor remains pinned and
searchable throughout; failed inference, cancellation, or a competing publisher
does not replace it with a partial migration.

Partial generations use `begin_partial_quality_model_migration` and the restricted
`NativePartialQualityUpdate::build` finalizer. The CLI supports this through the
same `--retained-source --migrate-model` options. A changed fast producer embeds
every source document; a changed quality producer embeds only the previously
covered IDs. Neither a model change nor a graph-policy change expands coverage.
The source checkpoint, complete stored documents, exact lexical image, and
covered-ID set stay fixed. Publication authenticates the candidate vector owners
and compares their actual membership before staging; equal coverage counts are
insufficient. The `--query` recovery executor supports this retained-source route
only with explicit `--mode partial` and a new `--rollback-receipt`. Partial
fresh-input rebuilds remain unsupported and refuse before model loading. Use
explicit completion to move to full coverage, or the ordinary partial update
operation to change source.

Retained documents are already prepared input. This route may change producers
and vector spaces, but must preserve each tier's complete input-preparation
contract. For a complete generation, changing canonicalization or document
encoding requires a complete freshly prepared corpus with `--input` instead.
Both tiers are checked before
either begins inference or creates the new work directory.
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

`--max-source-bytes N` on rebuild is available with `--retained-source` or recovery
`--query` and defaults to
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

### Rebuild a changed model and retry one exact query

Add `--query` and a new `--rollback-receipt` to an explicitly authorized model
migration to execute the offline identity-mismatch recovery path:

```sh
./target/release/frankensearch-native authority-rebuild \
  --from-receipt /absolute/path/to/g1.json \
  --work-dir /absolute/path/to/new-rebuild-work \
  --receipt /absolute/path/to/g2.json \
  --rollback-receipt /absolute/path/to/g2-rollback.json \
  --retained-source --migrate-model \
  --model-dir /absolute/path/to/replacement/models \
  --query 'retry network requests' --limit 10
```

This is a production consumer of the core `RecoveryPlan` for a verified local
producer identity change. The existing authority must still be
admissible. It supports fast-only, quality-only, complete two-tier and explicitly
selected partial-quality generations, including independently configured native
ANN tiers. All required producers are loaded locally; this operation does not
acquire models or execute plans supplied as JSON. Unchanged identities,
hash-control generations, empty replacement cohorts and unsupported recovery
states are refused.

For a partial predecessor, retain its source and explicitly request partial
retrieval:

```sh
./target/release/frankensearch-native authority-rebuild \
  --from-receipt /absolute/path/to/partial-g1.json \
  --work-dir /absolute/path/to/new-partial-rebuild-work \
  --receipt /absolute/path/to/partial-g2.json \
  --rollback-receipt /absolute/path/to/partial-g2-rollback.json \
  --retained-source --migrate-model --mode partial \
  --model-dir /absolute/path/to/replacement/models \
  --query 'retry network requests' --limit 10 --stream
```

The retained source and exact quality-covered IDs remain unchanged. Omitting
`--mode partial`, selecting full retrieval, supplying fresh input, or requesting
full-query tuning refuses before model loading. This recovery never fills the
missing quality rows or treats a successful fast-only query as validation of the
partial quality tier.

The command freezes one exact candidate and binds authorization to its authority
reference, the original authority and artifact generation, both independent
producer identities, source checkpoint, document counts and digests of the query
and its complete policy. For partial quality, it also binds the actual covered
document count and canonical digest of the sorted covered-ID set. The executor
compares the actual predecessor and candidate membership before preview and
again before publication; equal counts alone cannot authorize a different
subset. Retained-source migration preserves the complete source and both
preparation contracts. A freshly prepared complete `--input` cohort with
explicit source boundaries remains available when preparation must change.

Before any authority publication, a candidate preview opens the actual completed
component images and executes the requested semantic query. A full generation
or explicit partial query must complete both fast and quality phases.
`--filter JSON` fixes source membership before retrieval and fusion;
`--max-source-bytes` bounds source
materialization for that scope. Blank queries, zero limits and empty scopes
cannot substitute for semantic validation. Preview failure or cancellation leaves
the old authority selected.

An explicit `--reranker-dir` and `--rerank-window` also apply to fast-only and
quality-only recovery and to explicit partial-quality recovery. Both preview and
replay must complete the requested cross-encoder stage using the same retained
source, scope and model policy.
`--timeout-ms` supplies a fresh equal total query budget to each execution;
building and publishing do not spend those query budgets. Full-query tuning
remains restricted to full generations. A preview failure never publishes, and
a rerank failure or timeout during replay follows the rollback rules below.

Publication uses the original expected authority pair. An unknown outcome is
reconciled once before any replay; unresolved or superseded attempts retain their
actual status and saved receipt. The command never repeats publication or changes
its expectation to a competing publisher's head. After proven publication, it
replays the same query, filter and limit exactly once. Results are buffered until
the complete replay succeeds; `--stream` then preserves separate Initial and
Refined frames for full and explicit partial queries, followed by a reranked
frame when configured. Successful partial recovery reports
`partial_quality_coverage` readiness with the actual quality and total document
counts; it does not report complete semantic readiness.

An ordinary replay failure attempts one higher-sequence rollback to the retained
predecessor, using the just-published authority as its expected pair. The rollback
receipt is saved before changing authority, and every artifact image remains
intact. A rollback to a partial predecessor preserves its exact covered IDs; the
new receipt records the partial topology, quality count and producer identities.
Cancellation can prevent this compensation; the failure then reports the
known committed selection and that rollback is deferred. An unknown rollback
keeps its own exact attempt for `authority-recover`. A competing publication is
never overwritten. Output delivery failure after a successful replay preserves
the known commit rather than treating broken output as failed semantic retrieval.

The generic planner remains capability-blocked for other recovery operations,
including model acquisition, daemon alignment, ANN-only repair and rebuilding a
corrupt current generation. Library integrations can use
`NativeBuiltExactGeneration::preview_candidate` and the core
`BoundImmutableReindexPlan` with their own authorized executor; a serialized
report cannot recreate that process-local execution capability.

### Repair damaged vectors and graphs from authenticated source

`authority-repair` reconstructs the generation selected by a trusted receipt even
when its vector or ANN images are missing or corrupt. Its control state, metadata
and complete stored Quill source must still authenticate. No healthy ancestor or
original JSONL file is required. By default the repair re-embeds each stored tier's
authenticated document membership with the same local producers, preserving
their identities and persisted F16/F32 precision. It publishes a fully admitted
higher-sequence `Repair` successor with the same source checkpoint and document
fields. Explicit options below authorize full quality completion or a replacement
producer when those are needed.

Choose retrieval explicitly. To restore native ANN retrieval after vector loss:

```sh
./target/release/frankensearch-native authority-repair \
  --from-receipt "$HOME/frankensearch-authority/g1.json" \
  --work-dir "$HOME/frankensearch-authority/repair-work" \
  --receipt "$HOME/frankensearch-authority/repaired.json" \
  --retrieval hnsw --hnsw-seed 23 \
  --model-dir /absolute/path/to/installed/models
```

Both full and partial two-tier generations require their matching quality model
options.
`--quality-hnsw-seed` optionally selects an independent quality graph seed;
otherwise both tiers use `--hnsw-seed`. HNSW uses the default construction
parameters. Select a fresh seed: reproducing the selected ANN component's old
immutable address is refused before staging, even if a new embedding pass was
completed. `--retrieval exact` remains available and excludes seed flags.

If the vectors still authenticate and only the graphs need rebuilding, add
`--reuse-vectors` and omit model options. This route preserves the exact vector
values and Quill bytes without loading models, exporting the corpus or running
inference. It accepts the same explicit exact/HNSW choices and graph seeds.
This includes partial quality generations: the complete fast owner and the
sparse quality owner must both authenticate, and the repaired generation keeps
the same covered IDs, quality count, source cohort and per-tier vector values.
It remains a partial generation with explicit partial query mode; graph repair
does not fill missing quality coverage.

Newly built partial generations retain their exact covered IDs in authenticated
metadata. Ordinary source-only `authority-repair` uses this membership even when
both vector and graph images are missing or corrupt: the fast producer embeds
every retained document, and the quality producer embeds only the original
covered subset. The repaired generation keeps its partial topology, exact quality
membership, source fields, checkpoint and per-tier precision. Full and direct
quality query modes remain unavailable until coverage is completed explicitly.

The metadata field `native_quality_document_ids_v1` contains canonical, sorted
JSON document IDs derived from admitted live quality rows. Normal snapshot
admission verifies exact agreement with those rows; source-only admission checks
every ID against the authenticated live Quill source. A malformed declaration or
an unknown ID refuses repair. Historical partial generations without this field
remain searchable with healthy vectors, but cannot prove a lost subset from
source alone. They require either healthy retained vectors with `--reuse-vectors`
or explicit full completion. A retained-vector repair records the proven subset
in its successor, enabling subsequent source-only recovery.

Add `--complete-quality` to authorize a full successor instead. Every retained
document is then embedded by both matching local producers, including documents
that previously had no quality row. This route also accepts historical partial
metadata without the membership field. The source fields, checkpoint and per-tier
precision remain unchanged. The resulting receipt admits full and direct quality
search.

```sh
./target/release/frankensearch-native authority-repair \
  --from-receipt "$HOME/frankensearch-authority/partial.json" \
  --work-dir "$HOME/frankensearch-authority/complete-repair-work" \
  --receipt "$HOME/frankensearch-authority/full-repaired.json" \
  --complete-quality --retrieval exact \
  --model-dir /absolute/path/to/installed/models
```

`--complete-quality` is accepted only for a partial-quality receipt. It refuses a
fast-only, quality-only or already complete source before model loading.
Completion re-embeds the full source; it does not perform the bounded vector reuse
of `authority-complete-quality --quality-ids`.

If the original producer is unavailable or an intentional replacement is needed,
source repair accepts `--migrate-model` with explicitly chosen local model
options. The primary and quality producers may change independently. Both
replacement identities must retain their respective prepared-input contracts;
repair cannot reinterpret the stored document bodies using a different outer
canonicalization or content-selection policy. Persisted F16/F32 precision remains
the same, and every vector is newly inferred under its actual producing identity.
A replaced producer's original model and the old vector images are not required
for this route.

```sh
./target/release/frankensearch-native authority-repair \
  --from-receipt "$HOME/frankensearch-authority/g1.json" \
  --work-dir "$HOME/frankensearch-authority/migration-repair-work" \
  --receipt "$HOME/frankensearch-authority/migrated-repair.json" \
  --migrate-model --retrieval hnsw --hnsw-seed 29 \
  --model-dir /absolute/path/to/replacement/models
```

Add the matching quality backend/directory options for an explicitly selected
native quality model. For a partial generation with authenticated membership,
`--migrate-model` preserves the same quality subset under the new producers.
Combine `--migrate-model --complete-quality` when a damaged partial generation
needs both replacement producers and full coverage. Both flags conflict with
`--reuse-vectors`; unchanged producers remain mandatory without `--migrate-model`.

Every repair retains the original physical authority pair through preparation
and publication. A competing publisher causes refusal rather than replacing its
selection. Damaged source or control state, changed prepared-input contracts,
provider drift, cancellation and immutable object collisions also refuse; old
images remain intact. Repair accepts no replacement input or source advancement.
The only topology change it authorizes is explicit partial-to-full completion.
Source re-embedding obeys the retained-source and encoded-image budgets.
Successful output reports `event: "repaired"`, `publication: "committed"`, the
selected retrieval policy, `vectors_reused`, `quality_completed`, and
`model_migrated`. The last field reports whether the actual persisted producer
identities changed, even when migration was authorized. Repair does not preview
or replay a query. Its saved receipt supports the existing `authority-recover`
reconciliation after interrupted publication or lost output.

### Export stored documents after vector or graph loss

`authority-export` recovers the selected generation's complete stored source as
ordinary index JSONL without loading embedding models or publishing a successor.
It preserves each live document's identifier, content, nonempty title and metadata,
so the output can serve as input to a fresh index or an explicit source replacement.

```sh
./target/release/frankensearch-native authority-export \
  --receipt "$HOME/frankensearch-authority/partial.json" \
  --max-source-bytes 268435456 \
  > ./recovered-documents.jsonl
```

Current complete generations can export when vector or graph images are lost.
Partial generations can also export after both images are damaged or missing when
their metadata authenticates the original quality membership. Historical partial
metadata without that record uses independently admitted healthy vectors to prove
membership; the graph may still be damaged. Malformed membership, corrupt source
or metadata, and a receipt that no longer selects the current authority refuse
export. The recovery route does not change the authority or the damaged objects.

The command validates the entire inventory before writing output. The
`--max-source-bytes` limit bounds both raw stored fields and encoded JSONL; the
ordinary index input limits and separate encoded-image budget also apply. A source
or budget error therefore cannot emit a valid-looking incomplete inventory. An
empty selection emits zero records. Historical export with `--current-receipt`
continues to require healthy current and target generations with authenticated
ancestry.

### Authority source updates

`authority-update` applies a JSONL delta to the exact generation selected by
`--from-receipt`. It recovers the complete source from the retained Quill image
and reuses the stored vectors for unchanged document content. Title and metadata
changes update the retained source and lexical image without document inference.
Only the final new or changed content for each ID is embedded.

For a complete rescan, add `--replace-source` and supply the same document
JSONL used by `authority-index`, without `op` fields. That inventory is the
whole successor source: omitted IDs are deleted, while unchanged content keeps
its admitted vectors independently in each tier. Duplicate IDs are refused,
rather than resolved by input order. Title and metadata replacements still
avoid document inference. For example:

```sh
./target/release/frankensearch-native authority-update \
  --from-receipt "$HOME/frankensearch-authority/g2.json" \
  --work-dir "$HOME/frankensearch-authority/rescan-work" \
  --receipt "$HOME/frankensearch-authority/rescan.json" \
  --source-low 1 --source-high 63 \
  --replace-source --input ./documents.jsonl \
  --model-dir /absolute/path/to/installed/models
```

Both input modes preserve the selected producers, topology, precision and
retrieval policy, with the same source/record/image budgets and expected-old
publication. An empty replacement publishes a complete empty single-tier or
full generation. Partial replacements must retain a nonempty proper quality
subset; new IDs remain fast-only. A complete inventory never implicitly
changes models or expands quality coverage.

For a partial quality predecessor, membership follows the selected generation:
surviving covered IDs remain covered, and changed covered content is re-embedded
by both producers. New and previously uncovered IDs receive fast vectors only.
Deleting a covered ID removes its quality row; reinserting that ID in a later
generation leaves it fast-only. Within one delta, the final operation for an ID
wins, so a covered ID ending in an upsert retains its original coverage.
Title/metadata-only edits reuse vectors independently in both tiers.

The resulting quality subset must remain nonempty and smaller than the complete
source cohort. Deleting every covered ID, deleting every uncovered ID, or
deleting the entire partial corpus refuses before model loading or staging.
These updates cannot claim full coverage or silently discard the quality tier.
Use explicit quality completion before edits that need a complete topology.
The same rules apply to granted warm-server updates.

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
An empty delta can advance the checkpoint with zero document inference. For
single-tier and full generations, a delta that deletes the entire cohort
publishes a complete empty generation. Its
zero-row vector images, empty Quill source, metadata and optional graph are
admitted and committed together. Searches return no hits, source fetches report
the requested documents as unavailable, and later upserts repopulate the same
root. Full generations retain both independently bound tiers when empty.

Single-tier and full builds also support empty input through `authority-index`
and `authority-rebuild`, including `--retained-source` rebuilds of an empty
predecessor. Empty generations
keep their producer identities, storage and retrieval contracts, source
checkpoint and canonical empty document-set digest. Semantic builds and queries
still load and check the matching models, but zero documents require no embedding
inference. Lexical-only queries and source fetches remain model-free. Missing or
corrupt component images remain admission
errors. Readers retained before a delete-all publication keep their original
results, and rollback can restore either an empty or populated predecessor
through a new, higher-sequence activation.

The selected generation fixes the root, security profile, topology, complete
producer identities, F16/F32 vector encodings, and each tier's retrieval policy,
including all HNSW parameters and seeds. The matching local models are required even for
metadata-only updates; quality-only generations load their quality model
directly. Full generations independently reuse each tier's unchanged vectors;
metadata-only edits perform no document inference in either tier. New or changed
content is embedded by both producers. Use `authority-rebuild` for an intentional producer migration or
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
For a full generation, use `begin_progressive_update` with both providers and
finish with `NativeIndexUpdate::build_progressive_generation`. The existing
single-provider update method refuses a full snapshot before inference.
Both tiers retain their independent exact/HNSW configuration. Declared graphs
are rebuilt against the successor vector owners even when unchanged document
vectors can be reused without inference.

### Roll back to an authenticated retained authority

`authority-rollback` restores an ancestor authenticated by the current authority
and its retained activation-manifest chain. It works after a process restart and uses the
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
security profile, and topology. By default, both complete generations must pass
bounded admission. Each consecutive predecessor link must bind the exact next
manifest, down to the selected target. The target may be older than both current
physical slots; a receipt without the authenticated chain is insufficient.
Traversal permits at most 1,024 links. The image budget includes both complete
generations and all intermediate manifests. Ordinary rollback requires a healthy
current closure and does not repair corrupt component files or discover targets
by scanning the directory.

Rollback publishes a new `Rollback` activation at the current authority sequence
plus one. The target's artifact generation, source checkpoint, producer identities,
document set, and vector/lexical/ANN/metadata images remain unchanged. Its
retrieval mode is restored with those images. This can reverse a model migration;
subsequent search requires the target producers' local models. Full rollback
restores both original vector images and their independent identities. No model, source,
input, retrieval, or work-directory options are accepted by the rollback command.

Historical admission is pinned to the captured current authority pair. A
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
current authority, and `restores` for the selected ancestor. The new receipt
selects the higher authority sequence. Output failure after publication still
reports the known committed state. An already running authority server keeps
its retained generation until explicit activation; a producer-changing rollback
requires restarting that server with the restored model.

#### Recover when the current component images are damaged

If the current vector, lexical, metadata, or ANN image is missing or corrupt,
normal search and ordinary rollback refuse it. Explicit
`--allow-damaged-current` restores a healthy authenticated ancestor without
opening or serving the current component images:

```sh
./target/release/frankensearch-native authority-rollback \
  --from-receipt "$HOME/frankensearch-authority/g2.json" \
  --target-receipt "$HOME/frankensearch-authority/g1.json" \
  --receipt "$HOME/frankensearch-authority/recovered-g3.json" \
  --allow-damaged-current
```

This mode still authenticates the exact current `LOCK`/`AUTHORITY` state and
activation manifest. It follows only the exact predecessor references sealed
into successive manifests, checking each manifest's length, digest and consecutive
authority sequence. This can restore an older target after both physical slots
have been reused. It retains the authenticated manifest chain and fully admits
the target's vector, lexical, metadata, and optional ANN images. Intermediate
component images may also be missing or corrupt; they are never opened. Two
shared-lock resolutions must agree on the captured current pair. The command
never follows a new winner or creates a searchable handle for the damaged
current generation. An old receipt alone does not authorize recovery: every
manifest link from the current head to that target must survive and verify.

The mode does not bypass corrupt control anchors or a corrupt current activation
manifest. Pending publication still requires reconciliation; a corrupt target,
unretained ancestor, unsafe alias, WAL, or other invalid root entry still refuses
recovery. If a damaged object is also used by the target, that target cannot pass
complete admission. The command remains limited to the cooperative-local
profile recorded in both receipts.

The ancestry walk is bounded to 1,024 hops. The configured `--max-image-bytes`
budget covers the current activation manifest, every intermediate manifest,
and the target's manifest and component images; each object is charged before
its bytes are read. The target receipt's own image limit also remains enforced.
No model loads, source scans, or document
inference run. Only the new rollback activation manifest and external receipt are
created; damaged objects remain untouched. Publication uses the captured pair
and the existing durable attempt and `authority-recover` reconciliation path.

Successful output includes `current_admission: "control_manifest_only"` for
this explicit mode and `"complete_generation"` for ordinary rollback. This
reports what was validated; it does not claim that skipped current components
were examined or healthy. The restored generation always passes complete
admission before the command reports `serving: "activated"`.

### Read an authenticated ancestor without rolling back

`authority-search`, `authority-fetch`, and `authority-serve` accept
`--current-receipt CURRENT_JSON` to select a retained ancestor named by
`--receipt`. This lets you inspect or serve an earlier generation
while leaving the live authority unchanged:

```sh
./target/release/frankensearch-native authority-search \
  --receipt "$HOME/frankensearch-authority/g1.json" \
  --current-receipt "$HOME/frankensearch-authority/g2.json" \
  --lexical-only --query 'retry network requests' --limit 10
./target/release/frankensearch-native authority-fetch \
  --receipt "$HOME/frankensearch-authority/g1.json" \
  --current-receipt "$HOME/frankensearch-authority/g2.json" \
  --doc-id 'src/client.rs' --max-bytes 8192
./target/release/frankensearch-native authority-serve \
  --receipt "$HOME/frankensearch-authority/g1.json" \
  --current-receipt "$HOME/frankensearch-authority/g2.json" \
  --lexical-only --filter '{"metadata":{"project":"client"}}'
```

Both receipts must name the same physical root, root ID, and security profile.
The current receipt must select the actual current authority, and both complete
generations must pass admission on the same qualified root capability. The
target must be reached through exact, consecutive predecessor references in the
authenticated manifest chain, with at most 1,024 links. Missing or altered
intermediate manifests refuse history; ordinary current reads do not traverse
that chain and remain independent of its availability. An old receipt alone or
a matching artifact count cannot authorize access. Current and target may
have different producers, dimensions, or tier topologies; each is checked against
its own complete identity. After a rollback, authority lineage still determines
ancestry, even if artifact-generation numbers run in the opposite order.

For semantic search or serving, supply the target generation's matching local
models and quality options. The current generation's models are never loaded.
Lexical queries and source fetch remain model-free. Empty targets, full
progressive search, independently configured HNSW tiers, source windows, and
fixed/request filters use the same retained readers as ordinary current search.

`--max-image-bytes` bounds the **combined encoded images** of current and target
at startup: each activation manifest plus its declared vector, lexical, metadata,
and optional ANN components, together with all intermediate manifests. Shared images count in both independently admitted
closures. The default is 256 MiB; increase it explicitly when the pair requires
more. Source limits remain separate. Corrupt or missing current components,
stale current receipts, and changed control anchors refuse the read. A change to
the captured authority pair during historical admission also refuses the read.
This route does not use damaged-current recovery admission.

Historical responses keep the target in `authority` and `generation`, and add
`selected_history_of`, the current authority byte array that authenticated the
target at admission. An immediate predecessor reports
`selection_kind: "retained_predecessor"` and `selected_predecessor_of`; an older
ancestor reports `selection_kind: "retained_ancestor"`. Its reader retains the
complete current snapshot, intermediate manifest descriptors and target images,
without inventing a physical slot for an overwritten authority. In warm fetch
requests, `expected_authority` must match the target. A later publication does
not change that retained reader, its source scope, or its recorded provenance.
Fresh processes must authenticate the target against the then-current pair again.

Historical sessions are read-only: `--allow-activation`, `--allow-model-activation`,
and `--allow-updates` are rejected, and the corresponding requests are independently
refused from the retained snapshot's provenance. No new receipt or authority is
published. To change the live selection, use the explicit rollback command.

### Keep an authority generation warm across requests

`authority-serve` opens one selected generation and loads its local producers once,
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

For a full generation, requests may use `"mode":"fast"`, `"mode":"quality"`,
or `"mode":"full"` (the default). Add `"stream":true` to deliver separate
`initial` and `refined` result frames for a full query. Each frame preserves the
request ordinal and ID and uses the same retained authority. Single-tier query
modes emit one result frame unless a configured cross-encoder adds `reranked`.
The ready event declares `default_mode`, `progressive_available` and
`rerank_modes`. A full warm session loads both models at startup;
one-shot selective search can load only the requested tier.

Each query searches the retained vector, lexical and optional HNSW images with
the warm producers. It does not reload models, rediscover component paths or follow
the root's current authority implicitly. Publishing a newer generation elsewhere
does not redirect an already running session. Results identify the authority,
artifact generation, tier and actual retrieval mode used for that request.

With `--allow-activation`, the stdin controller can explicitly install a newer
receipt between requests. Send `op: "activate"`, `receipt` naming the new receipt,
and `expected_authority` copied exactly from the session's last successful
selection event. The new receipt must select a higher authority sequence in the
same root, security profile and topology, with the same producers, embedding
spaces and input contracts. Both identities are checked for full generations.
Intentional rollback can therefore install older artifact bytes
under its new, higher authority sequence. A producer migration uses the separate
model-activation grant and operation described below.

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

### Activate a rebuilt model generation without restarting

Start a semantic authority server with `--allow-model-activation` to permit
replacement of its local producer or both producers in a complete two-tier
or partial-quality generation:

```sh
./target/release/frankensearch-native authority-serve \
  --receipt "$HOME/frankensearch-authority/g1.json" \
  --model-dir /absolute/path/to/current/models --allow-model-activation
```

Build and publish the replacement separately using `authority-rebuild
--migrate-model`, then send an `activate_model` request. The ready frame reports
`model_activation_enabled`. The grant is independent of `--allow-activation`
and `--allow-updates`; neither of those grants permission to load another model.
Lexical-only startup rejects the model-activation grant.

| Request field | Meaning |
|---|---|
| `op` | `"activate_model"` |
| `id` | Optional request identifier |
| `expected_authority` | Exact authority byte array from the session's last successful selection event |
| `receipt` | Absolute path to the already published candidate's receipt |
| `model_dir` | Explicit local model root for fast or ONNX quality; required for the fast tier of full and partial generations |
| `quality_backend` | Optional explicit quality backend: `onnx`, `native-int8`, `native-f32`, or `native-multilingual` |
| `quality_model_dir` | Exact verified native-quality model directory when choosing a native backend |

Receipt and model paths must be absolute, UTF-8, NUL-free, and at most 4096 bytes.
Model selection never inherits an unspecified process default and never downloads
or acquires artifacts. Quality-only activation loads only its quality provider;
Full and partial generations load and validate fast and quality independently.

The candidate must belong to the original qualified root, security profile and
topology, and must advance the authority sequence. Each tier's prepared-input
contract stays unchanged. Both complete images and the server's fixed filter
membership are admitted before loading replacement models. Each new producer
must then match its selected embedding space, producer and input identities.
After loading, the server reopens that exact authority on its retained root to
detect an intervening publication. It never follows a different head.

For a partial session, build the successor using `authority-rebuild
--retained-source --migrate-model`. Warm adoption requires the exact same source
checkpoint, complete stored source, lexical image, and actual quality-covered
IDs. The server compares authenticated vector membership before loading either
model; equal counts cannot authorize a different subset. Source edits use the
separate partial update operation, and coverage expansion uses
`activate_quality_completion`. Each tier retains its own prepared-input contract,
including when those contracts differ between fast and quality.

Successful model activation replaces both query-embedding caches. A repeated
query is embedded under the admitted replacement identities before becoming
cacheable again. An activation refusal retains the previous models, caches,
reader, and fixed scope. The next permitted update embeds documents with the
installed producers and preserves that generation's partial coverage.

Only after all checks pass does the request loop install the receipt, reader,
fixed scope and complete model set together. The response reports
`event: "model_activated"`, `operation: "activate_model"`,
`serving: "activated"`, `previous_authority`, and `models_loaded` (one or two).
Subsequent queries and authorized updates use the installed producers. Library
readers already retaining the old generation remain usable.

Wrong or stale receipts, input-contract changes, missing models, either provider's
identity drift, source-budget failure, cancellation and concurrent publication
all preserve the previous complete serving selection. Loading is serialized with
requests, so a slow activation occupies this server's request loop. The operation
does not build, publish, modify authority controls or write a receipt.

### Publish edits through the warm authority server

Add `--allow-updates` to let the stdin controller apply durable source updates
using the already loaded producers:

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
| `changes` | Incremental array of upsert/delete objects, mutually exclusive with `documents` |
| `documents` | Complete replacement array of ordinary index documents, mutually exclusive with `changes` |

An update contains at most 1,000 mutation records, counting overwritten edits,
and must fit within the existing 1 MiB request limit. Every record is validated;
the final operation for each ID wins. Empty `changes` may advance the checkpoint.
Instead, supply a `documents` array to replace the complete source inventory.
Its records use `id`, `content`, optional `title` and string-valued `metadata`;
unknown fields and duplicate/invalid IDs refuse. The array is bounded to 1,000
documents within the same 1 MiB request limit. Supply exactly one input array.
Omitted IDs are deleted, and empty `documents` deliberately clears a complete
cohort. `edited_ids` includes both supplied documents and omitted predecessor
IDs, so this count can exceed the input-array length. Larger complete rescans
use the one-shot `--replace-source` command and its existing JSONL limits.
Single-tier and full sessions may produce an empty cohort. Partial sessions
must preserve a nonempty proper quality subset. Every result satisfies the same
100,000-document, 16 MiB record and 256 MiB encoded-source limits as the one-shot
command. Destinations must be UTF-8 paths of at most 4,096 bytes, outside the
selected root and separate from each other. `--batch-size` on the server
requires `--allow-updates`; `--max-source-bytes` also bounds filtered queries
and selected source retrieval when writes are disabled.

The server verifies that its retained authority pair is still current before
source export or inference. It preserves that exact pair throughout the build.
The original topology, producer identities, F16/F32 encodings, retrieval policy
and graph parameters remain fixed. Full sessions retain both producers and reuse
each tier's stored vector for unchanged content. Metadata-only edits therefore
perform no document inference in either tier; each final changed document is
embedded once by each producer. Complete successor images are built and admitted
through the same path as `authority-update`. Partial sessions use the restricted
subset-preserving builder: only changed covered documents reach quality
inference, and new or uncovered documents remain fast-only. The server keeps
both query embedding caches and recomputes the fixed source scope against the
successor. Existing retained readers keep their original source and coverage.

The new recovery receipt is synced before publication. A successful update
installs the exact admitted reader returned by the publisher, then acknowledges
`event: "updated"`, `publication: "committed"`, and `serving: "activated"`.
The response includes the new `authority`, `selection`, `receipt`, final
`documents` count, `edited_ids` count and `previous_authority`. Subsequent
queries use that selection without reloading the models. Update requests are
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

### Search a fixed authority without loading a model

`authority-search --lexical-only` and `authority-serve --lexical-only` explicitly
select keyword search over the authority's retained Quill image, for fast-only,
quality-only, and complete two-tier generations:

```sh
./target/release/frankensearch-native authority-search \
  --receipt "$HOME/frankensearch-authority/g3.json" \
  --lexical-only --query 'retry requests' --limit 10 \
  --filter '{"metadata":{"project":"client"}}'
./target/release/frankensearch-native authority-serve \
  --receipt "$HOME/frankensearch-authority/g3.json" \
  --lexical-only --allow-activation
```

This makes keyword retrieval available when the embedding model is unavailable.
It neither loads nor invokes a model. The complete selected authority, including
its vector and declared ANN images, must still pass normal admission; this is
not a partial-component recovery path. An old receipt still refuses a fresh open
after the root advances, while an existing session keeps its retained reader.

Results report `mode: "lexical"`, `retrieval: "lexical"`,
`semantic_used: false`, and `semantic_components_verified: true`. The
`retrieval_by_tier` map still describes the admitted vector policies; it does
not imply vector execution for this query. Scores are raw
Quill BM25 with lexical provenance; vector addresses and fast, quality and rerank
scores are absent. Metadata hydration uses the same retained lexical snapshot.
Fixed and request filters apply before the result cutoff and use the existing
source budgets. An empty scope stays empty without loading a model.

Authority servers with loaded models also accept an explicit lexical query:

```jsonl
{"id":"keywords","query":"retry requests","mode":"lexical","limit":10}
```

Omitting `mode` uses the server's startup choice. A server with loaded models
permits its available `fast`, `quality`, or `full` modes plus `lexical`; a
lexical-only server refuses all three model-backed modes with
`semantic_search_disabled`. Model failures never change the requested mode.
The ready frame lists the supported `query_modes`. One-shot search also accepts
`--mode lexical`; combining `--lexical-only` with a semantic mode is an error.

Lexical startup rejects model options and `--allow-updates`. Source fetching
remains available, and `--allow-activation` permits a complete newer authority
on the same qualified root, topology and security profile. Since no model is loaded,
the successor may have different producers. Activation still prepares the fixed
scope before changing any serving state. Library users can call
`search_lexical_text(&cx, query, limit)` on either a `NativeGenerationSnapshot`
or its prepared `NativeScopedGenerationSnapshot` without supplying an embedder.

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
and perform no query inference. Semantic execution still requires each selected
producer's identity and local model; explicit lexical execution requires neither
model loading nor query inference.
An empty full stream emits both empty `initial` and `refined` frames to complete
the request. Cancellation, inference failure, or a failed output write cannot
produce a successful completion frame.

Both vector and lexical candidates are restricted before normalization,
fusion, and winner hydration. Full queries apply the same eligible membership
to both independently retrieved vector tiers; initial and refined frames retain
that membership and the same authority. Retrieval widens within the same
immutable generation to find eligible candidates instead of discarding a
completed top-k page. HNSW remains approximate, and lexical scoring retains
whole-corpus statistics. A selective lexical query can examine the full matching
cohort.

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

The receipt selects the exact current authority by default. A newer publication
refuses that fresh open. To read an authenticated retained ancestor, add
`--current-receipt` as described above; an already running retained server also
keeps its original selection. The command reads the admitted vector and Quill images,
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

Single-tier receipts use `frankensearch.native.authority-selection.v2`. Full
receipts use `frankensearch.native.authority-selection.v3` and preserve the fast
identity plus an independent `quality_identity`. A v3 receipt missing either
tier, or a v1/v2 receipt claiming a second tier, is refused. All new receipts are
written durably before publication. In addition to the selected authority and
producers, each receipt preserves the original writer, exact attempt identity,
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
candidate or load embedding models. For a full generation it admits both vector
owners, their joined coverage, Quill, and metadata before reporting serving
activation. The exclusive publisher also refuses to
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
For an HNSW target, those receipts retain every declared graph and native owner
binding; rollback preserves each tier's retrieval mode, parameters, and seed as
well as its vectors.

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
admitted generation to preserve its validated primary policy. Full generations
expose the independent quality policy through `quality_retrieval_configuration()`;
this returns `Some(NativeBuildRetrieval::Exact)` for an exact quality tier and
`None` for a single-tier generation. This route admits a single exact or HNSW
tier, or a complete fast/quality pair with either policy chosen independently.
Explicit partial-quality authority generations use the separate APIs and
contracts described above. Sharded authority descriptors remain unsupported.

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

For a complete rescan, pass `--replace-source --input inventory.jsonl` to
`update`. Supply the same document records as `index`, in any order, with a
unique ID per record. Every omitted ID disappears from the successor, including
when the inventory is empty. The `edited_ids` count includes both supplied
and removed IDs. For a sharded receipt, provide the desired `--shard-size`.
Moving an unchanged document between partitions preserves its vector reuse.

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

Fast and quality-primary requests also support the configured cross-encoder,
for both ordinary and sharded receipts. `search --mode fast --reranker-dir ...`
retrieves with the fast model, then reranks; `--mode quality` retrieves directly
with the quality model, then reranks. No other query embedder runs and no
refinement phase is fabricated. Streaming emits `initial` or `quality` before
the final `reranked` page. Quality-primary scores, candidate counts and physical
row coordinates stay associated with the quality tier, including on failure.
Server readiness lists eligible `rerank_modes`; `reranker` and `rerank_window`
identify the loaded scorer. `full_mode_reranker` identifies that same scorer for
the full pipeline.
Individual requests cannot select arbitrary model paths, reload the provider,
or enlarge the rerank window. Full-query tuning remains specific to full mode.

The source text is the exact stored body from the query's pinned cohort, including
after live activation. Server/request scopes apply before retrieval and also
restrict the rerank inputs. No current filesystem body or unrelated text resolver
is used. Native response validation retains document IDs, source evidence and row
ownership; a malformed score envelope is a failure, not a partial reorder.

A successful final phase is `reranked` with actual `evaluated` count. Empty pools
skip the scorer; buffered output then retains the actual retrieval phase and
reports `rerank_applied: false`. Buffered reranked search fails when any required
phase fails. Streaming emits `rerank_failed` with the exact preceding page and a
`degraded` terminal, and the next request remains usable. The original total
deadline includes reranking: expiry after a retrieval page starts no scorer; late scoring
never replaces the delivered page. Synchronous native work is not preemptible.

## Add quality to a fast-only native index

`complete-quality` adds a complete, independently bound quality tier to an
ordinary or sharded fast-only selection. It retains the original prepared source
records and fast vectors, including their precision and exact/native-HNSW policy.
Only the new quality provider receives document inference requests. Both local
models must be available: the original fast model admits the retained fast owner,
and the selected quality model produces the new tier.

```sh
./target/release/frankensearch-native complete-quality \
  --receipt ./native-receipts/fast-g1.json \
  --index-dir ./native-indexes/full-g2 \
  --new-receipt ./native-receipts/full-g2.json \
  --model-dir /absolute/path/to/installed/models \
  --batch-size 16
./target/release/frankensearch-native search \
  --receipt ./native-receipts/full-g2.json \
  --model-dir /absolute/path/to/installed/models \
  --mode full --query 'retry failed network requests'
```

The quality tier uses F32 storage and native HNSW by default. `--quality-exact`
selects exact retrieval for that new tier; it does not change fast retrieval.
`--quality-backend` and `--quality-model-dir` select a local quality producer as
on fresh indexing. The provider must retain the original prepared-input
contract. `--input`, `--fast-only`, and `--exact` are invalid for completion.
A selection with an existing quality tier is refused; `rebuild` remains the
explicit full-reembedding operation for replacing models or damaged artifacts.

For a sharded receipt, pass an explicit `--shard-size N`. Repartitioning preserves
fast vectors by source ID even when shard boundaries change. An ordinary receipt
does not accept `--shard-size`. The command checks the same final source/document
limits as incremental update and supports `--batch-size 1..256`. Although fast
inference is reused, complete successor source/vector images, native graphs and
the global Quill index still require corpus-wide memory and I/O.

Both destinations must be new, non-overlapping and outside the predecessor.
The command seals all tiers and Quill before saving the new selection receipt,
then emits `event: "quality_completed"`, the predecessor, the new selection and
`fast_reused_documents`. The original receipt, artifacts and readers remain
usable. Failure or cancellation during building creates no completion receipt;
inert partial files can remain. A failure or cancellation during or after receipt
persistence can leave a complete new receipt, so inspect that destination before
retrying with fresh paths. Completion never rewrites an existing receipt or
activates a running server. Open the new receipt with both matching models to use
quality or full search, or adopt it through the explicit warm control below.

Library consumers can call `begin_quality_completion` on `NativeBuiltIndex`,
`NativeBuiltHybridIndex`, `NativeBuiltShardedIndex`, or
`NativeBuiltShardedHybridIndex`. The returned `NativeIndexQualityCompletion`
can build either layout, select quality storage and bound inference batches,
while keeping the fast producer and source content fixed. Live hybrid snapshots
also expose a completion builder; their completed candidates install through
the existing exact-predecessor check, so a stale completion cannot displace an
intervening activation.

### Adopt completed quality in an ordinary or sharded warm server

A fast-only `serve` session can adopt the sealed successor from `complete-quality`
without restarting or reloading its fast model. Grant this control at startup:

```sh
./target/release/frankensearch-native serve \
  --receipt ./native-receipts/fast-g1.json \
  --model-dir /absolute/path/to/installed/models \
  --allow-model-migration
```

After building and saving the completion receipt separately, send:

```json
{"op":"activate_quality_completion","id":"add-quality","receipt":"/absolute/native-receipts/full-g2.json","expected_generation":{"schema_version":1,"sequence":1,"nonce":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]},"model_dir":"/absolute/path/to/installed/models","quality_exact":false}
```

Copy `expected_generation`, including its nonce, from the session's current
ready, query or status frame; the example identity is illustrative. `receipt`
and `model_dir` must be absolute. `model_dir` is required even with an explicit
native quality backend; add `quality_backend` and `quality_model_dir` to select
that local producer. No model is downloaded. The required `quality_exact` field
must match the sealed quality tier: `true` for F32 exact retrieval built with
`--quality-exact`, or `false` for the default F32 native HNSW configuration and
seed 42. The fast tier keeps its existing precision, retrieval parameters and
seed independently.

The successor must be newer, use the same ordinary or sharded layout, and retain
every source ID, body, title and metadata field. Sharded partition boundaries may
change. An existing quality tier, changed fast producer, incompatible quality
input contract, damaged artifacts or a stale expected generation refuses the
request. Only the new quality model is loaded and wrapped with the session's
query-cache policy; the actual fast model and its warm query cache are retained.
Candidate installation checks the exact original serving snapshot, so a source
update or another activation that wins during admission is preserved.

The ready frame reports `quality_completion_enabled` and
`quality_completion_supported`. Success emits one `terminal` frame with
`operation: "activate_quality_completion"`, `status: "complete"`,
`selection_changed: true`, both generation identities and the admitted per-tier
retrieval facts. Subsequent full requests run both Initial and Refined phases,
and direct quality search becomes available. Old retained readers keep their
fast-only cohort. Failure before installation retains the serving selection;
failed acknowledgement delivery does not undo an installation.

This control admits and installs existing artifacts. It performs no document
inference, file writes or durable publication; process restart still uses the
explicit startup receipt. Library callers can prepare the same candidate through
`NativeHybridSnapshot::prepare_quality_completion_selected` or
`NativeShardedSnapshot::prepare_quality_completion_selected`, then use the
existing exact-predecessor `install` operation.

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
