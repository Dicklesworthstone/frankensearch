# Request-scoped native search tuning

The opt-in `frankensearch-native` stdio server can accept different retrieval
policies per full-mode query without reloading models or changing its retained
cohort. Ordinary flat requests and the `fsfs` CLI are unchanged.

## Enable the controller grant and set limits

```sh
frankensearch-native serve --receipt /absolute/path/to/selection.json \
  --model-dir /absolute/path/to/models --timeout-ms 5000 \
  --full-query '{"request_overrides":true,"candidate_multiplier":6,"fast_ef":128,"quality_ef":256}'
```

`request_overrides` defaults to false. Merely supplying `--full-query` does not
permit overrides. The ready frame reports `query_tuning_enabled`; startup
`full_query` values remain the defaults for ordinary full-mode requests.
This grants search tuning only, not activation, writes, model migration, or
access outside the server's source filter.

## Send an explicitly tuned request

```jsonl
{"op":"search","request":{"id":"quick","query":"retry network requests","limit":10,"mode":"full","timeout_ms":1000},"full_query":{"candidate_multiplier":2,"fast_ef":64,"quality_ef":128,"quality_weight":0.5}}
{"id":"ordinary","query":"retry network requests","limit":10}
```

The nested `request` uses the existing query/id/mode/limit/filter/timeout fields.
`full_query` is a partial object with `candidate_multiplier`, `fast_ef`,
`quality_ef`, `rrf_k`, `lexical_weight`, `semantic_weight`, and `quality_weight`.
Missing or null fields inherit the startup values; null cannot erase a cap.
Unknown or duplicate fields and non-object tuning values are rejected rather
than silently becoming an ordinary query. A request cannot grant itself
`request_overrides` or configure model paths, source scopes, or time limits
inside the tuning object.

The requested multiplier must be positive and at most the startup multiplier.
Each explicit beam requires an explicit startup beam for that tier and cannot
exceed it. Source weights remain finite and positive, RRF K finite and
nonnegative, quality weight within 0..1, and the maximum fused score must fit
f32. Reducing quality weight to zero does not disable a required quality tier.
The existing server timeout remains a ceiling on the nested request's timeout.
The server filter and nested request filter are intersected, not replaced.

Only full-mode requests accept tuning. Explicit fast/quality-primary modes in
a tuned envelope fail; ordinary flat fast/quality requests remain available.
The existing loaded reranker and its candidate window are retained unchanged.
Candidate requests include the larger of the displayed limit and rerank window.

Started and result frames carry the effective `full_query`, not the original
partial patch. They use the same generation, sequence numbers, sharded row
coordinates, progressive refinement, degradation pages and terminal protocol
as ordinary serving. Refused tuning emits a failed terminal without inference;
subsequent requests can continue. Output failures still stop serving. The next
ordinary request uses startup defaults, never a previous request's overrides.

## Boundaries and tests

The caps bound **requested** candidate multipliers and graph beams. Native
hit-count floors, physical-row caps, selective-filter widening and tombstone
routing still apply. These are not bounds on graph visits, model memory, RSS,
or hard wall time, and are not a recall or relevance guarantee.

Regression tests are in the executable's `query::tuning::tests` and
`serve::request_tuning` modules. They exercise actual FSVI/native/Quill owners
and the stdio dispatch with synthetic identified providers. They do not replace
real-model, platform or release qualification.

```sh
cargo test -p frankensearch --features hybrid --bin frankensearch-native request_tuning
cargo test -p frankensearch --features hybrid --bin frankensearch-native query::tuning
```
