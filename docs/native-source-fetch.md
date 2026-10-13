# Fetch the retained source behind a native search result

The opt-in `frankensearch-native serve` protocol supports a read-only `fetch`
operation. It resolves document IDs through the same retained source cohort as
search, not through current files, vector row ordinals, or a new model invocation.
This works for both single and sharded hybrid cohorts without flattening shards.

## Request and response

Copy `expected_generation` from the search response whose source you need,
including the nonce. In an existing text-mode subprocess session:

```python
request = {
    "op": "fetch",
    "id": "context-1",
    "expected_generation": search_result["generation"],
    "documents": [
        {"doc_id": "src/retry.rs", "max_bytes": 8192, "include_metadata": True},
        {"doc_id": "notes.md", "offset_bytes": 4096, "max_bytes": 4096},
    ],
    "timeout_ms": 1000,
}
session.stdin.write(json.dumps(request) + "\n")
session.stdin.flush()
reply = json.loads(session.stdout.readline())
```

The reply is one terminal frame with `operation: "fetch"`, the request ordinal,
correlation ID, `seq: 0`, and the actual pinned generation. Successful replies
contain `source: "retained_snapshot"` and a `documents` array in exactly the
requested order. Duplicate IDs/windows are not deduplicated.

A found document has `status: "found"`, `doc_id`, `content`, `offset_bytes`,
`end_offset_bytes`, `total_bytes`, and `next_offset_bytes`. Content is an exact
UTF-8 slice: no normalization, trimming, HTML rendering or query highlighting.
JSON escapes are transport encoding, not modifications of the decoded text.
`include_metadata` defaults to false. When true, the retained title (when present)
and complete metadata map accompany the window, subject to the response cap.

`offset_bytes` defaults to zero and must be a UTF-8 character boundary within
that document. The end rounds down to a character boundary. `max_bytes` defaults
to 8192 and must be 4..65536; four bytes guarantee progress for any UTF-8 scalar.
Continue using the returned `next_offset_bytes`, not a character count or an
assumed fixed increment. Null means the end was reached. An offset equal to the
source length returns an empty final window; an interior-scalar or out-of-range
offset rejects the entire request rather than returning a successful prefix.

## Snapshot and scope rules

Every request requires its complete `expected_generation`. If activation, an
update, or model migration has installed a different generation, the request
fails before source lookup. It does not silently return the new text under an
old result ID. Search again against the new generation, or use a separate server
opened with the original trusted selection while its immutable files remain.
The server pins one snapshot through generation checking, all windows, encoding,
and delivery. A concurrent installation cannot mix document versions in a reply.

The fixed `serve --filter` applies to source reads too. An optional request-level
`filter` uses the same fields as search and intersects, rather than replaces,
the server scope. Missing IDs and sources outside either scope both return only
`{"status":"unavailable","doc_id":"the-requested-id"}`. They disclose no source
length, title, metadata, existence reason or content. Document IDs are never
interpreted as file paths. Fetch grants no writes, activation, model loading,
model migration, or access outside the configured source scope.

## Bounds and delivery

One request accepts 1..32 windows. IDs are nonblank, NUL-free, at most 65535 bytes
each and 64 KiB in aggregate. The regular 1 MiB stdio request ceiling also applies.
The complete encoded response, including metadata, JSON escaping, framing and
newline, is capped at 1 MiB. Large responses fail before any source bytes are
written; request fewer/smaller windows or omit metadata. Metadata and content are
borrowed from the retained snapshot during serialization, not cloned into an
unbounded JSON value before the cap is applied. Readiness reports the three
source-fetch limits explicitly.

The server's existing timeout is a ceiling on `timeout_ms`. The same captured
timer covers source lookup and encoding. Cancellation and expiry reject the
complete result; synchronous lookup/encoding is not preempted inside a call.
These limits do not bound the memory already occupied by an opened cohort.

Query/validation refusals emit one failed terminal and permit subsequent
requests. Output write or flush failure stops serving immediately: delivered
bytes are not rolled back, and no contradictory second frame is appended.
Fetching changes neither search ranking nor source/vector/Quill artifacts.
It does not change `fsfs` defaults or the durable generation authority.

## Verification

The executable tests are under `serve::source::tests`:

```sh
cargo test -p frankensearch --features quill --bin frankensearch-native serve::source
```

They cover strict requests, UTF-8 continuation, source-scope non-disclosure,
encoded output bounds, cancellation/deadlines, complete single/sharded owners,
source updates with retained old readers, and failed delivery. Synthetic
identified providers exercise real native/Quill construction; these tests are
not model-quality, performance, platform, or release qualification.
