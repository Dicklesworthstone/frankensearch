# Searching from Claude Code

This tutorial shows a minimal pattern for using `fsfs` from an agent workflow.

## 1) Build or refresh index

Install a semantic-capable binary and complete the model acquisition, verification,
and loader check in [Indexing a Rust project](indexing-a-rust-project.md#1-install-fsfs)
first. A model-free `--lite` installation cannot run this semantic walkthrough.

```bash
fsfs index /data/projects/frankensearch
```

The command is one-shot and returns after sealing the index, which
makes it safe for an agent subprocess to await. Use `fsfs watch <path>` only in
a deliberately supervised long-running session.

## 2) Use stream mode for agent-friendly output

`jsonl` is the easiest format for incremental parsing:

```bash
fsfs search "where is rrf fusion implemented" --stream --format jsonl
```

Each line is standalone JSON, so tools can parse line-by-line without waiting for completion.
The `fsfs.stream.query.v1` envelope identifies frames with `event`; result fields
are under `payload.item`. The `query.stream.initial_ready` and
`query.stream.refined_ready` progress reasons identify the phase of subsequent
results. A rank may appear once in each phase; the later list replaces the earlier
list. A quality failure can leave only Initial results, so consumers must also
handle progress errors and the terminal outcome.

## 3) Filter top hits in shell

```bash
set -o pipefail
fsfs search "query classification" --stream --format jsonl \
  | jq -c 'select(.event == "result") | .payload.item | {rank, path, score}'
```

For a two-tier smoke check, retain the stream from a query that has hits and
verify that both phases actually returned results. Parsing an empty selection
is not a successful retrieval check:

```bash
stream_path=$(mktemp "${TMPDIR:-/tmp}/fsfs-stream.XXXXXX")
fsfs search "where is rrf fusion implemented" --no-daemon --stream --format jsonl > "$stream_path"
jq -se '
  reduce .[] as $f ({phase: null, initial: 0, refined: 0, completed: false};
    if $f.payload.reason_code == "query.stream.initial_ready" then .phase = "initial"
    elif $f.payload.reason_code == "query.stream.refined_ready" then .phase = "refined"
    elif $f.event == "result" and .phase != null then .[.phase] += 1
    elif $f.event == "terminal" then .completed = ($f.payload.status == "completed")
    else . end)
  | .initial > 0 and .refined > 0 and .completed
' "$stream_path"
```

## 4) Pair with exact text search

Use semantic retrieval first, then `rg` in the narrowed files:

```bash
fsfs search "adaptive budgets for short keyword queries" --limit 5
rg -n "candidate_multiplier|QueryClass" crates/frankensearch-fusion
```

## 5) Capture structured artifacts

For deterministic debugging or CI logs:

```bash
fsfs search "stream protocol contract" --format json > /tmp/fsfs-search.json
```
