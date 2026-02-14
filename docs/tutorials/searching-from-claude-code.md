# Searching from Claude Code

This tutorial shows a minimal pattern for using `fsfs` from an agent workflow.

## 1) Build or refresh index

```bash
fsfs index /data/projects/frankensearch
```

## 2) Use stream mode for agent-friendly output

`jsonl` is the easiest format for incremental parsing:

```bash
fsfs search "where is rrf fusion implemented" --stream --format jsonl
```

Each line is standalone JSON, so tools can parse line-by-line without waiting for completion.

## 3) Filter top hits in shell

```bash
fsfs search "query classification" --stream --format jsonl \
  | jq -c 'select(.kind=="hit") | {rank: .rank, path: .doc_id, score: .score}'
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
