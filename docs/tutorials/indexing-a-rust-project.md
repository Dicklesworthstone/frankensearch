# Indexing a Rust project

This walkthrough indexes a Rust workspace with `fsfs`, then verifies search quality quickly.

## 1) Install `fsfs`

```bash
curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/frankensearch/main/install.sh | bash -s -- --easy-mode
fsfs version
```

## 2) Index your repository

```bash
cd /path/to/your/rust-repo
fsfs index .
```

Use JSON when you want to capture machine-readable stats:

```bash
fsfs index . --format json | jq
```

## 3) Run a few targeted searches

```bash
fsfs search "structured concurrency context propagation" --limit 5
fsfs search "Cargo feature flags and default features" --limit 5
fsfs search "how retries and backoff are implemented" --limit 5
```

## 4) Ask for an explanation when ranking surprises you

```bash
fsfs explain <result-id>
```

The result id comes from a previous search result row or JSON payload.

## 5) Recommended next step

If the repository changes constantly, switch to watch mode:

```bash
fsfs watch .
```
