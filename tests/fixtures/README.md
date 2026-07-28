# frankensearch Shared Test Fixtures

This directory contains deterministic fixture data for workspace and cross-crate tests.

## Files

- `corpus.json`: Document corpus used by indexing/search tests.
- `relevance.json`: 20-query ground-truth mapping (`query` -> expected top-10 doc IDs).
- `queries.json`: Extended query set with `query_class` annotations (26 queries),
  including a single-token known-miss control.
- `edge_cases.json`: Canonicalization and query edge-case inputs.
- `treasure_island/`: A full public-domain novel plus lexical and semantic
  ground truth, for retrieval tests that must distinguish *real* semantic
  behavior from a degraded fallback. See the caveat below and that directory's
  own `README.md`.

## Corpus Layout

`corpus.json` uses:

- `doc_id`
- `title`
- `content`
- `created_at`
- `doc_type`
- `metadata.word_count`
- `metadata.reading_level`
- `metadata.language`

Current corpus composition:

- Core set: 100 documents across 5 clusters (`rust`, `ml`, `sysadmin`, `cooking`, `mixed`), 20 each.
- Supplemental set: 20 additional machine-wide style documents required by
  extended query fixtures:
  - `adversarial`: 14
  - `code`: 3
  - `config`: 2
  - `log`: 1

Total documents: 120.

## Ground Truth Notes

- `relevance.json` is the baseline 20-query fixture from bead `bd-3un.38`.
- `queries.json` is an extended parallel fixture used by newer test scenarios.
- All IDs referenced by both files are present in `corpus.json`.

## Hash Embedder Caveat

Relevance judgments are most meaningful for semantic models (for example, Model2Vec and MiniLM). For hash-based embeddings, these fixtures primarily validate pipeline correctness and deterministic behavior.

**This caveat has teeth, and it cost us a production bug.** A fixture that only
validates pipeline correctness passes just as happily when the embedder has
silently degraded to non-semantic FNV-1a hashing — which is exactly what happened
to a downstream consumer whose semantic search returned lexical-only results for
a long time without a single failing test (`bd-a6zt`).

Use `treasure_island/` for anything that needs to prove semantic retrieval is
*actually working*. Its queries are phrased to have no lexical overlap with the
passages that answer them, and the consuming test asserts the **gap** between a
real sentence embedder and a hash-embedder control. That comparison is what makes
degradation detectable; the synthetic corpus above cannot see it.

## Maintenance

- Keep fixture IDs stable.
- Prefer additive changes and avoid renaming existing IDs unless tests are migrated in the same commit.
- If a query file references new IDs, add matching corpus documents in the same change.
