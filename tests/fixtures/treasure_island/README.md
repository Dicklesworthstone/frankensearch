# Treasure Island — real-prose retrieval fixture

A full-length work of natural prose used to prove that lexical **and** semantic
retrieval actually work on realistic text, rather than only on the synthetic
`corpus.json` cluster fixture.

## Why this fixture exists

`tests/fixtures/README.md` carries an honest caveat about the synthetic corpus:

> For hash-based embeddings, these fixtures primarily validate pipeline
> correctness and deterministic behavior.

That is the whole problem. Pipeline-correctness fixtures pass just as happily
when the embedder has silently degraded to non-semantic FNV-1a hashing, so they
cannot detect the failure mode that made semantic search useless for downstream
consumers (see `frankensearch/tests/treasure_island_e2e.rs`).

This fixture is chosen so that the two retrieval modes can be told apart:

- **Lexical** questions have exact, rare surface forms — `Hispaniola`,
  `Ben Gunn`, `Admiral Benbow` — that BM25 must find.
- **Semantic** questions are phrased with *no meaningful lexical overlap* with
  the passage that answers them. "a marooned man left alone on an island for
  years" must retrieve the Ben Gunn chapter without sharing content words with
  it. A hash embedder cannot do this; a real sentence embedder can. The test
  asserts the gap.

## Provenance

| Field | Value |
|---|---|
| Work | *Treasure Island* |
| Author | Robert Louis Stevenson (1850–1894) |
| Source | Project Gutenberg eBook #120 |
| Source URL | `https://www.gutenberg.org/files/120/120-0.txt` |
| Retrieved | 2026-07-26 |
| Copyright | Public domain in the United States |

Project Gutenberg's own licence permits unrestricted reuse of public-domain
texts when the Project Gutenberg header/footer are removed, which they are here.
No Project Gutenberg trademark or boilerplate is retained, so this file carries
no licence obligations into the workspace.

## What was changed from the raw download

The raw file was reduced to the narrative body only:

1. Project Gutenberg header and footer boilerplate removed.
2. Title page and table of contents removed (the TOC repeats every chapter
   title and would otherwise pollute lexical scoring with duplicate headings).
3. Chapter and part headings rewritten into a canonical, machine-parseable
   form so tests can attribute a passage to a chapter:

   ```
   == PART ONE--The Old Buccaneer ==
   == CHAPTER 11 :: XI :: What I Heard in the Apple-Barrel ==
   ```

   `CHAPTER <n>` is a sequential 1..34 index assigned by position. The middle
   field is the roman numeral **as printed in the source**.

4. Runs of three or more blank lines collapsed to two.

Body prose is otherwise byte-faithful, including period spelling, em-dash
style (`--`), and curly quotation marks.

### Known source typo (deliberately preserved)

Chapter 17's heading is printed as `XXVII` in the Gutenberg source
(`120-0.txt` line 3539) where the table of contents says `XVII`. That is an
error in the source text, not in the fixture pipeline. The sequential
`CHAPTER 17` index is correct and is what tests key on; the roman numeral is
carried through verbatim rather than silently corrected.

## Shape

| Property | Value |
|---|---|
| Chapters | 34 (6 parts) |
| Lines | 7,184 |
| Bytes | 369,602 |
| SHA-256 | `be2e285d9b0fa633eac35b350490af1693e29c7bf19c54b9260ce6389fab5190` |

## Chunking

Chunking is **not** baked into this directory. The consuming test derives
passages from the raw text at run time so that the chunking rule stays visible,
reviewable, and adjustable in one place. See `chunk_book()` in
`frankensearch/tests/treasure_island_e2e.rs`.

## Ground truth

- `lexical_queries.json` — exact-term queries with the chapters that must be
  retrieved, plus terms that must return nothing.
- `semantic_queries.json` — concept/theme queries with their answering
  chapters, deliberately avoiding lexical overlap.

## Maintenance

- Treat `treasure_island.txt` as immutable. If it must change, update the
  SHA-256 above and re-verify every expected chapter in both query files.
- Keep `CHAPTER <n>` indices stable — both query fixtures reference them.
