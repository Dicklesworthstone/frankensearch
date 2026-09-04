# Search-quality BEIR harness (pure Python)

This is the historical Python experiment harness behind
[`SEARCH_QUALITY_FINDINGS.md`](../SEARCH_QUALITY_FINDINGS.md) and the July 2026
`search-QUALITY` entries in `../NEGATIVE_EVIDENCE.md`. It evaluates real BEIR
qrels with `model2vec` and the `rank_bm25` BM25Okapi proxy. It does not execute
Tantivy, Quill, Rust fusion, or `fsfs`; its quality deltas are not product proof.

Current Rust defaults are RRF (`TwoTierConfig::default()`), with pool-min-max
available as an option. Adaptive NQC weighting is default-on in the library
searchers after warm-up, but is not wired into `fsfs`'s separate orchestrator.
Use the existing Rust quality work under `bd-quill-e6-gauntlet-scale-rm3q.7` to
validate actual serving paths before promoting an experimental default.

## Historical reproduction environment

These scripts require a separate Python research environment with `model2vec`,
`rank_bm25`, `numpy`, and `snowballstemmer`. They are not part of the Cargo build
or its release gate. From this directory, download BEIR corpora into fresh
directories (do not overwrite existing datasets):

```bash
for ds in scifact nfcorpus arguana scidocs; do
  curl -fSL -A 'OpenAI File Downloader, XaiImageApiFetch/1.0' \
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$ds.zip" -o "$ds.zip"
  unzip -n "$ds.zip"
done
```

## Run
```bash
python3 beir_eval.py scifact nfcorpus arguana scidocs
```
Reports per corpus: nDCG@10 for the dense tier (model2vec `potion-retrieval-32M` cosine),
the lexical proxy (`rank_bm25` BM25Okapi), and their Python RRF hybrid, plus
`hybrid_ge_best`. That field is an observed comparison, not a guaranteed invariant.

## Historical baseline (2026-07-12, scifact, lowercase ASCII-alphanumeric tokenization)
dense 0.6331 · lexical 0.6523 · hybrid 0.6695 (+0.0172 over best single) · hybrid≥best ✓ —
records hybrid ≥ best single tier for that experiment, with lexical the stronger
tier. The scripts tokenize with `[a-z0-9]+`; they do not use the production
tokenizer. Stem/stop analysis and candidate-pool variants below are further
proxy experiments, not measurements of the Rust quality kernels.

## Stem/stop proxy experiment (`stem_stop.py`)

This script adds Python Snowball English stemming and a fixed stopword set to
the ASCII tokenizer and `rank_bm25` scoring. The historical result (2026-07-12,
300 SciFact queries, nDCG@10) was lexical 0.6523→0.6873 (+5.4%) and hybrid
0.6725→0.6970 (+3.6%). It measures that Python analyzer change. Matching a stemmer
name is not proof of Tantivy tokenizer/scoring parity or Quill conformance;
those require the pinned live oracle and actual Rust execution. These historical
numbers have not been rerun as part of the September documentation correction.
