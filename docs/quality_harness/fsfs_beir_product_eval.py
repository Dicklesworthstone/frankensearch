#!/usr/bin/env python3
"""Retrieval quality through the real fsfs product path, on a BEIR dataset.

Unlike the other scripts in this directory (Python proxies: model2vec + rank_bm25),
this drives a built `fsfs` binary through `fsfs serve`, so the numbers are the
product's: Quill BM25, the Model2Vec fast tier, the quality tier, RRF, blending
and the optional cross-encoder, exactly as shipped. Standard library only.

Steps (BEIR layout: corpus.jsonl, queries.jsonl, qrels/test.tsv):

  fsfs_beir_product_eval.py materialize --dataset D --out FILES
  fsfs index FILES --index-dir IDX --config CFG --format json
  fsfs_beir_product_eval.py ablate --fsfs BIN --dataset D --index-dir IDX --config CFG
  fsfs_beir_product_eval.py rerank --fsfs BIN --dataset D --index-dir IDX --config CFG

`ablate` compares lexical_only, the full-mode Initial payload and the Refined payload
(limit 100); `rerank` compares Refined with rerank=true at limit 10 and records request
latency (set `[search] rerank_timeout_ms` in CFG high enough for the stage to finish;
the reason codes show whether it applied). Metrics: nDCG@10 (BEIR graded formula, as
in beir_eval.py), MRR@10, Recall@100; paired bootstrap over queries, 10,000 resamples,
seed 20260922, 95% percentile interval. Decide comparisons before running; do not
retune on held-out judgments.
"""
import argparse
import json
import math
import os
import random
import subprocess
import time


def load(dataset):
    queries = {}
    for line in open(os.path.join(dataset, "queries.jsonl")):
        o = json.loads(line)
        queries[o["_id"]] = o["text"]
    qrels = {}
    with open(os.path.join(dataset, "qrels", "test.tsv")) as f:
        next(f)
        for line in f:
            q, doc, rel = line.rstrip("\n").split("\t")
            if int(rel) > 0:
                qrels.setdefault(q, {})[doc] = int(rel)
    qids = sorted(qrels, key=lambda q: (len(q), q))
    return queries, qids, qrels


def ndcg(ranked, rel, k=10):
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if rel.get(d, 0))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def mrr(ranked, rel, k=10):
    return next((1.0 / (i + 1) for i, d in enumerate(ranked[:k]) if rel.get(d, 0)), 0.0)


def recall(ranked, rel, k=100):
    return len([d for d in ranked[:k] if rel.get(d, 0)]) / len(rel)


def bootstrap(a, b, n=10000, seed=20260922):
    rnd = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    m = len(diffs)
    means = sorted(sum(diffs[rnd.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return sum(diffs) / m, means[int(0.025 * n)], means[int(0.975 * n) - 1]


class Serve:
    def __init__(self, fsfs, index_dir, config):
        self.proc = subprocess.Popen(
            [fsfs, "serve", "--index-dir", index_dir, "--config", config, "--format", "jsonl"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )

    def ask(self, request):
        started = time.time()
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise SystemExit("fsfs serve exited early")
            reply = json.loads(line)
            if "payloads" in reply or reply.get("ok") is False:
                return reply, time.time() - started

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=60)


def phase(reply, name):
    for payload in reply.get("payloads", []):
        if payload.get("phase") == name:
            return [os.path.basename(h["path"]).rsplit(".", 1)[0] for h in payload.get("hits", [])], payload
    return None, None


def report(label, runs, qids, qrels):
    n = len(qids)
    per = {
        "ndcg10": [ndcg(r, qrels[q]) for r, q in zip(runs, qids)],
        "mrr10": [mrr(r, qrels[q]) for r, q in zip(runs, qids)],
        "recall100": [recall(r, qrels[q]) for r, q in zip(runs, qids)],
    }
    print(f"{label:12} nDCG@10 {sum(per['ndcg10'])/n:.4f}  MRR@10 {sum(per['mrr10'])/n:.4f}  R@100 {sum(per['recall100'])/n:.4f}")
    return per


def compare(label, a, b):
    mean, lo, hi = bootstrap(a, b)
    verdict = "significant" if lo > 0 or hi < 0 else "not significant"
    print(f"  {label}: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] {verdict}")


def materialize(args):
    os.makedirs(args.out, exist_ok=True)
    count = 0
    for line in open(os.path.join(args.dataset, "corpus.jsonl")):
        o = json.loads(line)
        body = (o.get("title", "").strip() + "\n\n" + o.get("text", "").strip()).strip() + "\n"
        with open(os.path.join(args.out, f"{o['_id']}.txt"), "w") as f:
            f.write(body)
        count += 1
    print(f"wrote {count} documents to {args.out}")


def ablate(args):
    queries, qids, qrels = load(args.dataset)
    serve = Serve(args.fsfs, args.index_dir, args.config)
    lexical, initial, refined = [], [], []
    for q in qids:
        reply, _ = serve.ask({"query": queries[q], "limit": 100, "mode": "lexical_only"})
        lexical.append(phase(reply, "initial")[0] or [])
        reply, _ = serve.ask({"query": queries[q], "limit": 100})
        init = phase(reply, "initial")[0] or []
        initial.append(init)
        refined.append(phase(reply, "refined")[0] or init)
    serve.close()
    lex = report("lexical", lexical, qids, qrels)
    ini = report("initial", initial, qids, qrels)
    ref = report("refined", refined, qids, qrels)
    compare("nDCG@10 initial - lexical", ini["ndcg10"], lex["ndcg10"])
    compare("nDCG@10 refined - initial", ref["ndcg10"], ini["ndcg10"])
    compare("nDCG@10 refined - lexical", ref["ndcg10"], lex["ndcg10"])


def rerank(args):
    queries, qids, qrels = load(args.dataset)
    serve = Serve(args.fsfs, args.index_dir, args.config)
    serve.ask({"query": "warm up the cross-encoder", "limit": 10, "rerank": True})
    refined, reranked, codes, latency = [], [], {}, []
    for q in qids:
        reply, _ = serve.ask({"query": queries[q], "limit": 10})
        refined.append(phase(reply, "refined")[0] or phase(reply, "initial")[0] or [])
        reply, seconds = serve.ask({"query": queries[q], "limit": 10, "rerank": True})
        latency.append(seconds)
        ranked, payload = phase(reply, "refined")
        code = ((payload or {}).get("rerank") or {}).get("reason_code", "none")
        codes[code] = codes.get(code, 0) + 1
        reranked.append(ranked if ranked is not None else (phase(reply, "initial")[0] or []))
    serve.close()
    latency.sort()
    print(json.dumps({"rerank_reason_codes": codes,
                      "reranked_request_s": {"p50": round(latency[len(latency) // 2], 3),
                                             "p95": round(latency[int(0.95 * len(latency))], 3)}}))
    ref = report("refined", refined, qids, qrels)
    rr = report("reranked", reranked, qids, qrels)
    compare("nDCG@10 reranked - refined", rr["ndcg10"], ref["ndcg10"])
    compare("MRR@10 reranked - refined", rr["mrr10"], ref["mrr10"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("materialize")
    m.add_argument("--dataset", required=True)
    m.add_argument("--out", required=True)
    for name in ("ablate", "rerank"):
        p = sub.add_parser(name)
        p.add_argument("--fsfs", required=True)
        p.add_argument("--dataset", required=True)
        p.add_argument("--index-dir", required=True)
        p.add_argument("--config", required=True)
    args = parser.parse_args()
    {"materialize": materialize, "ablate": ablate, "rerank": rerank}[args.command](args)


if __name__ == "__main__":
    main()
