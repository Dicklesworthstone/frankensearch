#!/usr/bin/env python3
"""Code-search quality through the real fsfs product path, judged by this repo's history.

A query is a closed bead's title (natural language written before the fix); its relevant
documents are the Rust files changed by commits whose message names that bead. Beads touching
more than six files, titles under five words, and bead/GH identifiers in the text are dropped.
`code_queries.jsonl` is the frozen set used for the September 2026 fsfs ranking work (253
queries, built 2026-09-25; every relevant path is tracked at be75d713). Standard library
only; serving, metrics and the paired bootstrap come from fsfs_beir_product_eval.py.

  fsfs_code_product_eval.py materialize --rev REV --out TREE
  fsfs index TREE --index-dir IDX --config CFG --format json
  fsfs_code_product_eval.py ablate --fsfs BIN --tree TREE --index-dir IDX --config CFG
  fsfs_code_product_eval.py compare --tree TREE --arm LABEL BIN IDX CFG --arm ...
  fsfs_code_product_eval.py build --out QUERIES    # regenerate the set from history

`materialize` writes the Rust files tracked under crates/ and frankensearch/ at REV.
`ablate` compares lexical_only, the full-mode Initial payload and Refined (limit 100).
`compare` pairs every arm with the first; each arm names its binary, so the same call serves
a two-binary A/B on one index and a one-binary comparison of indexes. When a change should be
neutral somewhere, add that control arm (or the baseline twice as an A/A) and require 0.0000.
Relevance is binary; nDCG@10, MRR@10 and Recall@100 as in fsfs_beir_product_eval.py. Set
`[search] quality_timeout_ms` high so host load does not decide which queries refine.
"""
import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsfs_beir_product_eval as product  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FROZEN = os.path.join(HERE, "code_queries.jsonl")
BEAD = re.compile(r"\bbd-[a-z0-9]+(?:\.[0-9]+)*\b")


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], check=True, capture_output=True, text=True).stdout


def rust_files(rev):
    listed = git("ls-tree", "-r", "--name-only", rev, "--", "crates", "frankensearch").split("\n")
    return [path for path in listed if path.endswith(".rs")]


def materialize(args):
    files = rust_files(args.rev)
    for path in files:
        target = os.path.join(args.out, path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        blob = subprocess.run(["git", "-C", ROOT, "show", f"{args.rev}:{path}"],
                              check=True, capture_output=True).stdout
        with open(target, "wb") as handle:
            handle.write(blob)
    print(f"{len(files)} files at {git('rev-parse', '--short', args.rev).strip()} -> {args.out}")


def build(args):
    corpus = set(rust_files(args.rev))
    touched = {}
    log = git("log", "--format=%x01%H%x00%s%n%b%x00", "--name-only", args.rev)
    for record in log.split("\x01")[1:]:
        head, _, rest = record.partition("\x00")
        message, _, files = rest.partition("\x00")
        changed = {path for path in files.split("\n") if path in corpus}
        for bead in set(BEAD.findall(head + "\n" + message)):
            touched.setdefault(bead, set()).update(changed)
    titles = {}
    for line in open(os.path.join(ROOT, ".beads", "issues.jsonl")):
        bead = json.loads(line)
        if bead.get("status") == "closed":
            titles[bead["id"]] = bead["title"]
    rows = []
    for bead, files in sorted(touched.items()):
        if bead not in titles or not 1 <= len(files) <= 6:
            continue
        query = BEAD.sub(" ", titles[bead])
        query = re.sub(r"\b(GH|gh)\s*#?\d+\b|#\d+\b", " ", query)
        query = re.sub(r"^\s*(\[[^\]]*\]\s*)+", "", query)
        query = re.sub(r"\s+", " ", query).strip(" :-")
        if len(query.split()) >= 5:
            rows.append({"id": bead, "query": query, "relevant": sorted(files)})
    with open(args.out, "w") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    print(f"{len(rows)} queries over {len(corpus)} files")


def load(path):
    rows = [json.loads(line) for line in open(path)]
    return rows, [row["id"] for row in rows], {row["id"]: {p: 1 for p in row["relevant"]} for row in rows}


def ranked(reply, phase, tree):
    for payload in reply.get("payloads", []):
        if payload.get("phase") == phase:
            return [os.path.relpath(hit["path"], tree) if os.path.isabs(hit["path"]) else hit["path"]
                    for hit in payload.get("hits", [])]
    return None


def ablate(args):
    rows, qids, qrels = load(args.queries)
    serve = product.Serve(args.fsfs, args.index_dir, args.config)
    lexical, initial, refined = [], [], []
    for row in rows:
        reply, _ = serve.ask({"query": row["query"], "limit": 100, "mode": "lexical_only"})
        lexical.append(ranked(reply, "initial", args.tree) or [])
        reply, _ = serve.ask({"query": row["query"], "limit": 100})
        initial.append(ranked(reply, "initial", args.tree) or [])
        refined.append(ranked(reply, "refined", args.tree) or initial[-1])
    serve.close()
    lex = product.report("lexical", lexical, qids, qrels)
    ini = product.report("initial", initial, qids, qrels)
    ref = product.report("refined", refined, qids, qrels)
    product.compare("nDCG@10 initial - lexical", ini["ndcg10"], lex["ndcg10"])
    product.compare("nDCG@10 refined - initial", ref["ndcg10"], ini["ndcg10"])
    product.compare("nDCG@10 refined - lexical", ref["ndcg10"], lex["ndcg10"])


def compare_arms(args):
    rows, qids, qrels = load(args.queries)
    scored = []
    for label, fsfs, index_dir, config in args.arm:
        serve = product.Serve(fsfs, index_dir, config)
        initial, refined, missing = [], [], 0
        for row in rows:
            reply, _ = serve.ask({"query": row["query"], "limit": 100})
            initial.append(ranked(reply, "initial", args.tree) or [])
            phase = ranked(reply, "refined", args.tree)
            missing += phase is None
            refined.append(phase if phase is not None else initial[-1])
        serve.close()
        print(f"{label}: {missing} of {len(rows)} queries had no Refined payload")
        scored.append((label, product.report(f"initial {label}", initial, qids, qrels),
                       product.report(f"refined {label}", refined, qids, qrels)))
    base_label, base_initial, base_refined = scored[0]
    for label, initial, refined in scored[1:]:
        for name, arm, base in (("initial", initial, base_initial), ("refined", refined, base_refined)):
            product.compare(f"nDCG@10 {name} {label} - {base_label}", arm["ndcg10"], base["ndcg10"])
            product.compare(f"R@100 {name} {label} - {base_label}", arm["recall100"], base["recall100"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("materialize")
    m.add_argument("--rev", default="HEAD")
    m.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--rev", default="HEAD")
    b.add_argument("--out", required=True)
    a = sub.add_parser("ablate")
    for flag in ("--fsfs", "--tree", "--index-dir", "--config"):
        a.add_argument(flag, required=True)
    a.add_argument("--queries", default=FROZEN)
    c = sub.add_parser("compare")
    c.add_argument("--tree", required=True)
    c.add_argument("--queries", default=FROZEN)
    c.add_argument("--arm", nargs=4, action="append", required=True,
                   metavar=("LABEL", "FSFS", "INDEX_DIR", "CONFIG"),
                   help="repeat; the first arm is the baseline every other arm is paired with")
    args = parser.parse_args()
    if args.command == "compare" and len(args.arm) < 2:
        parser.error("compare needs at least two --arm")
    {"materialize": materialize, "build": build, "ablate": ablate, "compare": compare_arms}[args.command](args)


if __name__ == "__main__":
    main()
