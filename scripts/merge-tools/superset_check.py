#!/usr/bin/env python3
"""Decide whether taking one side of a conflicted file loses anything.

Builds the ours-resolved and theirs-resolved texts, then reports identifiers
(fn/struct/enum/const/static/trait/type/macro names and cfg feature gates)
that appear on the losing side but nowhere in the winning text. An empty
report means the winning side is a symbol-level superset, so taking it drops
no declared surface. A non-empty report lists exactly what must be re-applied
by hand.

usage: superset_check.py <ours|theirs> <file> [file...]
"""
import re
import sys

DECL = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+|const\s+|unsafe\s+|extern\s+\"[^\"]*\"\s+)*"
    r"(fn|struct|enum|trait|type|const|static|mod|macro_rules!)\s+([A-Za-z_][A-Za-z0-9_]*)"
)

keep = sys.argv[1]
assert keep in ("ours", "theirs")

for path in sys.argv[2:]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    sides = {"ours": [], "theirs": [], "common": []}
    i = 0
    while i < len(lines):
        if lines[i].startswith("<<<<<<< "):
            i += 1
            while not lines[i].startswith("======="):
                sides["ours"].append(lines[i])
                i += 1
            i += 1
            while not lines[i].startswith(">>>>>>> "):
                sides["theirs"].append(lines[i])
                i += 1
            i += 1
        else:
            sides["common"].append(lines[i])
            i += 1

    lose = "theirs" if keep == "ours" else "ours"
    win_text = "".join(sides["common"] + sides[keep])

    lost = []
    for line in sides[lose]:
        m = DECL.match(line)
        if m:
            name = m.group(2)
            if not re.search(r"\b" + re.escape(name) + r"\b", win_text):
                lost.append(f"{m.group(1)} {name}")

    name = path.split("/")[-1]
    if lost:
        print(f"  LOSES {len(set(lost))} symbol(s) in {name}:")
        for item in sorted(set(lost)):
            print(f"      - {item}")
    else:
        print(f"  OK superset: taking {keep} drops no declared symbol  ({name})")
