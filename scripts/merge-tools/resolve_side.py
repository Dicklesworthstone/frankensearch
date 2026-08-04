#!/usr/bin/env python3
"""Resolve conflict regions in a file by taking one side.

Only for hunks that have been individually verified (e.g. one side is a proven
superset of the other). Prints how many hunks were resolved so the count can be
checked against the expected number.

usage: resolve_side.py <ours|theirs> <file> [file...]
"""
import sys

side = sys.argv[1]
assert side in ("ours", "theirs"), side

for path in sys.argv[2:]:
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()

    out, i, hunks = [], 0, 0
    while i < len(lines):
        if lines[i].startswith("<<<<<<< "):
            hunks += 1
            i += 1
            ours = []
            while not lines[i].startswith("======="):
                ours.append(lines[i])
                i += 1
            i += 1  # skip =======
            theirs = []
            while not lines[i].startswith(">>>>>>> "):
                theirs.append(lines[i])
                i += 1
            i += 1  # skip >>>>>>>
            out.extend(ours if side == "ours" else theirs)
        else:
            out.append(lines[i])
            i += 1

    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(out)
    print(f"{hunks:3d} hunks -> {side}   {path}")
