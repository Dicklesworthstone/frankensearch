#!/usr/bin/env python3
"""Resolve conflict hunks by keeping BOTH sides (append-only ledgers).

usage: resolve_both.py <ours-first|theirs-first> <file> [hunk_index ...]

With no hunk indices every hunk is unioned; otherwise only the listed 1-based
hunk numbers are unioned and the rest are left conflicted for hand resolution.
"""
import sys

order = sys.argv[1]
assert order in ("ours-first", "theirs-first"), order
path = sys.argv[2]
wanted = {int(a) for a in sys.argv[3:]} or None

with open(path, encoding="utf-8") as handle:
    lines = handle.readlines()

out, i, h, done = [], 0, 0, 0
while i < len(lines):
    if lines[i].startswith("<<<<<<< "):
        h += 1
        head = lines[i]
        i += 1
        ours = []
        while not lines[i].startswith("======="):
            ours.append(lines[i])
            i += 1
        mid = lines[i]
        i += 1
        theirs = []
        while not lines[i].startswith(">>>>>>> "):
            theirs.append(lines[i])
            i += 1
        tail = lines[i]
        i += 1
        if wanted is not None and h not in wanted:
            out.extend([head] + ours + [mid] + theirs + [tail])
        else:
            done += 1
            out.extend((ours + theirs) if order == "ours-first" else (theirs + ours))
    else:
        out.append(lines[i])
        i += 1

with open(path, "w", encoding="utf-8") as handle:
    handle.writelines(out)
print(f"unioned {done}/{h} hunks ({order}) -> {path}")
