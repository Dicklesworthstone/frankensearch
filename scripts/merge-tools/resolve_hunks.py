#!/usr/bin/env python3
"""Resolve SPECIFIC conflict hunks to one side, leaving the rest conflicted.

resolve_side.py is all-or-nothing; this is its per-hunk counterpart, for files
whose hunks have different dispositions (take origin here, keep local there,
hand-write a third). Hunks not listed are left untouched and still conflicted,
so the file stays unmergeable until every hunk has been decided deliberately.

usage: resolve_hunks.py <ours|theirs> <file> [hunk_index ...]

With no hunk indices, nothing is done (use resolve_side.py if you really mean
every hunk). Hunk indices are 1-based and match conflict_stats.py's h<N>.
"""
import sys

side = sys.argv[1]
assert side in ("ours", "theirs"), side
path = sys.argv[2]
wanted = {int(a) for a in sys.argv[3:]}
if not wanted:
    sys.exit("refusing to run with no hunk indices; use resolve_side.py for all")

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
        if h in wanted:
            done += 1
            out.extend(ours if side == "ours" else theirs)
        else:
            out.extend([head] + ours + [mid] + theirs + [tail])
    else:
        out.append(lines[i])
        i += 1

missing = sorted(wanted - set(range(1, h + 1)))
if missing:
    sys.exit(f"error: file has {h} hunks; no such hunk(s): {missing}")

with open(path, "w", encoding="utf-8") as handle:
    handle.writelines(out)
print(f"resolved {done}/{h} hunks -> {side}   {path}   (left {h - done} conflicted)")
