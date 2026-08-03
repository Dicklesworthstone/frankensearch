#!/usr/bin/env python3
"""Summarize each conflict hunk: line counts and first line of each side.

Lets a large merge be triaged fast: hunks where one side is empty are pure
additions; hunks where both sides carry content need real reading.
"""
import sys

for path in sys.argv[1:]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    print(f"===== {path}")
    i = h = 0
    empty_ours = empty_theirs = both = 0
    while i < len(lines):
        if lines[i].startswith("<<<<<<< "):
            h += 1
            start = i + 1
            i += 1
            ours = []
            while i < len(lines) and not lines[i].startswith("======="):
                ours.append(lines[i])
                i += 1
            i += 1
            theirs = []
            while i < len(lines) and not lines[i].startswith(">>>>>>> "):
                theirs.append(lines[i])
                i += 1
            o, t = len(ours), len(theirs)
            if o == 0:
                empty_ours += 1
                kind = "ADD-theirs"
            elif t == 0:
                empty_theirs += 1
                kind = "ADD-ours"
            else:
                both += 1
                kind = "BOTH"
            osnip = ours[0].strip()[:58] if ours else ""
            tsnip = theirs[0].strip()[:58] if theirs else ""
            print(f"  h{h:<3} L{start:<6} {kind:<11} ours={o:<4} theirs={t:<4} | {osnip!r:60} | {tsnip!r}")
        i += 1
    print(f"  -> {h} hunks: {empty_ours} theirs-only, {empty_theirs} ours-only, {both} both-sides")
