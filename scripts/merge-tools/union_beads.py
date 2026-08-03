#!/usr/bin/env python3
"""Union the two sides of a .beads/issues.jsonl merge conflict.

Neither side's beads are dropped. When an id appears on both sides the record
with the later `updated_at` wins; ties keep origin's (theirs), which carries
the owner's correctness work. Output is sorted by id so the result is
deterministic and diffable.
"""
import json
import sys

ours_path, theirs_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]


def load(path):
    records = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["id"]] = rec
    return records


ours = load(ours_path)
theirs = load(theirs_path)

merged = {}
conflicts = ours_win = theirs_win = 0
for bead_id in set(ours) | set(theirs):
    a, b = ours.get(bead_id), theirs.get(bead_id)
    if a is None:
        merged[bead_id] = b
    elif b is None:
        merged[bead_id] = a
    else:
        conflicts += 1
        if a == b:
            merged[bead_id] = b
            continue
        # later updated_at wins; ties -> theirs
        if (a.get("updated_at") or "") > (b.get("updated_at") or ""):
            merged[bead_id] = a
            ours_win += 1
        else:
            merged[bead_id] = b
            theirs_win += 1

with open(out_path, "w", encoding="utf-8") as handle:
    for bead_id in sorted(merged):
        handle.write(json.dumps(merged[bead_id], separators=(",", ":"), sort_keys=True))
        handle.write("\n")

only_ours = len(set(ours) - set(theirs))
only_theirs = len(set(theirs) - set(ours))
print(f"ours={len(ours)} theirs={len(theirs)} union={len(merged)}")
print(f"  only-in-ours kept : {only_ours}")
print(f"  only-in-theirs kept: {only_theirs}")
print(f"  in both: {conflicts} (ours newer {ours_win}, theirs newer/tie {theirs_win})")
assert len(merged) == len(set(ours) | set(theirs)), "union lost ids"
print("ASSERT OK: no id dropped from either side")
