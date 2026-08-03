# INTERNAL defect documentation: tantivy PhraseScorer illegal seek on negated absent phrase

**OWNER RULING (2026-08-03, final): upstream filing DECLINED. Nothing is
submitted to the tantivy repository — not this document, not a minimal
reproducer, not a comment. Do not file, do not ask again.** This document is
retained solely as internal documentation of the oracle defect for anyone
operating the shipping tantivy lanes (`lexical-tantivy`/`cass-compat`/
`shadow-oracle`). Internal tracking: `bd-nqeb4`.

Standing remediation posture (owner-ruled): the tantivy lanes' results for
the `term NOT "absent phrase"` shape remain UNTRUSTED in release builds; the
`search_guarded` catch_unwind containment and the bsjw campaign exclusion
stay in force; the only in-repo release-build fix available is the defensive
Scorer wrapper (unsized follow-up on `bd-nqeb4`, only if the owner requests
it). The Quill migration removes the exposure wholesale.

---

**Title:** PhraseScorer: illegal seek after termination on excluded absent
phrase — `debug_assert` only, so release builds silently misexecute

## Summary

A boolean query that combines a positive term with a negated phrase whose
exact sequence does not occur in the corpus — the shape
`term NOT "phrase whose exact sequence is absent"` (every phrase term exists
individually; the sequence does not) — makes `PhraseScorer` seek a docset
that has already terminated.

In **debug builds** this panics:

```
target (3) should be greater than or equal to doc (2147483647)
```

at `src/query/phrase_query/phrase_scorer.rs:534` (tantivy 0.26.1; the same
assertion still exists on `main` as of 2026-08-02, in the seek path). Because
the guard is a `debug_assert!` and no code otherwise protects seeking after
`TERMINATED`, **release builds compile the check out and execute the illegal
seek silently** — the concerning half: production deployments do not crash,
they can return wrong results for this query shape.

## Reproduction

Index any corpus where the words of a phrase all occur but never as the exact
sequence, then execute a boolean query with one positive term clause and one
`MustNot` phrase clause for that absent sequence. Original crasher from our
differential campaign: `generic NOT "indexes Parser or minimal"` (all terms
present in the corpus; the 4-token sequence absent).

Mechanism: the phrase's terms each have posting lists, so the phrase docset
is constructed, but the intersection is empty and the docset terminates
immediately (doc = `TERMINATED` = `2147483647`). The exclusion path then
seeks the terminated docset to a real doc id, violating the seek precondition
the `debug_assert` documents.

## Versions

- Reproduced on tantivy `0.26.1` (pinned in our conformance harness).
- `0.27.0` changelog contains no phrase/seek/panic fix after `0.26.1`.
- The assertion is still a bare `debug_assert!` on `main` (checked
  2026-08-02); nothing guards the post-termination seek.

## Why we think this matters beyond the panic

We maintain a differential conformance harness that executes large generated
query corpora against tantivy; this shape was found within ~200 structured
queries, so real users hitting it is plausible. The failure mode differs by
profile: debug/CI crashes loudly; release returns silently wrong results —
the worse of the two. A `catch_unwind` boundary on the caller's side (which
we ship) contains the debug-build panic but cannot detect the release-build
misexecution.

## Suggested fix direction

Either clamp/guard the exclusion path's seek when the child docset is already
`TERMINATED` (seek on a terminated docset becoming a no-op returning
`TERMINATED` matches the documented docset contract), or promote the
`debug_assert` to a real branch so release builds take the safe path.

We're happy to provide the harness query corpus or a minimal standalone
reproducer if useful.
