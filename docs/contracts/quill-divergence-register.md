# Quill Divergence Register

**Status:** Living ledger. Machine history is append-only; corrections and
disposition changes append superseding events rather than rewriting prior
evidence.
**Owning beads:** `bd-quill-e0-contracts-j53p.4` (language policy) and
`bd-quill-e6-gauntlet-scale-rm3q.8` (campaign population and terminal census).
**Design of record:** plan §15.6. **Oracle:** tantivy `0.26.1` +
`frankensearch-lexical` (pinned by the gauntlet's version contract).

## Doctrine

Every **intentional or discovered-and-accepted** behavioral divergence between Quill and the oracle is recorded here. Two rules govern the ledger:

1. **An empty register is not the goal; an *unclassified* divergence is the only failure.** The conformance gate (G2) blocks on divergences that match no register class — never on the register being non-empty.
2. **`accept` decisions require a consumer-impact note and second-agent review** (fresh-eyes rule). `fix` decisions must name the spawned bead. Review sign-off is recorded in the entry.

The gauntlet's comparator auto-classifies against §2's classes; anything it cannot classify fails the run and lands in triage (bd-quill-duel-shrinker's factor-diff bucketing feeds this).

## 1. Machine contract and review workflow

The v1 machine contract is
[`schemas/quill-divergence-register-v1.schema.json`](../../schemas/quill-divergence-register-v1.schema.json).
Its synthetic schema-conformance fixture is
[`crates/frankensearch-quill-gauntlet/fixtures/divergence-register-v1.json`](../../crates/frankensearch-quill-gauntlet/fixtures/divergence-register-v1.json).
That fixture proves the format; it is **not** a production campaign register
and must never be cited as live mismatch evidence.

`DivergenceRegisterLedger` is the typed implementation. One immutable event
stream retains:

- exact first-seen gauntlet object address plus SHA-256 integrity digest, and
  engine, corpus, query, and generator revisions;
- minimized fixture identity, replay test, mismatch signatures, root cause,
  expected/observed behavior, consumer impact, and hashed redacted diagnostics;
- reviewed `accepted`, `fixed`, or `blocking` dispositions;
- predicted-class declarations followed by reviewed `observed` or `retired`
  revisions.

Every event has a contiguous sequence number, author, UTC timestamp, and an
optional `supersedes` link to the currently active event of the same kind and
logical ID. `validate_append_only_successor` compares complete event prefixes,
so deletion, reordering, or editing of historical evidence fails even when the
resulting JSON remains internally valid. Corrected observation evidence also
requires a later disposition event, preventing an old review from silently
blessing new evidence.

The campaign workflow is:

1. Append an observation as soon as a mismatch is emitted. Raw rank, snippet,
   and count failures may be recorded, but cannot be accepted as equivalence
   classes.
2. Append exactly one active disposition. A fix names its commit and regression
   test. An accept names an equivalence law, rationale, and reviewer independent
   of the observation author. An unresolved mismatch names an owning bead and
   remains an explicit blocker.
3. Append prediction revisions when a seeded class is forced or formally
   retired. A declaration is permitted during an active campaign, but prevents
   a terminal census.
4. Run the mismatch census over the sorted unique signatures emitted by the
   required campaigns. Missing signatures, a reappearance after `fixed`, active
   blockers, and unresolved predictions all make `flip_ready = false`;
   `require_terminal_census` fails closed on the same state.
5. Run source-sensitive canaries before commit. Diagnostics carry only a
   lowercase SHA-256 digest and a canonical `<redacted:...>` marker. The
   generated review table is derived from validated active events and never
   renders observed/query payloads.

The older `DivergenceRegistry` remains the narrow per-fixture runtime allowlist
embedded in existing campaign artifacts. It can classify only already accepted
semantic mismatches. It is not the append-only review ledger and cannot satisfy
the E6 terminal census by itself.

The following Markdown shape is the human projection retained for current
entries. The machine ledger carries the additional hashes, revisions, event
links, and disposition-specific proof described above.

```
### DIV-<NNN>: <short title>
- Class: <one of §2, or a NEW class added in the same commit>
- First seen: <date> · <suite/fixture id or shadow-generation stamp>
- Root cause: <precise mechanism, file/section refs>
- Consumer impact: <what a frankensearch user/agent could observe; "none observable" needs justification>
- Fixture: <committed fixture id that reproduces/pins the divergence>
- Decision: accept | fix (bead: <id>) | pending
- Reviewer: <second agent name + date for accepts>
```

The E6.8 bead remains open until the production ledger is populated from every
required campaign and two consecutive independent nightly receipts (different
fixed seeds and worker receipts) report no new or unclassified divergence.

## 2. Divergence classes (taxonomy)

| Class | Meaning | Default posture |
|---|---|---|
| `ScoreEpsilon` | `abs(a-b) / max(abs(a), abs(b), 1e-12) ≤ 1e-4` with identical result *sets*; rank flips only inside maximal connected components of epsilon-adjacent oracle scores, assigned in oracle total-order before inspecting Quill. Expected from segment-geometry-dependent stats and libm `ln` platform variation (see quill_contract.rs conventions: never bit-pin `ln` outputs) | accept-by-class (bounded) |
| `TieOrder` | Identical-score results differ only because Tantivy orders by ascending `DocAddress(segment_ord, segment-local doc_id)` while Quill orders by its global u32 docid. The comparator preserves native order, verifies the difference is confined to an expanded exact-score tie group (including top-k cutoff substitutions), and reports the public ordering impact; it never canonicalizes the difference away before classification. | accept-by-class |
| `SnippetWindow` | Same matched terms, different window choice on coverage ties; tags/lengths identical | accept-by-class (cosmetic) |
| `GlobExpansionLimit` | Wildcard expansion hits the bound with a different candidate subset/order than the oracle's expansion | accept per-entry (requires impact note) |
| `QueryCanonicalization` | Observable match- or score-affecting query-AST lowering differences, including parser repairs and any future score-affecting dedup (bd-quill-duel-ast-dedup). Score-neutral cursor sharing needs no entry. | fix/off by default |
| `OracleBug` | The oracle's behavior is wrong per its own documentation and Quill deliberately does not reproduce it | accept per-entry (needs upstream citation) |
| `StatsSemantics` | Deletes-vs-stats or delta-vs-sealed stats timing differences not covered by fixtures pinning oracle behavior (e0.1 row 3, e4.3/e5.2 notes) | pending → must converge to fix or a pinned accept |
| `PostingRecordSemantics` | A field's record option changes which posting details are semantically observable. `Basic` means effective tf=1 in the scorer and in every term, group, and block pruning upper bound, even when a sealed posting retains a raw occurrence count; snapshot field statistics remain independent. | fix-only; never accept |
| `UnicodeEdge` | Analyzer divergence on degenerate inputs (unpaired surrogates cannot occur in &str; this class covers e.g. exotic casing/width edge cases) proven byte-parity-impossible or oracle-inconsistent | accept per-entry |
| `OversizedQueryToken` | A >65,530-byte query token lowers to MatchNone under Quill's symmetric admission rule (ingest and query share `MAX_TERM_BYTES`), while the oracle keeps the unmatchable leaf as an empty posting list. Standalone/required/phrase oversized atoms are MatchNone; optional Should/MustNot oversized clauses are dropped only where sibling semantics are preserved. Public parser strings can never carry such a token (10,000-byte cap), so the class surfaces only on programmatic ASTs | accept-by-class (DIV-004 proof) |

Adding a class = a PR that adds the row here **and** teaches the comparator to classify it, in the same commit.

## 3. Seeded expectations (not yet observed — placeholders awaiting first evidence)

These are the classes the plan *predicts*; each becomes a numbered DIV entry when first observed with a real fixture:

- `ScoreEpsilon` from cross-platform `ln` (x86 vs Apple Silicon differential lanes).
- `TieOrder` on synthetic corpora with duplicated documents.
- `SnippetWindow` on documents where two windows have equal term coverage.
- `GlobExpansionLimit` on >limit-term dictionaries with `Complex` patterns.
- ~~A dedicated class for score-neutral oversized query-token normalization~~ — landed as `OversizedQueryToken` with the G1 comparator class and executable Boolean-shape proof (`bd-quill-e0-contracts-j53p.8`, DIV-004).

## 4. Entries

### DIV-001: standalone CASS negation loses complement semantics

- Class: `QueryCanonicalization`
- First seen: 2026-07-17 · `query-boolean-negative-standalone-universe`
- Root cause: `cass_build_boolean_query_clauses` emits a lone `MustNot` clause for `-term`; Tantivy's raw negative-only `BooleanQuery` matches nothing, while complement semantics require an `All` clause alongside the exclusion. OR-operand lifting already creates that wrapper, so the shapes disagree inside the shipping adapter.
- Consumer impact: a standalone negative CASS query returns zero hits instead of every live document not matching the excluded term. Positive `AND NOT` shapes are unaffected and must not receive an `All` scorer.
- Fixture: `query-boolean-negative-standalone-universe`
- Decision: fix completed 2026-07-18 (bead: `bd-2b2u`)
- Resolution: shipping now anchors every non-empty all-negative CASS root with `Must(All)` before filters are appended. Result-level tests pin `NOT`/`-` complements with and without filters, exact score neutrality for `MustNot`, and unchanged mixed `AND NOT` scoring; the Quill-oracle differential now treats the standalone shapes as ordinary parity cases.
- Reviewer: not required for a fix decision

### DIV-002: CASS anchored globs collapse to `AllQuery`

- Class: `OracleBug`
- First seen: 2026-07-17 · `query-glob-suffix` / `cass_parser_result_sets_match_the_shipping_tantivy_builder`
- Root cause: `CassWildcardPattern::to_regex` emits explicit `^`/`$` assertions, but pinned `tantivy-fst 0.5.0` rejects zero-width assertions and already matches regexes against the whole term. `cass_build_term_query_clauses` ignores both title/content construction failures; an empty top-level clause list then becomes `AllQuery`.
- Consumer impact: lone suffix globs such as `*bar` return every document. Affected complex wildcard operands can silently disappear from compound or filtered queries. Substring globs and complex globs bounded by `*` at both ends are unaffected by this anchor failure.
- Fixture: `query-glob-suffix` plus the result-level differential named above
- Decision: fix completed 2026-07-18 (bead: `bd-cass-wildcard-fst-anchors-t3f9`)
- Resolution: shipping suffix and complex globs now emit anchor-free FST regexes, and regex construction errors propagate to one explicit match-none root instead of silently removing an operand. Real Tantivy tests cover title/content suffixes, substrings, complex whole-term boundaries, compound operands, filters, and forced construction failure; the Quill-oracle result differential now treats the former DIV-002 shapes as ordinary parity cases.
- Reviewer: not required for a fix decision

### DIV-003: same-position phrase terms are alternatives

- Class: `QueryCanonicalization`
- First seen: 2026-07-18 · `phrase_same_position_alternatives_are_reviewed_or_divergence`
- Root cause: Quill groups analyzed phrase terms with the same query position into one OR slot, as required by the `HyphenDecompose` language contract. Pinned Tantivy instead puts every equal-offset term in its phrase intersection, so all of them must occur at that position.
- Consumer impact: a phrase produced by an analyzer that emits alternatives at one position can match a document containing any one alternative in Quill, while Tantivy requires every alternative. Ordinary phrases with one term per position are unaffected. Quill's fixed phrase weight still sums IDF in original term order, including every alternative.
- Fixture: `phrase_same_position_alternatives_are_reviewed_or_divergence`
- Decision: accept
- Reviewer: `/root/e45_algo_review` · 2026-07-18

### DIV-004: oversized query tokens lower to MatchNone (symmetric admission)

- Class: `OversizedQueryToken` (class added in the same commit as this entry)
- First seen: 2026-07-18 · `oversized_token_boolean_shapes_are_result_equivalent` (executable proof, argus)
- Root cause: Quill applies one admission rule on both sides of the engine — analyzed tokens longer than 65,530 bytes are dropped at ingest (`analyze_admitted`) and therefore can never name a dictionary entry. On the query side the same rule lowers an oversized atom to `Query::Empty` (MatchNone): standalone, required-conjunct, and phrase-member oversized atoms are all unsatisfiable. The scorer then shorts `Must(Empty)` to MatchNone and drops empty `Should`/`MustNot` clauses only where sibling semantics determine results (`boolean_with_mode`). The oracle keeps the same unmatchable leaf as an empty posting list; tantivy never matches it either.
- Consumer impact: none observable for public query strings — the 10,000-byte parser cap (`MAX_QUERY_LENGTH`) truncates long before a 65,530-byte token can occur, so the class is reachable only through programmatic AST construction. Result sets and per-hit scores are bit-identical to the oracle shape under the exhaustive Boolean-shape proof (all 258 scored clause combinations of length ≤ 3 over {matchable, oversized} × {Must, Should, MustNot}, plus the unscored doc-set shapes).
- Fixture: `oversized_token_boolean_shapes_are_result_equivalent` + `oversized_token_unscored_shapes_are_result_equivalent` (argus scorer-level proof); `public_query_strings_cannot_carry_oversized_tokens` + `oversized_standalone_and_phrase_atoms_lower_to_match_none` + `oversized_clauses_keep_boolean_sibling_semantics` (parser pins); comparator classification via `EngineObservation.ast_differences` with `AstLoweringKind::OversizedQueryToken`
- Decision: accept
- Reviewer: PeachStone (author) · second-agent sign-off requested via agent mail 2026-07-18

### DIV-005: non-finite query boosts recover or fail closed

- Class: `QueryCanonicalization`
- First seen: 2026-07-18 · `overflowing_boost_recovers_without_nonfinite_score`
- Root cause: the pinned grammar parses digit-only boosts as `f64`, can retain positive infinity, and later casts the factor to `f32`. Quill's parser and Argus scoring boundary instead require finite non-negative `f32` weights so result ordering, serialization, and exact-score comparisons never admit NaN or infinity.
- Consumer impact: a syntactically valid factor too large for finite `f32` is diagnosed and the branch executes unboosted instead of producing infinite oracle scores. Finite factors are unchanged. If individually finite nested factors overflow only when multiplied, Quill rejects the query deterministically rather than executing a non-finite scorer; matching semantics are otherwise unchanged.
- Fixture: `overflowing_boost_recovers_without_nonfinite_score` (parser recovery and syntax-key pin); `malformed_scoring_inputs_fail_without_reaching_idf_assertion` (Argus finite-score boundary)
- Decision: accept
- Reviewer: `/root/g1a_parser_dedup_review` · 2026-07-18

---

*Cross-references: comparator classes implemented in the gauntlet kernel (bead e0.5); auto-triage feeding this ledger (bd-quill-duel-shrinker); statistical gates consuming per-class pass rates (bead e6.6); G2 exit requires this register complete over two consecutive nightly runs (bead e6.8).*
