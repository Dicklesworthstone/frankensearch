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

The current v2 machine contract is
[`schemas/quill-divergence-register-v2.schema.json`](../../schemas/quill-divergence-register-v2.schema.json).
Its synthetic schema-conformance fixture is
[`crates/frankensearch-quill-gauntlet/fixtures/divergence-register-v2.json`](../../crates/frankensearch-quill-gauntlet/fixtures/divergence-register-v2.json).
That fixture proves the format; it is **not** a production campaign register
and must never be cited as live mismatch evidence. The historical
[`v1 schema`](../../schemas/quill-divergence-register-v1.schema.json) and
[`v1 fixture`](../../crates/frankensearch-quill-gauntlet/fixtures/divergence-register-v1.json)
remain unchanged for archival decoding tests only. A v1 ledger is explicitly
nonadmissible: it cannot produce a ledger hash, census, append-only successor
proof, review table, or flip-ready claim.

`DivergenceRegisterLedger` is the typed implementation. One immutable event
stream retains:

- the exact first-recorded gauntlet witness case and object address as
  `{scheme: "frankensearch-quill-gauntlet/artifact-object/v<N>/sha256", object_schema_version: <N>, digest: <64 lowercase hex>}`.
  Three generations are admitted, `N` in `{7, 8, 9}`, and the scheme pins the version — a scheme
  carrying another generation's `object_schema_version` is refused. v7 and v8 addresses are
  RETAINED, not reissued: the witnesses already in the ledger keep them and their bytes are
  unchanged, while a fresh mint records a v9 address (bd-4oiwf);
- canonical producer-build identity SHA-256, oracle dependency-contract
  SHA-256, historical lexical-contract audit revision, corpus and query
  manifest SHA-256s, query-suite source kind, and query-source identity
  SHA-256;
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

Observation class corrections remain forbidden except for one typed evidence-generation migration:
the same minimized fixture may move from a v8 `RankMismatch` witness to a v9 `OracleBug` witness.
The v8 artifact could not carry DIV-010's cross-case control and therefore correctly failed closed;
v9 can carry and re-derive that control. The predicate fixes both generations, both classes, the
first-recorded case, and every fixture-evidence field. Wrong-generation, changed-fixture, and every
other class transition remain rejected, while the external witness join proves the v9 object itself
actually earned `OracleBug` (bd-4oiwf).

Schema v2 deliberately removes v1's separate subject/oracle Git revisions.
Those fields could describe historical source snapshots but could not identify
the executable that produced an object or the exact dependency contract used
by its oracle. The producer-build and oracle-dependency digests are opaque,
domain-separated SHA-256 identities owned by their respective gauntlet
contracts; the lexical audit revision is retained only as historical contract
provenance. A v2 observation carrying the old 16-hex XXH3 address or the old
revision pair fails closed, as does a v1 record carrying the v2 shape.

There are deliberately two validation levels. `DivergenceRegisterLedger::validate`
checks the self-contained v2 shape, append-only history, review policy, and
canonical field bounds. It cannot authenticate an externally named object.
Evidence admission additionally requires
`validate_relational_integrity_against_witnesses`, which first runs each
witness's stored-evidence integrity contract and then compares the complete
closed binding: object schema version, hash-domain scheme, object digest,
producer-build identity, oracle-dependency identity, lexical-contract audit
revision, corpus/query manifests, query-suite source kind and identity,
first-recorded case ID, rank class, divergence class, and mismatch signatures.
Every observation event, including superseded history, must bind to the object
it actually references, so a register carrying N observations is joinable only
against N witnesses — one per observation, with an unreferenced witness refused
just as loudly as a missing one (bd-dxedq). A witness is presented as
`DivergenceWitness::Current` when it can reproduce its own bytes, or
`DivergenceWitness::Retained` when it is committed evidence addressed FROM
those bytes; a register that spans a generation boundary needs both forms in
one join, which is why the live register carries a v7 witness for DIV-008, a
v8 witness for DIV-009, and both the retained v8 and superseding v9 witnesses
for DIV-010. A multi-class object must be covered exactly; a
missing, extra, duplicate, or substituted class/signature claim fails.
Corrections for the same divergence may repeat the same claim, but a second
divergence ID may not claim it. Stored-evidence validation is used rather than
current-producer validation so an authentic historical v7 object does not age
out when the compiled producer or oracle revision advances.

The first-recorded campaign case and the minimized regression fixture are
deliberately separate identities: minimization may produce a new fixture ID.
`validate_fixture_content_witnesses` proves only exact byte availability and
SHA-256 equality for those minimized fixtures. It rejects missing, extra,
duplicate, or same-ID/different-hash witnesses. A content hash is not evidence
that the named regression test exists, replayed, or passed.

Both v2 case/fixture identifiers retain the gauntlet's 1,024-byte
query-identifier budget and are canonical printable ASCII with no leading or
trailing spaces. Rust's byte limits and JSON Schema's character limits are
therefore identical rather than silently disagreeing on multibyte input. The
archived v1 decoder preserves its historical Unicode/byte bounds exactly.

There is intentionally **no public terminal-census or flip-ready API** in v2.
The only census is a private, test-only structural projection over raw
signature slices, explicitly nonadmissible because callers could omit or
cherry-pick campaign results. Terminal admission remains unavailable until
report v7 exposes a store-verified, complete campaign matrix that derives both
ordinary and lexical mismatches and carries typed regression/pass/replay and
class-specific retirement receipts. Lexical mismatches are fix-only and can
never be waived through a Divergence Register class.

The seeded prediction policy is frozen and domain-hashed as
`quill-divergence-predictions-v1`; its ordered required set is
`ScoreEpsilon`, `TieOrder`, `SnippetWindow`, `GlobExpansionLimit`,
`StatsSemantics`, and `OversizedQueryToken`. A shape-valid `Retired` event is
archival state, not terminal proof; absent a future typed class-specific report
receipt it remains unresolved.

The campaign workflow is:

1. Append an observation as soon as a mismatch is emitted. Raw rank, snippet,
   and count failures may be recorded, but cannot be accepted as equivalence
   classes.
2. Append exactly one active disposition. A fix names its commit and regression
   test. An accept names an equivalence law, rationale, and reviewer independent
   of the observation author. An unresolved mismatch names an owning bead and
   remains an explicit blocker.
3. Append prediction revisions when a seeded class is forced or has an
   archival retirement proposal. `Declared` and unverified `Retired` states
   remain unresolved.
4. Bind every observation event to its store-verified v7 object and validate
   the exact multi-class signature coverage. Independently validate every
   minimized fixture's bytes. Neither check is a terminal release claim.
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
required campaign, report v7 supplies the missing verified terminal evidence,
and two consecutive independent nightly receipts (different fixed seeds and
worker receipts) report no new or unclassified divergence.

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
- `StatsSemantics` on delete/delta/sealed field-stat transitions.
- ~~A dedicated class for score-neutral oversized query-token normalization~~ — landed as `OversizedQueryToken` with the G1 comparator class and executable Boolean-shape proof (`bd-quill-e0-contracts-j53p.8`, DIV-004).

## 4. Entries

**WHICH OF THESE ENTRIES THE MACHINE ENFORCES** (bd-dxedq). The entries below are rendered
identically, and they are not equally binding. Only an entry with events in the committed ledger
[`fixtures/divergence-register-v2-live.json`](../../crates/frankensearch-quill-gauntlet/fixtures/divergence-register-v2-live.json)
is governed by `DivergenceRegisterLedger` — its class, its witness address, its revision set and
its reviewer are checked by code, and its accept had to satisfy
`DivergenceDisposition::validate`. Everything else on this page is an assertion in prose that
nothing verifies.

| Entry | Machine state |
| --- | --- |
| DIV-008 | **ENFORCED** — observation seq 1, blocking seq 2, `fixed` seq 3 (supersedes 2) |
| DIV-009 | **ENFORCED** — observation seq 4, reviewed `accepted` seq 5 |
| DIV-010 | **ENFORCED** — retained v8 `RankMismatch` observation seq 6 and `blocking` seq 7; appended v9 `OracleBug` observation seq 8 (supersedes 6) and reviewed `accepted` seq 9 (supersedes 7) |
| DIV-001 … DIV-007 | **PROSE-ONLY** — recorded before the typed ingestion contract existed; their first-seen artifacts were never retained, so they cannot be reconstructed after the fact |

A prose-only entry is not thereby wrong — DIV-001 through DIV-007 all cite regression tests that
exist and pass. It is *unverified by the mechanism this page describes*, and the point of the table
is that a reader can tell the difference without reading the ledger. DIV-010 demonstrates the
append-only correction state: its earlier, weaker machine conclusion remains visible even though a
new evidence generation now supports the reviewed decision.

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

### DIV-006: Basic posting fields used raw repeated term frequency

- Class: `PostingRecordSemantics`
- First seen: 2026-07-27 · strict-remote CASS nightly against subject `c0025dfaf3aafa10d49fec768e7a39bf4f7f85fd`; mismatch signature `15c251dd6671f6766082fc47e653cda26b72cb7b6e1b642ba4b795fb906133ec`, pointer `/comparison/subject/hits/0`, oracle object `tests/fixtures/quill_language_contract.json#39@41091897`, subject object `tests/fixtures/quill_language_contract.json#39@411532ef`
- Root cause: the CASS `content_prefix` field uses Tantivy's `Basic` record option, so repeated edge n-grams are observable only as document presence and BM25 must use effective tf=1. Quill retained the raw sealed-posting occurrence count and fed tf=7 to both scoring and pruning bounds. On the minimized 40-document witness (`df=16`, target field length 1,775, corpus field-token total 84,416), the predicted raw-vs-presence score delta was 0.756431132 and the observed delta was 0.756431579 (residual `4.47e-7`), excluding score epsilon as the cause.
- Consumer impact: the inflated Basic-field score produced a stable top-hit rank mismatch in `boolean-cass`, `boolean-cass-and`, `boolean-cass-or`, `contract-harvest-boolean`, `contract-harvest-glob`, `glob-exact`, `glob-prefix`, and `range-to`. Frequency- and position-bearing fields remain frequency-sensitive.
- Fixture: `tests/fixtures/quill_language_contract.json#39` (SHA-256 `633e614b13fc01365c0dea1cb1751bfe6b98b33d28efcc954f4e42b98a657f43`); executable regressions `basic_record_option_clamps_repeated_edge_ngram_frequency`, `basic_record_option_scores_and_bounds_repeated_delta_occurrences_as_presence`, `positionless_basic_scoring_uses_presence_for_repeated_edge_ngrams`, and `positionless_basic_scoring_survives_multiple_segments_and_reopen`, plus the Basic variants of `grouped_max_score_matches_exhaustive_and_prunes`, `randomized_maxscore_matches_exhaustive_for_pinned_k_matrix`, and `randomized_block_max_wand_matches_exhaustive_for_pinned_k_matrix`
- Decision: fix completed 2026-07-27 (bead: `bd-basic-field-tf-parity-y8cb`; source commit: `d95f1614130a3dc71ef7b63f828b009a68ca3ac0`)
- Resolution: query lowering now carries the schema record option into `TermScorer`. One effective-frequency policy clamps `Basic` to presence tf=1 for exhaustive scoring, cached term maxima, current block maxima, MaxScore/WAND, grouped execution, and Delta cursors; `WithFreqs` and `WithFreqsAndPositions` retain raw tf. The strict-remote nightly lane ran as RCH job `29949672659354095` on `hz2` with test-binary SHA-256 `0b840262a38e6e0f9de2bff499b264c3280b1ec6aadb9bcb485e0aa0c5d54a8b`, clean subject/oracle provenance, and zero mismatches in all eight default/CASS generated/repository first/replay reports. Generated corpus/query hashes were `24f68d1c24610581ada06ecfcda9d8e3b61a82502b16b10fbb788d2600a68358` / `933af603feec3950a7ed6e974b97bf82e51321195b5a89efd1f74771029063f6`; the complete 40-path repository hashes were `86e29bd008426ec1f793d6756347e6c60af4c9bf44fba75955b7745a768ccde6` / `23c7989a12b8f34f897b55700d6540755d371d7197de31a33d7bedff7338d93c`. First/replay report/reservation SHA-256 pairs were CASS generated `7754bd26269a7b4d47c2001b9960a8fa392d26e0994aabd5be9ef696ecddaebe` / `ea2de990f8f28d4b7a609a6f542d765b870ae387b4a1347c012448e15a616cdc`, CASS repository `cf30c92dfee1546664ec9b157a31ae65bc9541151071415af93de1a0be89a26b` / `8938c10a7aa211f514dff2e7bf6d6d8b36635748a9b97337b49e75c7b2b24a64`, default generated `88318f329f883b2dae32b9ee905aeed93973597225eb437999852eae764df9e1` / `9a09710bc60bcb097cbca5ed6dd21479d5581c6f4aa8fd901e130047252d7748`, and default repository `d5aaedcb79cb0299b95d77bb9e62bdee8e2b737a77ed9e6f004aeeebc8851c53` / `b1c4f91d851cd47ce8afbe895287136f12b67c70e178cbecd01ddcecc3098df9`.
- Reviewer: not required for a fix decision

### DIV-007: fused multi-field term scoring diverges from oracle summation association at ULP scale

- Class: `ScoreEpsilon` (typed reason `SummationAssociation`, bound ≤ 2 ULP — ULP-based, not the relative-1e-4 form; reason added by owner ruling on bd-55mvg)
- First seen: 2026-08-02 · bd-bsjw structure-aware campaign round 4 (bd-55mvg); reproduced at eight leaves, depth-1 grouping, pure Should, no boosts, no negation
- Root cause: Quill's Term scorer fuses each unfielded term's `[content, 2.0×title]` expansion into a single summed contribution per term; the pinned oracle expands every unfielded term into a two-clause boolean and accumulates 2N interleaved clause outputs. The f32 summation *association* differs, so scores diverge by 1–2 ULP as leaf count grows. Envelope members: (a) **≥3-leaf** pure-disjunctive shapes — 1–2 ULP `RankMismatch` on identical docs (reproduced bit pairs `0x415583bd`/`bc`, `0x41673288`/`87`, `0x4121c1e0`/`de`, `0x41addfb9`/`ba` at eight leaves; `0x4113fe32`/`33` at three leaves, read out of the DIV-008 witness object rather than transcribed); (b) a single top-level leaf boost over the multi-field expansion (1 ULP); (c) mixed-occur nesting that the Should-flatten (`0b9fad3b`) cannot splice (1–2 ULP). **Two-clause** disjunctions are bit-exact under the flatten and stay OUTSIDE this class.

  **Leaf-count boundary corrected 2026-08-04 (bd-gx7n4), by measurement.** This entry originally recorded member (a) as "≥8-leaf" and closed with "Pure-disjunctive spliceable shapes are bit-exact under the flatten and stay OUTSIDE this class." DIV-008 falsifies that sentence: a plain three-clause disjunction over the Core100 campaign corpus is NOT bit-exact, a 52-query sweep over the same corpus produced 9 such hits, the witness pair moves the ULP in OPPOSITE directions on the same document (which distinguishes summation association from a scoring bias in either engine), and only the two-clause control is bit-exact. The MECHANISM is unchanged and no new mechanism is admitted — the correction moves a measured lower bound from eight leaves to three and narrows the "outside" boundary to the shape actually measured as exact. The ≤2 ULP bound, the class, the typed reason, and the bd-55mvg ruling that the comparator's default config stays zero-tolerance are all untouched. Regression: `runner::tests::three_clause_or_diverges_at_one_ulp_without_the_div007_envelope`.
- Consumer impact: result *sets* are identical; only rank order within ULP-adjacent score pairs can flip. No membership change belongs to this class. (A negation inside a *boosted* group is a separate, *membership* divergence that classifies as `OracleBug`, never as `ScoreEpsilon` — see DIV-009. This entry previously described that as the oracle's lenient parse "dropping" the negation; measurement under bd-f20ye falsified it, and DIV-009 carries the corrected mechanism.)
- Owner ruling (2026-08-03, delegated to SandyGrove, recorded in mail thread `bd-55mvg`): keep Quill's fused scorer; adopt this bounded tolerance class rather than mirroring the oracle's interleaved per-field accumulation, which would surrender the fused-loop optimization on exactly the QG-6 query-latency axis. The comparator's default config REMAINS zero-tolerance; campaign lanes covering composite shapes opt in with the typed reason.
- Fixture: the four reproduced score-bit pairs above (bd-55mvg bead body); comparator typed-reason implementation and the generator unfence (groups, boosts, in-group negation except the bd-nqeb4 oracle-crash shape) tracked on bd-55mvg — blocked on active gauntlet file leases at ruling time.
- Decision: accept (owner ruling; bounded)
- Reviewer: SandyGrove (author of record for the ruling) · second-agent sign-off requested from LilacSquirrel (campaign author) via mail thread `bd-55mvg`

### DIV-008: the DIV-007 mechanism reaches the zero-tolerance default-profile lane as a raw rank mismatch

**This is the register's first WITNESSED entry.** DIV-001 through DIV-007 are documented — their
evidence was transcribed into the prose above by a human reading a campaign result. DIV-008 was
ingested by machine from the artifact that observed it, through
`DivergenceRegisterLedger::observation_from_artifact`, and the authoritative record is the ledger,
not this row. This row is the human projection of it.

- Class: `RankMismatch` (raw failure class — it may be fixed or blocked, never accepted)
- Machine record: `crates/frankensearch-quill-gauntlet/fixtures/divergence-register-v2-live.json`
  (register `quill-e6-divergence-register-live`, sequences 1–3). The ledger SHA-256 is a property of
  the whole register, not of this entry, and it moves whenever ANY divergence appends: it is
  `e846f1cac0ba5c191db5889a8436b86e909ea2b16928885a3495ec953fe80ef0` since DIV-009 was appended at
  sequences 4–5 (bd-dxedq); it was `b36c47186f47f119bc9469c75b852c0025282be9c96e5a5d58ee7e60498d2e3b`
  after the sequence-3 disposition below, and
  `6dfd9d1c7d9d07bbc261e1703c2bc1bc61b536adea5d17fd5daf9bd9b0ba276d` at sequence 2, when this row was
  first written. DIV-008's own three events are byte-unchanged across all three hashes, which is what
  `validate_append_only_successor` proves at every mint.
- First seen: 2026-08-04 · live default-profile oracle-differential campaign, run `e68-live-ingestion`,
  minted from a clean checkout at `4efe400cc80f55e85079400a7c54674116ab6f98`. Retained v7 witness object
  `65b1e4e89a3d1a2cc2202634fa448c397a48376fab90afb9a89390dfd823e763`, committed at
  `crates/frankensearch-quill-gauntlet/fixtures/artifact-object-v7-div007-live.json`; case
  `e68-div007-witness`; mismatch signature `15c251dd6671f6766082fc47e653cda26b72cb7b6e1b642ba4b795fb906133ec`
- Root cause: the DIV-007 mechanism observed OUTSIDE its documented qualifiers. A plain three-clause
  disjunction over the shared Core100 campaign corpus — three leaves, no boost, no mixed-occur
  nesting, where DIV-007 documents eight leaves — scores one document exactly one ULP away from the
  pinned oracle. A 52-query sweep over the same corpus produced 9 such hits, and the witness pair
  moves the ULP in OPPOSITE directions on the same document, which is what distinguishes summation
  association from a scoring bias in either engine. A two-clause control stays bit-exact.
- Consumer impact: result sets and rank positions are unchanged; only the order of two ULP-adjacent
  scores can flip. The load-bearing impact is on the campaign: the default-profile lane does not opt
  into the typed `SummationAssociation` reason, so this reaches it as a raw `RankMismatch` and the
  case fails closed as `Unclassified` on two independent axes — the total lexical contract mismatches,
  and the rank comparison carries an unregistered divergence.
- Fixture: campaign case `e68-div007-witness` (minimized-inputs SHA-256
  `210e480537e2cc6750df1699989cc1c818c8dab14ca71251772ddfe205136cb0`, re-derived by the selfcheck
  rather than trusted); executable regression
  `runner::tests::three_clause_or_diverges_at_one_ulp_without_the_div007_envelope`, which pins both
  halves of the bd-55mvg ruling: raw `RankMismatch` under the default comparator,
  `ScoreEpsilon`/`SummationAssociation` once a lane opts in.
- Decision: **fixed** 2026-08-04 (bead `bd-gx7n4`; fixing commit
  `78a2a189dd473ef641db7e99ac50b31b5500b1a1`), superseding the original **blocking** disposition by
  an APPENDED sequence-3 event — the blocking event is retained, not edited. The route taken was
  option (b) then (a) of the three the entry originally listed: DIV-007's documented envelope was
  first corrected by measurement to admit three-clause disjunctions (see the leaf-count boundary
  note on DIV-007), and only then did the default-profile lane opt in with
  `with_score_epsilon_reason(SummationAssociation)`. Doing (a) alone would have cited an envelope
  whose own text placed this shape outside the class. Option (c), re-associating Quill's fused
  summation, remains declined on the QG-6 latency grounds bd-55mvg recorded.
- Resolution: the lane refused this case on TWO axes, and both are closed. The rank axis is closed by
  the lane's typed reason. The lexical axis was the load-bearing one: `classify_case_with_lexical`
  short-circuits on any total-lexical-contract mismatch before `classify_case` runs, and
  `compare_score_bits` compares `normalized_score_bits` with plain `u32` equality and takes no
  comparator config at all, so the same one-ULP difference stayed a raw lexical mismatch whatever the
  rank comparator was told. `lexical_mismatches_are_the_classified_rank_divergence` now lets the
  lexical axis defer to a rank axis that ALREADY classified the case under a typed reason, and only
  when every lexical mismatch is that same score difference inside that reason's envelope. Nothing is
  hidden: the contract still reports `status == Mismatch` with every mismatch listed, and only the
  campaign disposition changes.
- Regression: `comparator::tests::the_lexical_axis_defers_only_to_a_rank_axis_that_actually_classified`
  (composed gate, rank axis built by the real comparator) and
  `comparator::tests::the_reviewed_score_envelope_refuses_everything_it_should` (six planted
  negatives: wider drift, non-Score class, another subsystem's score, presence asymmetry,
  unparseable diagnostic, wrong reason's envelope). The pre-existing
  `runner::tests::three_clause_or_diverges_at_one_ulp_without_the_div007_envelope` is unchanged and
  still pins both halves of the bd-55mvg ruling.
- Reviewer for the fix: `BlueOriole`, and independent review **is** required here. This line
  previously read "no independent reviewer is claimed and none is required —
  `DivergenceDisposition::validate` enforces `reviewer != recorded_by` only for **accepted**".
  That was true when written and `e2d992e8` made it false the same day, holding a `Fixed`
  disposition to the same fresh-eyes rule as an `Accepted` one. The rule is measured against the
  OBSERVER, not against whoever recorded the disposition, and this record satisfies it: the
  observation was recorded by `Claude-pane12` and the fix reviewed by `BlueOriole`. Verified
  rather than asserted — the committed ledger validates today, and mutating either identity onto
  the other is refused (bd-rm3q.8 adversarial pass, probes 2 and 3). What remains true from the
  old sentence is the second half: acceptance was never available here, because the observation's
  class is the raw `RankMismatch` and a raw failure class can never be accepted.
- Reviewer: recorded and blocked by `Claude-pane12`. No independent review is claimed, and none is
  required to block: `DivergenceDisposition::validate` enforces an independent reviewer only for
  acceptance, which is exactly the decision that needs one.
- Note for the terminal census (bead e6.8.1): the mismatch signature above is identical to DIV-006's,
  because a signature commits to the mismatch SHAPE (rank class, divergence class, normalized pointer,
  cause shape) rather than to the instance. DIV-006 and DIV-008 are mechanically unrelated, so a single
  ledger holding both active would trip the "one active mismatch signature cannot belong to multiple
  divergences" rule. That is a census-design constraint, not a defect in either entry.

### DIV-009: boosting a group changes its boolean meaning when the group negates (shipping path repaired, oracle pinned)

- Class: `OracleBug` (MEMBERSHIP — the engines disagree about which documents match, so no
  score-tolerance class can ever cover it; the blame is ATTRIBUTED, not asserted — see the machine
  record below). Recorded as a raw `RankMismatch` until bd-bxya1, because nothing in production
  could emit the semantic class.
- First seen: 2026-08-04 · probing for a divergence the default-profile lane still refuses after
  bd-gx7n4 opened it to the DIV-007 score envelope (bd-73ok3)
- Root cause: an ORACLE-side lowering defect in tantivy 0.26.1, with the conformance direction
  inverted — Quill executes these shapes correctly and the pinned oracle does not.
  **The mechanism first recorded here was wrong and is corrected under bd-f20ye.** DIV-007's entry
  and this row both said the lenient-parse fallback "drops" the negation. It does not: the `MustNot`
  clause is present in the parsed query, the strict and lenient parses agree, and
  `parse_query_lenient` reports NO errors for the offending shape. What actually happens is
  structural — an unboosted group lowers its negation as a `MustNot` clause OF the enclosing
  boolean, which excludes, while a boosted group nests it as
  `BooleanQuery { [(MustNot, …)], msm: 0 }` and attaches that as a clause of the outer boolean, so a
  matcher meaning "every document except B" becomes a positive alternative.
- Measured at the parser and at the result set (`frankensearch-lexical`, two documents
  `p1="alpha beta"`, `p2="alpha gamma"`), which is what falsified the original account:
  `alpha NOT beta` → `[p2]`; `(alpha NOT beta)` → `[p2]`; `(alpha NOT beta)^2` → `[p1, p2]`, the
  excluded document returns; `(alpha NOT alpha)` → `[]` but `(alpha NOT alpha)^2` → `[p1, p2]`, a
  self-contradictory group matching everything; and `(alpha AND NOT beta)^2` → `[]`, failing the
  OTHER way by losing a document it should return. A boosted group without a negation keeps its
  meaning, so the defect is specific to negation rather than to grouping or boosts.
- Measured with the lane's own enveloped comparator, shared Core100 corpus:
  `(release NOT release)^2` → `Failed`/`RankMismatch`; `(return NOT return)^2` → the same on a second
  operand; `(release NOT release)` → `Exact`, so the BOOST is what triggers it; and the reviewed
  `(release OR require) OR return` → `Classified`/`ScoreEpsilon` in the same run.
- Consumer impact: result SETS differ — a user query of this shape gets back documents the negation
  was supposed to exclude. Silent wrong results rather than a crash, which is worse, because nothing
  reports it. `frankensearch-lexical` is the shipping tantivy lane, so this is not gauntlet-only.
- Fixture: campaign case `e68-oracle-bug-refusal` in the E6.8 witness suite; executable regressions
  `runner::tests::live_default_profile_campaign_ingests_its_unclassified_divergence`, which asserts
  it still fails closed in the same run where the reviewed score mechanism classifies, and
  `frankensearch_lexical::tests::boosting_a_group_that_negates_changes_its_meaning_in_the_pinned_oracle`,
  which pins the oracle's exact behaviour in both directions so a tantivy upgrade that changes it
  becomes visible instead of silently moving the conformance target.
- Decision: **accept** — the two roles of `frankensearch-lexical` diverge here, deliberately, by
  owner ruling (2026-08-05, recorded on `bd-f20ye`; the owner may override).
- Equivalence law and rationale: the oracle is a pinned **comparator**, not a semantics authority. A
  boost is a score multiplier; a boost that changes boolean MEMBERSHIP is a defect by any reading,
  and `(a NOT b)^2` returning documents the query excluded is user-facing wrong. So the SHIPPING
  search path repairs the shape while the `oracle_observe_*` family stays bit-faithful to Tantivy
  0.26.1, defects included, and Quill continues to be measured against an unmoved target. Direct
  precedent: Quill already fixed the NOT+prefix double-negation defect this oracle carries (bd-bsjw),
  so the campaign has an established pattern for shipping-correct-while-oracle-faithful.
- Scope of the divergence, stated exactly: `TantivyIndex::parse_query_shipping` strips the boost from
  any parenthesized group containing a negation, and only the six user-facing `search*` methods call
  it. Membership is then identical to the unboosted form; the boost factor is dropped rather than
  redistributed, because a `MustNot` clause contributes no score and Tantivy cannot currently express
  the boosted form correctly at all. `oracle_observe_query` and `oracle_observe_page` are unchanged.
- Enforced by planted negatives in BOTH directions:
  disabling the repair reddens "the shipping path must not return a document the query excluded";
  leaking the repair into `oracle_observe_page` reddens "the oracle page surface must reproduce the
  defect, not the repair". Neither direction can regress silently.
- Machine record: `crates/frankensearch-quill-gauntlet/fixtures/divergence-register-v2-live.json`
  (register `quill-e6-divergence-register-live`, observation seq 4 and reviewed `accepted` seq 5;
  ledger SHA-256 `e846f1cac0ba5c191db5889a8436b86e909ea2b16928885a3495ec953fe80ef0`, which is also
  the census the terminal authorization compares against). COMMITTED since bd-dxedq — it was minted
  on demand and merged by nobody until then, so the ledger held DIV-008 alone while this row read as
  though it were governed. The mint now APPENDS to the committed register and proves the append with
  `validate_append_only_successor`, so DIV-008's history cannot be dropped or edited by a re-mint.
  Minted from a clean checkout at `0c4f9545e438ced7b824296bdda2782424c06a48`, run
  `e68-live-ingestion`, case `e68-oracle-bug-refusal` (minimized-inputs SHA-256
  `22cc1023a3f57aa6816aa88423d8f4d9fac2086abeeb2d5f6ad50e33afda0048`, re-derived by the selfcheck
  rather than trusted); mismatch signature
  `f708f094b609cf06c3a3d177ef8093db134c08c707914800cd185834040d1b05`; v8 witness object
  `85aecfb9f7aef2aaa51bbf27b5d7ab8410397c1794c75aa03023316974be2636`, committed at
  `crates/frankensearch-quill-gauntlet/fixtures/artifact-object-v8-div009-live.json`. The mint
  records **accepted**, agreeing with the reviewed decision above (bd-bxya1). What changed is the
  class, not the validator: `DivergenceDisposition::Accepted` still refuses every raw failure class, and this
  divergence is no longer raw because the campaign ATTRIBUTES it. The attribution is a stored
  comparator INPUT — `ComparatorConfig::oracle_bug_reason`, part of the artifact/report v8 shape — so
  the retained artifact re-derives the identical `OracleBug` report from its own observations, and
  the campaign verifier re-derives the attribution independently from the query rather than accepting
  the stored one. It is gated on three independent pieces of evidence, all required: the query SHAPE
  (a boosted group containing a negation, not a register row's opinion), the SYMPTOM (a membership
  `RankMismatch`), and the SIDE (the oracle returned a strict superset of the subject's documents,
  which is what attributes blame). Six planted negatives, each refused alone against an admitting
  control, pin the gate — the load-bearing one being a subject-side defect wearing the same shape and
  symptom, which is refused twice: the gate declines to attribute it, and the comparator refuses the
  attribution even when a configuration asserts it.
- What attribution does NOT buy: the campaign case still fails closed. It carries no per-fixture
  register row, and the total lexical contract still mismatches — the lexical axis defers only to a
  reviewed SCORE envelope and deliberately never to a membership class. Attribution earns the
  divergence a decidable class; it does not earn the case a pass.
- Reviewer: owner ruling of 2026-08-05, recorded by `Claude-pane12`. Distinct from `bd-nqeb4`, which
  is a `PhraseScorer` panic on a negated absent phrase — different shape, different failure mode.

### DIV-010: `A AND NOT B` matches nothing in tantivy 0.26.1; Quill now answers it correctly

- Class: `OracleBug` (MEMBERSHIP — the engines disagree about which documents match, so no
  score-tolerance class can cover it; v9 attributes the defect with a stored cross-case control)
- First seen: 2026-08-05 · found while implementing bd-f20ye's shipping repair, filed as `bd-eeq0q`,
  and made a DIVERGENCE by `bd-quill-shipping-conformance-parse-split-w7bsu`
- Root cause: `A AND NOT B` lowers to
  `Bool{[(Must, A), (Must, Bool{[(MustNot, B)], msm: 0})], msm: 0}`. The second `Must` operand is a
  boolean holding only a `MustNot` clause, so it has no positive term, matches nothing, and empties
  the whole conjunction. The same engine answers `A NOT B` and `A -B` correctly, so this is an
  inconsistency inside one query language rather than a defensible reading of `AND NOT`.
- THE ORDER OF EVENTS MATTERS, because it is what makes this entry honest. When `bd-eeq0q` measured
  the shape, BOTH engines returned nothing: Quill's `wrap_not_for_and` (quill `query.rs`)
  deliberately mirrored the oracle's lowering. There was therefore NO divergence, and no register
  entry was due — `bd-eeq0q` repaired only `frankensearch-lexical`'s shipping path and left the
  agreement pinned by a tripwire test. This bead then repaired Quill on purpose, which BROKE that
  agreement and created the divergence recorded here. The entry is the price of the second repair,
  not a discovery.
- Measured, two documents `p1="alpha beta"`, `p2="alpha gamma"`:
  `alpha NOT beta` → `[p2]` on both; `alpha -beta` → `[p2]` on both;
  `alpha AND NOT beta` → `[p2]` on Quill and on the lexical SHIPPING path, `[]` on the
  `oracle_observe_*` surface; `(alpha AND NOT beta)` and `(alpha AND NOT beta)^2` behave the same
  way. `NOT beta AND alpha` is repaired by the same change.
- **RETRACTED 2026-08-05 (`bd-iiidv`): this entry claimed `NOT alpha AND NOT beta` "still returns
  nothing on both, because an exclusion-only conjunction has no positive term — that is agreement,
  not a defect, and is pinned so a later change cannot silently turn it into match-all". That was
  wrong twice over, and there was no such pin.** The sentence is retracted rather than deleted,
  because it is what a reader checking whether this shape was covered would have relied on.
  Measured on the gauntlet's Core100 fixture at a non-truncating limit, and on the four-document
  lexical fixture:
  - `NOT release AND NOT small` → **59** documents on Quill, **0** on the `oracle_observe_*`
    surface. Not agreement.
  - 59 is exactly the declared complement, `|Core100| − |release ∪ small|` = 100 − 41, and *not*
    the complement of one operand (79). The operand pair is deliberately non-nested so those two
    readings differ: `NOT release AND NOT bounds` cannot discriminate them, because `bounds` matches
    one document and it is inside `release`. Quill honours BOTH exclusions.
  - The same oracle answers every spelling of the same intent that carries no explicit `AND` with
    that same complement: `NOT release NOT small` → 59 vs 59, `-release -small` → 59 vs 59, both
    `Exact`. An all-negative root therefore does **not** match nothing — it has complement
    semantics on both engines — and the `AND` spelling is the only one that diverges.
  - `NOT small AND NOT release` diverges identically, so it is not an operand-order artifact.
  - The shipping path was never affected, and it is covered TWICE over — measured, because the
    first version of the new pin passed with `repair_negated_conjunction` disabled and was
    therefore vacuous. `repair_negated_conjunction` normalises `NOT beta AND NOT gamma` →
    `-beta -gamma`, and `repair_and_not` independently deletes the `AND` before the `NOT`, giving
    `NOT beta NOT gamma`; either alone leaves `frankensearch-lexical`'s public search surface
    answering the complement, and only disabling BOTH reddens the pin, at `[]` against `["p4"]`.
    There was no shipping-side gap to fix; what was missing was the proof, now pinned by
    `frankensearch_lexical::tests::an_all_negative_conjunction_is_complement_on_shipping_and_empty_on_the_oracle`
    and by two normal-form cases in
    `a_negated_conjunction_normalises_to_explicit_occurrences`.
  - It is **this divergence, not a fourth one**: an explicit `AND` whose operand is a negation
    lowers to a positive-less boolean that empties the conjunction, which is the root cause stated
    above. The all-negative form is that mechanism with both operands negated, repaired by the same
    repair, and it needs no separate register entry or witness. The machine record's existing
    DIV-010 observation covers it.
- EXTENDED 2026-08-04 by `bd-8a2a8`, same root cause, same decision, one more ORDERING. When the
  conjunction is not the whole query — `A NOT B AND C` — the declared grammar reads it as
  `A OR (C AND NOT B)` (`quill-language-contract.md`: "default join := OR; explicit AND has
  precedence over OR"), and the defective lowering drops the entire `AND` conjunct rather than
  emptying the query, so the divergence hides as a SHORTER result instead of an empty one. Measured
  on the gauntlet's Core100 fixture, `release NOT bounds AND small` → 41 documents on Quill
  (`|release ∪ small|`, the declared reading) and 21 on the unrepaired lexical path (`|release|`).
  `repair_and_not` did not cover it, because it only deletes an `AND` that is immediately followed by
  a `NOT`. THE ATTRIBUTION, which is what makes this an oracle defect rather than a Quill one: the
  same engine answers `release ((small) NOT bounds)` — the identical query with its grouping written
  out — with the same 41 documents Quill returns, on BOTH of its roles. An engine that already agrees
  with the declared reading when the grouping is explicit is mislowering the implicit form, not
  asserting a different semantics.
- The bd-8a2a8 repair is `repair_negated_conjunction`, and it normalises rather than deletes: an
  `AND` chain containing a negation is re-spelled with explicit occurrences, `a NOT b AND c` →
  `a (-b +c)`, preserving operand order. That is exact rather than approximate, because `+`/`-` ARE
  the `Must`/`MustNot` the chain means. The narrow alternative — deleting that `AND` too — was
  MEASURED and rejected: it fixes `"ranked one" NOT bounds AND refactors` (19 documents, matching
  Quill) while breaking `explains NOT bounds AND refactors`, which already answered 60 correctly and
  drops to 59. A repair that regresses correct queries is not a repair. It runs BEFORE
  `repair_and_not` in `parse_query_shipping`, because it decodes whole chains and must see the `AND`
  that repair deletes; and it emits a chain that spans its whole level WITHOUT adding parentheses, so
  `repair_boosted_group_negation` — which records negation against the innermost open group — still
  recognises `(a AND NOT b)^2` as negating and DIV-009 does not silently regress.
- Consumer impact: before this repair, the most common negation spelling silently returned NOTHING on
  both backends — no error, no warning, just an empty result for a query the user reasonably expects
  to work. That is worse than a crash because nothing reports it. After it, both backends answer
  correctly and only the pinned comparator retains the defect.
- Fixture: executable regressions
  `frankensearch_quill::index::tests::quill_answers_and_not_correctly_and_diverges_from_the_pinned_oracle`
  (the former tripwire, inverted rather than deleted, so the arc from agreement to divergence stays
  legible) and
  `frankensearch_lexical::tests::and_not_returns_a_minus_b_on_shipping_while_the_oracle_stays_bit_faithful`
  (shipping repaired, oracle surface still defective, quoted literal untouched). For the `bd-8a2a8`
  ordering,
  `frankensearch_lexical::tests::a_negated_conjunction_reads_as_a_disjunct_on_shipping_while_the_oracle_stays_bit_faithful`
  (the declared reading on the shipping path, the dropped conjunct still dropped on the oracle
  surface, and the explicit-grouping control answered correctly by both roles) and
  `frankensearch_lexical::tests::a_negated_conjunction_normalises_to_explicit_occurrences`
  (the emitted normal form as TEXT, because the DIV-009 interlock and the borrow-on-no-change
  contract are both invisible in a result set).
- Decision: **accept** — same equivalence law as DIV-009, which is the direct precedent.
- Equivalence law and rationale: the oracle is a pinned **comparator**, not a semantics authority.
  Quill is the measured SUBJECT and the gauntlet observes it through `search_paginated`, the same
  public surface users call, so Quill has exactly ONE role and there is no shipping/conformance split
  to build in it — a split would make the gauntlet measure something users never run, which is the
  inverse of the fidelity it exists to provide. Repairing the one path is therefore both the
  user-facing fix and the deliberate divergence. `frankensearch-lexical` keeps its two roles: its
  shipping path is repaired (`repair_and_not`) and every `oracle_observe_*` caller stays bit-faithful
  to Tantivy 0.26.1, so Quill continues to be measured against an unmoved target.
- Scope of the divergence, stated exactly: Quill's `parse_and` no longer re-wraps a `NOT` operand of
  an explicit `AND` into a positive clause; `wrap_not_for_and` is deleted and
  `wrap_direct_negative_or_operand` is deliberately retained, because the `OR` side genuinely needs
  that wrapping so a bare negative operand cannot become a positive alternative. No other shape moves:
  the full `frankensearch-quill` lib suite is 560 passed / 0 failed across the change.
- Enforced by planted negatives in BOTH directions: on the lexical side, disabling `repair_and_not`
  reddens the shipping assertion and leaking it into `oracle_observe_query` reddens "the oracle must
  still reproduce the tantivy 0.26.1 defect" (both run as red proofs under `bd-eeq0q`); on the Quill
  side, the exclusion-only and leading-`NOT` cases are pinned beside the repaired forms so a broader
  change to negative handling cannot pass by loosening them.
- Machine record: `crates/frankensearch-quill-gauntlet/fixtures/divergence-register-v2-live.json`
  retains the original v8 `RankMismatch` observation seq 6 and `blocking` seq 7 on `bd-4oiwf`
  byte-for-byte, then appends v9 `OracleBug` observation seq 8 (supersedes 6) and reviewed
  `accepted` seq 9 (supersedes 7). The ledger hash is
  `2fa251818ab71467807a13b26f94a25c87d1c94ea33ef109870f6e48d8987816`. The v8 witness remains
  `46176fa8c9f7911eea852bc4089740323322f1be95c4563329d5c6dec29dd0f9` at
  `crates/frankensearch-quill-gauntlet/fixtures/artifact-object-v8-div010-live.json`; it is still
  required by the relational join because superseding evidence never erases history.
- The successor was minted by the sealed live lane from clean revision
  `70e1c01ba9e8f23b9b988ee5e1569e577148260c`, case
  `e68-negated-conjunction-refusal`, at a non-truncating limit of 512. Its v9 witness address is
  `201e3c43e3534692661a47ae8d5faa68c84361297dba02b0c94aa1e73657fb57`, committed at
  `crates/frankensearch-quill-gauntlet/fixtures/artifact-object-v9-div010-live.json`.
- **THE ARTIFACT NOW CARRIES THE ATTRIBUTION, NOT MERELY ITS CONCLUSION.** The stored
  `e68-negated-conjunction-explicit` control observes `small NOT bounds` through both engines. Both
  are exact on that equivalent spelling, and its membership equals the witness subject's; only the
  explicit `AND` makes the oracle empty the conjunction. v9 re-runs the comparison from those four
  observations and emits `OracleBug`. The verifier independently reconstructs the required control
  query and refuses truncation, a changed case, an inexact control, or a control whose subject does
  not match the witness subject.
- The load-bearing negative is the formerly indistinguishable failure direction: a planted
  SUBJECT-side membership defect with the same `A AND NOT B` shape is denied `OracleBug` because it
  disagrees with the stored control. Membership containment by itself therefore remains
  insufficient; the old single-case gate still declines DIV-010, and neither a register row nor
  disposition prose can assign blame.
- Which SHAPE is minted, and why it is not the other one: the bd-8a2a8 ordering (`A NOT B AND C`,
  where the conjunct is dropped and the result is merely shorter) was tried first and the mint
  REFUSED it — `one active mismatch signature cannot belong to multiple divergences`. A signature
  commits to the mismatch SHAPE, and that ordering produces `rank:hit:hit` at `hits/0`, which is the
  shape DIV-008 already holds active. DIV-008's own note predicted this collision for DIV-006 and
  called it a census-design constraint; it is now a measured one. The emptied form is this entry's
  own first measurement and its signature is a length disagreement, so one ledger carries both.
- Reviewer: `RubyJaguar`, reviewed 2026-08-10, independent of observation recorder
  `Claude-pane12`. The retained sequence-7 blocking disposition remains part of history; the
  independent review is bound to the superseding sequence-9 acceptance.

---

*Cross-references: comparator classes implemented in the gauntlet kernel (bead e0.5); auto-triage feeding this ledger (bd-quill-duel-shrinker); statistical gates consuming per-class pass rates (bead e6.6); G2 exit requires this register complete over two consecutive nightly runs (bead e6.8).*
