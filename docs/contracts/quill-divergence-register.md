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
  `{scheme: "frankensearch-quill-gauntlet/artifact-object/v7/sha256", object_schema_version: 7, digest: <64 lowercase hex>}`;
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
`validate_against_artifact_objects`, which first runs each `ArtifactObject`'s
stored-evidence integrity contract and then compares the complete closed
binding: object schema version, hash-domain scheme, object digest,
producer-build identity, oracle-dependency identity, lexical-contract audit
revision, corpus/query manifests, query-suite source kind and identity,
first-recorded case ID, rank class, divergence class, and mismatch signatures.
Every observation event, including superseded history, must bind to the object
it actually references. A multi-class object must be covered exactly; a
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
- Root cause: Quill's Term scorer fuses each unfielded term's `[content, 2.0×title]` expansion into a single summed contribution per term; the pinned oracle expands every unfielded term into a two-clause boolean and accumulates 2N interleaved clause outputs. The f32 summation *association* differs, so scores diverge by 1–2 ULP as leaf count grows. Envelope members: (a) ≥8-leaf pure-disjunctive shapes — 1–2 ULP `RankMismatch` on identical docs (reproduced bit pairs `0x415583bd`/`bc`, `0x41673288`/`87`, `0x4121c1e0`/`de`, `0x41addfb9`/`ba`); (b) a single top-level leaf boost over the multi-field expansion (1 ULP); (c) mixed-occur nesting that the Should-flatten (`0b9fad3b`) cannot splice (1–2 ULP). Pure-disjunctive spliceable shapes are bit-exact under the flatten and stay OUTSIDE this class.
- Consumer impact: result *sets* are identical; only rank order within ULP-adjacent score pairs can flip. No membership change belongs to this class. (The oracle's lenient-parse fallback silently DROPS every negation inside a boosted group — a *membership* divergence that classifies as `OracleBug`, never as `ScoreEpsilon`.)
- Owner ruling (2026-08-03, delegated to SandyGrove, recorded in mail thread `bd-55mvg`): keep Quill's fused scorer; adopt this bounded tolerance class rather than mirroring the oracle's interleaved per-field accumulation, which would surrender the fused-loop optimization on exactly the QG-6 query-latency axis. The comparator's default config REMAINS zero-tolerance; campaign lanes covering composite shapes opt in with the typed reason.
- Fixture: the four reproduced score-bit pairs above (bd-55mvg bead body); comparator typed-reason implementation and the generator unfence (groups, boosts, in-group negation except the bd-nqeb4 oracle-crash shape) tracked on bd-55mvg — blocked on active gauntlet file leases at ruling time.
- Decision: accept (owner ruling; bounded)
- Reviewer: SandyGrove (author of record for the ruling) · second-agent sign-off requested from LilacSquirrel (campaign author) via mail thread `bd-55mvg`

---

*Cross-references: comparator classes implemented in the gauntlet kernel (bead e0.5); auto-triage feeding this ledger (bd-quill-duel-shrinker); statistical gates consuming per-class pass rates (bead e6.6); G2 exit requires this register complete over two consecutive nightly runs (bead e6.8).*
