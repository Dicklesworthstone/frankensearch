# Quill Language and Scoring Contract

**Contract version:** 1.0.2

**Owning bead:** `bd-quill-e0-contracts-j53p.1`

**Oracle:** Tantivy 0.26.1, pinned by `Cargo.lock`
**Contract fixture:** `tests/fixtures/quill_language_contract.json`

**Amendment 1.0.1:** Corrects the default-parser duplicate fixture to match the pinned grammar's pre-analysis stable deduplication of exact raw siblings. Programmatically constructed scoring duplicates remain distinct in the post-parse canonicalizer.

**Amendment 1.0.2:** Pins the E5 mixed-residency execution surface: string and numeric ranges and sets, bounded globs, exact-count pagination, and scoreless global-id collection now execute identically across Delta, sealed, and mixed snapshots with invariant physical scoring history.

This document defines what “the same results as Tantivy” means for Quill. It is normative for the used lexical surface in `COMPREHENSIVE_PLAN_FOR_THE_QUILL_LEXICAL_ENGINE.md` §3.1, §5, and §8. Quill code does not merge past gate G0 unless its scalar reference path satisfies this contract. The lexical-crate loader currently executes schema, analyzer, helper, and BM25 operation-order goldens and structurally validates the other records; the G1/G2 differential runner is responsible for executing canonical query trees, lifecycle rows, and cross-engine result comparisons.

The conformance target is the observable behavior of frankensearch, not every Tantivy API. Analyzer behavior is pinned to the shipping scalar implementations. Query and scoring behavior is pinned to the shipping adapters plus the Tantivy 0.26.1 source used as the oracle. Intentional departures belong in `docs/contracts/quill-divergence-register.md`; an unclassified departure is a failure.

## 1. Normative sources and precedence

When sources disagree, apply this order:

1. This versioned contract and its executable fixtures.
2. Shipping frankensearch behavior in:
   - `crates/frankensearch-lexical/src/lib.rs`
   - `crates/frankensearch-lexical/src/cass_compat.rs`
   - `crates/frankensearch-core/src/traits.rs`
3. Tantivy 0.26.1 for scoring, Boolean `Occur`, parser-default, and token-admission semantics.
4. The comprehensive Quill plan.

The fixture distinguishes harvested queries from constructed boundary cases and records source paths or source facts for that query-class corpus. Analyzer and scoring rows are constructed contract vectors whose normative source is this section plus the pinned implementation named above. A later source discovery that contradicts a fixture requires a contract-version amendment, not a silent code change.

Two clarifications supersede older planning prose:

- `avgdl` is raw `total_num_tokens / total_num_docs`. Only a document’s `|d|` is decoded from its one-byte fieldnorm. Averaging decoded fieldnorm buckets is wrong and changes scores.
- Tantivy scoring statistics include deleted documents until their segment is merged. Tombstones suppress matches, but `N`, term document frequency, and total token counts remain the sealed-segment values until compaction.

## 2. Used-surface coverage

Every §3.1 row has at least one fixture ID in the executable `surface_coverage` table.

| Surface | Normative behavior |
|---|---|
| Schema | `id`: stored exact text (`Basic` postings); `content` and `title`: stored text using `frankensearch_default` with frequencies, positions, and fieldnorms; `metadata_json`: stored-only text; `ord`: stored fast `u64`. |
| Analyzers | `frankensearch_default`, `hyphen_normalize`, `prefix_normalize`, edge-prefix generation, and preview generation. |
| Queries | Term, Boolean, phrase, typed string/`i64`/`u64` range, typed string/`i64`/`u64` set, bounded glob-derived term matching, all-docs, boost, and the default lenient parser. |
| Collectors | Top-k plus offset, exact count, id-set collection, and count-free top-k where applicable. |
| Scoring | BM25 constants/formula, fieldnorm quantization, title boost, fixed f32 accumulation, and deterministic tie order. |
| Writer | Add, batch add, upsert, delete, clear, commit/seal, and cancellation taxonomy. |
| Reader | Immediate snapshot visibility for Quill, live document count, and stable doc-ID materialization. |
| Snippets | Default value 200, enforced by Tantivy against UTF-8 byte offsets at token boundaries despite the upstream `max_num_chars` name; configured tags and exact output comparison. |
| Durability | Checksummed FSLX segments/manifests, path safety, repair integration, and tombstone-aware open. |
| Errors | `IndexNotFound`, `Cancelled`, `InvalidConfig`, I/O/corruption, and `SubsystemError { subsystem: "quill" }`. |
| Concurrency | `Send + Sync`, `Cx` on operations, and an immediately-ready fusion read future. |

Anything in plan §3.2 is out of contract until added here with fixtures.

## 3. Analyzer contract

### 3.1 Token record

Analyzer equality is exact equality of the ordered token stream:

```text
(text: UTF-8 string, position: usize, offset_from: byte index,
 offset_to: byte index, position_length: usize)
```

Offsets are half-open byte ranges into the original input. Position duplicates are legal. Quill must not normalize, coalesce, or reorder duplicate-position alternatives.

### 3.2 Default analyzer

`frankensearch_default` is the shipping `FrankensearchTokenizer`:

- Split on characters for which `char::is_alphanumeric` is false.
- Lowercase ASCII in place and apply the full `char::to_lowercase` expansion for non-ASCII.
- Start positions at zero and increment once per emitted token.
- Preserve source byte offsets even when lowercasing changes the output byte length.
- Do not apply an analyzer-local length filter.

The Quill SIMD path may classify pure-ASCII spans in wide lanes. Encountering a high byte falls back to the scalar char walk for the affected span. SIMD and scalar streams must be byte-identical.

### 3.3 CASS analyzer family

The shipping `hyphen_normalize` pipeline is:

```text
CassTokenizer
  -> HyphenDecompose
  -> CjkBigramDecompose
  -> CassNormalizeAndLimit
```

`CassTokenizer` emits either:

- ASCII alphanumeric runs with interior hyphens only when both sides are ASCII alphanumeric; or
- contiguous runs in the exact CJK ranges listed in `cass_compat.rs`.

`HyphenDecompose` replaces an interior-hyphen token with the compound followed by each non-empty part. All alternatives retain the compound’s position, full source offsets, and position length.

`CjkBigramDecompose` replaces an all-CJK token of at least two characters with overlapping bigrams in source order. A single CJK character remains a unigram; non-CJK tokens pass through. All generated bigrams retain the source token’s position and full offsets.

`CassNormalizeAndLimit` drops tokens whose UTF-8 length exceeds 256 bytes, then ASCII-lowercases retained tokens. The boundary is deliberately inclusive: the shipping CASS analyzer retains exactly 256 bytes and starts dropping at 257. Tantivy 0.26.1's `RemoveLongFilter::limit(256)` instead retains only lengths strictly below 256, so it is not the boundary oracle. The restriction is safe because `CassTokenizer` emits ASCII/hyphen tokens or CJK runs.

The `prefix_normalize` pipeline omits `HyphenDecompose` because its input is generated whitespace-separated edge prefixes:

```text
CassTokenizer -> CjkBigramDecompose -> CassNormalizeAndLimit
```

Edge-prefix generation emits every prefix of length 2 through 20 Unicode scalar values for each alphanumeric word, in word and prefix-length order, separated by one ASCII space. Preview generation returns the first `max_chars` Unicode scalar values and appends `…` iff input remains.

### 3.4 Oversized terms

Tantivy 0.26.1 defines `MAX_TOKEN_LEN = u16::MAX - 5 = 65_530` bytes. Its postings writer admits an indexed token of exactly 65,530 bytes and drops a 65,531-byte token with a warning. Tantivy’s `QueryParser` does not repeat that check. Quill deliberately applies the 65,530-byte admission rule on both document and query analysis so a term cannot be accepted on only one side; this symmetric query-side check is an explicit Quill hardening amendment, not a claim about Tantivy internals. Ordinary frankensearch string queries are already capped at 10,000 characters, so the query-side boundary is reachable only through lower-level query construction. This limit is distinct from CASS’s analyzer-local 256-byte limit.

The hardening must preserve oracle result semantics, not merely erase the token: a standalone oversized term, required oversized clause, or phrase containing one lowers to `Empty`/match-none; an unmatchable optional `Should` or excluding `MustNot` may be removed only while valid sibling semantics remain intact. Dropped document tokens preserve subsequent position gaps. The AST/diagnostic classification and executable G1 shape proof are tracked by `bd-quill-e0-contracts-j53p.8`; until it lands, the rule is a G0 target and not an accepted divergence.

## 4. Query grammar contract

### 4.1 Canonical fixture AST

Fixtures use an engine-neutral JSON AST. Nodes are `Empty`, `All`, `Term`, `Phrase`, `PhrasePrefix`, typed `Range`, `Set`, `Glob`, `Boost`, or `Boolean`; `RangeI64` is the concrete CASS timestamp-range tag. Boolean children carry `Must`, `Should`, or `MustNot`. Default-parser and CASS `Term`/`Phrase` nodes list their ordered field/boost expansion explicitly; direct-Boolean semantic fixtures use synthetic unfielded terms. `Boolean.children` preserves construction order, while §5.4 separately defines runtime scorer order. Implementations may use different internal types but must canonicalize to the fixture tree. Classification matrices are represented as separate `Glob` cases, never as several inputs packed into one pseudo-tree.

### 4.2 Default lenient parser

The default parser searches unfielded literals across `[content, title]` and multiplies title literal/BM25 contributions by `2.0`. Its conjunction default is false, so adjacent plain fragments are a disjunction. It otherwise inherits the pinned Tantivy 0.26.1 query grammar: explicit `AND` binds tighter than `OR`; parentheses group; `+`/`-` set `Must`/`MustNot`; fields use `field:value`; `^` boosts a leaf or group; quoted literals support slop and phrase-prefix forms; typed fields support ranges and sets; and `*` denotes all documents. Regex and fuzzy parsing are not enabled by the shipping adapter.

```text
query        := expression
expression   := fragment ((AND | OR | whitespace) fragment)*
fragment     := ['+' | '-' | 'NOT'] (literal | group | field_fragment) ['^' boost]
default join := OR; explicit AND has precedence over OR
```

Every unfielded literal fragment is analyzed independently for each default field; a field-qualified literal is analyzed only for that field. Zero emitted tokens remove that field branch, one token makes a `Term`, and two or more tokens make a slop-zero `Phrase` even when the fragment was not quoted. Quoting therefore controls grammar grouping, not the token-count-to-query-node rule; a quoted one-token fragment is still a `Term`. Typed ranges, sets, and `All` bypass literal analysis and do not inherit the title literal boost.

Before field analysis, the default parser mirrors the pinned grammar's raw-syntax rewrite: at every Boolean level it recursively rewrites clause children, stably retains the first exact `(optional occurrence, syntax child)` pair, and flattens an implicitly occurring singleton group while transferring any explicit inner occurrence. The rewrite deliberately does not enter a non-identity `Boost`, so `(rust rust)^2` retains both scorers while bare `rust rust` scores one. Syntax identity is pre-analysis identity: case and quote delimiter remain significant, while parentheses and identity boosts do not. This parser-only rule is separate from post-parse `canonicalize_query`; programmatically constructed positive/scoring duplicates remain distinct there because the oracle's BM25 sum-union scores each branch (bd-quill-duel-ast-dedup-7hsu). Exact `MustNot` canonicalization and stable `Must`/`Should`/`MustNot` grouping belong to that later pass, and probe-cheap cursor sharing belongs to the planner.

Boosts have one finite-score safety exception to the pinned grammar (DIV-005): a syntactically valid boost that cannot be represented as a finite non-negative `f32` is diagnosed as `InvalidBoost` and recovered as the unboosted branch; its parser syntax key likewise treats the boost as absent. Individually finite nested boosts whose cumulative product becomes non-finite fail deterministically during lowering. Exact Tantivy score-bit parity excludes only this registered non-finite class.

Lenient mode recovers as much input as possible. Semantic failures such as an unknown field are diagnosed and their invalid branch is dropped while valid siblings remain. Syntax recovery may retain a repaired branch—for example, an unterminated quote can still yield a term—with a diagnostic. An all-negative AST is diagnosed and receives an `All` clause so it has complement semantics. The current adapter records diagnostics through tracing. Empty or whitespace-only input returns no results.

Queries are capped at 10,000 Unicode scalar values. Truncation selects the first 10,000 characters and therefore always ends at a UTF-8 boundary. This is a character contract, not a byte contract, and applies consistently to the Tantivy and FTS5 lexical adapters.

### 4.3 CASS Boolean grammar

CASS uses implicit `AND`; explicit `AND`/`&&`; explicit `OR`/`||`; and `NOT` or a leading `-`. Its intentionally non-standard precedence makes OR bind tighter than AND:

```text
query      := and_expr
and_expr   := or_expr ((AND | && | implicit_whitespace) or_expr)*
or_expr    := unary ((OR | ||) unary)*
unary      := (NOT | '-')* primary
primary    := term | quoted_phrase
```

For example, `auth OR token AND cache` means `(auth OR token) AND cache`. One or more adjacent negators lower to one `MustNot`; `NOT NOT a` is not logical double negation. In a conjunction, a negative is a raw top-level `MustNot`, so `auth AND NOT deprecated` is `Must(auth) + MustNot(deprecated)` and does not gain an `All` score. A negation used as a positive-valued OR operand is wrapped as `Must(All) + MustNot(primary)`. A standalone negative has the same complement target shape; the shipping CASS builder anchors every non-empty all-negative root with `Must(All)` while leaving mixed positive conjunctions unanchored. Sanitization preserves alphanumerics, `*`, `"`, and `-`, replacing other characters with spaces.

### 4.4 Boolean `Occur` semantics

- A Boolean node containing only `Should` children requires at least one match.
- If a positive `Must` child is present, ordinary `Should` children are optional: they affect score but do not gate matching.
- `Should` plus `MustNot`, with no positive `Must`, still requires at least one `Should` match. A raw `MustNot`-only Boolean node matches nothing; parser lowering adds `Must(All)` when complement semantics are intended.
- `MustNot` filters matches and contributes no score.
- Within an `Occur` group, clause construction order is stable. Across groups, score evaluation follows Tantivy’s scorer tree rather than raw parse order: the required aggregate is evaluated before the optional `Should` aggregate. Quill must match the oracle’s f32 evaluation order for each Boolean shape.

These rules are pinned independently of the parser because query planners also construct Boolean trees directly.

### 4.5 Phrase, range, glob, and CJK rules

- Ordinary default and CASS phrases require exact adjacent positions (slop zero) in `content` or `title`. An explicit default-parser `~n` suffix uses slop `n`, and a trailing phrase `*` creates a phrase-prefix query. A CJK phrase in the CASS path falls back to the compound bigram query used by shipping.
- `created_at` filters use inclusive i64 lower and upper bounds; either side may be unbounded. Shipping inserts the FAST-field range as a `Must` query, whose matched `ConstScorer` contributes `1.0`; “filter” describes its selection role, not a scoreless implementation.
- `CassWildcardPattern` classifies terms as `Exact`, `Prefix`, `Suffix`, `Substring`, or `Complex`. Exact and prefix terms issue exact `TermQuery` clauses over `title`, `content`, `title_prefix`, and `content_prefix`; the prefix fields contain only prefixes of 2 through 20 Unicode scalar values. Consequently `a*` does not generally prefix-match longer words, a prefix longer than 20 is not accelerated, and an `Exact` term of length 2 through 20 can match a longer indexed word through a prefix field. Suffix, substring, and complex `*` patterns issue anchor-free `RegexQuery` clauses over `content` and `title`: Tantivy FST regexes already match the complete term and reject explicit `^`/`$` assertions. If either field's regex construction fails, the complete CASS query fails closed to match-none instead of dropping the operand. Regex expansion is unbounded in the shipping path and each matching field uses Tantivy's constant scorer; introducing a bound requires a `GlobExpansionLimit` divergence. The sanitizer does not preserve `?`, so `?` is not a shipping wildcard operator.
- A term containing any CJK character drops its non-CJK characters for query bigram generation. The remaining CJK characters expand into overlapping bigrams; every bigram is `Must` and may match any searchable field (`title`, `content`, `title_prefix`, `content_prefix`). A single CJK character remains a unigram, but it cannot match inside a multi-character indexed run because that run is represented only by bigrams.
- Empty CASS input produces `All`, after which any supplied filters still apply.

## 5. Scoring and statistics contract

### 5.1 Formula and constants

For term frequency `f`, term document frequency `n`, corpus document count `N`, decoded document length `|d|`, and raw average field length `avgdl`:

```text
k1 = 1.2
b  = 0.75
idf = ln(1 + (N - n + 0.5) / (n + 0.5))
norm = k1 * (1 - b + b * |d| / avgdl)
weight = idf * (1 + k1)
tf_factor = f / (f + norm)
term_score = weight * tf_factor
```

All contract arithmetic is f32 in the displayed Tantivy operation order. In particular, do not algebraically reassociate the score to `idf * (f * (1 + k1) / (f + norm))`; that produces different low bits. Do not replace `(1.0 + x).ln()` with `ln_1p` or use FMA in contract mode.

For a phrase, Tantivy sums the per-term IDFs left-to-right in phrase-term order, multiplies that sum by `1 + k1`, and uses the number of phrase occurrences in the document as `f` in the same tf factor. Phrase score is not the sum of independently scored term queries.

### 5.2 Field length and `avgdl`

Each document stores one fieldnorm byte. `|d|` is `id_to_fieldnorm(byte)` using the complete table and functions vendored in `crates/frankensearch-lexical/src/quill_contract.rs`.

`avgdl` is **not** the mean of decoded fieldnorms. It is:

```text
total_num_tokens(field) as f32 / total_num_docs as f32
```

The 256-entry tf cache is built by combining every decoded per-document fieldnorm with that raw `avgdl`.

### 5.3 Deletes and scoring statistics

Tantivy’s scorer sums sealed segment statistics. Until merge/compaction:

- `N` is the sum of segment `max_doc`, including tombstoned documents.
- `total_num_tokens` includes tokens from tombstoned documents.
- term document frequency includes tombstoned documents.
- tombstones are excluded from matching and result collection.

After compaction, statistics are re-derived from retained live documents. The public live `doc_count` is separate from BM25’s pre-compaction `N` and must not be substituted into the formula.

### 5.4 Multi-field accumulation and ties

For default-parser literal/BM25 contributions, per-field scores use `content = 1.0` and `title = 2.0`; typed ranges, sets, and `All` are not field-boosted by that setting. The shipping CASS builder applies no field boost: `title`, `content`, `title_prefix`, and `content_prefix` are all `1.0` unless a future query node explicitly says otherwise. Scores are summed using the oracle scorer-tree order in f32. For an intersection, Tantivy stable-sorts required scorers by ascending runtime `Scorer::cost()`, then evaluates `left + right + sum(others)` in that order; equal-cost scorers retain construction order. Optional and exclusion aggregates are composed around that required scorer. Contract mode performs scalar score accumulation even if SIMD is used for decode and gather.

The Tantivy oracle's result order is the total order:

1. score descending via `f32::total_cmp`;
2. `DocAddress` ascending, lexicographically by `(segment_ord, segment-local doc_id)`.

Quill's native tie key is its global u32 docid. The differential comparator preserves each engine's native order for the primary comparison and records the oracle `DocAddress` for every hit; it must never collapse that address to the segment-local docid. A difference confined to an exactly equal-score group is classified as `TieOrder`, including a top-k boundary difference only when an expanded oracle tie group proves that every substituted document has the identical score. Any other order or cutoff difference fails `RankExact`.

## 6. Behavioral, collector, and runtime contract

- Upsert is delete-by-external-ID followed by add, with the new document becoming the sole visible version after publication.
- Delete removes matches immediately from the published snapshot; scoring statistics follow §5.3 until compaction.
- `delete_all` leaves a valid empty index.
- `doc_count` reports live documents.
- `limit = 0` returns no hits without invoking Tantivy’s nonzero-limit collector; a counted request still reports the exact total. Offset pagination returns page-local ranks starting at zero plus the exact total count when requested.
- Count-free top-k may skip exact-count work only when the caller did not request a count; rankings must match the counted path.
- Id-set collection is unscored and order-insensitive.
- Snippet defaults pass `200` to Tantivy and use `<b>…</b>` tags. Observable window enforcement compares tokenizer byte offsets, then cuts only at token boundaries; it is not a Unicode-scalar budget. Non-ASCII HTML is fixture-compared exactly. Any accepted tie-window difference requires a Divergence Register entry.
- Missing index maps to `IndexNotFound`; cancellation maps to `Cancelled`; corrupt/invalid input never panics and uses the shared error taxonomy with subsystem `quill`.
- The read path is synchronous under an immediately-ready `SearchFuture`, because fusion polls lexical work inside a Rayon join. It remains `Send + Sync` and does not create a runtime.
- Durable open validates checksums and path ownership. Repair/quarantine policy does not silently return partial results.

## 7. Conformance classes

### 7.1 RankExact

For equal-statistics configurations:

- identical result IDs and native top-k order, including Tantivy's full `DocAddress` tie-break;
- identical count, pagination, and behavioral outputs;
- bit-identical contract-mode BM25 factor/score outputs when Quill and Tantivy are compared in the same process/platform.

RankExact is the default gate for single-segment and otherwise equal-statistics fixtures. Persisted cross-platform artifacts do not use a universal bit pin for values containing `f32::ln`; they compare engines on the executing platform and then apply `ScoreEpsilon` when platform libm is the only classified cause.

### 7.2 ScoreEpsilon

For layouts where the oracle’s own segment geometry changes floating-point scores:

- result sets are identical;
- relative score delta is at most `1e-4`, using `abs(a-b) / max(abs(a), abs(b), 1e-12)`;
- rank flips occur only within the same deterministic epsilon component. Components are the maximal connected runs in oracle total-order: adjacent oracle scores share an edge when their relative delta is at most `1e-4`, and connected components are assigned before examining Quill order;
- the artifact records why RankExact was not applicable.

Every comparison is classified as exactly one of `RankExact`, `ScoreEpsilon`, or a reviewed Divergence Register entry. “Close enough” without a classification fails the gate.

## 8. Fixture maintenance

`tests/fixtures/quill_language_contract.json` contains:

- exact analyzer token streams and helper outputs;
- canonical parse trees for identifier, short-keyword, natural-language, phrase, Boolean, glob, typed range, and typed set classes;
- scoring/statistics cases;
- behavioral cases for every §3.1 row;
- a query-class corpus that labels genuinely harvested rows separately from constructed Boolean, glob, range, set, and boundary probes;
- `engine::tests::e55_mixed_residency_conformance_is_exact` plus its ignored seeded sibling, which execute the complete E5 mixed-residency query/collector matrix and mutation witnesses.

Fixture IDs are stable and additive. Editing expected behavior requires a contract-version bump and a rationale. The lexical crate test loads the JSON, rejects duplicate or dangling IDs, verifies required query/analyzer classes, introspects the shipping schema, executes analyzer/helper/BM25-order goldens, and proves every §3.1 surface row has coverage. Canonical parse trees, lifecycle rows, deleted-stat transitions, and end-to-end ranking cases remain inputs to the downstream Quill differential runner rather than claims of execution by this loader.

## 9. Update visibility contract (cross-process freshness)

**Owning bead:** `bd-quill-duel-visibility-contract-9rk3`. Bet Q3's "searchable immediately" (plan §6.3) is true **in-process only**: the delta segment (E5) is process-local memory, so the flagship topology — an `fsfs index --watch` daemon plus separate `fsfs search` CLI invocations — observes only sealed **and published** segments: commit-cadence freshness, exactly like the tantivy incumbent. This section pins that truth so no metric, gate, or claim may silently conflate the two topologies.

### 9.1 Visibility classes (normative)

- **InProcess visibility.** A reader in the writer's process observes delta-segment documents immediately after their batch returns (no seal, no publish). This class exists only where a delta exists and only inside that process.
- **CrossProcess visibility.** Any other process observes a document exactly when a MANIFEST generation containing it is durably published (§6.2 of the FSLX registry: temp + claim + two renames + directory fsync). Freshness is **publish-cadence** freshness.

### 9.2 `max_visibility_lag_ms` (configured guarantee)

`QuillConfig::max_visibility_lag_ms` (default 1,000) bounds CrossProcess staleness: once the oldest unpublished change has waited this long since the last durable publication, the writer must run a seal-and-publish barrier instead of waiting for the ordinary cadence. The bound piggybacks the existing seal/commit triggers — it adds no new artifact kind. The MANIFEST v2 `last_publish_unix_s` witness (registry §6.1, amendment v1.0.18) is the durable timestamp the lag is measured against; it is second-granular, so sub-second bounds behave as one second. `fsfs flush` is the operator-facing explicit barrier: it forces the same seal-and-publish on demand. Cross-process freshness is thereby a configured guarantee with a typed surface, not an emergent accident of debounce tuning.

### 9.3 Freshness surfacing

`segment_stats()` carries `published_generation`, `last_publish_unix` (from the MANIFEST v2 witness; absent for v1-era or in-memory images), and `live_writer` (from the D1 LOCK record plus POSIX `kill(pid, 0)` liveness — never from mtime staleness). It also carries the fail-loud quarantine state: `degraded`, `quarantined_segments`, `estimated_missing_docs`, and `unknown_missing_doc_segments`. Status surfaces (`fsfs status`, `--format json` search responses) carry this `index_freshness` block verbatim once the fsfs port lands (e7.2); they must never report InProcess freshness while serving CrossProcess readers or omit a persisted quarantine from degraded output.

### 9.4 Honest-topology measurement rule

Any update-to-searchable latency claim (QG-3 in `quill-perf-gates.toml`) must be measured in **both** topologies: (a) in-process reader (delta-visible), and (b) fresh-process reader (publish-visible). Publishing only the in-process number as "the" freshness figure is a contract violation: it is the topology almost no deployment runs.

### 9.5 Visibility is not durability

A published process-local Delta epoch is visible but uncommitted. A process
failure may therefore lose that epoch even after its FSLX bytes have been
fsynced and installed under their canonical segment name: the segment becomes
durable authority only when a successor MANIFEST references it. Recovery must
ignore an installed-but-unreferenced segment. The configured commit cadence
and `max_visibility_lag_ms` bound this loss window; immediate InProcess
visibility does not shorten it.

A successful seal uses one ordered transition: retain the frozen Delta, build
and fsync the segment, install it, durably publish the successor MANIFEST, then
atomically replace the process-local view with `(new Keeper, replacement
Deltas)`. No cancellation point is allowed between durable MANIFEST authority
and the local epoch swap. Readers that already pinned the old epoch retain the
old Delta; later readers see the sealed Keeper. Thus seal publication has no
visibility gap even though visibility and durability remain distinct states.

### 9.6 Unrepairable-segment quarantine

`QuillConfig::quarantine_on_unrepairable` is false by default. A
durability-enabled library writer therefore preserves the hard open error and
does not mutate an unrepairable segment. The fsfs watch topology enables the
policy because it also owns the mandatory recovery loop: the corrupt FSLX and
repair sidecar are atomically retained under `.quarantine` names, a durable
successor MANIFEST omits the segment, and the watcher synchronously runs the
index-freshness audit before it starts. Every `MissingLexical` finding becomes
an `EnqueueReindex` action; watcher startup fails closed unless the post-reindex
audit is clean. Search-only readers never perform this mutation. Once a
quarantine exists they open the retained healthy generation, log the degraded
state on every open, and expose it through doctor, status, and JSON search
freshness metadata.

The public scorer and collectors execute the currently supported query tree
(`All`, term, exact phrase, Boolean, boost, string and numeric range, string and
numeric set, and bounded glob) over Keeper leaves and every frozen Delta leaf
through one composite BM25 statistics snapshot. Range, set, and each matching
glob field contribute their finite inherited boost exactly once per matching
document. Set values and expanded terms are deduplicated before scoring. Glob
expansion is planned snapshot-globally per field in byte-lexicographic order;
the configured expansion limit applies once to that unique field-term set, not
once per residency leaf. Winners are globally merged and external IDs are
materialized from either residency. The seal guarantee therefore means
continuous query findability, ranking, counts, pagination, and doc-set
membership, not merely continuous internal residency.

The E5 mixed-state conformance corpus holds logical rows, Q1 leaf boundaries,
sealed tombstone history, composite BM25 `N`, field token totals, and term
document frequencies fixed while transitioning from all live rows in Delta,
through one sealed plus one Delta leaf, to all sealed leaves. Every supported
query class runs through full ranking, pagination, exact counting, and
scoreless global-id collection. Rank, score bits, external IDs, native global
IDs, diagnostics, counts, and statistics must be identical in all three
states. Delta upsert, sealed-row replacement, and delete-then-readd histories
are mandatory fixture mutations. A larger deterministic corpus repeats the
same matrix as a seeded, seed-logged nightly test.

Once FSLX construction has produced a complete proposal, the writer retains
the canonical bytes, a zero-timestamp successor MANIFEST, the complete
replacement-Delta table, and the prepared local swap as one pending
transaction. Every currently published non-source shard must appear through
the private-lineage rebind of that exact frozen epoch in the replacement
table; omission and an earlier freeze of the same mutable shard are rejected
before construction so a seal cannot make newer Delta rows disappear. All other
writer mutations are rejected until that transaction finishes. Segment
reconciliation or temp write/fsync/install runs under one
writer mutation guard, so a dropped await cannot race its retry or create a
second temp after the exact canonical segment won. A cancellation before
MANIFEST publication leaves the old Delta visible; `resume_pending_delta_seal`
reuses the retained bytes and logical MANIFEST exactly. Keeper alone stamps the
MANIFEST timestamp and normalizes it when reconciling an ambiguous prior
publication. After MANIFEST success, the local swap and writer-counter advance
contain no await or cancellation checkpoint.

Dropping the seal future during the earlier FSLX-build await is also safe: no
filesystem authority or local visibility has changed. The original seal
request may be submitted again when the caller retained, or can reconstruct,
its sealed and replacement-Delta inputs. The resumable pending-transaction
state begins only after those canonical bytes and the successor proposal have
been retained; calling `resume_pending_delta_seal` before that point correctly
reports that no proposal awaits reconciliation.
