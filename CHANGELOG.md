# Changelog

All notable changes to [frankensearch](https://github.com/Dicklesworthstone/frankensearch) are documented here.

Entries correspond to [GitHub Releases](https://github.com/Dicklesworthstone/frankensearch/releases) unless noted otherwise. Tags that share a commit with another release are called out explicitly. Each entry links to representative commits using full commit URLs.

Scope window: [v1.6.0](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.6.0) (2026-08-14) through HEAD on 2026-08-19 ([`450fc2cfc1be20851678d35313681d1336edceab`](https://github.com/Dicklesworthstone/frankensearch/commit/450fc2cfc1be20851678d35313681d1336edceab)), plus missing version rows for the 1.4.x / 1.5.0 spine that shipped after the old "proposed v1.4.0" draft. **v1.4.1 and v1.4.2 are git tags with no GitHub Release.** v1.4.0, v1.4.3, v1.5.0, and v1.6.0 are published Releases.

## Version Timeline

| Version | Kind | Date | Summary |
|---------|------|------|---------|
| [Unreleased](https://github.com/Dicklesworthstone/frankensearch/compare/v1.6.0...main) | HEAD | 2026-08-19 | Hash-control fuse follow-through, Quill CASS ingest, janitor docs-reorg |
| [v1.6.0](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.6.0) | Release | 2026-08-14 | Hash control no longer presented as semantic search |
| [v1.5.0](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.5.0) | Release | 2026-08-05 | Multi-platform assets; remedy for #31 |
| [v1.4.3](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.4.3) | Release | 2026-08-02 | First complete release since v1.2.5 |
| [v1.4.2](https://github.com/Dicklesworthstone/frankensearch/tree/v1.4.2) | Tag | 2026-08-02 | Windows/macOS installer fixes (no GitHub Release) |
| [v1.4.1](https://github.com/Dicklesworthstone/frankensearch/tree/v1.4.1) | Tag | 2026-08-02 | Replacement tag for empty v1.4.0 assets (no GitHub Release) |
| [v1.4.0](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.4.0) | Release | 2026-07-30 | Quill + native rerank epoch |

---

## [Unreleased] -- development on `main` since [v1.6.0](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.6.0) (as of 2026-08-19)

Compare: <https://github.com/Dicklesworthstone/frankensearch/compare/v1.6.0...main>

114 non-merge commits after the v1.6.0 tag.

### Janitor docs-reorg (2026-08-19)

- Untracked skill-loop scratch and ignored the workspace pattern ([`0980a48242d89d100058fc1d15d8fa5ac7c303f9`](https://github.com/Dicklesworthstone/frankensearch/commit/0980a48242d89d100058fc1d15d8fa5ac7c303f9)).
- Relocated remaining root reports: `UPGRADE_LOG.md` → [`docs/planning/UPGRADE_LOG.md`](https://github.com/Dicklesworthstone/frankensearch/blob/main/docs/planning/UPGRADE_LOG.md) ([`450fc2cfc1be20851678d35313681d1336edceab`](https://github.com/Dicklesworthstone/frankensearch/commit/450fc2cfc1be20851678d35313681d1336edceab)).

### Representative commits

- [`450fc2cfc1be20851678d35313681d1336edceab`](https://github.com/Dicklesworthstone/frankensearch/commit/450fc2cfc1be20851678d35313681d1336edceab) — janitor relocate remaining root reports.
- [`0980a48242d89d100058fc1d15d8fa5ac7c303f9`](https://github.com/Dicklesworthstone/frankensearch/commit/0980a48242d89d100058fc1d15d8fa5ac7c303f9) — untrack skill-loop scratch.

### Hash-control fusion continues past v1.6.0

v1.6.0 stopped presenting hash/FNV/JL generations as semantic search. The follow-on window keeps hash ranks off semantic fields through fuse, persist, explain, and dashboard, and routes hash-control search through lane-aware fuse APIs.

- Hash-control ranks remapped at the RRF fuse boundary ([`00b6452a986f264fcca46c0bfa0588d703dc0202`](https://github.com/Dicklesworthstone/frankensearch/commit/00b6452a986f264fcca46c0bfa0588d703dc0202)); hash-control search routed through lane-aware fuse APIs ([`b1e6409645a079c277b1fb5e26e52402cde28203`](https://github.com/Dicklesworthstone/frankensearch/commit/b1e6409645a079c277b1fb5e26e52402cde28203)).
- Fused lexical-only and hash-only hits are no longer tagged `Hybrid` ([`03e9431d690440233040ccb3b64ef3bd5589d255`](https://github.com/Dicklesworthstone/frankensearch/commit/03e9431d690440233040ccb3b64ef3bd5589d255)).

### Quill CASS schema-general ingest

- Durable create/open bound to an explicit compiled schema ([`1d0dd64c4e7d0bb56ba4bd2a80bcb9e922fb3aff`](https://github.com/Dicklesworthstone/frankensearch/commit/1d0dd64c4e7d0bb56ba4bd2a80bcb9e922fb3aff)).
- Native CASS ingest surface with a Quill-only schema generation ([`6fd85d8022cc9089c5d2e2e81e849db5b5590aa5`](https://github.com/Dicklesworthstone/frankensearch/commit/6fd85d8022cc9089c5d2e2e81e849db5b5590aa5)); `conversation_id` stored as 8 LE bytes and `CassDocument` projected onto `SchemaDocument` ([`869ba30572c39915cfa53b0cb0e962f2587be44e`](https://github.com/Dicklesworthstone/frankensearch/commit/869ba30572c39915cfa53b0cb0e962f2587be44e)).
- Document identity is conversation-scoped, not `source#ordinal` ([`400eabef73bdde38f2058ff03d205e56b3fbb026`](https://github.com/Dicklesworthstone/frankensearch/commit/400eabef73bdde38f2058ff03d205e56b3fbb026)).
- Differential Quill CASS path against the Tantivy incumbent ([`6d899a6f9510ced8a55878c58edf735c410ece88`](https://github.com/Dicklesworthstone/frankensearch/commit/6d899a6f9510ced8a55878c58edf735c410ece88)); dev-only Quill-vs-Tantivy equivalence comparator behind `cass-equivalence` ([`8db4589ee12b529e35898aa644eff8c9dc88e831`](https://github.com/Dicklesworthstone/frankensearch/commit/8db4589ee12b529e35898aa644eff8c9dc88e831)).

### Cancel, transients, and ANN

- Observe cancel before FastEmbed ONNX work ([`572548eada5ef1534045f59bb61aecaa7b39f02e`](https://github.com/Dicklesworthstone/frankensearch/commit/572548eada5ef1534045f59bb61aecaa7b39f02e)).
- Lockstep asupersync 0.4.4 and retry FrankenSQLite transients ([`ad0326ed055503bff4e6e9e1c8423007fd7ef790`](https://github.com/Dicklesworthstone/frankensearch/commit/ad0326ed055503bff4e6e9e1c8423007fd7ef790)); asupersync 0.4.4 → 0.4.5 ([`160a7dc750aa46c4325ee3b20551f2de1c189ad3`](https://github.com/Dicklesworthstone/frankensearch/commit/160a7dc750aa46c4325ee3b20551f2de1c189ad3)).
- Admit HNSW graphs that are weakly connected but not fully directed-reachable ([#32](https://github.com/Dicklesworthstone/frankensearch/issues/32), [`766f45a011ad773df38c3d8079c15f90e1cdcf5b`](https://github.com/Dicklesworthstone/frankensearch/commit/766f45a011ad773df38c3d8079c15f90e1cdcf5b)).

---

## [v1.6.0](https://github.com/Dicklesworthstone/frankensearch/compare/v1.5.0...v1.6.0) -- 2026-08-14

Product CLI `fsfs` 1.6.0. Crate versions stay decoupled (workspace members 0.2.x, facade 0.3.2). 481 commits since v1.5.0.

### Hash control is no longer presented as semantic search

Hash / FNV / JL vector generations are labeled as control artifacts everywhere an operator or agent can read them: doctor, status, dashboards, search JSON (`skip_reason`, `vector_generation_id`, `vector_generation_is_hash`), table/CSV/explain ranks (`[L H]`, `hash_rank`), stream start, and degradation advice (`degrade.advice.hash_control`).

- `auto_detect` / `require_semantic` no longer classify a two-hash stack as `Full`.
- `IndexBuilder` will not write a quality FSVI from a hash embedder.
- `TwoTierSearcher` will not refine quality from hash-control ranks.
- Full search with a hash FSVI and a lexical index continues as lexical plus typed hash-control advice; it does not pretend to be semantic. Full search with no vector index still fails closed (`IndexNotFound`).
- FrankenSQLite edges are lockstep on crates.io **0.3.1** (same 0.3 API: autocommit durability, concurrent-open prepare, freelist safety).

### Storage schema

Current storage `SCHEMA_VERSION` is **7** (FTS5 rebuild-version marker). Fresh databases still bootstrap directly to latest; historical v1–v6 fixtures migrate.

### Not in this release

- Mach-O ArtifactStore F1 (needs Darwin/unsafe).
- Live 50k/100k `--full` corpus numbers were not re-run for this tag.
- Windows assets remain absent (same as v1.5.0: no Windows build host).
- GitHub Actions is not used; binaries are produced with dsr.

---

## [v1.5.0](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.5.0) -- 2026-08-05 (GitHub Release)

Compare: <https://github.com/Dicklesworthstone/frankensearch/compare/v1.4.3...v1.5.0>

728 non-merge commits since v1.4.3. Remedy for [#31](https://github.com/Dicklesworthstone/frankensearch/issues/31): since v1.2.5 the pipeline shipped only `x86_64-unknown-linux-musl`, so `fsfs update` advertised a new version and then 404'd on every other platform. This release ships an asset for every target triple the self-updater and installer construct (except Windows — the build host ran out of disk).

- `st_nlink` widened to `u64` for musl portability ([`9966b6d876b6926631a97ae6c0323444d090c8bb`](https://github.com/Dicklesworthstone/frankensearch/commit/9966b6d876b6926631a97ae6c0323444d090c8bb)).
- `frankensearch-fsfs` package version bumped to 1.5.0 ([`6c1140aa04f6ca1dcdf427edbf93f41082a999ff`](https://github.com/Dicklesworthstone/frankensearch/commit/6c1140aa04f6ca1dcdf427edbf93f41082a999ff)).
- Library API sweep in this window (`LexicalRead` / `LexicalWrite` replacing combined `LexicalSearch`) is breaking for library consumers — hence the minor bump.

---

## [v1.4.3](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.4.3) -- 2026-08-02 (GitHub Release)

Compare: <https://github.com/Dicklesworthstone/frankensearch/compare/v1.4.2...v1.4.3>

First complete GitHub Release since v1.2.5. Native Quill lexical engine in fsfs (no more writer-lock failures on concurrent searches); `fsfs index` terminates cleanly; loader-capable stock default; typed embedding-identity foundation; working Linux self-update (the updater previously requested a nonexistent asset name).

- Self-update requests the lite asset family on lite-only targets ([`a5f464432f7514d54f4e43b31a839b9536ac22f0`](https://github.com/Dicklesworthstone/frankensearch/commit/a5f464432f7514d54f4e43b31a839b9536ac22f0)).
- Release bump that must carry the self-update asset-family fix ([`4446f75bc6978dd65e4ef4cf0d27d578ba4ebbd8`](https://github.com/Dicklesworthstone/frankensearch/commit/4446f75bc6978dd65e4ef4cf0d27d578ba4ebbd8)).

---

## v1.4.2 -- 2026-08-02 (Tag only — no GitHub Release)

> Git tag `v1.4.2` exists; **no GitHub Release was published for this tag.**
> Compare: <https://github.com/Dicklesworthstone/frankensearch/compare/v1.4.1...v1.4.2>
> 8 non-merge commits. Windows-compatible Quill (`stat_dev_as_u64` cfg-gated to unix) and macOS bash 3.2 installer-script compatibility. Tag: [`c74818be81f52031b8106e3331ff3d0eab5a5bed`](https://github.com/Dicklesworthstone/frankensearch/commit/c74818be81f52031b8106e3331ff3d0eab5a5bed).

---

## v1.4.1 -- 2026-08-02 (Tag only — no GitHub Release)

> Git tag `v1.4.1` exists; **no GitHub Release was published for this tag.** The tag message calls it a replacement for the empty v1.4.0 asset set.
> Compare: <https://github.com/Dicklesworthstone/frankensearch/compare/v1.4.0...v1.4.1>
> 323 non-merge commits. Four release-blocking CI failures repaired ([`15353948d620b90b699198aaa4d77ea12f8bdb65`](https://github.com/Dicklesworthstone/frankensearch/commit/15353948d620b90b699198aaa4d77ea12f8bdb65)); fsfs bumped to 1.4.1 ([`0b41ec66a7a74628866cc608ea8785bb1f0f1ef3`](https://github.com/Dicklesworthstone/frankensearch/commit/0b41ec66a7a74628866cc608ea8785bb1f0f1ef3)).

---

## [v1.4.0](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.4.0) -- 2026-07-30 (GitHub Release)

> **Scope window: v1.3.0..v1.4.0 (~1,488 non-merge commits), reconstructed as one epoch.** This window is too
> large to enumerate commit-by-commit, so this section is organized as capability
> waves with representative live-linked commits, reconstructed from `git log`,
> `git diff --stat v1.3.0..v1.4.0`, and the checked-in beads tracker
> ([`.beads/issues.jsonl`](https://github.com/Dicklesworthstone/frankensearch/blob/main/.beads/issues.jsonl)).
> It is a navigation aid, not an exhaustive record.
>
> **Release status:** [v1.4.0](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.4.0) is a published GitHub Release (2026-07-30). v1.3.0 remains a git tag with **no** GitHub Release. The previous published release before this epoch was [v1.2.5](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.2.5) (2026-04-08). Crate versions are intentionally decoupled from the v1.x tag
> series: workspace member crates are at 0.2.x and the `frankensearch` facade
> crate at 0.3.2 while the repo tag series continues at v1.x.
>
> This section was originally drafted as "Unreleased — proposed v1.4.0" before the tag shipped; the body is the v1.4.0 reconstruction and is kept here as the version row.

### Post-publication scope correction — 2026-07-30

- **Native ANN scope:** v1.4.0 landed the in-tree native HNSW engine and its
  FSVI-bound graph and receipt foundations, but it has not yet replaced the
  shipping ANN path. `TwoTierIndex` and the `ann` feature still use the
  git-pinned `hnsw_rs` adapter; native production wiring, tombstone handling,
  and external-dependency removal remain follow-up work.
- **Embedding-integrity scope:** v1.4.0 landed canonical identity types,
  remote-vector attestation, zero-signal classification, sink-level
  validation, and FSVI v2 admission foundations. Enforcement is not yet
  end-to-end: fusion and fsfs still contain raw-vector and
  model-ID/dimension-only seams, current production writers still emit legacy
  FSVI v1, and generic auto-detection can return an explicitly degraded
  `HashOnly` stack. “Fail closed” therefore describes the implemented trust
  boundaries, not every production ingest and search boundary.

### Quill: Native Pure-Rust Lexical Engine (new crate `frankensearch-quill`)

The dominant workstream of the epoch: a ground-up, memory-safe BM25 lexical
engine built to replace Tantivy inside frankensearch (design doc:
`COMPREHENSIVE_PLAN_FOR_THE_QUILL_LEXICAL_ENGINE.md`). Epics E0–E5 of the
`bd-quill-*` family are closed (89 closed issues in the quill epic family);
E6 (gauntlet at scale), E7 (default flip), E8 (perf doctrine), and E9 (Tantivy
retirement) remain open — Quill is the recommended `quill` feature path, while
Tantivy stays in-tree behind `lexical-tantivy` as the pinned conformance oracle.

- Scribe ingest pipeline: term interner + bump arenas, columnar accumulator, deterministic radix flush kernel ([ba26e5dc](https://github.com/Dicklesworthstone/frankensearch/commit/ba26e5dcc6679f9e4cb272004a2a197a1e4232b4))
- Grimoire/Quiver formats: prefix-block term dictionary ([c497bac6](https://github.com/Dicklesworthstone/frankensearch/commit/c497bac61115784b57894dc27b30819850edc723)), canonical posting codec, block-max skip tables, positions codec, IDMAP/IDHASH ([db1bf12c](https://github.com/Dicklesworthstone/frankensearch/commit/db1bf12c839938205a52249f361558ad19fa833e)), STOREDMETA stored fields
- Keeper lifecycle: FSLX segment container ([eae50313](https://github.com/Dicklesworthstone/frankensearch/commit/eae5031316194c11d79a82ba7ff09833dc00775e)), atomic manifest keeper ([a60bb8b3](https://github.com/Dicklesworthstone/frankensearch/commit/a60bb8b39596162c7c51b50f3946e3bfd3f28bef)), crash-only recovery and GC ([f3ab5dbe](https://github.com/Dicklesworthstone/frankensearch/commit/f3ab5dbe97bb093a5de7e59b36095d8ca4148458)), Q1-preserving tombstone compaction ([f1081753](https://github.com/Dicklesworthstone/frankensearch/commit/f108175316b011ea56fe0f0f8ad572ad1e8b659c)), tiered Keeper lifecycle ([bc210b77](https://github.com/Dicklesworthstone/frankensearch/commit/bc210b777f60286bac06cfc85b21e9dd135406ba))
- Argus query layer: native lenient + CASS query parsers ([9c8ebecd](https://github.com/Dicklesworthstone/frankensearch/commit/9c8ebecd77ebb35e46baa4dd71ef92e230e5d001)), exhaustive BM25 scorer correctness anchor ([522d2d3c](https://github.com/Dicklesworthstone/frankensearch/commit/522d2d3ce4aebcd8a902aeeadc32bf8111d31bda)), exact phrase scorer ([b4500dc1](https://github.com/Dicklesworthstone/frankensearch/commit/b4500dc126394264f3fe8df491aa3ad78edffa99)), native snippet generation ([6f78b38f](https://github.com/Dicklesworthstone/frankensearch/commit/6f78b38f867f24ef7da91a41e6268481b199d560))
- Searchable-while-indexing delta segments: mutable delta ([73bde1d6](https://github.com/Dicklesworthstone/frankensearch/commit/73bde1d644b872f33a736a534b443aa2aee5bb0d)), epoch sealing without visibility gaps ([327b1a6b](https://github.com/Dicklesworthstone/frankensearch/commit/327b1a6b68e29d28b2282f9c1b06b7dcc974b147))
- Operational safety: cross-process writer admission ([554c9ef2](https://github.com/Dicklesworthstone/frankensearch/commit/554c9ef2398e9b29fffe84e90d3340edd8606698)), blue-green CURRENT pointer machinery ([8ba6cab4](https://github.com/Dicklesworthstone/frankensearch/commit/8ba6cab402855efc7a5180d148d3fdcfdf5f1842)), unrepairable-segment quarantine ([3fa54a8a](https://github.com/Dicklesworthstone/frankensearch/commit/3fa54a8a2cd6ef5f5c767d89ed188b8004df8fc0)), content-witness resumable bulk builds ([6c5478dc](https://github.com/Dicklesworthstone/frankensearch/commit/6c5478dc4ca4b49f3325914ea41f3a615797b0ad)), dark-launch shadow oracle ([7e3a8dbf](https://github.com/Dicklesworthstone/frankensearch/commit/7e3a8dbfe3bcc8b57cbdcd94a9e7ddacffd03549)), deterministic query fuel metering ([ae5baa0d](https://github.com/Dicklesworthstone/frankensearch/commit/ae5baa0dbc433f3a92591f93e8f652e406ecdbeb))
- Integration flip machinery (E7, in progress): lexical backend contract ([9c468af6](https://github.com/Dicklesworthstone/frankensearch/commit/9c468af6cc6723f273b2bd462ebefe8a3f6f1a34)), fsfs lexical runtime ported to Quill ([ab4dc2bd](https://github.com/Dicklesworthstone/frankensearch/commit/ab4dc2bdab8da62f24016672853ebfe9e012b8f1)), blue-green Tantivy upgrade path ([e708abad](https://github.com/Dicklesworthstone/frankensearch/commit/e708abad2c79d63a0c1db186b76ba4d1a3d1e8a1)), committed performance ratchet ([cc36e146](https://github.com/Dicklesworthstone/frankensearch/commit/cc36e146038d24d018dd8f6d2d1f3bb2e59018c4))

### Quill Gauntlet: Differential Verification (new crate `frankensearch-quill-gauntlet`)

A dedicated differential-testing crate that certifies Quill against pinned
Tantivy as an oracle before any default flip.

- Conformance gauntlet skeleton ([1139973a](https://github.com/Dicklesworthstone/frankensearch/commit/1139973a5da74b0930132f454646fa7a6ba82c7b)) and deterministic corpus + query generators for differential campaigns ([d9fd0743](https://github.com/Dicklesworthstone/frankensearch/commit/d9fd074366b4a5ea04865bb615df8027d8beb8d6))
- ddmin divergence shrinker with auto-triage ([cc78425c](https://github.com/Dicklesworthstone/frankensearch/commit/cc78425c1b199a0e5e85d70c6091869c2801ecd2)) and append-only divergence ledger ([d3b5b303](https://github.com/Dicklesworthstone/frankensearch/commit/d3b5b303616146e47a13b25c97c8611040743f5a))
- Exact CASS oracle campaigns ([ace575cb](https://github.com/Dicklesworthstone/frankensearch/commit/ace575cb84a94a8591d418a93335a4da0b81c263)); versioned QG evidence layer with sealed atomic artifacts and metric-specific estimands ([dc301b28](https://github.com/Dicklesworthstone/frankensearch/commit/dc301b28d2ee52ed6cecebd8af14ae79c7d20827)); repaired paired estimators wired into the bench harness ([8912f04f](https://github.com/Dicklesworthstone/frankensearch/commit/8912f04fd380670d75d4bfe4baebf5fc1051a18c))

### Native Cross-Encoder and Embedder (frankentorch) — ONNX/ort Retired from the Default Rerank Lane

- Replace the ONNX/ort cross-encoder with a pure-Rust frankentorch `NativeReranker` ([e717b8b4](https://github.com/Dicklesworthstone/frankensearch/commit/e717b8b4785a3e087e2ca17c22c946b4335b268e)), pinned by git rev so the `native` feature is consumable ([2eaf7539](https://github.com/Dicklesworthstone/frankensearch/commit/2eaf753955f58d8ce0f6203224d1ee2759b7cc49))
- Pure-Rust `NativeEmbedder` (all-MiniLM-L6-v2) ([a18943de](https://github.com/Dicklesworthstone/frankensearch/commit/a18943de844abfc63fb711ad23b10f5f0761ccc1))
- Kernel work to beat the ONNX baseline: tape-free fully-fused encoder layer ([30084c3d](https://github.com/Dicklesworthstone/frankensearch/commit/30084c3d8ef79c036dfee8d233e58203c6f29b79)), raw gemm-based attention ([e022c2b3](https://github.com/Dicklesworthstone/frankensearch/commit/e022c2b348a65573923b6bd6affc5ca55eed90f0)), fused softmax/GELU/LayerNorm and int8 GEMM paths
- `fastembed-reranker` remains available as an optional feature for the previous FlashRank-style lane

### SIMD and Systems Performance Offensive

Hundreds of measured, bit-identity-gated optimizations across every crate,
recorded pass-over-pass in `docs/PERF_LEDGER.md` with rejected hypotheses in
`docs/NEGATIVE_EVIDENCE.md` (the ledger discipline itself is a deliverable of
this epoch: ~150+ negative-evidence entries prevent re-litigating dead ends).

- Runtime-dispatched AVX2/F16C kernel SELF-SPEEDUP maintenance versus the preceding frankensearch paths: f16 dot products 3.6–4.0x ([7239d585](https://github.com/Dicklesworthstone/frankensearch/commit/7239d585e765a430e0acff5a573a6ddbaf12f936)), 4-bit slab pack 10.3–13.6x ([dc60d618](https://github.com/Dicklesworthstone/frankensearch/commit/dc60d618c649e33d2d596abb44750b65f3ed3bad)), FSVI slab write 6.4–7.3x ([2a4d3334](https://github.com/Dicklesworthstone/frankensearch/commit/2a4d333445b14e80f3d7d865516056b2619efbf5)), 384-dim specializations ([cb0bb785](https://github.com/Dicklesworthstone/frankensearch/commit/cb0bb785771d1fae6a87d922f0c3bffb7ba9564e))
- Lossless quantized search: FSVI 4-bit two-pass returned the same top-k as an exact full-precision scan in a 32/32 third-party BLAS-class exhaustive-search check, at one eighth the pass-1 working set ([conversion card](docs/evidence/fsvi-4bit-vs-incumbent-20260731.md)), wired into the sync fast tier ([226814a1](https://github.com/Dicklesworthstone/frankensearch/commit/226814a1298959f1265afb878a3a65ed76d9044d)); SELF-SPEEDUP maintenance versus preceding frankensearch paths: parallelized MRL truncated scan 8.64x ([31c3d9cf](https://github.com/Dicklesworthstone/frankensearch/commit/31c3d9cfbd2fe1bfcda421937b00338256f46645)); selective-filter gather fast path 6.9–50x on filtered search ([ec76859a](https://github.com/Dicklesworthstone/frankensearch/commit/ec76859ace6dec12e5d569cc4b27781528845495))
- Fusion-path SELF-SPEEDUP maintenance versus preceding frankensearch paths: doc_id moves instead of clones (7.8–21.5x on the fuse step) ([832c2613](https://github.com/Dicklesworthstone/frankensearch/commit/832c261396c54d936615bd2200153aafe9ccb04e)), merge-structured `rrf_fuse` 1.31–1.46x ([4aeb66b1](https://github.com/Dicklesworthstone/frankensearch/commit/4aeb66b1004e8e23fdfdd948449c19fa35bc8258))
- Analyzer/canonicalizer SELF-SPEEDUP maintenance versus preceding frankensearch paths: ASCII NFC fast path, ~45–368x on the analyzer hot path ([9d7e8d00](https://github.com/Dicklesworthstone/frankensearch/commit/9d7e8d0022b9351bd586ff34e9e6057896074fb7)); S3-FIFO query-embedding cache ([b83b25d6](https://github.com/Dicklesworthstone/frankensearch/commit/b83b25d637e393692d5b9086692888cdaf5cb2e2)); Tantivy fast-field id materialization up to 6.32x ([14e87e4a](https://github.com/Dicklesworthstone/frankensearch/commit/14e87e4aa7e6cfb26078c58c5e1dc27b72c7d859))
- Representative negative evidence: HNSW route-next refuted at production scale ([b8aec7b2](https://github.com/Dicklesworthstone/frankensearch/commit/b8aec7b292fe1d7c9284c6f7cfe1dc5aeab52de1)), AVX-512/VNNI ruled out as hardware-unavailable ([da7de808](https://github.com/Dicklesworthstone/frankensearch/commit/da7de808455d1aa094079bac9d39a23743688e5d))

### HNSW / ANN Maturation

- Persist native HNSW graph sidecars instead of rebuilding on load ([b5c3bab4](https://github.com/Dicklesworthstone/frankensearch/commit/b5c3bab4acc0ec6f6767c4cc522d28d0f5c03359)), with a v2 sidecar guard against silent vector swaps ([acf3f866](https://github.com/Dicklesworthstone/frankensearch/commit/acf3f866c96903bec986ea153a3d130567622c6a))
- Native in-tree HNSW graph engine replacing the external dependency ([0b59e600](https://github.com/Dicklesworthstone/frankensearch/commit/0b59e6008fb17c403ad836361811aed34da4156f))
- Correctness hardening: atomic generation publish ([a816f6d9](https://github.com/Dicklesworthstone/frankensearch/commit/a816f6d93b408f5dece7bcb9f13e38c8c4778ac4)), per-vector fingerprinting ([f96a9008](https://github.com/Dicklesworthstone/frankensearch/commit/f96a900885128cb6192b14b563f2dac2fa84fdc5)), READY-generation recovery ([b23dbe4e](https://github.com/Dicklesworthstone/frankensearch/commit/b23dbe4e51103b175e201aa069a6706a9cbb4f07)), duplicate-row-preserving repair ([95a636f9](https://github.com/Dicklesworthstone/frankensearch/commit/95a636f95d12da436a8453f685ef6b09c76f033c))

### Embedding Integrity: Fail-Closed Trust Boundaries

- Fail closed on unverifiable embedding identity ([ab0f69f8](https://github.com/Dicklesworthstone/frankensearch/commit/ab0f69f8c966844dbd5e595a07c1c317a8ca2e2d)) with frozen canonical embedding identity contracts ([d615076c](https://github.com/Dicklesworthstone/frankensearch/commit/d615076c15c2f6ffee975eb637638821dfd09b83))
- Authenticate remote API vectors ([46ebeebf](https://github.com/Dicklesworthstone/frankensearch/commit/46ebeebf72c3d7b1fec2101be5a21d0f49df87b9)) and daemon vectors ([e580d460](https://github.com/Dicklesworthstone/frankensearch/commit/e580d4604ee9b73b800620fa1db1160d2dc08f2d)); acquire frozen models atomically ([e2851547](https://github.com/Dicklesworthstone/frankensearch/commit/e2851547c2598616dd229afa89487370436c890b))
- Typed semantic zero-signal vocabulary ([7a8c6eb9](https://github.com/Dicklesworthstone/frankensearch/commit/7a8c6eb9efba6165637f1645f2020b66fbd073cc)); reject unusable embeddings at the ingestion sink ([df91954b](https://github.com/Dicklesworthstone/frankensearch/commit/df91954b2bed942db5d41caff4904619bc0a157b)); make degraded embedder stacks observable instead of silent ([5097f545](https://github.com/Dicklesworthstone/frankensearch/commit/5097f545c627c58f8f0d148f8b105128dbbedaed))
- Lift the 16 MiB download cap so built-in model artifacts fetch (closes [#27](https://github.com/Dicklesworthstone/frankensearch/issues/27)) ([0ca8dc45](https://github.com/Dicklesworthstone/frankensearch/commit/0ca8dc457f4fce20f3e6eeed337dc7a3b1f9d1d2))

### fsfs Operability Wave (bd-pkl0 roadmap)

A May burst delivering the "agent-scale resilience and operability" roadmap
(epic `bd-pkl0`, closed with 15+ children), followed by lifecycle hardening.

- New diagnostic/ops surfaces: self-calibrating profile report ([0581b039](https://github.com/Dicklesworthstone/frankensearch/commit/0581b03940ee9c04af03b06249908d3430a8648a)), index freshness audit ([7c82d27f](https://github.com/Dicklesworthstone/frankensearch/commit/7c82d27f166bbc832ba395da426b6e6f4e421403)), resource pressure governor ([3c7cda27](https://github.com/Dicklesworthstone/frankensearch/commit/3c7cda27e0b9ed053f959564f38434babb7970bd)), index footprint advisor ([ea64a0f0](https://github.com/Dicklesworthstone/frankensearch/commit/ea64a0f05a58dba761601b9435a6a5a577f81b55)), benchmark drift dashboard ([a4d3cec8](https://github.com/Dicklesworthstone/frankensearch/commit/a4d3cec852a8cc862b531ca042742c012cfc460c)), corpus privacy preflight, model cache diagnostics, degraded incident suite, query-plan metamorphic suite
- Incremental change evaluator and contract-parity validators ([468488f0](https://github.com/Dicklesworthstone/frankensearch/commit/468488f0f0a1eb54f89c025b05c3f4acd9e00bd2)); per-feature CI smoke lanes ([d305eda2](https://github.com/Dicklesworthstone/frankensearch/commit/d305eda22ab2e7d358cde8d6c79089eb0e41a886))
- Streaming search command with a phase-sink, flush-visible writer ([bd1821ce](https://github.com/Dicklesworthstone/frankensearch/commit/bd1821ce1c5a5ba3e69e4fefd514208135c51660)); resumable interrupted indexing ([b596cca8](https://github.com/Dicklesworthstone/frankensearch/commit/b596cca8c6f2bb6f9775823e48d72c310fdfc806)); shutdown-aware concurrent serve accept loop ([77b68c7f](https://github.com/Dicklesworthstone/frankensearch/commit/77b68c7faefbdc0a812808a0d86809c05d6a2ba1)); executable lifecycle commands ([c5bd2c6d](https://github.com/Dicklesworthstone/frankensearch/commit/c5bd2c6d3694c9b1f29f5632f49f35c3dceb8cc3))

### CASS Compatibility and Lexical Scale

- Bounded-merge API to avoid `vm.max_map_count` exhaustion on huge rebuilds ([ceaba154](https://github.com/Dicklesworthstone/frankensearch/commit/ceaba154449b931fd58d9b1c1c90be7673e5f38c))
- Query-semantics hardening: wildcard regexes fail closed ([3b65ab32](https://github.com/Dicklesworthstone/frankensearch/commit/3b65ab3270a3ae8e52410ab726fdeff1de3b17e1)); standalone NOT complement semantics preserved ([054dab0d](https://github.com/Dicklesworthstone/frankensearch/commit/054dab0d92ea52149efeabfadc70785db0830058)); ASCII-token CJK allocation elision ([a982f33a](https://github.com/Dicklesworthstone/frankensearch/commit/a982f33a5b5109f6fc6be7853dd4598f35a31482))
- Feature-graph rename: `cass-compat` is now an alias of `lexical-tantivy` (which implies `lexical`); the CASS schema-v8 Tantivy adapter is an explicitly foreign-format lane outside default builds

### Hybrid Quality Harness

- Statistically disciplined retrieval-quality evidence (`docs/quality_harness/`): tier-asymmetry finding — lexical decisive on 4/4 corpora, dense tier ~4.4x smaller contribution ([8a90fa13](https://github.com/Dicklesworthstone/frankensearch/commit/8a90fa13bcb558bd1240822810d028cdc26bedd6)); bootstrap-CI dense-tier marginal value ([e0dab5cd](https://github.com/Dicklesworthstone/frankensearch/commit/e0dab5cde0b39fd7abdf07e131a9075979b4d900)); pool-size-dependent fusion comparisons ([91abbe8d](https://github.com/Dicklesworthstone/frankensearch/commit/91abbe8d898211da6aac1bed8228f5cd088c2831))

### Toolchain, Dependencies, and Packaging

- asupersync tracked through the 0.3.x series: 0.3.4 alignment ([f130ec4c](https://github.com/Dicklesworthstone/frankensearch/commit/f130ec4cffbcada856776140812db39bd9ae62e3)), floored at 0.3.9 to exclude a sleep regression ([b3b5d618](https://github.com/Dicklesworthstone/frankensearch/commit/b3b5d6185d26fdc2ce9efd3a15a0793976c1a75b))
- crates.io republish of the cass-closure crates: members 0.2.1, rerank 0.2.2, umbrella `frankensearch` 0.3.1/0.3.2 ([c8fd6654](https://github.com/Dicklesworthstone/frankensearch/commit/c8fd66544614b434f82eab86e86e189ea284dada), [2cad158f](https://github.com/Dicklesworthstone/frankensearch/commit/2cad158f4468ece7076e3fe529c8e5c20b2e020e))
- Vendored OpenSSL / forced rustls TLS for fastembed/ort/hf-hub ([10383e3d](https://github.com/Dicklesworthstone/frankensearch/commit/10383e3dd0310c8ca7c17a2bd06f9275ed192a06)); test-internals de-leaked from production dependency graphs ([e82ba84c](https://github.com/Dicklesworthstone/frankensearch/commit/e82ba84cd76ad7456587e524b8e491625cd3f127))

---

## v1.3.0 -- 2026-04-24

> **Git tag only — no GitHub Release was published for v1.3.0.** -- [Full diff from v1.2.5](https://github.com/Dicklesworthstone/frankensearch/compare/v1.2.5...v1.3.0)
>
> 84 commits (2026-04-08 through 2026-04-22). Workspace version bump: 11 member
> crates to 0.2.0, fsfs to 1.3.0, `frankensearch` binary crate to 0.3.0
> ([3dbab624](https://github.com/Dicklesworthstone/frankensearch/commit/3dbab624fd05d4b22dac8ad4b8b02bee49db6b39)).

### CASS-Compat Lexical Indexing Overhaul

- Custom `CassTokenizer` with unified content prefix/preview builder; schema v7 hash bump ([84566328](https://github.com/Dicklesworthstone/frankensearch/commit/84566328322e0ae95982a9f46aacf040fae75c90))
- Writer scaling: `CASS_MAX_WRITER_THREADS` raised to 32, batched adds via `Writer::run`, bulk-load merge threshold lifted to 256 ([4be27b96](https://github.com/Dicklesworthstone/frankensearch/commit/4be27b963e82d436f926db5682dccb227ff59dd5))

### fsfs Contract Parity

- Pressure profile contract types, `Reranked` search phase, and config contract schema ([c146fdfe](https://github.com/Dicklesworthstone/frankensearch/commit/c146fdfe495fb7231ea9af047e3fca2818cce458)); `reranked` phase wired through search_events with explanation goldens ([4ef8e090](https://github.com/Dicklesworthstone/frankensearch/commit/4ef8e0904d0ac6f953065506e6e323344e1fb257))

### Runtime & Dependencies

- asupersync moved from a git pin to crates.io v0.3.0, then 0.3.1 ([f6dc73aa](https://github.com/Dicklesworthstone/frankensearch/commit/f6dc73aac0343358b213743cefcba0c679669749))
- `PR_SET_PDEATHSIG` installed on spawned search daemons so orphans cannot outlive their parent ([0134f073](https://github.com/Dicklesworthstone/frankensearch/commit/0134f0735794ae47f311f571cc1b783fc93d8f2d))

---

## [v1.2.1 – v1.2.5](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.2.5) -- 2026-04-07/08

> **Five CI-stabilization respins, each published as a GitHub Release**
> ([v1.2.1](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.2.1),
> [v1.2.2](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.2.2),
> [v1.2.3](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.2.3),
> [v1.2.4](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.2.4),
> [v1.2.5](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.2.5)) --
> [Full diff from v1.2.0](https://github.com/Dicklesworthstone/frankensearch/compare/v1.2.0...v1.2.5)
>
> Cut in rapid succession to get green release binaries out after v1.2.0:
> flaky ops/ingestion test fixes and removal of ort-incompatible build targets
> ([0a325fa5](https://github.com/Dicklesworthstone/frankensearch/commit/0a325fa5654cb59d8e6fa400adbc609575202e70)),
> proptest temp-path fixes for macOS CI and nightly-clippy `tempfile` adaptations
> ([b6fbf0db](https://github.com/Dicklesworthstone/frankensearch/commit/b6fbf0db922f5e77e6f9ebaf8e8f09dde8b5e0ea),
> [c71cd45f](https://github.com/Dicklesworthstone/frankensearch/commit/c71cd45fb99b42acce2d2d8ae514dc16903902c2)).
> v1.2.5 is the latest GitHub Release as of this changelog's reconstruction.

---

## [v1.2.0](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.2.0) -- 2026-04-07

> **CJK search, performance improvements, async embedder fix** -- [Full diff from v1.1.7](https://github.com/Dicklesworthstone/frankensearch/compare/v1.1.7...v1.2.0)
>
> Lightweight tag. Published as a GitHub Release.

- CJK bigram tokenization for Chinese/Japanese/Korean search ([b7a6cf61](https://github.com/Dicklesworthstone/frankensearch/commit/b7a6cf61c730b140793ce628adb813360f9572c3))
- Raw reranker logit propagation, `config set/reset`, and WAL compaction on shutdown ([fa945601](https://github.com/Dicklesworthstone/frankensearch/commit/fa94560180bdef2e143447749533f8f22e7ff228))
- Parallelized embedding + lexical search via `rayon::join` with `poll_immediate` for sync-in-async futures ([cba3632a](https://github.com/Dicklesworthstone/frankensearch/commit/cba3632a794f73bf0fa4a0e3925963f130b7ba90))

---

## v1.1.5, v1.1.6, and [v1.1.7](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.1.7) -- 2026-03-22/23

> **v1.1.5 and v1.1.6 are git tags only (no GitHub Releases); v1.1.7 was
> published as a GitHub Release.** -- [Full diff from v1.1.4](https://github.com/Dicklesworthstone/frankensearch/compare/v1.1.4...v1.1.7)
>
> Compilation/CI respins of the v1.1.4 feature set: adaptation to the
> fsqlite-types 0.1.2 API (`Arc<str>`/`Arc<[u8]>` values, `Cx` type)
> ([37dc7e18](https://github.com/Dicklesworthstone/frankensearch/commit/37dc7e185f6e3e8a67258b56133dcb79ca3e59f9)),
> warnings-as-errors and OpenSSL cross-compilation fixes
> ([007ffadb](https://github.com/Dicklesworthstone/frankensearch/commit/007ffadb970e541ea929cd6c85ecfee1f6dcf077)),
> and partial release publishing when some targets lack ORT prebuilts
> ([fb623f37](https://github.com/Dicklesworthstone/frankensearch/commit/fb623f37f33b9186ac5a3a6900762a99b3823a8c)).

---

## [v1.1.4](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.1.4) -- 2026-03-22

> **Release with binary assets for all platforms** -- [Full diff from v1.1.3](https://github.com/Dicklesworthstone/frankensearch/compare/v1.1.3...v1.1.4)
>
> The CI was fixed in March to include aarch64-linux-musl and x86_64-apple-darwin targets,
> but no release was tagged after v1.1.3. This release makes binaries available.
> Closes #7, closes #23.
>
> Commits since v1.1.3 (2026-02-23) through v1.1.4 (2026-03-22).

### Cloud API Embedding Providers

- Add pluggable cloud API embedding abstraction supporting OpenAI and Gemini backends, with HTTP transport, automatic retry, token-bucket rate limiting, and L2 normalization ([e5b7bab](https://github.com/Dicklesworthstone/frankensearch/commit/e5b7bab7d6a303c3503b6b7e99509d94b484c812), [d37d506](https://github.com/Dicklesworthstone/frankensearch/commit/d37d506da533fa7ded49ceb0b36af3e5214e0a40))
- Support query-param authentication for Gemini via `request_url()` trait method ([9846bb0](https://github.com/Dicklesworthstone/frankensearch/commit/9846bb03d4e1097138960fd500b44b83e4da426e))
- Fix rate limiter token drain and separate OpenAI/Gemini auto-detect paths ([e57f2de](https://github.com/Dicklesworthstone/frankensearch/commit/e57f2de8c4fe0f5e557a58be8e5e399acb03834e))
- Thread `Cx` through download and API embedding pipelines for asupersync 0.2.9 cancellation support ([8e369b4](https://github.com/Dicklesworthstone/frankensearch/commit/8e369b4c640510ea109d751659bd73ff2dde8f2a))

### Tokenizer & Search

- Preserve hyphenated bead IDs (e.g. `bd-q3fy`) in cass tokenizer by replacing `SimpleTokenizer` with a regex tokenizer and `HyphenDecompose` filter; schema bumped v6 to v7 ([11db96a](https://github.com/Dicklesworthstone/frankensearch/commit/11db96ae541659f7422c9b66f98c52f9d294b872))
- Use `[a-zA-Z0-9]` instead of `\w` in tokenizer regex to exclude underscores, fixing index/query mismatch for underscore-separated terms ([472322f](https://github.com/Dicklesworthstone/frankensearch/commit/472322fdcedea90d72209e79f845875e1d8703e3))

### In-Memory Vector Index

- Add fully-resident in-memory vector index with f16 quantization, enabling use cases that skip disk entirely ([eee7b73](https://github.com/Dicklesworthstone/frankensearch/commit/eee7b73dee846d4601cee9b8b67ad66147e4f880))
- Add synchronous two-tier search API alongside in-memory index improvements ([e081bc5](https://github.com/Dicklesworthstone/frankensearch/commit/e081bc578ada4ed9e9dab06476b3737e3dad6541))

### WAL-Based Incremental Mutations (fsfs CLI)

- Add `append-batch`, `delete`, `compact`, and `daemon` commands for WAL-based incremental index mutation without full rebuilds ([0fadc4d](https://github.com/Dicklesworthstone/frankensearch/commit/0fadc4d3fdfc7c5f0ccafcbf5eb3f93747e375af))
- Document WAL commands in help text and shell completions ([1b88d5e](https://github.com/Dicklesworthstone/frankensearch/commit/1b88d5e46e494116defb13dde03499f22397c2bc))

### Async Runtime Compatibility

- Rename `LockError::PolledAfterCompletion` to `Cancelled` across all crates for clarity ([5151d47](https://github.com/Dicklesworthstone/frankensearch/commit/5151d473b1ff09836c89e381ed49f317f4ec7ffb))
- Handle `PolledAfterCompletion` LockError variant in embed, lexical, and rerank crates ([1359551](https://github.com/Dicklesworthstone/frankensearch/commit/1359551eaaa8c098474083738e68f8593456921c))
- Update `asupersync` to 0.2.8 and then 0.2.9 ([ba3ab85](https://github.com/Dicklesworthstone/frankensearch/commit/ba3ab85f0c8a0d713706d15556101cf32fce9683), [ede9fc8](https://github.com/Dicklesworthstone/frankensearch/commit/ede9fc8ec4f64f8e452e02ce9a0fbbd97535c0a8))
- Remove 6 unreachable duplicate `Cancelled` match arms across lexical, rerank, and embed crates ([3e6b29e](https://github.com/Dicklesworthstone/frankensearch/commit/3e6b29e4a6898ebc9ae9d8dea50eb95e2958c3e3))

### Build & CI

- Decouple release CI builds from quality gate so tag pushes produce artifacts ([9b9accb](https://github.com/Dicklesworthstone/frankensearch/commit/9b9accb51ea9c2c264aec428df886c91f71fc471))
- Exclude `tools/optimize_params` from default build; add `--recurse-submodules` to install.sh ([35becff](https://github.com/Dicklesworthstone/frankensearch/commit/35becff5ceda7865e49f3c4923410a6bcb1c61e3))
- Remove invalid `const fn` qualifiers from phase gate builder methods ([900898b](https://github.com/Dicklesworthstone/frankensearch/commit/900898b8c770df2b758793331870e82c8fd607c9))

### Tests

- Add comprehensive test coverage for fsfs update verification path ([da7acf5](https://github.com/Dicklesworthstone/frankensearch/commit/da7acf55a633b5a28e6f82118bd14b4b43ce1383))

---

## [v1.1.3](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.1.3) -- 2026-02-23

> **Bug fixes, PDF extraction, lite builds** -- [Full diff from v1.1.2](https://github.com/Dicklesworthstone/frankensearch/compare/v1.1.2...v1.1.3)
>
> Annotated tag. Published as a GitHub Release.

### PDF & Document Processing

- Native PDF text extraction -- `fsfs index` and `fsfs search` can now process PDF files directly without external tooling ([efc26cc](https://github.com/Dicklesworthstone/frankensearch/commit/efc26cc2fd5b6444efb1efcfa45cabb8a030982e))

### Lite / Offline Builds

- `embedded-models` feature flag for lite/offline builds that bundle models at compile time ([9fbdbd6](https://github.com/Dicklesworthstone/frankensearch/commit/9fbdbd6a82c4a0443aef5756c5b433d24cdd42d7))

### Search Quality & Observability

- Rank movement explanations -- `TwoTierSearcher` now surfaces why a result moved between Phase 1 and Phase 2 ([117f955](https://github.com/Dicklesworthstone/frankensearch/commit/117f95539b4387c1a1e96516618cfc386d663b59))

### Installer & Update Pipeline

- Beautiful download progress with file-size display ([2886323](https://github.com/Dicklesworthstone/frankensearch/commit/2886323fa2de42738e3e51681ed2f1f8fef7f53a))
- Fix six security and correctness issues in installer and update logic ([97ac428](https://github.com/Dicklesworthstone/frankensearch/commit/97ac428f729eb830e06f19f7f6b04f2049f9506f))
- Harden release asset URL and checksum handling ([20aa045](https://github.com/Dicklesworthstone/frankensearch/commit/20aa045be191d4ce76b7dbf490362c7ae906e9b3))

### Bug Fixes

- Fix seven code-review bugs: ANSI box border rendering, `.max` vs `.min` confusion, `pdf_extract` panic guard, model ID lookup, else-if cleanup, `--expand`+`--daemon` warning ([f21ba72](https://github.com/Dicklesworthstone/frankensearch/commit/f21ba7225e0c181c0eafd1e161236a8eb4802507))
- Improve searcher module reliability ([97e340d](https://github.com/Dicklesworthstone/frankensearch/commit/97e340da0ddff51ba2bdb2a76b8bab018f6b151a))

---

## [v1.1.2](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.1.2) -- 2026-02-22

> **Fix version reporting, update mechanism, and TUI visibility** -- [Full diff from v1.1.0](https://github.com/Dicklesworthstone/frankensearch/compare/v1.1.0...v1.1.2)
>
> Lightweight tag. Published as a GitHub Release.

### Release & Update Infrastructure

- Overhaul release asset system with proper Windows support and SHA256SUMS, refactor pipeline dedup logic ([7f4f8c7](https://github.com/Dicklesworthstone/frankensearch/commit/7f4f8c732f0918adcf3a3ce974fa84bf78dbae6b))
- **Version reporting** -- binary now correctly reports its actual version instead of `v0.1.0`
- **Update mechanism** -- `fsfs update` constructs correct download URLs matching release asset naming (`fsfs-{version}-{triple}.{ext}`)
- **SHA256SUMS** -- update verification downloads the release-level checksum file instead of per-artifact sidecars
- **Windows target triple** -- correct detection for `x86_64-pc-windows-msvc` and `aarch64-pc-windows-msvc`

### CLI & UX

- **TUI visibility** -- running `fsfs` with no args now prints diagnostic messages when the TUI exits
- Tighten root probe limits, expand excluded directories, and improve first-run UX ([3cdae25](https://github.com/Dicklesworthstone/frankensearch/commit/3cdae25fa2e47360edab7bed9e9e6cd3fe07db1d))

### Performance & Correctness

- Use `HashSet` for O(1) duplicate `doc_id` detection in `TwoTierIndexBuilder` ([8cd272a](https://github.com/Dicklesworthstone/frankensearch/commit/8cd272a9b714dfdf7ce5ad294d1095c3e24d9ce4))
- Use `BTreeSet` for deterministic path ordering in `PendingEvents` ([378af78](https://github.com/Dicklesworthstone/frankensearch/commit/378af788b60319a1715cc56a9c3e069723a258e7))
- Populate actual index path in WAL `IndexCorrupted` error for better diagnostics ([b72bfd4](https://github.com/Dicklesworthstone/frankensearch/commit/b72bfd41e4b9088dedbc382f1ebb7252e797c54b))

---

## [v1.1.1](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.1.1) -- 2026-02-22

> [Full diff from v1.1.0](https://github.com/Dicklesworthstone/frankensearch/compare/v1.1.0...v1.1.1)
>
> Lightweight tag pointing to the same commit as v1.1.0 (`82c18ff`). Published as a separate GitHub Release with patched release notes describing fixes that were folded into the v1.1.0 binary.

**Note:** v1.1.0 and v1.1.1 share the same commit. v1.1.1 was cut as a quick-follow release to document first-run fixes that shipped in the v1.1.0 binary but warranted explicit callout.

### Bug Fixes (documented retroactively)

- **Fix first-run hang on macOS** -- reduced filesystem probe budget (depth 2, 10K entries max) and excluded macOS system directories (`Library`, `Pictures`, `Movies`, etc.)
- **Add progress indicator** -- prints "Scanning for indexable directories..." before filesystem probe begins
- **SHA256SUMS filename** -- checksum file now has the correct name

### Platforms

| Platform | Asset |
|----------|-------|
| Linux x86_64 | `fsfs-1.1.1-x86_64-unknown-linux-musl.tar.xz` |
| macOS arm64 (Apple Silicon) | `fsfs-1.1.1-aarch64-apple-darwin.tar.xz` |
| Windows x86_64 | `fsfs-1.1.1-x86_64-pc-windows-msvc.zip` |

---

## [v1.1.0](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.1.0) -- 2026-02-22

> **Crates.io publishing, resilient indexing, macOS Apple Silicon support** -- [Full diff from v1.0.0](https://github.com/Dicklesworthstone/frankensearch/compare/v1.0.0...v1.1.0)
>
> Lightweight tag. Published as a GitHub Release.

### Crates.io Publishing & Dependency Migration

- Switch all workspace dependencies from local path references to crates.io registry versions ([1fceef4](https://github.com/Dicklesworthstone/frankensearch/commit/1fceef4a5e62a33ce2b50ab5edede0516763a145))
- Bump core to v0.1.2 and index to v0.1.2 for crates.io republish ([c1676fb](https://github.com/Dicklesworthstone/frankensearch/commit/c1676fbe5b0d78f0ec0ac2735b02d2572fa8d25f))
- Add README.md files for all 12 crates for crates.io presentation ([790cbb0](https://github.com/Dicklesworthstone/frankensearch/commit/790cbb0d45a6b7cf241d44415e1e6b971f6d8700))
- Adapt to ftui-text 0.2.1 API changes ([be772f7](https://github.com/Dicklesworthstone/frankensearch/commit/be772f7e80286b6b66fc0cf6103f8d316451ae9a))

### Resilient Indexing Pipeline

- Checkpoint resume, embedding retries, degraded-mode completion, watcher auto-restart, and heap/normalization fixes ([4547323](https://github.com/Dicklesworthstone/frankensearch/commit/45473238fa4cdcf5b3408a5303b8efd026c48563))
- Prevent infinite loop in snapshot walker on symlink cycles ([af02a5d](https://github.com/Dicklesworthstone/frankensearch/commit/af02a5d73693b99ad59cb761c6519e3b096c319e))
- Recognize Johnson-Lindenstrauss embedders as hash embedders in storage ([1738a6b](https://github.com/Dicklesworthstone/frankensearch/commit/1738a6bd2f87da22f55eaf87acbbbd0a787911b0))

### Search Pipeline Hardening

- Fix nested markdown links, optimize diff, stabilize MMR, add bounds checks ([ee88129](https://github.com/Dicklesworthstone/frankensearch/commit/ee881297165bb75912a9a0df158d1b5d22474539))
- Improve identifier detection, WAL-first lookups, score normalization, and job queue dedup ([a74d3f2](https://github.com/Dicklesworthstone/frankensearch/commit/a74d3f2c0fe6e597cd5e7a73d191776c3bad042b))
- Model manifest expansion, index reconciliation, and storage pipeline hardening ([adc3f45](https://github.com/Dicklesworthstone/frankensearch/commit/adc3f454a189179eb95ab39bc568c761c5b9f3fd))
- Deduplicate WAL entries and fix ULID generation with telemetry prefix ([595aa48](https://github.com/Dicklesworthstone/frankensearch/commit/595aa48937d56fb48677daaa54f0f1f39dfc9d59))

### Platform Support

- **macOS arm64 (Apple Silicon)** added as a first-class release target with pre-built binary
- Tighten root probe limits and expand excluded directories for better first-run UX on macOS ([3cdae25](https://github.com/Dicklesworthstone/frankensearch/commit/3cdae25fa2e47360edab7bed9e9e6cd3fe07db1d))
- Fix `version` subcommand (use subcommand instead of `--version` flag) in install script self-test ([35dbcd5](https://github.com/Dicklesworthstone/frankensearch/commit/35dbcd5f3cf8a21ef86745bbe1444fda4727039c))

### Platforms

| Platform | Asset |
|----------|-------|
| Linux x86_64 | `fsfs-1.1.0-x86_64-unknown-linux-musl.tar.xz` |
| macOS arm64 (Apple Silicon) | `fsfs-1.1.0-aarch64-apple-darwin.tar.xz` |
| Windows x86_64 | `fsfs-1.1.0-x86_64-pc-windows-msvc.zip` |

---

## [v1.0.0](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.0.0) -- 2026-02-21

> **First stable release** -- [Full diff from v0.1.0](https://github.com/Dicklesworthstone/frankensearch/compare/v0.1.0...v1.0.0)
>
> Lightweight tag. Published as a GitHub Release.

First stable release of the `fsfs` CLI, marking the two-tier hybrid local search engine as production-ready.

### Canonicalization & Reranking Overhaul

- Rewrite the query canonicalization pipeline and add fastembed-based cross-encoder reranker ([1445c78](https://github.com/Dicklesworthstone/frankensearch/commit/1445c7805cad0b6768b724a5e93b74f19cf9de65))
- Add multi-model ONNX embedder support via `OnnxEmbedderConfig` ([41c69be](https://github.com/Dicklesworthstone/frankensearch/commit/41c69be0791bf5689fee97134983a7e69e9ebdfc))

### Stability & Correctness

- Work around VDBE `sqlite_master` parameterized query limitation ([567d816](https://github.com/Dicklesworthstone/frankensearch/commit/567d8160efe1d6ef74c7070667888b174fa6e65a))
- Fix daemon modules, exclude pattern suffix matching, and clippy warnings ([2ca9040](https://github.com/Dicklesworthstone/frankensearch/commit/2ca90406e8fcf829b9a8f3ac29063916cff9fa7d))
- Resolve workspace-level dead code errors and fix 2 failing fsfs tests ([d7f144f](https://github.com/Dicklesworthstone/frankensearch/commit/d7f144f71573e04f6ae6b4173f2b787422131223))
- Resolve compilation errors, lifetime annotations, and clippy warnings across workspace ([3a280c6](https://github.com/Dicklesworthstone/frankensearch/commit/3a280c663f59e3dcaf0cbee62ea0157195467ef4))
- Fix Windows build portability and installer checksum fallback ([7f8649e](https://github.com/Dicklesworthstone/frankensearch/commit/7f8649e04a0a1953a21d9a43970d12611f3a681c))
- Remove spurious lifetime annotations, update sibling dep refs, fix embed auto_detect ([0bdbac9](https://github.com/Dicklesworthstone/frankensearch/commit/0bdbac941b9e3deefeeb280a2debbf9a9f4105ae))

### Runtime & Configuration

- Refine FSFS runtime configuration and update ops dashboard screens ([139b70d](https://github.com/Dicklesworthstone/frankensearch/commit/139b70dfdd6d07f82df85b92cb6c83d7e43038e8))

### Platforms

| Platform | Asset |
|----------|-------|
| Linux x86_64 | `fsfs-1.0.0-x86_64-unknown-linux-musl.tar.xz` |
| macOS arm64 (Apple Silicon) | `fsfs-1.0.0-aarch64-apple-darwin.tar.xz` |
| Windows x86_64 | `fsfs-1.0.0-x86_64-pc-windows-msvc.zip` |

---

## [v0.1.0](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v0.1.0) -- 2026-02-19

> **Initial public release** -- [Full diff from initial commit](https://github.com/Dicklesworthstone/frankensearch/compare/42156796c494...v0.1.0)
>
> Annotated tag. Published as a GitHub Release.

The first public release of frankensearch -- a two-tier hybrid local search engine for Rust with the `fsfs` standalone CLI. This release includes the full workspace of 11 crates and approximately 240 commits over 6 days of intensive multi-agent development (2026-02-13 through 2026-02-19).

### Two-Tier Progressive Search Engine

- Fast initial results (<15 ms target) followed by quality-refined results (~150 ms target)
- Reciprocal Rank Fusion (RRF) combining lexical BM25 and semantic vector similarity
- Configurable score blending between fast and quality tiers (`quality_weight`)
- Graceful degradation -- `SearchPhase::RefinementFailed` preserves Phase 1 results when quality tier errors or times out
- Deterministic ordering for reproducible ranking via stable tie-break logic
- Query classification (`identifier`, `short keyword`, `natural language`) with adaptive budgets
- Exclusion queries (`-term`) for filtering unwanted results
- Negation-aware canonicalization pipeline

### Embedding System

- Hash embedder (zero model downloads) for dev/CI
- `model2vec` embedder (potion-multilingual-128M for fast tier)
- `fastembed` embedder (all-MiniLM-L6-v2 for quality tier)
- Automatic embedder stack detection and fallback ([54b11f6](https://github.com/Dicklesworthstone/frankensearch/commit/54b11f63f66b201d6df62731fee1c871c2c4ce04))
- Streaming model downloads with manifest validation and batch size clamping ([5a09ad9](https://github.com/Dicklesworthstone/frankensearch/commit/5a09ad9dfdb16ed159db0c376c61ef48f1fc8e9f))
- Filesystem-backed verification cache for model manifests ([0d1b78a](https://github.com/Dicklesworthstone/frankensearch/commit/0d1b78a1feb2023f566d7b18549afdb17ffdef9f))
- Dimension reduction support

### Vector Index (FSVI Format)

- Memory-mapped on-disk format with f16 quantization by default
- SIMD dot products with NaN-safe total ordering
- Zero-alloc byte-level SIMD dot products eliminating thread-local scratch buffers ([3ae2caa](https://github.com/Dicklesworthstone/frankensearch/commit/3ae2caa70d9078189170ad33b529c45120ea61b4))
- Heap-based top-k selection
- WAL journaling for crash recovery with corrupted trailer detection ([cb43051](https://github.com/Dicklesworthstone/frankensearch/commit/cb43051f3da60527360d03576b45831b635c6c98))
- HNSW approximate nearest-neighbor path with persistence and graph-ranking integration ([3409623](https://github.com/Dicklesworthstone/frankensearch/commit/3409623b66fecb3a1d7f7880e1e9ddb39ffa6a55), [3ad4326](https://github.com/Dicklesworthstone/frankensearch/commit/3ad4326fd63a3f53d0d90d48f0e5ea1c74c2a4da))
- Soft-delete with rollback support ([8dd09e6](https://github.com/Dicklesworthstone/frankensearch/commit/8dd09e6b8e2e7ce030271acb17053dd2509f1d7a))
- Mmap-backed VectorIndex ([a0a96fd](https://github.com/Dicklesworthstone/frankensearch/commit/a0a96fd4dee03ca5eb1d559bcae8ba6148109858))

### Fusion Pipeline

- Adaptive fusion with circuit breaker and phase gating ([e9551bb](https://github.com/Dicklesworthstone/frankensearch/commit/e9551bb60254826c451bc9435017b2be597e31ad))
- Pseudo-relevance feedback (PRF) and Maximal Marginal Relevance (MMR)
- Conformal calibration for score confidence
- Query-biased PageRank graph ranking and 3-input RRF ([da611e6](https://github.com/Dicklesworthstone/frankensearch/commit/da611e6596ef05e099bcb2697869300bc1f7bbfc))
- RRF explain support ([b1541ca](https://github.com/Dicklesworthstone/frankensearch/commit/b1541ca254c387f9dfd4ac348aad3906b0d27faf))
- NaN/Infinity fallback in RRF scoring ([5dba362](https://github.com/Dicklesworthstone/frankensearch/commit/5dba3628ae604a1f45634ac6d2b5901d3de03ff2))
- Federated search with interaction testing infrastructure ([de2bddc](https://github.com/Dicklesworthstone/frankensearch/commit/de2bddc80961a5d2c4f9e3d96577d0ab051a79c7))

### FSFS CLI Product

- `fsfs search` with progressive delivery (`--stream`, `--format jsonl/toon/csv/table/json`)
- `fsfs index` with filesystem watching and incremental updates
- `fsfs explain` for result explanation surfaces
- `fsfs doctor` for health checks
- `fsfs status` for runtime diagnostics
- Daemon transport with query caching, unbounded-recall tuning, and fallback when daemon unavailable ([e4870f8](https://github.com/Dicklesworthstone/frankensearch/commit/e4870f82809ea42969ddf05a513ee9a60ab35321), [068121d](https://github.com/Dicklesworthstone/frankensearch/commit/068121dca3382a618a8dd8ad770c9f2b7dfe8bb7))
- Auto-detect output format for non-TTY environments ([b7813295](https://github.com/Dicklesworthstone/frankensearch/commit/b7813295ca0f280197f958f0e5db11d28c6f22f8))
- Adaptive debounce engine for search execution ([85a4380](https://github.com/Dicklesworthstone/frankensearch/commit/85a4380838ef7fe81d1e430e1401480f2159f77a))
- Semantic VOI gate, cass-compatible Tantivy index, and TUI render/timing overhaul ([3e5bcfb](https://github.com/Dicklesworthstone/frankensearch/commit/3e5bcfba3b9ad093f5cbdac52d8fa9f9b9e5053e))
- Pressure-aware backoff, redaction hardening, and repro diagnostic expansion ([40839e1](https://github.com/Dicklesworthstone/frankensearch/commit/40839e1abfec2ab1197fde30737e9677f60cf4a2))
- Batch indexing pipeline, vector skip diagnostics, and WAL stale-detection fix ([5d40423](https://github.com/Dicklesworthstone/frankensearch/commit/5d40423d3f23f7935b359fafdbebb8b56b2bdf16))

### Storage & Durability

- FrankenSQLite-backed metadata persistence with content-hash dedup
- Immediate transactions, ingest pipeline, staleness detection ([fe4fca4](https://github.com/Dicklesworthstone/frankensearch/commit/fe4fca4d09f5e76328778fc6c27d539c7d8a8622))
- Concurrent schema bootstrap with race-safe upsert ([780b676](https://github.com/Dicklesworthstone/frankensearch/commit/780b676e78314750414c722963f5c7e8fa4a60cf))
- File protector with e2e corruption recovery and atomic operations ([edf2c4a](https://github.com/Dicklesworthstone/frankensearch/commit/edf2c4a82f65d78628d1b27eeb41eecc4e22ee00))
- fsync hardening for all production metadata writes, including sidecar and repair files ([477b713](https://github.com/Dicklesworthstone/frankensearch/commit/477b7137c4f7552f767701a04b012fb69941150a), [391f1fa](https://github.com/Dicklesworthstone/frankensearch/commit/391f1fa8e01766bec72b61295faf537374f1c832), [3a2e61b](https://github.com/Dicklesworthstone/frankensearch/commit/3a2e61b5484a8d761f9ba1da5f69ef433c99cea2))
- StorageDataSource with storage-backed ingest pipeline and IndexBuilder durability/lexical wiring ([bf2ed68](https://github.com/Dicklesworthstone/frankensearch/commit/bf2ed68ac78cf00887281f38aaa89f263b0603f6))

### Ops Dashboard & Telemetry

- Multi-screen TUI dashboard: alerts/SLO, fleet, resources, analytics, timeline ([fc155e3](https://github.com/Dicklesworthstone/frankensearch/commit/fc155e344bcfeda0d805219670b0bd6eb1eedde3), [590f3e0](https://github.com/Dicklesworthstone/frankensearch/commit/590f3e04c1a343a35e5a2116ff4e855338f84954))
- Telemetry ingest pipeline with backpressure, attribution, and lifecycle tracking ([dac02d3](https://github.com/Dicklesworthstone/frankensearch/commit/dac02d3556c9dba1931b48c5e767304ae7e0fbd6))
- Control-plane health alerting and self-monitoring ([814331a](https://github.com/Dicklesworthstone/frankensearch/commit/814331aff12b0752877e44ce08dff6c48825f663))
- Live resource telemetry collection and ops ingestion pipeline ([2f61155](https://github.com/Dicklesworthstone/frankensearch/commit/2f61155e644e95cafced01baf39b9a96a1fae3d5))
- FrankenSQLite-backed telemetry storage ([4e18736](https://github.com/Dicklesworthstone/frankensearch/commit/4e1873664b4640c50232c231a1ab8ae524c7779c))
- Migration to FrankenTUI backend ([85e44e0](https://github.com/Dicklesworthstone/frankensearch/commit/85e44e09589512e18d57a4a4dc942634312f7661))

### Resilience & Safety

- Graceful lock recovery (no panic-on-poison) across all concurrent components ([968a6b8](https://github.com/Dicklesworthstone/frankensearch/commit/968a6b8f5a1036861fc57edfcb8027566065f3a3))
- Fix 14 NaN-blindness and safety bugs across 8 files ([c0eea15](https://github.com/Dicklesworthstone/frankensearch/commit/c0eea158a4e9407fda63949a8bf6636b7da9b7b1))
- Eliminate TOCTOU race in sentinel and PID file acquisition ([28f4ab3](https://github.com/Dicklesworthstone/frankensearch/commit/28f4ab34ef3017f42e5514ff3aa556974aaee08f))
- Block path traversal in model manifests and harden searcher indexing ([b994a1c](https://github.com/Dicklesworthstone/frankensearch/commit/b994a1c6fac194708d04bfb4f2644d503514773b))
- Harden self-update pipeline, add ULID telemetry IDs, improve WAL atomicity ([08569f5](https://github.com/Dicklesworthstone/frankensearch/commit/08569f5d638b3f6b3894ae3793b0ff6dedabf81d))

### Cross-Platform & Packaging

- Linux x86_64 (`musl` static binary)
- Windows x86_64 (zip + standalone exe) with CI build target ([af41cdc](https://github.com/Dicklesworthstone/frankensearch/commit/af41cdc3089350b3006869679936dd0cbe823715))
- curl|bash installer with HTTP proxy forwarding and background daemon service installation (systemd / launchd / schtasks) ([83f3298](https://github.com/Dicklesworthstone/frankensearch/commit/83f32989373b16f3c950dd6220568cad7be4a185))
- MIT License with OpenAI/Anthropic Rider ([9f60b71](https://github.com/Dicklesworthstone/frankensearch/commit/9f60b71daefe58496503d78a0c5303de5f716c70))
- Published to crates.io (core crates v0.1.1) ([a8d64a0](https://github.com/Dicklesworthstone/frankensearch/commit/a8d64a0605b373ea5875b4fa95266c23a3529fec))

### Build System & Quality

- Feature-flag tier system: `default` (hash), `semantic`, `hybrid`, `persistent`, `durable`, `full`, `full-fts5`
- CMA-ES hyperparameter optimizer for fusion pipeline tuning ([979b09c](https://github.com/Dicklesworthstone/frankensearch/commit/979b09c0309c02180d9cad5cb770a1ec9119fc12))
- IR evaluation metrics: nDCG@K, MRR, Recall@K, MAP with bootstrap confidence intervals
- Benchmark baseline matrix and pressure simulation harness ([dff3387](https://github.com/Dicklesworthstone/frankensearch/commit/dff33875558285ee8cf6f06bf223b73fd6618d49))
- Dependency upgrades: tantivy 0.25, fastembed 5.8, ort rc10 ([df53b6e](https://github.com/Dicklesworthstone/frankensearch/commit/df53b6ee43b8d5d63f3a7c769c4ef0cd90f20a61))
- Pin asupersync to v0.2.0 ([3818edb](https://github.com/Dicklesworthstone/frankensearch/commit/3818edbc973d1bde7ffeee13043d2217c1df2b2f))

### Concurrency Model

- Built on `asupersync` and capability context (`Cx`), not Tokio
- Cancellation-aware search phases and timeouts
- Deterministic TUI replay and evidence ledger hooks ([251e618](https://github.com/Dicklesworthstone/frankensearch/commit/251e618c9a0443df866b11fca74ab37203a0a107))

### Architecture (11-Crate Workspace)

| Crate | Responsibility |
|-------|---------------|
| `frankensearch` | Facade crate with top-level public API and re-exports |
| `frankensearch-core` | Shared types, traits, errors, config, query canonicalization/classification, metrics/eval helpers |
| `frankensearch-embed` | Embedding backends and fallback stack (`hash`, `model2vec`, `fastembed`), streaming model downloads |
| `frankensearch-index` | FSVI vector storage, SIMD dot products, top-k search, WAL, optional ANN (HNSW) |
| `frankensearch-lexical` | Tantivy schema/index/search for BM25 lexical retrieval |
| `frankensearch-fusion` | RRF fusion, two-tier orchestration, blending, adaptive fusion, PRF, MMR, circuit-breaker, federated search |
| `frankensearch-rerank` | Cross-encoder reranking integration |
| `frankensearch-storage` | FrankenSQLite metadata persistence, dedup/content-hash tracking, embedding queue |
| `frankensearch-durability` | Repair/protection primitives for index artifacts and segment health |
| `frankensearch-tui` | Shared TUI shell, input, theme, replay framework |
| `frankensearch-ops` | Fleet observability/control-plane TUI and telemetry materialization |

---

## Pre-Release History

> 2026-02-13 -- Project scaffolding and planning phase before the first tagged release.

- Initial commit: project scaffolding, README, AGENTS.md, and beads task graph ([4215679](https://github.com/Dicklesworthstone/frankensearch/commit/42156796c494a74d4dfad83301d25bec04058c61))
- Document `asupersync` as mandatory async runtime, purge Tokio references ([3f29794](https://github.com/Dicklesworthstone/frankensearch/commit/3f2979450fb9c924cbcea07ecf76a8279b183f5a))
- Initialize Rust workspace with six-crate hybrid search architecture ([f39c793](https://github.com/Dicklesworthstone/frankensearch/commit/f39c793a94e1e687cae0db104c6f1c5f6df02f89))
- Implement full search pipeline across embed, index, lexical, fusion, and rerank crates ([3965991](https://github.com/Dicklesworthstone/frankensearch/commit/39659917d5a3a55be73a68cbc39aa30eba3b39c1))
- Add storage, durability, fsfs, tui, and ops crates to workspace ([484f9cc](https://github.com/Dicklesworthstone/frankensearch/commit/484f9cc9e266f02348b44ff3c09557838eb06be0))
- Expand workspace config, facade crate, and README with feature-flag tier system ([efe8ca6](https://github.com/Dicklesworthstone/frankensearch/commit/efe8ca6d4f92d6da2346e9a58a479ea1af5ca22f))
