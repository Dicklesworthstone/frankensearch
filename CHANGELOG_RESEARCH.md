# Changelog research: fsfs 1.10.0 / FrankenSearch 0.5.0

Requested 2026-09-08 using `changelog-md-workmanship`, small-update mode.
Scope: every landed commit after the 1.9.1 release source
`9d132a0315e12da0442aa7a943852091fe043037` through the 1.10.0 source
`9c5d8867cbbc7edf696a24410a86af08402fc468`, plus publication records for these
binary/library versions. Earlier changelog history is preserved, not reconstructed
by this update. Governing repository/suite docs were read during release work.

Source priority: git diffs and history; live tags/releases; checked-in Beads;
existing changelog/release notes; README. Durable release evidence lives at
`/data/release-work/frankensearch-release-1.10.0-recovered-20260908`.

| Chunk | State | Coverage / open questions |
|---|---|---|
| Version spine and publication | Validated | v1.10.0 and crates-v0.5.0 are both public GitHub Releases; 13 crates.io versions published; canonical timeline updated after bundle publication |
| Complete landed commit window | Validated | All 21 commits classified below; independent audit of all eight changed fsfs/facade paths completed, corrections incorporated |
| Tracker workstreams | Validated | Durability closed; bd-2ba5 native migration remains open. Release bd-apqfa closed after public bundle verification; immutable links retain the pre-publication source records |
| Synthesis and link validation | Validated | Independent audit corrections incorporated; final 19-link scoped validator passes without warnings; canonical structural check passes with only the preserved-history warning |

Initial live check: GitHub v1.10.0 Release ID385005152 published
2026-09-08T19:36:49Z, latest=true. Previous v1.9.1 published
2026-09-07T05:34:20Z; crates-v0.4.3 published2026-09-07T05:32:33Z.
The existing changelog's timeline/scope note still stopped at1.9.1 even though
its new1.10.0 entry existed. Do not describe the pending crate-bundle tag as an
already-published GitHub Release.

## Complete commit coverage

The release window contains 21 commits (including one merge), touching 36 files.
Every commit is accounted for here; the canonical changelog selects the commits
that best explain landed behavior instead of repeating this inventory.

| Commit | Classification and distilled result |
|---|---|
| `71551e93` | Previous-release README version bookkeeping; no new 1.10 capability |
| `0bc24c38` | Previous-release qualification documentation; no new 1.10 capability |
| `798a9337` | Merge of concurrent README publication update; no production change |
| `46ce8a6a` | All 20 FrankenSQLite members and four consumers advance to 0.3.18; 396 FTS5/storage tests and reopen evidence in upgrade log |
| `4e789c2d` | Checked aligned little-endian weight copy, scalar fallback, exceptional-bit/alignment/trailing-byte tests; no certified speedup |
| `00f00cf9` | Daemon retains initialized quality model after timed-out waiter; producer/generation checks and pool replacement reset |
| `e21bfbb8` | Explicit native multilingual selection reaches index/search/doctor/append and daemon producer-bound cache; conflict refusal and real cross-language tests |
| `b31e792d` | Shared semantic-support feature and ONNX-free semantic-native source profile; English native quality default, configured-pair downloads, refusal of binary self-update/unclassified rollback replacement |
| `c002fd85` | Facade native/rerank API parity; hash-dependent examples/tests declare required features |
| `19bf2195` | NativeReranker async trait and caller-owned pool; worker retains model admission, cancellation checkpoints, no inline fallback |
| `e1f935e7` | NativeEmbedder stopped-pool/shutdown-race refusal; request and child cancellation before output; exact producer-bit tests |
| `12339664` | Cold reranker load on caller pool; selected-model cache identity frozen, completed initialization retained, worker owns admission after waiter drop |
| `e0b7ca7b` | Rerank stage deadline spans file reads/capacity/inference; cold model selection/loading precedes that stage and is outside its timer; preserves fused results, propagates cancellation, prevents timeout response caching |
| `96a66f39` | Actual runtime/model regressions for stopped pool, shutdown after submission and cancellation winning over a load error |
| `cd4c73a9` | Detection preflight, rejected-submission fallback refusal and post-join request cancellation implement those lifecycle cases |
| `bc5e86d1` | Creator-owned operation guards release durability locks despite retained descriptors; read guard covers CRC/sidecars/mapping, repair relocks published inode |
| `87dbb5f4` | Release bead starts; no product behavior |
| `1b67c0cb` | All 13 forward version bumps and historical adapter fingerprint reconstruction; candidate not independently the final release source |
| `194c04c0` | Release freeze tracker update; no product behavior |
| `e82dee23` | Quill engine 0.2.4/CRC with exact historical wire round trips, oracle v7/profile v6, current semantic packaging composition, cold-load limits |
| `9c5d8867` | Only ChaCha 0.10.1→0.10.2 lock record changes; final qualified and published source |

Research used full commit messages, scoped production/test diffs, the existing
upgrade log and feature manifests. Runtime/facade changes receive an independent
read-only draft audit while the root checks dependencies, durability and version
contracts. Tests mentioned as historical qualification are evidence from their
landed records; no claim that this documentation pass reran those tests.

## Tracker and release checks

- At frozen release source, `.beads/issues.jsonl` line 102 is bd-2ba5 (open
  native migration), line 711 is bd-apqfa (release in progress), and line 778 is
  bd-durability-lock-release-v5lj4 (closed at 2026-09-08T00:11:06Z). HEAD at the
  initial research check agreed. Canonical links pin that immutable file and
  exact line; bd-apqfa subsequently closed after final publication verification.
- The durability closure cites baseline retained-descriptor failures, candidate
  regressions, 160 durability tests and an unchanged 10-stage gate. The original
  intermittent repair failure's lock holder was never observed; no fork-holder
  attribution or performance claim is inferred.
- Live GitHub and remote refs rechecked during this pass: v1.10.0 is public
  Release 385005152 with 25 assets, both release tags peel to
  `9c5d8867cbbc7edf696a24410a86af08402fc468`, and Actions are disabled.
- Upstream FrankenSQLite 0.3.18 release and RustCrypto stream-ciphers PR583 were
  read live to confirm dependency-fix descriptions. The local ChaCha upgrade is
  an upstream defect correction, not a claim of locally reproducing UB.
- Publication qualification and evidence recovery are documented separately in
  the release bundle README. Original root temporary evidence disappeared at
  19:48Z; the recovery record distinguishes exact surviving originals,
  reconstructed summaries and repeated public delivery probes.

## Validation

The independent runtime review caught an overstatement in the draft: cold
reranker initialization uses the caller's pool but occurs before the rerank
stage timer. The changelog now states that boundary explicitly. This is a
documentation correction against final source, not a runtime behavior change.
The review also identified native-only delivery boundaries worth making explicit:
English F32 defaults, configured-pair downloads and refusal of self-update or
unclassified rollback replacement. The canonical profile bullet now includes
those behaviors, with the native-profile commit as its representative link.

- [x] Independent runtime/facade coverage audit incorporated: corrected timer
  boundary, source-profile update behavior and retry after model provisioning.
  The reviewer found no other missing capability waves or open questions.
- [x] Skill structural validator on canonical CHANGELOG.md: no errors, one
  warning about possible bare hashes entirely in preserved older content.
  The v1.9.1 and older entries are byte-identical to the pre-skill version
  (SHA256 `f9c4ca2d7ee641875c7f695b777ec56f450ab9e50b38f1ab953b875c803e48c1`).
- [x] Initial 18-link scoped audit passed with no warnings; the added retained
  initialization link passed a live GitHub commit lookup. The final bundle URL
  receives the final publication check. Older
  historical links are outside this reconstruction window.
- [x] Scope, dates, actual Release-versus-tag states, versions, tracker states
  and upgrade/platform limitations cross-checked against final publication.
- [x] Canonical changelog and this research memo included in the final evidence
  bundle as post-release documentation, distinct from the frozen product source.

Public-registry consumers completed at 2026-09-08T21:05:28Z: all nine passed;
the combined-feature run executed both Quill and Tantivy queries. The bundle
snapshot includes that terminal qualification. Its own public download check
and final repository closure occur after assembly and are recorded in the
canonical files, rather than retroactively modifying the bundled snapshots.

## Final publication and validation

- Binary Release 385005152 remains latest with its original 25 assets. Crate
  bundle Release 385065727 was published at 2026-09-08T21:13:03Z with exactly
  `frankensearch-crates-0.5.0.tar.xz` and its checksum sidecar. Both release tags
  still peel to the frozen product source. GitHub Actions remained disabled.
- Both bundle assets were freshly downloaded and matched their staged bytes and
  GitHub digests; every one of the 779 archive entries was checked. The 13 public
  crate archives are included. Bundle SHA256 is
  `c85ed203c21c3f5e92d54f40d61e82ebfeff497a7c1c2cc84da6cace1ae18bc9`.
- Final `validate-changelog-md.py --verify-links --max-links 100` on the new
  release section and timeline rows passed with all 19 HTTP links checked and
  no warnings. The canonical structural pass has no errors; its one warning
  concerns unchanged earlier history. `git diff --check` passes.
- Evidence: `changelog-final-validation-command.json`,
  `changelog-final-validation.log`, `final-release-verification3.json`, and
  `public-bundle-final-verification.json` in the durable release work directory.
  The latter records actual public downloads after bundle publication.
- A new aggregate reader initially expected `case_count` in the Mac summary;
  the original schema uses `cases_passed`. The corrected reader checks six
  successful cases, zero failures, all original receipt/raw-stream hashes,
  three installers, three updaters and 21 direct commands. No original receipt,
  test result or acceptance condition was changed.

This update completes the requested changelog skill pass for the stated window.
It does not reconstruct or revalidate every older changelog entry, certify a
performance win, or close the broader native migration workstream.

## Narrow unreleased update, 2026-09-09

Scope: post-publication history from frozen product source
`9c5d8867cbbc7edf696a24410a86af08402fc468` through merge `9f965533`.
The non-merge commit inventory contains three product changes: progressive
daemon streaming (`35e8c231`), Quill replacement-ingest/live-document protection
(`e0d24c3e`), and the asupersync requirement correction (`a6d664cf`). The other
commits record release evidence, changelog research, assessment or tracker
updates. The two Quill entries arrived with their implementation and are
preserved; this update supplies the missing daemon capability and behavior
change, and marks all three as unreleased.

Evidence checked: complete implementation diff and affected runtime/CLI tests,
the closed `bd-fsfs-progressive-daemon-dq48i` record at `462035f3`, current README,
and the live GitHub commit and release metadata. GitHub still reports v1.10.0
published at 2026-09-08T19:36:49Z and crates-v0.5.0 at 21:13:03Z; the new work is
not a release. The initial local tag-range lookup found no local v1.10.0 ref,
so the inventory uses the verified frozen source SHA and the remote tag API.

Daemon acceptance is explicitly source-bound to `35e8c231`: clean DSR 8/8 checks,
all ten stock stages, seven real-model tests, and a subsequent exact release
replay passed. The replay served two distinct uncached queries on PID3311628
with one quality-model initialization, ordered frames, direct-result parity,
JSONL/TOON replay and clean disconnect/signal handling. Release executable
SHA256: `a62398f078cd82fff36e2aceba85ce2ddc098368a44e00f1d7c62016ab24fcb6`.
Raw commands and logs are retained under
`/data/tmp/frankensearch-progressive-daemon-20260908.flI1Y2`, particularly
`dsr-qualified-summary.json` and `progressive-release-receipt.json`.
These functional checks do not certify performance, cross-platform publication
or the subsequently merged Quill implementation.

Coverage status: validated. The skill's structural validator passes with its
existing older-history bare-hash warning. All four unique links in the
unreleased section return HTTP 200, and the immutable tracker permalink points
to the exact completed bead on line 838. Every entry from v1.10.0 backward is
byte-identical to the pre-update file (SHA256
`e52cebfc57458900e8786b6329f1df0aeb11107e371c3872ff3e9d35bd6e5b67`).
The remote annotated v1.10.0 tag was peeled to the frozen source above.
Validation receipt: `changelog-unreleased-validation.json` in the same artifact
directory. No release date, published version or old capability entry changed.

## Quill gate and replay follow-through, 2026-09-09

Bounded inventory: `29b8a71e..820e83e1`, five non-merge commits. The two
implementation fixes are the atomic-publication identity rebind (`802784a1`)
and the schema-pinned seed multiplier (`acee3572`). `0147e799` adds the native
witness gate; `820e83e1` adds replay regressions and complete default-feature
inventory to its probes. `b384ccca` is tracker-only. No public package version
or release tag changed.

Read the complete Rust, driver, test-selection and contract diffs. The existing
seed and hostile replay assertions are unchanged; the additional mutation
rewrites a staged inode and restores mtime, then requires digest rejection.
RCH ran the original fuzz-harness selection: 11 passed, zero failed/ignored,
943 filtered, 0.77s, vmi1227854. The revised real-host driver ran 59 tests
(39 witness validators, 12 typed-query regressions, eight real-engine cases),
all passing with zero ignored. Actual zero-selection, missing-oracle build,
timeout and missing-terminal probes each produced their required refusal.
Default inventory: 906 library tests, five existing ignored; all-feature
inventory: 1012 library tests, nine existing ignored. Integration/bin targets
are inventoried separately, including an empty target never counted as a pass.

Evidence: `quill-typed-query-repair.log`, `dsr-quill-development.log`, and
`dsr-quill-negative-probes.log` under the same retained artifact directory as
the daemon acceptance. The first expanded DSR development pass completed all
eight checks, including all eleven stock-plus-Quill stages and seven real-model
E2Es. This development snapshot was not clean. The final clean-source run at
`820e83e1` then passed all eight DSR checks, all eleven stages, 59 Quill tests
and seven real-model E2Es, with identical source/lock identities before and
after. It is recorded in `dsr-quill-qualified.log` and
`quill-final-validation.json`; DSR receipt SHA256 is
`a1ec0b5ba617fe33b46ce828f6c8a141dd418871115493ae76adae7d12696a35`.
One subsequent change adds a ten-second timeout only to Git metadata lookup;
the exact changed call and Python syntax were verified separately. No Rust or
test-selection change follows the qualified revision.

The unchanged full baseline was stopped after 2492.389 seconds of library
execution, with its unchanged ELF hash and exit -15 retained. It exposed
12 failures; replay/seed failures were then repaired, while the QG2 golden
digest and E6 expected-divergence assertions remain unchanged and unresolved.
Cargo-configuration fixtures also encountered an in-repository RCH temporary
directory; the driver now uses a private canonical system-temp directory.
This is incomplete full-suite coverage, not a full-conformance, performance or
release claim. Earlier release entries are preserved byte for byte.
