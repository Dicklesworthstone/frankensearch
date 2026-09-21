# Changelog research: fsfs 1.10.0 / FrankenSearch 0.5.0

## September 17 supplement: lexical retry and pager qualification limits

Follow-through also covers the resumable-ordering implementation
`9135a6288be4bc340d4717618bede9d0b9031653` and integration tests in
`1bf70fd4c267dcafa789903318ffcdd65e0cc9a0`. The author had no Rust toolchain.
RCH on `vmi1152480` first executed four passing tests and one failed fixture
precondition: Quill publishes replacements of existing rows immediately, so
that fixture could not establish the pending-write state it claimed. The
corrected test verifies published replacement then restoration through a fresh
adapter. A sixth test creates a real pending insertion and verifies that both
documents survive resuming through another adapter. All six integration tests
and all 36 lexical unit tests pass, with one existing unit-test ignore.
The first failure remains in `lexical-ordering-3c9785c0-failure-receipt.json`;
the final run uses `lexical-ordering-3c9785c0-r3-spec.json`, frozen merge
`3c9785c0`, and explicit formatting/comment/test overlays. All-target fsfs
Clippy passes with warnings denied. The intermediate run's Clippy failure
exceeded the compiler's future-type recursion limit in the seed helper;
using the existing boxed `LexicalWrite::commit` interface fixes it without
suppressing a lint or changing the underlying commit operation. Both failed
attempts are retained. The final receipt is
`lexical-ordering-3c9785c0-r3-receipt.json` under the release work directory.
These are focused checks, not full release qualification.

This narrow supplement covers `db13bc1c4391e62d5a81c35a2c2f3735c3c2c19a`
(lexical queue ownership) and `05aca4de38f7560de94dc41b3bd8d28eff4bfb6d`
(pager-only lock update), with the corrected blocker record in `76a9e327`.
Reviewed the actual diffs and Quill write/delete publication boundaries.
The lexical commit's author explicitly reported no Rust execution. Subsequent
RCH validation on frozen `5aaa0352` passes all 2,110 fsfs library tests, with
14 existing ignores, and all-target Clippy with warnings denied. The initial
run had one permission-fixture failure because root could still read a mode-000
directory; the unchanged executable passes after dropping DAC override
capabilities. Source and sibling hashes match before and after execution.
The retained receipt is `lexical-base-dac-receipt.json` under the September 12
release work directory; no full release clearance is inferred from this check.

The follow-through regression drives the public Quill backend with a malformed
empty-ID upsert after a completed upsert/delete barrier and before another valid
action. All 36 lexical tests pass with one existing ignore, including retention
across repeated ordinary and resumable flush failures. A negative control with
the byte-identical pre-fix `flush_inner` body fails the new test because the
pending queue is empty instead of retaining the two-action suffix. Both runs
use RCH, with unchanged dependencies and source/ELF receipts; the expected
negative-control failure is not counted as a passing product test. All-target
fsfs Clippy also passes with warnings denied after adding the regression.

The retained pager storage receipt reports 351 passed, zero failed/ignored.
Upstream Mail 42321 and its raw savepoint/churn logs independently report
zero passed and one failed for each unchanged orphan-page keeper on published
facade/core 0.4.0 plus pager 0.4.3, with before/after production-source and
executable receipts. The changelog records both outcomes and keeps publication
pending. Upstream repair `e21d008b40476d13ce49aedeb1a51fb2bef0850c` now keeps
native-reader abandoned pages in the durable reclamation path. Its live gate
on `62d802014` plus that patch reports 1,004 pager tests and nine savepoint
tests passing, but remaining corruption arms are nonterminal and the repair
is not yet published. These upstream direct-Cargo results do not substitute
for the required RCH checks of exact and fresh public dependency graphs.

Live GitHub release metadata on September 17 still lists `v1.10.0` and
`crates-v0.5.0` from September 8 as the newest binary and bundle releases.
Live registry metadata still lists pager 0.4.3 and facade/core 0.4.2.
This supplement does not claim complete September 17 commit coverage.

## September 16 supplement: actual ARM native qualification

The native F32 limitation recorded on September 15 is superseded by actual
ARM64 macOS execution of source `31acc37c68cfa3af987ebf64a257c3bd9b576e97`.
The existing F32 certificate/batching/repeatability test and Int8 certificate
test each pass once, with no failures or ignores. All 116 F32 checkpoints match
the retained Linux x86 candidate. The changelog moves this result into Fixed,
retains populated CASS acceptance as unresolved, and makes no release or
performance claim. Commands, source and executable hashes, model barriers, and
the original failed linker attempt are recorded in the September 16 entry of
`docs/planning/UPGRADE_LOG.md` and its referenced release-work receipts.

## September 12 producer correction after registry publication

`b62074148d7fd38029818e638aa31f7699dc93fd` corrects Potion's execution
protocol from SafeTensors 0.7.0 to the actual 0.8.0 selected by `347be73e`.
The archive publication source `dd093fb2` predates this correction, so the
changelog labels it unreleased and includes published adapter 0.3.0 among the
producer identities that require rebuilding. No vector certificate or
historical manifest fingerprint was replaced.

RCH validation: the unchanged historical fixture fails on the bumped adapter;
the repaired candidate passes 404 embedder unit tests, eight fresh-process
logging tests, three real-Potion checks, formatting and focused Clippy. The
fresh-process preflight compares the protocol against the actual locked
Tokenizers and SafeTensors versions. Source matches all 1,777 tracked inputs;
both test-binary hashes and the terminal log are retained under
`/data/release-work/frankensearch-release-20260912/dependencies/` in
`producer-tests-receipt.json` and `producer-independent-embed-vmi.log`.
The full core suite separately exposed an existing anti-rollback publication
race. Commit `a4e86336761d982783ab0f4fec002cbbcee0e6bf` fixes it with a
per-root kernel lock across publication and durability. Independent-process
reader/competitor and interrupted-writer regressions preserve torn-head refusal.
The subsequent RCH batch passed all 1,131 core tests, formatting, focused Clippy,
and the Windows index compile guard; the single ignored core entry is a helper
executed by the subprocess tests. `floor-publication-receipt.json` retains the
source and executable identities. The broader floor bead remains open for
consumer profile wiring. These focused results do not qualify the complete release.

The new repair links name exact local commits; public link availability
must be checked after the qualified source is pushed.

## September 12 registry-publication reconciliation

Small-update scope: correct the publication status of the existing post-1.10.0
inventory and document the new runtime boundary. This is not a reconstruction
of earlier history or a claim that the next binary release is qualified.

- Live crates.io metadata plus downloads verified ten new library archives:
  facade 0.6.0, rerank 0.4.0, core/embed/index/lexical/fusion/Quill/storage/
  durability 0.3.0. Every SHA-256 matches the registry and every
  `.cargo_vcs_info.json` identifies `dd093fb230404ab08be2ed6f27776ed6c4796485`.
  Publication times span 2026-09-12 14:28:46–14:33:59 UTC; all are unyanked.
  Receipt: `/data/release-work/frankensearch-release-20260912/registry-20260912-family.json`.
- GitHub API confirms annotated tag `frankensearch-v0.6.0`, object
  `8c9ed3c53cede7d6489bdc97f6e87d61afc1215f`, tagged 14:28:41 UTC, targets that
  same commit. Live Releases list still ends at September 8. Therefore the
  new timeline entry is crates.io publication plus a plain tag, not a GitHub
  Release. fsfs remains 1.10.0; TUI/ops remain 0.2.1.
- Inspected the final ten nonmerge publication commits and their relevant
  diffs: `347be73e` runtime/component boundary, `f20e9446` durability type
  alignment, `4fcbdb55` FTS5 Send proof, `06e053e1`/`309ae25a` caller-owned
  shadow polling and admission, `ad0a6083`/`ff29cf76` persisted pooled
  comparison regressions, `b9e0c109` lock closure, `dd093fb2` assertion
  formatting, plus `8b588964` descriptor-bound mapping inspection. Existing
  research below covers the earlier loader/HNSW/Quill changes in the shared
  post-1.10.0 inventory. The full source window contains 82 nonmerge commits.
- Corrected the obsolete claims that all post-September-8 library changes were
  unreleased and that current consumers still resolve Asupersync 0.4.10. The
  earlier 0.4.11 latency failure remains in the upgrade log; historical
  0.4.10 receipts are not current 0.5.0 qualification.
- Bead `bd-dsbym` remains in progress. Cross-platform binary publication,
  current full gates and fresh registry-consumer execution are not established
  by the archive checks. No new release/tag/publication was performed by this
  reconciliation.
- Structural changelog validation passes with the existing historical bare-hash
  warning. All newly added GitHub commit/tag/issue URLs return HTTP 200. The
  crates.io HTML routes return 404 in this environment despite available API
  metadata and archives; the new version entry therefore links to its verified
  HTTP-200 official version API record.
- Independent draft audit corrected a mistaken adapter/package distinction:
  `model_manifest.rs` derives implementation revision from `CARGO_PKG_VERSION`,
  so published package 0.3.0 also identifies producer 0.3.0. The preparation
  version 0.2.7 is historical, not the current producer. Changelog migration
  advice now requires rebuilding both older producer generations. The audit
  also found a stale 0.2.7 prefix in historical fixture reconstruction; repair
  and validation are tracked separately under the ongoing dependency bead.

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

## Executable quickstart and FastEmbed receipt follow-through, 2026-09-09

Narrow inventory: `fc893e6e..8789e044`. Read all three commits: `f5eb50e2`
records the A5 claim, `3d1c2fdd` implements receipt reuse and the strengthened
quickstart, and `8789e044` repairs Cargo discovery under the isolated HOME.
The actual `bd-fsfs-executable-quickstart-ci-ve3ul` acceptance and its original
history were read. Historical records establish two executed RED cases; the
later refinement requires twelve named classes. This update preserves all
twelve and adds six explicitly labeled controls, without inventing a historical
twelve-case pass. GitHub metadata still reports v1.10.0 as latest, published
2026-09-08T19:36:49Z, and crates-v0.5.0 at 21:13:03Z. Actions remain disabled.

The stronger loader assertion exposed FastEmbed's direct full-hash call. Its
three registered model mappings now call the existing cached verifier. Exact
artifact matching, stale-file invalidation and execution certificate validation
remain intact. The Rust real-model test adds a direct FastEmbed stale-receipt
control and requires actual result records in both progressive phases.

Retained evidence is under
`/data/tmp/frankensearch-progressive-daemon-20260908.flI1Y2`.
RCH build 30012625538515439 ran 122 model-manifest tests successfully, with one
existing ignored case; that run preceded the FastEmbed caller change. The
rebuilt executable's positive and all eighteen controls passed in
`/data/tmp/fsfs-quickstart.IrpzmjHA`. This supplied-binary smoke correctly records
unknown source. `--binary --require-source` independently refuses, and unknown
options exit 2. The first clean installation run failed before compilation
because the host Cargo launcher resolves below HOME; its failure receipt is
retained at `/data/tmp/fsfs-quickstart.ns65acIV`. The correction preserves the
explicit toolchain cache locations and fresh build/runtime HOME/XDG.

Current coverage status: implementation landed; final clean-source installation,
the new direct real-ONNX regression, and complete DSR qualification are still
pending. The changelog describes code behavior without claiming publication,
complete Quill conformance or measured performance. Final execution results and
live-link verification must be recorded below before this workstream closes.

Final qualification: clean `719fdd0e425f25f44a871170e1e024204fc11e7f`
passed all eight DSR checks and all eleven stock quality stages, including
59 bounded Quill tests and seven real-model tests. The new direct FastEmbed
test initially expected `HashMismatch`; the existing frozen verifier instead
aggregates checksums into `ModelLoadFailed`. The first full run retained that
single failure (six other real-model tests passed). The corrected test requires
the exact `tokenizer.json:sha256-or-size-mismatch` diagnostic and an unchanged
receipt; its complete real-model run then passed. No production rejection,
native deadline, workload, or ignore was relaxed.

The final installed quickstart passed all eighteen controls, retained 1,136
runtime log lines / 318,235 bytes, and left no owned processes. Installed ELF
SHA256 is `343b12a660e2558eea2779bc94c32671695677671300e35dc585e5ae7865c565`,
identical to the invocation-built executable. Optimized real-model ELF SHA256
is `197a3bcd1f9a1cccd96ed2226bb9f0e8a23cf040c58e3cbeea22ab0c794976e5`,
retrieved from RCH build 30012625538515447 on vmi1227854; local and remote
hashes match, and all 297 production/build inputs match the clean qualification
source through the retained source-content receipt. This is not an assertion
that the older build's documentation/test files were identical.

`a5-final-validation.json` retains the complete stage list and artifact paths.
Final DSR receipt SHA256:
`ff10629a6b5e56ddaccb1b61934459f299db4d53560b611ac817a6f2101b859c`.
The clean source and lock identity were rechecked after execution. Both new
representative commit links resolve live, the changelog structural validator
passes with its existing older-history warning, and published entries from
v1.10.0 backward are byte-identical. DSR's inconsistent duration_ms fields are
excluded from timing claims. This completes Linux A5 acceptance, not a new
release, complete Quill conformance, all-platform installation or performance
certification.

## GH #43/#46 follow-through — 2026-09-11 UTC

This narrow update covers one-shot cancellation/pressure in `90e8fb14` and the
producer-revision consequences of the loader work in `c92d193c` / `6ec30ef1`.
The GitHub API still reports v1.10.0 as the latest published release, published
2026-09-08T19:36:49Z. Its annotated tag `74e2a47a` resolves to `9c5d8867`.
The local checkout lacks that tag, so release identity was checked through the
live tag object rather than inferred from local refs. The new implementation
commit resolves through the live GitHub API; no version or release was created.

Evidence reviewed: exact diffs of those three commits, GH #43/#46 discussion,
Beads `bd-pz8va` and `bd-u8cof`, current fsfs producer-revision admission, and the
strict library activation test that refuses a conformance-compatible foreign
producer. Stable space and golden outputs do not imply admission: the 0.2.7
implementation revision changes the producer/bundle fingerprint. Corrected the
overbroad no-rebuild comment and documented the explicit fsfs rebuild step.

GH #43 has five passing focused RCH tests plus six real-Potion CLI controls
(signal, same-index retry, double interrupt, two low-threshold profiles, and a
successful high-threshold run), and a separate real discovery-phase signal
probe. The process probes used ELF SHA256
`2a0acfcf2ef358506483c076a6d9d5aaf596cad753fd8d47a63582a676ffbbb9`.
The 128 MiB threshold runs reached about 1.07 GiB RSS during synchronous model
loading before rejection: this is a soft pressure threshold, not a memory cap.
The first signal is cooperative; the second interrupt terminated promptly.
Same-index retry succeeded, but that does not establish retention of a previous
committed generation. These are bounded development diagnostics, not a
performance comparison.

The required full gate ended with three failures, retained in
`/data/tmp/fsfs-gh43-quality-gate-final-20260911.log`: the workspace library
stage exposed an existing 120 ms watch-test timing assumption; two of seven
real-model E2Es lacked registered native/multilingual fixture files; and the
quickstart refused dirty-source provenance. Formatting, checks, clippy,
cross-target, bounded Quill correctness, all 2,400 selected fsfs tests and
the facade stage passed in that invocation. That snapshot preceded the watch
test repair, which now waits for actual watcher startup and checks finalization
once. A later RCH invocation passed final-source formatting, checks, clippy
(including hybrid) and cross-target compilation; its log is
`/data/tmp/fsfs-final-compiler-gates-20260911.log`.

GH #46's three fixture-backed loader tests passed: frozen conformance,
same-process shared construction, and bit-identical streamed matrix decoding.
The new production loader integration test also passed: modifying either real
artifact is rejected before a resident-cache hit, preserves the verification
receipt, and allows reuse after restoring the original bytes. Logs are
`/data/tmp/fsfs-gh46-real-model-tests-vmi1227854-20260911.log` and
`/data/tmp/fsfs-gh46-production-cache-recovery-20260911.log`. The test executable
SHA256 values are respectively
`bd8010c828e458bd596521d3c97ce4924ff5e743a269fa654ed65d6e62c85abf`
and `6b7c0919b7a538d27dd2f6d5c8d11207a1de16844ec39c93719eabdf76d59d58`.
The draft expected a raw hash error; before execution it was corrected to require
the production verifier's exact aggregated `ModelLoadFailed` diagnostic.
No admission rule or golden fingerprint changed. The changelog also avoids
calling shared loads free: verification still runs and reuse requires a live
caller retaining the model.

A final watch-only RCH attempt lost its client without terminal test evidence
and remains `NO_VERDICT`. After confirming its compilers had exited, the three
failed stages were scheduled against clean `4ea318c0`, with all required model
artifacts hash-verified. One initial follow-up was cancelled before tests to
correct its Cargo cache routing; both attempts' logs remain available. The
replacement run records a remote log and terminal exit receipt independently
of client output. Its final receipt reports exit 1 at 2026-09-11T06:18:05Z on
vmi1149989. All three selected stages failed; this is not a full-gate pass.

The repaired watch-start test and all five GH #43 tests passed in that clean
run. The fsfs library aggregate was 2,102 passed, one failed and 14 ignored.
The unchanged unreadable-subtree test correctly rejected its fixture
precondition: root could still read the directory after chmod 000. A separate
workspace-stage rerun removed `CAP_DAC_OVERRIDE` and `CAP_DAC_READ_SEARCH`,
preserving the test and its assertions. Its client exited 143 during compilation,
with no terminal test output or exit receipt and no remaining owned compiler
observed afterward. That attempt remains `NO_VERDICT`.

All seven real-model E2Es were executed: six passed and one failed in 164.28 s.
The multilingual CLI case passed. Native English F32 loading, producer admission,
indexing, status and doctor succeeded, but search exceeded the unchanged 500 ms
quality budget and returned `refinement_failed`. This was not a certificate
mismatch. The executed fsfs ELF SHA256 is
`88e88901c1f7476cda88debc57e000e0dbc65f0f98e19a9981a0c7919d7a976f`;
the test driver is
`d2ca61f58620c84bb1821b588dd8c205b3a82544911f408d34dd8654ffd60f08`.

The clean quickstart built and installed identical executable bytes, SHA256
`458465d861d6cdd4a288e7e5e9c10348a69763ffeda23dddb9c10a25934f1181`,
with source identity `4ea318c0`. Both index passes succeeded. The subsequent
ONNX hybrid query also exceeded its 500 ms quality budget, so the gate failed
with `MISSING-PHASE`; its negative controls were not reached. The receipt reports
no remaining owned processes. This resolves the earlier provenance refusal,
but does not qualify the executable quickstart.

The enclosing RCH connection later timed out at 06:25:00Z and reported its remote
process group stopped. The retained gate exit receipt and quickstart failure
receipt establish the failures above despite that transport error. Local copies
are `/data/tmp/fsfs-gh43-clean-followup-2-4ea318c0-20260911.remote.log`, the
matching `.exit.json`, and
`/data/tmp/fsfs-gh43-validation-20260911/quickstart-Z6m9dthU/receipt.json`.
The incomplete workspace attempt is retained in
`/data/tmp/fsfs-gh43-permission-workspace-vmi1227854-20260911.remote.log`.

An isolated native rerun was prepared with byte-identical binaries on
vmi1156319, but that host became busy and RCH refused admission with exit 103
(`RCH-I003`, no free slots). No test ran there and no local fallback occurred.
Neither timeout budgets, expected phases nor golden fingerprints were changed.
The cause of the two quality-budget failures remains unresolved; the results
do not establish an inference-performance comparison or ARM qualification.

Another session committed and pushed the reserved GH #43 edits as `90e8fb14`
while validation was running. This does not change the evidence boundary.
Previous-generation retention remains unresolved under the existing publisher
HOLD, cross-process tokenizer reuse is absent, and both original beads remain
open. Retained commands and outcomes are linked from their Beads comments.

## GH #41 continuous-writer lease repair — 2026-09-11 UTC

This update covers `bd-458gu` and the concat retirement path in
`frankensearch-quill/src/index.rs`. The unchanged-production baseline used
`021ad1ac` plus the two test-only probes preserved in
`/data/tmp/fsfs-gh41-combined-baseline-20260911.patch`. RCH on vmi1152480
executed both probes: 0 passed, 2 failed. A continuous writer committing 192
single-document batches produced 24 segments and last docid 1,507,335. The
same regression with conditional retirement produced 3 segments and docids
0–191, with every original document identity preserved. All 21 selected
concat, cross-shard, compaction and lease controls passed. These are structural
correctness results, not an incumbent-relative performance claim.

The separate durable diagnostic retained 24 searchable identities across 24
writer reopenings but left 24 segments, versus 3 for its 24-document continuous
control. The real concat builder expanded eight reopened-session segments
from 10,672 total source bytes to 16,345,526 bytes over a 458,753-row hull.
This supports retaining the existing hole-ratio guard. The temporary failing
diagnostic was removed after capture; its patch and terminal failure remain
available. Reopen fragmentation, merge fuel and visibility checkpoints remain
open obligations, so this repair does not close GH #41.

The baseline ELF SHA256 is
`d05a4f6b802dc7ade87187c47ae431f6f6c135d8b7fecf488815e3e58635f62a`;
the passing candidate ELF SHA256 is
`beaf165fca329edf974fdfdab0cdb3496df8d4e277ccda0c4780382ca1d93c25`.
Candidate `index.rs` SHA256 is
`3f15d88e1162ced4a244b722c99284a1c177c52dec25e39594f4eaa051c8007f`.
Exact commands, source/worker hashes and terminal output are retained under
`/data/tmp/fsfs-gh41-combined-baseline-20260911.*` and
`/data/tmp/fsfs-gh41-candidate-20260911.*`. The candidate worker-source receipt
is the `worker-source-final-sha256` file captured after synchronization; the
earlier capture still contains the baseline and is not candidate provenance.
Earlier attempts include a disk-pressure cancellation and a compile failure
in the temporary diagnostic, corrected before the observed baseline. Neither
is a test verdict. Existing workspace quality-gate failures described above
are not cleared by the focused Quill result.

Final Quill validation completed remotely on vmi1152480 at
2026-09-11T21:03:42Z: workspace formatting passed, Quill all-targets clippy
passed with `-D warnings`, and the complete default-feature Quill library suite
reported 681 passed, 0 failed, 3 existing ignored, 0 filtered. The exact job and
terminal output are in `/data/tmp/fsfs-gh41-quill-validation-warm-20260911.log`.
An earlier wrapper job selected a cold target directory and was canceled before
completion; the final job explicitly reused the candidate's Cargo target. This
does not certify the full workspace gate, release lanes or all-feature gauntlet.

## 2026-09-12 dependency and release continuation

Release bead `bd-dsbym` tracks the user-authorized dependency refresh and next
full publication. The complete granular checklist and per-dependency state live
at `/data/release-work/frankensearch-release-20260912/progress.json`.
The live 67-package registry census and upstream source research found 19 newer
direct packages; FrankenSQLite 0.3.18 remains current. Asupersync 0.4.11 was
rejected after a source-verified remote core run returned 1,122 passes and a
255.521109 ms shadow serving-latency failure against the unchanged 100 ms guard.
Its new current-thread post-root drain executes the deliberately slow shadow
before returning. The prior runtime family is restored, with consumer resolution
capped below 0.4.11; the unchanged baseline rerun passes all 1,123 core tests
and 986 fusion tests, with four existing fusion ignores. Tokenizers 0.23.2
and its required Daachorse 3.0.3 transition pass 402 embedder tests, eight logging
contracts, and three real Potion tests. FastEmbed 6.0.3 removes the separate
Tokenizers 0.22.2 dependency; it passes 426 embedder and 35 reranker tests plus
the exact MiniLM, Snowflake, and Nomic certificates. The stale producer protocol
fields were corrected, all historical fingerprint fixtures retained, and the
426 + 35 unit tests and six real-model checks passed again. The new changelog
entry explicitly records the semantic-index rebuild consequence. crc32fast
1.5.1 passes 160 durability, 774 index, and 681 Quill tests. Wide 1.7.0 passes
3,093 focused tests plus the real native F32 certificate/batching/repeatability
test on x86. TOML 1.1.6 passes 1,123 core tests. FrankenTUI 0.7.0 passes 2,103
fsfs, 827 ops, and 205 shared-TUI tests; all 1,776 tracked non-coordination
source files match after execution. After the first ureq 3.4.1 attempt lost its
worker SSH connection during compilation, all nine client tests passed on an
admitted alternate worker. JSONSchema 0.56 passes 119 fsfs schema tests and eight
gauntlet schema tests; Ed25519-Dalek 3.0 passes all 21 supervisor signing and
verification tests. Each worker's source matched after these runs. Tantivy
0.26.2 passes all 129 lexical tests, but the broader gauntlet has reported
cancellation-receipt, Cargo configuration guard, and startup-deadline failures.
After restoring accurate worker Git metadata and isolating TMPDIR from workspace
Cargo configuration, all 14 selected tests pass without assertion changes.
The broad run was canceled after its observed failures and is not a full pass.
The next oracle run passes 11 tests but fails the live Q1 merge fixture because
it did not construct the required interior burned lease tail. Source comparison
again matches all 1,776 files. The requested updater skill paused repairs after
11 accumulated test-failure events; the owner then authorized continuation.
The fixture depended on an obsolete assumption that a single-writer merge
always retires its lease. It now reopens the committed first-stage snapshot,
making the unused reserved tail unavailable to subsequent batches while keeping
every original gap, identity, merge-order, and query check. The existing snapshot
constructor is also exposed to the conformance feature. Default Q1 (one test),
oracle contracts (12 tests), the unchanged continuous-writer regression (one
test), and a library-only no-default-features compile check all pass remotely.
Matching source manifests and executable hashes are retained. Full release
qualification remains outstanding; capacity and cache-permission refusals are
not passing tests.
Oracle v8 and QG-1's
new screen version distinguish the new dependency from retained historical
evidence. Further dependency and release qualification remains in
progress. Publication-contract positive and negative self-tests pass;
`cargo audit` reports zero vulnerability advisories, four unmaintained warnings
and the existing lru 0.16.4 unsoundness warning (RUSTSEC-2026-0253).

The existing native ARM issue #47 is now independently reproduced through an
isolated RCH Darwin route. The exact ignored native Int8 certificate test ran
and failed during loader admission after model verification: zero passes, one
failure, 0.15 s, exit 101. Source comparison matched 1,776 files; the ARM64
Mach-O SHA-256 is `e6765302951134f17302ed633e628b4f4efcec1de8fd760e0c44d1449ff98381`.
The terminal log and source/executable records live under the release work
directory's `rch-mac/`. A matched-source x86 run passes the same Int8 certificate
test (one pass, 1.78 s); its ELF SHA-256 is
`29cecb02573818ad6e7f170b7cc8369fdbe61ef3c0340ba85a2db6956238d9ba`.
The two builds select the same native features without CPU-feature overrides.
This does not prove a specific numerical root cause,
qualify F32 or reranking, or validate the reporter's proposed hash. The
changelog names this existing limitation and distinguishes the standard ONNX
quality default. Bead `bd-6kafg` is reopened for numerical investigation now
that native execution is available; its acceptance criteria are unchanged.

The v1.10.0 source-to-current history review found two capabilities missing from
the unreleased prose: HNSW v7 durable row maps and refusal-safe incremental append
(`ecbc6598`, `8f429d2e`, `33b8dacb`, `ee95e999`), and same-descriptor streamed
FSLX witness authentication (`b51b46ab`). Their implementation diffs and existing
regressions were inspected; the new notes describe behavior and migration
requirements without inferring a fresh test pass or performance certification.

CASS search timed out while its index refresh was owned by another process;
`cass view` successfully read the prior release session directly. Original
v1.10.0 receipts confirm six binary variants, 13 crates and nine public consumer
lanes. The recovered initial DSR configuration is historical, not the final
configuration or authorization to bypass RCH. The live Homebrew tap has no fsfs
formula. No version bump, new tag or publication has occurred in this run.

## September 15 supplement: installer profile and model selection

Scope is the installer workstream `bd-fsfs-cross-platform-semantic-installer-46z3u`,
not a reconstruction of all intervening commits. Reviewed implementation and
policy-test diffs in `6d345be35ca48925c42a5c4ac8c54805cc19cca6` and
`4d73425ae6238368017024ec234ba2dee6ee5325`, the bead's retained failure comments,
and the shipped full/lite 1.10.0 binaries. The existing installer suite passes
with exact model arguments, same-version replacement, offline download refusal,
separate quality-model rejection, checksum preservation and rollback checks.
Its scripted CLI fixtures are policy evidence, not real-model qualification.

Real RCH execution reproduced the old version-only no-op. The intermediate
installer made an HTTPS connection during `--offline`; that run timed out and
is not a successful transition. The final explicit-pair implementation installed
the genuine full binary over lite with only the nine default-model files and
completed indexing without observed internet socket calls. Its first search
returned hits but exceeded the unchanged 500 ms quality-refinement deadline.
The broad four-platform installer bead and release gates remain open.

Live `gh release list` on September 15 still reports September 8's `v1.10.0`
and `crates-v0.5.0` as the newest GitHub Releases. The September 12
`frankensearch-v0.6.0` remains a plain tag associated with the existing crates.io
publication. This supplement adds unreleased behavior notes without inventing
a new release or claiming complete September 14–15 history coverage.

### September 16 Windows runtime follow-through

Reviewed `54a6fdc80af048496be71f0f5cdadc52f4bdf002` against the retained
Windows failure and the actual native regression executable. The original full
binary stopped indexing at the unsupported publication lease. The corrected
binary indexes successfully; five publication tests pass, including separate
child contention and post-release acquisition. Full search still returned
Initial because the ONNX constructor rejected the executing output certificate.
This is recorded as a failed full runtime gate, not a passing semantic release.

The prior September 7 ONNX platform probe was recovered from the Mac release
worktree and rebuilt through RCH with FastEmbed 6.0.3, the current workspace's
registry package identities, and nightly-2026-08-31. On Windows and Linux, two
fresh processes per model reproduce their platform certificates. All three
Linux values match their existing registered certificates. Raw vectors for 355
texts per model and diagnostic f16 ranking comparisons are retained under
`/data/release-work/frankensearch-release-20260912/windows-onnx-platform-probe/`.
The ranking comparison is a Python reference calculation, not a production
index execution or a performance claim. The actual Windows owning-loader
executable subsequently passed all three real-model tests, including historical
and foreign-platform certificate rejection, and the historical manifest fixture
passed separately. PE SHA256 is
`cc22ea415a9a37b273b228079da057c82ec8aed9a6da860aa106182bf6dc5bf8`;
receipts are retained in `windows-onnx-qualified-native-tests/`. Linux regression
execution and the full CLI gate remain required before release qualification.

### September 17 published SQLite repair

Reviewed the ten manifest floors and twenty lock entries in
`e90b9e9b2edde2fb7eaea3a65c8baf7cbbc4fc9c`. Published facade, core, pager,
types, and FTS5 archive checksums and production sources match upstream tag
`v0.4.4` at `9d3d98778a372aba95d76d05c5c974ac0238c96a`. Their internal
requirements also floor SQLite at 0.4.4. The unchanged savepoint/growth and
composite UNIQUE churn tests each pass 1/0/0 through RCH on vmi1152480, using
only registry packages. Their executable hashes are respectively
`87439b4142e7453e4c4b8da4a17d46df7297927f6b1cc47decef1d8aa058c0b0` and
`1715c97c913b97fb555907979bf411474c6e50915b1eddd2383bad3fd38c8ca1`.
Source barriers pass before and after each run. Logs are retained under
`/data/release-work/frankensearch-release-20260912/sqlite044-validation-r2/`
on that worker and in the local dependency controller log.

The worker uses source `5c6e6516` plus the SQLite manifest/lock overlay;
later queue/refresh commits are not covered by these focused results.
Consumer suites finished with 3,449 passes, no failures and 15 existing ignores:
storage 351, durability 160, ops 827 and fsfs 2,111. The final receipt and four
hash-verified logs are retained in `dependencies/sqlite044-stage-evidence/`.
The combined graph and new binaries require release qualification. Live GitHub release metadata on September 17
still reports `v1.10.0` and `crates-v0.5.0` as the newest published releases.
This supplement does not claim a new frankensearch release or complete
coverage of the concurrently arriving queue/refresh workstream.

The subsequent direct dependency refresh includes FastEmbed 7.0.1. Its
manifest-history suite passes 126 tests with one existing ignore, and all three
existing Linux ONNX certificates pass in separate processes. Historical hashes
and numerical certificates were not regenerated. Executable SHA-256 is
`598e23d5100d0fae94de2f5bfd6ef8355d51fa7f2c6ac3c6b33511125984af5a`;
the receipt and hash-verified stage logs are in
`dependencies/fastembed701-stage-evidence/` under the same release work directory.
The subsequent RCH cross-build passes native execution on `Mac-mini-max`:
126 manifest tests with one existing ignore and all three unchanged macOS
certificates. Mach-O SHA-256 is
`cd3b76353cad3415855e1be9a015088e5b2455a1a177633d55a4da707d6bb39c`;
the receipt is `dependencies/darwin-fastembed701-native-results/receipt.json`.
This does not yet qualify Windows or the combined release.

### September 20 post-0.6.1 source supplement

Reviewed the complete non-merge inventory from `7cc86150` through
`aaf8bae1169f39acdaf12c0e58b65a0ab876ab7c`, including commit descriptions,
changed surfaces, numeric constructor/test diffs, shutdown signal handling,
native API declarations, and the retained qualification receipts. The new
changelog section is explicitly unpublished. A live GitHub release-list read
on September 20 still reports `crates-v0.5.0` and `v1.10.0` as the newest
GitHub Releases; the existing 0.6.1 registry publication is a separate event.

Coverage:

- `0334d547`, `8b312bac`, `a8d2dce9`, `643127ee`, and `d7ef46e4` repair
  output/fixture assumptions and independently pin physical row provenance.
  The historical 0.6.1 diagnosis corrections remain in that release entry.
- `1e8d986d`, `5672f088`, and `aa6bf6b0` cover heap allocation, staged
  hydration/provenance validation, and the final lint/format correction.
  The focused receipt `dependencies/native567-lintfix-evidence/receipt.json`
  binds formatting, strict Clippy, 18 hydration tests, and three public
  provenance tests; its SHA-256 is
  `ea21381e6015935ce8affe3eeb8abb0a4683c877ea4b8e932a50b64429bea54b`.
- `1ce6ad0f`, `923e18c7`, `17021d4c`, `c4fdf881`, `8b1c89ed`,
  `9f2e2d2b`, `8268b88f`, and `9835ba53` add staged/live generations,
  deadline-aware progressive queries, source-cohort admission, immutable
  scopes, and tuning. Their original commits report no Rust execution;
  current combined-source qualification remains in progress. No shipped
  fsfs cutover, preemptive deadline, or performance certification is claimed.
- `6f2ad609` shares the three-second force-exit window between SIGINT and
  SIGTERM, including mixed sequences; read directly against shutdown.rs.
- `86337594` separates persisted numeric-column cardinality from the live
  scorer domain. `e2eb8d74` migrates all 20 existing constructor test calls
  and adds the tombstone/Boolean regression. The verified receipt at
  `dependencies/numeric863-r2-evidence/receipt.json` records 17 passes,
  zero failures/ignores, formatting, and strict library/test Clippy. SHA-256:
  `5ea2a79b17412634602c7979434ef0d774b4186390d85c9cc218e7dc1e895b7d`.
- `6181cc4b` and `83c7ba74` preserve an unintegrated publication-reuse
  proposal and archive its preparation workflow. No enabled optimization or
  workflow execution is claimed. Beads-only bookkeeping commits add no
  product capability.

The receipt paths above are relative to
`/data/release-work/frankensearch-release-20260912/`. Linux GNU and musl
candidates and the Windows embedded candidate at source `643127ee` have
separate successful runtime receipts, but do not contain later numeric,
shutdown, or native API changes. Darwin builds and the unchanged default
quality gate now target frozen `aaf8bae1`. The complete Quill inventory,
exact public R1 comparison, and all current-source release gates remain open.
An identical dependency-update bullet was removed from the changelog without
changing its recorded qualification scope.

### September 20 follow-through through `9c37418e`

Reviewed the complete commit inventory after `1a5a5b87`, including the merged
Quill document-frequency accounting fix `0642c5fc`, its linear-ceiling tests
`86c2e99c` and `0a779d12`, the configuration explanation `c3e15633`, publication
regressions `500b89ff` and `dc02682a`, and signal ownership fix `cfcb6831`.
The production fixes are recorded in the existing unreleased section.
Native ANN helper/fixture repairs and formatting are validation maintenance;
publication tests do not imply that the archived reuse proposal is enabled.

Both new representative commit URLs resolved through the GitHub API on
September 20. A fresh release-list query still returned `crates-v0.5.0` and
`v1.10.0` as the latest crate-bundle and binary releases. No new publication
is inferred from the source commits.

The unchanged clean-source quickstart gate passed on `6fd1362a`: 12 executable
checks, 18 negative controls, no remaining owned processes. Its retrieved
receipt SHA-256 is
`34c3fefbc20b6e0a6e396ab5604ff07ae0936c62579aeb7049c053015013ac29`.
The complete default gate and full Quill inventories on `9c37418e` are running
through RCH jobs `30023605353973348` and `30023605353973346`, respectively;
these are not terminal success receipts. Exact public R1 acceptance remains
open independently. Detailed evidence and the current checklist remain in
the release-work directory cited above.

The default gate subsequently finished with two failures: strict workspace
Clippy rejected `.err().expect()` in the new publication cancellation test,
and the native-quality daemon's first refinement exceeded its unchanged
500 ms deadline. All other stages, including the executable quickstart,
passed. The retrieved full-gate log SHA-256 is
`e822f9a974289e2ea5742b2baca90844765c16230e784ec0cf23f04dd816c3b2`.
The assertion is repaired with `expect_err`; the timeout remains under
investigation. Neither this repair nor an isolated replay replaces a complete
release verdict. The full Quill invocation remains nonterminal.

### September 21 executable generation and configuration supplement

Narrow scope: the complete-generation commits `83a558f6`, `ceea888b`,
`1ff0bb1d`, `a2bae41d`, and `a6992b70`, integrated at `e57980ae`, plus
the owned repairs `4f30ffee` and `e7808fe6`. Reviewed their source diffs,
the command parser, config init/reset callers, and release bead `bd-dsbym`.
This is a supplement to the earlier history, not a new publication record.
The GitHub API still reports binary release v1.10.0 (September 8); Actions
remain disabled, checked September 21.

RCH worker vmi1264463 validated frozen source `e57980ae` plus the exact
repair diff now committed in `e7808fe6`. Retrieved evidence is under
`/data/release-work/frankensearch-release-20260912/dependencies/config-toml-validation/`:
42 generation unit tests, two configuration regressions, five executable
tests including the explicitly enabled real-Potion case, and strict fsfs
all-target Clippy passed. The production executable test log SHA-256 is
`0db2f1fdb510f40020b6afb051450067f87ed458b6a9b4dd6bbf34f0189634a7`;
its test ELF is `6e7ff18d80ea64efd7040683aed3a61126bbc633a327fa72b28e68066fbc053e`.
The tested production binary was separately hashed after the test as
`9994481a7a8b761198575f0d2e91b649b93bc99ece9f5bc24dd397a320c64f2f`.

The overall receipt remains FAIL: four pre-existing golden receipt checks
still differed. Diagnostics isolated JSON map ordering and receipt hashes;
that follow-through is not credited as complete here. An earlier golden
rerun reused a test ELF embedding the previous snapshot's fixture path and
is explicitly excluded. Subsequent fresh-target runs reproduced the mismatch.
No failed test, ignored test, pending full gate, R1 comparison, native quality
deadline, or platform build is promoted to a release verdict. This was solo
re-execution and review, not independent verification.

The subsequent JSON-order repair completed its fresh remote run on worker
vmi1264463: 117 tests passed (13 golden, 55 producer identity, 49 config and
generation/CLI), plus strict fsfs all-target Clippy. The explicit production
`serde_json/preserve_order` feature removes accidental workspace feature
unification from receipt hashing. Four CLI fixtures retain their pre-format
values and serialization layout; expected hashes were not regenerated.
Retrieved logs and receipt are under
`/data/release-work/frankensearch-release-20260912/dependencies/json-contract-validation/`.
Every stage log was checked against its receipt hash. Receipt SHA-256:
`f3698907fbdf35b7384827a5030726f73ec10befcbd0e8ccc07594aa8636a4d2`;
strict Clippy log: `adb6bb39e9d14dc1b3cd2b50c76a6281ea3e6c8bd33c258e194abf3fc867e1ac`.
This closes the four observed CLI golden failures only. The separate complete
quality gate exposed TUI/sealed-Quill fixture formatting damage and a Quill
test guard lifetime lint; those repairs and the native cold-start test correction
remain under validation. R1 and platform release qualification remain open.

The native cold-start correction subsequently passed its actual production CLI
test on RCH worker vmi1264463 in 94.95 seconds. The pre-existing README explicitly
allows cold model initialization to exceed 500 ms. The test now admits only that
typed timeout with populated Initial hits, or successful refinement; negative
controls reject empty results, wrong counts, unrelated failures, inconsistent
refined payloads, and changed timeout budgets. Runtime deadlines are unchanged.
The original forced 50 ms cold timeout, observed loader completion, uncached
500 ms warm refinement, single constructor, and clean shutdown assertions all
remain and executed successfully. Missing/corrupt models, producer mismatch,
append, and vector preservation checks also passed.

Observed split: direct and streaming cold requests returned the typed 500 ms
timeout with Initial hits; the daemon and post-append requests refined. These
degraded responses are not performance wins. The unchanged old test failed again
in the separate full gate on worker vmi1152480 (seven passes, one native failure).
This is solo re-execution against the existing contract, not independent review
or full release acceptance.

Retrieved native log SHA-256:
`a3c7751d0c27a2cb237109683b7aef647132248e0cdaaca51eaf478b6b7d0ac4`;
test ELF: `f56cb403d658bba829d120051499ecb05b24208160c6806fd6734e61dd0f008f`;
production executable: `3a4b87584f4d61a6100e97bad5f92189880fbd1713e9700592aa1bad9824aefd`.
The negative-control test and both Quill snapshot-statistics regressions passed.
Strict fsfs/Quill all-target Clippy passed after shortening the test mutex guard
lifetime; its log SHA-256 is
`aa753839346cfad88db44272df91ef4566aebc90e50ca8fc8ba63c6d36b070b0`.
The two Rust files received 94 explained, site-specific UBS annotations for
91 intentional test panics and three validated test-executable launches. Those
statements remain intact; scanner configuration did not change. The static scan
exited zero with zero critical findings, 3,773 warnings, and 488 informational
findings. It is not a warning-free result. Byte-fixture and full-gate validation
remain separate obligations.

The full gate on the earlier dirty JSON-order snapshot finished FAIL. Its
retrieved receipt SHA-256 is
`72d92795e80e9defabfa69dbd2961560602e6d891691e811a02664ccacb5ffe9`.
The failing stages were Clippy, workspace library tests (three TUI goldens),
bounded Quill (five sealed-campaign failures), fsfs (one benchmark-fixture hash),
native E2E (the cold assertion), and quick-start provenance. The last failure
was my validation setup: `--require-source` correctly rejects a dirty checkout,
even when the overlay is frozen and separately hashed. The next complete gate
must use a clean checkout of the committed repairs; that check is not relaxed.

Byte restoration covers 15 Quill archive fixtures, the normative machine
registry, three TUI goldens, and three benchmark baseline fixtures. Every JSON
value matches the historical source at `4f30ffee`; every file matches those bytes
modulo its final newline. Sealed SHA constants, benchmark thresholds, and test
expectations remain untouched. The three restored TUI tests passed remotely;
the Quill archive tests and benchmark rerun remain pending here. A read-only
scan of all 420 reformatted JSON paths against 554 Rust source files identified
the additional fixed-hash consumers. UBS returned exit 3 for JSON alone because
it has no JSON scanner; this is not a pass. A cumulative scan including the two
actual Rust files repaired in this work block exited zero, but does not validate
JSON. Historical-byte comparisons and the consumer tests provide that evidence.
