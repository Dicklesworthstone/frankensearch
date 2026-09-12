# Dependency Upgrade Log

## 2026-09-12 — Stable dependency refresh (in progress)

### Current integration status at 19:49 UTC

Main now resolves Asupersync 0.5.0 and FrankenSQLite 0.4, following a separate
publication session. Live crates.io records confirm facade 0.6.0, core/lexical/
Quill 0.3.0, and rerank 0.4.0 were published today. The downloaded facade and
lexical archives both identify source `dd093fb230404ab08be2ed6f27776ed6c4796485`.
fsfs remains 1.10.0. These published versions are immutable; publication alone
does not establish that the complete release gates passed.

The earlier compiler and dependency-family results below describe the isolated
Asupersync 0.4.10 candidate, not the current runtime/storage graph. Both broad
product and retained-output Quill runs reached RCH's 1,800-second transport
deadline during compilation. Neither produced a complete test verdict. The
installed RCH accepts per-invocation `RCH_BUILD_TIMEOUT_SEC` and
`RCH_TEST_TIMEOUT_SEC`; the current focused run uses finite 7,200-second
transport allowances without changing test deadlines or shared configuration.

The package bumps left three integration defects: the current oracle contract
still named lexical 0.2.5, the built-in profile still named Quill 0.2.4, and the
standalone fuzz crate still selected a different Asupersync `Cx` type. The
candidate introduces oracle v9 and profile v7 for the actual 0.3.0 adapters,
retains exact historical v8/v6 identities for archive inspection only, and
aligns the fuzz runtime closure to 0.5.0. Focused remote tests are pending.

### Earlier dependency census and isolated candidate

The live crates.io census found 19 newer direct packages among 67 registry
dependencies. FrankenSQLite 0.3.18 was current at that census. Updates were applied and
validated one library family at a time under bead `bd-dsbym`; publication waits
for the complete release gates.

The combined dependency candidate passed the compiler stages on `hz3` at
2026-09-12 13:51:44 UTC: formatting, workspace/all-target checking, workspace
and hybrid-feature Clippy with warnings denied, and the Windows index compile
guard. The remote command exited 0. All 1,776 tracked non-coordination source
files matched during the final stage and after completion. The Windows check
emitted nine warnings; that stage does not deny warnings. This is compiler
qualification, not a complete release gate. The later broad attempts timed out
during compilation, as recorded above.

The follow-up native F32 MiniLM test passes with the new certificate-refusal
diagnostics: one test, 3.15 seconds, remote exit 0 on `ovh-a`. It checks exact
certification, batching, repeatability, and bounded diagnostics without exposing
conformance inputs. The 1,776 source files match after execution. The executable
was removed before later hash retrieval, so this run has no retained ELF hash.
It does not resolve the separate ARM Int8 certificate refusal.

The full Quill attempt ended without a terminal test result; its successful
build is not a suite pass. The driver previously buffered both output streams
until child exit, losing test output when interrupted. It now writes both
streams incrementally and reports observed byte growth. A real child that
requires readable output files before exiting fails against the former driver
(exit 9) and passes against the revised driver. An inherited-pipe control still
times out and kills the process group. Test selection, validation, and budgets
are unchanged. The existing repository probe suite and full default/all-feature
run did not complete before the transport timeout.

### Earlier Asupersync 0.4.11 rejection and 0.4.10 baseline verification

The attempted runtime-family upgrade compiled, but the core suite returned
1,122 passes and one failure on `vmi1156319` (RCH job `30017369531219996`).
`shadow::tests::wall_clock_guard_never_awaits_slow_shadow_backend` took
255.521109 ms against its unchanged 100 ms limit; its shadow backend deliberately
blocks for 250 ms. Cargo stopped before running the fusion suite.

The [0.4.11 current-thread driver](https://github.com/Dicklesworthstone/asupersync/blob/v0.4.11/src/runtime/current_thread.rs)
drains queued work after the root result becomes ready, before `block_on`
returns. A dispatch-count bound cannot prevent an individual shadow poll from
blocking the caller. Replacing this guard with a different runtime or moving
its timer inside the root would stop checking the existing caller-visible
contract; neither change was made.

The runtime and its coupled lock entries are restored to the prior versions.
The manifest retains the 0.4.10 API floor and temporarily excludes 0.4.11 from
consumer resolution. The separate fuzz manifest's stale exact 0.4.4 pin is
aligned to 0.4.10. The unchanged baseline rerun on the same worker passes:
core 1,123/1,123, including the shadow latency guard; fusion 986 passed,
four existing ignores, zero failures (RCH `30017369531220011`, exit 0).
The failed run's source files matched the isolated checkout both during
compilation and after execution; it is a real failed test, unlike the earlier
excluded RCH transfer/source-drift attempts. Terminal logs and source manifests
are retained under `/data/release-work/frankensearch-release-20260912/dependencies/`.

### Tokenizers 0.23.1 → 0.23.2 — verified

The [upstream patch](https://github.com/huggingface/tokenizers/releases/tag/v0.23.2)
adds a default pretokenized-model hook and vocabulary-size improvements; the
used tokenizer constructors and encode APIs remain compatible. The existing
`default-features = false` and `fancy-regex` selection are preserved. Its required
`daachorse` dependency moves from 1.0.1 to 3.0.3. No other resolved package
version changes in this step. Fresh-process receipt identities follow the new
tokenizer version. Remote validation passes 402 unit tests and all eight logging
contract tests (RCH `30017369531220019`). The three explicitly selected real-Potion
tests also pass: conformance certification, shared initialization, and bit-exact
streamed matrix/embedding parity. The five default ignored tests include these
opt-in model lanes and an unrelated performance probe. All 1,812 tracked source
files matched the worker during compilation and after testing.

### FastEmbed 6.0.0 → 6.0.3 — API and ONNX conformance verified

The [published 6.0.3 package](https://crates.io/crates/fastembed/6.0.3)
retains the builder APIs used here; new CPU-fallback and dimension-override
options have defaults. The existing disabled defaults and
`ort-download-binaries-rustls-tls` feature remain unchanged. FastEmbed now shares
Tokenizers 0.23.2 with the native adapters, removing the separate 0.22.2 package.
The embedder suite passes 426 tests (eight existing opt-in ignores), and the
optional ONNX reranker suite passes 35. Explicit MiniLM, Snowflake, and Nomic
real-model tests all match the unchanged exact vector certificates and reject
historical certificates. Every additional fixture file was checked against its
pinned revision, size, and SHA-256. The audit reports no vulnerability advisories
and retains four unmaintained-package warnings and the existing `lru` soundness
warning; none is hidden.

The review also caught stale producer provenance in the model manifests.
Current protocols now name Tokenizers 0.23.2 and FastEmbed 6.0.3. All seven
historical fingerprint fixtures remain verbatim: their reconstruction changes
only those dependency fields, preserving artifact, numeric, preprocessing, and
vector-certificate fields. The corrected manifest passes the same 426 + 35
unit tests. All six explicitly selected real-model tests also pass with the
corrected producer (three ONNX certificates and three Potion checks).
The full release gates remain pending. Existing semantic indexes require rebuilding
when their stored producer identity differs.

### crc32fast 1.5.0 → 1.5.1 — verified

The published patch adds wider x86 VPCLMULQDQ implementations and documents
the existing AArch64 CRC path. The used `Hasher` API is unchanged. This step
updates only crc32fast in the lockfile. The admitted `hz3` worker passes 774
index tests (15 existing ignores), 681 Quill tests (three existing ignores),
and 160 durability tests. All 1,776 tracked non-coordination files match before
and after execution. No performance gain is claimed.

### wide 1.6.1 → 1.7.0 — focused suites verified

The published release consolidates SIMD implementations and adds operations.
Used vector arithmetic APIs remain compatible; changed `signum` semantics and
deprecated swizzle methods are not used here. Validation targets the existing
SIMD/scalar parity, exceptional-float, quantization, and postings tests. The
separate transitive wide 0.7 line remains under its own consumers' constraints.
Remote tests pass core 1,123, embedder 355 (two existing ignores), index 774
(15 existing ignores), Quill 681 (three existing ignores), and durability 160.
The native F32 MiniLM certificate passes with both the prior wide version and
wide 1.7.0. The latter explicitly selected test checks certification, batching,
and repeatability on the pinned real model (one pass, zero failures, 8.43 s).

### toml 1.1.4 → 1.1.6 — focused configuration suite verified

The patch fixes ownership of borrowed numeric values and avoids unnecessary
table cloning. The used serde/configuration APIs remain compatible. Only the
toml package changes in this lockfile step. The remote core suite passes all
1,123 tests, including configuration round-tripping, partial defaults, and
invalid-input fallback. All 1,776 tracked non-coordination source files matched
during compilation. Full fsfs coverage remains part of the release gate.

### FrankenTUI 0.5.x → 0.7.0 — consumer suites verified

The nine direct framework packages move together. Published 0.7.0 retains the
frame, layout, and widget APIs used here. The existing markdown feature remains
enabled; runtime telemetry and its HTTP dependencies remain disabled. The
optional Asupersync executor is not enabled by these consumers. Validation
covers the shared TUI and both fsfs and ops consumers: 205 TUI tests, 2,103 fsfs
tests (14 existing ignores), and 827 ops tests (one existing ignore), all passing.
All 1,776 tracked non-coordination source files match after execution. The
lockfile changes only the twelve coupled framework packages in this step.

### ureq 3.4.0 → 3.4.1 — client tests verified

The patch repairs timeout handling around TLS and connection establishment and
buffer reuse. The JSON feature and default TLS configuration remain unchanged.
The project uses ureq in its query-expansion client; existing tests cover client
configuration and response parsing, not live transport behavior. The required
ureq-proto dependency moves from 0.6.1 to 0.6.2; no other package changes in this
step. After hz3 lost its SSH connection during compilation, an admitted ovh-a
rerun passes all nine query-expansion tests. The other two selected consumer
binaries have zero matching tests and contribute no additional passes. All
1,776 tracked non-coordination source files match after execution.

### jsonschema 0.50.0 → 0.56.0 — schema contracts verified

Both development dependencies advance together. Defaults remain disabled;
the newly optional `idna` feature is enabled explicitly to retain the earlier
internationalized-name behavior. Validation targets the fsfs schema fixtures
and the gauntlet's current and retained divergence-register schema checks.
All 119 fsfs schema tests and eight gauntlet tests pass. The required supporting
packages jsonschema-regex, jsonschema-value, and referencing move to 0.56.0;
fraction moves to 0.17.0. Both workers match all 1,776 tracked non-coordination
source files after execution. Historical schemas and fixtures are unchanged.

### ed25519-dalek 2.2.0 → 3.0.0 — signing contracts verified

The signing-key, signature, and verification APIs used by ArtifactStore v4 remain
compatible. The update changes the RustCrypto dependency family; stored signed
records, domain separation, and rejection checks must retain their existing
contracts. Validation targets the real supervisor signing and verification tests.
Asupersync's nkeys dependency still requires the separate 2.x line. Updating the
gauntlet consumer adds the five required 3.x-family packages while retaining that
transitive edge; forcing every old 2.x requirement to 3.0 correctly fails
resolution. No dependency requirement was loosened to evade that constraint.
All 21 supervisor tests pass remotely, including valid signatures, tampering,
wrong keys, retired/revoked keys, cancellation, and timeout handling. All 1,776
tracked non-coordination source files match after execution.

### Tantivy 0.26.1 → 0.26.2 — validation in progress

The patch fixes nested aggregation flushing and buffered union seeking. Only
Tantivy changes in this lockfile step. The current oracle dependency record
advances to v8 with the published 0.26.2 checksum; the exact v7 record remains
readable as historical evidence and cannot authorize a new run. Earlier frozen
oracle versions and hashes remain unchanged. The QG-1 incumbent screen also
advances its protocol version and rejects screens from the prior dependency.
Validation runs the lexical consumer and the complete gauntlet library suite.
The lexical suite passes all 129 tests. The first broad gauntlet attempt was
stopped after seven failures, before its slow evidence-assembly tests finished;
it is not a completed suite. Four cancellation failures shared a missing
compiled Git revision because RCH excludes `.git`. The worker now has accurate
Git metadata for the isolated checkout's actual base and retains its dirty
overlay. `TMPDIR=/tmp` also keeps Cargo configuration isolation fixtures outside
the real workspace's ancestor configuration. With that environment, all 14
selected cancellation, configuration-guard, and unchanged startup-deadline tests
pass. No validator or assertion was weakened. Full qualification remains pending.
The first subsequent oracle-contract run passed 11 tests, including the current v8
identity, every retained v2–v7 identity, exact lock resolution, and rejection of
retired incumbent screens. One live Q1 merge fixture failed with `Q1 E3.5 did not
construct an interior burned lease tail`. All 1,776 tracked source files match
the worker after execution. Repair paused at the library-updater skill's
explicit checkpoint: 11 test-failure events across this upgrade run, including
the seven observations in the incomplete broad run and one repeated diagnostic.
The environment-related failures subsequently passed; Q1 was unresolved at that
checkpoint. No new version, tag, or publication has been made.

The owner authorized continuation after that checkpoint. The Q1 fixture still
assumed that every concat merge retired an ingest lease, but commit
`9e3117dffa980b73e6292e29fdbd0cc7fdcf1fd4` correctly preserves safe live leases
for continuous single-writer ingestion. The candidate fixture now reopens its
committed first-stage snapshot before appending later batches; the existing
allocator starts beyond the prior reserved lease range, creating the required
real unused tail inside the final merge hull.
All original gap, merge-order, identity, and query assertions remain. The existing
snapshot constructor is exposed to `conformance-internals` as well as benchmarks
so the default gauntlet can compile this fixture. The exact fixture passes in
the default-feature build; all 12 oracle-contract tests pass with the oracle
feature. The unchanged continuous-writer regression also passes (192 documents,
IDs 0–191, three segments), and a library-only `--no-default-features` check
confirms the helper is available without development-feature unification.
Both workers match all 1,776 tracked source files after testing; executable
hashes are retained. The first candidate's `Arc<KeeperSnapshot>` argument error
was corrected by cloning the referenced snapshot; its failed compiler attempt
is not counted as a passing gate. Broader release qualification remains pending.

The standalone fuzz workspace also passes `cargo check --locked --manifest-path
crates/frankensearch-quill-gauntlet/fuzz/Cargo.toml --bins` on `ovh-a` (100 seconds,
exit 0). Both fuzz targets resolve the reconciled lockfile, and all 1,776 source
files match after checking. This verifies compilation; no fuzz campaign ran.

## 2026-09-07 — FrankenSQLite 0.3.18 source follow-through

The storage, durability, fsfs, and ops manifests now require FrankenSQLite
0.3.18. All 20 registry family members move together in `Cargo.lock`; no other
package version changes. Asupersync remains a single registry identity at
0.4.10, with the existing feature selections and registry renames preserved.

The [upstream release](https://github.com/Dicklesworthstone/frankensqlite/releases/tag/v0.3.18)
fixes WAL journal-switch sequence handling, read-only WAL reader registration,
Linux I/O cancellation safety, and mount namespace permissions. Its parameterized
rowid-IN optimization is not a demonstrated FrankenSearch speedup. The used
public APIs require no caller migration; the FTS5, value, error, context, and
RaptorQ source files are byte-identical to 0.3.17. Published source provenance
is `1600766ca698dae99b6018474bc8c150ece4a82d`.

The publish contract's audited family pin moves with the actual lock. Its
positive and negative self-tests pass, including mixed-family, foreign-source,
duplicate-runtime, and source-cleanliness cases. The live audit accepts these
dependency identities while retaining dirty-source and occupied-version
blockers. `cargo audit` reports zero vulnerability advisories and the unchanged
four unmaintained, one unsoundness, and one yanked-package warnings.

FTS5-enabled storage validation passes 386 library tests and 10 pipeline
integration tests; the existing scaling probe remains ignored. Commit/reopen,
rollback, concurrent-reader isolation, and repeated schema-open read-only
checks all pass. The first full gate has nine passing checks and one failed
check: its real-model CLI stage has five passes and one fixture failure because
the selected cache lacks native MiniLM `model.safetensors`. That test stops
before native indexing or search, so it establishes no native deadline result.
The candidate's ordinary registered downloader then downloaded and verified
91,337,345 bytes into a private native cache; the failed run remains retained.

The complete unchanged rerun passes all ten default checks, including real-model
CLI E2E 6/6, from 06:32:03Z to 06:36:58Z on `thinkstation1`. It uses the existing
`MINILM_FIXTURE_DIR` override for that verified native directory and
`FSFS_E2E_BINARY` for the freshly built optimized default-profile executable,
SHA256 `9c9eccb905933c6d9ddecd405c115c44c25b0deae3059a8b50ec4aefe84bf8d0`.
The final quick-start stage independently builds and runs the stock debug
executable, SHA256
`53651ebb8d082e396b681a4c5c9e3bed6fd243a311a90f4d1b764b01e3f7c84d`.
No stage, assertion, timeout, feature, model checksum, or producer identity was
changed. Native E2E qualifies the optimized executable; it does not resolve the
previously observed stock-debug native cold-load deadline failure.

Candidate source base is
`798a933743f2d87802b0faa75947497af94d5bfa`; the new lockfile SHA256 is
`daaba04e7e3858b13309f7bdc99458af8205abcd60c168744f5af47447374704`.
Terminal evidence is retained under
`/data/tmp/frankensearch-fsqlite-0.3.18-20260907/` on `thinkstation1`.

This source follow-through does not replace the published `v1.9.1` or
`crates-v0.4.3` artifacts, which retain their validated 0.3.17 lock. The
stock-debug native deadline, native-default, and no-C-toolchain criteria of
bd-2ba5 remain open; upstream `stacker`/`psm` still needs an assembler driver.
Watcher reader exclusion, bd-z2nfa, also remains open and depends on unfinished
composite-generation publication and retention work owned elsewhere.

## 2026-09-07 — Published fsfs 1.9.1 and the 0.4.3 crate bundle

Both annotated tags bind `9d132a0315e12da0442aa7a943852091fe043037`.
The [binary release](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.9.1)
contains six variants, two installer aliases, source/profile metadata, checksums,
and terminal verification evidence. The [crate bundle](https://github.com/Dicklesworthstone/frankensearch/releases/tag/crates-v0.4.3)
contains all 13 published registry archives and their provenance checks.

- The unchanged quality gate passed 10/10 stages, including real-model E2E 6/6.
  All 13 Cargo archives passed build verification and publication dry runs.
  Their public downloads match both registry checksums and the verified archives;
  all VCS records identify the clean tagged source.
- The actual public-registry consumer passed nine feature configurations without
  source replacement: minimal, lexical, hybrid, persistent, durable, full,
  lexical-tantivy, cass-compat, and all. Quill and Tantivy consumers index two
  documents and check matching and empty results. This is feature compilation
  and lexical correctness coverage; model execution is checked separately.
- Linux and Apple Silicon full executables passed new-corpus indexing, refined
  search, native ms-marco reranking with three actual scores, and doctor.
  The first Mac rerank check failed because its older cache lacked native
  `model.safetensors`; the ordinary registered download to a private cache and
  a new-index repeat passed. The initial failed receipt remains in the release
  evidence. ARM Linux and Intel Mac version execution used QEMU and Rosetta.
- Public 1.9.0 testing exposed an updater wiring bug: Linux GNU full selected
  MUSL lite, while Mac ARM lite selected full. The 1.9.1 fix carries the compiled
  ABI and semantic-loader profile through exact asset lookup, with a missing-full
  negative. Old clients still need the installer route documented in the README.
  Direct 1.9.0 index opening and rollback both returned refined results without
  changing either of the two vector files.

The release lock is the fully validated FrankenSQLite 0.3.17 family. Upstream
published 0.3.18 at 2026-09-07T04:20:02Z, after this dependency freeze. Fresh
consumer resolution uses that compatible patch and a single Asupersync 0.4.10,
with no git dependency or forbidden runtime. No released tag was moved to chase
the new patch, and no 0.3.18 binary claim is made.

The earlier 4096-document/2.4576 GB run retains the automatic 1.8 migration
failure and seven candidate refinement timeouts at the unchanged 500 ms budget.
Explicit rebuilding and rollback preserve relevance, but do not qualify refined
search at that scale. The user's explicit instruction that backwards compatibility
is not required authorizes the documented rebuild requirement for this release;
it does not turn the automatic migration failure into a pass. Native-default,
stock-debug deadline, and no-C-toolchain work remains open in bd-2ba5.

## 2026-09-07 — macOS ARM64 ONNX producer qualification

The fsfs 1.9.0 candidate built on Apple Silicon but its real index/search/doctor
probe rejected MiniLM's executing output certificate. Model artifact checksums
passed. Both `thinkstation1` and `mmini` report ORT 1.28.0, revision `da9b5e3`,
but the platform builds produce different f32 bits. Independent fresh Mac
processes reproduced the same four-text batch certificates for all three models.

**GOLDEN-CHANGE bd-2ba5:** qualify those Mac ARM64 outputs as separate producers.
Their numeric profile is
`ort-1.28.0-da9b5e3-macos-aarch64-cpu-f32-host-default-intra-threads-v1`.
The existing Linux profile and certificates remain unchanged. Model files,
tokenization, pooling, normalization, corpus bytes, and the owning loader's
bit-exact admission check are unchanged. A platform certificate cannot authorize
the other platform's output or silently reuse its vector generation.

| Model | Mac ARM64 four-text normalized batch SHA256 |
|---|---|
| MiniLM | `5693dd454b03d7c4ae3a96ea429eddbbaf60519e11ded361a26a1f221f996843` |
| Snowflake Arctic S | `f9f9b1071d82dd22614086a7a0e05bcdc785ca3b04158c4f914e678c75fd6c8d` |
| Nomic v1.5 | `7041b782516edfb91097d668443130d098bca7a035c8a85024150b5a09aebc67` |

Semantic review used 339 Treasure Island passages plus 16 queries on both hosts.
Corpus metadata SHA256:
`ba9534014a619e27b349385122418194c3835e4fe0d0be76bfb6eb8ba2d72e54`.
The minimum paired cosines are 0.9999999999992835 (MiniLM),
0.9999999999996847 (Snowflake), and 0.9999999999980949 (Nomic); maximum absolute
component differences are respectively 2.5332e-7, 1.3411e-7, and 2.6823e-7.
The production `InMemoryVectorIndex` f16 path returns identical ordered top-10
lists for all 16 queries under each model. Mean nDCG@10 on both platforms is
0.267848928855, 0.221664612396, and 0.453100816876 respectively. This certifies
bounded correctness parity, not performance, all-corpus equivalence, or identical
bits across batching shapes.

The gate's admission rule is unchanged: the newly registered Mac producer must
match its exact output; historical and foreign-platform certificates must fail
the actual owning constructor. No previously rejected Linux output is admitted
by this change, and no native producer or auto-detection preference changes.
The three real-model owning-loader tests pass on both hosts, including the
historical MiniLM/Snowflake and opposite-platform rejection cases. Linux's
exact manifest fixture, formatting, and strict fastembed Clippy checks pass.
Probe source, Cargo lock, vectors, checksums, and comparison receipts are under
`/data/tmp/frankensearch-release-1.9.0-20260907/onnx-platform-probe/` on
`thinkstation1`; production-index replay source is in the sibling
`platform-ranking/` directory. Mac originals are under
`~/release-work/frankensearch-1.9.0-20260907/onnx-platform-probe/` on `mmini`.

## 2026-09-06 — FrankenSQLite 0.3.17

**Scope:** owner-directed FrankenSQLite update from 0.3.8 to the latest
published stable release, 0.3.17. All 20 `fsqlite*` packages in `Cargo.lock`
move together; the storage, durability, fsfs, and ops manifests require
0.3.17. Existing feature selections and Rust call sites are unchanged.

Release preparation also updates the publish contract's audited identity pins
to FrankenSQLite 0.3.17 and Asupersync 0.4.10, and the fresh-process logging
receipt's Asupersync pin to 0.4.10. These had remained at 0.3.8 / 0.4.9 after
the lockfile moved. Before/after live census receipts remove exactly the two
dependency-version blockers; occupied versions, dirty tracked source, and
untracked package inputs remain blocked. The planner's positive and negative
self-tests pass, including mixed-family and non-registry rejection; all eight
fresh-process logging tests pass with the new identity. No source, checksum,
single-runtime, package-provenance, or cleanliness check was removed.

- **Upstream fixes:** the releases since 0.3.8 address prepared-read
  transaction release, cross-process WAL visibility/checkpoint horizons,
  FTS5 visibility and maintenance, and savepoint allocation ownership.
  See the [tagged changelog](https://github.com/Dicklesworthstone/frankensqlite/blob/v0.3.17/CHANGELOG.md).
  These are upstream changes, not claims that every corresponding failure
  has been reproduced and fixed in FrankenSearch.
- **Dependency changes:** one Asupersync 0.4.10 identity remains. New
  transitive packages are `stacker` 0.1.25, `psm` 0.1.32, `object` 0.39.1,
  and `ar_archive_writer` 0.5.3. Cargo also consolidates existing-version
  dependency edges to `windows-sys` 0.61.2, `itertools` 0.14.0, and
  `getrandom` 0.4.3; no unrelated package versions were upgraded.
- **Native build requirement:** upstream `fsqlite-core` now includes
  `stacker` on non-wasm32 targets. Its `psm` build uses a C/assembler
  driver. This update does not meet the no-C-toolchain acceptance criterion
  of the still-open native migration, bd-2ba5. The gate's Windows check
  covers the index crate, not this new database build dependency.
- **Validation:** 551 focused tests passed (155 durability, 386 storage,
  and 10 pipeline integrations), with the storage `fts5` feature enabled;
  one storage test remained ignored. Formatting, workspace check, both
  Clippy lanes, the index cross-target check, workspace library tests,
  fsfs tests, and facade hybrid integrations passed in the full gate.
  The real-model CLI stage had five passes and one failure: the existing
  stock-debug native-quality test returned `refinement_failed` at its
  unchanged 500 ms budget, matching bd-2ba5's earlier failure. The executable
  quick-start stage passed. Overall: nine gate checks passed and the e2e
  check failed; the full gate exited 1. The timeout remains unresolved.
- **Audit:** `cargo audit` reports zero vulnerability advisories, with the
  same four unmaintained, one unsoundness, and one yanked-package warning
  as the previous lockfile. UBS scanned zero Rust source files for this
  manifest-only change and supplies no additional code-scan evidence.

Validation used source base `13585b29a3e867b0221146f2926a091a2bba893f` and
lockfile SHA256 `6b0663501ab0ff9aeed97f483a4d897877f1a26a2726f1498ae1490fb29b8d82`.
The candidate was first tested through `CARGO_RESOLVER_LOCKFILE_PATH`;
the applied workspace lockfile is byte-identical. Terminal logs and the
before/after audit reports are under
`/tmp/frankensearch-fsqlite-0.3.17.lrSs6X/` on `thinkstation1`.
The gate ran from 19:12:23Z to 19:42:30Z. Its final stock-debug quick-start
binary SHA256 was `53267a580cdcea946ba6d732b7054da7a67ca66a03af67422d5906ae37ade4d8`.

## 2026-08-21/23 — Registry refresh: FrankenSQLite 0.3.8, Asupersync 0.4.9, fastembed 6, jsonschema 0.50

**Scope:** owner-directed refresh of every direct dependency to its crates.io
latest (bd-r0ar1, bd-lnexf). Verified on remote workers: `cargo check
--workspace --all-targets`, `cargo clippy --workspace --all-targets -D
warnings`, the fastembed-gated clippy lane for embed + rerank, and the
workspace test suite. No frankensearch call-site change was needed.

- **FrankenSQLite 0.3.1 → 0.3.8** (storage / durability / fsfs / ops pins).
  Ships the fixes surfaced by the cass/frankensearch cross-repo triage:
  GH#366 `ReservedEmpty` reopen of a coherent populated file, GH#370
  orphaned FTS5 `%_content` reclaim, GH#371 bounded WITHOUT-ROWID / large
  `DROP` teardown memory, GH#244 attached-schema writes inside explicit
  transactions, plus the bd-xv5cm concurrency-hardening facets. The public
  surface we use (`AsyncConnection`, `FrankenError`, `Row`,
  `ConnectionEnv`, `raptorq_integration::*`, `Fts5Table`/`snippet`,
  `cx::Cx`, `value::SqliteValue`) is unchanged. Note: a selective
  `cargo update -p fsqlite` left seven sub-crates (btree/mvcc/pager/
  planner/vdbe/vfs/wal) on 0.3.1 because upstream's internal pins are
  caret `0.3`; the whole 20-crate family must be moved together.
- **Asupersync 0.4.5 → 0.4.9** (lockfile only; workspace floor stays
  `>=0.4.4, <0.5`). `asupersync-macros`, `franken-kernel`,
  `franken-decision`, `franken-evidence` move in lockstep; exactly one
  `asupersync` identity resolves. The fresh-process contract pin in
  `frankensearch-embed/tests/scoped_logging_contract.rs` now binds
  `asupersync@0.4.9` (it deliberately fails on silent lock drift).
- **fastembed =5.17.4 → =6.0.0.** Sole upstream break: `fastembed::Error`
  became a typed enum instead of an `anyhow::Error` alias. Every
  frankensearch consumer formats it through `Display`, so no change; the
  `ort-download-binaries-rustls-tls` feature and the `ort =2.0.0-rc.13` pin
  are identical in 6.0.0. Model-bearing tests remain a local/feature lane.
- **jsonschema 0.49 → 0.50** (quill-gauntlet, fsfs). 0.50's changes are
  confined to the `CanonicalSchema` set-operation API, which we do not use
  (`draft202012::new` + `Validator` only).
- **Transitive in-range sweep** (`cargo update`): 30 semver-compatible
  bumps (ICU 2.3 family, blake3, cc, either, rangemap, safe_arch, uuid,
  zerovec …); `arrayref` dropped. Git pins (frankentorch `ft-*`, the
  `hnsw_rs` fork rev, the tantivy 0.27.0 rev) are deliberate packaging
  boundaries and were left alone.

## 2026-08-14 — Asupersync 0.4.4 lockstep + FrankenSQLite 0.3.1 usage

**Scope:** workspace Asupersync floor is now `>=0.4.4, <0.5`. `Cargo.lock`
resolves `asupersync` / `asupersync-macros` 0.4.4 from crates.io. FrankenSQLite
stays on published `0.3.1` (latest). No public frankensearch API break.

- **Asupersync 0.4.4:** native-task abort preserves an acknowledged
  `Cancelled` result; HTTP/1 streaming cancel/reuse is additive. Download
  now uses the caller `Cx` (plus `checkpoint()` in the manifest loop)
  instead of minting `Cx::for_request()`. Watcher join already prefers the
  typed task result over a generic join-cancel, which matches the 0.4.4
  contract.
- **FrankenSQLite 0.3.1 usage:** storage open, schema bootstrap, and
  storage transactions retry the published `is_transient()` family
  (`Busy` / `BusyRecovery` / `BusySnapshot` / `DatabaseLocked` /
  `WriteConflict` / `SerializationFailure` / `PageBufferCapacityExhausted`)
  with bounded backoff. Ops open retry walks the error source chain to
  `FrankenError::is_transient()` instead of matching the word "busy" in
  display text. Catalog file-open retries the same transient family.
- **Contract:** publish-identity gate and gauntlet fuzz pin now bind
  `asupersync@0.4.4`.

## 2026-08-14 — FrankenSQLite 0.3.1 crates.io bump

**Scope:** every frankensearch `fsqlite*` edge now resolves the published
`0.3.1` registry crates. This is a lockstep bug-fix release on the 0.3 API
(autocommit durability, concurrent-open prepare, committed-freelist safety,
read-only open sidecars). No FrankenSearch call-site signature change.

- **fsfs / storage / ops** stay on `AsyncConnection` plus the `*_sync` facade.
- **durability** RaptorQ `SymbolCodec` now binds `fsqlite-core` 0.3.1.

## 2026-08-14 — FrankenSQLite 0.3.0 crates.io cutover

**Scope:** every frankensearch `fsqlite*` edge now resolves the published
`0.3.0` registry crates. The remaining 0.1.2/0.1.19 compatibility line in
fsfs and durability is gone, and the workspace git patches that split
FrankenSQLite across two revs are removed.

- **fsfs catalog** uses FrankenSQLite 0.3's `AsyncConnection` synchronous
  facade (`open_sync` / `execute_sync` / `query_sync`).
- **durability** RaptorQ `SymbolCodec` now binds `fsqlite-core` 0.3.0.
- **storage / ops** keep the 0.3.0 API they already used, but now take it
  from crates.io instead of a git rev behind the 0.3.0 tag.

## 2026-08-11 — Asupersync 0.4.3 and FrankenSQLite 0.3 migration: IN PROGRESS

**Scope:** the workspace now resolves one Asupersync runtime family at 0.4.3.
The earlier resolution barrier was removed by lifting FrankenSearch's production
FrankenSQLite edge from the synchronous 0.1 API to the asynchronous 0.3 API.

### Asupersync: `0.3.10` → `0.4.3`

- **Target provenance:** stable `v0.4.3` crates.io release and tagged upstream
  source/changelog.
- **Constraint:** `>=0.4.3, <0.5`, with `default-features = false` and the
  production `proc-macros` feature retained. Test-only crates opt into
  `test-internals` explicitly.
- **Graph result:** the root workspace, FrankenSQLite 0.3 production edge, and
  retained FrankenSQLite 0.1 compatibility edge all resolve Asupersync 0.4.3;
  `cargo tree -d` shows no second Asupersync version.
- **Contract updates:** dependency-identity assertions and the separately rooted
  gauntlet fuzz workspace now bind 0.4.3 instead of the stale 0.3.10 identity.

### FrankenSQLite / `fsqlite`: production edge `0.1.19` → `0.3.0`

- Storage operations were migrated to the async connection, statement, row,
  transaction, and FTS5 APIs.
- FrankenSearch's synchronous public storage contract is preserved through
  FrankenSQLite 0.3's `AsyncConnection` synchronous facade. The connection owns
  its worker boundary; FrankenSearch does not create an ambient runtime or mint
  test contexts in production storage code.
- Commit and rollback now execute the real FrankenSQLite transaction operations;
  a dropped or panicking transaction is rolled back before the worker accepts
  another request.
- The legacy 0.1 package remains only where the current FrankenSQLite extension
  graph still exposes that audited compatibility edge. Both package lines share
  Asupersync 0.4.3, so they no longer split the runtime universe.

### Validation

Remote workspace/profile checks, focused storage tests, fuzz-workspace compile,
dependency audit, and CI validation are recorded with the completing commit.

### Primary sources

- Asupersync [`v0.4.3` changelog](https://github.com/Dicklesworthstone/asupersync/blob/v0.4.3/CHANGELOG.md)
  and [tagged source](https://github.com/Dicklesworthstone/asupersync/tree/v0.4.3).
- FrankenSQLite [`v0.2.1` changelog](https://github.com/Dicklesworthstone/frankensqlite/blob/v0.2.1/CHANGELOG.md)
  and [tagged source](https://github.com/Dicklesworthstone/frankensqlite/tree/v0.2.1).

## 2026-06-17 — straggler third-party majors

Two leftovers from earlier passes, finished now. `Cargo.lock` is gitignored in
this repo, so only the `Cargo.toml` edits are committable; the local lock was
re-resolved for validation.

- **sha2 0.10 → 0.11 (workspace) in `frankensearch-storage`.** The workspace
  root already declared `sha2 = "0.11.0"` and every other crate used
  `sha2 = { workspace = true }`; `frankensearch-storage` was the lone straggler
  pinning `"0.10"` directly. Switched it to `{ workspace = true }`. No code
  change — `content_hash.rs` already hex-encodes with a manual `write!("{:02x}")`
  loop (not the `LowerHex`-on-digest pattern that sha2 0.11 dropped).
- **jsonschema 0.17 → 0.46 (dev-dep) in `frankensearch-fsfs`.** Migrated the
  `schema_conformance` test off the removed `JSONSchema` builder API:
  `JSONSchema::options().with_draft(Draft::Draft202012).compile(&s)` →
  `jsonschema::draft202012::new(&s)` returning a `Validator`; validation now uses
  `validator.iter_errors(&value)` (collect-all for should-pass) and
  `validator.is_valid(&value)` (for should-fail) instead of the old
  iterator-returning `validate()`.

### Validation
- `cargo check -p frankensearch-fsfs --tests`: ✅ (jsonschema 0.46 API compiles)
- `cargo test -p frankensearch-fsfs --test schema_conformance`: ✅ **118 passed,
  0 failed** — including `test_schema_fixtures_validate_against_jsonschema`.
- `cargo audit`: no vulnerabilities (3 pre-existing allowed "unmaintained" advisories).

---

**Date:** 2026-02-17  
**Project:** frankensearch  
**Language:** Rust

## Summary
- **Upgraded major core deps** to current Rust-1.85-compatible versions
- **Updated manifests** in workspace root and crate-level manifests
- **Applied API migrations** required by newer `ort`, `fastembed`, `safetensors`, `notify`, and `criterion`
- **Validated:** `cargo check --workspace`, `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`

## Direct / Workspace Dependency Upgrades

### Search / IR
- `tantivy`: `0.22.1` -> `0.25.0`

### Embeddings / Tokenization / ONNX
- `fastembed`: `4.9.1` -> `5.8.0`
- `tokenizers`: `0.21.4` -> `0.22.2`
- `safetensors`: `0.5.3` -> `0.7.0`
- `ort`: `2.0.0-rc.9` -> `2.0.0-rc.10`
- `ort-sys`: `2.0.0-rc.9` -> `2.0.0-rc.10`

### Tooling / Runtime
- `criterion`: `0.5.1` -> `0.7.0`
- `sysinfo`: `0.33.1` -> `0.36.1`
- `toml`: `0.8.23` -> `1.0.2+spec-1.1.0`
- `notify` (crate-level in `frankensearch-fsfs`): `7.0.0` -> `8.2.0`

## Code Migrations Performed

### `ort` rc10 migration (`crates/frankensearch-rerank/src/lib.rs`)
- `SessionOutputs<'_, '_>` -> `SessionOutputs<'_>`
- `Tensor::from_array((shape, slice))` -> owned arrays (`Vec`) input form
- `ort::inputs!` handling adjusted (no `map_err` on macro result)
- `try_extract_raw_tensor` -> `try_extract_tensor`
- `Session::run` mutability update (`&mut Session`)

### `fastembed` 5.8 migration (`crates/frankensearch-embed/src/fastembed_embedder.rs`)
- mutable model/session handles where embed APIs now require mutable receiver
- batch embed call adjusted to avoid unnecessary owned conversion

### `safetensors` 0.7 migration
- `serialize(&tensors, &None)` -> `serialize(&tensors, None)` in:
  - `crates/frankensearch-embed/src/auto_detect.rs`
  - `crates/frankensearch-embed/src/model2vec_embedder.rs`

### Minor compatibility / lint updates
- tensor-name discovery adjusted for current string types (`model2vec_embedder`)
- deprecated `criterion::black_box` replaced with `std::hint::black_box` in benchmark files:
  - `crates/frankensearch-durability/benches/durability_bench.rs`
  - `frankensearch/benches/search_bench.rs`

## Remaining Behind Latest (after update)
`cargo update --verbose` still reports unresolved newest versions for some crates, primarily due Rust-version constraints (`rust-version > 1.85`) or upstream selection constraints:
- Rust version constrained: `criterion 0.8.2`, `ort 2.0.0-rc.11`, `sysinfo 0.38.2`, `time 0.3.47`, `time-core 0.1.8`, `time-macros 0.2.27`, `wide 1.1.1`, `wasip2/wasip3/wit-bindgen*`
- Also unresolved at latest despite wildcarded manifest: `fastembed 5.9.0`, `generic-array 0.14.9`, `indexmap 2.13.0`, `libc 0.2.182`, `signal-hook 0.4.3`, `smallvec 2.0.0-alpha.12`

## Validation Run
- `cargo check --workspace` ✅
- `cargo fmt --check` ✅
- `cargo clippy --workspace --all-targets -- -D warnings` ✅

## Notes
- External dependency warnings from `/data/projects/fast_cmaes` are still printed during checks, but they do not fail builds for this workspace.

---

## 2026-02-18 Follow-up Update

### Summary
- Ran `cargo update --verbose` and `cargo update --verbose --ignore-rust-version`
- Updated workspace dependency constraints and lockfile to latest practical versions in this environment
- Revalidated formatting and workspace compile after upgrades

### Workspace manifest updates
- `fastembed`: `5.8.0 -> 5.9.0`
- `ort`: `2.0.0-rc.10 -> 2.0.0-rc.11`
- `ndarray`: `0.16 -> 0.17`
- `toml`: `1.0.2 -> 1.0.3`
- `criterion`: `0.7.0 -> 0.8.2`
- `time`: `0.3 -> 0.3.47`
- `sysinfo`: `0.36.1 -> 0.38.2`
- `wide`: `0.7 -> 1.1.1`

### Lockfile updates observed
- `fastembed 5.8.0 -> 5.9.0`
- `ort 2.0.0-rc.10 -> 2.0.0-rc.11`
- `ort-sys 2.0.0-rc.10 -> 2.0.0-rc.11`
- `ndarray 0.16.1 -> 0.17.2`
- `criterion 0.7.0 -> 0.8.2`
- `criterion-plot 0.6.0 -> 0.8.2`
- `sysinfo 0.36.1 -> 0.38.2`
- `time 0.3.45 -> 0.3.47`
- `time-core 0.1.7 -> 0.1.8`
- `time-macros 0.2.25 -> 0.2.27`
- `wide 1.1.1` added
- plus related transitive graph updates/removals

### Post-update validation
- `cargo fmt` ✅
- `cargo fmt --check` ✅
- `cargo check --workspace` ✅

### Clippy status
- `cargo clippy --workspace --all-targets -- -D warnings` ❌
- Current failure is concentrated in `crates/frankensearch-lexical/src/cass_compat.rs` with strict pedantic/nursery lint violations (e.g. `missing_errors_doc`, `too_many_lines`, `iter_with_drain`, `derive_partial_eq_without_eq`, etc.).
- These are style/lint policy failures, not dependency-resolution or compile failures. No rollback was applied because core compile and tests for dependent cass integration remained functional.

---

## 2026-02-19 Dependency Update

### Summary
- Bumped MSRV from `1.85` to `1.95` to match nightly toolchain and unlock all dependency updates
- Updated `fastembed` 5.9.0 → 5.11.0 (workspace Cargo.toml)
- Updated `signal-hook` 0.3 → 0.4 (frankensearch-fsfs/Cargo.toml)
- Ran `cargo update` to pull all semver-compatible transitive bumps
- No source code changes required — all API surfaces remained compatible

### Manifest changes
- `rust-version`: `1.85` → `1.95` (workspace Cargo.toml, matches nightly toolchain)
- `fastembed`: `5.9.0` → `5.11.0` (workspace Cargo.toml)
- `signal-hook`: `0.3` → `0.4` (crates/frankensearch-fsfs/Cargo.toml)

### Lockfile updates
- `fastembed` 5.9.0 → 5.11.0
- `signal-hook` 0.3.18 removed (replaced by 0.4.x)
- `bumpalo` 3.20.1 → 3.20.2
- `clap` 4.5.59 → 4.5.60
- `clap_builder` 4.5.59 → 4.5.60
- `darling` 0.21.3 → 0.23.0
- `security-framework` 3.6.0 → 3.7.0
- `security-framework-sys` 2.16.0 → 2.17.0
- `wasip2` 1.0.1 → 1.0.2

### Still behind latest (held by upstream constraints)
- `generic-array` 0.14.7 (available 0.14.9) — held by transitive dependents
- `indexmap` 2.12.1 (available 2.13.0) — held by transitive dependents
- `libc` 0.2.180 (available 0.2.182) — held by transitive dependents
- `objc2-core-foundation` 0.3.1 (available 0.3.2) — macOS-only, held by upstream
- `objc2-io-kit` 0.3.1 (available 0.3.2) — macOS-only, held by upstream

### Already at latest
serde 1.0.228, serde_json 1.0.149, thiserror 2.0.18, tracing 0.1.44,
tracing-subscriber 0.3.22, rayon 1.11.0, half 2.7.1, memmap2 0.9.10,
safetensors 0.7.0, tokenizers 0.22.2, tantivy 0.25.0, ort 2.0.0-rc.11,
ndarray 0.17.2, hnsw_rs 0.3.3, crc32fast 1.5.0, unicode-normalization 0.1.25,
dirs 6.0.0, toml 1.0.3, sha2 0.10.9, criterion 0.8.2, proptest 1.10.0,
time 0.3.47, sysinfo 0.38.2, toon-rust 0.1.3, tempfile 3.25.0,
tracing-test 0.2.6, xxhash-rust 0.8.15, notify 8.2.0, ignore 0.4.25, wide 1.1.1

### Validation
- `cargo check` (all non-TUI crates): ✅
- `cargo clippy -- -D warnings` (all non-TUI crates): ✅
- `cargo test` (2,706+ tests across 10 crates): ✅ all passing, 0 failures
- **Note:** TUI crates (`frankensearch-fsfs`, `frankensearch-tui`, `frankensearch-ops`) blocked by a pre-existing `ftui-text` nightly regression (`E0515` in `markup.rs:143`) unrelated to this dependency update
