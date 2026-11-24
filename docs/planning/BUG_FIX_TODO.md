# Open bug repair TODO

## November 24 release continuation — OrangeSparrow

Owner-requested execution checklist; Beads remains authoritative. This section
gates completion claims for `bd-dsbym` and retires from active use when this
release block is published or handed off with exact unmet conditions. It earns
no capability credit. Existing qualified work below remains unchanged.

- [x] Refresh origin, source/worktree state and the current dependency graph.
  Merge remote `bbd81cdb` without rewriting either history; the integrated
  checkpoint is `a35b1769`. Later merge source replacement commit `12b7c0d1`
  at `cea53452`; preserve both histories and the shared checkout's peer edits.
- [x] Recover the established six-variant inventory and current release via
  the GitHub connector and existing release checker; retain the actual profiles.
- [x] Inspect release-host admission. The configured Apple host answers SSH
  but DSR rejects its APFS writable roles: 6.41 GiB available versus the
  existing 40 GiB minimum. No threshold change or cleanup is authorized.
  Fresh 05:27 UTC admission remains refused at 4.79 GiB; retain that receipt.
- [ ] Upgrade one stable dependency at a time under `bd-dsbym`.
  - [x] Read the updater/research skills and primary sysinfo 0.40.4 source,
    migration guide and changelog; compare every actual call site.
  - [x] Retain the unchanged-callsite compile failure after the sysinfo bump.
    Ten actual errors; `sysinfo-baseline-red.log` in the new evidence root.
  - [x] Handle fallible sensor construction and disk inventory explicitly;
    retain unavailable load as unavailable and existing pressure semantics.
  - [x] Run affected pressure/storage/doctor tests and strict Clippy.
    Pressure 224/0/0, doctor 33/0/0; workspace, hybrid and HTTP strict
    Clippy pass. Optional fsfs Windows builds stop in ring's missing host
    tools/SDK headers, before fsfs; they are not cross-platform passes.
  - [x] Audit the lock against the current advisory database; patch the
    pre-existing cookie 0.18.2 entry to 0.18.3 for RUSTSEC-2026-0594.
    Same 1,563-advisory snapshot: one vulnerability before, zero after;
    five informational warnings remain. Only cookie changes in this step.
    The HTTP/proxy gate passes. No production exploit is demonstrated.
  - [ ] Execute the complete stock gate and actual signal/pressure controls.
    - [x] Preserve the clean `8de81f7f` stock attempt: all stages through
      Quill passed, then fsfs linking hit EDQUOT on quota-enabled `/dev/shm`.
      Stop only the owned gate tree; retain terminal 143 and linker receipts.
    - [x] Copy the cache without deletion to the separately probed volume
      with 15 TB available: rsync terminal 0, 64,572,120,354 bytes transferred.
    - [ ] Rerun the entire unchanged stock gate on the copied cache.
      - [x] Retain the completed `a35b1769` attempt, terminal 1: fourteen
        labels pass; hybrid Clippy, workspace tests, facade tests and real-model
        e2e fail. The fsfs, HTTP, quickstart and lite stages pass. No skips.
      - [x] Supply the genuine retained `2e2b8026` prior executable to the
        two-executable test; the previous full run omitted this required input.
      - [ ] Complete all eighteen labels on the repaired committed source.
      - [x] Retain the `60a07028` workspace failure: twelve Quill receipt
        fixtures hardcode quota-exhausted `/dev/shm`. The same executable's
        exact twelve cases pass on fresh private tmpfs, UID 1000 with the
        original groups, including the sixty-second production age floor.
        No source, gate, filesystem admission or host files change. Stop only
        the already-failed owned run; retain actual SIGTERM result -15.
      - [x] Rerun privately on unchanged `60a07028`: workspace tests pass
        (500 seconds), alongside all compilation/Clippy/cross/proxy/installer
        labels. Retire this incomplete run when origin advances to `12b7c0d1`;
        retain SIGTERM -15. Neither retired run is a full stock pass.
    - [ ] Recheck actual foreground signals, both 128 MiB pressure profiles,
      and cancellation while paused under an explicit 100-per-mille load ceiling.
      This configured pressure probe cannot establish default-pressure behavior.
      The first probe used 1, which the existing minimum-100 validator correctly
      refused. That receipt is not pressure proof; the corrected 13-command
      probe passes on preserved candidate ELF `bc531f43`, with no owned
      processes left and the incumbent independently searchable after each stop.
    - [ ] Recheck strict Clippy and the affected pressure/doctor tests on
      the separate `semantic-native` profile; retain its exact feature set.
  - [ ] Preserve the legacy symlink watch repair as a permanent regression.
    - [x] Reproduce historical ELF `83ff0d18`: watcher exit 2 and foreign
      cache/explain artifacts become searchable. Its source is not attested.
    - [x] Fresh known-source `2e2b8026` optimized executable passes: live
      document stays in the admitted original index, moved-to index remains
      byte-identical, explicit new alias search follows its new destination.
    - [x] Strengthen the existing real-model handoff test with two real
      destinations, alias movement before watcher initialization, unchanged
      byte inventory, and all original vector/content/shutdown assertions.
    - [x] Compile/lint the stronger test and execute its old-executable
      negative control plus current-source positive; retain every attempt.
      One test executes in each run. Historical semantic watcher exits 0
      but changes seventeen alternate-index artifacts; the unchanged byte
      assertion fails. Candidate passes all assertions. The first observer
      wrongly required the earlier lexical startup-error signature; retain
      its rejection and the reviewed actual artifact differences.
    - [ ] Qualify current-source lexical watch with sibling and nested storage;
      preserve the broader upgrade-reuse Bead and its original acceptance.
- [ ] Repair the independently reproduced timeout-control fixture before
  claiming a complete gate pass.
  - [x] Preserve clean `e591bf57` stock failures in both workspace and HTTP
    lanes: the one-second "stalled" child can already have exited successfully.
  - [x] Replay the same ELF `68f690a0` on cores 0 and 1: all eight cases pass
    there. No affinity setting becomes part of the stock gate.
  - [x] Actual scheduling control pauses only its own parent for two seconds;
    the finite child becomes a zombie with wait status 0 and the original
    test fails. Initial slower child discovery missed the interception window;
    retain that invalid control and use direct-child discovery plus wait status.
  - [x] Stop only the already-failed stock run's owned process tree, terminal
    143, before editing. Change only the timeout fixture to park until reaped;
    preserve the 50 ms monitor deadline, all assertions and other child cases.
  - [x] Recompile, lint, replay the same schedule against the blocked fixture,
    and run every default and HTTP contract case.
    - [x] Default compile and strict Clippy pass. Matched workspace build
      preserves rustc, features, dependency fingerprints, flags and profile.
      Corrected ELF `7fa91351` passes all eight cases on cores 0 and 1.
    - [x] Under the same two-second parent pause, the corrected child is
      still sleeping and the unchanged timeout/reaping assertion passes.
    - [x] HTTP-feature cohort: all eight pass, zero failures or ignores.
  - [ ] Complete the stock gate on the committed fixture checkpoint.
  - [ ] Publish the integrity split: this corrects a maintenance fixture;
    no competitive benchmark rows, WINs, LOSEs, or performance gates change.
  - [x] Record the exact dependency change, primary sources and terminal
    results in the existing UPGRADE_LOG; preserve failed attempts.
- [ ] Qualify the integrated partial-quality changes under the original gates.
  - [x] Fix four strict-Clippy errors in the included activation tests: remove
    an already inherited import and borrow the existing query without cloning.
  - [x] Match the flag-specific two-tier refusal text in the repair test;
    preserve its model-call, no-publication and exact-byte assertions.
  - [x] Reproduce the privacy-capture failure with an actual subscriber-less
    first use of the production callsite. One isolated test fails before repair.
  - [x] Keep a second capture dispatch registered to avoid the pinned
    tracing-core single-dispatch cache shortcut. The same first-use order and
    every query/content exclusion, event-presence and query-length assertion
    remain. This corrects a test fixture, not production tracing behavior.
  - [x] Run the entire core unit cohort with the repaired capture:
    1,239 pass, zero failures, one pre-existing ignore.
  - [x] Run strict core/hybrid Clippy and affected native repair/activation tests.
    Both Clippy lanes pass; the repair case and all three activation cases pass.
  - [x] Review the scoped UBS findings and explicit formatting of included files.
    Three-file scan: zero critical; six new notices are fixture assertions,
    joins/constructor expectations and necessary shared-handle clones. No
    suppressions. Explicit include-file rustfmt and workspace fmt pass.
  - [ ] Complete the unchanged full stock gate with all required inputs.
- [ ] Qualify the newly published source replacement API, preserving peer credit.
  - [x] Read both complete incoming diffs and merge `12b7c0d1` cleanly.
    Complete and sparse-quality replacements reuse admitted surviving vectors;
    empty/duplicate IDs, byte limits and cancellation precede staging.
  - [x] Run strict hybrid all-target Clippy on the merged implementation.
  - [x] Correct four formatter differences in new tests, without changing
    any assertion, refusal, inference count, storage policy or time limit.
  - [x] Execute replacement inventory positive/negative and authority tests:
    all five cases pass, zero failures or ignores, including actual native HNSW.
    Formatting passes. The two-file UBS scan flags two unchanged test-only
    panic assertions requiring whole-component activation; review retains both
    guards, all warnings and the nonzero scanner exit without suppressions.
  - [ ] Run the entire stock gate on the new clean combined checkpoint with
    the already demonstrated private tmpfs and the genuine prior executable.
    The tests use control providers for replacement behavior; they do not
    establish ordinary fsfs authority convergence or real-model replacement.
- [ ] Complete remaining dependency review without relaxing audited identity,
  forbidden-runtime, profile, wire, installer or consumer contracts.
- [ ] Recheck platform admission before any release. Keep the original
  six-variant acceptance; preserve the current release until all gates pass.
  - [x] Live source/registry audit finds thirteen public packages, all already
    published from `d1b9da15`; only `PUBLISHED_VERSION_SOURCE_MISMATCH` blocks
    the current unbumped candidate. Do not republish those versions.
  - [x] Determine Homebrew applicability from the actual complete tap tree:
    `3244f87d` has 56 files and no fsfs/frankensearch formula, cask or update
    route. The connector was rate-limited; a fresh public Git snapshot supplies
    the evidence. No existing entry needs updating; no brew install is claimed.
- [ ] Publish qualified increments and scoped Beads/TODO evidence, verify
  both remote refs and dependency cycles, and fill the bounded honesty review.

## November 23 execution — OrangeSparrow

Consumer: the owner explicitly requested a complete, granular execution TODO
after the reality check. This checklist gates my completion claims; Beads
remain the task authority. Retire this section from active use when the listed
work is completed or handed off with exact blockers. Retain it as history;
no file deletion is authorized. The observed failure it prevents is losing
pending implementation/validation behind an assessment report.

- [x] Read the anti-ceremony skill; keep the next block on product code.
- [x] Refresh source, dirty-worktree and active Beads state; preserve peer edits.
- [x] Claim `bd-fsfs-update-offline-g81fb` and reserve its runtime/test surfaces.
- [x] Locate the existing updater repair and verify its ancestry and scope.
- [x] Deliver the offline updater repair (`bd-fsfs-update-offline-g81fb`).
  - [x] Read the production guard, executable regression and native-only assertion.
  - [x] Verify retained exact-source gate and binary identities.
  - [x] Replay CLI, environment and config offline checks/application against
    baseline and repaired executables; keep all connection traces.
  - [x] Exercise positive online override and local offline rollback listing.
  - [x] Reconcile required default/native-only and unit/executable gate evidence.
    Clean `2e2b8026` stock gate: all 18 stages pass, zero failures, no model or
    cross skip allowance; terminal 01:00:58 UTC November 24. Native-only
    `semantic-native` all-target Clippy and the actual updater CLI regression
    pass. Fresh optimized default and native-only executables each pass 14
    trace-controlled invocations: all six offline source/action combinations
    against the failing old executable, explicit online override and local
    rollback. `final-updater-build-identity.json` binds source, lock, compiler,
    profiles and both executables; raw logs are `updater-{default,native}-final-probe.log`.
  - [x] Confirm the repair already landed on remote main (`474ed8bf`, `a723d161`).
  - [x] Verify remote main and the compatibility mirror both select `ea23592f`.
  - [x] Reverify original acceptance in a separate pass; close only on cited
    evidence, or retain the exact unmet conditions on the original Bead.
    Comment 5411 records the exact original matrix and all identities;
    closed at 01:10:19 UTC. The optimized default additionally passes twelve
    quickstart commands and 22 controls, zero owned processes, at
    `tmp/fsfs-quickstart.jUZjGdo1/receipt.json` under the execution root.
- [x] Qualify the search correctness improvements already integrated on remote main.
  - [x] Refresh origin and confirm the prepared chain is already present; do not
    cherry-pick duplicate implementations or claim their authorship.
  - [x] Review the prepared search chain, especially two-tier identity,
    filtered refill, failure fallback, scan policy and HNSW reuse.
  - [x] Separate actual product changes from test/census corrections; preserve
    all original guards, deadlines, ignores and negative cases.
  - [x] Run populated filter, quality-policy, staging and interaction regressions
    with the required feature profiles and exact source identity.
    - [x] Clean `ea23592f` stock workspace lane: 1,274 fusion library tests
      passed, zero failed, four ignored. Populated async-filter, late metadata,
      failure-fallback, cancellation, explicit quality policy and semantic-tail
      exclusion regressions all executed. Existing Beads have the scoped evidence.
    - [x] The complete workspace unit/integration stage passed in 676 seconds;
      this includes compilation and remains separate from the unfinished gate.
    - [x] Qualify the new partial-quality authority library/CLI regressions
      through the product `hybrid` feature profile and real-model commands.
      - [x] Actual current-source native executable: partial HNSW index, two
        real-model phases and expected winner; full-on-partial and forged
        coverage refuse; original source path unavailable during retained
        quality completion; warm server adopts the complete successor and
        adds quality scoring for its previously uncovered scoped document.
        Probe: `/data/tmp/fs-exec.SoVtXGnZ/native-product-probe-v2.log`;
        fixture `/home/ubuntu/.frankensearch-native-product-l1i55zu5`.
        Initial observer expected the wrong diagnostic wording; that failed
        receipt is retained separately. No product change was needed.
      - [x] Fix actual preflight ordering at `f5c5ba6c`: invalid quality IDs
        loaded models before refusing. Strengthened existing regression fails
        on old code (one loader call instead of zero); all six partial CLI
        cases pass after validating membership before constructors. The broad
        FSVI migration Bead remains open.
  - [x] Run appropriate workspace/lint/real-model gates on the combined source.
    - [x] Preserve the clean baseline: fmt, check, proxy Clippy and Windows
      compile guard pass; workspace/hybrid Clippy fail on `RequestFilter`
      visibility. Raw log: `/data/tmp/fs-exec.SoVtXGnZ/current-stock-gate.log`.
    - [x] Correct the private-module visibility without changing the crate-only
      re-export; commit `6684adc8` is isolated and not yet published.
    - [x] Preserve all lite assertions and retain its artifacts instead of
      deleting them; the initial baseline explicitly omits lite for this reason.
    - [x] Reverify solver-depth corrections in both interaction test crates;
      preserve the explicit Send requirements, all assertions and workloads.
      Matched baseline `6684adc8` fails strict Clippy in both unchanged crates
      with `quill,graph,rerank`; `150306ab` bounds the request future once and
      passes. No raised test recursion limits remain. Final focused runtime:
      1,305 library + 31 integration + 25 interaction-unit tests pass, four
      existing performance ignores. This adds one bounded allocation per
      request; no latency or performance improvement is claimed.
    - [x] Run the complete stock gate on the committed final candidate,
      including the previously omitted lite stage; retain all failed attempts.
  - [x] Land and publish qualified source increments; update existing Beads.
    Both remote refs select `2e2b8026`; tracker/TODO publication is a separate
    documentation/data commit. Rust sources, manifests, lock and toolchain
    remain unchanged; executed binaries and their build provenance bind `2e2b8026`.
- [ ] Resolve the next ready product gap using observed behavior and the graph.
  - [ ] Reconcile one-shot cancellation/pressure (`bd-pz8va`) with prepared repairs.
    - [x] Fresh installed default executable: successful incumbent, SIGTERM
      and SIGINT after discovery, both 128 MiB memory profiles, both actual
      host-load pause profiles cancelled while paused; exact previous selection
      and independently searchable incumbent after each. Thirteen commands,
      no owned processes remain. `foreground-product-probe-v4.log` and
      `/data/tmp/fsfs-foreground-n30xeikz/summary.json`; Bead comment 5410.
    - [ ] Existing peer-owned broader phase/repeated-interrupt/RCH acceptance;
      this scoped native pass does not close the original issue.
  - [ ] Finish useful upgrade-reuse acceptance (`bd-fsfs-cli-upgrade-reuse-rsqq7`).
  - [ ] Advance native ARM qualification (`bd-6kafg`) when an admitted host exists.
  - [ ] Advance the existing complete-store product-cost task on booked hardware.
  - [ ] Execute authority/recovery/CASS and Quill conformance work through their
    existing prerequisites, without inventing replacement frameworks.
- [ ] Qualify and deliver the next release through `bd-dsbym` and its actual
  installer/platform/crate prerequisites; do not infer release clearance from
  a focused pass or an unavailable worker.
  - [x] Close the named updater blocker on original optimized/default and
    native-only acceptance; preserve profile refusal and signing/install gates.
  - [x] Live registry audit identifies thirteen existing versions from the
    older published source. No version reuse or republishing is attempted.
  - [x] Both configured release hosts answer read-only SSH (Linux x86_64 and
    Darwin arm64). Connectivity gives no capacity/build/platform pass.
  - [ ] Refresh dependencies from primary sources, one at a time as required
    by the existing release Bead; rerun affected consumers.
  - [ ] Bump versions and all internal requirements forward after qualification.
  - [ ] Build and qualify all six established variants, signed installer and
    updater paths, platform and registry consumers; publish and verify release.
- [x] Correct the bounded active-contract drift (`bd-current-oracle-routing-docs-2fzfn`).
  - [x] Correct both architecture maps against actual native v2 routing and
    Ops' direct FrankenSQLite dependencies; keep the fsfs default explicit.
  - [x] Correct the active oracle label to 0.26.2; preserve historical version
    contracts, measured receipts and v6/v7 placeholders byte for byte.
  - [x] Update only current unmeasured sentinel identities; independently
    compare parsed manifests and confirm targets, rules and activation unchanged.
  - [x] Regenerate the existing operator plan through its compiled renderer,
    inspect the exact diff, and run version/manifest/bootstrap refusal tests.
    `b0f3023e`: 23 existing checks pass. Frozen-hash amendment retains an exact
    old-label control. Parsed manifest comparison changes only the oracle
    label; ten current unmeasured identities change only their manifest hash;
    measured and historical files stay byte-identical. Eight local links resolve.
    No activated gate, changed target or new decidable measurement results.
  - [x] Reverify original acceptance, publish, and then close on exact evidence.
    Separate fresh solo pass executes all 23 checks on `2e2b8026` without
    regeneration permission; comment 5409, closed at 01:03:33 UTC. This is
    self-verification, not independent review.
- [x] Before reporting completion: perform the skill's written work/honesty
  review, verify changed-record scope, sync Beads and inspect dependency cycles.
  Seven intended tracker records, two bounded own closures, original peer
  status/acceptance/comments preserved, zero active dependency cycles. Tracker
  additive-only mode correctly refused relation additions; monotonic reconcile
  applied the reviewed newer rows in an isolated publication DB. Stale-path
  export and missing-ancestor merge were refused and retained; a fresh export
  path succeeded without force. Only this seven-record scope is published.

Source correction: refreshing origin found `ea23592f`, already containing the
updater and prepared search chain. The local fast-forward to `163e6b55` added
no capability beyond remote main; both attempted pushes correctly refused
non-fast-forward. No force push, reset or stash occurred. Current execution is
on committed `2e2b8026` in `/data/projects/.fs-exec-fix-20261123-SoVtXGnZ`,
including both newer peer completion/activation commits through `b06d3268`.
The clean baseline checkout is retained separately at `6684adc8`.
The old 5ac check was explicitly retired and provides no compiler verdict.
The updater replay passed all four paired offline cases, online override and
local rollback, using the independently hashed retained executable. Closure
and release remain separate from that scoped result.

Current validation: the full stock gate (no stage selection or skip allowance)
ran from 00:31:52 to 01:00:58 UTC on November 24: all 18 stages pass, terminal
zero. Exact log SHA256 is `5983be7856497e10ac180c811198c135f1cdd03863efeb0ec5112ac36834b9bd`.
Source-attested installed quickstart passes 12 commands and 22 controls,
zero owned processes remaining. Earlier failed/retired attempts stay in
`/data/tmp/fs-exec.SoVtXGnZ`; retirement exit 143 is never a pass. The optimized
default build completes in 13m29s, and both updater profile matrices pass.
Source `2e2b8026` is published to main and its compatibility mirror.
The existing source gate itself authorizes native-host compilation; these are
native results, not RCH worker evidence.

Bounded real-work/honesty review (OrangeSparrow, this execution block; update
terminal outcomes before closure): purpose is reusable progressive two-tier
hybrid search plus the fsfs product. Mechanical inputs: `git log --stat
b06d3268..HEAD`, complete test/gate diffs, 12-entry reflog, five recent closed
Beads and the oldest eight open Beads. Five own commits: two build/lint
ENABLERs (`6684adc8`, `150306ab`), one USER fix (`f5c5ba6c`), one contributor
USER documentation chore (`b0f3023e`), one test-lint ENABLER (`2e2b8026`).
The integration merge preserves peer capability and earns no authorship
credit. TODO/assessment/Beads edits are PROCESS, zero capability credit.
The bounded oracle/routing and offline-updater items are closed after fresh
original acceptance. No umbrella, peer-owned or broader feature item is closed.

Work-audit answers: (1) demo invalid quality IDs with an empty model cache:
the actual executable refuses before constructor startup; the populated
partial/completion/warm probe demonstrates the positive path. Publication is
still pending. (2) Without the process edits runtime behavior is identical;
they serve the explicit owner request and completion discipline only.
(3) Both unchanged interaction consumer crates now compile with normal trait
depth, and the stock gate exercises the library; these are observed enablers.
(4) The oldest open user-facing work includes Quill conformance and complete
semantic lifecycle/restart qualification from July; this block first removed
current build failures and qualified already-wired runtime paths, without
claiming those older umbrella items complete. (5) Solo: no dispatched agents
or closure-count comparison. (6) No follow-up was minted to retire unmet
acceptance; broad migration, pressure, reuse and release items remain open.
Verdict: ENABLER-dominant, with bounded requested process work. Do not call
this a finished project; next work remains executable acceptance/publication,
not another planning framework.

Honesty inventory, original acceptance retained:
1. Yes, the initial baseline explicitly omitted lite because it deleted its
   fixture; it was never called a full pass. `6684adc8` retains artifacts and
   every assertion; current stock run includes lite. Temporary recursion-limit
   test edits were removed; both final interaction test diffs are empty.
2. No (checked: production and test diffs); the existing invalid-membership
   probe gained a constructor counter, and actual model-backed commands ran.
3. Yes, current unmeasured sentinels, rendered plan and frozen digest changed
   for the oracle-label correction. Explicit GOLDEN-CHANGE, old-hash control,
   historical byte equality and zero new decidable rows price this amendment;
   old measurements are not relabeled. No competitive win follows.
4. Yes, lite cleanup and the identity-bound test changed as disclosed above.
   No assertion, threshold, ignore, deadline or feature-default relaxation;
   native RCH bypass is labeled native. The commit guard initially refused
   missing AGENT_NAME; rerun used the reserved identity, no bypass.
5. No (checked: all own production diffs); no corpus/model/benchmark special
   case or demo success branch was added.
6. No (checked: named test counts and raw terminal logs); zero-test harnesses,
   RCH admission failures and retired runs add no pass credit.
7. No (checked: retained terminal command logs); historical receipts remain
   labeled historical. Final stock, both updater matrices and optimized
   quickstart now have observed terminal passes.
8. Yes, stale local source initially misdirected updater selection. The origin
   refresh exposed the already-landed repair, the mistake was disclosed, and
   current-source validation replaced the stale assumption. The first native
   observer's diagnostic mismatch is retained, not reported as a product bug.
   Foreground observers also initially expected unsanitized JSON detail and
   used an invalid load ceiling of 1. Both failed attempts are retained; the
   corrected observer requires typed JSON plus exact stderr and valid minimum
   ceiling100. Original signal/pressure conditions are unchanged.
9. No (checked: retained Clippy, frozen-hash and preflight failures plus all
   retired logs); none supports a full-clearance claim.
10. No (checked: cited build/probe invocations); stdout and stderr retained.
11. No (checked: the two own bounded closures cite fresh original acceptance);
    original umbrella scope and peer ownership remain unchanged.
12. No (checked: descriptions and source diff); oracle documentation corrects
    actual pins, while frozen performance targets and release conditions stay.
13. Solo: no agent or peer closures accepted in this block.
14. Solo: no subagent dispatch.
15. Solo: no subagent report substituted for execution.
16. Solo: no refusal-farming pane; refusal-only checks do not close umbrellas.
17. No (checked: evidence references); agreement or inherited claims receive
    no independent-evidence credit.
18. No (checked: no comparative performance metric reported); test counts are
    observed results, not a post-selected benchmark denominator.
19. Origin should have been refreshed first. Both refused pushes, stale
    assumption, temporary solver experiments, observer mismatch and the first
    two commit messages' missing Bead IDs are retained. Existing Bead comments
    map those commits without rewriting cited history. First source push also
    omitted AGENT_NAME and was correctly refused by my own reservation guard;
    supplying the real identity passed without bypass.
20. Strongest current proof: matched old/new unchanged consumer builds, the
    constructor-count red/green, and the actual real-model native warm probe.
    The final stock 18-stage gate, both populated updater trace matrices and
    optimized positive quickstart now pass. These are self-executed checks,
    not an independent certificate or complete-release claim.

Disposition: corrections are recorded and disclosed; countermeasures are
RH-1/RH-13/RH-14/RH-15 (no weaker gates), RH-2 (proof class), RH-3 (explicit
identity amendment), RH-4/RH-17 (no commit-count credit or hidden history),
RH-9/RH-10 (original scope stays open), SM-11 (retain first attempts).
Bounded CASS history audit used six searches and five exact context views;
hits mostly contained owner instructions and explicit schema amendments,
not independent proof of misconduct. Raw `honesty-search-*`/`honesty-view-*`
artifacts are retained under the execution root. All verification here is
self-executed; no independent verifier has certified this block.

Creation-gate verdict: LEGITIMATE GATE by the owner's explicit TODO request.
This checklist earns no capability credit. The next deliverable is the updater
behavior; an hour on another planning artifact would be worse use of the time.

Requested by the owner on 2026-09-12. Maintainer: PurpleBay.
This is a working checklist, not a substitute for Beads or evidence of closure.
Inventory baseline: `062f5b8f`; GitHub issues and JSONL read live on September 12.
Checked items mean only the specific stated step is complete. Parent issues stay
open until their acceptance criteria, including consumer execution, are met.

## September 15 active handoff (supersedes older pending probe notes)

This owner-requested checklist is process, not shipped capability. Consumer:
the owner and PurpleBay; retire it from active maintenance when this bug campaign
is reconciled against Beads (no file deletion implied).

- [x] Review committed Tantivy fixture repair `25a8d576`: copy the production
  tokenizer manager into the foreign-schema fixture before writing its retained
  document. Schema rejection and vector-preservation assertions remain unchanged.
- [ ] Execute post-fix facade validation (initial baseline `f7c2e727`).
  - [x] `hash,lexical-tantivy`: focused foreign-schema regression and full matrix.
    Strict RCH clean-overlay/no-overlay invocation admitted on `vmi1264463`
    at 17:07:28 UTC but exited 101 before compilation: missing sibling
    `fast_cmaes/Cargo.toml`. No tests ran. Dependency-aware strict RCH job
    `30022149863374855` on `am-release-css` transferred both roots and began
    compilation at 17:09 UTC. Cancelled via `rch cancel` after delayed Mail
    41961 confirmed a duplicate; terminal 143, no test verdict from this run.
    Prior peer session 71871 timed out before execution, also no test verdict.
    Peer session 2151 finished at 17:11:53 UTC: fmt, all-target Clippy, one
    focused foreign-schema test and all 144 matrix tests pass at `f7c2e727`.
    PurpleBay inspected the retained receipt and matching terminal log:
    `dependencies/facade-f7c2e727-tantivy-css-receipt.json` under the release-work
    root below; log SHA256 `df9ba5638e3235786a11db36a0e5440659223215e9b0d36a31cd564c83ad7e9e`.
    Counts are 2 + 54 + 24 + 42 + 22; two zero-test executables add no credit.
    Receipt binds 1777 project files, 36 sibling files and emitted executables.
  - [ ] `hash,lexical-tantivy,quill`: the same focused tests.
    Peer session 74592 stopped at two Clippy errors in
    `treasure_island_e2e.rs`; peer owns repair and rerun. Do not duplicate it.
  - [x] Complete remaining Tantivy-only facade integration executables after
    focused success (included in the 144 above).
  - [ ] Retain terminal result and artifact identity; report any failure without
    converting compilation, a filtered zero-test binary, or admission into a pass.
- [x] Read retained x86 scalar-candidate receipt:
  `dependencies/scalar-candidate-hz3-native-receipt.json` under
  `/data/release-work/frankensearch-release-20260912` binds native source
  `685f5e7c97201e79731cebae980dda9a13afb2753f48fbeb591416674f5e4243`
  and executed ELF `167c353683f0a90dd1d2825c0f66475c00e98296ad59d0ff4b51af91a5390066`.
  It records 48 normal tests, the live GNU exponential oracle, the F32
  certificate, and 116 trace records equal to the preceding x86 execution.
  This is retained focused evidence, not a new execution by PurpleBay.
- [ ] Qualify the committed scalar repair `5953b314` on admitted ARM capacity;
  prior paired Int8 success does not qualify this later F32 candidate.
- [ ] Execute populated CASS semantic acceptance with qualified consumer artifacts;
  hash-only or lexical fallback does not count (handoff to GreenFinch).
- [ ] Let ChartreuseCarp qualify the `engine.rs` E63 diagnostic, now committed
  as `8ff42f65`; do not retire DIV007 or alter its existing tolerance.
- [ ] Complete current-source full quality and release acceptance before closure.
- [ ] Newly found installer same-version profile mismatch (Mail 41952): the
  version-only shortcut can retain lite during a requested full install, or
  vice versa; substring matching can also confuse 1.1 with 1.10. ChartreuseCarp
  claimed the narrow repair after checking the prior owner's inactivity and
  obtaining exact reservations (Mail 41956). Require exact
  version/profile transition and incumbent-preservation tests; keep the broad
  cross-platform installer bead open.

## Execution rules and immediate queue

- [x] Read governing documentation and inspect current source/history.
- [x] Inventory all 22 nonclosed bug beads and GitHub issues 41, 43, 46, 47, 48.
- [x] Ask existing owners for current progress and bounded handoffs (Mail 41382).
- [ ] Reconcile owner replies; do not interpret silence or null assignee as permission to overwrite work.
  - ChartreuseCarp resumed release/dependency ownership (Mail 41394), owns oracle/profile/fuzz and full Quill qualification. Per Mail 41407, PurpleBay will execute the genuine-model doctor regression on the existing ovh fsfs target after the active gate; peer's old fsfs target was reaped. No independent watcher subpiece is ready.
- [x] Restore tracker mutation capability with its graph owner: migrate schema 17 to 19.
  - [x] Inspect explicit migration plan read-only and its source preconditions: eligible 17→19, integrity OK, 1298 issues/2257 dependencies/3779 comments. Prestate-bound receipt reported in Agent Mail; no apply performed.
  - [x] Coordinate a single migration writer before any apply; preserve JSONL and DB family. ChartreuseCarp delegated PurpleBay in Mail 41399; peers notified in 41401. Snapshot `.beads/recovery_20260912T194346Z`; supported migration run `20260912T194351.279401Z-476704-0` retains raw prestate under `.beads/.br_recovery/schema-migrations/` and supports undo.
  - [x] Verify ready/show, import and dependency-cycle checks after recovery: reads work, sync healthy with 1298 DB/JSONL issues and no drift, import zero changes, zero active cycles (two archived closed cycles). Doctor remains degraded for retained historical recovery artifacts; no deletion or blanket repair performed.
  - [x] Record claim bd-6kafg as PurpleBay/in_progress and doctor follow-up under bd-pz8va after recovery; flush succeeded.
- [x] Land doctor producer-admission fix `062f5b8f` (main and compatibility mirror).
- [x] Run remote formatting and 16 doctor library tests (semantic-support/no-default; all pass).
- [x] Validate doctor fix with a genuine model and stale index through the production CLI.
  - [x] Add isolated production-CLI regression `loader_only::doctor_rejects_stale_real_model_producer_revisions` in the existing quickstart suite; real index vectors, both tiers, stale-revision refusal, exact-byte restoration acceptance, observer-only assertions.
  - [x] Compile and execute the ignored genuine-model regression with both pinned caches: corrected fixture passed, one test/zero failures, 18.07s, terminal exit 0 at 20:49:10 UTC on ovh-a.
  - [x] Remote rustfmt passed for this new regression on hz4 at 19:37:30 UTC.
  - [x] Source review caught and corrected the new test's diagnostic expectation: doctor embeds the safe `SearchError` display text, not its internal variant code. The production error envelope remains `subsystem_error`; refusal/recovery/observer-only assertions are retained. Corrected-source remote formatting passed on hz4 at 19:53:34 UTC; subsequent execution results are below.
  - [x] Further source review found the raw revision-byte mutation invalidates FSVI's header CRC. Replaced it with the production writer using the genuine vectors, original shape/quantization and nonce, plus a read-only structural reopen and bitwise vector checks. The f17e0a73 run used the earlier invalid fixture; its result is retained as fixture-development evidence, not product acceptance.
  - [x] Retained old-fixture terminal result: RCH 30017802786046230 exit 101 at 20:48 UTC, zero passed/one failed in 11.01s; pinned models verified and ten-document fast+quality indexing succeeded, then raw revision mutation produced the predicted CRC refusal. This is fixture-development evidence, not a product-regression verdict. Source/executable hashes and full output retrieved under `.rch-results/doctor-producer-20260912-f17e0a73`.
  - [x] Checksum-valid fixture `041950ce` passed on ovh-a at 20:49:10 UTC; receipt directory `.rch-results/doctor-producer-20260912-041950ce` includes transferred producer manifest hash, full source checks, terminal output and executable hashes (test `75bfafd1...`, fsfs `d5f13039...`). Fast and quality stale identity refusals, exact-original restoration acceptance and observer-only vector assertions all executed. Formatting passed at 20:30:52 UTC.
- [ ] Validate default, native, lite and full workspace gates for the doctor fix.
- [ ] Reproduce ARM native refusal with the new digest diagnostic; locate first numerical divergence.
- [ ] Implement the supported numerical repair or explicitly distinct qualified producer only after evidence.
- [ ] Continue code-bearing bug work in dependency order; retain each new discovered task below.
  - Ordinary `br ready --json` now works: bd-6kafg was the only unowned ready bug and is now claimed; other ready entries are the owned installer, publication reconciliation and bd-d7xk1.

## GH #47 / bd-6kafg: native MiniLM ARM64 exact producer rejection

Owner: PurpleBay (Beads assignee); ChartreuseCarp owns the current coordinated ARM probe. Bead returned to in-progress after normal admission on September 13; native migration parent bd-2ba5 remains peer-owned.

- [x] Reproduce native Int8 refusal on ARM and success on x86 with matched source/model (existing receipts).
- [x] Preserve exact certificate gate; no tolerance or reporter-hash substitution.
- [x] Add bounded expected/observed digest, target and precision diagnostics (`1698b288`).
- [x] Verify F32 positive/negative diagnostic regression remotely (1 pass; executable hash was not retained).
- [x] Refresh private Mac RCH controller health/capacity: healthy, zero jobs, eight available slots. Preserve original dirty release clone; use a separate clean current-source clone.
- [x] Run one native certificate probe with the diagnostic patch; retain terminal output and executable hash.
  - September 13 current-source `41e72e56` ARM execution reproduced the exact `cf39f307` versus registered `bed15455` digest mismatch: exit 101, zero passed/one failed/56 filtered. Mach-O SHA256 `9b7d1b1df0897c97f18e48e359a5fa32e002bdd1545d96fd5eefab58a5211aec`; retained receipt `rch-mac/native-current-41e72e56-source-after-executable.json` under `/data/release-work/frankensearch-release-20260912`. This completes reproduction, not repair.
  - Attempt 2026-09-12 19:32 UTC: strict private RCH refused before execution (exit 103, memory_pressure_critical). No model/compile verdict; retry only after fresh host-pressure admission. Own clean clone exists at `/tmp/fsfs-release-root-20260912/frankensearch-purplebay`; peer release clone is untouched.
  - Fresh 19:39 UTC status confirms critical memory pressure 95.081%, high confidence, telemetry age 59s. Host-owner relief or another admitted ARM worker is required; no blind retries or threshold changes.
  - Read-only refresh at 20:07 UTC: memory pressure 93.798%, still critical/high confidence with 30-second telemetry. No ARM job submitted. User asked asynchronously for host-owner relief or another admitted ARM64 worker.
  - Read-only refresh at 21:05 UTC: memory pressure 98.959%, critical, fresh 59-second telemetry, zero active jobs. Bead marked blocked with exact reentry: noncritical admitted ARM RCH capacity, then the unchanged current-source certificate/intermediate probe. No repeated job submission or gate weakening.
- [ ] Bind compiler target features, model hashes and source to both platforms.
- [x] Compare tokenizer IDs and positions before numerical kernels: paired 41-record trace agrees through tokens, initial normalization, QKV and attention scores (`dependencies/native-trace-comparison.json`, same release-work root).
- [x] Compare embedding layer-normalization outputs: initial output agrees in the retained paired trace.
- [ ] Compare Q/K/V quantization bytes, scales, integer accumulations and dequantized bits.
- [ ] Compare attention softmax, GELU and first divergent encoder layer.
  - [x] Earlier trace locates scalar softmax and vector GELU differences, but they reconverge by layer-zero output; layer-one output also agrees. Layer-two output is the first persistent encoder-output difference. Only document 2 differs in the final embedding (105/384 values). Do not attribute the final refusal to the earlier erased differences.
  - [x] Committed bounded all-layer observer `eb935aa1`; retained x86 receipt `dependencies/native-observer-lintfixed-fetch-verification.json`: fmt, native Clippy, 45 normal tests and exact certificate passed; 116 records/117 files retained, executable SHA256 `b54c19dff3fba7dbad20be06269d53a9420a8555bfcd87518f26419180afc5fc`. Initial lint failure remains recorded separately.
  - [ ] Match the committed all-layer ARM trace before arithmetic changes. At 04:20 UTC September 13, fresh pressure score 81.40058 is warning, eight slots free; ChartreuseCarp owns one normal-admission probe (Mail 41588). PurpleBay submits no competing build.
- [ ] Prove which arithmetic/dispatch difference causes the first divergence.
- [ ] Implement deterministic arithmetic or an explicitly separate qualified producer; never silently relabel old vectors.
- [ ] Test repeated and batched genuine-model execution on both platforms.
- [ ] Test negative cross-producer index admission and required rebuild behavior.
- [ ] Run a populated CASS semantic query using the qualified artifacts.
- [ ] Complete linked acceptance and update GitHub/Beads with exact evidence before closure.

## GH #43 / bd-pz8va: shutdown, pressure and retained generation

Owner: ChartreuseCarp; doctor follow-up implemented by PurpleBay.

- [x] Existing one-shot cancellation/RSS sampling repair is on main (90e8fb14 lineage).
- [x] Existing bounded SIGTERM, second-SIGINT and strict/performance pressure probes are recorded.
- [x] Keep cooperative pressure threshold distinct from an OS hard memory limit.
- [ ] Preserve the previous committed generation across interruption of replacement indexing.
  - [ ] Finish composite-generation prerequisites with bd-xomn.1 owner.
  - [ ] Stage lexical, fast and quality artifacts under a new generation identity.
  - [ ] Seal/verify complete generation before publication.
  - [ ] Publish once atomically, retaining the old readable generation on cancellation/error.
  - [ ] Test interruption during discovery, loading, batches, sealing and pointer publication.
  - [ ] Test concurrent search while indexing is interrupted and then retried.
- [ ] Repeat bounded original-workload acceptance without recreating the unsafe 12.9 GiB incident.
- [x] Investigate new doctor false-healthy report (comment 5645302291).
- [x] Reuse loaded producer and unchanged search admission in doctor; no second model construction.
- [x] Test current/missing/historical revision, foreign ID, width mismatch, absent tier and read-only inspection for both tiers.
- [x] Post scoped implementation/test update on GitHub (comment 5648153122); do not close parent.
- [x] Genuine-model CLI stale-index refusal and exact-original restoration acceptance, both tiers, with read-only vector assertions (041950ce; remote one test passed). Actual full rebuild/release matrix remains under the surrounding open gates.
- [ ] Complete full-feature, native-only and explicit-lite validation.
- [ ] Complete unchanged real-model deadlines, quickstart and full quality gate.
- [ ] Publish only if separately authorized; verify reporter acceptance before closing the complete incident.

## GH #46 / bd-u8cof: Model2Vec startup allocations and tokenizer rebuild

Owner: ChartreuseCarp.

- [x] Existing streamed F32 matrix decoding avoids redundant whole-file allocation.
- [x] Existing same-process shared loader and cache-mutation rejection tests are recorded.
- [x] Correct identity/reindex guidance: equal vector certificate does not imply equal complete producer revision.
- [ ] Preserve exact bytes, artifact verification and external mutation/truncation safety in further work.
- [ ] Identify a supported persistent compiled Unigram representation (JSON serialization rebuilds the trie).
- [ ] Design cache binding to exact tokenizer/model/implementation identity, with bounded corrupt-cache refusal.
- [ ] Implement cross-process reuse without unsafe borrowing from mutable/truncatable files.
- [ ] Test cold creation, warm reopen, corrupt/truncated cache, tokenizer swap and concurrent builders.
- [ ] Run real Potion token/vector parity against the current implementation.
- [ ] Measure cold-process RSS, faults and wall time against a live baseline in the same invocation.
- [ ] Retain honest loss/no-verdict results and negative cache tests; no speed claim from correctness alone.
- [ ] Validate with the reporting ee workload and complete required gates before closure.

## GH #41 / bd-458gu: fragmented writer sessions, merge fuel, visibility

Owner: ChartreuseCarp; coordinate scorer work with GoldenMink.

- [x] Existing continuous-writer lease repair preserves safe live leases (9e3117df lineage).
- [x] Existing unchanged 192-document control proves 3 segments, IDs 0–191.
- [x] Keep hole-ratio guard: recorded naive concat expands sparse 10,672 bytes to 16,345,526 bytes.
- [ ] Implement sparse-safe merge/reencoding for repeated writer reopen sessions.
- [ ] Preserve stable document identities, tombstones, positions and exact query results.
- [ ] Repair dictionary/merge fuel accounting without granting unbounded work.
- [ ] Avoid sealing every shard solely for the one-second visibility checkpoint.
- [ ] Test 24-session reopen fragmentation, cross-shard overlaps and crash/reopen behavior.
- [ ] Validate bounded fuel errors and successful search after publication/compaction.
- [ ] Run full Quill default/all-feature conformance and unchanged negative controls.
- [ ] Retain real incumbent-relative evidence for any performance claim and close only on full acceptance.

## Other nonclosed bug beads: dependency-ordered work inventory

Each row needs source reconciliation, ownership/reservation confirmation, a minimal
reproduction, implementation where missing, focused negative/control tests, strict
RCH validation, and evidence-backed Beads closure. None is silently waived.

- [ ] bd-z2nfa — watcher-held writer excludes search. Implement through sealed composite generations, not relaxed locking; test independent-process search during watch and teardown. Depends in practice on generation publisher and bd-fsfs-identity-bound-watch-staging-t9m9m.
- [ ] bd-fsfs-identity-bound-watch-staging-t9m9m — identity-bound watch staging/publication; prerequisites bd-9xuj, bd-xomn.1. Test late writers, interrupted seals, old readers and tier identity agreement.
- [ ] bd-raw-vector-api-retirement-8sc8a — remove public raw-vector/infallible cross-tier bypasses; prerequisites bd-9xuj, bd-xomn.3. Inventory callers, migrate to authenticated typed inputs, test wrong-space rejection.
- [ ] bd-8utj — AzureCove: validated generation-aware caches and same-ID revision invalidation; prerequisites bd-07os, bd-r65a, bd-xomn. Test generation changes during active queries and failed replacement.
- [ ] bd-jbfg — AzureCove: terminal embedding-space enforcement across all entry points; prerequisites bd-4v5n, bd-8utj, bd-fsvi-readonly-semantic-inspection-qxo6, bd-lk1g, bd-r65a, bd-raw-vector-api-retirement-8sc8a, bd-remote-api-space-attestation-fcfj. Inventory every bypass and prove typed terminal refusal.
- [ ] bd-a6zt — semantic search silently degrades to hash; prerequisite bd-3fy9. Reconcile already-landed fail-closed work, test real consumer behavior and remove any remaining implicit fallback.
- [ ] bd-quill-union-horizon-exactness-salej — GoldenMink: exact TopDocs across UNION_HORIZON refills/ties. Preserve live Tantivy differential and adversarial boundary fixtures.
- [ ] bd-r1-exact-repair-or-residual-1i4j4 — scorer first-divergence repair or controlled residual; prerequisites bd-5o5z8, bd-r1-preconstruction-scorer-trace-rtnwu. Establish owner before editing; no generic tolerance substitution.
- [ ] bd-qg6-exact-case-v8-migration-vyun6 — migrate exact/case-bound V8 evidence; prerequisite bd-quill-flip-real-prose-lexical-ghhh. Remove generic V7 tolerance only with valid current witness and rejection controls.
- [ ] bd-qg2-lifecycle-quarantine-w1-rerun-l4ra4.1 — AzureCove: summed gauge bound to continuous terminal lifecycle. Require fresh live producer and missing-terminal negatives.
- [ ] bd-6xhh9 — AzureCove: prohibit watchdog daemon restart under active evidence jobs; test live leases and interruption preservation across restart routes.
- [ ] bd-6xhh9.5 — harden remaining restart surfaces; prerequisites bd-6xhh9, bd-6xhh9.3. Inventory each restart caller and classify terminal interruptions without fake passes.
- [ ] bd-quill-e8-perf-doctrine-x4e4.6 — QG-1 true indexing/tokenizer denominators; prerequisites bd-qg1-class-applicability-repair-89o4o, bd-qg1-final-multiclass-adjudication-07xik, bd-quill-e8-perf-doctrine-x4e4.15.1, bd-tqi3, bd-uh2f, bd-vmpb.
- [ ] bd-quill-e8-perf-doctrine-x4e4.7 — QG-3 real update-to-searchable visibility, not completion time; prerequisites bd-quill-e8-perf-doctrine-x4e4.15.1, bd-tqi3, bd-uh2f.
- [ ] bd-quill-e8-perf-doctrine-x4e4.8 — QG-4 durable steady-state commit latency excluding rebuild contamination; same prerequisites as QG-3.
- [ ] bd-quill-e8-perf-doctrine-x4e4.9 — QG-6 realistic queries, tail samples and absolute per-class latency; prerequisites bd-live-total-lexical-contract-gxwy, bd-qg6-exact-case-v8-migration-vyun6, bd-quill-e8-perf-doctrine-x4e4.15, bd-quill-e8-perf-doctrine-x4e4.15.1, bd-tqi3, bd-uh2f.
- [ ] bd-quill-e8-perf-doctrine-x4e4.10 — QG-9 prove cold-open and distinguish warm/page-fault costs; prerequisites bd-quill-e8-perf-doctrine-x4e4.15.1, bd-tqi3, bd-uh2f.
- [ ] bd-quill-e8-perf-doctrine-x4e4.11 — QG-10 deterministic dependency/build-footprint facts outside timing A/A; prerequisites bd-quill-e8-perf-doctrine-x4e4.15.1, bd-uh2f.

## GH #48: optional allocator enhancement (not a confirmed bug)

- [x] Classify separately from the four bug reports; do not silently add a dependency/default allocator.
- [ ] Evaluate safety/API/maintenance and actual workload evidence if owner wants this enhancement after bugs.
- [ ] If accepted, implement opt-in only, real cross-platform tests and live baseline comparison; otherwise explain disposition.

## Validation, handoff and discovered-work ledger

- [ ] Preserve exact command, source, selected test count, terminal exit and executed artifact identity where required.
- [ ] Complete `scripts/quality-gate.sh` remotely with registered models; retain failures instead of weakening gates.
  - Strict RCH job 30017802786046197 on ovh-a from transferred 062f5b8f source, started 19:33 UTC, explicitly cancelled at 19:54 UTC (terminal exit 143) to prioritize current-source doctor execution after stale contracts were confirmed. No workspace test verdict. Remote Git metadata reports dd093fb2+dirty, so this invocation is not release-source provenance. The new genuine-model regression was not part of that transferred tree.
  - Formatting, workspace/all-target check (282s), Clippy plus hybrid Clippy, and Windows index cross-check (92s) passed before cancellation during library-test compilation. External fast_cmaes emits seven deprecated-constant warnings and Windows index check emits nine dead-code warnings; no unrelated dependency edits or lint suppressions made.
  - Earlier doctor fixture f17e0a73: two-slot queue timed out without execution (exit 103 at 20:00:29 UTC); reduced Cargo parallelism to one to fit actual spare capacity, job 30017802786046230 admitted on ovh-a at 20:01:31 UTC. Same retained target (dependencies nevertheless rebuilt), finite 7200-second transport budget, unchanged model-test deadline; source/executable/output retained under `.rch-results/doctor-producer-20260912-f17e0a73`. It ended with the CRC fixture failure recorded above, superseded by the passing checksum-valid fixture. Initial receipt wrapper was blocked before execution for dynamic redirect targets; explicit new paths replaced them without disabling the guard.
  - Read-only worker checks confirm both model directories exist and all five recorded source hashes match local: runtime `38cad088...`, corrected regression `090622ee...`, root/fsfs manifests and lockfile. Actual pinned model-byte verification remains part of the test, not inferred from directory presence.
- [ ] Complete additional Quill full/probe lanes for affected changes.
  - September 13 older frozen full Quill run ended SIGKILL/137 after 483 partial passes, without a suite terminal or all-feature phase. This is NO_VERDICT; ChartreuseCarp is investigating timeout/cancellation evidence before another expensive run. Do not convert partial passes into closure.
  - ChartreuseCarp reports stale oracle v8/engine profile v6 contracts against new 0.3.0 adapters in the already-transferred gate source (Mail 41406). Their isolated five-file repair is under RCH validation; old identities remain historical. Expected contract refusal is not a doctor regression or permission to weaken conformance.
  - [x] Peer-reported focused repair validation (Mail 41421): RCH 30017802786046208 terminal exit 0 at 20:16:52 UTC; 12 oracle/version plus four profile tests pass, formatter and standalone locked fuzz-bin check pass. Peer retained executable/source hashes. This does not satisfy full Quill/full workspace gates.
- [x] Accepted bounded handoff from ChartreuseCarp (Mail 41441): strict RCH job 30017802786046288 on ovh-a completed 21:13:13 UTC, exit 0. Built both default library test executables with `cargo test -j1 --locked -p frankensearch-storage -p frankensearch-durability --lib --no-run`, then ran the exact emitted binaries from the repository root: durability 160 passed and storage 351 passed, zero failed/ignored/filtered. All 25 recorded source hashes match current files. Receipt `.rch-results/storage-durability-20260912` retains build output, full test log, terminal status and executable hashes; test-log SHA256 `b448003b2b788e2562e73472fc237043ad3855c11c540b5d58bddfd494e792c0`. Durability ELF `3bf3a74fe8e93e2c6dae77af86021bbd066cee0681624bd07398bdfefec925a5`; storage ELF `f190f27c51459f2256acaa3d5e943e6a7bca50733be91324e294c70c5f591c26`. ChartreuseCarp independently verified the receipt (Mail 41454). This predates the floor repair and is consumer qualification, not full workspace/release closure.
  - [x] Released the warm ovh target to ChartreuseCarp for coordinated remaining quality lanes (Mail 41455); no competing PurpleBay build or duplicate Quill lane.
- [ ] Reconcile actual shipped fixes with every GH/bead acceptance criterion; do not close a broad issue for one sub-fix.
- [ ] Commit only intended paths; preserve the unrelated golden `.actual.json` file.
- [ ] Keep main/mirror synchronization and coordinate concurrent remote histories without force pushes.
- [ ] Update this checklist on every new finding or completed validation step.
  - [x] September 13 reconciliation: retained clean-source `41e72e56` quickstart receipt `dependencies/css-clean-quickstart-pass-receipt.json` covers 12 commands and 18 negative controls, private verified models, and warm offline zero-network execution. Optimized default CLI E2E receipt `dependencies/css-release-e2e-receipt.json` reports eight passed/zero failed in 89.42s, with unchanged deadlines and executable SHA256 `4e84e31d5119a99c6b013dbfedab81c9b1c862b6b37df4a431b613bcebc08c7f`. Both receipts are under `/data/release-work/frankensearch-release-20260912`; they do not qualify later source automatically.
  - [x] Peer-reported fsfs suite: 2105 passed/zero failed/14 ignored after fixture repairs. Index suite: 774 passed after retained-reader overlap repair `85612f0a`. Earlier failures remain in the upgrade log; final-source full/package/platform gates remain open.
- [ ] Newly discovered: RCH source-content receipt refuses a symlink in sibling fast_cmaes `.claude/worktrees`; use a supported isolated-source route, do not delete peer files or weaken proof.
- [ ] Newly discovered: retain executable provenance before remote cleanup; old F32 diagnostic run has source/log evidence but no executable digest.
- [x] Newly discovered genuine-model doctor regression gap after 16 library passes: closed by checksum-valid production CLI regression 041950ce and the retained 20:49 UTC pass. Full workspace/release gaps remain separately open.
- [ ] Newly discovered: ChartreuseCarp reports today's registry publication of facade 0.6.0 / component 0.3.0 / rerank 0.4.0 from dd093fb2 (Mail 41404/41406); reconcile publication-owner receipts before release qualification. These immutable versions must not be reused, and they do not contain the later doctor repair. This is peer-reported registry evidence, not an independent publication receipt collected by PurpleBay.
- [x] Historical producer-fingerprint reconstruction repaired in `b6207414`: current adapter 0.3.0 is distinguished from archival adapters, retaining historical fingerprints and certificates. Receipt `dependencies/producer-tests-receipt.json` covers 404 embedder tests, eight fresh-process logging tests and three real-model tests; terminal Clippy also passed (upgrade log, 20:55:07 UTC).
- [x] Current-execution dependency assertions and exact publication guard reconciled to qualified Asupersync 0.5.0 / FSQLite 0.4.0 (`b6207414`, `19aeba11`). Guard self-test retains old-version/source/duplicate/mixed-family rejection controls; consumer execution is recorded separately above. This is not publication or full release qualification.
- [x] Potion production protocol now identifies the actual SafeTensors 0.8.0 loader (`b6207414`), with historical-preservation and same-adapter/old-protocol rejection coverage. Real Potion certificate, shared-load and streamed-decoding tests passed in the producer receipt. Earlier f17 doctor fixture predates this fix; passing 041950ce doctor execution includes manifest SHA `68d1d185...`. No blanket golden regeneration.
- [ ] Newly reproduced: anti-rollback floor publisher race, `bd-generation-antirollback-floor-ynwdi` (Mail 41444): core suite 1128 passed/one failed; a reader can observe a newly created but incomplete floor file before the winning publisher writes/fsyncs it. ChartreuseCarp owns the repair after checking the stale prior owner. Preserve genuine torn/tampered-head `UnresolvedAttempt` refusal; add deterministic paused-writer/competitor coverage and crash-releasing cross-process serialization. Do not widen the test or blindly remap all invalid heads.
  - [x] Peer-reported repair validation (Mail 41453): core 1131 passed, one subprocess helper ignored in parent; embed 404 passed, five pre-existing ignored; fmt and focused all-target Clippy passed. Both real subprocess regressions passed. Windows compile guard and commit still pending at receipt time; broad release closure remains open.
  - [x] Subsequent source inspection confirms the floor repair committed as `a4e86336`; producer identity correction is `b6207414`, and publication-guard reconciliation is `19aeba11`. These commits do not replace the remaining current-source full quality receipts.
- [ ] bd-d7xk1 (ready problem task, not bug-typed): inspect seven unreferenced source files (`core/metrics.rs`, `durability/tantivy_wrapper.rs`, `fusion/repro_blend.rs`, `fusion/repro_rrf.rs`, `tui/repro_input.rs`, `rerank/test_api.rs`, `rerank/test_inputs.rs`). Register only real supported functionality or move valid repros into actual tests; deletion requires explicit written permission. Add a meaningful source-reachability check without compiling dead code solely to satisfy a census.
