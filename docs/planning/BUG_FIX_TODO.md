# Open bug repair TODO

## November 25 fresh review of legacy compactor repair — in progress

`bd-compactor-repair-admission-40m0c`, OrangeSparrow. The previously
qualified installed `3d9adea8…` baseline exits 2 during sidecar repair
with the actual direct query holding both reader locks. A quality-only
reader also exposes premature fast WAL truncation when its valid prefix
has a torn suffix. Both fixtures retain the acknowledged ID after repair
or maintenance. Baselines are in `/data/tmp/fs-fresh-eyes-*-baseline.log`.

- [x] Re-read all three modified Rust files and their callers. Confirm the
  empty ready queue and preserve the dirty shared checkout.
- [x] Reproduce both faults against the exact installed default binary,
  record original acceptance, claim one Bead and reserve its edit surface.
- [x] Classify OS WouldBlock before RaptorQ repair writes, retry on the same
  compactor poll and preserve explicit compact and hard-error diagnostics.
- [x] Acquire all pending writer locks before any WAL-open recovery.
  Preserve normal indexed writer and read-only behavior.
- [x] Strengthen the existing populated regression with repairable vector
  damage and a torn WAL suffix. Check the fast lock with a plain descriptor
  so the test cannot trim the WAL. Keep the healthy direct case.
- [x] Prove old installed failures, candidate passage, no byte changes while
  blocked, both-tier recovery, recorded identities, output and selected stop.
- [ ] Review the final diff again, pass strict compiler checks and UBS,
  and run all eighteen stock stages on frozen exact source.
- [ ] Run fresh acceptance on the stock installed ELF, publish scoped code
  and evidence, verify both remote refs, close the Bead and release leases.

Fresh review of the test also found its stream observation could race the
release of both index locks. `review-candidate-v1-result.json` failed that
witness and does not qualify the candidate. Synchronize on both actual kernel
locks before suspending the owned query and check Initial and Refined after
resuming it. No production hooks or widened deadlines. With final harness
`2c5fec55…`, `review-baseline-v2-result.json` records the expected repair
failure on the old installed default and `review-candidate-v2-result.json`
records all three cases passing in 29.6 s on Cargo-test CLI `425fbd34…`.
Both receipts are in `/data/tmp/fs-daemon-startup.sKDROXdS/`.

The independent quality-only probe also passes on `425fbd34…`: zero
main/WAL/FEC changes under lock, both tiers compact after release, and
Refined contains the acknowledged ID. See `/data/tmp/fs-fresh-eyes-torn-wal-under-reader-candidate.log`.
Strict Rust Clippy for affected crates passed remotely on hz4 for the
implementation. The final fixture, native checks and stock gate are
tracked separately. The first native compile was superseded by the group
lock API and its own process group was stopped at exit 143.

## November 25 active direct-reader compactor progress — qualified source published

`bd-compactor-direct-reader-progress-pup0v`, OrangeSparrow. Actual installed
default `12762434…` exits 2 when a real direct progressive search retains
both kernel reader locks beyond two seconds. Pending artifacts stay unchanged;
Refined succeeds after resuming the query, but maintenance needs a restart.
Baseline and same-input recovery control:
`/home/ubuntu/.fs-active-reader-tdnwa8aw/reader-held-observation.json` and
`reader-released-control.json`. Original acceptance is recorded together in
the Bead before implementation. Shared dirty checkout and peer tasks stay intact.

- [x] Refresh graph/ref/ownership state; reproduce with the installed CLI,
  actual pinned producers, acknowledged payload and kernel reader ownership.
- [x] Claim one bug with complete positive/adverse criteria and reserve the
  precise index/runtime/test/TODO surface under the graph structure lease.
- [x] Distinguish OS WouldBlock from other lock/I/O/validation errors without
  changing existing writer/reader admission or adding unsafe sites.
- [x] Defer only contended maintenance on the normal daemon poll; release
  partial tier writers and publication exclusion, retain pending baselines,
  preserve strict two-tier WAL/FEC preflight and responsive selected stop.
- [x] Add the named real CLI regression: actual suspended direct query,
  unchanged pending bytes beyond the old budget, same live daemon, resume to
  Refined, automatic both-tier merge with every acknowledged ID/producer.
- [x] Exercise quality-only contention after a successful fast writer open;
  prove the earlier writer and publication lease are released between polls.
- [x] Check absent/corrupt artifacts and non-contention lock errors stay hard
  failures; retain the original seven startup and idle-writer controls.
- [x] Execute the named regression baseline-red/candidate-green, focused
  compiler/lint/unit checks, review/UBS and diff checks with retained receipts.
- [x] Freeze clean source and pass all eighteen unchanged stock gate labels;
  record exact source/lock/toolchain/model/installed ELF and execution class.
- [x] Perform fresh solo original acceptance on the stock installed ELF.
- [x] Verify published qualified source on both refs, close this Bead on
  cited evidence, and export the closed status.

The live candidate probe keeps both main/WAL/FEC sets byte-identical while
the actual direct reader holds both kernel locks beyond the old budget.
The reader emits Refined after resuming and the same daemon merges both tiers.
Raw receipts are in `/home/ubuntu/.fs-active-reader-o4jfqf9m/`.
The checked-in named test additionally passes quality-only contention with
an independent fast writer and publication lease acquired between polls.
Candidate `35f04799…` passes in 20.60 s. Previously installed `12762434…`
fails at the asserted compactor exit in 11.79 s with the same harness
`b70a5f64…`. Full receipts are `reader-candidate-v2-result.json` and
`reader-baseline-v2-result.json` in `/data/tmp/fs-daemon-startup.sKDROXdS/`.
The initial harness failed earlier because the expected stored MiniLM ID
used the model directory's casing. Preserve `reader-baseline-v1-result.json`
as a fixture failure outside the baseline regression evidence. Initial
compiler refusals and error diagnostics remain in the focused build logs.
Strict remote Clippy found a needless move of the OS lock error. The stock
run on `ae93dcb0` stopped at the formatting stage before the correction.
`stock-gate-active-reader-v1-result.json` records exit -15 and the partial
denominator. The correction borrows the error and leaves the OS classification
and reported error unchanged. Strict native workspace Clippy with all targets
passes on the corrected source in 87 s (`fs-active-reader-native-clippy-v2.log`).
UBS exits 0 with zero critical findings on all changed Rust files, including
the corrected index file.

All eighteen unchanged stock stages passed on clean
`b32b9ae8c10d5a309c74ece4a7b830a587c501fd` in 3,977.01 s. The native
thinkstation1 run used UID 1000, a private 4 GiB tmpfs, and the pinned
nightly. See `stock-gate-active-reader-v2-result.json` and
`reader-stock-source-toolchain-b32b9ae8.txt` in the evidence directory above.
The fsfs stage includes 3,368 library tests and 120 schema tests passed.
The end-to-end stage explicitly ran the real-model reader, writer, and seven
startup scenarios. The Quill result covers the bounded stock witness.

The installed default build has SHA256
`3d9adea85b5b9d1ac169c29179087c47837f02c2762b975d528d9aabf3e32efe`.
`reader-quickstart-b32b9ae8-receipt.json` records 12 commands, 22 controls
and zero remaining owned processes. Fresh solo original acceptance on that
binary passed in 21.60 s, including actual suspended direct-reader and
quality-only contention. `reader-installed-acceptance-v2-result.json`
binds the clean source, installed executable, harness, and log. Selected
stop preserved both live query locks and pending bytes, then the resumed
query returned Refined. A write-permission refusal preserved pending bytes
and restoring permissions allowed both tiers to compact. These controls
are in `/data/tmp/fs-active-reader-installed-stop.log` and
`/data/tmp/fs-active-reader-installed-denied-write.log`.

Release qualification remains on `bd-dsbym`. At 14:51 UTC DSR found `mmini`
reachable with 21.74 GiB against its 40 GiB floor, `wlap` unreachable, and
`trj` healthy. `/data/tmp/fs-reader-release-hosts-20261125.json` preserves
the admission results. The maintained tap at `61378995` has no fsfs formula.

Both remote refs were verified at `ee23dd924b5a8efddb6c9786de8d7e9e773d7d0c`
before closing `bd-compactor-direct-reader-progress-pup0v` at 16:07:55 UTC.
The original acceptance text and all unrelated published rows are identical.
The final ref and lease release receipt is
`/data/tmp/fs-daemon-startup.sKDROXdS/reader-final-publication-result.json`.

## November 25 idle compactor writer progress — qualified source published

`bd-idle-compactor-writer-progress-u5mm1`, OrangeSparrow. Existing ready
queue is empty; blocked publisher migration and peer-owned work remain intact.
Actual qualified default ELF `7aaad237…` reproduced append exit 1 while the
same idle compactor was alive; index creation and selected daemon stop exit 0.
Artifacts: `/home/ubuntu/.fs-compactor-idle-nx56boh1/observation.json` and
`/data/tmp/fs-compactor-idle-repro-20261125-v2.log`. Original installed
quickstart directory was absent; the exact executable was recovered from the
Cargo cache and retained separately, as recorded in
`/data/tmp/fs-daemon-startup.sKDROXdS/default-elf-recovery-20261125.json`.

- [x] Recheck source/ref/graph state and preserve the shared dirty checkout.
- [x] Reproduce actual ordinary writer refusal with a live idle daemon.
- [x] Record the complete original positive and adverse acceptance together
  in one bug; claim it and reserve the precise source/test surface.
- [x] Separate the lifetime compactor lease from the maintenance publication
  lease using the existing native lock/fence implementation on every platform.
- [x] Retry publication contention on the normal poll cadence; retain pending
  work, recheck both tiers under exclusion, and capture the reset baseline
  before unlock so a later acknowledged append is not consumed unmerged.
- [x] Keep correct process ownership for stop and live-compactor uninstall
  exclusion; retain sealed/fixed-authority admission on both lease kinds.
- [x] Extend real process-owner controls for compactor versus writer-only
  identity, second-compactor exclusion, and uninstall refusal.
- [x] Add and execute a real Potion/MiniLM long-lived daemon regression:
  repeated ordinary appends, both tiers compacted, all IDs/identities retained,
  Refined search, deferred maintenance behind an independent publisher, and
  selected stop while another writer holds publication exclusion.
- [x] Execute the new regression against the retained old executable first;
  retain its intended writer-progress failure.
- [x] Complete focused compile/lint, ownership/admission tests and both old
  seven-scenario startup and new writer-progress regressions.
- [x] Freeze clean source and complete all eighteen unchanged stock gate
  labels. Preserve failures, original ignores, exact source/ELF/model identity
  and native versus remote execution class; do not weaken acceptance.
- [x] Review the final production/test changes, run UBS and diff checks,
  then perform fresh solo original acceptance against the frozen revision.
- [x] Publish only qualified source and this scoped Bead row; verify both
  remote refs and preserve every unrelated published tracker row.
- [x] Close only with cited published evidence and export the closed status;
  keep release/migration limits explicit. Post-commit ref verification and
  release of the exact seven owned leases are recorded in
  `idle-final-publication-result.json` after the final status push.

Initial focused controls: 25/0/0 native tests; preserved log
`/data/tmp/fs-idle-compactor-focused-unit-v1.log`. Remote compiler check
passed an initial snapshot; strict Clippy correctly refused a redundant
`pub(crate)` constant in the private kernel module. Move that constant to
the public admission module; no lint relaxation. The new real-model harness
fails the old executable at the first ordinary append, as intended:
`idle-writer-baseline-v1-result.json`, exit 101, log SHA256
`15d2ce53e36dc8af7a04638e21a13d5231f6349a612cbb80855ad00d5d15b52e`.
The first candidate command mistakenly copied the old cached ELF again;
its retained SHA proves that mistake and it earns no candidate credit.
The actual freshly built Cargo-test CLI ELF `01310494…` accepts all three CLI
appends, merges both tiers and retries the observed contended publication.
Its initial harness then correctly refuses the deliberately new vector-only
ID because that ID has no source-window manifest membership. Correct this
new test's contention stimulus to replay an existing row's actual vector and
ID. It still requires observed deferral, pending WAL preservation, eventual
both-tier merging, all acknowledged IDs, exact producer identities, Refined
search and selected stop. No production manifest check or original criterion
is changed. Preserve both failed candidate receipts; full qualification and
fresh solo acceptance remain pending.

UBS scans all four changed Rust files: 0 critical; existing/test-heavy
warnings are triaged against the diff. New production findings are bounded
array indices from the same two-element shape, two internal lock names
supplied only by fixed constants, existing CLI output and tracing formatting.
New test unwrap/assert/JSON findings are test assertions and retained-fixture
I/O; none is silently suppressed. UBS does not replace compilation or tests.

Corrected long-lived regression PASS in 14.69s:
`idle-writer-candidate-v3-result.json`; Cargo-test CLI ELF SHA256
`605fcb8768c61bb296f9dc077b0e30e6172926ba76f2c47b084d267429ba7ddc`,
harness `445e00fd4b17f98bd1d35d7bab625de5df35868921c3e5b298eff793efcd9d77`,
log `6ef6c0e912147bb3b7b9214a6c9d5398f6dea6bb2ce1eabbb086e34f8b652b0c`.
All three actual CLI appends succeed with the same daemon PID. Independent
publisher contention is observed, the original pending WALs remain, both
merge after release, all acknowledged IDs/producers and Refined hits remain,
and stop selects only the compactor while the writer remains fenced.
All seven original startup scenarios also pass in 72.57s on the initial
repaired executable: `idle-startup-regression-v1-result.json`. Remote strict
all-target fsfs Clippy passes on hz4 after the visibility correction;
`/data/tmp/fs-idle-compactor-rch-clippy-v2.log`. Final staged UBS v3 exits 0,
four Rust files, 0 critical, 10,470 warnings and 2,900 info; broad unchanged
source and test findings are not hidden. Full frozen-source stock qualification
and fresh solo acceptance remain open below.

Fresh review found the first contention fixture insufficient: replaying the
same ID/vector has no observable payload effect, and accepting any earlier
deferral log could mistake a prior append's contention for the named case.
The 14.69s pass therefore does not qualify contention acceptance. Retire the
first stock run before source edits: verified own process group 3860986 was
terminated with SIGTERM and had no remaining processes. Preserve
`stock-gate-idle-compactor-v1-result.json`; terminal -15, only the six compiler
labels completed, no full-gate credit. Strengthen the existing test instead:
stop only its owned daemon, observe the actual stopped-child event, append a
new document through the real CLI, hold an independent publication lease,
resume the daemon and require one NEW deferral after the prior log count.
Both pending WALs must still contain that newly acknowledged ID under
contention; after release it must appear in both compacted mains and Refined
results. Four CLI appends are now the predeclared denominator. No new process
hooks, fake vectors, production changes or relaxed assertions/deadlines.

The stronger named test passes in 16.76s (four actual CLI appends):
`idle-writer-candidate-v4-result.json`; matching baseline with the same new
harness fails at the first ordinary append, `idle-writer-baseline-v2-result.json`.
The existing fixture child guard now signals only when `Child::try_wait`
reports `Ok(None)`: a stop observer may already have reaped a terminated
child, so an error must not become permission to signal its old PID. The
new stopped-state observation remains safe Rust using the existing rustix
dependency; no new runtime hook, dependency or unsafe site.
An additional actual baseline same-input control confirms append exit 1 with
the live old daemon, then exit 0 after selected stop:
`/home/ubuntu/.fs-compactor-idle-c73r0qw8/post-stop-control.json`, transcript
`/data/tmp/fs-compactor-idle-repro-20261125-v3.log`. Staged UBS v5 exits 0
(one changed test file, 0 critical, 2,125 warning, 327 info), triaged as test
assertions/fixture I/O and unchanged code. Production source remains exactly
the implementation in `216575fc`; freeze the stronger fixture and rerun the
entire unchanged stock gate before original acceptance and publication.

Final functional and stock acceptance is complete on clean unchanged
`b39f432dd5982a98467294793af359b1d542dda8`. Every one of the exact eighteen
stock labels passes, terminal 0, 11:10:51–12:11:53 UTC, 3,662.16s;
`stock-gate-idle-compactor-v2-result.json` binds log SHA256
`89392bd000db7490ebbd60a52de3093f5a74cc9d2f77e7ecf54bcc7dc28c3792`.
Execution is native thinkstation1, UID1000, original groups, private 4 GiB
tmpfs; remote compiler success is recorded separately. No label skips, skip
allowances, gate edits, lint relaxations, golden regeneration or widened
deadlines. The library fsfs lane passes 3,368/0/16 (original unit ignores
unchanged), schema 120/0/0; the newly declared real-model test executes
explicitly in the unchanged stock e2e lane. Cargo.lock remains `ad2c99a0…`,
the stock script `db7fae7f…`, nightly `1.100.0 (908501772)`, and the clean
unpublished tool dependency fast_cmaes stays at `18b39102`.

Final fresh acceptance uses the exact installed STOCK DEFAULT executable,
not merely the earlier Cargo-test feature closure. Quickstart receipt
`fsfs-quickstart.X5N42TUb/receipt.json` binds checker/source `b39f432d`,
default profile, invocation-build origin, 12 commands, 22 controls and no
remaining owned processes. Retained `idle-installed-b39f432d` has SHA256
`127624341d1396d0e16bbc1b5b7bef17d8ec111c22d12b0db3a3a00362d5659b`.
`idle-installed-writer-result.json` passes all four real CLI appends,
same-daemon continuity, a fresh contention event, the new acknowledged ID in
both pending WALs and then both mains and Refined results, exact producer
identity, duplicate exclusion and selected stop, 17.83s; log SHA256
`184cc873988c36dce51a77db4f6c7dd206d07c55e9697fec21960aab982fffeb`.
`idle-installed-startup-result.json` passes all seven original cases in
76.91s; log SHA256
`f421d10d1cb06894ab3e480e6bfb79ce6e378e1760195008b818e46991ced8e0`.
Both use final harness SHA256
`a3fb245bcf7c0453b71a8b64a43430739ff896f6ba8091a6712b753ded4fc176`.
The same harness's intended baseline failure is in
`idle-final-writer-baseline-result.json`, log `2e1af6db…`. All eight installed
fixtures' actual index-command JSON records Potion 256 and MiniLM 384 with
`is_hash_control: false`; both identities survive compaction. The original
Bead description remains byte-identical. Joined evidence is
`idle-original-acceptance-result.json`; source publication remains required
before closure. Earlier 605fcb87 CLI checks are separate subjects, not the
installed-default proof. All referenced run receipts live under
`/data/tmp/fs-daemon-startup.sKDROXdS` unless otherwise stated.

Qualified code and its in-progress Bead evidence were published and both
remote refs verified at `ccf5d50f666c5f569e2425587c2e8c17a09265fd`,
12:24:03 UTC, before closure. `idle-source-publication-result.json` binds that
ordering and confirms the production/test/gate/lock body is unchanged from
qualified `b39f432d`. Only then was the bug closed at 12:24:26 UTC on its
original description, and closed status exported. Initial source publication
preserves all 1,491 unrelated published rows. The final status export also
updates the owned release Bead's two scoped comments (5454 and 5460), retaining
its original description and in-progress state; all 1,490 other rows remain
byte-identical. Active dependency cycles are zero and no ready work remains.
This block records a completed product repair. After the final status push,
`idle-final-publication-result.json` records the actual final refs and exact
reservation release; this document does not pretend to contain its own Git
commit ID or a post-commit event before it happens. The dated block retires
from active work after that recorded confirmation; broader Beads stay open.

Release remains open. Fresh unchanged DSR Mac admission at 12:10:37 UTC
refuses 22.20 GiB free against the configured two-slot 40 GiB floor:
`/data/tmp/fs-idle-compactor-macos-admission-v2.json`, SHA256
`a491cf90b108cd33ecd6651f23d0a86fdc4a0d3454dd8b34db8347b105d9f0ad`.
No capacity setting, version or tag changed. Six binary variants, forward
crate/consumer qualification, full Quill campaigns and the separately held
publisher/watch migration are not qualified by this source repair.

End-of-block real-work audit (10:16 UTC through qualified publication):
purpose from README: reusable two-tier hybrid search and its fsfs product.
Actual commits inspected: `216575fc` USER (writers work with an idle daemon),
`b39f432d` ENABLER (non-tautological contention proof); evidence/status edits
are PROCESS and earn zero capability credit; UNKNOWN 0. The two-minute demo
is the same-input old-daemon refusal/stopped success, then four appends with
the repaired daemon alive and all IDs in Refined results. Removing the
process notes would not change runtime behavior; they serve the requested
TODO and honest closure. The enabler is exercised by the actual installed
CLI, not speculative infrastructure. Oldest user-facing open work includes
February's native ONNX replacement; it is peer-owned, while release is
externally refused and the fixed publisher explicitly held. No ready work
was displaced, no subagents were spawned, and no original criterion was
split into a follow-up or edited to manufacture completion. Verdict HEALTHY;
correction: retain the stronger new-ID control and keep broader work open.

Honesty inventory, this same bounded block (numbers follow the invoked
skill's worksheet):
1. No (checked both commit test diffs): no existing test removed or newly
   ignored; the new pinned-model case is explicitly executed, including the
   stock e2e lane. Original fsfs library's sixteen ignores remain unchanged.
2. Yes: initial identical-vector replay was inadequate positive proof.
   Removed in `b39f432d`; real CLI new-ID append replaces it. Earlier pass
   withdrawn in Bead comment 5456 and the operator update.
3. No (checked changed paths/diffs): no golden or snapshot regenerated.
4. Yes: production and regression tests change together; existing stop
   controls follow the actual new compactor role, retain their state/signal
   assertions and add independent writer exclusion. No gate, CI, validator,
   suppression, bypass flag or tolerance changed; failed controls retained.
5. No (checked production diff): no fixture ID, benchmark path or test-only
   success branch. Publication contention remains real kernel exclusion.
6. No (checked receipts): each named model run executes one test, the stock
   lane executes the new case, and exactly eighteen distinct labels pass.
7. No (checked terminal outcomes and source/ELF receipts): no unrun command
   is reported as executed, and incomplete gate v1 earns no full-gate credit.
8. Yes: the early replay/stale-log witness could overstate contention proof.
   Replaced by a newly acknowledged real ID and a new log event after the
   observed stopped child resumes; final installed readback proves its effect.
9. Yes: first candidate copied the baseline cache by mistake; later weak
   contention success was premature. Both were disclosed and corrected;
   old failed/withdrawn receipts are preserved, not silently replaced.
10. No (checked runner redirections): all cited stderr is retained in combined
    or separate logs; no evidence command silences it.
11. No (checked statuses/publication timestamps): this bug closed only after
    both refs held the qualified source. No release or peer task is closed.
12. No (checked creation description and unrelated rows): original acceptance
    is byte-identical; no requirement or plan was rewritten as implementation.
13–17. N/A, genuinely solo: no agent/pane delegation, peer certification or
    agreement used as evidence. Remote compiler jobs are not reviewers.
18. Expanded the fixture from three CLI appends plus synthetic replay to four
    real appends before running the stronger version; the old result is
    withdrawn, not silently assigned the new denominator. Countermetrics are
    acknowledged IDs, both tiers, producer identity and unchanged adverse
    cases. No performance win is claimed.
Older-session check: six bounded cass queries, including this test's name;
followed each first hit by exact archive ordinal. Tool titles and earlier
partial work are not treated as historical certification.
19. Explain the mistaken ELF and weak fixture before review: done in operator
    updates, this history and Bead comments 5456/5459. No false-close remains.
20. Strongest evidence: intended original-stock failure, same-input stopped
    control, exact installed-stock four-append/new-ID readback and all seven
    original cases, joined to all eighteen labels on clean unchanged b39f432d.
Disposition: corrected in place, disclosed, and encoded as RH-2 (bind actual
ELF/origin/proof class) and RH-5 (observable new payload and fresh contention
witness). RH-1 control: exact stock script/labels, old assertions and all
original acceptance are retained. This is solo verification, not independent
certification. No unresolved test/gate relaxation is hidden.

Creation worksheet (requested granular TODO and scoped evidence): runtime
does not branch on this block. Consumer: operator requesting completion of
the idle-compactor repair. Gate: cannot claim that repair complete without
the positive writer/compactor proof and unchanged stock gates. Observed
defect: the live append refusal above. Retirement: after qualified publication
and original acceptance, this becomes history. No integrity exception is
claimed. Highest ready capability: this reproduced ordinary-write failure;
an hour of implementation exceeds an hour of report work. Verdict:
LEGITIMATE GATE, minimum updates in the existing TODO and Bead only.

## November 25 daemon repairs — qualified source published

Owner-requested execution checklist for `bd-legacy-daemon-startup-wal-420fg`.
The overdue-turn repair is `bd-daemon-overdue-tick-cooperation-z3qax`; its code
and native regression coverage remain in the same Bead. Qualified source was
published and verified on both remote refs at `cc00e0d1`. Only then were the
fairness and startup bugs closed, in that order, on their original criteria.
This dated block is retired from active product work; preserve its history.
Beads remains the task authority. Use
`/data/tmp/fs-daemon-startup.sKDROXdS/final-qualified-publication-result.json`
for post-commit status publication, ref checks and exact reservation releases.

Final qualified code: `e8043378ad24db7d892e32f87c8bb2182e86e860`.
All eighteen unchanged stock labels PASS, terminal 0, no label skips,
08:54:40–09:55:39 UTC, 3,659.25 seconds. `stock-gate-v9-result.json` binds
the clean unchanged source and log SHA256
`45f5c19e525ae30987e2fe7b06d845ce34343db0fd71b94e030681622f97ec68`.
Execution is native thinkstation1, original UID/groups, private 4 GiB tmpfs;
remote admission refusals earn no test credit. The lockfile and stock script
are unchanged, and `fast_cmaes` remains clean at `18b39102`. Results include
fsfs 3,366/0/16, schema/conformance 120/0/0, facade library 618/0/0 and native
CLI 407/0/0. The original sixteen ignored fsfs tests are unchanged; the named
real-model startup probe is explicitly executed. Same-source optional graph
strict Clippy and 26/0/0 regressions pass; `graph-feature-tests-v4-result.json`
records log SHA256 `b2159bffdfa83fb5ccea0aa459cf4805711b4f3678103d696585f08538472559`.

Fresh solo original acceptance is complete. `final-v9-fairness-result.json`
binds three intended old-ELF failures (exit 101) and three repaired passes
(exit 0), retaining the exact count/drop/socket/shutdown oracles. Final unit
ELF SHA256: `3661ccde714173820dcb79408bf647b98d86ca7a4e273d03d0a1ea0990773f22`.
`final-v9-acceptance-result.json` binds the real-model startup baseline's
main-count failure and all seven installed-default scenarios passing in
77.09 seconds. Installed default ELF SHA256:
`7aaad237a1241ffd422f2e5bf93c56d20dcfa9bb14cc5b6b131d74709a297a02`;
harness SHA256: `2166ceb8ee1c394b1cc4bbd26415d8ef7b1475b59a7eeb8b0bc88edc0eaf0c56`.
Both original Bead descriptions remain byte-identical to their creation files.

`post-v9-product-checks-result.json` completes at terminal 0 with clean
unchanged source. The retained normal native CLI ELF is
`/data/tmp/fs-daemon-startup.sKDROXdS/native-cli-e8043378`, SHA256
`82ec438f6c8e8f0fd008981fe4cf52b3a4c0aec01bd66a5b4e791bbff1613f24`.
`native-preview-product-v2-result.json` records all twelve predeclared actual
Potion/MiniLM command outcomes, including both proper refusals, in 104.97s.
`warm-coverage-product-v1-result.json` records fourteen actual-model
observations in 74.85s: exact promotes partial coverage to full; HNSW retires
quality; denied requests preserve authority/lock and create no successor;
same-stream and fresh-process queries use the committed default and IDs.
Artifact roots: `/home/ubuntu/.native-preview-s4kkncyo` and
`/home/ubuntu/.native-warm-coverage-rfdjyzl6`. These are solo functional checks,
not independent certification, performance wins, full Quill campaigns or
binary/crate release qualification. Source publication is verified in
`initial-qualified-publication-result.json`: both refs at
`cc00e0d1297673963613a3d5546eb958ed0817dd`, product code byte-identical to the
qualified revision. Fairness closed at 10:08:35 UTC, startup at 10:08:36 UTC;
their original descriptions remain unchanged. Broader Beads remain open.

- [x] Refresh current source and governing instructions; preserve the dirty
  shared checkout and fast-forward the clean isolated checkout to `b27feb94`.
- [x] Triage the existing graph: no ready work; preserve blocked migration and
  peer performance acceptance. Claim one concrete ordinary-product bug under
  the exclusive graph lease, with its code and named real-model test together.
- [x] Inspect IcySnow's preserved unqualified candidate; credit its design and
  leave that worktree untouched. Confirm the old counter defect at `b27feb94`.
- [x] Retain a matched baseline executable and reproduce the named populated
  daemon-startup test failing because both pending WALs remain unmerged.
- [x] Finish the production fix: process existing work on the first poll,
  preflight both vector sidecars, retain later growth/stop/cancellation behavior,
  and report failed maintenance honestly.
- [x] Verify both actual model tiers merge their WALs, preserve identities and
  Refined search, and avoid rewriting on the second idle startup.
- [x] Verify repairable corruption and separate corrupt-fast/corrupt-quality
  sidecar refusals before any merge; preserve vector/WAL bytes. Exercise the
  existing healthy unprotected policy with preserved sidecars outside the root.
- [x] Compile/lint the integrated source and execute the named test red/green;
  triage the exact changed files with UBS and check formatting.
- [x] Freeze a clean source checkpoint and finish all eighteen unchanged stock
  labels with real models and the genuine prior executable. Keep execution
  classes, all failed attempts and source/lock/ELF/model identities explicit.
  - [x] Preserve the first clean `b692f5d4` run: fmt/check/clippy-proxy pass;
    strict workspace and hybrid Clippy fail on upstream code. Retire the owned
    process group before editing; terminal -15, no full-gate credit.
  - [x] Repair the upstream integration blockers without changing assertions,
    lints or gates: two unit-valued mutation statements need semicolons; an HNSW
    test needs its explicit default type; receipt values need Clone; native CLI
    preview checks need the authenticated policies and bounded source export.
  - [x] Re-execute the existing populated partial-quality migration, completion,
    source-recovery and transition cases; freeze and rerun all eighteen labels.
  - [x] Preserve and correct the warm-completion test's wrong API invocation:
    raw `ServingEmbedder::embed` deliberately bypasses caching, while production
    queries use `embed_bound`. The isolated old test fails with the injected
    inference prohibition. Exercise the bound path without changing faults,
    call-count/cache identity/scope/membership assertions or production caching.
  - [x] Preserve the second isolated failure: the forged receipt fixture writes
    its input to the output path it later asserts absent. Separate the input
    path, require the exact document-count refusal, and verify a matched valid
    receipt reaches the injected loader stop without staging work or output.

Additional scanner triage: the nine integration-repair files initially return
UBS exit 1, with three critical `panic!` findings in the existing
`model_migration_tests.rs` fixture. All three are unchanged intentional failure
sentinels under `model_migration.rs`'s test-only module: raw inference, loading
fast during completion, or partial loading during whole-model migration must
fail. Three local explanatory UBS annotations now classify these exact test
sentinels; their panic bodies and the strict guards remain intact. The full
unfiltered prior result is retained. The annotated full-file scan exits 0 with
0 critical, 1,361 warnings and 405 informational findings. No production path,
scanner rule, stock label or test assertion is suppressed. This admits exactly
three intentional test sentinels for scanner triage, with no performance claim.
The separate unannotated production-panic control still returns exit 1 with
one critical finding; annotations do not disable that rule or alter panic
behavior. After both test-driver corrections the final integration scan covers
ten files: exit 0, 0 critical, 1,867 warnings, 455 informational findings. All
45 findings near changed lines were reviewed: deliberate test failures and
fixed fixture values, path-string false positives, bounded retained `Arc`
clones, existing constant-size casts, and the unchanged inherited retrieval
default. No whole-project scanner clearance is claimed.

The native CLI's first actual suite run is retained: 400 pass, two fail, zero
ignored, 360.35 seconds. The two failures reproduce separately in fresh private
tmpfs namespaces. Corrected warm cache retention passes in 92.42 seconds across
all four original topology/retrieval combinations; corrected source-completion
preflight passes in 32.58 seconds, including the exact refusal and matched
valid-input control. Both affected core migration tests pass. Workspace strict
Clippy and final hybrid strict Clippy pass; the complete stock run passes on
the qualified source below. Neither correction changes production caching, default tests,
model mocks, pass thresholds, or time budgets. The quality script, lockfile and
startup production/test implementation are byte-identical to `b692f5d4`.
- [x] Reverify original acceptance in a fresh solo pass and write the bounded
  work/honesty review. This is solo verification, not independent certification.
- [x] Exercise the newly connected preview API with twelve predeclared actual
  local-model command outcomes, including exact/HNSW and proper refusals.
- [x] Preserve upstream changes encountered at publication: identical core lint
  repair, warm coverage transitions (`03cf0caa`) and graph refinement
  (`851021f1`). Keep failed push and retired partial gates as failed/partial
  evidence; none qualifies the later source.
- [x] Repair the demonstrated default-feature graph integration lint with a
  feature-gated private argument and conditional local mutability. Preserve the
  graph algorithm and all strict lint settings.
- [x] Freeze the combined source; pass all eighteen unchanged stock labels.
- [x] Pass strict Clippy with the optional graph feature and execute its actual
  graph-filtered library regressions at clean `7a7dee9a`: 26 passed, zero failed
  or ignored, including the incoming refinement cases. Retain this separate
  result; the later fixture-only repair does not change graph production code.
- [x] Preserve v5's failed native matrix: 406 pass/one fail/zero ignored in
  393.71 seconds; reproduce its failed update independently. Diagnose the
  `partial` case's collision with both initial `partial-work` and `partial.json`.
- [x] Verify all sixteen original warm transition combinations with distinct
  successor paths and four occupied-receipt controls; preserve every original
  cache, producer, scope, result, predecessor and follow-up assertion.
- [x] Refresh the existing Mac release admission check: DSR refuses `mmini` at
  06:34:48 UTC with 22.17 GiB free against its unchanged 40 GiB requirement.
  This is not an Apple build result or release qualification.
- [x] Retain v6's three genuine fsfs ownership failures (3,361 passed, three
  failed, sixteen ignored). Retire only its verified owned process group before
  edits. Trace overdue periodic ticks re-entering pending work in one poll.
- [x] Create/claim the separate fairness bug and add the startup blocker edge;
  preserve every older criterion and keep the active dependency graph acyclic.
- [x] Strengthen the three existing probes with real caller-owned native
  contexts and first-poll work exceeding the unchanged 10 ms tick. Preserve
  exact poll-count, cleanup, socket, output, shutdown and deadline assertions.
- [x] Retain the old executable and execute all three strengthened probes red:
  raw 3 versus 1, concurrent 16 versus 8, shutdown 2 versus 1, each exit 101.
- [x] Wire the existing cancellation-aware `yield_turn` helper after overdue
  intake, request, framing, producer and blocked-response turns. Keep every
  ready result, strict control and original timeout ahead of repeat work.
- [x] Execute all three repaired probes green and all 152 daemon-focused tests
  green, zero ignored; formatting, strict all-target fsfs Clippy and three-file
  scanner triage pass. No runtime, dependency, unsafe site or suppression added.
- [x] Preserve the additional upstream cache/recovery work at `de47fbda` before
  freezing the next combined source. Review producer-contract invalidation,
  cancellation/epoch fencing, legacy raw adapters and neutral recovery advice.
- [x] Requalify the optional graph feature on this combined source: its cache
  dependency changed, so the earlier graph result cannot qualify this tree.
- [x] Retain v7's strict-Clippy failure in the incoming legacy sync test
  provider. Return its literal as `&'static str`, preserving the trait contract,
  all assertions and strict lints. The refreshed graph run at `fcc2b15b` passes
  26/0/0; keep that scoped receipt separate from final stock qualification.
- [x] Review all 155 touched scanner findings in the four incoming Rust files.
  Full scan exits 1: nine critical, 1,625 warnings, 325 informational. All nine
  critical are unchanged schema-test failure sentinels (lines 132–253), with
  byte-identical prefix proof and zero critical in changed spans. Preserve the
  failed scan; add no suppression. Correct the first triage receipt's mistaken
  canonical HEAD in a separate retained v2 receipt bound to `fcc2b15b`.
- [x] Preserve v8's two fixture-consumer failures: 3,364 pass/two fail/sixteen
  ignored in the default fsfs unit binary. Both incoming recovery kinds pass
  their behavioral tests, and all three strengthened daemon probes pass in
  the complete suite. Old consumers still expect nine advice kinds and eleven
  quoted commands, while the expanded fixture has eleven kinds/thirteen
  commands. Retire only the verified owned group; no full-gate credit.
- [x] Correct these two consumers while preserving the original nine-kind and
  eleven-command censuses. Require both new kinds, their exact wire types and
  diagnostic action, and exact quoted doctor argv separately. Preserve all
  existing round-trip, hit, freshness, omission and quoting assertions.
- [x] Execute the corrected consumers and existing model/topology refusal
  controls; preserve the failed baseline ELF/logs and report the admitted
  scope. This admits exactly two additional diagnostic kinds, no performance
  WIN/LOSE rows, relaxed thresholds, goldens or production changes.

V8 at clean `6e458457` retains twelve PASS labels and FAIL fsfs, then verified
owned-group retirement (terminal -15, 2,999.34 seconds). The fsfs binary reports
3,364 passed/two failed/sixteen ignored in 60.48 seconds; build time is 35m39s.
`stock-gate-v8-result.json` and log SHA256
`6086036bf4d5756445b2e9a4397daea33a9eaf3c45a73d144d95abe5da9ed8d5`
are retained. The two consumers still census the old shared fixture. The
correction preserves nine original advice kinds and eleven quoted commands,
then separately requires each new diagnostic kind, its exact wire type and
readiness action, and exact quoted doctor argv. Every old hit, payload equality,
freshness, omission and shell-quoting assertion remains. Production prefixes
are byte-identical; no golden, gate driver, policy or tolerance changes.
`advice-census-pair-result.json` binds two fresh old-ELF failures (13 versus 11
commands, 11 versus 9 kinds) and the corrected advice group (24/0/0) plus payload
round trip (1/0/0), including existing model/topology refusal controls. Retained
old ELF SHA256: `d4bd87a1263df1c3411ca875657d5c06fcefd567f28cc68326705b8e37b50376`.
Strict fsfs all-target Clippy and formatting pass. Two-file UBS exits 0:
0 critical, 437 warnings, 88 informational; all nineteen touched findings are
test assertions/unwraps, fixed 0/1 diagnostic indices or checks after length
validation. No annotation is added. This admits exactly two diagnostic kinds;
no performance rows or WIN/LOSE decisions are admitted. V8 is failed history;
the later full clean-source qualification is recorded above. Mac admission at 08:30 UTC is
still refused: 20.26 GiB available against the unchanged 40 GiB requirement;
`macos-admission-refresh-v2-command.json` records actual exit 1.
- [x] Repeat the original baseline/default-installed-binary acceptance pair on
  the combined source, retaining both executable and harness identities.
- [x] Repeat the twelve actual-model native preview command outcomes and add
  fourteen predeclared actual-model warm coverage observations: exact promotes
  partial coverage to full, HNSW retires quality, denied requests preserve the
  authority, and same-stream/fresh-process queries use the committed selection.
- [x] Scan both incoming changes and review all touched-line findings. Warm:
  two files, 0 critical/3,426 warnings/479 informational, 174 nearby findings;
  graph: nine files, 0 critical/4,222 warnings/757 informational, 69 nearby
  findings. Test fixture assertions, bounded indices, retained handles and
  comment text explain these reports; no new scanner exclusions.
- [x] Export only the two owned tracker records, publish qualified code and evidence,
  and verify both remote refs without disturbing the peer's unpublished row.
- [x] Close the fairness bug, then the startup bug on their original acceptance;
  export only their final statuses. The final publication receipt records the
  post-commit push, ref verification and release of the 23 exact owned leases.

Earlier qualification: clean `b286a61d737ebbba0aff63dc73a67fec558d5686`, all
eighteen quality labels PASS, no label skips, terminal 0, 04:22:51–05:20:34 UTC,
3,463.35 seconds. `stock-gate-v2-result.json` records source, commands, labels
and terminal state; log SHA256:
`6bb94ed4256f3ac9429fdb01af214e61942fdc58dc09f3e4a3c6ca1257dcd797`.
Execution is native thinkstation1 at the original UID/groups in a fresh private
4 GiB tmpfs. The genuine prior executable is retained. `fast_cmaes` remains
clean at `18b3910246862468893ca1d0cd345a379e4f7082`; lock/script hashes are
unchanged. The native CLI suite passes 402/0/0, including both corrected drivers.
The bounded Quill lane is not the separate full campaign or performance gate.

The fresh final-harness acceptance pair uses the actual installed default
quickstart binary: baseline main-count failure, exit 101; repaired executable,
all seven scenarios, exit 0, 78.10 seconds. `final-acceptance-result.json` binds
both ELFs, harness, logs and unchanged source. Installed default ELF SHA256:
`b544d60bfeb1676e283a8ea024b12975b86b8191a6dd73da531ca5ce0a27da24`.
The final combined scan of all twelve changed Rust files exits 0: 0 critical,
11,883 warnings, 3,154 informational findings; the scoped triage and the three
test-sentinel classifications above apply, without a warning-free claim.

`native-preview-product-v1-result.json` records twelve actual registered
Potion/MiniLM outcomes in 96.84 seconds: ten successful commands and two proper
regression refusals. Exact and HNSW each index fast-only, backfill a strict
quality subset, refuse regressing coverage before model loading/staging,
replace the complete source with an explicitly authorized full-coverage
successor, and reopen in fresh processes for Initial/Refined search and source
fetch. Producer identities, result IDs, replacement title and metadata survive;
refusals preserve AUTHORITY/LOCK and create no work or receipt. Artifact root:
`/home/ubuntu/.native-preview-4rh7b5ba`. Native ELF SHA256:
`d3bad6483ab47fdedb42b28da285f3ae9e4a4101ecc409a6e47833ddd78b311f`.

Origin advanced only by `5b7bac4b` with the same two semicolon fixes already
qualified here. Merge `c3f986d7` preserves that commit; its entire Git tree is
identical to the qualified tree `6b420a4503a39dcd66096c0b101192ee1e7c4df7`.
`merge-tree-equivalence.json` records the equality; the post-merge import makes
zero updates. That merge changed no product code. The later publication attempt
was refused by the Agent Mail pre-push guard because the moving remote revision
was not fetched locally; no guard bypass or remote write occurred. Origin then
advanced through `03cf0caa` and `851021f1`. These substantive peer changes are
preserved in merges `8e365f68` and `31e4e166`; their runtime qualification is
separate from the earlier full pass. The clean v3 gate was retired after eleven
PASS labels with terminal -15. The v4 gate demonstrated default/hybrid strict
Clippy failures at the new graph argument, while graph-enabled Clippy passed;
it too was retired before repair. Both partial/failed receipts remain retained.
The small private cfg repair adds no lint exemption or runtime branch.

The next clean run, v5 at `7a7dee9a`, passes fourteen stock labels but fails
the native CLI suite as reported above. Its owned process group is retired
before edits (terminal -15); no full-gate credit. Separate optional graph
runtime evidence is terminal 0: 26/0/0, 14.21 seconds of test execution;
`graph-feature-tests-result.json` includes build/queue time and log SHA256
`b09112c63f0aba561a75986cfb46d2bf66d333806e9c75587403a0393048e665`.
The combined 22-file Rust scanner exits 0 with 0 critical, 19,400 warnings and
4,277 informational findings. The existing scoped triage applies.

The native failure is a third demonstrated test-driver defect. Temporary
test-only internal diagnostics identify the existing-destination refusal;
that logging is removed, leaving production code byte-identical to `7a7dee9a`.
Both initial fixture paths collide only when the scenario label is `partial`.
The first correction separates the receipt but misses the work-directory
collision; the added negative control catches that mistake, and its failed
receipt remains retained. Both successor paths now have a distinct prefix.
The original sixteen positive scenarios and assertions remain intact, with
four new exact-refusal controls proving occupied receipts preserve authority,
lock, receipt bytes, reader identity, inference counts and absent successor
work. This admits valid fixture destinations and still rejects occupied ones;
it changes no production policy, cache behavior, gate, tolerance or budget.
The corrected focused test passes in 329.88 seconds: all sixteen original
combinations plus four occupied-receipt refusals. Formatting, strict hybrid
all-target Clippy and the corrected test-file scanner pass (0 critical, 3,307
warnings, 375 informational across the whole file). The new-line findings are
intentional test assertions, fixture unwraps, bounded frame access and retained
test handles; no scanner exclusion is added. The source file remains under its
existing test-only module. Full clean-source stock requalification is pending.

V6 at clean `6bfb8d44` exposes the overdue-turn bug rather than a fixture
failure. Three original tests observe repeated pending-owner polls during one
owning poll; the unmodified stock lane reports 3,361/3/16. The request and
nested output drivers can outlast a 10 ms tick during synchronous work and
immediately loop again. The retained old harness with strengthened native
stimuli fails all three exact original oracles. The repaired harness passes
each and the complete daemon group (152/0/0, 3.08 seconds). These are actual
native runtime/Unix socket tests with existing pending-work controls, not
real-model relevance or release evidence. `overdue-native-pair-result.json`
binds both ELFs and the six logs; the old ELF SHA256 is
`7f58b86415801eebc3888a5858aaa1595311618bcf80986cb37cc2c620acad59`.
The existing helper supplies a real scheduler yield and cancellation checks;
no sleep is added to production and no deadline or assertion is weakened.
Three-file UBS exits 0: 0 critical, 855 warnings, 168 informational findings;
the three touched-line findings are deliberate native test setup unwraps.
Strict fsfs all-target Clippy passes. Both closures were withheld until the full
combined stock run, fresh solo original-criteria checks and publication recorded
at the top of this block.

Real-work worksheet (these repairs and their integrated validation): the
README purpose is two-tier hybrid local search for Rust and fsfs. Inventory:
`b692f5d4` fixes startup compaction with its real-model test; `b286a61d` connects
the missing native preview APIs and repairs demonstrated test-driver defects;
`d327790c` fixes overdue daemon cooperation with its native regressions;
upstream warm/graph/cache/recovery changes retain their authorship. By work
item: USER 3, ENABLER 1
(executed qualification), PROCESS 1 (requested tracking/publication record),
UNKNOWN 0. The two-minute demonstration is append, start daemon, inspect both
merged tiers and query the appended document. Removing the tracking would not
change runtime capability; it would lose the requested evidence and honest
closure boundary. Tests consumed the actual fixes. The oldest open epic,
`bd-quill-e6-gauntlet-scale-rm3q`, requires its blocked children and two full
nightly runs; this stock bounded lane cannot close it. No peer acceptance was
split, rewritten or closed. Solo work; no swarm. Verdict: HEALTHY, with zero
capability credit for this worksheet or the tracker record.

Honesty inventory (bounded to these changes and retained attempts):

1. No (checked: both source diffs and the stock script/lock hashes): no original
   assertion, tolerance or budget was weakened. The new real-model case follows
   the existing ignored-by-default convention and is executed by the e2e lane.
2. No (checked: new executable regression and native diffs): no new model mock
   or fake implementation; the matched negative uses the existing loader seam.
3. No (checked: golden/config/dependency diffs): no golden regeneration.
4. Yes: three explained local UBS annotations classify existing test-only panic
   sentinels; their bodies remain intact and the unannotated production control
   still blocks. Three earlier driver defects and two consumers of the expanded
   advice fixture are corrected above, preserving/strengthening old assertions.
   Those corrections change no production cache, policy, threshold or budget;
   the separately preserved upstream cache change is qualified on final source.
5. No (checked: runtime/API diffs): no demo-path, environment or benchmark branch.
6. No (checked: actual result counts): startup 1/7 scenarios, three paired
   ownership probes, graph 26, fsfs 3,366/0/16, facade 618/0/0, native 407/0/0,
   twelve live command outcomes, fourteen warm observations, eighteen labels.
7. No (checked: command logs/receipts): no unexecuted check reported as passed.
8. No (checked: execution classes): baseline remote build is distinguished from
   native qualification; synthetic existing unit providers are not live proof.
9. No (checked: failed receipts and progress comments): copy race, wrong expected
   CRC diagnostic, WAL-quarantine defect, upstream compile/lint failures, both
   failed native tests, all retired gates, the missed work-path collision,
   overdue-turn failures and both fixture-consumer failures remain visible.
10. No (checked: command redirections): stdout/stderr evidence is retained.
11. No (checked: Beads): both bugs closed only after qualified code was verified
    on both remote refs; their original acceptance descriptions are unchanged;
    broader migration, performance and release acceptance remains open.
12. No (checked: original Bead description): acceptance/specification unchanged.
13–17. Not applicable: truly solo; the prior prototype is credited and rechecked,
    and no peer report or agreement substitutes for executed verification.
18. No (checked: declarations): seven startup scenarios, three ownership probes,
    eighteen labels, twelve live outcomes, fourteen warm observations and both
    explicit new diagnostic-kind checks were selected before their results.
19. The copy race and initial test assumptions warranted explanation; both were
    disclosed and corrected, with failed runs preserved. No independent review
    or competitive/performance/binary-release qualification is claimed.
20. Strongest evidence: re-executable paired installed-binary red/green, actual
    artifact/Refined-result assertions, the clean full gate, and live native
    source/policy consumers. All are bound to the qualified source and ELFs.

Six bounded CASS searches plus archived hit context checked older work and this
named probe; this is a sample, not an exhaustive history audit. Disposition:
the observed defects and annotation exceptions are corrected, disclosed and
guarded by retained negative controls. Patterns guarded against are false/zero
green, gate self-weakening, proof-class inflation and false closure. Qualified
code is published and the two bugs are closed. Consult the final publication
receipt for the post-commit status push, ref verification and lease releases.

Current evidence: `/data/tmp/fs-daemon-startup.sKDROXdS`. The clean-overlay
`b27feb94` build with explicit `fast_cmaes` revision completed on hz4 and its
artifact was retrieved (terminal acknowledgement and exit 0). Retained baseline
ELF SHA256: `33ef59c2395b67a05c8cb176c984fe03f8c5b15c583cb1951b319efc9367977c`.
The native-built permanent probe executes that baseline and fails at the actual
startup result: daemon exit 0, fast main count 1 instead of 2. Both pending WALs
also remain byte-identical in the separately retained product reproduction.
Admission refusals (external containment, slots, queue timeout, disk) are saved
and earn no compilation/test credit. Final stock/fresh results are recorded above.
The first candidate launch raced an unfinished executable copy and failed to
spawn; the next retained run passed all four positive scenarios, then caught
an incorrect expected diagnostic in the new test. A first-byte sidecar edit
trips the existing CRC check before magic validation (`repair_trailer.rs:98`).
The assertion now requires that CRC diagnostic, with the original nonzero,
typed-corruption and byte-preservation checks retained. No stock test changed.
The next run passed both malformed-sidecar controls but caught an actual
production defect in the added quality-WAL control: writer-open recovery
quarantines an unusable WAL and continues, so writer opens alone are not strict
preflight. The repair adds read-only validation of all pending tiers before
opening either writer; the failed run is retained alongside the complete rerun.

Focused result: `candidate-named-probe-v4.log`, terminal 0, one actual test,
seven predeclared scenarios, 73.50 seconds. Four positive scenarios merge both
tiers and preserve Refined hits, producer/dimension/quantization and idle
bytes/inodes/mtimes; three malformed-input scenarios refuse before merging and
preserve the watched bytes. The quality-WAL control now reports bad magic
without quarantine. Default executable SHA256:
`43661b1adc2505418ac65a7ff9e1d7ffa65953394c26023f6e25030726869df4`.
Focused Clippy passes with warnings denied; formatting and diff checks pass.
UBS exits 0 on the two changed Rust files, with 0 critical, 10,016 warnings and
2,703 informational findings across their entire contents. The 114 findings
in touched spans were reviewed: test assertions/unwraps and deliberate fixture
writes fail the test on bad setup; indexed arrays share the fixed tier count,
test tier selectors are 0/1, and corrupted bytes come from successfully opened
nonempty files. Paths use fixed tier constants; apparent division/modulo are
path strings/tracing syntax. Bounded fixture copies and command/error logging
are intentional. No suppression or scanner/gate change was made. Unchanged
file regions were not qualified by this scoped triage.

Creation worksheet (November 25, this checklist and retained run evidence):
running code does not branch on these artifacts. Consumer: the owner explicitly
requested a granular TODO and honest Beads execution. Gate: publication/closure
of the startup-compaction bug requires the original populated red/green and
unchanged stock checks. Observed defect: an exit-0 daemon silently leaves both
startup WALs pending, demonstrated above. Retirement: stop active tracking at
qualified publication or an explicit incomplete handoff; preserve dated history.
Verdict: legitimate minimal gate. The highest-value ready capability is this
repair; another process framework would displace it and is not being created.

## November 24 source-replacement wiring — published, OrangeSparrow

Owner-requested implementation checklist for `bd-native-authority-replace-source-6o6m1`.
Qualified code `f02742f3` is published and verified on both remote refs. The
feature Bead is closed on its original acceptance; this block is now dated
history, retired from active tracking. It earns no capability credit.

- [x] Refresh origin and preserve the dirty shared checkout in the existing
  isolated worktree. Governing docs are unchanged from the full prior read.
- [x] Find the actual unused connection: the library can replace a complete
  inventory with vector reuse, while authority CLI/warm input only accepts deltas.
- [x] Record the old real executable's refusal of `--replace-source`.
- [x] Preserve the blocked parent migration. Its claim is refused; create one
  additive feature under the exclusive graph lease, with original parent criteria
  unchanged and all implementation/tests in the same unit.
- [x] Wire cold CLI standard JSONL and mutually exclusive warm documents into
  the existing complete-source library method; retain all source/publication limits.
- [x] Verify exact/HNSW and fast/quality/full/partial flows: unchanged vectors,
  omitted-ID deletion, metadata, new IDs, empty complete source, warm scopes/cache
  refresh, independent quality membership and retained old readers.
- [x] Verify malformed/duplicate/oversize/mixed input, authorization, stale head,
  cancellation and partial-boundary refusals before inference/staging.
  Four populated focused cases pass, including eight topology/retrieval combinations
  and both cold and warm replacement. The stale control uses a valid different
  authority. Full/single-tier empty inventories succeed; partial collapse refuses.
- [x] Execute actual-model cold and warm commands plus fresh-process readback;
  retain ELF/source/feature/lock identity and adverse controls.
  Twenty populated outcomes pass on clean `f02742f3`, with actual registered
  Potion/MiniLM models and full/partial HNSW authorities. New IDs are searchable,
  omitted IDs unavailable, and stored metadata survives cold and warm publication.
  Malformed, mixed and stale controls refuse. The debug hybrid ELF is
  `333b439ba6a5bc168e4ed7e27b0a62790855bc94aca7f0ace5abe8ae76f29383`.
- [x] Run focused strict checks, scanner triage and the complete unchanged stock
  gate on a clean source checkpoint. RCH admission currently queues all workers;
  native quality-host runs must retain their actual execution class.
  - [x] Retire the initial `dbfccddb` run before changing source: six labels pass,
    then owned group termination yields -15. A parser negative lacked a valid
    base-command control; add valid bases and exact refusal classification.
    Retain the original run and all actual-model evidence without full-gate credit.
  - [x] Retire `b1fe37b7` before further test edits: two labels pass, terminal -15.
    Finish the negative-control review: valid oversized JSON, unique over-limit
    warm IDs, exact refusal codes, and a valid cancelled request that succeeds
    unchanged when cancellation is cleared. Production implementation is unchanged.
  - [x] Final stock run: all eighteen labels PASS, no SKIP, terminal 0,
    15:58:27–16:47:33 UTC, 2,946.66 seconds. Clean `f02742f3`, native
    thinkstation1, original UID/groups and a fresh private 4 GiB tmpfs. The host
    mount, quota and retained cache remain unchanged. No gate source or budget changes.
  - [x] Facade library 598/0/0 and native CLI 352/0/0, including all four added
    replacement cases. Focused v4 cases and strict hybrid Clippy pass. Final
    UBS scans all three current Rust files: zero critical, 140 changed-line
    notices triaged, no suppressions. Seven production notices concern bounded
    owned ID keys and a HELP-string slash. Existing warnings are not claimed fixed.
- [x] Reverify the feature's original criteria in a fresh solo pass, update Beads
  honestly, publish source/evidence and verify both refs. The broader migration,
  platform and release tasks retain their original acceptance.
  - [x] Read all five original criteria again and bind each to actual code,
    populated controls, source/ELF/model identities and terminal gate results.
    This is solo verification, not independent certification.
  - [x] Publish qualified `f02742f3` and verify both refs before closing the
    feature. Preserve `bd-xomn.3`'s original prerequisites and acceptance.
  - [x] Close only the owned feature Bead on cited evidence. Scope the tracker
    export to that feature and owned `bd-dsbym` comments; preserve the unpublished
    peer `bd-fsfs-current-product-envelope-lz056` update and all other records.
  - [x] Add the user-facing Unreleased changelog entry and retain this detailed
    implementation/verification history. Subsequent metadata changes contain
    no product-code, dependency, model, gate or acceptance changes.

Receipts: `/data/tmp/fs-wire.Vki6tW2z/stock-gate-v3-result.json`,
`product-probe-final-result.json`, `original-acceptance-review.json`,
`model-files-final.json`, `ubs-final-triage.json`, `focused-tests-v4.log` and
`source-publication-refs.txt`. The complete gate log SHA256 is
`f977935aad91cf5c0ad96490b53e0de1592abfa53049206fe3d963e18e9576bf`.
Cargo.lock remains `ad2c99a06a8ffe5353a0775f396c95fd6c8ffebc7c095d0c61f2863bab83410c`.
All nine registered model files are unchanged across the real-model probes.
No performance, strict-remote, physical-Apple, binary-release or full-Quill-
conformance result is claimed. `bd-dsbym` remains in progress: the fresh
16:45:29 UTC Apple admission refuses 2.79 GiB available against 40 GiB. Its
original 15:41 probe and both retired stock runs remain retained.

### Bounded work and honesty review for source replacement

Window: `e473ee17` through `dbfccddb`, `b1fe37b7`, `f02742f3` and this metadata
update. Inspect the three actual source commits, their test diffs, unchanged
manifests/gate/configuration, the published refs and this one owned Bead close.
The README's purpose is two-tier hybrid local search: fast first-pass results,
then quality refinement.

**Process worksheet.** Runtime does not branch on this TODO or tracker comments.
Consumer: the owner explicitly requested a granular TODO and current Beads.
Gate and defect provenance: that request covers tracking this native rescan
feature; the observed weak negative controls show why qualification must remain
distinct from partial runs. Retirement: source publication and evidence-bound
closure, now reached; preserve this as dated history. Raw logs/identity receipts
are an integrity control: they prevent losing exact retired runs or confusing
the ELF/source/model that ran; yes, needed for multiple profiles and attempts;
yes, raw receipts are the minimum; no, a commit name alone cannot identify the
executable. Highest ready capability work was this independent source-ingress
feature after the blocked parent refused claiming; `br ready` was otherwise
empty. Another hour of report design would deliver less value. Verdict:
LEGITIMATE GATE for the requested checklist; no new report framework.

**Real-work worksheet.** `dbfccddb` is USER: a cold rescan or warm replacement
can remove omitted IDs and reuse unchanged vectors. `b1fe37b7` and `f02742f3`
are ENABLER: corrected refusal controls for that feature. This tracker/evidence
update is PROCESS, with a small user-facing changelog entry; UNKNOWN is zero.
Tally: USER 1, ENABLER 2, PROCESS 1; no commit-count credit. A two-minute demo
is the documented replacement command followed by fresh search/fetch. Removing
PROCESS leaves runtime unchanged but loses the requested progress/evidence
boundaries. The new controls are exercised by focused and full stock tests;
actual model behavior is separately exercised by twenty live outcomes. The
oldest bounded open item remains Quill E6, July 16; it has its own original
dependency/conformance requirements, which this additive CLI feature cannot
close. No swarm or close farming. No plan-only commit substitutes for code, and
no follow-up carries away an unmet original condition. Verdict: DRIFTING in
cost: starting gates before completing the negative-control review wasted two
retired runs. Correction: finish that review before freezing source, then
publish the qualified capability without adding more reporting machinery.

**Honesty inventory.** Answers are bounded to this window and cited artifacts.

1. No (checked: the full test diff adds/strengthens cases; no old assertion,
   ignore, gate, tolerance or deadline is weakened).
2. No (checked: existing providers are labeled unit controls; actual model
   CLI/process proof is separate, and no new fake implementation is introduced).
3. No (checked: changed paths contain no regenerated golden or snapshot).
4. New regression tests and shared product document validation accompany the
   feature. No test gate, suppression, skip, retry budget or tolerance changes
   (checked: original predicate and unchanged gate/configuration diff).
5. No (checked: all eighteen stock labels run on the recorded native host;
   no benchmark hardcoding, special expected results or narrowed stock stages).
6. No accepted test result is zero-run (checked: four populated focused cases,
   all 352 native cases, 598 facade library cases and twenty live outcomes;
   empty Quill inventory targets explicitly receive no pass credit).
7. No (checked: cited results have observed terminal exits and raw receipts;
   source publication is verified on both remote refs).
8. No (checked: unit providers, actual Potion/MiniLM, debug profile, native
   host and refused remote/Apple admission retain their actual proof classes).
9. No material failure is omitted (checked: weak controls, both -15 retired
   runs, queued RCH admission and both Apple refusals remain recorded).
10. No (checked: claimed runs retain stdout/stderr or combined raw logs;
    incidental reader/session checks are not qualification evidence).
11. No (checked: only the owned additive feature closes after original-criteria
    review and publication; broader migration/release requirements stay open).
12. No (checked: the feature's original description and acceptance are unchanged;
    parent prerequisites and peer acceptance are not edited).
13. No (checked: this window is solo; the one owned close cites exact source).
14. No (checked: no subagent dispatch or success-only delegation).
15. No (checked: no subagent report substitutes for executed evidence).
16. No (checked: no panes or refusal-only closes; the feature has live positives).
17. No (checked: no agent agreement or independent certification is claimed).
18. No (checked: topology/retrieval combinations, twenty live outcomes and
    eighteen stock labels are defined before their results; no performance ratio).
19. Yes: initial parser, oversize and cancellation negatives could fail for
    another reason. Correct them in place with valid bases, exact refusal
    classes and the identical valid cancelled request succeeding afterward.
    Retain both prematurely started gate runs without full-pass credit.
    One tracker-read formatter also expected a list instead of an issues object;
    correct that read without changing its original JSON or counting it as a gate.
20. Strongest evidence: the re-executable clean-source eighteen-stage stock run
    plus source-bound ELF/model identities and the twenty actual cold/warm/fresh
    process outcomes. The receipt files above identify every command and result.

Older-session review remains the prior bounded CASS sweep and direct views.
This window adds `cass-replacement-search.json` and `cass-replacement-view.json`:
the inspected hit concerns Cargo vendoring, not this rescan feature, and supplies
no current qualification or misconduct claim. No second corpus writer is started.
Disposition: the weak controls are corrected, disclosed here and in commentary,
and covered by RH-2's valid positive/negative comparisons; RH-1 retains the
unchanged stock gates, RH-4 gives retired runs and commit counts no capability
credit, and RH-9/RH-10 preserve the original broader acceptance. The source is
qualified for this feature; overall release and project completion remain open.

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
  The 06:50 UTC recheck still refuses admission: 4.31 GiB versus 40 GiB.
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
  - [x] Execute the complete stock gate and actual signal/pressure controls.
    - [x] Preserve the clean `8de81f7f` stock attempt: all stages through
      Quill passed, then fsfs linking hit EDQUOT on quota-enabled `/dev/shm`.
      Stop only the owned gate tree; retain terminal 143 and linker receipts.
    - [x] Copy the cache without deletion to the separately probed volume
      with 15 TB available: rsync terminal 0, 64,572,120,354 bytes transferred.
    - [x] Rerun the entire unchanged stock gate on the copied cache.
      - [x] Retain the completed `a35b1769` attempt, terminal 1: fourteen
        labels pass; hybrid Clippy, workspace tests, facade tests and real-model
        e2e fail. The fsfs, HTTP, quickstart and lite stages pass. No skips.
      - [x] Supply the genuine retained `2e2b8026` prior executable to the
        two-executable test; the previous full run omitted this required input.
      - [x] Complete all eighteen labels on the repaired committed source.
        Clean `ebae5455`, 06:04:09–06:49:39 UTC, terminal 0, no skip allowances.
        Native thinkstation1, original UID/groups, private supported 4 GiB
        tmpfs; original host quota and retained cache remain unchanged.
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
    - [x] Recheck actual foreground signals, both 128 MiB pressure profiles,
      and cancellation while paused under an explicit 100-per-mille load ceiling.
      This configured pressure probe cannot establish default-pressure behavior.
      The first probe used 1, which the existing minimum-100 validator correctly
      refused. That receipt is not pressure proof; the corrected 13-command
      probe passes on preserved candidate ELF `bc531f43`, with no owned
      processes left and the incumbent independently searchable after each stop.
      The final invocation-built default ELF `83d0c462` passes the same
      thirteen commands; `/data/tmp/fsfs-foreground-95mjla_b/summary.json`
      binds its exact hash, incumbent selection and independently served hits.
    - [x] Recheck strict Clippy and the affected pressure/doctor tests on
      the separate `semantic-native` profile; retain its exact feature set.
      `--no-default-features --features semantic-native`: all-target strict
      Clippy passes; pressure 223/0/0 and doctor 33/0/0. No platform pass.
  - [x] Preserve the legacy symlink watch repair as a permanent regression.
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
    - [x] Qualify current-source lexical watch with sibling and nested storage;
      preserve the broader upgrade-reuse Bead and its original acceptance.
      Final ELF `83d0c462` passes both layouts; historical `83ff0d18` still
      reproduces the original bad behavior. One initial polling run encounters
      the documented typed generation-change retry during publication. Retain
      that failed observer run; permit only the exact typed reasons within its
      unchanged twenty-second poll deadline. Final hit/inventory/shutdown
      assertions remain; neither successful replay encounters that retry path.
      The permanent stronger real-model handoff case also passes in the stock
      e2e stage, with both original vector tiers and alternate bytes checked.
- [x] Repair the independently reproduced timeout-control fixture before
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
  - [x] Complete the stock gate on the committed fixture checkpoint.
  - [x] Publish the integrity split: this corrects a maintenance fixture;
    no competitive benchmark rows, WINs, LOSEs, or performance gates change.
  - [x] Record the exact dependency change, primary sources and terminal
    results in the existing UPGRADE_LOG; preserve failed attempts.
- [x] Qualify the integrated partial-quality changes under the original gates.
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
  - [x] Complete the unchanged full stock gate with all required inputs.
    All 598 facade library tests and 348 native CLI cases pass; real-model
    CLI/recovery and genuine two-executable reuse also pass. This is native
    host qualification, not a strict-remote or cross-platform release claim.
- [x] Qualify the newly published source replacement API, preserving peer credit.
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
  - [x] Run the entire stock gate on the new clean combined checkpoint with
    the already demonstrated private tmpfs and the genuine prior executable.
    The tests use control providers for replacement behavior; they do not
    establish ordinary fsfs authority convergence or real-model replacement.
- [ ] Complete remaining dependency review without relaxing audited identity,
  forbidden-runtime, profile, wire, installer or consumer contracts.
- [ ] Recheck platform admission before any release. Keep the original
  six-variant acceptance; preserve the current release until all gates pass.
  - [x] Re-execute the named DSR admission probe at 06:50 UTC: exit 1,
    all five writable Apple roles refused at 4.31 GiB, minimum 40 GiB.
  - [ ] Complete the remaining dependency migrations, then qualify all six
    release variants, installers/updaters and all thirteen public crates
    together with their registry consumer lanes;
    select forward versions and publish/verify the actual releases. The native
    source gate does not fulfill these original release conditions.
  - [x] Live source/registry audit finds thirteen public packages, all already
    published from `d1b9da15`; only `PUBLISHED_VERSION_SOURCE_MISMATCH` blocks
    the current unbumped candidate. Do not republish those versions.
  - [x] Determine Homebrew applicability from the actual complete tap tree:
    `3244f87d` has 56 files and no fsfs/frankensearch formula, cask or update
    route. The connector was rate-limited; a fresh public Git snapshot supplies
    the evidence. No existing entry needs updating; no brew install is claimed.
- [x] Publish qualified increments and scoped Beads/TODO evidence, verify
  both remote refs and dependency cycles, and fill the bounded honesty review.
  Qualified source `ebae5455` is already verified on both refs. Comments 5421
  and 5422 bind the final results; the Beads-generated publication changes only
  those two records among 1,488, preserving all prior comments and original
  criteria/owners/status. Active cycles remain zero. This evidence-only update
  adds no product-code change; final remote tips are retained separately.

### Bounded work and honesty review for this continuation

Window: after `c5a7464a` through qualified `ebae5455`, plus this evidence-only
update. Inspect the actual twelve-commit log, owned test diffs, manifests,
gate/configuration diff and reflog. The source purpose remains the README's
fast first-pass local search followed by quality refinement.

**Process worksheet.** Running search code does not branch on this checklist,
bridge or tracker comments: these are PROCESS. The owner explicitly requested
the granular TODO; the operator and `bd-dsbym` release decision consume it.
It prevents the observed mistake of confusing focused controls, retired runs
or missing test inputs with a completed source/release gate. Retire this block
from active use after publication/handoff with the listed unmet conditions;
retain it as dated history, without deletion. Raw receipts are the integrity
exception: they prevent loss of exact failed runs and source/ELF identities;
yes, needed at this multi-profile scale; yes, raw logs plus small identity
records are the minimum; no, a commit hash alone cannot establish which ELF
ran. No new project report framework or schema is created. Highest ready work:
`br ready` is empty; the owned release task is still in progress. An extra hour
of report design would add less value than the remaining product/dependency
work, so stop at the existing three documents and scoped tracker comments.
Verdict: LEGITIMATE GATE for the requested checklist, not feature credit.

**Real-work worksheet.** My five nonmerge commits classify as USER for
`8de81f7f` (doctor/metric handling), ENABLER for `e591bf57`, `8ba1188c`,
`60a07028`, `ebae5455` (regression/fixture/build corrections). The two merges
are ENABLER. The five peer native feature commits are USER/API work by their
authors, not five new features authored here. This final tracker/docs update
is PROCESS; UNKNOWN is zero. Tally including peer commits and the evidence
update: USER 6, ENABLER 6, PROCESS 1. This is a diagnostic, not a quota.
The visible authored change is honest unavailable-metric reporting; demonstrate
it with doctor and its explicit OS-error controls, without claiming a live OS
failure was induced. Removing the PROCESS update would leave runtime behavior
identical, but would hide the owner-requested completion boundaries. Enablers
are exercised by the actual stock gate, downstream product tests, installed
CLI and native profile. Oldest bounded open item is Quill E6 `bd-quill-e6-gauntlet-scale-rm3q`
(July 16); this block instead finishes the owner's existing release predecessor
and unqualified integrated source, leaving E6's original criteria intact.
There is no swarm or close farming, and no new follow-up launders acceptance.
Verdict: DRIFTING in cost—qualification and fixture/infrastructure repair
dominate this authored block. Correction: stop adding reporting machinery,
publish the tested increment, and keep the next capability block on the
existing authority/model-provisioning/dependency graph when its prerequisites
are admitted. Do not present this block as broad new search functionality.

**Honesty inventory.** Each answer is bounded to the artifacts above.

1. No (checked: every owned test diff and unchanged gate/configuration;
   assertions, ignores, deadlines and tolerances are retained).
2. No (checked: new OS-error controls are labeled unit injections; actual
   process/filesystem/model proof is separate; no fixture masquerades as it).
3. No (checked: changed paths contain no regenerated golden or snapshot).
4. Yes: timeout/capture fixtures and one flag-specific diagnostic assertion
   change; old actual counterexamples and unchanged behavioral guards justify
   them. The external live-poll observer also honors only the exact documented
   generation-change retry within its original deadline. No stock gate,
   production admission, suppression, retry ceiling or test budget is relaxed.
5. No (checked: stock invocation has every original label, genuine prior
   executable, actual models and no skip allowance or affinity override).
6. No accepted result is zero-run (checked: all eighteen gate labels, exact
   focused counts and process commands). The empty initial UBS baseline is
   excluded and replaced by the populated scan; it remains recorded.
7. No (checked: cited commands have observed terminal results and raw receipts;
   retired and blocked invocations receive no pass credit).
8. No (checked: control-provider replacement tests, injected OS errors,
   historical unattested ELF, actual model/process tests and private native
   host results retain their distinct proof classes).
9. No material failure is omitted (checked: original quota, fixture, missing
   input, observer failures and each retired run remain in this TODO/log and
   Beads). This is qualified source; release and project completion stay open.
10. No cited qualification result loses stderr (checked: combined or separate
    raw stdout/stderr files; routine absence probes are not pass evidence).
11. No (checked: this continuation closes zero Beads and creates none;
    `bd-dsbym` and the peer reuse task keep their original acceptance).
12. No (checked: tracker status/owner/priority/description/dependencies match
    the published originals; documentation appends evidence, not new targets).
13. No (checked: solo continuation; no subagent tracker closes).
14. No (checked: no subagent dispatches or swarm panes).
15. No (checked: peer code is read and the combined source re-executed;
    no peer report substitutes for compilation/runtime qualification).
16. No (checked: no panes farming refusals and no new closures).
17. No (checked: no inter-agent agreement is claimed; this is solo verification,
    not independent certification).
18. No (checked: eighteen stock labels, thirteen foreground commands,
    sibling/nested layouts and original acceptance predate their results;
    no competitive rows or selected performance denominator is reported).
19. Yes: explain the avoidable wrong precise-update command, initial imports,
    empty UBS extraction, incorrect old-watch observer expectation, invalid
    load ceiling 1, late SIGSTOP discovery, missing prior-executable input and
    final live-poll conflict classification. All failed attempts remain; each
    corrected observer/configuration is labeled, not silently substituted.
    Waiting-driver setup errors and retired source selections are not evidence.
20. Strongest evidence: the re-executable clean-source full stock invocation
    and source-bound installed ELF, plus adversarial old failures and actual
    final foreground/watch controls. `dependency-stock-gate-v6-result.json`
    and `final-candidate-validation-completion.json` bind the terminal results.

Older-session check remains the bounded CASS sweep and five direct views in
`/data/tmp/fs-exec.SoVtXGnZ/honesty-{search,view}-*`; the release-history review
adds no current pass credit. Disposition for the yes answers: corrections are
in place, failures are disclosed in commentary and retained here/Beads, and
countermeasures are explicit under RH-1 (independent before/after fixture
controls, unchanged gates), RH-2 (proof classes and nonzero census), RH-4
(no commit-count credit), RH-15 (fixed deadlines) and RH-9/RH-10 (no
closure/spec laundering). Failed attempts stay explicit throughout.
Two unchanged test-only panic assertions remain visible in the additional UBS
scan; do not call it zero-critical. Zero new competitive rows, WINs or LOSEs.

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
