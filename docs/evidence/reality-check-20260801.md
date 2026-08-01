# Reality Check — frankensearch, 2026-08-01

**Author:** MossyPine (claude-code/fable-5), user-directed `/reality-check-for-project`, end to end.
**Basis:** clean `origin/main` `6e537b31` extracted by `git archive` (never the dirty shared tree),
plus this session's three hostile review sweeps. **Every claim below was executed or read at a
named line — nothing is inferred from bead titles.**

---

## 0. The headline

The engine works. The product does not ship.

`fsfs search` returns the correct document ranked #1 with a proper snippet, `in_both_sources: true`
(lexical and semantic agreeing), and honest freshness metadata. That is the core value proposition,
and it genuinely delivers.

`fsfs index` **never terminates.** It completes all of its work — writes the vector index, the Quill
lexical engine, the CURRENT pointer, and a sentinel saying `generation_complete: true` — and then
hangs forever. Reproduced twice on a 3-file, 196-byte corpus.

And the **default source build cannot index at all.**

Neither defect is visible from the test suite, because tests call library functions and never
exercise the binary's shutdown path or its default feature set.

---

## 1. Vision Checklist (extracted from README.md + AGENTS.md)

| # | Goal | Source | Status | Evidence |
|---|---|---|---|---|
| 1 | `fsfs index <dir>` builds an index | README Quick Start | **REGRESSED** | Work completes; **process never exits** (EXIT 124 at a 150 s hard cap, sentinel `generation_complete: true`) |
| 2 | `fsfs search <q>` returns ranked hybrid results | README Quick Start | **WORKING** | Correct doc rank #1, snippet, `in_both_sources: true`, semantic ranks present |
| 3 | Default developer build is usable | README Cargo Install | **REGRESSED** | `default = []` (fsfs Cargo.toml:13) ⇒ `index` exits 78 `embedder_unavailable`, no index dir created |
| 4 | Progressive phases (Initial <15 ms, Refined ~150 ms) | README perf table | **UNPROVEN** | Only `phase: "initial"` observed; `duration_ms: 18519` in a debug build — no release measurement exists |
| 5 | Quill replaces Tantivy as default lexical engine | AGENTS/campaign | **NOT STARTED** | `default = ["hash"]`; `lexical` still resolves to Tantivy; `quill` in no aggregate feature |
| 6 | QG-1..QG-10 performance gates certified | quill-perf-gates.toml | **NOT STARTED** | **10 of 10 `activated = false`**; zero certified baselines |
| 7 | Semantic search is reliable | campaign goal | **PARTIAL / defective** | 5 CRITICAL wrong-result defects filed today (`bd-5fsy6`, `bd-cnby1`, `bd-180wl`, `bd-3zh67`, `bd-ihb98`) |
| 8 | Typed embedding identity prevents cross-space scoring | bd-9xuj | **PARTIAL** | C1r2/C3/C2 reviewed GO — but **zero production call sites**; enforces nothing at query time |
| 9 | Publishable to crates.io | README | **BLOCKED** | `hnsw_rs` is a git dependency (Cargo.toml:136) |
| 10 | Doctor diagnoses real problems | README | **WORKING** | Correctly reported "models on disk but binary lacks loaders" + exact remedy |
| 11 | fsfs cross-process write safety | implied | **PARTIAL** | `bd-jr74s`: no index-root lease on origin/main; fix in flight (GoldThrush) |

**Scoreboard: 3 WORKING, 3 PARTIAL, 2 REGRESSED, 2 NOT STARTED, 1 BLOCKED, 1 UNPROVEN.**

---

## 2. The two blocking defects, with root-cause analysis

### D1 — `fsfs index` never exits (P0, no bead)

Reproduced: `timeout 150 fsfs index <3-file corpus>` → **EXIT 124**, with
`index_sentinel.json` already at `generation_complete: true`. First run sat idle **10+ minutes**
at 1.3 GB RSS / 70 threads before being killed. CPU profile: ~36 s of work, then idle.

Thread states of the hung process:

```
main            futex_do_wait
fsfs-signal-lis unix_stream_data_wait   <-- signal listener, never shut down
asupersync-work ep_poll
asupersync-work futex_do_wait
```

**Root cause (first principles):** a signal-listener task on a unix socket is spawned outside the
command's cancelled scope, so it is never cancelled or joined when the index transaction completes.
The asupersync runtime therefore cannot quiesce and the process cannot exit. AGENTS.md states the
contract explicitly — *"structured concurrency: `Cx`, `Scope`, `region()` — no orphan tasks"* — and
this is precisely an orphan task.

**Impact:** every script, cron job, CI step, and agent harness that runs `fsfs index` hangs forever.
Data is durably committed, so this is pure lifecycle damage — which is exactly why no test caught it.

### D2 — default build cannot index (P0, no bead)

`crates/frankensearch-fsfs/Cargo.toml:13` declares `default = []`. A plain
`cargo build -p frankensearch-fsfs` therefore produces a model-free binary whose `index` command
fails hard (exit 78, `embedder_unavailable`) and creates **no index directory at all**. There is no
lexical-only fallback, so the binary is inert.

The README's documented developer path (`rch-ensure-deps.sh --models-only` → `cargo install --path
crates/frankensearch-fsfs`) therefore installs a non-functional binary, and the 60-second quick
start is not reproducible from source. Building with `--features embedded-models` **does** work.

Worth stating clearly: the *diagnostics* around this failure are excellent — the error names the
exact rebuild command, and `doctor` independently reports "models present on disk but this binary
has no Model2Vec/FastEmbed loader." The defect is that the **default is a non-functional
configuration**, not that the failure is opaque.

---

## 3. Would implementing every open bead close the gap?

**No.** 904 of 1104 beads are closed (82%) and 127 of the remainder are P0 — but four vision-level
gaps have **zero bead coverage**, and they include both blocking defects:

1. **D1** — the index hang. No bead. Invisible to unit tests by construction.
2. **D2** — default-features usability regression + README drift. No bead.
3. **Searcher parity** — `searcher.rs` and `sync_searcher.rs` implement one pipeline and have
   drifted, with the correct behaviour on *opposite sides* in different cases. Three of today's
   HIGH findings are symptoms of this one divergence. No bead; a shared conformance test would
   kill the whole class.
4. **Executable documentation** — nothing runs the README's own commands. A CI job that builds the
   binary the way the README tells a user to, runs the quick start against a fixture corpus, and
   asserts a non-empty ranked result **would have caught both D1 and D2 automatically**.

---

## 4. The honest diagnosis

The campaign has been optimizing **measurement quality far ahead of product completeness**. The
rigor is real and worth keeping: the evidence machinery has no forgeable-verdict path, the ELF
identity TOCTOU is genuinely closed, honest MISSes are recorded as MISSes, and hostile review
routinely catches overclaims within minutes — including several of mine.

But the artifacts that rigor exists to certify — activated gates, a flipped default, a working
install — do not exist yet, and while attention was on the gates, the user-facing path regressed
underneath. A project can be simultaneously *more rigorous* and *less shippable* than it was a week
ago. That is the state today, and it is fixable cheaply: D1 and D2 are small, surgical, and
high-blast-radius.

**Recommended sequencing:** D1 → D2 → executable-README gate → searcher-parity conformance test →
then resume gate certification. Fixing the four no-bead gaps costs days, not weeks, and converts
"impressive engine" into "installable product" — which is the precondition for every performance
claim being worth making.
