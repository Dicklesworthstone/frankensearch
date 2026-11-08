# fsfs Root Discovery and Exclusion Precedence Contract v1

Issue: `bd-2hz.2.1`  
Parent: `bd-2hz.2`

## Goal

Define deterministic traversal roots and exclusion precedence for heterogeneous machines, including explicit safety semantics for symlinks, loops, and mount boundaries.

## Normative Terms

- `MUST`: hard requirement
- `SHOULD`: expected default
- `MUST NOT`: forbidden behavior

## Default Root Policy

Default roots are home-centric:

- user home directories and explicit user-provided roots under home scope
- no implicit scanning of global/system roots by default

System/global roots (`/etc`, `/proc`, `/sys`, `/dev`, `/run`, `/var`) require explicit opt-in.

## Override Modes

`fsfs` MUST support:

- `strict`: safety-first mode; deny on ambiguity and do not cross safety boundaries
- `permissive`: allows explicit include overrides within bounded safety limits

Both modes MUST emit auditable reason codes for final decisions.

## Explicit Root Exclusion Scope

Bare component exclusions (such as `build`, `target`, or a custom directory name)
apply to the selected root itself and its descendants, not to unrelated ancestors.
Thus `/home/user/build/notes` is an eligible root, while a selected root named
`build` and a `build` directory inside `notes` remain excluded. Indexing and watch
discovery MUST use the actual traversal root for this boundary. Custom path and
glob exclusions still match the full path; explicit root selection does not
override those rules. Scope matching MUST use path components, reject parent
traversal below the root, and preserve the existing symlink and mount guards.

## Deterministic Precedence Order

Rule precedence MUST be applied in this exact order:

1. `hard_deny`
2. `explicit_exclude`
3. `explicit_include`
4. `fsfs_config_exclude`
5. `fsfs_config_include`
6. `gitignore`
7. `dot_ignore`
8. `system_exclude`
9. `default_root`

## Traversal Safety Guards

Required guardrails:

- symlink policy (`no_follow` or bounded follow mode)
- loop detection MUST be enabled; loop detections MUST be excluded
- mount-boundary policy (`stay_on_device` by default)
- bounded symlink depth and bounded mount hops

Safety violations MUST produce guard events with reason codes.

## Required Decision Fields

Every discovery decision MUST include:

- `path`
- `override_mode`
- `rules_evaluated[]` with source/match/effect
- `final_decision`
- `reason_code`
- `symlink_detected`
- `mount_crossing`
- `loop_detected`

## Validation Artifacts

- `schemas/fsfs-root-discovery-v1.schema.json`
- `schemas/fixtures/fsfs-root-discovery-contract-v1.json`
- `schemas/fixtures/fsfs-root-discovery-decision-v1.json`
- `schemas/fixtures/fsfs-root-discovery-guard-event-v1.json`
- `schemas/fixtures-invalid/fsfs-root-discovery-invalid-*.json`
- `scripts/check_fsfs_root_discovery_contract.sh`

## Validation Command

```bash
scripts/check_fsfs_root_discovery_contract.sh --mode all
```
