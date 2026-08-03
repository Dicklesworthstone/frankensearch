# ArtifactStore v4 F0: trust and threat-model contract

Status: frozen design contract for the ArtifactStore v4 series. It is not an
implementation, a migration approval, a release qualification, or a performance
claim. Implementations in F1-F4 must conform to this document or introduce a
new schema version.

The machine-readable structural contract is
[`artifactstore-v4-f0.schema.json`](../../schemas/artifactstore-v4-f0.schema.json).
It rejects unknown state values and the illegal trust/admission/decision/release
combinations that can be expressed structurally; implementations must still
validate signatures, predecessor bytes, nonce state, and policy coverage.

## Scope and non-goals

ArtifactStore v4 persists the evidence chain for one source snapshot, build,
execution, and terminal completion. It must preserve authentic failure evidence
as durably as success evidence. It does not make a result admissible, decide a
quality gate, or authorize a release by merely storing or authenticating it.

The existing v7 ArtifactObject/CampaignReport storage is integrity-only. Its
`IntegrityCheckedCampaign` wrapper proves replayed relational links, not an
external signer, admission, decision, or promotion. Existing legacy schemas
remain decode-only; v4 must neither rewrite their bytes nor reinterpret them as
authenticated.

## Independent state axes

Every v4 completion carries all four axes below. No enum may be inferred from
another axis, including on a successful process exit.

| Axis | Closed values | Meaning |
| --- | --- | --- |
| Authentication | `VerifiedReceiptChain`, `IntegrityOnly`, `UnauthenticatedLegacy` | What cryptographic authority, if any, verified the complete chain. |
| Admission | `Admitted`, `Unadmitted`, `NoDecision` | Whether the independent evidence policy admitted this exact chain. |
| Decision | `Pass`, `Miss`, `NoDecision`, `Quarantine` | The evaluator's result for an admitted scope; `Miss` remains authentic evidence. |
| Release | `Qualified`, `NotQualified` | Whether a separately configured release policy accepted the decision and coverage. |

`VerifiedReceiptChain` requires a complete source-to-completion chain signed by
trusted, non-revoked keys. `IntegrityOnly` means domain-separated hashes and
relational replay passed but no external identity was verified.
`UnauthenticatedLegacy` means historical bytes may be decoded for inspection
only. It must never become `Admitted`, `Pass`, or `Qualified` without a new v4
chain produced from independently verified source bytes.

## Principals and object chain

The principals are: the source publisher, build service, execution supervisor,
completion issuer, evidence-policy evaluator, release policy, verifier, and
reader. A single deployment may host more than one role, but a receipt always
records the principal role and signing key identity so roles cannot be silently
collapsed.

The immutable object chain is strictly ordered:

1. `Source`: exact source archive/tree bytes and source provenance.
2. `Build`: exact source object, Cargo.lock bytes, declared build command, and
   executable digest.
3. `Execution`: exact build object, supervisor-issued nonce, bounded job
   identity, fixture/query-contract identities, and start receipt.
4. `Completion`: exact execution object, terminal outcome, end receipt,
   bounded artifact index, and retention disposition.

Each object has a canonical byte representation and an identity of
`SHA-256(domain || canonical_bytes)`. The domains are exactly
`frankensearch.artifactstore.v4.source`, `.build`, `.execution`, and
`.completion`; domains must be encoded as UTF-8 bytes followed by one NUL byte.
An object references its predecessor by the predecessor's complete 32-byte
identity, never by a path, display name, or mutable run label.

The supported content hash is SHA-256 only. The supported receipt signature is
Ed25519 only. Unknown algorithms, unknown key identifiers, malformed encodings,
unknown object kinds, and unknown schema versions fail closed before any state
is classified. Key rotation is explicit: each receipt names an immutable
`key_id`; verification uses a versioned trust-root set with validity interval
and revocation state. A retired key may verify receipts issued before retirement
but may not issue a new receipt after retirement.

## Canonical bytes, nonce, and terminal rules

V4 canonical serialization is UTF-8 RFC 8785 JSON with no trailing newline.
Required fields are present even when their value is null; unknown fields are
rejected at every nesting boundary. Integers are exact JSON integers, byte
strings are lower-case hexadecimal, timestamps are signed nanoseconds from the
Unix epoch, and maps use canonical key order. Implementations must reserialize
after decoding and require byte-for-byte equality.

The supervisor issues one non-zero 128-bit nonce for one execution object. The
nonce is bound to the source, build, requested command digest, and expiration.
Its lifetime is at most 15 minutes; it is single-use and may not be refreshed,
replayed across a build, or used after expiry. A duplicate nonce is a terminal
`Quarantine`, not a retryable success path.

Completion is terminal and immutable. Terminal outcomes are `Succeeded`,
`Failed`, `Cancelled`, `TimedOut`, `Interrupted`, and `Unknown`. `Unknown` is
mandatory when a durable completion receipt cannot prove whether execution
finished. Every terminal object retains its bounded failure artifact index;
success is not permitted to erase failure, cancellation, or interruption
evidence from the same execution identity.

## Admission, decision, and release transition rules

Authentication verifies bytes and issuer authority only. Admission verifies the
policy-defined provenance, fixture, coverage, environment, and evidence
requirements. Decision evaluates the admitted evidence. Release evaluates the
decision plus release-specific coverage and approval rules.

| Authentication | Admission | Decision/release consequence |
| --- | --- | --- |
| `UnauthenticatedLegacy` | must be `Unadmitted` | only inspection; never `Pass` or `Qualified` |
| `IntegrityOnly` | `Unadmitted` or `NoDecision` | preserve/replay only; never `Qualified` |
| `VerifiedReceiptChain` | `Unadmitted` | authentic but cannot influence a gate |
| `VerifiedReceiptChain` | `NoDecision` | authentic incomplete evidence; retry predicate required |
| `VerifiedReceiptChain` | `Admitted` | evaluator may emit `Pass`, `Miss`, `NoDecision`, or `Quarantine` |
| any | any | `Qualified` requires `Admitted` plus `Pass` and a satisfied release policy |

`Miss` is authentic when its receipt chain is verified; it is not a failure of
authentication. `Quarantine` overrides `Pass` and `Qualified`. Missing
coverage, incomplete completion, source/build substitution, nonce reuse, or a
verification failure yields `NoDecision` or `Quarantine` according to the
predeclared policy; neither may silently become a retry success.

## Privacy and diagnostics

Every payload field is classified as `Public`, `Restricted`, or `Secret`.
`Secret` values (credentials, private keys, raw tokens, and raw query/document
content where the policy marks it secret) are never serialized. `Restricted`
values are represented by domain-separated digests and bounded lengths unless a
separate authorized encrypted store is named in the receipt. Logs contain only
object identities, key IDs, reason codes, bounded redacted labels, and state
transitions. They must not contain paths, raw corpus/query text, credentials, or
unbounded verifier errors.

Any unsupported privacy label, attempted downgrade from `Secret`, or missing
redaction witness is a fail-closed `Quarantine` outcome. A verifier may expose a
bounded reason code but never the rejected sensitive value.

## Mandatory conformance cases for F1-F4

The implementation must test at least these substitutions and reject each one:

- authentic-but-unadmitted evidence presented as a `Pass`;
- admitted evidence with incomplete required coverage presented as `Qualified`;
- `IntegrityOnly` or hash-only data presented as `VerifiedReceiptChain`;
- `UnauthenticatedLegacy` bytes presented as promotable;
- successful exit presented as automatic `Pass` or `Qualified`;
- `Miss` presented as unauthentic or discarded;
- one-bit mutation of each source/build/execution/completion identity;
- unknown schema, enum, algorithm, signer, nonce, or object kind;
- signature valid under a revoked/expired/wrong-role key;
- nonce reuse, cross-build reuse, expiry, and completion substitution;
- omitted terminal/failure artifact, redaction witness, or required predecessor.

F1-F4 may add strictly stronger checks, but may not weaken these rules. Any
semantic expansion requires an explicitly versioned successor contract and a
separate security and evidence-policy review.
