//! Terminal replacement authorization for the Quill lexical flip.
//!
//! Every other witness in this crate deliberately refuses to authorize the
//! replacement. [`NativeEnrichedReceiptV1::authorizes_replacement`] is a
//! `const fn` returning `false` with no serialized field to flip, and the
//! cancellation receipt makes no authorization claim at all. Each of them
//! defers to "the terminal release-gate aggregator". This module is that
//! aggregator, and [`ReplacementAuthorizationV1`] is the only type in the
//! repository whose `authorizes_replacement` can answer `true`.
//!
//! **Conformance alone still cannot authorize the flip.** [`authorize`] refuses
//! every bundle, including a complete and clean one, because it presents no QG
//! release state at all (bd-quill-e8-perf-doctrine-x4e4.15.1). Read a passing
//! conformance suite as "correct", never as "authorized".
//!
//! [`authorize_with_qg_release`] is the entry that can answer `Ok`, and only
//! against a retained release state that covers every required gate exactly
//! once with artifacts that replay against their externally retained
//! `Qg1ExpectedAuthority` references, were measured at this candidate by a
//! clean tree, are bound to an admitted machine class, are ratchet-admissible,
//! and record an `Allow` decision. Every one of those facts is read out of the
//! artifacts; no caller-supplied verdict participates. Whether such a release
//! state exists today is a question about the QG manifests, not about this
//! module: this module's job is to consume one correctly if it is presented and
//! to refuse precisely if it is not.
//!
//! # What this module is NOT
//!
//! It does not run campaigns, fuzz query ASTs, or score retrieval quality --
//! that empirical work is the flip conformance gate's. This module answers the
//! separate structural question: is the evidence for the flip *complete*,
//! *current*, and *of the right coverage class*? A campaign can be perfectly
//! green and still fail here, because green is not the same as covering the
//! surface the replacement needs.
//!
//! # Why every slot is separately named
//!
//! A validator that takes one opaque "evidence" blob refuses as a unit and can
//! never say which piece is missing. Each required slot is therefore its own
//! variant, each is proved independently load-bearing by a test that removes
//! exactly that one, and every refusal names the slot it refused for.

use serde::{Deserialize, Serialize};

use crate::GauntletError;
use crate::campaign_contract::{
    CampaignCellEvidenceV1, CampaignContractModeV1, validate_replacement_completeness,
};
use crate::comparator::QuillCancellationReceipt;
use crate::native_enriched_witness::{AcceptedCandidateBindingV1, VerifiedNativeEnrichedReceiptV1};
use crate::perf::{PerfGate, Qg1ExpectedAuthority};
use crate::perf_evidence::{EvidenceDecisionStatus, PerfEvidenceArtifact};
use crate::runner::{CampaignReport, DivergenceRegisterLedger};

/// Stable schema identity for a terminal replacement authorization.
pub const REPLACEMENT_AUTHORIZATION_SCHEMA_VERSION: &str = "quill-replacement-authorization-v1";

/// One required evidence slot in the terminal authorization.
///
/// The set is closed on purpose: adding a slot is a deliberate contract change
/// that breaks every caller, which is the correct cost for widening what the
/// flip is allowed to rest on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplacementEvidenceSlotV1 {
    /// Core Lexical V3 candidate binding derived from a real campaign report.
    CoreLexicalV3Binding,
    /// CASS-visible total contract binding for the same candidate.
    CassTotalBinding,
    /// Admissible native enriched receipt (both engines, clean provenance).
    NativeEnrichedReceipt,
    /// Method-bound Quill cancellation receipt.
    CancellationReceipt,
    /// Accepted divergence census snapshot digest.
    DivergenceCensus,
}

impl ReplacementEvidenceSlotV1 {
    /// Every slot the terminal authorization requires.
    pub const REQUIRED: [Self; 5] = [
        Self::CoreLexicalV3Binding,
        Self::CassTotalBinding,
        Self::NativeEnrichedReceipt,
        Self::CancellationReceipt,
        Self::DivergenceCensus,
    ];

    const fn slot_name(self) -> &'static str {
        match self {
            Self::CoreLexicalV3Binding => "core lexical v3 binding",
            Self::CassTotalBinding => "cass total binding",
            Self::NativeEnrichedReceipt => "native enriched receipt",
            Self::CancellationReceipt => "cancellation receipt",
            Self::DivergenceCensus => "divergence census",
        }
    }

    fn missing(self) -> GauntletError {
        GauntletError::InvalidContract {
            reason: format!(
                "replacement authorization is missing required evidence: {}",
                self.slot_name()
            ),
        }
    }
}

/// Evidence presented for terminal authorization.
///
/// Every field is `Option` so that an absent slot is a *refusable state this
/// type can represent*, rather than something a caller cannot express. The
/// fail-closed behaviour is therefore testable today, while the CASS binding
/// slot is still unbuilt.
#[derive(Debug, Clone, Copy)]
pub struct ReplacementEvidenceBundleV1<'evidence> {
    /// Canonical 40-hex candidate revision every slot must bind.
    pub candidate_source_revision: &'evidence str,
    /// The core-v3 campaign REPORT, not a pre-derived binding.
    ///
    /// Taking the report is the whole point: [`AcceptedCandidateBindingV1`] is
    /// a public struct with public fields, so a caller handed a binding slot
    /// could simply declare `CoreLexicalV3` coverage with no campaign behind
    /// it. Deriving the binding here makes the coverage class a fact read out
    /// of the report's own coverage summary instead of an assertion the caller
    /// makes about itself.
    pub core_lexical_v3: Option<&'evidence CampaignReport>,
    /// The frozen replacement campaign CELL MATRIX, not a pre-derived binding.
    ///
    /// The CASS-visible total contract has no `CampaignReport` derivation --
    /// `CampaignLexicalCoverageSummary` has no CASS variant and
    /// `from_campaign_report` hardcodes `CoreLexicalV3`. Its coverage is
    /// established a different way: by the frozen campaign matrix, whose
    /// `CassTotalV1`-profile cells `validate_replacement_completeness` checks
    /// against the frozen key set, seed bundles, and per-profile required
    /// contract mode. Taking the cells and validating them here means the
    /// caller cannot declare CASS coverage any more than it can declare
    /// core-v3 coverage.
    ///
    /// WHAT THIS SLOT DOES NOT DO, stated so the grant is not over-read: cells
    /// carry no source revision, so unlike the core report and the enriched
    /// receipt this slot cannot be candidate-bound. It proves the campaign
    /// matrix is COMPLETE, not that it was run on this candidate. That is the
    /// same class of limit as bd-drize.
    pub cass_total: Option<&'evidence [CampaignCellEvidenceV1]>,
    /// Admissible native enriched receipt.
    pub native_enriched: Option<&'evidence VerifiedNativeEnrichedReceiptV1>,
    /// Method-bound cancellation receipt.
    pub cancellation: Option<&'evidence QuillCancellationReceipt>,
    /// The divergence census LEDGER, not a digest of one.
    ///
    /// This slot previously took a `&str` digest and checked only that it was
    /// lowercase 64-hex, so any well-formed hex string satisfied it — and every
    /// grant harvested before bd-lvhfh passed a digest of nothing. A slot
    /// satisfiable by a constant is a missing receipt that reports as present.
    /// Taking the ledger applies the same derive-never-accept doctrine the
    /// core-v3 slot got in d3668b3c and the CASS slot got in e075a370: the
    /// digest is computed here, from a ledger that must first validate, and
    /// must equal the REGISTERED census state.
    pub divergence_census: Option<&'evidence DivergenceRegisterLedger>,
}

/// Why a slot that SATISFIED its own contract still could not be tied to the
/// candidate revision (bd-s1xrl).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplacementSlotUnboundReasonV1 {
    /// The evidence carries no source revision anywhere in its shape, so there
    /// is nothing to compare the candidate against. The frozen campaign matrix
    /// is this: its cells prove COMPLETENESS, and completeness is not currency.
    EvidenceCarriesNoSourceRevision,
    /// The evidence is a bare digest with no producer that binds it to a
    /// revision. The divergence census snapshot is this.
    DigestHasNoProducer,
}

/// How strongly one required slot was tied to the candidate revision.
///
/// This distinction is the whole point of the record: "the slot was satisfied"
/// and "the slot was satisfied BY THIS CANDIDATE" are different claims, and
/// before bd-s1xrl a granted authorization stated only the first while looking
/// like it stated both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "binding")]
pub enum ReplacementSlotBindingV1 {
    /// The slot's evidence names a source revision and [`authorize`] compared
    /// it against the candidate, so a mismatch refuses by this slot's name.
    CandidateBound,
    /// The slot's own contract was satisfied and nothing ties it to THIS
    /// candidate. Evidence of this class from an older candidate satisfies the
    /// slot exactly as well as evidence from the one being authorized.
    ContractOnly {
        /// Why the binding could not be made.
        reason: ReplacementSlotUnboundReasonV1,
    },
}

/// One slot's entry in a granted authorization's binding record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplacementSlotBindingRecordV1 {
    /// The required slot this entry describes.
    pub slot: ReplacementEvidenceSlotV1,
    /// What [`authorize`] was able to prove about it.
    pub binding: ReplacementSlotBindingV1,
}

/// The binding strength of each required slot, as a fact about the evidence
/// SHAPE rather than about any particular bundle.
///
/// Written as a `match` on the closed slot set on purpose: adding a slot will
/// not compile until its binding strength is decided, which is the correct cost
/// for widening what the flip rests on. Three slots are `CandidateBound`
/// because `authorize` really does compare a revision for each -- the core
/// report's producer revision, the enriched receipt's candidate AND producer,
/// and (since bd-drize) the cancellation receipt's producer revision. The other
/// two carry no revision to compare.
/// The record every grant carries, built once so the production path and the
/// tests that pin its invariants cannot drift apart.
fn slot_binding_record()
-> [ReplacementSlotBindingRecordV1; ReplacementEvidenceSlotV1::REQUIRED.len()] {
    ReplacementEvidenceSlotV1::REQUIRED.map(|slot| ReplacementSlotBindingRecordV1 {
        slot,
        binding: slot_binding_strength(slot),
    })
}

const fn slot_binding_strength(slot: ReplacementEvidenceSlotV1) -> ReplacementSlotBindingV1 {
    match slot {
        ReplacementEvidenceSlotV1::CoreLexicalV3Binding
        | ReplacementEvidenceSlotV1::NativeEnrichedReceipt
        | ReplacementEvidenceSlotV1::CancellationReceipt => {
            ReplacementSlotBindingV1::CandidateBound
        }
        ReplacementEvidenceSlotV1::CassTotalBinding => ReplacementSlotBindingV1::ContractOnly {
            reason: ReplacementSlotUnboundReasonV1::EvidenceCarriesNoSourceRevision,
        },
        ReplacementEvidenceSlotV1::DivergenceCensus => ReplacementSlotBindingV1::ContractOnly {
            reason: ReplacementSlotUnboundReasonV1::DigestHasNoProducer,
        },
    }
}

/// The seal that makes a granted authorization unforgeable INSIDE this crate
/// as well as outside it.
///
/// A private field of unit type reads as a hand-rolled `#[non_exhaustive]`,
/// and the two are not interchangeable here: `#[non_exhaustive]` only refuses
/// literal construction from OTHER crates, so every module of this one — the
/// runner included — could still mint an authorization it never earned. That
/// is precisely the forgery this type exists to refuse, so the seal is a named
/// private type rather than `()`, and the rule is stated instead of being an
/// artifact of which crate the caller happens to be in (bd-916qm).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AuthorizationSeal;

/// A granted terminal authorization for the Quill lexical replacement.
///
/// The private [`AuthorizationSeal`] field makes literal construction
/// impossible outside this module, so the only way to hold this type is to
/// have passed [`authorize`]. It is deliberately NOT `Deserialize`: a
/// deserializable authorization could be minted in a text editor, which is
/// exactly the failure the enriched receipt's private-field design already
/// refuses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReplacementAuthorizationV1 {
    /// Stable schema identity.
    pub schema_version: String,
    /// The exact candidate every slot agreed on.
    pub candidate_source_revision: String,
    /// Content address of the admitting enriched receipt.
    pub native_enriched_receipt_address: String,
    /// Canonical body hash of the admitting cancellation receipt.
    pub cancellation_body_sha256: String,
    /// Engine identity the cancellation receipt was produced against.
    pub cancellation_engine_revision: String,
    /// Accepted divergence census snapshot digest.
    pub divergence_census_sha256: String,
    /// What this grant actually proved about EACH required slot (bd-s1xrl).
    ///
    /// Serialized, and deliberately so. The two limits it records were
    /// previously stated only in doc comments on the bundle's fields, and doc
    /// comments do not travel with the artifact — this type is `Serialize`
    /// precisely so it can be read somewhere the source is not. A reader that
    /// sees one `candidate_source_revision` and five satisfied slots would
    /// otherwise have no way to learn that two of the five were never checked
    /// against that revision at all.
    ///
    /// In `ReplacementEvidenceSlotV1::REQUIRED` order and the same length, so
    /// the record cannot omit a slot: an omitted slot is exactly the unstated
    /// limit this field exists to remove, and here it is a compile error rather
    /// than a test.
    pub slot_bindings: [ReplacementSlotBindingRecordV1; ReplacementEvidenceSlotV1::REQUIRED.len()],
    /// Unconstructible from outside this module. Skipped on the wire: the seal
    /// is an in-process construction proof, and serializing it would suggest a
    /// reader could verify authorization from bytes alone, which it cannot.
    #[serde(skip)]
    sealed: AuthorizationSeal,
}

impl ReplacementAuthorizationV1 {
    /// This type, and only this type, may authorize the lexical replacement.
    ///
    /// Holding one is currently impossible outside this module: [`authorize`]
    /// refuses terminally until a validated all-required-target QG WIN release
    /// state is consumable, so this method describes the contract a future
    /// grant must satisfy rather than a grant anyone can obtain today. A
    /// caller that reaches this method has either been handed a value from
    /// inside this module or is reading a test's unreachable branch.
    ///
    /// It does NOT mean every slot was tied to the candidate revision. Read
    /// [`Self::every_slot_is_candidate_bound`] for that; it answers `false`
    /// today, and the reason is in [`Self::slot_bindings`].
    #[must_use]
    pub const fn authorizes_replacement(&self) -> bool {
        true
    }

    /// Whether EVERY required slot's evidence was tied to this candidate.
    ///
    /// `false` today: the frozen campaign matrix and the divergence census
    /// carry no source revision, so their slots are satisfied without being
    /// bound. Exposed as a question a caller can ask rather than a paragraph a
    /// caller must read, so a consumer that needs a fully-bound grant can
    /// refuse one instead of discovering the asymmetry later. The day either
    /// slot grows a producer that stamps a revision, this starts answering
    /// `true` and [`slot_binding_strength`] must change with it.
    #[must_use]
    pub fn every_slot_is_candidate_bound(&self) -> bool {
        self.slot_bindings
            .iter()
            .all(|record| record.binding == ReplacementSlotBindingV1::CandidateBound)
    }

    /// What this grant proved about one required slot.
    #[must_use]
    pub fn binding_for(&self, slot: ReplacementEvidenceSlotV1) -> ReplacementSlotBindingV1 {
        self.slot_bindings
            .iter()
            .find(|record| record.slot == slot)
            .map_or_else(|| slot_binding_strength(slot), |record| record.binding)
    }
}

/// The committed Divergence Register state a grant must be about.
///
/// Compiled in rather than read at runtime: a census loaded from a path the
/// caller chooses is a census the caller controls, which is the hole this
/// replaces rather than relocates.
const REGISTERED_DIVERGENCE_CENSUS: &str =
    include_str!("../fixtures/divergence-register-v2-live.json");

/// Digest of the registered census, derived from the committed ledger.
///
/// # Errors
///
/// Returns an error when the committed census does not parse or does not
/// validate — a state in which no authorization should be issued at all.
fn registered_divergence_census_sha256() -> Result<String, GauntletError> {
    let ledger: DivergenceRegisterLedger = serde_json::from_str(REGISTERED_DIVERGENCE_CENSUS)
        .map_err(|error| GauntletError::InvalidContract {
            reason: format!("the registered divergence census does not parse: {error}"),
        })?;
    ledger.ledger_hash()
}

fn is_canonical_git_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

// `is_lower_sha256` lived here to shape-check a caller-supplied census digest.
// bd-lvhfh removed that parameter: the census is now derived from the
// registered ledger and compared against the registered digest, so there is no
// free-form digest left to shape-check. Deleted rather than `#[allow]`ed —
// a helper kept alive by an allow is a claim that something still validates.

fn require_binding(
    binding: &AcceptedCandidateBindingV1,
    expected_mode: CampaignContractModeV1,
    candidate: &str,
    slot: ReplacementEvidenceSlotV1,
) -> Result<(), GauntletError> {
    if binding.contract_mode != expected_mode {
        return Err(GauntletError::InvalidContract {
            reason: format!(
                "{} carries {:?} coverage, which can never authorize a replacement",
                slot.slot_name(),
                binding.contract_mode
            ),
        });
    }
    if binding.candidate_source_revision != candidate {
        return Err(GauntletError::ManifestMismatch {
            reason: format!(
                "{} binds candidate {} but the authorization is for {candidate}",
                slot.slot_name(),
                binding.candidate_source_revision
            ),
        });
    }
    Ok(())
}

/// One retained QG artifact paired with the exact external expectations needed
/// to replay it outside the process that produced it.
///
/// The pairing is the whole point. A `PerfEvidenceArtifact` carrying QG-1 cells
/// cannot authenticate itself — `Qg1ExpectedAuthority` is deliberately not
/// serializable, so the expectations must arrive from whoever retained them —
/// and an artifact presented without them is refused by its own replay rather
/// than admitted on its own say-so.
#[derive(Debug, Clone, Copy)]
pub struct RetainedQgReleaseArtifactV1<'evidence> {
    /// The gate slot this artifact is presented FOR.
    ///
    /// Compared against the artifact's own `gate`, so a QG-2 artifact filed in
    /// the QG-1 slot is a named refusal rather than a silent substitution.
    pub gate: PerfGate,
    /// The sealed evidence artifact exactly as persisted.
    pub artifact: &'evidence PerfEvidenceArtifact,
    /// Externally retained QG-1 expectations for this artifact.
    ///
    /// Gates that carry no QG-1 authority present an empty slice. A QG-1
    /// artifact whose expectations are absent, foreign, or duplicated fails
    /// closed inside [`PerfEvidenceArtifact::verify_integrity_against_qg1_authorities`],
    /// which is where that judgement belongs.
    pub retained_qg1_authorities: &'evidence [&'evidence Qg1ExpectedAuthority],
}

/// The complete terminal QG release state offered for consumption.
///
/// Deliberately a set of ARTIFACTS rather than a verdict: the caller cannot
/// hand in the answer being gated on, exactly as the core-v3 and CASS slots
/// take a report and a matrix instead of a pre-derived binding. Every verdict
/// below is read out of the artifacts themselves.
#[derive(Debug, Clone, Copy)]
pub struct QgReleaseAuthorityRequestV1<'evidence> {
    /// One entry per required gate, in any order.
    pub artifacts: &'evidence [RetainedQgReleaseArtifactV1<'evidence>],
}

/// Refuse a terminal replacement authorization, naming the reason.
///
/// **This function cannot return `Ok`.** It presents no QG release state, so a
/// bundle that clears every conformance slot below still meets the terminal
/// refusal (bd-quill-e8-perf-doctrine-x4e4.15.1). Every slot is checked in full
/// and still refuses for its own reason first, so a dirty producer, a foreign
/// candidate, or an incomplete campaign matrix is reported as such.
///
/// Callers that hold a retained QG release state use
/// [`authorize_with_qg_release`]; this entry is conformance-only and is kept so
/// every existing caller compiles and keeps its exact prior behaviour.
///
/// # Errors
///
/// Refuses when any required slot is absent, when any slot carries a coverage
/// class that cannot support a replacement, when the enriched receipt is not
/// release-admissible, when the cancellation receipt does not validate, when
/// any slot binds a different candidate than the one under authorization, and
/// — after all of those pass — for the absent QG WIN release state.
pub fn authorize(
    bundle: &ReplacementEvidenceBundleV1<'_>,
) -> Result<ReplacementAuthorizationV1, GauntletError> {
    authorize_with_qg_release(bundle, None)
}

/// Authorize the replacement against conformance evidence AND a retained QG
/// release state.
///
/// This is the entry a real release consumes. [`authorize`] remains the
/// conformance-only entry and delegates here with no release state, which still
/// refuses terminally — conformance alone may not authorize the flip, and that
/// has not changed.
///
/// # Errors
///
/// Returns every refusal [`authorize`] can, and — after all of them pass —
/// refuses when the retained release state is absent, does not cover every
/// required gate exactly once, files an artifact under a gate it does not
/// belong to, cannot be replayed against its retained expectations, was not
/// produced from the candidate under authorization by a clean tree, is not
/// bound to an admitted machine class, is not ratchet-admissible, or does not
/// carry an `Allow` promotion decision.
pub fn authorize_with_qg_release(
    bundle: &ReplacementEvidenceBundleV1<'_>,
    qg_release: Option<&QgReleaseAuthorityRequestV1<'_>>,
) -> Result<ReplacementAuthorizationV1, GauntletError> {
    let candidate = bundle.candidate_source_revision;
    if !is_canonical_git_revision(candidate) {
        return Err(GauntletError::InvalidContract {
            reason: "replacement authorization requires a canonical lowercase 40-hex candidate"
                .to_owned(),
        });
    }

    // EACH SLOT IS VALIDATED COMPLETELY, IN ORDER: presence, then that slot's
    // own contract, before the next slot is touched. Cheapest and most
    // structural first; heaviest and most environment-sensitive last.
    //
    // The ordering is load-bearing rather than stylistic. The enriched
    // receipt's admissibility depends on whether the checkout that PRODUCED it
    // was clean, which no working tree here can guarantee, and the receipt
    // itself is the one artifact no unit test can mint. Validating it early
    // would make every other refusal unreachable in practice: a bundle with a
    // deliberately wrong coverage class would refuse for the missing receipt
    // instead, and the fail-closed tests would all pass for the wrong reason.
    // That is not hypothetical -- it is exactly what the first version of this
    // function did, and its own tests caught it.
    let core_report = bundle
        .core_lexical_v3
        .ok_or_else(|| ReplacementEvidenceSlotV1::CoreLexicalV3Binding.missing())?;
    // DERIVED, never accepted. from_campaign_report fails closed on seven
    // independent gates -- not passed, core-v3-but-inadmissible, rank-envelope
    // only, no declared coverage scope, non-shipping semantic contract, absent
    // provenance, and a dirty or non-canonical producer revision -- so the
    // coverage class cannot be asserted into existence by the caller.
    let core_lexical_v3 = AcceptedCandidateBindingV1::from_campaign_report(core_report)?;
    require_binding(
        &core_lexical_v3,
        CampaignContractModeV1::CoreLexicalV3,
        candidate,
        ReplacementEvidenceSlotV1::CoreLexicalV3Binding,
    )?;

    let cass_cells = bundle
        .cass_total
        .ok_or_else(|| ReplacementEvidenceSlotV1::CassTotalBinding.missing())?;
    // DERIVED, never accepted -- the same move slice 3 made for the core slot.
    // A caller cannot assert CASS coverage; it must present a campaign matrix
    // that satisfies the frozen completeness policy.
    validate_replacement_completeness(cass_cells).map_err(|error| {
        GauntletError::InvalidContract {
            reason: format!("cass total binding is not a complete frozen campaign matrix: {error}"),
        }
    })?;
    if !cass_cells
        .iter()
        .any(|cell| cell.contract_mode() == CampaignContractModeV1::CassTotalV1)
    {
        return Err(GauntletError::InvalidContract {
            reason: "cass total binding carries no CassTotalV1 campaign cell".to_owned(),
        });
    }

    let cancellation = bundle
        .cancellation
        .ok_or_else(|| ReplacementEvidenceSlotV1::CancellationReceipt.missing())?;
    cancellation.validate()?;
    // bd-drize. `validate` proves the receipt is a real, replayable, live
    // cancellation matrix; it cannot prove WHOSE. Until the body carried a
    // source revision this slot was the only required one with no candidate
    // binding to check, so a receipt observed from another source tree
    // satisfied it — measured, not supposed: two candidates produced the same
    // body_sha256 `9b81b211...` while the enriched address moved beneath it.
    if cancellation.body.producer_source_revision != candidate {
        return Err(GauntletError::ManifestMismatch {
            reason: format!(
                "{} was produced from {} but the authorization is for {candidate}",
                ReplacementEvidenceSlotV1::CancellationReceipt.slot_name(),
                cancellation.body.producer_source_revision
            ),
        });
    }

    let census_ledger = bundle
        .divergence_census
        .ok_or_else(|| ReplacementEvidenceSlotV1::DivergenceCensus.missing())?;
    // DERIVED, never accepted. ledger_hash() validates the ledger before
    // hashing it, so a malformed census cannot reach the comparison at all.
    let divergence_census_sha256 = census_ledger.ledger_hash()?;
    let registered = registered_divergence_census_sha256()?;
    if divergence_census_sha256 != registered {
        return Err(GauntletError::ManifestMismatch {
            reason: format!(
                "divergence census {divergence_census_sha256} is not the registered census \
                 {registered}"
            ),
        });
    }

    let native_enriched = bundle
        .native_enriched
        .ok_or_else(|| ReplacementEvidenceSlotV1::NativeEnrichedReceipt.missing())?;
    let receipt = native_enriched.receipt();
    if receipt.candidate.candidate_source_revision != candidate {
        return Err(GauntletError::ManifestMismatch {
            reason: format!(
                "native enriched receipt binds candidate {} but the authorization is for {candidate}",
                receipt.candidate.candidate_source_revision
            ),
        });
    }
    if receipt.producer.source_git_revision != candidate {
        return Err(GauntletError::ManifestMismatch {
            reason: format!(
                "native enriched receipt was produced by {} but the authorization is for {candidate}",
                receipt.producer.source_git_revision
            ),
        });
    }

    // LAST, and deliberately so: this is the only check whose answer depends on
    // the cleanliness of the checkout that produced the evidence.
    native_enriched.require_release_admissible()?;

    // TERMINAL, and after every conformance slot above has already passed
    // (bd-quill-e8-perf-doctrine-x4e4.15.1). Placing it here rather than first
    // is load-bearing: a bundle that fails an existing slot must still refuse
    // for THAT slot's reason, so this cannot mask a dirty producer, a foreign
    // candidate, or an incomplete campaign matrix behind a performance answer.
    require_consumable_qg_release_authority(qg_release, candidate)?;

    Ok(ReplacementAuthorizationV1 {
        schema_version: REPLACEMENT_AUTHORIZATION_SCHEMA_VERSION.to_owned(),
        candidate_source_revision: candidate.to_owned(),
        native_enriched_receipt_address: native_enriched.address().to_owned(),
        cancellation_body_sha256: cancellation.body_sha256.clone(),
        cancellation_engine_revision: cancellation.body.engine_revision.clone(),
        divergence_census_sha256,
        // Derived from the closed slot set, never from the bundle: the record
        // describes what THIS FUNCTION checked above, so a caller cannot
        // present a bundle that claims stronger binding than the code performs.
        slot_bindings: slot_binding_record(),
        sealed: AuthorizationSeal,
    })
}

/// Refuse the flip until a validated all-required-target QG WIN state can
/// actually be consumed (bd-quill-e8-perf-doctrine-x4e4.15.1).
///
/// Conformance is not authorization. Every slot above proves the candidate is
/// *correct*; none of them says anything about whether it is *fast enough*, and
/// before this refusal existed a complete conformance bundle authorized the
/// replacement while the QG matrix was absent, `NoDecision`, `Block`,
/// `Quarantine`, or measured on a machine class the gate does not apply to.
/// That is the hole this closes.
///
/// It takes ARTIFACTS, never a verdict. A verdict slot would let the caller
/// hand in the very answer being gated on, which is the failure mode this
/// module already refuses for the core and CASS slots by taking a report and a
/// matrix rather than pre-derived bindings. Every judgement below —
/// admissibility, promotion decision, candidate, machine class — is read out of
/// the artifact, and no caller-supplied string participates.
///
/// The refusal this replaced was unconditional because a persisted
/// `PerfEvidenceArtifact` carrying QG-1 cells could not be replayed outside its
/// producing process. That is now expressible: the caller presents the
/// externally retained `Qg1ExpectedAuthority` references alongside each
/// artifact, and replay authenticates against them. An absent release state
/// still refuses, so [`authorize`] behaves exactly as before.
///
/// # Errors
///
/// Refuses for an absent release state, a gate that is missing or presented
/// more than once, an artifact filed under a gate it does not belong to, a
/// replay that fails against the retained expectations (which also proves the
/// seal), evidence not produced from this candidate by a clean tree, an
/// unbound machine class, non-ratchet-admissible evidence, and any promotion
/// decision other than `Allow`.
fn require_consumable_qg_release_authority(
    request: Option<&QgReleaseAuthorityRequestV1<'_>>,
    candidate: &str,
) -> Result<(), GauntletError> {
    let Some(request) = request else {
        return Err(GauntletError::InvalidContract {
            reason: "replacement authorization requires a validated all-required-target QG WIN \
                     release state, and none was presented: conformance alone may not authorize \
                     the flip"
                .to_owned(),
        });
    };

    // Every required gate, in the normative manifest order, so a refusal always
    // names the earliest failing gate rather than whichever entry the caller
    // happened to list first.
    for gate in PerfGate::ALL {
        let mut matching = request.artifacts.iter().filter(|entry| entry.gate == gate);
        let entry = match (matching.next(), matching.next()) {
            (Some(entry), None) => entry,
            (None, _) => {
                return Err(GauntletError::InvalidContract {
                    reason: format!(
                        "QG release state is missing required gate {gate}; an all-required-target \
                         WIN cannot rest on a partial matrix"
                    ),
                });
            }
            (Some(_), Some(_)) => {
                return Err(GauntletError::InvalidContract {
                    reason: format!(
                        "QG release state presents gate {gate} more than once; a duplicated gate \
                         cannot name one release state"
                    ),
                });
            }
        };

        // A swapped artifact is a substitution, not a mislabel: refuse before
        // any of its content is read, so a foreign artifact can never be
        // partially credited to this gate.
        if entry.artifact.gate != gate {
            return Err(GauntletError::ManifestMismatch {
                reason: format!(
                    "QG release state files {} evidence under gate {gate}",
                    entry.artifact.gate
                ),
            });
        }

        // Replay against the retained expectations. This also re-derives the
        // seal, so an unsealed or edited artifact refuses here, and a QG-1
        // artifact whose expectations are absent, foreign, or duplicated fails
        // closed inside the artifact rather than being judged by this module.
        entry
            .artifact
            .verify_integrity_against_qg1_authorities(entry.retained_qg1_authorities)
            .map_err(|error| GauntletError::InvalidContract {
                reason: format!(
                    "gate {gate} evidence does not replay against its retained authorities: \
                     {error}"
                ),
            })?;

        let build = &entry.artifact.provenance.build;
        if build.git_dirty || build.git_revision != candidate {
            return Err(GauntletError::ManifestMismatch {
                reason: format!(
                    "gate {gate} evidence was measured at {}{} but the authorization is for \
                     {candidate}",
                    build.git_revision,
                    if build.git_dirty { " (dirty)" } else { "" }
                ),
            });
        }

        // Machine applicability: an artifact with no admitted runner identity
        // was never bound to a machine class the gate applies to.
        if entry.artifact.machine_class.identity().is_none() {
            return Err(GauntletError::InvalidContract {
                reason: format!(
                    "gate {gate} evidence carries no admitted machine-class identity, so its \
                     applicability to this release is unproven"
                ),
            });
        }

        // NoDecision, InvalidNull, an incomplete QG-1 incumbent screen, and
        // partial plan coverage all land here, which is why the verdict is read
        // from admissibility rather than from a status string.
        if !entry.artifact.ratchet_admissible() {
            return Err(GauntletError::InvalidContract {
                reason: format!(
                    "gate {gate} evidence is not ratchet-admissible (status {}), so it cannot \
                     carry a WIN",
                    entry.artifact.gate_status
                ),
            });
        }

        match entry.artifact.gate_decision {
            Some(EvidenceDecisionStatus::Allow) => {}
            Some(decision) => {
                return Err(GauntletError::InvalidContract {
                    reason: format!(
                        "gate {gate} evidence records decision {decision}, which can never \
                         authorize a replacement"
                    ),
                });
            }
            None => {
                return Err(GauntletError::InvalidContract {
                    reason: format!(
                        "gate {gate} evidence records no promotion decision, so no WIN was ever \
                         adjudicated"
                    ),
                });
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const OTHER_CANDIDATE: &str = "5eb995d524705ef8b17834e9ce005125179b9af2";
    /// The registered census, parsed from the same committed ledger the gate
    /// derives its expected digest from.
    fn registered_census() -> DivergenceRegisterLedger {
        serde_json::from_str(REGISTERED_DIVERGENCE_CENSUS).expect("the registered census parses")
    }

    /// A census that is WELL FORMED and validates, but is not the registered
    /// state: one extra character of prose in its register id.
    ///
    /// This is the planted negative bd-lvhfh exists for. Under the old slot it
    /// was unrepresentable — the slot took a digest, so a forgery was simply a
    /// different 64-hex string and every one of them passed.
    fn forged_census() -> DivergenceRegisterLedger {
        let mut ledger = registered_census();
        ledger.register_id.push('x');
        ledger
    }

    fn baseline_provenance() -> crate::runner::CampaignProvenance {
        crate::runner::CampaignProvenance {
            producer_build_identity_sha256: "0".repeat(64),
            cargo_lock_sha256: "1".repeat(64),
            rustc_version_verbose: "rustc 0.0.0 (authorization baseline)".to_owned(),
            rust_toolchain_channel: "nightly-0000-00-00".to_owned(),
            unicode_version: "0.0.0".to_owned(),
            unicode_normalization_version: "0.0.0".to_owned(),
            unicode_normalization_table_version: "0.0.0".to_owned(),
            query_generator_id: "authorization-baseline".to_owned(),
            query_generator_schema_version: 1,
            query_seed: 0,
            query_source_identity_sha256: "2".repeat(64),
            query_profile_sha256: "3".repeat(64),
            analyzer_contract_hash: "4".repeat(64),
            schema_contract_hash: "5".repeat(64),
            corpus_manifest_hash: "6".repeat(64),
            query_manifest_hash: "7".repeat(64),
            corpus_seed: None,
        }
    }

    /// THE REAL PINNED ARTIFACT, untouched. It is `passed: true` and
    /// rank-envelope-only, which is exactly the object the acceptance says can
    /// never authorize.
    fn pinned_report() -> CampaignReport {
        crate::runner::load_pinned_campaign_report_v8().expect("the pinned V8 campaign report")
    }

    /// The pinned fixture repaired along every axis `from_campaign_report`
    /// gates. Repairing the REAL report rather than hand-building one keeps the
    /// positive case honest: if the gates change, this stops satisfying them
    /// and the tests fail loudly instead of drifting.
    fn accepted_report() -> CampaignReport {
        let mut report = pinned_report();
        report.lexical_coverage = crate::runner::CampaignLexicalCoverageSummary::CoreLexicalV3 {
            subject: Box::new(crate::runner::LexicalSideCoverageCounts::default()),
            oracle: Box::new(crate::runner::LexicalSideCoverageCounts::default()),
            admissible: true,
        };
        report.semantic_contract = crate::runner::SemanticContract::shipping_default();
        report.provenance = Some(baseline_provenance());
        report.producer_build_identity.source_git_dirty = false;
        report
    }

    /// The complete frozen campaign matrix, built the way the contract's own
    /// tests build it: every frozen key, each with the contract mode its
    /// profile requires and the seed bundle its slot pins.
    fn cass_cells() -> Vec<CampaignCellEvidenceV1> {
        crate::campaign_contract::frozen_replacement_cell_keys()
            .into_iter()
            .map(|key| {
                let mode = match key.campaign_profile() {
                    crate::campaign_contract::CampaignProfileV1::ShippingDefaultCoreV3 => {
                        CampaignContractModeV1::CoreLexicalV3
                    }
                    crate::campaign_contract::CampaignProfileV1::CassTotalV1 => {
                        CampaignContractModeV1::CassTotalV1
                    }
                };
                let seeds =
                    crate::campaign_contract::frozen_replacement_seed_bundle(key.seed_slot());
                CampaignCellEvidenceV1::new(
                    key,
                    crate::campaign_contract::CampaignEvidenceRole::BuiltInEvidence(
                        crate::campaign_contract::BuiltInEvidenceBindingV1::new(
                            crate::campaign_contract::CampaignSha256V1::parse(&"a".repeat(64))
                                .expect("strict lower-case hex"),
                            crate::campaign_contract::CampaignSha256V1::parse(&"b".repeat(64))
                                .expect("strict lower-case hex"),
                        ),
                    ),
                    mode,
                    seeds,
                )
            })
            .collect()
    }

    /// A bundle complete in every slot EXCEPT the enriched receipt, which no
    /// unit test can mint: it needs a live both-engines run from a clean
    /// checkout. Every refusal below is proved against a bundle whose only
    /// other absence is that one slot, and each test changes exactly one
    /// further thing.
    fn bundle<'a>(
        core: &'a CampaignReport,
        cass: &'a [CampaignCellEvidenceV1],
        cancellation: &'a QuillCancellationReceipt,
        candidate: &'a str,
        census: &'a DivergenceRegisterLedger,
    ) -> ReplacementEvidenceBundleV1<'a> {
        ReplacementEvidenceBundleV1 {
            candidate_source_revision: candidate,
            core_lexical_v3: Some(core),
            cass_total: Some(cass),
            native_enriched: None,
            cancellation: Some(cancellation),
            divergence_census: Some(census),
        }
    }

    fn refusal(bundle: &ReplacementEvidenceBundleV1<'_>) -> String {
        authorize(bundle)
            .expect_err("this bundle must never authorize a replacement")
            .to_string()
    }

    /// The LIVE cancellation matrix, re-sealed onto the candidate under test.
    ///
    /// The observation is real — every phase is exercised against a live Quill
    /// index and the re-seal re-runs `validate`, so nothing here can turn a
    /// broken matrix into an accepted one. Only the producer revision is
    /// restated, and it has to be: these bundles authorize a HYPOTHETICAL
    /// candidate (the pinned fixture's producer), while a receipt observed here
    /// necessarily records the revision THIS test binary was built from. Left
    /// live, bd-drize's own check would refuse first and mask every refusal
    /// below — the exact vacuity de006886 repaired in this function, which is
    /// why the fix ships with this helper rather than after it.
    async fn cancellation_receipt_for(
        cx: &asupersync::Cx,
        candidate: &str,
    ) -> QuillCancellationReceipt {
        let observed = crate::comparator::observe_live_quill_cancellation_receipt(cx)
            .await
            .expect("observe the live Quill cancellation matrix");
        let mut body = observed.body;
        body.producer_source_revision = candidate.to_owned();
        QuillCancellationReceipt::seal(body)
            .expect("re-seal the live matrix onto the candidate under test")
    }

    #[test]
    fn every_required_slot_is_independently_load_bearing() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let core = accepted_report();
            let candidate = core.producer_build_identity.source_git_revision.clone();
            let cancellation = cancellation_receipt_for(&cx, &candidate).await;
            let cass = cass_cells();
            let census = registered_census();
            let complete = bundle(&core, &cass, &cancellation, &candidate, &census);

            let removals: [(ReplacementEvidenceSlotV1, ReplacementEvidenceBundleV1<'_>); 4] = [
                (
                    ReplacementEvidenceSlotV1::CoreLexicalV3Binding,
                    ReplacementEvidenceBundleV1 {
                        core_lexical_v3: None,
                        ..complete
                    },
                ),
                (
                    ReplacementEvidenceSlotV1::CassTotalBinding,
                    ReplacementEvidenceBundleV1 {
                        cass_total: None,
                        ..complete
                    },
                ),
                (
                    ReplacementEvidenceSlotV1::CancellationReceipt,
                    ReplacementEvidenceBundleV1 {
                        cancellation: None,
                        ..complete
                    },
                ),
                (
                    ReplacementEvidenceSlotV1::DivergenceCensus,
                    ReplacementEvidenceBundleV1 {
                        divergence_census: None,
                        ..complete
                    },
                ),
            ];
            for (slot, emptied) in removals {
                let message = refusal(&emptied);
                assert!(
                    message.contains(slot.slot_name()),
                    "removing {slot:?} must refuse naming that slot, got: {message}"
                );
            }

            let message = refusal(&complete);
            assert!(
                message.contains(ReplacementEvidenceSlotV1::NativeEnrichedReceipt.slot_name()),
                "an absent enriched receipt must refuse naming that slot, got: {message}"
            );
        });
    }

    /// THE ACCEPTANCE CLAUSE, PROVED AGAINST THE REAL ARTIFACT rather than a
    /// straw man: "a generic passed `CampaignReport` or rank-envelope-only
    /// coverage can never authorize the flip". The only committed report in
    /// this repository passed its own campaign, and the aggregator refuses it.
    ///
    /// Before slice 3 this was unprovable here, because the caller handed over
    /// a pre-derived binding and could simply declare core-v3 coverage.
    #[test]
    fn the_real_passed_rank_envelope_report_can_never_authorize() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let pinned = pinned_report();
            assert!(
                pinned.passed,
                "the fixture must really be a PASSED report, or this test proves nothing"
            );
            let candidate = pinned.producer_build_identity.source_git_revision.clone();
            let cancellation = cancellation_receipt_for(&cx, &candidate).await;
            let cass = cass_cells();
            let census = registered_census();

            let message = refusal(&bundle(&pinned, &cass, &cancellation, &candidate, &census));
            assert!(
                message.contains("rank-envelope"),
                "a passed rank-envelope-only report must refuse by coverage, got: {message}"
            );
        });
    }

    /// Each gate `from_campaign_report` owns is separately load-bearing, proved
    /// by breaking the repaired report one axis at a time. Without this, a
    /// single surviving gate could carry all of them.
    #[test]
    fn each_report_gate_is_independently_load_bearing() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let candidate = accepted_report()
                .producer_build_identity
                .source_git_revision
                .clone();
            let cancellation = cancellation_receipt_for(&cx, &candidate).await;
            let cass = cass_cells();
            let census = registered_census();

            let mut not_passed = accepted_report();
            not_passed.passed = false;
            assert!(
                refusal(&bundle(
                    &not_passed,
                    &cass,
                    &cancellation,
                    &candidate,
                    &census
                ))
                .contains("did not pass"),
                "a failed campaign must refuse"
            );

            let mut dirty = accepted_report();
            dirty.producer_build_identity.source_git_dirty = true;
            assert!(
                refusal(&bundle(&dirty, &cass, &cancellation, &candidate, &census))
                    .contains("dirty"),
                "a dirty-produced report must refuse"
            );

            let mut no_provenance = accepted_report();
            no_provenance.provenance = None;
            assert!(
                refusal(&bundle(
                    &no_provenance,
                    &cass,
                    &cancellation,
                    &candidate,
                    &census
                ))
                .contains("provenance"),
                "a provenance-free report must refuse"
            );

            let mut inadmissible = accepted_report();
            inadmissible.lexical_coverage =
                crate::runner::CampaignLexicalCoverageSummary::CoreLexicalV3 {
                    subject: Box::new(crate::runner::LexicalSideCoverageCounts::default()),
                    oracle: Box::new(crate::runner::LexicalSideCoverageCounts::default()),
                    admissible: false,
                };
            assert!(
                refusal(&bundle(
                    &inadmissible,
                    &cass,
                    &cancellation,
                    &candidate,
                    &census
                ))
                .contains("not admissible"),
                "core-v3 coverage that is not admissible must refuse"
            );
        });
    }

    #[test]
    fn evidence_from_a_different_candidate_can_never_authorize() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let core = accepted_report();
            let candidate = core.producer_build_identity.source_git_revision.clone();
            let cancellation = cancellation_receipt_for(&cx, &candidate).await;

            // NOTE ON WHAT MOVED: the CASS slot can no longer be candidate-
            // stale, because it is now a campaign CELL MATRIX and cells carry
            // no source revision. That is a real limit of this slot, recorded
            // on the bundle type, not something to fake an assertion about.
            // Its refusal property is COMPLETENESS, proved separately below.
            //
            // THE STALE-EVIDENCE HOLE still applies to the slots that DO carry
            // a revision: an authorization for a candidate the core report does
            // not describe.
            let cass = cass_cells();
            let census = registered_census();
            let message = refusal(&bundle(
                &core,
                &cass,
                &cancellation,
                OTHER_CANDIDATE,
                &census,
            ));
            assert!(
                message.contains(&candidate),
                "a report from another candidate must refuse naming it, got: {message}"
            );
        });
    }

    /// bd-drize, THE SLOT THAT COULD NOT BIND A CANDIDATE.
    ///
    /// The receipt below is fully valid — observed live, every phase covered,
    /// re-sealed so its content address matches its body — and describes a
    /// DIFFERENT source revision than the one under authorization. Before the
    /// body carried a producer revision this bundle authorized, because no
    /// field of the receipt could disagree with the candidate.
    ///
    /// bd-s1xrl N1. The record covers the closed slot set exactly once each and
    /// states the two limits by name.
    ///
    /// Omission is a compile error rather than a test failure — the field is an
    /// array over `REQUIRED` — which is the stronger guarantee, so what this
    /// pins is the part the type cannot: that no slot appears twice, that the
    /// three bound slots really are the three `authorize` compares a revision
    /// for, and that the two unbound ones carry the reason they are unbound
    /// rather than a generic absence.
    #[test]
    fn the_binding_record_covers_every_required_slot_once_and_names_both_limits() {
        let record = slot_binding_record();
        for slot in ReplacementEvidenceSlotV1::REQUIRED {
            let recorded = record.iter().filter(|entry| entry.slot == slot).count();
            assert_eq!(
                recorded, 1,
                "{slot:?} appears {recorded} times; a duplicate means some required slot is \
                 unrecorded while the array length still checks out"
            );
        }

        for slot in [
            ReplacementEvidenceSlotV1::CoreLexicalV3Binding,
            ReplacementEvidenceSlotV1::NativeEnrichedReceipt,
            ReplacementEvidenceSlotV1::CancellationReceipt,
        ] {
            assert_eq!(
                slot_binding_strength(slot),
                ReplacementSlotBindingV1::CandidateBound,
                "{slot:?} is compared against the candidate and must say so"
            );
        }
        assert_eq!(
            slot_binding_strength(ReplacementEvidenceSlotV1::CassTotalBinding),
            ReplacementSlotBindingV1::ContractOnly {
                reason: ReplacementSlotUnboundReasonV1::EvidenceCarriesNoSourceRevision
            }
        );
        assert_eq!(
            slot_binding_strength(ReplacementEvidenceSlotV1::DivergenceCensus),
            ReplacementSlotBindingV1::ContractOnly {
                reason: ReplacementSlotUnboundReasonV1::DigestHasNoProducer
            }
        );
    }

    /// The accessor answers the question the record exists to make askable, and
    /// answers it `false` today. A consumer that needs a fully-bound grant can
    /// therefore refuse one rather than discover the asymmetry downstream.
    #[test]
    fn a_grant_reports_that_not_every_slot_is_candidate_bound() {
        let grant = ReplacementAuthorizationV1 {
            schema_version: REPLACEMENT_AUTHORIZATION_SCHEMA_VERSION.to_owned(),
            candidate_source_revision: OTHER_CANDIDATE.to_owned(),
            native_enriched_receipt_address: "a".repeat(64),
            cancellation_body_sha256: "b".repeat(64),
            cancellation_engine_revision: "quill-engine-131073".to_owned(),
            divergence_census_sha256: registered_divergence_census_sha256()
                .expect("registered census digest"),
            slot_bindings: slot_binding_record(),
            sealed: AuthorizationSeal,
        };

        assert!(grant.authorizes_replacement());
        assert!(
            !grant.every_slot_is_candidate_bound(),
            "two slots carry no source revision, so this cannot be true yet"
        );
        assert_eq!(
            grant.binding_for(ReplacementEvidenceSlotV1::CancellationReceipt),
            ReplacementSlotBindingV1::CandidateBound
        );
        assert_eq!(
            grant.binding_for(ReplacementEvidenceSlotV1::CassTotalBinding),
            ReplacementSlotBindingV1::ContractOnly {
                reason: ReplacementSlotUnboundReasonV1::EvidenceCarriesNoSourceRevision
            }
        );

        // The record travels with the artifact, which is the entire point.
        let encoded = serde_json::to_string(&grant).expect("a grant serializes");
        assert!(encoded.contains("slot_bindings"), "{encoded}");
        assert!(encoded.contains("contract_only"), "{encoded}");
        assert!(
            encoded.contains("evidence_carries_no_source_revision"),
            "{encoded}"
        );
    }

    /// bd-s1xrl N3. EVERY `CandidateBound` LABEL IS BACKED BY A REFUSAL.
    ///
    /// A label nobody can falsify is decoration. For each slot the record calls
    /// `CandidateBound` and a unit test can mint, rebinding that slot's evidence
    /// to another revision must make `authorize` refuse BY THAT SLOT'S NAME.
    ///
    /// THE ENRICHED SLOT IS THE STATED ASYMMETRY: it is also `CandidateBound`
    /// and `authorize` compares both its candidate and its producer revision,
    /// but no unit test can mint one — it requires a live both-engines run from
    /// a clean checkout — so that leg is covered by the live integration lane
    /// (`tests/native_enriched_witness_live.rs`) and not here. Saying so is the
    /// point; a test that quietly skipped it would leave the strongest slot the
    /// least proved.
    #[test]
    fn every_candidate_bound_slot_is_backed_by_a_refusal() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let cells = cass_cells();
            let census = registered_census();
            let core = accepted_report();
            let candidate = core.producer_build_identity.source_git_revision.clone();

            for slot in ReplacementEvidenceSlotV1::REQUIRED {
                if slot_binding_strength(slot) != ReplacementSlotBindingV1::CandidateBound {
                    continue;
                }
                let message = match slot {
                    ReplacementEvidenceSlotV1::CoreLexicalV3Binding => {
                        let mut stale = accepted_report();
                        stale.producer_build_identity.source_git_revision =
                            OTHER_CANDIDATE.to_owned();
                        let cancellation = cancellation_receipt_for(&cx, &candidate).await;
                        refusal(&bundle(&stale, &cells, &cancellation, &candidate, &census))
                    }
                    ReplacementEvidenceSlotV1::CancellationReceipt => {
                        let foreign = cancellation_receipt_for(&cx, OTHER_CANDIDATE).await;
                        refusal(&bundle(&core, &cells, &foreign, &candidate, &census))
                    }
                    // Covered by the live lane; see this test's doc comment.
                    ReplacementEvidenceSlotV1::NativeEnrichedReceipt => continue,
                    ReplacementEvidenceSlotV1::CassTotalBinding
                    | ReplacementEvidenceSlotV1::DivergenceCensus => {
                        unreachable!("{slot:?} is ContractOnly and cannot reach this arm")
                    }
                };
                assert!(
                    message.contains(slot.slot_name()),
                    "{slot:?} is labelled CandidateBound but rebinding it did not refuse by \
                     its own name, got: {message}"
                );
                assert!(
                    message.contains(OTHER_CANDIDATE),
                    "{slot:?}'s refusal must name the foreign revision, got: {message}"
                );
            }
        });
    }

    /// bd-s1xrl N2. THE `ContractOnly` LABEL IS OBSERVED, NOT DECLARED.
    ///
    /// One unchanged frozen campaign matrix satisfies its slot under two
    /// DIFFERENT candidates. Both bundles differ only in the candidate — the
    /// core report's producer revision and the cancellation receipt are rebound
    /// to each — and both reach the enriched slot, which means both got PAST
    /// the CASS slot. That is what "satisfied but not bound" means, stated as a
    /// behaviour rather than as a doc comment.
    ///
    /// Asserting that neither refusal NAMES the CASS slot is the load-bearing
    /// half: a test that only checked "both refuse" would pass just as happily
    /// if the CASS slot rejected both for a reason of its own.
    ///
    /// This is a tripwire, not a freeze. If the cells ever grow a source
    /// revision and start discriminating, this test goes red and
    /// `slot_binding_strength` must be updated in the same change — which is
    /// precisely the coupling the grant's record exists to force.
    #[test]
    fn one_cass_matrix_satisfies_its_slot_under_two_different_candidates() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let cells = cass_cells();

            let first_core = accepted_report();
            let first_candidate = first_core
                .producer_build_identity
                .source_git_revision
                .clone();
            let mut second_core = accepted_report();
            second_core.producer_build_identity.source_git_revision = OTHER_CANDIDATE.to_owned();
            assert_ne!(
                first_candidate, OTHER_CANDIDATE,
                "the two candidates must differ, or this test proves nothing"
            );

            let first_cancellation = cancellation_receipt_for(&cx, &first_candidate).await;
            let second_cancellation = cancellation_receipt_for(&cx, OTHER_CANDIDATE).await;

            for (label, core, cancellation, candidate) in [
                (
                    "first",
                    &first_core,
                    &first_cancellation,
                    first_candidate.as_str(),
                ),
                (
                    "second",
                    &second_core,
                    &second_cancellation,
                    OTHER_CANDIDATE,
                ),
            ] {
                let message = refusal(&bundle(
                    core,
                    &cells,
                    cancellation,
                    candidate,
                    &registered_census(),
                ));
                assert!(
                    message.contains(ReplacementEvidenceSlotV1::NativeEnrichedReceipt.slot_name()),
                    "the {label} candidate must reach the enriched slot, got: {message}"
                );
                assert!(
                    !message.contains(ReplacementEvidenceSlotV1::CassTotalBinding.slot_name()),
                    "the {label} candidate must not be refused by the CASS slot, got: {message}"
                );
            }
        });
    }

    /// The refusal is asserted BY SLOT NAME and by BOTH revisions, not merely
    /// as "some error": every other slot here binds the candidate correctly, so
    /// an implementation that refused for an unrelated reason would still turn
    /// this test green. It is placed after the census check in intent but
    /// reached before it, so nothing later can be masking the result.
    #[test]
    fn a_cancellation_receipt_from_another_source_revision_can_never_authorize() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let core = accepted_report();
            let candidate = core.producer_build_identity.source_git_revision.clone();
            let cass = cass_cells();
            let census = registered_census();
            assert_ne!(
                candidate, OTHER_CANDIDATE,
                "the two revisions must differ, or this test proves nothing"
            );

            let foreign = cancellation_receipt_for(&cx, OTHER_CANDIDATE).await;
            foreign
                .validate()
                .expect("the foreign receipt must be VALID: the defect is whose it is, not what");

            let message = refusal(&bundle(&core, &cass, &foreign, &candidate, &census));
            assert!(
                message.contains(ReplacementEvidenceSlotV1::CancellationReceipt.slot_name()),
                "a foreign cancellation receipt must refuse naming that slot, got: {message}"
            );
            assert!(
                message.contains(OTHER_CANDIDATE) && message.contains(&candidate),
                "the refusal must name the revision it was produced from AND the one under \
                 authorization, got: {message}"
            );

            // THE OTHER HALF, or the assertion above would also pass for a
            // validator that refused every cancellation receipt: the same
            // matrix, bound to the candidate, gets past this slot and refuses
            // only for the enriched receipt no unit test can mint.
            let matching = cancellation_receipt_for(&cx, &candidate).await;
            let message = refusal(&bundle(&core, &cass, &matching, &candidate, &census));
            assert!(
                message.contains(ReplacementEvidenceSlotV1::NativeEnrichedReceipt.slot_name()),
                "a matching receipt must pass its slot and refuse later, got: {message}"
            );
        });
    }

    /// The binding is read from the BUILD, never from a caller: a live receipt
    /// observed here names the revision this test binary was compiled from.
    /// Without this, `observe_live_quill_cancellation_receipt` could grow a
    /// candidate parameter and every test above would still pass while the
    /// receipt attested whatever its caller asked for.
    #[test]
    fn a_live_cancellation_receipt_names_the_revision_it_was_built_from() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let observed = crate::comparator::observe_live_quill_cancellation_receipt(&cx)
                .await
                .expect("observe the live Quill cancellation matrix");
            let compiled = crate::artifact::GauntletProducerBuildIdentity::compiled()
                .expect("this binary has a build-sealed producer identity");
            assert_eq!(
                observed.body.producer_source_revision, compiled.source_git_revision,
                "the receipt must name the source revision of the build that produced it"
            );
            assert!(
                is_canonical_git_revision(&observed.body.producer_source_revision),
                "the bound revision must be a canonical lowercase 40-hex git revision"
            );
        });
    }

    /// The CASS slot's own refusal property: an INCOMPLETE frozen matrix
    /// cannot authorize. This is what replaced "the caller declared CASS
    /// coverage" -- the caller must now present a matrix that satisfies the
    /// frozen completeness policy, and dropping any single cell refuses.
    #[test]
    fn an_incomplete_cass_campaign_matrix_can_never_authorize() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let core = accepted_report();
            let candidate = core.producer_build_identity.source_git_revision.clone();
            let cancellation = cancellation_receipt_for(&cx, &candidate).await;

            let census = registered_census();
            let complete = cass_cells();
            assert!(
                complete
                    .iter()
                    .any(|cell| cell.contract_mode() == CampaignContractModeV1::CassTotalV1),
                "the frozen matrix must really contain a CassTotalV1 cell, or this proves nothing"
            );

            // Drop exactly one cell.
            let mut incomplete = complete.clone();
            incomplete.pop();
            let message = refusal(&bundle(
                &core,
                &incomplete,
                &cancellation,
                &candidate,
                &census,
            ));
            assert!(
                message.contains("complete frozen campaign matrix"),
                "an incomplete CASS matrix must refuse by completeness, got: {message}"
            );
        });
    }

    #[test]
    fn a_non_canonical_candidate_and_census_digest_are_refused() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let core = accepted_report();
            let candidate = core.producer_build_identity.source_git_revision.clone();
            let cancellation = cancellation_receipt_for(&cx, &candidate).await;
            let cass = cass_cells();
            let census = registered_census();

            let short_candidate = ReplacementEvidenceBundleV1 {
                candidate_source_revision: "702dee9a",
                ..bundle(&core, &cass, &cancellation, &candidate, &census)
            };
            assert!(
                refusal(&short_candidate).contains("40-hex"),
                "an abbreviated candidate revision must refuse"
            );

            // THE PLANTED NEGATIVE bd-lvhfh EXISTS FOR: a census that is WELL
            // FORMED and validates, but is not the registered state. Under the
            // old slot this was unrepresentable -- the slot took a digest, so
            // a forgery was just a different 64-hex string and every one of
            // them passed. The forged ledger must refuse, and must refuse by
            // naming the mismatch rather than by failing to parse.
            let forged = forged_census();
            let forged_bundle = ReplacementEvidenceBundleV1 {
                divergence_census: Some(&forged),
                ..bundle(&core, &cass, &cancellation, &candidate, &census)
            };
            let message = refusal(&forged_bundle);
            assert!(
                message.contains("is not the registered census"),
                "a forged census must refuse against the registered state, got: {message}"
            );
        });
    }

    /// PLANTED NEGATIVE: no release state at all.
    ///
    /// The refusal must now be about ABSENT INPUT, not about consumption being
    /// impossible. The old message claimed verified QG evidence "cannot be
    /// replayed outside its producing process"; that claim is what this slice
    /// removed, and a gate that keeps asserting it after gaining the ability to
    /// replay would be refusing for a reason that is no longer true.
    #[test]
    fn an_absent_qg_release_state_refuses_for_absence_rather_than_impossibility() {
        let candidate = "a".repeat(40);
        let error = require_consumable_qg_release_authority(None, &candidate)
            .expect_err("an absent release state can never authorize")
            .to_string();
        assert!(
            error.contains("none was presented"),
            "the refusal must name the absent input, got: {error}"
        );
        assert!(
            !error.contains("cannot be replayed outside its producing process"),
            "the gate must no longer claim consumption is impossible, got: {error}"
        );
    }

    /// PLANTED NEGATIVE: a release state that covers no gate at all.
    ///
    /// Refusal names the FIRST required gate in normative order, so a partial
    /// matrix cannot be reported as a generic incompleteness.
    #[test]
    fn an_empty_qg_release_set_refuses_naming_the_first_missing_gate() {
        let candidate = "a".repeat(40);
        let request = QgReleaseAuthorityRequestV1 { artifacts: &[] };
        let error = require_consumable_qg_release_authority(Some(&request), &candidate)
            .expect_err("an empty release set can never authorize")
            .to_string();
        assert!(
            error.contains(PerfGate::Qg1.label()) && error.contains("missing required gate"),
            "an empty set must refuse for the first required gate, got: {error}"
        );
    }

    /// THE MODEL SEAM, without activating any manifest: the required coverage
    /// is the whole normative gate set, and the refusal walks it in that order.
    ///
    /// A positive grant cannot be written here, and the reason is a real limit
    /// rather than an omission: a genuine complete release set needs ten sealed
    /// `PerfEvidenceArtifact`s each carrying an admitted runner identity, and no
    /// unit test in this crate can mint even one — the same limit that keeps the
    /// enriched-receipt slot untestable above. Asserting per-gate independence
    /// would need nine artifacts per case, so this pins what is genuinely
    /// checkable and claims nothing further.
    #[test]
    fn the_required_release_coverage_is_the_whole_normative_gate_set() {
        assert_eq!(
            PerfGate::ALL.len(),
            10,
            "the release state must cover QG-1 through QG-10"
        );
        let candidate = "a".repeat(40);
        let request = QgReleaseAuthorityRequestV1 { artifacts: &[] };
        let error = require_consumable_qg_release_authority(Some(&request), &candidate)
            .expect_err("an empty release set can never authorize")
            .to_string();
        assert!(
            error.contains(PerfGate::Qg1.label()),
            "refusal names the first gate in normative order, got: {error}"
        );
    }

    /// The terminal gate must never mask an earlier slot's refusal, which is
    /// the precedence the conformance slots were ordered for. A bundle missing
    /// the CASS matrix refuses for THAT slot even when a release state is
    /// present and itself unsatisfiable.
    #[test]
    fn the_qg_release_gate_never_masks_an_earlier_slot_refusal() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let core = accepted_report();
            let candidate = core.producer_build_identity.source_git_revision.clone();
            let cancellation = cancellation_receipt_for(&cx, &candidate).await;
            let cass = cass_cells();
            let census = registered_census();
            let complete = bundle(&core, &cass, &cancellation, &candidate, &census);
            let request = QgReleaseAuthorityRequestV1 { artifacts: &[] };

            let emptied = ReplacementEvidenceBundleV1 {
                cass_total: None,
                ..complete
            };
            let masked = authorize_with_qg_release(&emptied, Some(&request))
                .expect_err("an incomplete bundle can never authorize")
                .to_string();
            assert!(
                masked.contains(ReplacementEvidenceSlotV1::CassTotalBinding.slot_name()),
                "the earlier slot must still refuse first, got: {masked}"
            );
            assert!(
                !masked.contains("QG release state"),
                "the terminal gate must not answer for an earlier slot, got: {masked}"
            );

            // The conformance-only entry is exactly the release-bearing entry
            // with no release state, so its behaviour cannot drift.
            assert_eq!(
                authorize(&complete)
                    .expect_err("a bundle without an enriched receipt can never authorize")
                    .to_string(),
                authorize_with_qg_release(&complete, None)
                    .expect_err("a bundle without an enriched receipt can never authorize")
                    .to_string(),
                "authorize must delegate without changing behaviour"
            );
        });
    }
}
