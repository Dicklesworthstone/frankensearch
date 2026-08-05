//! Terminal replacement authorization for the Quill lexical flip.
//!
//! Every other witness in this crate deliberately refuses to authorize the
//! replacement. [`NativeEnrichedReceiptV1::authorizes_replacement`] is a
//! `const fn` returning `false` with no serialized field to flip, and the
//! cancellation receipt makes no authorization claim at all. Each of them
//! defers to "the terminal release-gate aggregator". This module is that
//! aggregator, and it is the only type in the repository whose
//! `authorizes_replacement` can answer `true`.
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
use crate::runner::CampaignReport;

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
    /// Lowercase 64-hex digest of the accepted divergence census snapshot.
    pub divergence_census_sha256: Option<&'evidence str>,
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
    /// Unconstructible from outside this module. Skipped on the wire: the seal
    /// is an in-process construction proof, and serializing it would suggest a
    /// reader could verify authorization from bytes alone, which it cannot.
    #[serde(skip)]
    sealed: AuthorizationSeal,
}

impl ReplacementAuthorizationV1 {
    /// This type, and only this type, may authorize the lexical replacement.
    #[must_use]
    pub const fn authorizes_replacement(&self) -> bool {
        true
    }
}

fn is_canonical_git_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

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

/// Grant a terminal replacement authorization, or refuse naming the reason.
///
/// # Errors
///
/// Refuses when any required slot is absent, when any slot carries a coverage
/// class that cannot support a replacement, when the enriched receipt is not
/// release-admissible, when the cancellation receipt does not validate, or when
/// any slot binds a different candidate than the one under authorization.
pub fn authorize(
    bundle: &ReplacementEvidenceBundleV1<'_>,
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

    let divergence_census_sha256 = bundle
        .divergence_census_sha256
        .ok_or_else(|| ReplacementEvidenceSlotV1::DivergenceCensus.missing())?;
    if !is_lower_sha256(divergence_census_sha256) {
        return Err(GauntletError::InvalidContract {
            reason: "divergence census digest must be a lowercase 64-hex sha256".to_owned(),
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

    Ok(ReplacementAuthorizationV1 {
        schema_version: REPLACEMENT_AUTHORIZATION_SCHEMA_VERSION.to_owned(),
        candidate_source_revision: candidate.to_owned(),
        native_enriched_receipt_address: native_enriched.address().to_owned(),
        cancellation_body_sha256: cancellation.body_sha256.clone(),
        cancellation_engine_revision: cancellation.body.engine_revision.clone(),
        divergence_census_sha256: divergence_census_sha256.to_owned(),
        sealed: AuthorizationSeal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const OTHER_CANDIDATE: &str = "5eb995d524705ef8b17834e9ce005125179b9af2";
    const CENSUS: &str = "9d283b3445b042ac24f2c1d9d65af62c416acc1af4acdad8cca74d0aa70dde31";

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
    ) -> ReplacementEvidenceBundleV1<'a> {
        ReplacementEvidenceBundleV1 {
            candidate_source_revision: candidate,
            core_lexical_v3: Some(core),
            cass_total: Some(cass),
            native_enriched: None,
            cancellation: Some(cancellation),
            divergence_census_sha256: Some(CENSUS),
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
            let complete = bundle(&core, &cass, &cancellation, &candidate);

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
                        divergence_census_sha256: None,
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

            let message = refusal(&bundle(&pinned, &cass, &cancellation, &candidate));
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

            let mut not_passed = accepted_report();
            not_passed.passed = false;
            assert!(
                refusal(&bundle(&not_passed, &cass, &cancellation, &candidate))
                    .contains("did not pass"),
                "a failed campaign must refuse"
            );

            let mut dirty = accepted_report();
            dirty.producer_build_identity.source_git_dirty = true;
            assert!(
                refusal(&bundle(&dirty, &cass, &cancellation, &candidate)).contains("dirty"),
                "a dirty-produced report must refuse"
            );

            let mut no_provenance = accepted_report();
            no_provenance.provenance = None;
            assert!(
                refusal(&bundle(&no_provenance, &cass, &cancellation, &candidate))
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
                refusal(&bundle(&inadmissible, &cass, &cancellation, &candidate))
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
            let message = refusal(&bundle(&core, &cass, &cancellation, OTHER_CANDIDATE));
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
            assert_ne!(
                candidate, OTHER_CANDIDATE,
                "the two revisions must differ, or this test proves nothing"
            );

            let foreign = cancellation_receipt_for(&cx, OTHER_CANDIDATE).await;
            foreign
                .validate()
                .expect("the foreign receipt must be VALID: the defect is whose it is, not what");

            let message = refusal(&bundle(&core, &cass, &foreign, &candidate));
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
            let message = refusal(&bundle(&core, &cass, &matching, &candidate));
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
            let message = refusal(&bundle(&core, &incomplete, &cancellation, &candidate));
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

            let short_candidate = ReplacementEvidenceBundleV1 {
                candidate_source_revision: "702dee9a",
                ..bundle(&core, &cass, &cancellation, &candidate)
            };
            assert!(
                refusal(&short_candidate).contains("40-hex"),
                "an abbreviated candidate revision must refuse"
            );

            let bad_census = ReplacementEvidenceBundleV1 {
                divergence_census_sha256: Some("not-a-digest"),
                ..bundle(&core, &cass, &cancellation, &candidate)
            };
            assert!(
                refusal(&bad_census).contains("64-hex"),
                "a malformed divergence census digest must refuse"
            );
        });
    }
}
