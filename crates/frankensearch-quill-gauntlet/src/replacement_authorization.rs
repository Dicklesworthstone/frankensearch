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
use crate::campaign_contract::CampaignContractModeV1;
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
    /// CASS-visible total coverage binding.
    ///
    /// ASYMMETRY, STATED RATHER THAN LEFT TO BE DISCOVERED: this slot is still
    /// a hand-constructible binding, because no derivation for it exists.
    /// `CampaignLexicalCoverageSummary` has exactly three variants --
    /// `CoreLexicalV3`, `RankEnvelopeOnly`, `LegacyMissing` -- with no CASS
    /// variant, and `from_campaign_report` hardcodes `CoreLexicalV3` on
    /// success. Until bd-8nqz.5 supplies a real producer, this slot is exactly
    /// as strong as its caller, and no stronger.
    pub cass_total: Option<&'evidence AcceptedCandidateBindingV1>,
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

    let cass_total = bundle
        .cass_total
        .ok_or_else(|| ReplacementEvidenceSlotV1::CassTotalBinding.missing())?;
    require_binding(
        cass_total,
        CampaignContractModeV1::CassTotalV1,
        candidate,
        ReplacementEvidenceSlotV1::CassTotalBinding,
    )?;

    let cancellation = bundle
        .cancellation
        .ok_or_else(|| ReplacementEvidenceSlotV1::CancellationReceipt.missing())?;
    cancellation.validate()?;

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
        crate::runner::load_pinned_campaign_report_v7().expect("the pinned V7 campaign report")
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

    fn cass_binding(revision: &str) -> AcceptedCandidateBindingV1 {
        AcceptedCandidateBindingV1 {
            candidate_source_revision: revision.to_owned(),
            contract_mode: CampaignContractModeV1::CassTotalV1,
        }
    }

    /// A bundle complete in every slot EXCEPT the enriched receipt, which no
    /// unit test can mint: it needs a live both-engines run from a clean
    /// checkout. Every refusal below is proved against a bundle whose only
    /// other absence is that one slot, and each test changes exactly one
    /// further thing.
    fn bundle<'a>(
        core: &'a CampaignReport,
        cass: &'a AcceptedCandidateBindingV1,
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

    async fn cancellation_receipt(cx: &asupersync::Cx) -> QuillCancellationReceipt {
        crate::comparator::observe_live_quill_cancellation_receipt(cx)
            .await
            .expect("observe the live Quill cancellation matrix")
    }

    #[test]
    fn every_required_slot_is_independently_load_bearing() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let cancellation = cancellation_receipt(&cx).await;
            let core = accepted_report();
            let candidate = core.producer_build_identity.source_git_revision.clone();
            let cass = cass_binding(&candidate);
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
            let cancellation = cancellation_receipt(&cx).await;
            let pinned = pinned_report();
            assert!(
                pinned.passed,
                "the fixture must really be a PASSED report, or this test proves nothing"
            );
            let candidate = pinned.producer_build_identity.source_git_revision.clone();
            let cass = cass_binding(&candidate);

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
            let cancellation = cancellation_receipt(&cx).await;
            let candidate = accepted_report()
                .producer_build_identity
                .source_git_revision
                .clone();
            let cass = cass_binding(&candidate);

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
            let cancellation = cancellation_receipt(&cx).await;
            let core = accepted_report();
            let candidate = core.producer_build_identity.source_git_revision.clone();

            // THE STALE-EVIDENCE HOLE: the report is valid and of the right
            // coverage class, the CASS binding is well formed, and the BUNDLE
            // still must not authorize because they describe different
            // candidates. A per-receipt validator cannot see this at all.
            let stale_cass = cass_binding(OTHER_CANDIDATE);
            let message = refusal(&bundle(&core, &stale_cass, &cancellation, &candidate));
            assert!(
                message.contains(OTHER_CANDIDATE),
                "a stale CASS binding must refuse naming the mismatch, got: {message}"
            );

            // And the mirror: an authorization for a candidate the report does
            // not describe.
            let cass = cass_binding(OTHER_CANDIDATE);
            let message = refusal(&bundle(&core, &cass, &cancellation, OTHER_CANDIDATE));
            assert!(
                message.contains(&candidate),
                "a report from another candidate must refuse naming it, got: {message}"
            );
        });
    }

    #[test]
    fn a_non_canonical_candidate_and_census_digest_are_refused() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let cancellation = cancellation_receipt(&cx).await;
            let core = accepted_report();
            let candidate = core.producer_build_identity.source_git_revision.clone();
            let cass = cass_binding(&candidate);

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
