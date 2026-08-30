//! Same-snapshot cutoff and offset completeness certificate (bd-pjvl1).
//!
//! The post-Salej correctness boundary replaces the boolean
//! `cutoff_tie_complete` — historically inferred from requested fetch
//! capacity or mutable counters — with one closed certificate built only
//! from engine-native, same-snapshot observations:
//!
//! * `M = exact_total` from a separate exact-count invocation on the same
//!   pinned snapshot/searcher (it can never influence hit selection);
//! * `O = offset`, `L = limit` as requested;
//! * the checked page interval `P = [min(O, M), min(O + L, M))`;
//! * the expanded evidence `E = [a, b)`: contiguous, unique, absolute ranks
//!   with non-increasing canonical finite scores, `P ⊆ E`;
//! * boundary witnesses: either `a = 0` or rank `a - 1` is the immediate
//!   strictly-higher predecessor; either `b = M` or rank `b` is the
//!   immediate strictly-lower successor;
//! * exhaustion is exactly `b == M`.
//!
//! Requested capacity proves nothing, `returned_count == exact_total` is
//! never authority (least of all when `O > 0`, including `M = 0`), a
//! zero-limit page carries no score boundary, and for `L > 0` an empty page
//! proves `M <= O`. The certificate is content-addressed under its own
//! domain so a stored one can be re-verified byte-for-byte.
//!
//! This module is the runtime authority only. Stored V7 bytes and their
//! replay are untouched; distinct V8 persistence belongs to the downstream
//! schema node, and no comparator waiver or Salej closure is claimed here.

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

/// Schema tag of the runtime certificate.
pub const CUTOFF_CERTIFICATE_SCHEMA_V1: &str = "quill-gauntlet-cutoff-certificate-v1";

/// Domain separator for [`CutoffCertificateV1::digest_sha256`].
const CUTOFF_CERTIFICATE_HASH_DOMAIN: &[u8] =
    b"frankensearch/quill-gauntlet/cutoff-certificate/v1\0";

/// Upper bound on expanded ranks one certificate may carry.
pub const MAX_EXPANDED_RANKS: usize = 65_536;

/// Why a certificate is not a valid same-snapshot completeness proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CutoffCertificateError {
    #[error("certificate schema is not {CUTOFF_CERTIFICATE_SCHEMA_V1}")]
    Schema,
    #[error("offset + limit overflows the checked rank domain")]
    RankOverflow,
    #[error("page interval does not equal [min(O, M), min(O + L, M))")]
    PageInterval,
    #[error("expanded evidence carries more than {MAX_EXPANDED_RANKS} ranks")]
    ExpandedTooLarge,
    #[error("expanded ranks are not contiguous and unique from the declared start")]
    ExpandedRanks,
    #[error("expanded evidence [a, b) does not cover the page interval")]
    PageNotCovered,
    #[error("expanded evidence ends past the exact total")]
    ExpandedPastTotal,
    #[error("an expanded score is not canonical and finite")]
    NonCanonicalScore,
    #[error("expanded scores are not non-increasing in rank order")]
    ScoreOrder,
    #[error("leading boundary witness disagrees with the expanded start")]
    LeadingBoundary,
    #[error("trailing boundary witness disagrees with the expanded end")]
    TrailingBoundary,
    #[error("a zero-limit page must carry no expanded evidence or score boundary")]
    ZeroLimit,
    #[error("an empty page for a positive limit must prove exact_total <= offset")]
    EmptyPage,
    #[error("a provenance or observation digest is unset")]
    UnsetDigest,
    #[error("the same-snapshot authority is unset")]
    UnsetAuthority,
}

/// One observed absolute rank inside the expanded evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpandedRankV1 {
    /// Absolute rank within the full same-snapshot ordering.
    pub absolute_rank: u64,
    /// Native score, raw bits, never reinterpreted.
    pub score_bits: u32,
}

impl ExpandedRankV1 {
    fn score(self) -> f32 {
        f32::from_bits(self.score_bits)
    }
}

/// Why a boundary carries no witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryNotApplicableV1 {
    /// `a = 0`: nothing precedes the expanded evidence.
    AtStart,
    /// `b = M`: the expanded evidence reaches the exact total.
    Exhausted,
    /// `L = 0`: the page has no score boundary at all.
    ZeroLimit,
    /// `M <= O` with `L > 0`: the page lies beyond the end.
    BeyondEnd,
}

/// Boundary witness at one edge of the expanded evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BoundaryWitnessV1 {
    /// No witness is required at this edge.
    NotApplicable { reason: BoundaryNotApplicableV1 },
    /// The immediate neighbour outside the expanded evidence.
    Neighbour {
        /// `a - 1` for the leading edge, `b` for the trailing edge.
        absolute_rank: u64,
        /// Its native score bits: strictly higher than the leading group,
        /// strictly lower than the trailing group.
        score_bits: u32,
    },
}

/// Checked absolute page interval `P = [start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PageIntervalV1 {
    pub start: u64,
    pub end: u64,
}

/// Bound provenance: which physical snapshot, which arm, and which native
/// observations the certificate was derived from. Digests only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificateProvenanceV1 {
    /// Physical snapshot / pinned searcher digest.
    pub snapshot_sha256: [u8; 32],
    /// Arm (engine + configuration) provenance digest.
    pub arm_sha256: [u8; 32],
    /// Digest of the native ranked `TopDocs` observation.
    pub ranked_observation_sha256: [u8; 32],
    /// Digest of the native expanded (tie-completing) observation.
    pub expanded_observation_sha256: [u8; 32],
    /// One-shot same-snapshot authority: the ranked, expanded, and exact
    /// count observations were taken under this single token.
    pub same_snapshot_authority: [u8; 16],
}

/// The closed same-snapshot completeness certificate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CutoffCertificateV1 {
    pub schema: String,
    /// `M`: exact match total from the separate count invocation.
    pub exact_total: u64,
    /// `O`.
    pub offset: u64,
    /// `L`.
    pub limit: u64,
    /// `P`.
    pub page: PageIntervalV1,
    /// `a`: first absolute rank of the expanded evidence.
    pub expanded_start: u64,
    /// `E` in rank order; `len == b - a`.
    pub expanded: Vec<ExpandedRankV1>,
    pub leading_boundary: BoundaryWitnessV1,
    pub trailing_boundary: BoundaryWitnessV1,
    pub provenance: CertificateProvenanceV1,
}

/// Canonical finite score: finite and not negative zero, so one score group
/// has exactly one bit pattern.
fn is_canonical_finite(bits: u32) -> bool {
    let score = f32::from_bits(bits);
    score.is_finite() && bits != (-0.0_f32).to_bits()
}

fn checked_page(
    exact_total: u64,
    offset: u64,
    limit: u64,
) -> Result<PageIntervalV1, CutoffCertificateError> {
    let end = offset
        .checked_add(limit)
        .ok_or(CutoffCertificateError::RankOverflow)?;
    Ok(PageIntervalV1 {
        start: offset.min(exact_total),
        end: end.min(exact_total),
    })
}

impl CutoffCertificateV1 {
    /// Build and validate a certificate from native observations.
    ///
    /// `expanded` is the engine's expanded evidence in rank order starting
    /// at absolute rank `expanded_start`; boundary witnesses are the native
    /// neighbours outside it. Nothing here consults a requested fetch
    /// capacity or a returned count.
    ///
    /// # Errors
    ///
    /// Returns the first violated invariant.
    pub fn new(
        exact_total: u64,
        offset: u64,
        limit: u64,
        expanded_start: u64,
        expanded: Vec<ExpandedRankV1>,
        leading_boundary: BoundaryWitnessV1,
        trailing_boundary: BoundaryWitnessV1,
        provenance: CertificateProvenanceV1,
    ) -> Result<Self, CutoffCertificateError> {
        let page = checked_page(exact_total, offset, limit)?;
        let certificate = Self {
            schema: CUTOFF_CERTIFICATE_SCHEMA_V1.to_owned(),
            exact_total,
            offset,
            limit,
            page,
            expanded_start,
            expanded,
            leading_boundary,
            trailing_boundary,
            provenance,
        };
        certificate.validate()?;
        Ok(certificate)
    }

    /// `b`: one past the last expanded absolute rank.
    #[must_use]
    pub fn expanded_end(&self) -> u64 {
        self.expanded_start
            .saturating_add(u64::try_from(self.expanded.len()).unwrap_or(u64::MAX))
    }

    /// Exhaustion is exactly `b == M`.
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.expanded_end() == self.exact_total
    }

    /// Whether the page is empty.
    #[must_use]
    pub const fn page_is_empty(&self) -> bool {
        self.page.start >= self.page.end
    }

    /// Validate every closed invariant. Pure; no I/O.
    ///
    /// # Errors
    ///
    /// Returns the first violated invariant.
    pub fn validate(&self) -> Result<(), CutoffCertificateError> {
        if self.schema != CUTOFF_CERTIFICATE_SCHEMA_V1 {
            return Err(CutoffCertificateError::Schema);
        }
        let page = checked_page(self.exact_total, self.offset, self.limit)?;
        if self.page != page {
            return Err(CutoffCertificateError::PageInterval);
        }
        if self.expanded.len() > MAX_EXPANDED_RANKS {
            return Err(CutoffCertificateError::ExpandedTooLarge);
        }
        let a = self.expanded_start;
        let len = u64::try_from(self.expanded.len())
            .map_err(|_| CutoffCertificateError::ExpandedTooLarge)?;
        let b = a
            .checked_add(len)
            .ok_or(CutoffCertificateError::RankOverflow)?;
        if b > self.exact_total {
            return Err(CutoffCertificateError::ExpandedPastTotal);
        }
        for (index, rank) in self.expanded.iter().enumerate() {
            let expected = a
                .checked_add(
                    u64::try_from(index).map_err(|_| CutoffCertificateError::ExpandedTooLarge)?,
                )
                .ok_or(CutoffCertificateError::RankOverflow)?;
            if rank.absolute_rank != expected {
                return Err(CutoffCertificateError::ExpandedRanks);
            }
            if !is_canonical_finite(rank.score_bits) {
                return Err(CutoffCertificateError::NonCanonicalScore);
            }
        }
        if self
            .expanded
            .windows(2)
            .any(|pair| pair[1].score() > pair[0].score())
        {
            return Err(CutoffCertificateError::ScoreOrder);
        }
        for witness in [self.leading_boundary, self.trailing_boundary] {
            if let BoundaryWitnessV1::Neighbour { score_bits, .. } = witness
                && !is_canonical_finite(score_bits)
            {
                return Err(CutoffCertificateError::NonCanonicalScore);
            }
        }

        // L = 0: no evidence and no score boundary at either edge.
        if self.limit == 0 {
            let zero = BoundaryWitnessV1::NotApplicable {
                reason: BoundaryNotApplicableV1::ZeroLimit,
            };
            if !self.expanded.is_empty()
                || self.leading_boundary != zero
                || self.trailing_boundary != zero
                || a != page.start
            {
                return Err(CutoffCertificateError::ZeroLimit);
            }
            self.validate_provenance()?;
            return Ok(());
        }

        // L > 0 with an empty page: only beyond the end, and it proves M <= O.
        if page.start >= page.end {
            let beyond = BoundaryWitnessV1::NotApplicable {
                reason: BoundaryNotApplicableV1::BeyondEnd,
            };
            if self.exact_total > self.offset
                || !self.expanded.is_empty()
                || a != self.exact_total
                || self.leading_boundary != beyond
                || self.trailing_boundary != beyond
            {
                return Err(CutoffCertificateError::EmptyPage);
            }
            self.validate_provenance()?;
            return Ok(());
        }

        // Non-empty page: P ⊆ E and E is non-empty.
        if self.expanded.is_empty() || a > page.start || b < page.end {
            return Err(CutoffCertificateError::PageNotCovered);
        }
        let first = self.expanded[0];
        let last = self.expanded[self.expanded.len() - 1];
        match (a, self.leading_boundary) {
            (
                0,
                BoundaryWitnessV1::NotApplicable {
                    reason: BoundaryNotApplicableV1::AtStart,
                },
            ) => {}
            (
                a,
                BoundaryWitnessV1::Neighbour {
                    absolute_rank,
                    score_bits,
                },
            ) if a > 0 && absolute_rank == a - 1 && f32::from_bits(score_bits) > first.score() => {}
            _ => return Err(CutoffCertificateError::LeadingBoundary),
        }
        match self.trailing_boundary {
            BoundaryWitnessV1::NotApplicable {
                reason: BoundaryNotApplicableV1::Exhausted,
            } if b == self.exact_total => {}
            BoundaryWitnessV1::Neighbour {
                absolute_rank,
                score_bits,
            } if b < self.exact_total
                && absolute_rank == b
                && f32::from_bits(score_bits) < last.score() => {}
            _ => return Err(CutoffCertificateError::TrailingBoundary),
        }
        self.validate_provenance()
    }

    fn validate_provenance(&self) -> Result<(), CutoffCertificateError> {
        let p = &self.provenance;
        if [
            p.snapshot_sha256,
            p.arm_sha256,
            p.ranked_observation_sha256,
            p.expanded_observation_sha256,
        ]
        .contains(&[0; 32])
        {
            return Err(CutoffCertificateError::UnsetDigest);
        }
        if p.same_snapshot_authority == [0; 16] {
            return Err(CutoffCertificateError::UnsetAuthority);
        }
        Ok(())
    }

    /// Canonical JSON bytes of a validated certificate.
    ///
    /// # Errors
    ///
    /// Returns the first violated invariant; serialization of a valid
    /// certificate cannot fail.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, CutoffCertificateError> {
        self.validate()?;
        Ok(serde_json::to_vec(self).unwrap_or_default())
    }

    /// Domain-separated content address of a validated certificate.
    ///
    /// # Errors
    ///
    /// Returns the first violated invariant.
    pub fn digest_sha256(&self) -> Result<[u8; 32], CutoffCertificateError> {
        let bytes = self.canonical_bytes()?;
        let mut hasher = Sha256::new();
        hasher.update(CUTOFF_CERTIFICATE_HASH_DOMAIN);
        hasher.update(&bytes);
        Ok(hasher.finalize().into())
    }

    /// Decode and validate a stored certificate.
    ///
    /// # Errors
    ///
    /// Returns [`CutoffCertificateError::Schema`] for undecodable bytes or
    /// the first violated invariant.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, CutoffCertificateError> {
        let certificate: Self =
            serde_json::from_slice(bytes).map_err(|_| CutoffCertificateError::Schema)?;
        certificate.validate()?;
        Ok(certificate)
    }
}

/// Why a certificate cannot be derived from a native ranked prefix.
///
/// None of these is a completeness verdict: an insufficient prefix means
/// the engine must expand further on the same snapshot, never that the
/// page is (or is not) complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CutoffDerivationError {
    #[error("native prefix is longer than the exact total")]
    PrefixLongerThanTotal,
    #[error("native prefix scores are not non-increasing in rank order")]
    PrefixNotRankOrdered,
    #[error("a native prefix score is not canonical and finite")]
    NonCanonicalScore,
    #[error(
        "the trailing score group is cut by the fetch at rank {prefix_len}; expand on the same snapshot"
    )]
    TrailingGroupTruncated { prefix_len: u64 },
    #[error("derived certificate is invalid: {0}")]
    Certificate(#[from] CutoffCertificateError),
}

impl CutoffCertificateV1 {
    /// Derive the certificate from one engine-native ranked prefix.
    ///
    /// `prefix` holds the score bits of absolute ranks `0..n` of the full
    /// same-snapshot ordering (the engine's expanded `TopDocs` fetch);
    /// `exact_total` comes from the separate count on that snapshot. The
    /// expanded evidence is the smallest `[a, b)` that covers the page and
    /// whole score groups at both edges, with the neighbours at `a - 1` and
    /// `b` taken from the same prefix. If the trailing group runs off the end
    /// of the prefix while ranks remain (`n < M`), no certificate exists:
    /// the fetch capacity is not evidence, and the caller must expand.
    ///
    /// # Errors
    ///
    /// Returns a typed derivation error; never infers completeness.
    pub fn from_native_prefix(
        exact_total: u64,
        offset: u64,
        limit: u64,
        prefix: &[u32],
        provenance: CertificateProvenanceV1,
    ) -> Result<Self, CutoffDerivationError> {
        let n = u64::try_from(prefix.len())
            .map_err(|_| CutoffDerivationError::PrefixLongerThanTotal)?;
        if n > exact_total {
            return Err(CutoffDerivationError::PrefixLongerThanTotal);
        }
        if prefix.iter().any(|bits| !is_canonical_finite(*bits)) {
            return Err(CutoffDerivationError::NonCanonicalScore);
        }
        if prefix
            .windows(2)
            .any(|pair| f32::from_bits(pair[1]) > f32::from_bits(pair[0]))
        {
            return Err(CutoffDerivationError::PrefixNotRankOrdered);
        }
        let page = checked_page(exact_total, offset, limit)?;

        if limit == 0 {
            let zero = BoundaryWitnessV1::NotApplicable {
                reason: BoundaryNotApplicableV1::ZeroLimit,
            };
            return Ok(Self::new(
                exact_total,
                offset,
                limit,
                page.start,
                Vec::new(),
                zero,
                zero,
                provenance,
            )?);
        }
        if page.start >= page.end {
            let beyond = BoundaryWitnessV1::NotApplicable {
                reason: BoundaryNotApplicableV1::BeyondEnd,
            };
            return Ok(Self::new(
                exact_total,
                offset,
                limit,
                exact_total,
                Vec::new(),
                beyond,
                beyond,
                provenance,
            )?);
        }

        // The page is non-empty, so every page rank must be in the prefix.
        if page.end > n {
            return Err(CutoffDerivationError::TrailingGroupTruncated { prefix_len: n });
        }
        let at = |rank: u64| prefix[usize::try_from(rank).unwrap_or(usize::MAX)];
        let same = |x: u32, y: u32| f32::from_bits(x).to_bits() == f32::from_bits(y).to_bits();

        // Leading edge: walk back to the start of the score group at P.start.
        let leading_bits = at(page.start);
        let mut a = page.start;
        while a > 0 && same(at(a - 1), leading_bits) {
            a -= 1;
        }
        // Trailing edge: walk forward to the end of the score group at P.end - 1.
        let trailing_bits = at(page.end - 1);
        let mut b = page.end;
        while b < n && same(at(b), trailing_bits) {
            b += 1;
        }
        if b == n && n < exact_total {
            // The group may continue beyond the fetch: no witness, no proof.
            return Err(CutoffDerivationError::TrailingGroupTruncated { prefix_len: n });
        }

        let expanded = (a..b)
            .map(|rank| ExpandedRankV1 {
                absolute_rank: rank,
                score_bits: at(rank),
            })
            .collect();
        let leading = if a == 0 {
            BoundaryWitnessV1::NotApplicable {
                reason: BoundaryNotApplicableV1::AtStart,
            }
        } else {
            BoundaryWitnessV1::Neighbour {
                absolute_rank: a - 1,
                score_bits: at(a - 1),
            }
        };
        let trailing = if b == exact_total {
            BoundaryWitnessV1::NotApplicable {
                reason: BoundaryNotApplicableV1::Exhausted,
            }
        } else {
            BoundaryWitnessV1::Neighbour {
                absolute_rank: b,
                score_bits: at(b),
            }
        };
        Ok(Self::new(
            exact_total,
            offset,
            limit,
            a,
            expanded,
            leading,
            trailing,
            provenance,
        )?)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn provenance() -> CertificateProvenanceV1 {
        CertificateProvenanceV1 {
            snapshot_sha256: [0x11; 32],
            arm_sha256: [0x22; 32],
            ranked_observation_sha256: [0x33; 32],
            expanded_observation_sha256: [0x44; 32],
            same_snapshot_authority: [0x55; 16],
        }
    }

    fn rank(absolute_rank: u64, score: f32) -> ExpandedRankV1 {
        ExpandedRankV1 {
            absolute_rank,
            score_bits: score.to_bits(),
        }
    }

    fn at_start() -> BoundaryWitnessV1 {
        BoundaryWitnessV1::NotApplicable {
            reason: BoundaryNotApplicableV1::AtStart,
        }
    }

    fn exhausted() -> BoundaryWitnessV1 {
        BoundaryWitnessV1::NotApplicable {
            reason: BoundaryNotApplicableV1::Exhausted,
        }
    }

    fn neighbour(absolute_rank: u64, score: f32) -> BoundaryWitnessV1 {
        BoundaryWitnessV1::Neighbour {
            absolute_rank,
            score_bits: score.to_bits(),
        }
    }

    /// A ten-document ordering with a tie group of three at scores 5.0
    /// (ranks 3..6) and a tie group of two at 2.0 (ranks 7..9).
    fn scores() -> [f32; 10] {
        [9.0, 8.0, 7.0, 5.0, 5.0, 5.0, 3.0, 2.0, 2.0, 1.0]
    }

    fn expanded(range: std::ops::Range<u64>) -> Vec<ExpandedRankV1> {
        let scores = scores();
        range
            .map(|r| rank(r, scores[usize::try_from(r).expect("small")]))
            .collect()
    }

    fn certificate(
        exact_total: u64,
        offset: u64,
        limit: u64,
        a: u64,
        b: u64,
    ) -> Result<CutoffCertificateV1, CutoffCertificateError> {
        let scores = scores();
        let leading = if a == 0 {
            at_start()
        } else {
            neighbour(a - 1, scores[usize::try_from(a - 1).expect("small")])
        };
        let trailing = if b == exact_total {
            exhausted()
        } else {
            neighbour(b, scores[usize::try_from(b).expect("small")])
        };
        CutoffCertificateV1::new(
            exact_total,
            offset,
            limit,
            a,
            expanded(a..b),
            leading,
            trailing,
            provenance(),
        )
    }

    #[test]
    fn page_interval_is_checked_and_clamped() {
        let c = certificate(10, 0, 3, 0, 3).expect("plain top-3");
        assert_eq!(c.page, PageIntervalV1 { start: 0, end: 3 });
        assert!(!c.is_exhausted());
        let c = certificate(10, 8, 5, 7, 10).expect("offset page reaching the end");
        assert_eq!(c.page, PageIntervalV1 { start: 8, end: 10 });
        assert!(c.is_exhausted());
        assert_eq!(
            CutoffCertificateV1::new(
                10,
                u64::MAX,
                1,
                0,
                Vec::new(),
                at_start(),
                exhausted(),
                provenance()
            ),
            Err(CutoffCertificateError::RankOverflow)
        );
    }

    #[test]
    fn complete_ties_at_both_boundaries_are_accepted_and_truncated_ties_refused() {
        // Page [3, 8) starts and ends inside tie groups; the evidence must
        // expand to [3, 9) so both groups are complete.
        let complete = certificate(10, 3, 5, 3, 9).expect("both tie groups expanded");
        assert_eq!(complete.expanded.len(), 6);
        // Trailing group truncated: witness at rank 8 has the SAME score as
        // rank 7, so it is not strictly lower.
        assert_eq!(
            certificate(10, 3, 5, 3, 8).unwrap_err(),
            CutoffCertificateError::TrailingBoundary
        );
        // Leading group truncated: predecessor rank 3 is not strictly higher
        // than rank 4.
        assert_eq!(
            certificate(10, 4, 4, 4, 9).unwrap_err(),
            CutoffCertificateError::LeadingBoundary
        );
    }

    #[test]
    fn exhaustion_is_exactly_b_equals_m_never_capacity_or_returned_count() {
        // Limit larger than the total: the evidence reaches M and is exhausted.
        let c = certificate(10, 0, 50, 0, 10).expect("limit > total");
        assert!(c.is_exhausted());
        assert_eq!(c.page, PageIntervalV1 { start: 0, end: 10 });
        // Same page, evidence stops one short with a witnessed successor:
        // a valid certificate that is NOT exhausted, although the page
        // covers everything the requester asked for.
        let c = certificate(10, 0, 9, 0, 9).expect("nine of ten");
        assert!(!c.is_exhausted());
        // Claiming exhaustion (b = M) while ranks stop short is refused.
        assert_eq!(
            CutoffCertificateV1::new(
                10,
                0,
                9,
                0,
                expanded(0..9),
                at_start(),
                exhausted(),
                provenance()
            ),
            Err(CutoffCertificateError::TrailingBoundary)
        );
        // returned_count == exact_total is not a field: with O > 0 a page
        // can return exactly M - O hits and still not be exhausted unless b = M.
        let c = certificate(10, 7, 3, 7, 10).expect("last page");
        assert!(c.is_exhausted());
        let c = certificate(10, 6, 3, 6, 9).expect("penultimate page");
        assert!(!c.is_exhausted());
    }

    #[test]
    fn empty_totals_and_beyond_end_pages_are_typed() {
        let beyond = BoundaryWitnessV1::NotApplicable {
            reason: BoundaryNotApplicableV1::BeyondEnd,
        };
        // M = 0 with O = 0 and O > 0: empty page proves M <= O.
        for offset in [0, 5] {
            let c =
                CutoffCertificateV1::new(0, offset, 3, 0, Vec::new(), beyond, beyond, provenance())
                    .expect("empty corpus");
            assert!(c.page_is_empty());
            assert!(c.is_exhausted());
        }
        // Beyond the end of a non-empty corpus.
        let c = CutoffCertificateV1::new(10, 10, 3, 10, Vec::new(), beyond, beyond, provenance())
            .expect("beyond end");
        assert!(c.page_is_empty());
        // An empty page that does not prove M <= O is refused.
        assert_eq!(
            CutoffCertificateV1::new(10, 4, 3, 10, Vec::new(), beyond, beyond, provenance()),
            Err(CutoffCertificateError::PageNotCovered)
        );
        // Beyond-end evidence must sit exactly at M.
        assert_eq!(
            CutoffCertificateV1::new(10, 12, 3, 9, Vec::new(), beyond, beyond, provenance()),
            Err(CutoffCertificateError::EmptyPage)
        );
    }

    #[test]
    fn zero_limit_has_no_score_boundary() {
        let zero = BoundaryWitnessV1::NotApplicable {
            reason: BoundaryNotApplicableV1::ZeroLimit,
        };
        let c = CutoffCertificateV1::new(10, 4, 0, 4, Vec::new(), zero, zero, provenance())
            .expect("zero limit");
        assert!(c.page_is_empty());
        assert!(!c.is_exhausted());
        // Any evidence or witness with L = 0 is refused.
        assert_eq!(
            CutoffCertificateV1::new(10, 4, 0, 4, expanded(4..5), zero, zero, provenance()),
            Err(CutoffCertificateError::ZeroLimit)
        );
        assert_eq!(
            CutoffCertificateV1::new(
                10,
                4,
                0,
                4,
                Vec::new(),
                neighbour(3, 5.0),
                zero,
                provenance()
            ),
            Err(CutoffCertificateError::ZeroLimit)
        );
        // Zero limit beyond the end still has no boundary and sits at M.
        let c = CutoffCertificateV1::new(10, 12, 0, 10, Vec::new(), zero, zero, provenance())
            .expect("zero limit beyond end");
        assert_eq!(c.page, PageIntervalV1 { start: 10, end: 10 });
    }

    #[test]
    fn gaps_duplicates_order_and_non_canonical_scores_are_refused() {
        let base = certificate(10, 3, 5, 3, 9).expect("valid");
        let mut gap = base.clone();
        gap.expanded[2].absolute_rank = 9;
        assert_eq!(gap.validate(), Err(CutoffCertificateError::ExpandedRanks));
        let mut duplicate = base.clone();
        duplicate.expanded[2].absolute_rank = duplicate.expanded[1].absolute_rank;
        assert_eq!(
            duplicate.validate(),
            Err(CutoffCertificateError::ExpandedRanks)
        );
        let mut order = base.clone();
        order.expanded[1].score_bits = 9.5_f32.to_bits();
        assert_eq!(order.validate(), Err(CutoffCertificateError::ScoreOrder));
        for bits in [
            f32::NAN.to_bits(),
            f32::INFINITY.to_bits(),
            (-0.0_f32).to_bits(),
        ] {
            let mut bad = base.clone();
            bad.expanded[0].score_bits = bits;
            assert_eq!(
                bad.validate(),
                Err(CutoffCertificateError::NonCanonicalScore)
            );
            let mut bad_witness = base.clone();
            bad_witness.trailing_boundary = BoundaryWitnessV1::Neighbour {
                absolute_rank: 9,
                score_bits: bits,
            };
            assert_eq!(
                bad_witness.validate(),
                Err(CutoffCertificateError::NonCanonicalScore)
            );
        }
        let mut past = base.clone();
        past.exact_total = 8;
        past.page = PageIntervalV1 { start: 3, end: 8 };
        assert_eq!(
            past.validate(),
            Err(CutoffCertificateError::ExpandedPastTotal)
        );
        let mut non_adjacent = base;
        non_adjacent.leading_boundary = neighbour(1, 8.0);
        assert_eq!(
            non_adjacent.validate(),
            Err(CutoffCertificateError::LeadingBoundary)
        );
    }

    #[test]
    fn every_field_mutation_changes_or_refuses_the_digest() {
        let base = certificate(10, 3, 5, 3, 9).expect("valid");
        let digest = base.digest_sha256().expect("digest");
        let bytes = base.canonical_bytes().expect("bytes");
        assert_eq!(
            CutoffCertificateV1::from_canonical_bytes(&bytes).expect("round trip"),
            base
        );
        assert_eq!(base.digest_sha256().expect("stable"), digest);

        type Mutation = Box<dyn Fn(&mut CutoffCertificateV1)>;
        let mutations: Vec<(&str, Mutation)> = vec![
            ("schema", Box::new(|c| c.schema = "other".to_owned())),
            ("exact_total", Box::new(|c| c.exact_total = 11)),
            ("offset", Box::new(|c| c.offset = 4)),
            ("limit", Box::new(|c| c.limit = 4)),
            ("page.start", Box::new(|c| c.page.start = 2)),
            ("page.end", Box::new(|c| c.page.end = 9)),
            ("expanded_start", Box::new(|c| c.expanded_start = 2)),
            (
                "expanded.pop",
                Box::new(|c| {
                    c.expanded.pop();
                }),
            ),
            (
                "expanded.score",
                Box::new(|c| c.expanded[3].score_bits = 4.5_f32.to_bits()),
            ),
            ("leading", Box::new(|c| c.leading_boundary = at_start())),
            ("trailing", Box::new(|c| c.trailing_boundary = exhausted())),
            (
                "snapshot",
                Box::new(|c| c.provenance.snapshot_sha256 = [0; 32]),
            ),
            ("arm", Box::new(|c| c.provenance.arm_sha256 = [0x99; 32])),
            (
                "ranked",
                Box::new(|c| c.provenance.ranked_observation_sha256 = [0; 32]),
            ),
            (
                "expanded_obs",
                Box::new(|c| c.provenance.expanded_observation_sha256 = [0x98; 32]),
            ),
            (
                "authority",
                Box::new(|c| c.provenance.same_snapshot_authority = [0; 16]),
            ),
        ];
        for (name, mutate) in mutations {
            let mut mutated = base.clone();
            mutate(&mut mutated);
            if let Ok(other) = mutated.digest_sha256() {
                assert_ne!(other, digest, "{name} must change the digest");
            }
            assert!(
                mutated.validate().is_err() || mutated.digest_sha256() != Ok(digest),
                "{name} silently kept the certificate"
            );
        }
        // Unknown fields are refused on decode.
        let mut json: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        json["returned_count"] = serde_json::Value::from(5);
        let with_returned = serde_json::to_vec(&json).expect("bytes");
        assert_eq!(
            CutoffCertificateV1::from_canonical_bytes(&with_returned),
            Err(CutoffCertificateError::Schema)
        );
    }

    #[test]
    fn a_score_group_inside_the_evidence_may_be_partially_paged_but_not_cut() {
        // Page [4, 6) sits entirely inside the 5.0 group; the evidence must
        // still expand to the whole group [3, 6) with strict neighbours.
        let c = certificate(10, 4, 2, 3, 6).expect("group fully expanded");
        assert_eq!(c.page, PageIntervalV1 { start: 4, end: 6 });
        assert_eq!(c.expanded_end(), 6);
        // Evidence exactly equal to the page cuts the group on both sides.
        assert_eq!(
            certificate(10, 4, 2, 4, 6).unwrap_err(),
            CutoffCertificateError::LeadingBoundary
        );
    }

    fn prefix(len: usize) -> Vec<u32> {
        scores()[..len].iter().map(|s| s.to_bits()).collect()
    }

    #[test]
    fn derivation_expands_to_whole_groups_and_witnesses_neighbours_from_the_prefix() {
        // Page [3, 8) touches the 5.0 group (3..6) and the 2.0 group (7..9);
        // a nine-rank prefix carries the witness at rank 9 (1.0).
        let c = CutoffCertificateV1::from_native_prefix(10, 3, 5, &prefix(10), provenance())
            .expect("full prefix");
        assert_eq!(c.expanded_start, 3);
        assert_eq!(c.expanded_end(), 9);
        assert_eq!(c.leading_boundary, neighbour(2, 7.0));
        assert_eq!(c.trailing_boundary, neighbour(9, 1.0));
        assert!(!c.is_exhausted());
        assert_eq!(
            c,
            CutoffCertificateV1::from_native_prefix(10, 3, 5, &prefix(10), provenance())
                .expect("deterministic")
        );
        // Exactly enough prefix (rank 9 present) derives the same certificate.
        assert_eq!(
            CutoffCertificateV1::from_native_prefix(10, 3, 5, &prefix(10), provenance())
                .expect("ten")
                .digest_sha256(),
            c.digest_sha256()
        );
    }

    #[test]
    fn a_fetch_that_cuts_the_trailing_group_is_insufficient_not_incomplete() {
        // Prefix of 8 ends inside the 2.0 group: the engine must expand.
        assert_eq!(
            CutoffCertificateV1::from_native_prefix(10, 3, 5, &prefix(8), provenance()),
            Err(CutoffDerivationError::TrailingGroupTruncated { prefix_len: 8 })
        );
        // Prefix of 9 ends exactly at the group end but rank 9 is unseen:
        // the group MAY continue, so still insufficient.
        assert_eq!(
            CutoffCertificateV1::from_native_prefix(10, 3, 5, &prefix(9), provenance()),
            Err(CutoffDerivationError::TrailingGroupTruncated { prefix_len: 9 })
        );
        // A page shorter than the prefix can be covered.
        assert_eq!(
            CutoffCertificateV1::from_native_prefix(10, 0, 9, &prefix(9), provenance()),
            Err(CutoffDerivationError::TrailingGroupTruncated { prefix_len: 9 })
        );
        // Requested capacity larger than the total proves nothing by itself:
        // with M = 10 and a prefix of 10, exhaustion comes from b == M.
        let c = CutoffCertificateV1::from_native_prefix(10, 0, 50, &prefix(10), provenance())
            .expect("exhausted");
        assert!(c.is_exhausted());
        assert_eq!(c.trailing_boundary, exhausted());
    }

    #[test]
    fn derivation_handles_zero_limit_empty_totals_and_beyond_end() {
        let zero = CutoffCertificateV1::from_native_prefix(10, 4, 0, &prefix(10), provenance())
            .expect("zero limit");
        assert!(zero.page_is_empty());
        assert_eq!(
            zero.leading_boundary,
            BoundaryWitnessV1::NotApplicable {
                reason: BoundaryNotApplicableV1::ZeroLimit
            }
        );
        for (total, offset, prefix_len) in [(0, 0, 0), (0, 4, 0), (10, 10, 10), (10, 12, 3)] {
            let c = CutoffCertificateV1::from_native_prefix(
                total,
                offset,
                3,
                &prefix(prefix_len),
                provenance(),
            )
            .expect("empty page beyond end");
            assert!(c.page_is_empty());
            assert_eq!(c.expanded_start, total);
            assert_eq!(
                c.trailing_boundary,
                BoundaryWitnessV1::NotApplicable {
                    reason: BoundaryNotApplicableV1::BeyondEnd
                }
            );
        }
        // A prefix longer than the exact total is a contradiction.
        assert_eq!(
            CutoffCertificateV1::from_native_prefix(5, 0, 3, &prefix(6), provenance()),
            Err(CutoffDerivationError::PrefixLongerThanTotal)
        );
        // Non-ranked or non-canonical prefixes are refused before derivation.
        let mut unordered = prefix(10);
        unordered.swap(1, 2);
        assert_eq!(
            CutoffCertificateV1::from_native_prefix(10, 0, 3, &unordered, provenance()),
            Err(CutoffDerivationError::PrefixNotRankOrdered)
        );
        let mut nan = prefix(10);
        nan[4] = f32::NAN.to_bits();
        assert_eq!(
            CutoffCertificateV1::from_native_prefix(10, 0, 3, &nan, provenance()),
            Err(CutoffDerivationError::NonCanonicalScore)
        );
    }

    #[test]
    fn derivation_at_the_offset_edge_expands_the_leading_group() {
        // Page starts at rank 4, inside the 5.0 group: a walks back to 3 and
        // the predecessor witness is rank 2 (7.0).
        let c = CutoffCertificateV1::from_native_prefix(10, 4, 2, &prefix(10), provenance())
            .expect("leading expansion");
        assert_eq!(c.expanded_start, 3);
        assert_eq!(c.expanded_end(), 6);
        assert_eq!(c.leading_boundary, neighbour(2, 7.0));
        assert_eq!(c.trailing_boundary, neighbour(6, 3.0));
        // Last page: b reaches M and is exhausted without any witness.
        let c = CutoffCertificateV1::from_native_prefix(10, 8, 5, &prefix(10), provenance())
            .expect("last page");
        assert_eq!(c.expanded_start, 7);
        assert!(c.is_exhausted());
    }

    mod properties {
        use super::*;
        use proptest::prelude::*;

        /// A non-increasing ordering over a small score alphabet so ties are
        /// frequent at both page edges.
        fn ordering() -> impl Strategy<Value = Vec<f32>> {
            proptest::collection::vec(
                prop_oneof![Just(7.0_f32), Just(5.0), Just(3.0), Just(2.0), Just(1.0)],
                0..=12,
            )
            .prop_map(|mut scores| {
                scores.sort_by(|left, right| right.total_cmp(left));
                scores
            })
        }

        /// Brute-force oracle for what `from_native_prefix` must decide.
        fn expected(
            scores: &[f32],
            offset: u64,
            limit: u64,
            prefix_len: usize,
        ) -> Result<(u64, u64, bool), ()> {
            let m = u64::try_from(scores.len()).expect("small");
            let n = u64::try_from(prefix_len).expect("small");
            let start = offset.min(m);
            let end = offset.saturating_add(limit).min(m);
            if limit == 0 {
                // No evidence and no boundary, but exhaustion is still the
                // closed `b == M` fact: true exactly when the page sits at M.
                return Ok((start, start, start == m));
            }
            if start >= end {
                return Ok((m, m, true));
            }
            if end > n {
                return Err(());
            }
            let at = |rank: u64| scores[usize::try_from(rank).expect("small")];
            let mut a = start;
            while a > 0 && at(a - 1).to_bits() == at(start).to_bits() {
                a -= 1;
            }
            let mut b = end;
            while b < n && at(b).to_bits() == at(end - 1).to_bits() {
                b += 1;
            }
            if b == n && n < m {
                return Err(());
            }
            Ok((a, b, b == m))
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(512))]

            #[test]
            fn derivation_agrees_with_the_brute_force_oracle_and_is_closed(
                scores in ordering(),
                offset in 0_u64..=14,
                limit in 0_u64..=6,
                prefix_fraction in 0_u8..=12,
            ) {
                let prefix_len = usize::from(prefix_fraction).min(scores.len());
                let bits = scores[..prefix_len].iter().map(|s| s.to_bits()).collect::<Vec<_>>();
                let m = u64::try_from(scores.len()).expect("small");
                let derived = CutoffCertificateV1::from_native_prefix(m, offset, limit, &bits, provenance());
                match expected(&scores, offset, limit, prefix_len) {
                    Err(()) => prop_assert!(
                        matches!(derived, Err(CutoffDerivationError::TrailingGroupTruncated { .. })),
                        "expected insufficiency, got {derived:?}"
                    ),
                    Ok((a, b, exhausted)) => {
                        let certificate = derived.expect("brute force says certifiable");
                        prop_assert_eq!(certificate.validate(), Ok(()));
                        prop_assert_eq!(certificate.expanded_start, a);
                        prop_assert_eq!(certificate.expanded_end(), b);
                        prop_assert_eq!(certificate.is_exhausted(), exhausted);
                        prop_assert_eq!(certificate.page.start, offset.min(m));
                        prop_assert_eq!(certificate.page.end, offset.saturating_add(limit).min(m));
                        // P ⊆ E for a non-empty page.
                        if certificate.page.start < certificate.page.end {
                            prop_assert!(a <= certificate.page.start && certificate.page.end <= b);
                            // Whole groups at both edges, witnessed by strict neighbours.
                            let at = |rank: u64| scores[usize::try_from(rank).expect("small")];
                            if a > 0 {
                                prop_assert!(at(a - 1) > at(a));
                            }
                            if b < m {
                                prop_assert!(at(b) < at(b - 1));
                            }
                        }
                        // Content addressing is deterministic and round-trips.
                        let bytes = certificate.canonical_bytes().expect("bytes");
                        let decoded = CutoffCertificateV1::from_canonical_bytes(&bytes).expect("decode");
                        prop_assert_eq!(&decoded, &certificate);
                        prop_assert_eq!(decoded.digest_sha256(), certificate.digest_sha256());
                    }
                }
            }

            #[test]
            fn the_trailing_witness_is_essential_capacity_never_substitutes(
                scores in ordering(),
                offset in 0_u64..=14,
                limit in 1_u64..=6,
            ) {
                let m = u64::try_from(scores.len()).expect("small");
                let bits = scores.iter().map(|s| s.to_bits()).collect::<Vec<_>>();
                let Ok(certificate) = CutoffCertificateV1::from_native_prefix(m, offset, limit, &bits, provenance()) else {
                    return Ok(());
                };
                if certificate.is_exhausted() || certificate.page_is_empty() {
                    return Ok(());
                }
                // Drop everything from the witness rank `b` onward: the same
                // page with the same requested capacity is no longer certifiable.
                let b = usize::try_from(certificate.expanded_end()).expect("small");
                let without_witness = &bits[..b];
                let truncated = matches!(
                    CutoffCertificateV1::from_native_prefix(m, offset, limit, without_witness, provenance()),
                    Err(CutoffDerivationError::TrailingGroupTruncated { .. })
                );
                prop_assert!(truncated);
                // And a certificate that CLAIMS exhaustion over that shorter
                // prefix is refused by the validator.
                let mut forged = certificate.clone();
                forged.trailing_boundary = exhausted();
                prop_assert!(forged.validate().is_err());
            }
        }
    }
}
