//! Generation activation controller for Native Mode distributed search.
//!
//! The [`GenerationController`] manages the lifecycle of search generations:
//! verifying artifacts against the manifest, checking activation invariants,
//! atomically swapping the active generation pointer, and rolling back on failure.
//!
//! The active generation pointer is an `Arc`-swapped reference: readers acquire
//! a clone of the current `Arc<ActiveGeneration>` and hold it for the duration of
//! a single query, guaranteeing no mixed-generation reads.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::generation::{
    ActivationInvariant, EmbedderRevision, GenerationManifest, InvariantKind, ValidationResult,
    validate_manifest,
};
use crate::{SearchError, SearchResult};

// ---------------------------------------------------------------------------
// Active generation snapshot
// ---------------------------------------------------------------------------

/// A verified, ready-to-serve generation snapshot.
///
/// Query handlers acquire an `Arc<ActiveGeneration>` at request start and hold it
/// for the request lifecycle. This ensures a single query never mixes artifacts
/// from different generations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveGeneration {
    /// The manifest this generation was built from.
    pub manifest: GenerationManifest,
    /// Monotonic activation sequence number (increases with each activation).
    pub activation_seq: u64,
    /// Unix timestamp (millis) when this generation was activated.
    pub activated_at: u64,
    /// Resolved paths to verified vector artifacts (keyed by manifest path).
    pub vector_paths: BTreeMap<String, String>,
    /// Resolved paths to verified lexical artifacts (keyed by manifest path).
    pub lexical_paths: BTreeMap<String, String>,
}

// ---------------------------------------------------------------------------
// Artifact verification
// ---------------------------------------------------------------------------

/// Result of verifying a single artifact against its manifest descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactVerification {
    /// Manifest-relative path of the artifact.
    pub path: String,
    /// Whether the artifact passed verification.
    pub passed: bool,
    /// Details about the verification (checksum match, size match, etc.).
    pub detail: String,
}

/// Trait for verifying that artifacts on disk match their manifest descriptors.
///
/// Consumers provide an implementation that knows how to locate and checksum
/// files in their environment. This keeps the controller decoupled from
/// filesystem specifics.
pub trait ArtifactVerifier: Send + Sync {
    /// Verify a single artifact exists and matches the expected checksum.
    ///
    /// `manifest_path` is the relative path from the manifest; `checksum` is
    /// the expected hex-encoded SHA-256 hash.
    fn verify(
        &self,
        manifest_path: &str,
        checksum: &str,
        expected_size: u64,
    ) -> ArtifactVerification;

    /// Return the resolved absolute path for a manifest-relative artifact path.
    fn resolve_path(&self, manifest_path: &str) -> String;

    /// Return runtime embedder revision for a tier key (e.g. `"fast"`, `"quality"`).
    ///
    /// Implementations that cannot provide runtime embedder identity should
    /// return `None`. In that case, `EmbedderRevisionMatch` invariants fail.
    fn runtime_embedder(&self, _tier: &str) -> Option<EmbedderRevision> {
        None
    }

    /// Evaluate a deployment-defined custom invariant.
    ///
    /// Return `None` when this verifier does not implement custom checks.
    /// Return `Some((passed, reason))` when the check was evaluated.
    fn check_custom_invariant(
        &self,
        _check_name: &str,
        _manifest: &GenerationManifest,
        _previous: Option<&ActiveGeneration>,
    ) -> Option<(bool, String)> {
        None
    }
}

// ---------------------------------------------------------------------------
// Activation gate
// ---------------------------------------------------------------------------

/// Result of evaluating a single activation invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvariantCheck {
    /// The invariant that was checked.
    pub invariant_id: String,
    /// Whether the invariant passed.
    pub passed: bool,
    /// Reason for pass or failure.
    pub reason: String,
}

/// Evaluate all activation invariants for a manifest.
///
/// Returns a list of check results. All must pass for activation to proceed.
pub fn check_invariants(
    manifest: &GenerationManifest,
    verifier: &dyn ArtifactVerifier,
    previous_generation: Option<&ActiveGeneration>,
) -> Vec<InvariantCheck> {
    manifest
        .activation_invariants
        .iter()
        .map(|inv| evaluate_invariant(inv, manifest, verifier, previous_generation))
        .collect()
}

fn evaluate_invariant(
    inv: &ActivationInvariant,
    manifest: &GenerationManifest,
    verifier: &dyn ArtifactVerifier,
    previous: Option<&ActiveGeneration>,
) -> InvariantCheck {
    match &inv.kind {
        InvariantKind::AllArtifactsVerified => {
            evaluate_all_artifacts_verified(inv, manifest, verifier)
        }
        InvariantKind::EmbedderRevisionMatch => {
            evaluate_embedder_revision_match(inv, manifest, verifier)
        }
        InvariantKind::VectorCountConsistency { expected_total } => {
            evaluate_vector_count_consistency(inv, manifest, *expected_total)
        }
        InvariantKind::CommitContinuity { previous_high } => {
            evaluate_commit_continuity(inv, manifest, previous, *previous_high)
        }
        InvariantKind::Custom { check_name } => {
            evaluate_custom_invariant(inv, check_name, verifier, manifest, previous)
        }
    }
}

fn evaluate_all_artifacts_verified(
    inv: &ActivationInvariant,
    manifest: &GenerationManifest,
    verifier: &dyn ArtifactVerifier,
) -> InvariantCheck {
    let mut failures = Vec::new();
    for art in &manifest.vector_artifacts {
        let v = verifier.verify(&art.path, &art.checksum, art.size_bytes);
        if !v.passed {
            failures.push(format!("{}: {}", art.path, v.detail));
        }
    }
    for art in &manifest.lexical_artifacts {
        let v = verifier.verify(&art.path, &art.checksum, art.size_bytes);
        if !v.passed {
            failures.push(format!("{}: {}", art.path, v.detail));
        }
    }

    if failures.is_empty() {
        InvariantCheck {
            invariant_id: inv.id.clone(),
            passed: true,
            reason: "all artifacts verified".into(),
        }
    } else {
        InvariantCheck {
            invariant_id: inv.id.clone(),
            passed: false,
            reason: format!(
                "{} artifact(s) failed: {}",
                failures.len(),
                failures.join("; ")
            ),
        }
    }
}

fn evaluate_embedder_revision_match(
    inv: &ActivationInvariant,
    manifest: &GenerationManifest,
    verifier: &dyn ArtifactVerifier,
) -> InvariantCheck {
    let mut failures = Vec::new();
    for (tier, expected) in &manifest.embedders {
        let Some(actual) = verifier.runtime_embedder(tier) else {
            failures.push(format!("runtime embedder missing for tier '{tier}'"));
            continue;
        };
        if actual != *expected {
            failures.push(format!(
                "tier '{tier}' mismatch: expected model={} hash={} dim={} quant={:?}, got model={} hash={} dim={} quant={:?}",
                expected.model_name,
                expected.weights_hash,
                expected.dimension,
                expected.quantization,
                actual.model_name,
                actual.weights_hash,
                actual.dimension,
                actual.quantization,
            ));
        }
    }

    let passed = failures.is_empty();
    InvariantCheck {
        invariant_id: inv.id.clone(),
        passed,
        reason: if passed {
            "runtime embedder revisions match manifest".into()
        } else {
            failures.join("; ")
        },
    }
}

fn evaluate_vector_count_consistency(
    inv: &ActivationInvariant,
    manifest: &GenerationManifest,
    expected_total: u64,
) -> InvariantCheck {
    let actual: u64 = manifest
        .vector_artifacts
        .iter()
        .map(|artifact| artifact.vector_count)
        .sum();
    let passed = actual == expected_total;
    InvariantCheck {
        invariant_id: inv.id.clone(),
        passed,
        reason: if passed {
            format!("vector count {actual} matches expected {expected_total}")
        } else {
            format!("vector count {actual} != expected {expected_total}")
        },
    }
}

fn evaluate_commit_continuity(
    inv: &ActivationInvariant,
    manifest: &GenerationManifest,
    previous: Option<&ActiveGeneration>,
    previous_high: u64,
) -> InvariantCheck {
    let (passed, reason) = previous.map_or_else(
        || {
            (
                true,
                "commit continuity skipped (no previous generation)".into(),
            )
        },
        |prev| {
            let actual_prev_high = prev.manifest.commit_range.high;
            let actual_low = manifest.commit_range.low;
            let expected_low = previous_high.saturating_add(1);
            if actual_prev_high != previous_high {
                (
                    false,
                    format!("expected previous high {previous_high}, found {actual_prev_high}"),
                )
            } else if actual_low != expected_low {
                (
                    false,
                    format!(
                        "expected current low {expected_low} to follow previous high {previous_high}, found {actual_low}"
                    ),
                )
            } else {
                (
                    true,
                    format!(
                        "commit continuity verified: prev_high={previous_high}, current_low={actual_low}"
                    ),
                )
            }
        },
    );

    InvariantCheck {
        invariant_id: inv.id.clone(),
        passed,
        reason,
    }
}

fn evaluate_custom_invariant(
    inv: &ActivationInvariant,
    check_name: &str,
    verifier: &dyn ArtifactVerifier,
    manifest: &GenerationManifest,
    previous: Option<&ActiveGeneration>,
) -> InvariantCheck {
    if let Some((passed, reason)) = verifier.check_custom_invariant(check_name, manifest, previous)
    {
        return InvariantCheck {
            invariant_id: inv.id.clone(),
            passed,
            reason,
        };
    }

    InvariantCheck {
        invariant_id: inv.id.clone(),
        passed: false,
        reason: format!("custom check '{check_name}' is not implemented"),
    }
}

// ---------------------------------------------------------------------------
// Generation controller
// ---------------------------------------------------------------------------

/// Manages the active search generation with atomic pointer swap and rollback.
///
/// Thread-safe: multiple query handlers can read the active generation concurrently
/// while the controller swaps to a new generation.
pub struct GenerationController {
    /// Currently active generation (readers acquire Arc clone).
    /// Uses `std::sync::RwLock` (not asupersync) because generation pointer reads
    /// must be synchronous on the query hot path — no `&Cx` available at read time.
    active: RwLock<Option<Arc<ActiveGeneration>>>,
    /// Monotonically increasing activation counter.
    activation_counter: AtomicU64,
    /// Previous generation kept for rollback.
    rollback: RwLock<Option<Arc<ActiveGeneration>>>,
    /// Serializes activate/rollback writers to prevent concurrent state races.
    write_lock: Mutex<()>,
}

impl GenerationController {
    /// Create a new controller with no active generation.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            active: RwLock::new(None),
            activation_counter: AtomicU64::new(0),
            rollback: RwLock::new(None),
            write_lock: Mutex::new(()),
        }
    }

    /// Get the currently active generation, if any.
    ///
    /// Returns an `Arc` that the caller holds for the duration of a query,
    /// ensuring consistent reads even if a new generation is activated mid-query.
    #[must_use]
    pub fn active(&self) -> Option<Arc<ActiveGeneration>> {
        self.active
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Current activation sequence number.
    #[must_use]
    pub fn activation_seq(&self) -> u64 {
        self.activation_counter.load(Ordering::Acquire)
    }

    /// Attempt to activate a new generation.
    ///
    /// Performs in order:
    /// 1. Validate the manifest structurally.
    /// 2. Verify all artifacts via the provided verifier.
    /// 3. Check all activation invariants.
    /// 4. Atomically swap the generation pointer.
    ///
    /// On failure at any step, the previous generation remains active.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if validation fails, or
    /// `SearchError::SubsystemError` if invariant checks fail.
    pub fn activate(
        &self,
        manifest: GenerationManifest,
        verifier: &dyn ArtifactVerifier,
        now_millis: u64,
    ) -> SearchResult<Arc<ActiveGeneration>> {
        let _write_guard = self
            .write_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // Step 1: Structural validation.
        let validation = validate_manifest(&manifest);
        if !validation.is_valid() {
            return Err(activation_error(&validation));
        }

        // Step 2 & 3: Check invariants (which includes artifact verification).
        let previous = self
            .active
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let checks = check_invariants(&manifest, verifier, previous.as_deref());
        let failed: Vec<&InvariantCheck> = checks.iter().filter(|c| !c.passed).collect();
        if !failed.is_empty() {
            let reasons: Vec<String> = failed
                .iter()
                .map(|c| format!("{}: {}", c.invariant_id, c.reason))
                .collect();
            return Err(SearchError::SubsystemError {
                subsystem: "generation_activation",
                source: reasons.join("; ").into(),
            });
        }

        // Step 4: Build resolved paths and swap.
        let mut vector_paths = BTreeMap::new();
        for art in &manifest.vector_artifacts {
            vector_paths.insert(art.path.clone(), verifier.resolve_path(&art.path));
        }
        let mut lexical_paths = BTreeMap::new();
        for art in &manifest.lexical_artifacts {
            lexical_paths.insert(art.path.clone(), verifier.resolve_path(&art.path));
        }

        let seq = self.activation_counter.fetch_add(1, Ordering::AcqRel) + 1;
        let new_gen = Arc::new(ActiveGeneration {
            manifest,
            activation_seq: seq,
            activated_at: now_millis,
            vector_paths,
            lexical_paths,
        });

        // Atomic swap under serialized writer guard: install new active and retain old for rollback.
        let previous_active = self
            .active
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        {
            let mut rollback = self
                .rollback
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            rollback.clone_from(&previous_active);
        }
        {
            let mut active = self
                .active
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *active = Some(Arc::clone(&new_gen));
        }

        Ok(new_gen)
    }

    /// Roll back to the previous generation.
    ///
    /// Returns `true` if rollback succeeded (previous generation existed),
    /// `false` if there was no previous generation to roll back to.
    pub fn rollback(&self) -> bool {
        let _write_guard = self
            .write_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prev = {
            self.rollback
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
        };
        prev.is_some_and(|prev_gen| {
            let mut active = self
                .active
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *active = Some(prev_gen);
            true
        })
    }

    /// Check if the controller is in a degraded state (no active generation).
    #[must_use]
    pub fn is_degraded(&self) -> bool {
        self.active
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_none()
    }
}

impl Default for GenerationController {
    fn default() -> Self {
        Self::new()
    }
}

fn activation_error(validation: &ValidationResult) -> SearchError {
    let messages: Vec<String> = validation
        .errors()
        .iter()
        .map(|f| f.message.clone())
        .collect();
    SearchError::InvalidConfig {
        field: "generation_manifest".into(),
        value: String::new(),
        reason: format!("manifest validation failed: {}", messages.join("; ")),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::*;

    fn sample_embedder() -> EmbedderRevision {
        EmbedderRevision {
            model_name: "potion-128M".into(),
            weights_hash: "abcdef1234567890".into(),
            dimension: 256,
            quantization: QuantizationFormat::F16,
        }
    }

    /// Test verifier that always passes.
    struct AlwaysPassVerifier {
        base_dir: String,
        runtime_embedders: BTreeMap<String, EmbedderRevision>,
    }

    impl AlwaysPassVerifier {
        fn new(base: &str) -> Self {
            let mut runtime_embedders = BTreeMap::new();
            runtime_embedders.insert("fast".into(), sample_embedder());
            Self {
                base_dir: base.into(),
                runtime_embedders,
            }
        }
    }

    impl ArtifactVerifier for AlwaysPassVerifier {
        fn verify(
            &self,
            manifest_path: &str,
            _checksum: &str,
            _expected_size: u64,
        ) -> ArtifactVerification {
            ArtifactVerification {
                path: manifest_path.into(),
                passed: true,
                detail: "mock: always passes".into(),
            }
        }

        fn resolve_path(&self, manifest_path: &str) -> String {
            format!("{}/{manifest_path}", self.base_dir)
        }

        fn runtime_embedder(&self, tier: &str) -> Option<EmbedderRevision> {
            self.runtime_embedders.get(tier).cloned()
        }
    }

    /// Test verifier that fails specific paths.
    struct FailingVerifier {
        fail_paths: Vec<String>,
    }

    impl ArtifactVerifier for FailingVerifier {
        fn verify(
            &self,
            manifest_path: &str,
            _checksum: &str,
            _expected_size: u64,
        ) -> ArtifactVerification {
            let passed = !self.fail_paths.iter().any(|p| p == manifest_path);
            ArtifactVerification {
                path: manifest_path.into(),
                passed,
                detail: if passed {
                    "ok".into()
                } else {
                    "checksum mismatch".into()
                },
            }
        }

        fn resolve_path(&self, manifest_path: &str) -> String {
            format!("/data/{manifest_path}")
        }
    }

    struct CustomHookVerifier {
        base: AlwaysPassVerifier,
        allowed_check: String,
    }

    impl CustomHookVerifier {
        fn new(base_dir: &str, allowed_check: &str) -> Self {
            Self {
                base: AlwaysPassVerifier::new(base_dir),
                allowed_check: allowed_check.to_owned(),
            }
        }
    }

    impl ArtifactVerifier for CustomHookVerifier {
        fn verify(
            &self,
            manifest_path: &str,
            checksum: &str,
            expected_size: u64,
        ) -> ArtifactVerification {
            self.base.verify(manifest_path, checksum, expected_size)
        }

        fn resolve_path(&self, manifest_path: &str) -> String {
            self.base.resolve_path(manifest_path)
        }

        fn runtime_embedder(&self, tier: &str) -> Option<EmbedderRevision> {
            self.base.runtime_embedder(tier)
        }

        fn check_custom_invariant(
            &self,
            check_name: &str,
            _manifest: &GenerationManifest,
            _previous: Option<&ActiveGeneration>,
        ) -> Option<(bool, String)> {
            Some((
                check_name == self.allowed_check,
                format!("custom check '{check_name}' evaluated by test verifier"),
            ))
        }
    }

    fn sample_manifest() -> GenerationManifest {
        let mut embedders = BTreeMap::new();
        embedders.insert("fast".into(), sample_embedder());

        let mut manifest = GenerationManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            generation_id: "gen-001".into(),
            manifest_hash: String::new(),
            commit_range: CommitRange { low: 1, high: 100 },
            build_started_at: 1_700_000_000_000,
            build_completed_at: 1_700_000_060_000,
            embedders,
            vector_artifacts: vec![VectorArtifact {
                path: "vectors/shard_0.fsvi".into(),
                size_bytes: 1024,
                checksum: "deadbeef".into(),
                vector_count: 100,
                dimension: 256,
                embedder_tier: EmbedderTierTag::Fast,
            }],
            lexical_artifacts: vec![LexicalArtifact {
                path: "lexical/segment_0".into(),
                size_bytes: 2048,
                checksum: "cafebabe".into(),
                document_count: 100,
            }],
            repair_descriptors: vec![RepairDescriptor {
                protected_artifact: "vectors/shard_0.fsvi".into(),
                sidecar_path: "vectors/shard_0.fsvi.fec".into(),
                source_symbols: 64,
                repair_symbols: 13,
                overhead_ratio: 0.2,
            }],
            activation_invariants: vec![ActivationInvariant {
                id: "all_artifacts".into(),
                description: "All artifacts verified".into(),
                kind: InvariantKind::AllArtifactsVerified,
            }],
            total_documents: 100,
            metadata: BTreeMap::new(),
        };
        manifest.manifest_hash = compute_manifest_hash(&manifest).expect("hash");
        manifest
    }

    #[test]
    fn controller_starts_degraded() {
        let ctrl = GenerationController::new();
        assert!(ctrl.is_degraded());
        assert!(ctrl.active().is_none());
        assert_eq!(ctrl.activation_seq(), 0);
    }

    #[test]
    fn successful_activation() {
        let ctrl = GenerationController::new();
        let verifier = AlwaysPassVerifier::new("/data/gen");
        let manifest = sample_manifest();

        let active_gen = ctrl
            .activate(manifest, &verifier, 1_700_000_100_000)
            .unwrap();
        assert_eq!(active_gen.activation_seq, 1);
        assert_eq!(active_gen.manifest.generation_id, "gen-001");
        assert!(!ctrl.is_degraded());
        assert_eq!(ctrl.activation_seq(), 1);

        // Resolved paths should include base dir.
        assert_eq!(
            active_gen.vector_paths.get("vectors/shard_0.fsvi").unwrap(),
            "/data/gen/vectors/shard_0.fsvi"
        );
    }

    #[test]
    fn activation_increments_seq() {
        let ctrl = GenerationController::new();
        let verifier = AlwaysPassVerifier::new("/data");

        let mut m1 = sample_manifest();
        m1.generation_id = "gen-001".into();
        m1.manifest_hash = compute_manifest_hash(&m1).expect("hash");
        ctrl.activate(m1, &verifier, 100).unwrap();
        assert_eq!(ctrl.activation_seq(), 1);

        let mut m2 = sample_manifest();
        m2.generation_id = "gen-002".into();
        m2.manifest_hash = compute_manifest_hash(&m2).expect("hash");
        ctrl.activate(m2, &verifier, 200).unwrap();
        assert_eq!(ctrl.activation_seq(), 2);

        let active = ctrl.active().unwrap();
        assert_eq!(active.manifest.generation_id, "gen-002");
    }

    #[test]
    fn invalid_manifest_rejected() {
        let ctrl = GenerationController::new();
        let verifier = AlwaysPassVerifier::new("/data");

        let mut bad = sample_manifest();
        bad.generation_id = String::new(); // Invalid.
        let err = ctrl.activate(bad, &verifier, 100).unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
        assert!(ctrl.is_degraded()); // No activation happened.
    }

    #[test]
    fn artifact_verification_failure_blocks_activation() {
        let ctrl = GenerationController::new();
        let verifier = FailingVerifier {
            fail_paths: vec!["vectors/shard_0.fsvi".into()],
        };
        let manifest = sample_manifest();

        let err = ctrl.activate(manifest, &verifier, 100).unwrap_err();
        assert!(matches!(err, SearchError::SubsystemError { .. }));
        assert!(ctrl.is_degraded());
    }

    #[test]
    fn rollback_restores_previous() {
        let ctrl = GenerationController::new();
        let verifier = AlwaysPassVerifier::new("/data");

        // Activate gen-001.
        let mut m1 = sample_manifest();
        m1.generation_id = "gen-001".into();
        m1.manifest_hash = compute_manifest_hash(&m1).expect("hash");
        ctrl.activate(m1, &verifier, 100).unwrap();

        // Activate gen-002.
        let mut m2 = sample_manifest();
        m2.generation_id = "gen-002".into();
        m2.manifest_hash = compute_manifest_hash(&m2).expect("hash");
        ctrl.activate(m2, &verifier, 200).unwrap();
        assert_eq!(ctrl.active().unwrap().manifest.generation_id, "gen-002");

        // Rollback to gen-001.
        assert!(ctrl.rollback());
        assert_eq!(ctrl.active().unwrap().manifest.generation_id, "gen-001");
        assert!(
            !ctrl.rollback(),
            "rollback slot should be consumed after a successful rollback"
        );
    }

    #[test]
    fn rollback_with_no_previous_returns_false() {
        let ctrl = GenerationController::new();
        assert!(!ctrl.rollback());
    }

    #[test]
    fn failed_activation_preserves_current() {
        let ctrl = GenerationController::new();
        let good_verifier = AlwaysPassVerifier::new("/data");
        let bad_verifier = FailingVerifier {
            fail_paths: vec!["vectors/shard_0.fsvi".into()],
        };

        // First successful activation.
        let m1 = sample_manifest();
        ctrl.activate(m1, &good_verifier, 100).unwrap();
        assert_eq!(ctrl.active().unwrap().manifest.generation_id, "gen-001");

        // Second activation fails — gen-001 should still be active.
        let mut m2 = sample_manifest();
        m2.generation_id = "gen-002".into();
        assert!(ctrl.activate(m2, &bad_verifier, 200).is_err());
        assert_eq!(ctrl.active().unwrap().manifest.generation_id, "gen-001");
    }

    #[test]
    fn invariant_all_artifacts_pass() {
        let manifest = sample_manifest();
        let verifier = AlwaysPassVerifier::new("/data");
        let checks = check_invariants(&manifest, &verifier, None);
        assert!(checks.iter().all(|c| c.passed));
    }

    #[test]
    fn invariant_all_artifacts_fail() {
        let manifest = sample_manifest();
        let verifier = FailingVerifier {
            fail_paths: vec!["vectors/shard_0.fsvi".into()],
        };
        let checks = check_invariants(&manifest, &verifier, None);
        assert!(checks.iter().any(|c| !c.passed));
    }

    #[test]
    fn invariant_vector_count_consistency() {
        let mut manifest = sample_manifest();
        manifest.activation_invariants = vec![ActivationInvariant {
            id: "vec_count".into(),
            description: "Vector count check".into(),
            kind: InvariantKind::VectorCountConsistency {
                expected_total: 100,
            },
        }];
        let verifier = AlwaysPassVerifier::new("/data");

        // Should pass: 100 vectors = 100 expected.
        let checks = check_invariants(&manifest, &verifier, None);
        assert!(checks[0].passed);

        // Should fail: expect 200 but only have 100.
        manifest.activation_invariants[0].kind = InvariantKind::VectorCountConsistency {
            expected_total: 200,
        };
        let checks = check_invariants(&manifest, &verifier, None);
        assert!(!checks[0].passed);
    }

    #[test]
    fn invariant_commit_continuity() {
        let mut manifest = sample_manifest();
        manifest.commit_range = CommitRange { low: 51, high: 100 };
        manifest.activation_invariants = vec![ActivationInvariant {
            id: "continuity".into(),
            description: "Commit continuity".into(),
            kind: InvariantKind::CommitContinuity { previous_high: 50 },
        }];
        let verifier = AlwaysPassVerifier::new("/data");

        // No previous generation: always passes.
        let checks = check_invariants(&manifest, &verifier, None);
        assert!(checks[0].passed);

        // Previous with matching high: passes.
        let mut prev_manifest = sample_manifest();
        prev_manifest.commit_range = CommitRange { low: 1, high: 50 };
        let prev = ActiveGeneration {
            manifest: prev_manifest,
            activation_seq: 1,
            activated_at: 100,
            vector_paths: BTreeMap::new(),
            lexical_paths: BTreeMap::new(),
        };
        let checks = check_invariants(&manifest, &verifier, Some(&prev));
        assert!(checks[0].passed);

        // Previous with different high: fails.
        let mut prev2 = prev.clone();
        prev2.manifest.commit_range.high = 49;
        let checks = check_invariants(&manifest, &verifier, Some(&prev2));
        assert!(!checks[0].passed);

        // Previous high matches but current low is not contiguous: fails.
        let mut non_contiguous = manifest;
        non_contiguous.commit_range = CommitRange { low: 60, high: 100 };
        let checks = check_invariants(&non_contiguous, &verifier, Some(&prev));
        assert!(!checks[0].passed);
    }

    #[test]
    fn invariant_custom_fails_without_verifier_hook() {
        let mut manifest = sample_manifest();
        manifest.activation_invariants = vec![ActivationInvariant {
            id: "custom_1".into(),
            description: "Custom check".into(),
            kind: InvariantKind::Custom {
                check_name: "my_check".into(),
            },
        }];
        let verifier = AlwaysPassVerifier::new("/data");
        let checks = check_invariants(&manifest, &verifier, None);
        assert!(!checks[0].passed);
        assert!(checks[0].reason.contains("not implemented"));
    }

    #[test]
    fn invariant_custom_uses_verifier_hook() {
        let mut manifest = sample_manifest();
        manifest.activation_invariants = vec![ActivationInvariant {
            id: "custom_1".into(),
            description: "Custom check".into(),
            kind: InvariantKind::Custom {
                check_name: "my_check".into(),
            },
        }];
        let verifier = CustomHookVerifier::new("/data", "my_check");
        let checks = check_invariants(&manifest, &verifier, None);
        assert!(checks[0].passed);
        assert!(checks[0].reason.contains("evaluated by test verifier"));
    }

    #[test]
    fn active_generation_serde_roundtrip() {
        let active_gen = ActiveGeneration {
            manifest: sample_manifest(),
            activation_seq: 42,
            activated_at: 1_700_000_000_000,
            vector_paths: BTreeMap::from([("v.fsvi".into(), "/data/v.fsvi".into())]),
            lexical_paths: BTreeMap::from([("l/seg".into(), "/data/l/seg".into())]),
        };
        let json = serde_json::to_string(&active_gen).expect("serialize");
        let back: ActiveGeneration = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            active_gen.manifest.generation_id,
            back.manifest.generation_id
        );
        assert_eq!(active_gen.activation_seq, back.activation_seq);
    }

    #[test]
    fn default_controller_is_degraded() {
        let ctrl = GenerationController::default();
        assert!(ctrl.is_degraded());
    }

    #[test]
    fn multiple_invariants_all_checked() {
        let mut manifest = sample_manifest();
        manifest.activation_invariants = vec![
            ActivationInvariant {
                id: "inv_1".into(),
                description: "Check 1".into(),
                kind: InvariantKind::AllArtifactsVerified,
            },
            ActivationInvariant {
                id: "inv_2".into(),
                description: "Check 2".into(),
                kind: InvariantKind::EmbedderRevisionMatch,
            },
            ActivationInvariant {
                id: "inv_3".into(),
                description: "Check 3".into(),
                kind: InvariantKind::VectorCountConsistency {
                    expected_total: 100,
                },
            },
        ];
        let verifier = AlwaysPassVerifier::new("/data");
        let checks = check_invariants(&manifest, &verifier, None);
        assert_eq!(checks.len(), 3);
        assert!(checks.iter().all(|c| c.passed));
    }
}
