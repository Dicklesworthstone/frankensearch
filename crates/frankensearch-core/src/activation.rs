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
    ActivationInvariant, EmbedderRevision, EmbedderTierTag, GenerationManifest, InvariantKind,
    ValidationResult, validate_manifest,
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
    /// return `None`. Activation requires an exact, valid identity for every
    /// declared tier, even when the manifest omits `EmbedderRevisionMatch` or
    /// contains only empty index images.
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

/// Evaluate mandatory admission and every additional manifest invariant.
///
/// Artifact verification and exact runtime embedder identity are always
/// required. Declared vector tiers must also exist in the embedder map and use
/// that identity's dimension. A manifest can add checks, never omit this baseline.
///
/// Configured checks retain their order and identifiers. Missing mandatory
/// checks are appended with `required_all_artifacts_verified` and
/// `required_embedder_revision_match` identifiers. Repeated artifact or embedder
/// checks reuse one evaluation, so listing a built-in cannot double verification
/// work. Every returned check must pass before activation proceeds.
pub fn check_invariants(
    manifest: &GenerationManifest,
    verifier: &dyn ArtifactVerifier,
    previous_generation: Option<&ActiveGeneration>,
) -> Vec<InvariantCheck> {
    let mut checks = Vec::new();
    let mut artifacts = None;
    let mut embedders = None;
    for invariant in &manifest.activation_invariants {
        let mut check = match &invariant.kind {
            InvariantKind::AllArtifactsVerified => artifacts
                .get_or_insert_with(|| {
                    evaluate_all_artifacts_verified(&invariant.id, manifest, verifier)
                })
                .clone(),
            InvariantKind::EmbedderRevisionMatch => embedders
                .get_or_insert_with(|| {
                    evaluate_embedder_revision_match(&invariant.id, manifest, verifier)
                })
                .clone(),
            _ => evaluate_invariant(invariant, manifest, verifier, previous_generation),
        };
        check.invariant_id.clone_from(&invariant.id);
        checks.push(check);
    }
    if artifacts.is_none() {
        checks.push(evaluate_all_artifacts_verified(
            "required_all_artifacts_verified",
            manifest,
            verifier,
        ));
    }
    if embedders.is_none() {
        checks.push(evaluate_embedder_revision_match(
            "required_embedder_revision_match",
            manifest,
            verifier,
        ));
    }
    checks
}

fn evaluate_invariant(
    inv: &ActivationInvariant,
    manifest: &GenerationManifest,
    verifier: &dyn ArtifactVerifier,
    previous: Option<&ActiveGeneration>,
) -> InvariantCheck {
    match &inv.kind {
        InvariantKind::AllArtifactsVerified => {
            evaluate_all_artifacts_verified(&inv.id, manifest, verifier)
        }
        InvariantKind::EmbedderRevisionMatch => {
            evaluate_embedder_revision_match(&inv.id, manifest, verifier)
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
    invariant_id: &str,
    manifest: &GenerationManifest,
    verifier: &dyn ArtifactVerifier,
) -> InvariantCheck {
    let mut failures = Vec::new();
    for art in &manifest.vector_artifacts {
        let v = verifier.verify(&art.path, &art.checksum, art.size_bytes);
        if v.path != art.path {
            failures.push(format!(
                "{}: verifier returned a different artifact path",
                art.path
            ));
        } else if !v.passed {
            failures.push(format!("{}: {}", art.path, v.detail));
        }
    }
    for art in &manifest.lexical_artifacts {
        let v = verifier.verify(&art.path, &art.checksum, art.size_bytes);
        if v.path != art.path {
            failures.push(format!(
                "{}: verifier returned a different artifact path",
                art.path
            ));
        } else if !v.passed {
            failures.push(format!("{}: {}", art.path, v.detail));
        }
    }

    if failures.is_empty() {
        InvariantCheck {
            invariant_id: invariant_id.to_owned(),
            passed: true,
            reason: "all artifacts verified".into(),
        }
    } else {
        InvariantCheck {
            invariant_id: invariant_id.to_owned(),
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
    invariant_id: &str,
    manifest: &GenerationManifest,
    verifier: &dyn ArtifactVerifier,
) -> InvariantCheck {
    let mut failures = Vec::new();
    for (index, artifact) in manifest.vector_artifacts.iter().enumerate() {
        let tier = match artifact.embedder_tier {
            EmbedderTierTag::Fast => "fast",
            EmbedderTierTag::Quality => "quality",
        };
        let Some(expected) = manifest.embedders.get(tier) else {
            failures.push(format!(
                "vector_artifacts[{index}] has no declared identity for tier '{tier}'"
            ));
            continue;
        };
        if artifact.dimension != expected.identity.space.dimension {
            failures.push(format!(
                "vector_artifacts[{index}] dimension {} differs from tier '{tier}' identity dimension {}",
                artifact.dimension, expected.identity.space.dimension,
            ));
        }
    }
    for (tier, expected) in &manifest.embedders {
        let safe_tier = if tier.len() <= 128 && !tier.chars().any(char::is_control) {
            tier.as_str()
        } else {
            "<redacted-tier>"
        };
        let Some(actual) = verifier.runtime_embedder(tier) else {
            failures.push(format!("runtime embedder missing for tier '{safe_tier}'"));
            continue;
        };
        if expected.validate().is_err() || actual.validate().is_err() {
            failures.push(format!(
                "tier '{safe_tier}' mismatch: expected or runtime identity failed canonical validation"
            ));
        } else if actual.fingerprint != expected.fingerprint {
            failures.push(format!(
                "tier '{safe_tier}' mismatch: expected model={} identity={} dim={}, got model={} identity={} dim={}",
                expected.identity.space.logical_model_id,
                expected.fingerprint,
                expected.identity.space.dimension,
                actual.identity.space.logical_model_id,
                actual.fingerprint,
                actual.identity.space.dimension,
            ));
        }
    }

    let passed = failures.is_empty();
    InvariantCheck {
        invariant_id: invariant_id.to_owned(),
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
    /// 3. Check exact runtime producer identity and all additional invariants.
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

        // Step 2 & 3: Mandatory artifact/producer admission cannot be disabled
        // by omitting an invariant. Manifest predicates add restrictions only.
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
        sample_embedder_for("potion-128M", 256)
    }

    fn sample_embedder_for(model: &str, dimension: u32) -> EmbedderRevision {
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(model, dimension);
        identity.storage.format = "fsvi-v2".to_owned();
        identity.storage.quantization = QuantizationFormat::F16;
        identity.storage.endianness = "little-endian".to_owned();
        identity.freeze().unwrap()
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

    /// Records admission hook calls and can refuse one required boundary.
    struct ObservedVerifier {
        base: AlwaysPassVerifier,
        fail_path: Option<String>,
        substitute_path: bool,
        custom_pass: bool,
        verified: Mutex<Vec<String>>,
        resolved: Mutex<Vec<String>>,
        producers: Mutex<Vec<String>>,
    }

    impl ObservedVerifier {
        fn new() -> Self {
            Self {
                base: AlwaysPassVerifier::new("/data"),
                fail_path: None,
                substitute_path: false,
                custom_pass: true,
                verified: Mutex::new(Vec::new()),
                resolved: Mutex::new(Vec::new()),
                producers: Mutex::new(Vec::new()),
            }
        }
    }

    impl ArtifactVerifier for ObservedVerifier {
        fn verify(&self, path: &str, checksum: &str, expected_size: u64) -> ArtifactVerification {
            self.verified.lock().unwrap().push(path.to_owned());
            let mut verification = self.base.verify(path, checksum, expected_size);
            if self.fail_path.as_deref() == Some(path) {
                verification.passed = false;
                verification.detail = "fixture artifact failed verification".to_owned();
            }
            if self.substitute_path {
                verification.path = "another-artifact".to_owned();
            }
            verification
        }

        fn resolve_path(&self, path: &str) -> String {
            self.resolved.lock().unwrap().push(path.to_owned());
            self.base.resolve_path(path)
        }

        fn runtime_embedder(&self, tier: &str) -> Option<EmbedderRevision> {
            self.producers.lock().unwrap().push(tier.to_owned());
            self.base.runtime_embedder(tier)
        }

        fn check_custom_invariant(
            &self,
            _check_name: &str,
            _manifest: &GenerationManifest,
            _previous: Option<&ActiveGeneration>,
        ) -> Option<(bool, String)> {
            Some((self.custom_pass, "fixture deployment predicate".to_owned()))
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

    // ─── bd-2cz8 tests begin ───

    #[test]
    fn artifact_verification_debug_clone_eq() {
        let v = ArtifactVerification {
            path: "a.fsvi".into(),
            passed: true,
            detail: "ok".into(),
        };
        let v2 = v.clone();
        assert_eq!(v, v2);
        assert_eq!(format!("{v:?}"), format!("{v2:?}"));
    }

    #[test]
    fn artifact_verification_not_equal_different_passed() {
        let v1 = ArtifactVerification {
            path: "a.fsvi".into(),
            passed: true,
            detail: "ok".into(),
        };
        let v2 = ArtifactVerification {
            path: "a.fsvi".into(),
            passed: false,
            detail: "ok".into(),
        };
        assert_ne!(v1, v2);
    }

    #[test]
    fn invariant_check_debug_clone_eq() {
        let c = InvariantCheck {
            invariant_id: "inv_1".into(),
            passed: true,
            reason: "ok".into(),
        };
        let c2 = c.clone();
        assert_eq!(c, c2);
        assert_eq!(format!("{c:?}"), format!("{c2:?}"));
    }

    #[test]
    fn invariant_check_not_equal_different_reason() {
        let c1 = InvariantCheck {
            invariant_id: "inv_1".into(),
            passed: true,
            reason: "ok".into(),
        };
        let c2 = InvariantCheck {
            invariant_id: "inv_1".into(),
            passed: true,
            reason: "different".into(),
        };
        assert_ne!(c1, c2);
    }

    #[test]
    fn active_generation_debug_clone() {
        let ag = ActiveGeneration {
            manifest: sample_manifest(),
            activation_seq: 1,
            activated_at: 100,
            vector_paths: BTreeMap::new(),
            lexical_paths: BTreeMap::new(),
        };
        let ag2 = ag.clone();
        assert_eq!(ag.activation_seq, ag2.activation_seq);
        assert_eq!(ag.activated_at, ag2.activated_at);
        assert_eq!(ag.manifest.generation_id, ag2.manifest.generation_id);
        let dbg = format!("{ag:?}");
        assert!(dbg.contains("ActiveGeneration"));
    }

    #[test]
    fn embedder_revision_mismatch_fails() {
        let mut manifest = sample_manifest();
        manifest.activation_invariants = vec![ActivationInvariant {
            id: "emb_rev".into(),
            description: "Embedder revision match".into(),
            kind: InvariantKind::EmbedderRevisionMatch,
        }];
        // Use a verifier whose runtime embedder has different model name.
        let mut verifier = AlwaysPassVerifier::new("/data");
        verifier.runtime_embedders.insert(
            "fast".into(),
            EmbedderRevision::explicit_test_model("different-model", 256),
        );
        let checks = check_invariants(&manifest, &verifier, None);
        assert_eq!(checks.len(), 2);
        assert!(!checks[0].passed);
        assert!(checks[0].reason.contains("mismatch"));
    }

    #[test]
    fn embedder_revision_match_passes() {
        let mut manifest = sample_manifest();
        manifest.activation_invariants = vec![ActivationInvariant {
            id: "emb_rev".into(),
            description: "Embedder revision match".into(),
            kind: InvariantKind::EmbedderRevisionMatch,
        }];
        let verifier = AlwaysPassVerifier::new("/data");
        let checks = check_invariants(&manifest, &verifier, None);
        assert_eq!(checks.len(), 2);
        assert!(checks[0].passed);
        assert!(checks[0].reason.contains("match manifest"));
    }

    #[test]
    fn embedder_revision_with_tampered_canonical_digest_fails() {
        let mut manifest = sample_manifest();
        manifest.activation_invariants = vec![ActivationInvariant {
            id: "emb_rev".into(),
            description: "Embedder revision match".into(),
            kind: InvariantKind::EmbedderRevisionMatch,
        }];
        let mut verifier = AlwaysPassVerifier::new("/data");
        let mut tampered = manifest.embedders["fast"].clone();
        tampered.fingerprint = "0".repeat(64);
        verifier.runtime_embedders.insert("fast".into(), tampered);

        let checks = check_invariants(&manifest, &verifier, None);
        assert_eq!(checks.len(), 2);
        assert!(!checks[0].passed);
        assert!(checks[0].reason.contains("mismatch"));
    }

    #[test]
    fn malformed_embedder_identity_diagnostics_are_bounded_and_redacted() {
        let mut manifest = sample_manifest();
        manifest.activation_invariants = vec![ActivationInvariant {
            id: "emb_rev".into(),
            description: "Embedder revision match".into(),
            kind: InvariantKind::EmbedderRevisionMatch,
        }];
        let expected = manifest.embedders.remove("fast").unwrap();
        manifest
            .embedders
            .insert("fast\nforged-log-line".to_owned(), expected.clone());
        let mut verifier = AlwaysPassVerifier::new("/data");
        let mut malformed = expected;
        malformed.identity.space.logical_model_id = "model\nforged-model-line".to_owned();
        verifier
            .runtime_embedders
            .insert("fast\nforged-log-line".to_owned(), malformed);

        let checks = check_invariants(&manifest, &verifier, None);
        assert_eq!(checks.len(), 2);
        assert!(!checks[0].passed);
        assert!(checks[0].reason.contains("<redacted-tier>"));
        assert!(checks[0].reason.contains("failed canonical validation"));
        assert!(!checks[0].reason.contains("forged-log-line"));
        assert!(!checks[0].reason.contains("forged-model-line"));
    }

    #[test]
    fn embedder_revision_missing_runtime_fails() {
        let mut manifest = sample_manifest();
        manifest.activation_invariants = vec![ActivationInvariant {
            id: "emb_rev".into(),
            description: "Embedder revision match".into(),
            kind: InvariantKind::EmbedderRevisionMatch,
        }];
        // Use a verifier with no runtime embedders.
        let mut verifier = AlwaysPassVerifier::new("/data");
        verifier.runtime_embedders.clear();
        let checks = check_invariants(&manifest, &verifier, None);
        assert!(!checks[0].passed);
        assert!(checks[0].reason.contains("missing"));
    }

    #[test]
    fn custom_invariant_failing_check_name() {
        let mut manifest = sample_manifest();
        manifest.activation_invariants = vec![ActivationInvariant {
            id: "custom_fail".into(),
            description: "Custom check".into(),
            kind: InvariantKind::Custom {
                check_name: "wrong_check".into(),
            },
        }];
        // CustomHookVerifier only passes "my_check", so "wrong_check" should fail.
        let verifier = CustomHookVerifier::new("/data", "my_check");
        let checks = check_invariants(&manifest, &verifier, None);
        assert!(!checks[0].passed);
    }

    #[test]
    fn lexical_artifact_verification_failure() {
        let mut manifest = sample_manifest();
        manifest.activation_invariants = vec![ActivationInvariant {
            id: "all_art".into(),
            description: "All artifacts verified".into(),
            kind: InvariantKind::AllArtifactsVerified,
        }];
        // Fail on lexical artifact path.
        let verifier = FailingVerifier {
            fail_paths: vec!["lexical/segment_0".into()],
        };
        let checks = check_invariants(&manifest, &verifier, None);
        assert!(!checks[0].passed);
        assert!(checks[0].reason.contains("lexical/segment_0"));
    }

    #[test]
    fn both_vector_and_lexical_failures_reported() {
        let mut manifest = sample_manifest();
        manifest.activation_invariants = vec![ActivationInvariant {
            id: "all_art".into(),
            description: "All artifacts verified".into(),
            kind: InvariantKind::AllArtifactsVerified,
        }];
        let verifier = FailingVerifier {
            fail_paths: vec!["vectors/shard_0.fsvi".into(), "lexical/segment_0".into()],
        };
        let checks = check_invariants(&manifest, &verifier, None);
        assert!(!checks[0].passed);
        assert!(checks[0].reason.contains("2 artifact(s) failed"));
    }

    /// Default `runtime_embedder` returns None.
    struct MinimalVerifier;

    impl ArtifactVerifier for MinimalVerifier {
        fn verify(
            &self,
            manifest_path: &str,
            _checksum: &str,
            _expected_size: u64,
        ) -> ArtifactVerification {
            ArtifactVerification {
                path: manifest_path.into(),
                passed: true,
                detail: "minimal".into(),
            }
        }

        fn resolve_path(&self, manifest_path: &str) -> String {
            manifest_path.into()
        }
    }

    #[test]
    fn default_runtime_embedder_returns_none() {
        let v = MinimalVerifier;
        assert!(v.runtime_embedder("fast").is_none());
        assert!(v.runtime_embedder("quality").is_none());
    }

    #[test]
    fn default_check_custom_invariant_returns_none() {
        let v = MinimalVerifier;
        let m = sample_manifest();
        assert!(v.check_custom_invariant("anything", &m, None).is_none());
    }

    #[test]
    fn empty_invariants_still_returns_mandatory_admission_checks() {
        let mut manifest = sample_manifest();
        manifest.activation_invariants.clear();
        let verifier = AlwaysPassVerifier::new("/data");
        let checks = check_invariants(&manifest, &verifier, None);
        assert_eq!(
            checks
                .iter()
                .map(|check| check.invariant_id.as_str())
                .collect::<Vec<_>>(),
            [
                "required_all_artifacts_verified",
                "required_embedder_revision_match"
            ]
        );
        assert!(checks.iter().all(|check| check.passed));
    }

    #[test]
    fn vector_count_consistency_multiple_artifacts() {
        let mut manifest = sample_manifest();
        manifest.vector_artifacts = vec![
            VectorArtifact {
                path: "vectors/shard_0.fsvi".into(),
                size_bytes: 1024,
                checksum: "aaa".into(),
                vector_count: 60,
                dimension: 256,
                embedder_tier: EmbedderTierTag::Fast,
            },
            VectorArtifact {
                path: "vectors/shard_1.fsvi".into(),
                size_bytes: 1024,
                checksum: "bbb".into(),
                vector_count: 40,
                dimension: 256,
                embedder_tier: EmbedderTierTag::Fast,
            },
        ];
        manifest.activation_invariants = vec![ActivationInvariant {
            id: "vec_count".into(),
            description: "Vector count check".into(),
            kind: InvariantKind::VectorCountConsistency {
                expected_total: 100,
            },
        }];
        let verifier = AlwaysPassVerifier::new("/data");
        let checks = check_invariants(&manifest, &verifier, None);
        assert!(checks[0].passed);
        assert!(checks[0].reason.contains("100"));
    }

    #[test]
    fn commit_continuity_saturating_add_at_max() {
        let mut manifest = sample_manifest();
        manifest.commit_range = CommitRange {
            low: u64::MAX,
            high: u64::MAX,
        };
        manifest.activation_invariants = vec![ActivationInvariant {
            id: "continuity".into(),
            description: "Commit continuity at max".into(),
            kind: InvariantKind::CommitContinuity {
                previous_high: u64::MAX - 1,
            },
        }];
        let prev_manifest_inner = {
            let mut m = sample_manifest();
            m.commit_range = CommitRange {
                low: 1,
                high: u64::MAX - 1,
            };
            m
        };
        let prev = ActiveGeneration {
            manifest: prev_manifest_inner,
            activation_seq: 1,
            activated_at: 100,
            vector_paths: BTreeMap::new(),
            lexical_paths: BTreeMap::new(),
        };
        let verifier = AlwaysPassVerifier::new("/data");
        // expected_low = (u64::MAX - 1).saturating_add(1) = u64::MAX
        // actual_low = u64::MAX → should pass
        let checks = check_invariants(&manifest, &verifier, Some(&prev));
        assert!(checks[0].passed);
    }

    #[test]
    fn commit_continuity_saturating_add_overflow() {
        let mut manifest = sample_manifest();
        manifest.commit_range = CommitRange { low: 1, high: 100 };
        manifest.activation_invariants = vec![ActivationInvariant {
            id: "continuity".into(),
            description: "Commit continuity overflow".into(),
            kind: InvariantKind::CommitContinuity {
                previous_high: u64::MAX,
            },
        }];
        let prev_manifest_inner = {
            let mut m = sample_manifest();
            m.commit_range = CommitRange {
                low: 1,
                high: u64::MAX,
            };
            m
        };
        let prev = ActiveGeneration {
            manifest: prev_manifest_inner,
            activation_seq: 1,
            activated_at: 100,
            vector_paths: BTreeMap::new(),
            lexical_paths: BTreeMap::new(),
        };
        let verifier = AlwaysPassVerifier::new("/data");
        // expected_low = u64::MAX.saturating_add(1) = u64::MAX (saturated)
        // actual_low = 1 → should fail (1 != u64::MAX)
        let checks = check_invariants(&manifest, &verifier, Some(&prev));
        assert!(!checks[0].passed);
    }

    #[test]
    fn rollback_after_three_activations_only_one_slot() {
        let ctrl = GenerationController::new();
        let verifier = AlwaysPassVerifier::new("/data");

        let mut m1 = sample_manifest();
        m1.generation_id = "gen-001".into();
        m1.manifest_hash = compute_manifest_hash(&m1).expect("hash");
        ctrl.activate(m1, &verifier, 100).unwrap();

        let mut m2 = sample_manifest();
        m2.generation_id = "gen-002".into();
        m2.manifest_hash = compute_manifest_hash(&m2).expect("hash");
        ctrl.activate(m2, &verifier, 200).unwrap();

        let mut m3 = sample_manifest();
        m3.generation_id = "gen-003".into();
        m3.manifest_hash = compute_manifest_hash(&m3).expect("hash");
        ctrl.activate(m3, &verifier, 300).unwrap();
        assert_eq!(ctrl.active().unwrap().manifest.generation_id, "gen-003");

        // Rollback restores gen-002 (the one just before gen-003).
        assert!(ctrl.rollback());
        assert_eq!(ctrl.active().unwrap().manifest.generation_id, "gen-002");

        // No further rollback possible.
        assert!(!ctrl.rollback());
    }

    #[test]
    fn activation_error_formats_validation_failures() {
        let mut manifest = sample_manifest();
        manifest.generation_id = String::new(); // triggers validation failure
        manifest.schema_version = 0; // triggers another validation failure
        let validation = validate_manifest(&manifest);
        assert!(!validation.is_valid());
        let err = activation_error(&validation);
        let msg = err.to_string();
        assert!(msg.contains("manifest validation failed"));
    }

    #[test]
    fn activation_with_empty_invariants_succeeds() {
        let ctrl = GenerationController::new();
        let verifier = AlwaysPassVerifier::new("/data");
        let mut manifest = sample_manifest();
        manifest.activation_invariants.clear();
        manifest.manifest_hash = compute_manifest_hash(&manifest).expect("hash");
        let result = ctrl.activate(manifest, &verifier, 100).unwrap();
        assert_eq!(result.activation_seq, 1);
        assert!(!ctrl.is_degraded());
    }

    #[test]
    fn empty_or_custom_only_invariants_cannot_bypass_required_admission() {
        for custom_only in [false, true] {
            for fault in [
                "vector",
                "lexical",
                "returned_path",
                "missing_runtime",
                "foreign_producer",
                "noncanonical_runtime",
                "undeclared_tier",
                "dimension",
            ] {
                let ctrl = GenerationController::new();
                let good = AlwaysPassVerifier::new("/data");
                let first = ctrl.activate(sample_manifest(), &good, 100).unwrap();
                let mut second = sample_manifest();
                second.generation_id = "gen-002".to_owned();
                second.manifest_hash = compute_manifest_hash(&second).unwrap();
                let current = ctrl.activate(second, &good, 200).unwrap();
                let mut candidate = sample_manifest();
                candidate.generation_id = "gen-003".to_owned();
                candidate.activation_invariants.clear();
                if custom_only {
                    candidate.activation_invariants.push(ActivationInvariant {
                        id: "deployment".to_owned(),
                        description: "A passing deployment predicate cannot waive admission"
                            .to_owned(),
                        kind: InvariantKind::Custom {
                            check_name: "deployment".to_owned(),
                        },
                    });
                }
                let mut verifier = ObservedVerifier::new();
                let expected_reason = match fault {
                    "vector" => {
                        verifier.fail_path = Some(candidate.vector_artifacts[0].path.clone());
                        "fixture artifact failed verification"
                    }
                    "lexical" => {
                        verifier.fail_path = Some(candidate.lexical_artifacts[0].path.clone());
                        "fixture artifact failed verification"
                    }
                    "returned_path" => {
                        verifier.substitute_path = true;
                        "verifier returned a different artifact path"
                    }
                    "missing_runtime" => {
                        verifier.base.runtime_embedders.clear();
                        "runtime embedder missing"
                    }
                    "foreign_producer" => {
                        let mut identity = sample_embedder().identity;
                        identity
                            .producer
                            .implementation_revision
                            .push_str("-another-producer");
                        verifier
                            .base
                            .runtime_embedders
                            .insert("fast".to_owned(), identity.freeze().unwrap());
                        "mismatch"
                    }
                    "noncanonical_runtime" => {
                        verifier
                            .base
                            .runtime_embedders
                            .get_mut("fast")
                            .unwrap()
                            .canonical_bytes
                            .push(0);
                        "failed canonical validation"
                    }
                    "undeclared_tier" => {
                        candidate.vector_artifacts[0].embedder_tier = EmbedderTierTag::Quality;
                        "no declared identity for tier 'quality'"
                    }
                    "dimension" => {
                        candidate.vector_artifacts[0].dimension += 1;
                        "differs from tier 'fast' identity dimension"
                    }
                    _ => unreachable!("fixture fault"), // ubs:ignore — cfg(test) fixture assertion.
                };
                candidate.manifest_hash = compute_manifest_hash(&candidate).unwrap();
                assert!(validate_manifest(&candidate).is_valid(), "{fault}");
                let error = ctrl.activate(candidate, &verifier, 300).unwrap_err();
                assert!(matches!(
                    &error,
                    SearchError::SubsystemError {
                        subsystem: "generation_activation",
                        ..
                    }
                ));
                assert!(
                    error.to_string().contains(expected_reason),
                    "{fault}: {error}"
                );
                assert!(verifier.resolved.lock().unwrap().is_empty(), "{fault}");
                assert_eq!(ctrl.activation_seq(), 2, "{fault}");
                assert!(Arc::ptr_eq(&ctrl.active().unwrap(), &current), "{fault}");
                // Refusal preserves the rollback slot as well as the active Arc.
                assert!(ctrl.rollback(), "{fault}");
                assert!(Arc::ptr_eq(&ctrl.active().unwrap(), &first), "{fault}");
            }
        }
    }

    #[test]
    fn repeated_mandatory_predicates_reuse_one_artifact_and_producer_evaluation() {
        let mut manifest = sample_manifest();
        manifest.activation_invariants = [
            ("artifacts-a", InvariantKind::AllArtifactsVerified),
            ("producer-a", InvariantKind::EmbedderRevisionMatch),
            ("artifacts-b", InvariantKind::AllArtifactsVerified),
            ("producer-b", InvariantKind::EmbedderRevisionMatch),
        ]
        .into_iter()
        .map(|(id, kind)| ActivationInvariant {
            id: id.to_owned(),
            description: String::new(),
            kind,
        })
        .collect();
        let verifier = ObservedVerifier::new();
        let checks = check_invariants(&manifest, &verifier, None);
        assert!(checks.iter().all(|check| check.passed));
        assert_eq!(
            checks
                .iter()
                .map(|check| check.invariant_id.as_str())
                .collect::<Vec<_>>(),
            ["artifacts-a", "producer-a", "artifacts-b", "producer-b"]
        );
        assert_eq!(
            *verifier.verified.lock().unwrap(),
            ["vectors/shard_0.fsvi", "lexical/segment_0"]
        );
        assert_eq!(*verifier.producers.lock().unwrap(), ["fast"]);
    }

    #[test]
    fn mandatory_admission_preserves_tier_shapes_physical_rows_and_cumulative_ranges() {
        for shape in ["fast", "quality", "progressive", "shards", "empty"] {
            let mut manifest = sample_manifest();
            manifest.activation_invariants.clear();
            // Physical rows can include tombstones. They need not equal the
            // live document count, nor the sum across two independently stored tiers.
            manifest.vector_artifacts[0].vector_count = 120;
            match shape {
                "quality" => {
                    manifest.embedders = BTreeMap::from([(
                        "quality".to_owned(),
                        sample_embedder_for("quality-model", 384),
                    )]);
                    manifest.vector_artifacts[0].embedder_tier = EmbedderTierTag::Quality;
                    manifest.vector_artifacts[0].dimension = 384;
                }
                "progressive" => {
                    manifest.embedders.insert(
                        "quality".to_owned(),
                        sample_embedder_for("quality-model", 384),
                    );
                    let mut quality = manifest.vector_artifacts[0].clone();
                    quality.path = "vectors/quality.fsvi".to_owned();
                    quality.embedder_tier = EmbedderTierTag::Quality;
                    quality.dimension = 384;
                    manifest.vector_artifacts.push(quality);
                }
                "shards" => {
                    manifest.vector_artifacts[0].vector_count = 70;
                    let mut shard = manifest.vector_artifacts[0].clone();
                    shard.path = "vectors/shard_1.fsvi".to_owned();
                    shard.vector_count = 50;
                    manifest.vector_artifacts.push(shard);
                }
                "empty" => {
                    manifest.total_documents = 0;
                    manifest.vector_artifacts[0].vector_count = 0;
                    manifest.lexical_artifacts[0].document_count = 0;
                }
                "fast" => {}
                _ => unreachable!("fixture topology"), // ubs:ignore — cfg(test) fixture assertion.
            }
            manifest.manifest_hash = compute_manifest_hash(&manifest).unwrap();
            assert!(validate_manifest(&manifest).is_valid(), "{shape}");
            let mut verifier = ObservedVerifier::new();
            verifier.base.runtime_embedders = manifest.embedders.clone();
            let ctrl = GenerationController::new();
            let retained = ctrl.activate(manifest.clone(), &verifier, 100).unwrap();
            assert_eq!(
                verifier.verified.lock().unwrap().len(),
                manifest.vector_artifacts.len() + manifest.lexical_artifacts.len()
            );
            assert_eq!(
                *verifier.producers.lock().unwrap(),
                manifest.embedders.keys().cloned().collect::<Vec<_>>()
            );

            manifest.generation_id = "gen-002".to_owned();
            manifest.commit_range.high = 200;
            manifest.manifest_hash = compute_manifest_hash(&manifest).unwrap();
            let next = ctrl.activate(manifest, &verifier, 200).unwrap();
            assert_eq!(
                next.manifest.commit_range,
                CommitRange { low: 1, high: 200 }
            );
            assert_eq!(
                retained.manifest.commit_range,
                CommitRange { low: 1, high: 100 }
            );
        }
    }

    #[test]
    fn empty_generation_still_requires_the_default_verifier_to_declare_runtime_identity() {
        let mut manifest = sample_manifest();
        manifest.activation_invariants.clear();
        manifest.total_documents = 0;
        manifest.vector_artifacts[0].vector_count = 0;
        manifest.lexical_artifacts[0].document_count = 0;
        manifest.manifest_hash = compute_manifest_hash(&manifest).unwrap();
        let ctrl = GenerationController::new();
        let error = ctrl.activate(manifest, &MinimalVerifier, 100).unwrap_err();
        assert!(error.to_string().contains("runtime embedder missing"));
        assert!(ctrl.is_degraded());
        assert_eq!(ctrl.activation_seq(), 0);
    }

    #[test]
    fn an_unused_declared_tier_still_requires_its_exact_runtime_identity() {
        let mut manifest = sample_manifest();
        manifest.activation_invariants.clear();
        manifest.embedders.insert(
            "quality".to_owned(),
            sample_embedder_for("quality-model", 384),
        );
        manifest.manifest_hash = compute_manifest_hash(&manifest).unwrap();
        let mut verifier = ObservedVerifier::new();
        let ctrl = GenerationController::new();
        let error = ctrl.activate(manifest.clone(), &verifier, 100).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("runtime embedder missing for tier 'quality'")
        );
        assert!(ctrl.is_degraded());
        verifier.base.runtime_embedders = manifest.embedders.clone();
        assert!(ctrl.activate(manifest, &verifier, 200).is_ok());
    }

    #[test]
    fn custom_predicates_still_add_restrictions_after_mandatory_checks_pass() {
        let mut manifest = sample_manifest();
        manifest.activation_invariants = vec![ActivationInvariant {
            id: "deployment".to_owned(),
            description: "Additional deployment restriction".to_owned(),
            kind: InvariantKind::Custom {
                check_name: "deployment".to_owned(),
            },
        }];
        manifest.manifest_hash = compute_manifest_hash(&manifest).unwrap();
        let mut verifier = ObservedVerifier::new();
        verifier.custom_pass = false;
        let ctrl = GenerationController::new();
        let error = ctrl.activate(manifest, &verifier, 100).unwrap_err();
        assert!(error.to_string().contains("deployment"));
        assert_eq!(verifier.verified.lock().unwrap().len(), 2);
        assert_eq!(*verifier.producers.lock().unwrap(), ["fast"]);
        assert!(verifier.resolved.lock().unwrap().is_empty());
        assert!(ctrl.is_degraded());
    }

    #[test]
    fn active_generation_resolved_lexical_paths() {
        let ctrl = GenerationController::new();
        let verifier = AlwaysPassVerifier::new("/mnt/gen");
        let manifest = sample_manifest();
        let result = ctrl.activate(manifest, &verifier, 100).unwrap();
        assert_eq!(
            result.lexical_paths.get("lexical/segment_0").unwrap(),
            "/mnt/gen/lexical/segment_0"
        );
    }

    #[test]
    fn activation_records_timestamp() {
        let ctrl = GenerationController::new();
        let verifier = AlwaysPassVerifier::new("/data");
        let manifest = sample_manifest();
        let timestamp = 1_700_555_000_000_u64;
        let result = ctrl.activate(manifest, &verifier, timestamp).unwrap();
        assert_eq!(result.activated_at, timestamp);
    }

    #[test]
    fn is_degraded_after_single_activation_then_no_rollback() {
        let ctrl = GenerationController::new();
        let verifier = AlwaysPassVerifier::new("/data");
        let manifest = sample_manifest();
        ctrl.activate(manifest, &verifier, 100).unwrap();
        assert!(!ctrl.is_degraded());
        // No rollback slot yet (no previous gen), so rollback returns false.
        assert!(!ctrl.rollback());
        // Controller still has the activated generation.
        assert!(!ctrl.is_degraded());
    }

    #[test]
    fn concurrent_arc_snapshot_isolation() {
        let ctrl = GenerationController::new();
        let verifier = AlwaysPassVerifier::new("/data");

        let mut m1 = sample_manifest();
        m1.generation_id = "gen-001".into();
        m1.manifest_hash = compute_manifest_hash(&m1).expect("hash");
        ctrl.activate(m1, &verifier, 100).unwrap();

        // Acquire snapshot before second activation.
        let snapshot = ctrl.active().unwrap();
        assert_eq!(snapshot.manifest.generation_id, "gen-001");

        // Activate gen-002.
        let mut m2 = sample_manifest();
        m2.generation_id = "gen-002".into();
        m2.manifest_hash = compute_manifest_hash(&m2).expect("hash");
        ctrl.activate(m2, &verifier, 200).unwrap();

        // Snapshot still sees gen-001 (Arc isolation).
        assert_eq!(snapshot.manifest.generation_id, "gen-001");
        // Current sees gen-002.
        assert_eq!(ctrl.active().unwrap().manifest.generation_id, "gen-002");
    }

    #[test]
    fn vector_count_consistency_zero_expected_and_empty() {
        let mut manifest = sample_manifest();
        manifest.vector_artifacts.clear();
        manifest.activation_invariants = vec![ActivationInvariant {
            id: "vec_count".into(),
            description: "Zero vectors".into(),
            kind: InvariantKind::VectorCountConsistency { expected_total: 0 },
        }];
        let verifier = AlwaysPassVerifier::new("/data");
        let checks = check_invariants(&manifest, &verifier, None);
        assert!(checks[0].passed);
        assert!(checks[0].reason.contains('0'));
    }

    // ─── bd-2cz8 tests end ───
}
