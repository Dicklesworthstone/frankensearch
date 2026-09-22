//! Explicitly selected, artifact-pinned native `Model2Vec` producers.
//!
//! Registration is not qualification: the caller supplies a previously measured
//! certificate, and loading verifies the real tokenizer/matrix execution before
//! exposing an embedder. This never invents a certificate from the loaded model.

use std::fmt;
use std::path::Path;
use std::sync::{Arc, PoisonError};

use frankensearch_core::generation::{EmbeddingIdentityBundleV1, GoldenVectorCertificateV1};
use frankensearch_core::{SearchError, SearchResult};

use crate::model_manifest::{
    MODEL_CONFORMANCE_TEXTS_V1, ModelArtifactManifestV1, ModelManifest, ModelTier,
};

use super::{
    Model2VecEmbedder, REQUIRED_FILES, SharedModelKey, canonical_model_dir, shared_model_cache,
    validate_registered_execution_contract,
};

/// One explicitly selected native `Model2Vec` model and its qualified certificate.
///
/// A registration owns both manifests so download selection, byte verification,
/// model identity, dimension, and producer qualification cannot drift apart.
/// Execution semantics are generated from this build's native `Model2Vec` adapter,
/// not accepted as caller-supplied backend names or implementation revisions.
///
/// The caller is responsible for selecting trustworthy, immutable artifact pins
/// and an independently qualified certificate for those artifacts. Construction
/// checks the registration; it does not download, load, or qualify the model.
/// [`Model2VecEmbedder::load_registered`] checks the actual output bits.
#[derive(Clone)]
pub struct RegisteredModel2Vec {
    download_manifest: ModelManifest,
    artifact_manifest: ModelArtifactManifestV1,
}

impl fmt::Debug for RegisteredModel2Vec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RegisteredModel2Vec")
            .field("dimension", &self.artifact_manifest.dimension)
            .finish_non_exhaustive()
    }
}

impl RegisteredModel2Vec {
    /// Register a pinned model using this build's native execution contract.
    ///
    /// The bundle must contain exactly `model.safetensors` and `tokenizer.json`,
    /// have a fixed dimension and fast-tier assignment, and pin every artifact's
    /// size and SHA-256 at an immutable upstream revision. The certificate must
    /// describe the unchanged, ordered [`MODEL_CONFORMANCE_TEXTS_V1`] corpus.
    /// The vectors' digest is supplied by the caller, never learned at load time.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` for incomplete pins, an unsupported artifact
    /// layout, or a certificate with the wrong corpus, count, or dimension.
    pub fn new(
        download_manifest: ModelManifest,
        provider: &str,
        golden_vectors: GoldenVectorCertificateV1,
    ) -> SearchResult<Self> {
        if download_manifest.tier != Some(ModelTier::Fast) {
            return Err(registration_error(
                "the native Model2Vec registration must serve the fast tier",
            ));
        }
        if download_manifest.files.len() != REQUIRED_FILES.len()
            || !REQUIRED_FILES.iter().all(|required| {
                download_manifest
                    .files
                    .iter()
                    .any(|file| file.name == *required)
            })
        {
            return Err(registration_error(
                "the native Model2Vec bundle must contain exactly model.safetensors and tokenizer.json",
            ));
        }

        // Reuse the implemented adapter contract, not Potion's artifact/space
        // identity. Each selected model supplies its own pins and certificate.
        let mut execution = ModelArtifactManifestV1::potion_128m_native()?.execution;
        if golden_vectors.corpus_sha256 != execution.golden_vectors.corpus_sha256
            || golden_vectors.vector_count != execution.golden_vectors.vector_count
        {
            return Err(registration_error(
                "the certificate must bind the ordered native Model2Vec conformance corpus",
            ));
        }
        execution.golden_vectors = golden_vectors;
        let artifact_manifest = ModelArtifactManifestV1::from_download_manifest(
            &download_manifest,
            provider,
            execution,
        )?;
        Ok(Self {
            download_manifest,
            artifact_manifest,
        })
    }

    /// Select the existing multilingual Potion model without changing its identity.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the built-in pinned registration is inconsistent.
    pub fn potion_128m() -> SearchResult<Self> {
        Ok(Self {
            download_manifest: ModelManifest::potion_128m(),
            artifact_manifest: ModelArtifactManifestV1::potion_128m_native()?,
        })
    }

    /// Operational model identifier; compatibility still requires the full identity.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.artifact_manifest.logical_model_id
    }

    /// Output dimension declared by the selected artifact and execution contract.
    #[must_use]
    pub const fn dimension(&self) -> u32 {
        self.artifact_manifest.dimension
    }

    /// Exact download manifest to pass to the existing consent-aware downloader.
    #[must_use]
    pub const fn download_manifest(&self) -> &ModelManifest {
        &self.download_manifest
    }

    /// Frozen artifact/execution registration used for admission and identity.
    #[must_use]
    pub const fn artifact_manifest(&self) -> &ModelArtifactManifestV1 {
        &self.artifact_manifest
    }

    fn admit(&self, model_dir: &Path) -> SearchResult<EmbeddingIdentityBundleV1> {
        // This runs on EVERY load, including a shared-cache hit. A fingerprint
        // key alone cannot detect files replaced under an already-resident model.
        let verified = self
            .artifact_manifest
            .verify_dir_cached(&self.download_manifest, model_dir)?;
        let identity = verified.identity_bundle(
            frankensearch_core::generation::QuantizationFormat::F32,
            "in-memory-f32-v1",
        )?;
        validate_registered_execution_contract(&identity)?;
        Ok(identity)
    }

    fn verify_execution(&self, embedder: &Model2VecEmbedder) -> SearchResult<()> {
        if embedder.tokenizer.get_padding().is_some() {
            return Err(registration_error(
                "the native Model2Vec execution contract does not permit tokenizer padding",
            ));
        }
        let probe = embedder.embed_batch_sync(&MODEL_CONFORMANCE_TEXTS_V1)?;
        self.artifact_manifest
            .execution
            .golden_vectors
            .verify_exact_f32(&MODEL_CONFORMANCE_TEXTS_V1, &probe)
            .map_err(|_| SearchError::ModelLoadFailed {
                path: embedder.model_dir.clone(),
                source: "native Model2Vec execution does not match the selected producer certificate; use a qualified model/runtime registration before rebuilding the index"
                    .into(),
            })
    }

    fn load_admitted(
        &self,
        model_dir: &Path,
        identity: EmbeddingIdentityBundleV1,
    ) -> SearchResult<Model2VecEmbedder> {
        let embedder = Model2VecEmbedder::load_preverified(model_dir, self.id(), identity)?;
        self.verify_execution(&embedder)?;
        // Recheck admission before publication as well: a detected source
        // mutation during loading must not leave an instance in the cache.
        self.admit(model_dir)?;
        Ok(embedder)
    }
}

impl Model2VecEmbedder {
    /// Load an explicitly selected, pinned native `Model2Vec` model.
    ///
    /// Unlike [`Self::load_with_name`], the registration chooses the actual artifacts,
    /// space, dimension, and producer rather than changing a display name.
    /// This reuses the bounded single-matrix loader and checks the selected
    /// producer's exact certificate before returning. No download is implicit.
    ///
    /// # Errors
    ///
    /// Returns an artifact-admission, tokenizer, matrix-shape, or producer
    /// qualification error. Failed qualification never returns an embedder.
    pub fn load_registered(
        model_dir: impl AsRef<Path>,
        registration: &RegisteredModel2Vec,
    ) -> SearchResult<Self> {
        let model_dir = model_dir.as_ref();
        let identity = registration.admit(model_dir)?;
        registration.load_admitted(model_dir, identity)
    }

    /// Load a selected model with process-wide, weak-reference sharing.
    ///
    /// Every call re-admits the selected files before cache lookup. The complete
    /// identity is part of the key, so equal names or dimensions cannot mix two
    /// models. Construction and qualification share the existing single-flight
    /// lock; an unqualified instance is never inserted into the cache.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::load_registered`], including when an
    /// artifact changes while an older instance is still resident.
    #[allow(clippy::significant_drop_tightening)]
    pub fn load_shared_registered(
        model_dir: impl AsRef<Path>,
        registration: &RegisteredModel2Vec,
    ) -> SearchResult<Arc<Self>> {
        let model_dir = model_dir.as_ref();
        let identity = registration.admit(model_dir)?;
        let key = SharedModelKey {
            dir: canonical_model_dir(model_dir),
            name: registration.id().to_owned(),
            identity: identity.fingerprint(),
        };
        let mut cache = shared_model_cache()
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(existing) = cache.get(&key).and_then(std::sync::Weak::upgrade) {
            // The legacy built-in loader shares this cache. Therefore even a
            // cache hit must prove the selected certificate, not assume that a
            // prior constructor executed it merely because the identity agrees.
            registration.verify_execution(&existing)?;
            registration.admit(model_dir)?;
            return Ok(existing);
        }
        let embedder = Arc::new(registration.load_admitted(model_dir, identity)?);
        cache.retain(|_, weak| weak.strong_count() > 0);
        cache.insert(key, Arc::downgrade(&embedder));
        Ok(embedder)
    }
}

fn registration_error(reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "model2vec.registration".to_owned(),
        value: "invalid".to_owned(),
        reason: reason.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Barrier;

    use frankensearch_core::traits::Embedder;
    use sha2::{Digest, Sha256};

    use crate::model_manifest::ModelFile;

    use super::*;

    const PROVIDER: &str = "registered-model2vec-fixture";

    fn sha256_hex(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut encoded = String::with_capacity(64);
        for &byte in &Sha256::digest(bytes) {
            encoded.push(char::from(HEX[usize::from(byte >> 4)]));
            encoded.push(char::from(HEX[usize::from(byte & 15)]));
        }
        encoded
    }

    // These are artifact-admission fixtures, not evidence that any upstream
    // model is qualified. Their expected vectors are analytical unit vectors,
    // independent of the loader or embedding implementation being exercised.
    fn fixture(id: &str, dimension: u32, axis: usize) -> (tempfile::TempDir, RegisteredModel2Vec) {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": {"type": "Lowercase"},
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "hello": 1, "world": 2},
                "unk_token": "[UNK]"
            }
        });
        fs::write(
            dir.path().join("tokenizer.json"),
            serde_json::to_vec(&tokenizer).unwrap(),
        )
        .unwrap();

        let width = usize::try_from(dimension).unwrap();
        let mut unit = vec![0.0_f32; width];
        unit[axis] = 1.0;
        let values = unit.repeat(3);
        let data = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let mut header = serde_json::to_vec(&serde_json::json!({
            "embeddings": {
                "dtype": "F32",
                "shape": [3, width],
                "data_offsets": [0, data.len()]
            }
        }))
        .unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut weights = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
        weights.extend_from_slice(&header);
        weights.extend_from_slice(&data);
        fs::write(dir.path().join("model.safetensors"), weights).unwrap();

        let files = REQUIRED_FILES
            .iter()
            .map(|name| {
                let bytes = fs::read(dir.path().join(name)).unwrap();
                ModelFile {
                    name: (*name).to_owned(),
                    sha256: sha256_hex(&bytes),
                    size: u64::try_from(bytes.len()).unwrap(),
                    url: None,
                }
            })
            .collect::<Vec<_>>();
        let download_manifest = ModelManifest {
            id: id.to_owned(),
            version: "1".to_owned(),
            display_name: None,
            description: None,
            repo: "fixtures/registered-model2vec".to_owned(),
            revision: "a".repeat(40),
            download_size_bytes: files.iter().map(|file| file.size).sum(),
            files,
            license: "MIT".to_owned(),
            dimension: Some(dimension),
            tier: Some(ModelTier::Fast),
        };
        let expected = vec![unit; MODEL_CONFORMANCE_TEXTS_V1.len()];
        let certificate =
            GoldenVectorCertificateV1::from_exact_f32(&MODEL_CONFORMANCE_TEXTS_V1, &expected)
                .unwrap();
        let registration = RegisteredModel2Vec::new(download_manifest, PROVIDER, certificate)
            .expect("fully pinned analytical fixture");
        (dir, registration)
    }

    fn certificate(registration: &RegisteredModel2Vec) -> GoldenVectorCertificateV1 {
        registration
            .artifact_manifest
            .execution
            .golden_vectors
            .clone()
    }

    #[test]
    fn builtin_registration_preserves_the_existing_frozen_identity() {
        let selected = RegisteredModel2Vec::potion_128m().unwrap();
        assert_eq!(selected.download_manifest(), &ModelManifest::potion_128m());
        assert_eq!(
            selected.artifact_manifest(),
            &ModelArtifactManifestV1::potion_128m_native().unwrap()
        );
        assert_eq!(selected.id(), super::super::DEFAULT_MODEL_NAME);
        assert_eq!(selected.dimension(), 256);
    }

    #[test]
    fn selected_models_load_their_own_dimensions_values_and_identities() {
        for (dimension, axis) in [(2, 0), (3, 2), (8, 5)] {
            let (dir, selected) = fixture("selected-model", dimension, axis);
            let embedder = Model2VecEmbedder::load_registered(dir.path(), &selected).unwrap();
            assert_eq!(embedder.id(), selected.id());
            assert_eq!(embedder.dimension(), usize::try_from(dimension).unwrap());
            let values = embedder.embed_sync("hello world").unwrap();
            for (index, value) in values.iter().enumerate() {
                let expected = if index == axis { 1.0_f32 } else { 0.0_f32 };
                assert_eq!(value.to_bits(), expected.to_bits());
            }
            assert!(
                embedder
                    .embed_sync("")
                    .unwrap()
                    .iter()
                    .all(|value| *value == 0.0)
            );
            let expected_identity = selected.admit(dir.path()).unwrap();
            assert_eq!(embedder.identity().unwrap(), &expected_identity);
            assert_eq!(expected_identity.space.logical_model_id, "selected-model");
            assert_eq!(expected_identity.producer.backend, "model2vec-native");
        }
    }

    #[test]
    fn shared_selected_model_reuses_the_qualified_instance() {
        let (dir, selected) = fixture("shared-selected", 2, 0);
        let first = Model2VecEmbedder::load_shared_registered(dir.path(), &selected).unwrap();
        let second = Model2VecEmbedder::load_shared_registered(dir.path(), &selected).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn padding_cannot_masquerade_as_the_native_no_padding_contract() {
        let (dir, selected) = fixture("padding-control", 2, 0);
        let mut embedder = Model2VecEmbedder::load_registered(dir.path(), &selected).unwrap();
        embedder
            .tokenizer
            .with_padding(Some(tokenizers::PaddingParams::default()));
        assert!(selected.verify_execution(&embedder).is_err());
    }

    #[test]
    fn same_directory_name_and_dimension_cannot_mix_two_producers() {
        let (dir, first_registration) = fixture("same-name", 2, 0);
        let (other_dir, second_registration) = fixture("same-name", 2, 1);
        let first =
            Model2VecEmbedder::load_shared_registered(dir.path(), &first_registration).unwrap();
        let second_bytes = fs::read(other_dir.path().join("model.safetensors")).unwrap();
        fs::write(dir.path().join("model.safetensors"), second_bytes).unwrap();
        let second =
            Model2VecEmbedder::load_shared_registered(dir.path(), &second_registration).unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
        assert_ne!(first.identity().unwrap(), second.identity().unwrap());
        assert_eq!(first.embed_sync("hello").unwrap(), vec![1.0, 0.0]);
        assert_eq!(second.embed_sync("hello").unwrap(), vec![0.0, 1.0]);
        assert!(
            Model2VecEmbedder::load_shared_registered(dir.path(), &first_registration).is_err()
        );
    }

    #[test]
    fn mutated_artifacts_are_rejected_before_a_resident_cache_hit() {
        let (dir, selected) = fixture("mutation-control", 2, 0);
        let resident = Model2VecEmbedder::load_shared_registered(dir.path(), &selected).unwrap();
        for name in REQUIRED_FILES {
            let path = dir.path().join(name);
            let original = fs::read(&path).unwrap();
            let mut changed = original.clone();
            let last = changed.last_mut().unwrap();
            *last ^= 1;
            fs::write(&path, changed).unwrap();
            assert!(Model2VecEmbedder::load_shared_registered(dir.path(), &selected).is_err());
            // A failed new admission does not mutate an already-owned instance.
            assert_eq!(resident.embed_sync("hello").unwrap(), vec![1.0, 0.0]);
            fs::write(&path, original).unwrap();
            let restored =
                Model2VecEmbedder::load_shared_registered(dir.path(), &selected).unwrap();
            assert!(Arc::ptr_eq(&resident, &restored));
        }
    }

    #[test]
    fn wrong_certificate_is_rejected_and_never_published_to_shared_cache() {
        let (dir, selected) = fixture("certificate-control", 2, 0);
        let mut wrong_certificate = certificate(&selected);
        wrong_certificate.vectors_sha256 = "0".repeat(64);
        let wrong = RegisteredModel2Vec::new(
            selected.download_manifest().clone(),
            PROVIDER,
            wrong_certificate,
        )
        .unwrap();
        let error = Model2VecEmbedder::load_shared_registered(dir.path(), &wrong).unwrap_err();
        assert!(matches!(error, SearchError::ModelLoadFailed { .. }));
        let key = SharedModelKey {
            dir: canonical_model_dir(dir.path()),
            name: wrong.id().to_owned(),
            identity: wrong.admit(dir.path()).unwrap().fingerprint(),
        };
        assert!(!shared_model_cache().lock().unwrap().contains_key(&key));
        let valid = Model2VecEmbedder::load_shared_registered(dir.path(), &selected).unwrap();
        assert_eq!(valid.embed_sync("world").unwrap(), vec![1.0, 0.0]);
        assert!(Model2VecEmbedder::load_registered(dir.path(), &wrong).is_err());
    }

    #[test]
    fn a_legacy_cache_entry_does_not_bypass_selected_execution_qualification() {
        let (dir, selected) = fixture("legacy-cache-control", 2, 0);
        let mut wrong_certificate = certificate(&selected);
        wrong_certificate.vectors_sha256 = "0".repeat(64);
        let wrong = RegisteredModel2Vec::new(
            selected.download_manifest().clone(),
            PROVIDER,
            wrong_certificate,
        )
        .unwrap();
        // Model the existing legacy constructor: byte-admitted but without the
        // selected-registration execution probe. This is not a new public bypass.
        let identity = wrong.admit(dir.path()).unwrap();
        let legacy =
            Model2VecEmbedder::load_shared_preverified(dir.path(), wrong.id(), identity).unwrap();
        assert_eq!(legacy.embed_sync("hello").unwrap(), vec![1.0, 0.0]);
        assert!(Model2VecEmbedder::load_shared_registered(dir.path(), &wrong).is_err());
    }

    #[test]
    fn selected_shape_is_not_relabelled_to_fit_the_actual_tensor() {
        let (dir, selected) = fixture("shape-control", 2, 0);
        let mut download = selected.download_manifest().clone();
        download.dimension = Some(3);
        let expected = vec![vec![1.0, 0.0, 0.0]; MODEL_CONFORMANCE_TEXTS_V1.len()];
        let wrong_shape = RegisteredModel2Vec::new(
            download,
            PROVIDER,
            GoldenVectorCertificateV1::from_exact_f32(&MODEL_CONFORMANCE_TEXTS_V1, &expected)
                .unwrap(),
        )
        .unwrap();
        assert!(Model2VecEmbedder::load_registered(dir.path(), &wrong_shape).is_err());
    }

    #[test]
    fn registration_rejects_unpinned_artifacts_and_unsupported_layouts() {
        let (_dir, selected) = fixture("pin-control", 2, 0);
        for case in 0..5 {
            let mut download = selected.download_manifest().clone();
            match case {
                0 => download.revision = "main".to_owned(),
                1 => download.files[0].sha256 = "PLACEHOLDER_VERIFY_AFTER_DOWNLOAD".to_owned(),
                2 => download.files[0].size = 0,
                3 => download.files[0].name = "other.safetensors".to_owned(),
                _ => download.tier = Some(ModelTier::Quality),
            }
            assert!(RegisteredModel2Vec::new(download, PROVIDER, certificate(&selected)).is_err());
        }
    }

    #[test]
    fn registration_rejects_foreign_corpus_count_and_dimension() {
        let (_dir, selected) = fixture("corpus-control", 2, 0);
        for case in 0..3 {
            let mut golden = certificate(&selected);
            match case {
                0 => golden.corpus_sha256 = "0".repeat(64),
                1 => golden.vector_count = 3,
                _ => golden.dimension = 3,
            }
            assert!(RegisteredModel2Vec::new(
                selected.download_manifest().clone(),
                PROVIDER,
                golden,
            )
            .is_err());
        }
    }

    #[test]
    fn concurrent_first_loads_share_one_qualified_owner() {
        let (dir, selected) = fixture("concurrent-selected", 2, 0);
        let barrier = Barrier::new(4);
        let owners = std::thread::scope(|scope| {
            let handles = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        Model2VecEmbedder::load_shared_registered(dir.path(), &selected).unwrap()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(owners.iter().all(|owner| Arc::ptr_eq(&owners[0], owner)));
    }

    #[test]
    fn registration_debug_does_not_disclose_artifact_locations() {
        let (_dir, selected) = fixture("debug-control", 2, 0);
        let debug = format!("{selected:?}");
        assert!(debug.contains("dimension"));
        assert!(!debug.contains("fixtures/"));
        assert!(!debug.contains("https://"));
        assert!(!debug.contains(&selected.download_manifest.revision));
    }
}
