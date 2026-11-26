//! Explicit local-model activation of an already published authority.
//!
//! A separate startup grant permits changing the serving producer. This path
//! retains the original qualified root, admits the selected successor, and
//! replaces the model, reader and fixed scope together. Partial model changes
//! retain the exact source cohort, checkpoint and authenticated quality subset.
//! Explicit quality completion retains the fast model and admits a full or
//! strictly expanded partial successor of the same source cohort. An existing
//! partial quality tier must retain its producer identity and every covered ID
//! as coverage expands. Explicit rollback activation restores an already
//! published rollback's admitted topology, source and quality coverage. Matching
//! producers and query caches stay warm; missing or changed producers require
//! explicit local model paths. These operations never rebuild, publish, write a
//! receipt or download a model.

use frankensearch_core::generation::ActivationManifest;

use super::{
    Arc, AuthorityModels, AuthorityRefV1, AuthoritySession, Cx, Deserialize, Embedder,
    EmbedderTierTag, Error, GenerationAuthorityActionV1, Models, NativeGenerationTopology, Path,
    PathBuf, Receipt, Result, SearchError, admit_serving_model, admit_serving_models, checkpoint,
    models, prepare_scope, same_query_contract, serve_refused,
};

const MAX_PATH_BYTES: usize = 4096;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Operation {
    ActivateModel,
    ActivateQualityCompletion,
    ActivateRollback,
}

#[derive(Debug)]
pub(super) struct Activation {
    pub(super) models_loaded: usize,
    pub(super) primary_retained: bool,
    pub(super) quality_retained: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    #[serde(rename = "op")]
    operation: Operation,
    pub(super) id: Option<String>,
    receipt: PathBuf,
    expected_authority: Vec<u8>,
    model_dir: Option<PathBuf>,
    quality_backend: Option<String>,
    quality_model_dir: Option<PathBuf>,
}

impl Request {
    pub(super) const fn completes_quality(&self) -> bool {
        matches!(self.operation, Operation::ActivateQualityCompletion)
    }

    pub(super) const fn restores_rollback(&self) -> bool {
        matches!(self.operation, Operation::ActivateRollback)
    }

    pub(super) const fn operation(&self) -> &'static str {
        match self.operation {
            Operation::ActivateModel => "activate_model",
            Operation::ActivateQualityCompletion => "activate_quality_completion",
            Operation::ActivateRollback => "activate_rollback",
        }
    }

    fn model_options(
        &self,
        tier: EmbedderTierTag,
        has_quality_component: bool,
    ) -> Result<models::QualityOptions> {
        let options = self
            .validated_model_options(tier == EmbedderTierTag::Quality || has_quality_component)?;
        // Do not let a request inherit a process default, environment-selected
        // cache, or the old model directory. Native quality selects its exact
        // directory through QualityOptions; fast/ONNX require an explicit root.
        if (tier == EmbedderTierTag::Fast
            || options.backend.unwrap_or_default() == models::QualityBackend::Onnx)
            && self.model_dir.is_none()
        {
            return Err(serve_refused("model_activation_model_path_required"));
        }
        Ok(options)
    }

    fn rollback_model_options(
        &self,
        selected: &Receipt,
        retain_primary: bool,
        retain_quality: bool,
    ) -> Result<models::QualityOptions> {
        let load_primary = !retain_primary;
        let load_quality = (load_primary && selected.tier == EmbedderTierTag::Quality)
            || (selected.quality_identity.is_some() && !retain_quality);
        let options = self.validated_model_options(load_quality)?;
        if !load_primary && !load_quality && self.model_dir.is_some() {
            return Err(serve_refused("rollback_activation_unused_model_options"));
        }
        if ((load_primary && selected.tier == EmbedderTierTag::Fast)
            || (load_quality
                && options.backend.unwrap_or_default() == models::QualityBackend::Onnx))
            && self.model_dir.is_none()
        {
            return Err(serve_refused("model_activation_model_path_required"));
        }
        Ok(options)
    }

    fn validated_model_options(&self, load_quality: bool) -> Result<models::QualityOptions> {
        validate_path(&self.receipt)?;
        if let Some(path) = &self.model_dir {
            validate_path(path)?;
        }
        if let Some(path) = &self.quality_model_dir {
            validate_path(path)?;
        }
        let options = models::QualityOptions {
            backend: self
                .quality_backend
                .as_deref()
                .map(models::QualityBackend::parse)
                .transpose()
                .map_err(|_| serve_refused("model_activation_invalid_model_options"))?,
            directory: self.quality_model_dir.clone(),
        };
        options
            .validate_usage(load_quality)
            .map_err(|_| serve_refused("model_activation_invalid_model_options"))?;
        Ok(options)
    }
}

pub(super) fn validate_path(path: &Path) -> Result<()> {
    if !path.is_absolute()
        || path
            .to_str()
            .is_none_or(|value| value.len() > MAX_PATH_BYTES || value.contains('\0'))
    {
        return Err(serve_refused("model_activation_invalid_path"));
    }
    Ok(())
}

pub(super) fn bounded_error(
    error: Box<dyn Error + Send + Sync>,
    code: &'static str,
) -> Box<dyn Error + Send + Sync> {
    if error
        .downcast_ref::<SearchError>()
        .is_some_and(|error| matches!(error, SearchError::Cancelled { .. }))
    {
        error
    } else {
        serve_refused(code)
    }
}

impl AuthoritySession {
    pub(super) fn activate_model(&mut self, cx: &Cx, request: &Request) -> Result<Activation> {
        self.activate_model_with_loader(cx, request, |cx, directory, tier, options| {
            models::load_single(cx, directory, tier, options, cx.blocking_pool_handle())
        })
    }

    pub(super) fn activate_model_with_loader<F>(
        &mut self,
        cx: &Cx,
        request: &Request,
        mut load_model: F,
    ) -> Result<Activation>
    where
        F: FnMut(
            &Cx,
            Option<&Path>,
            EmbedderTierTag,
            &models::QualityOptions,
        ) -> Result<Arc<dyn Embedder>>,
    {
        checkpoint(cx)?;
        self.require_current_selection()?;
        if !self.allow_model_activation {
            return Err(serve_refused("model_activation_disabled"));
        }
        let completes_quality = request.completes_quality();
        let restores_rollback = request.restores_rollback();
        let retains_partial_quality = !completes_quality
            && !restores_rollback
            && matches!(
                self.receipt.topology(),
                NativeGenerationTopology::PartialQuality { .. }
            );
        if !matches!(self.models, AuthorityModels::Semantic(_)) {
            return Err(serve_refused("model_activation_requires_hybrid"));
        }
        super::super::serve::validate_id(request.id.as_deref())
            .map_err(|_| serve_refused("invalid_request_id"))?;
        let expected = AuthorityRefV1::from_canonical_bytes(&request.expected_authority)
            .map_err(|_| serve_refused("invalid_expected_authority"))?;
        if expected != self.snapshot.authority() {
            return Err(serve_refused("stale_expected_authority"));
        }
        let options = if restores_rollback {
            // The restored receipt determines which producers are needed.
            // Never inherit model requirements from the topology being undone.
            validate_path(&request.receipt)?;
            None
        } else if completes_quality {
            if !matches!(
                self.receipt.topology(),
                NativeGenerationTopology::FastOnly
                    | NativeGenerationTopology::PartialQuality { .. }
            ) {
                return Err(serve_refused(
                    "quality_completion_requires_incomplete_quality",
                ));
            }
            // The fast provider is retained, not reloaded from request paths.
            // Admit both existing producers for a partial tier before reading
            // a candidate or loading its quality model.
            let admission = self.models.admit(&self.receipt);
            checkpoint(cx)?;
            admission?;
            Some(request.model_options(EmbedderTierTag::Quality, true)?)
        } else {
            Some(
                request
                    .model_options(self.receipt.tier, self.receipt.quality_identity.is_some())?,
            )
        };
        let selected = Receipt::read(&request.receipt);
        checkpoint(cx)?;
        let selected =
            selected.map_err(|error| bounded_error(error, "model_activation_invalid_receipt"))?;
        if selected.root != self.receipt.root
            || selected.root_id != self.receipt.root_id
            || selected.profile != self.receipt.profile
            || selected.tier != self.receipt.tier
        {
            return Err(serve_refused("model_activation_root_or_tier_mismatch"));
        }
        if completes_quality {
            match selected.topology() {
                NativeGenerationTopology::FullProgressive => {}
                NativeGenerationTopology::PartialQuality {
                    quality_documents, ..
                } => {
                    if quality_documents <= self.receipt.quality_documents.unwrap_or(0) {
                        return Err(serve_refused("quality_completion_coverage_not_expanded"));
                    }
                }
                _ => return Err(serve_refused("quality_completion_requires_quality")),
            }
            if selected.identity != self.receipt.identity {
                return Err(serve_refused("quality_completion_fast_identity_mismatch"));
            }
            if self.receipt.quality_identity.is_some()
                && selected.quality_identity != self.receipt.quality_identity
            {
                return Err(serve_refused(
                    "quality_completion_quality_identity_mismatch",
                ));
            }
            if selected.generation.sequence <= self.receipt.generation.sequence {
                return Err(serve_refused("quality_completion_not_newer_generation"));
            }
            if selected.documents != self.receipt.documents {
                return Err(serve_refused("quality_completion_source_mismatch"));
            }
        } else if !restores_rollback && selected.topology() != self.receipt.topology() {
            return Err(serve_refused("model_activation_topology_mismatch"));
        }
        if selected.authority()?.sequence <= self.snapshot.authority().sequence {
            return Err(serve_refused("model_activation_not_newer"));
        }
        // This grant changes producers and vector spaces, not prepared-input
        // semantics. A different canonicalization/document-encoding contract
        // requires a separately initialized serving selection.
        if selected.identity.identity.input != self.receipt.identity.identity.input
            || selected
                .quality_identity
                .as_ref()
                .zip(self.receipt.quality_identity.as_ref())
                .is_some_and(|(next, previous)| next.identity.input != previous.identity.input)
            || ((completes_quality || restores_rollback)
                && self.receipt.topology() == NativeGenerationTopology::FastOnly
                && selected.quality_identity.as_ref().is_some_and(|quality| {
                    quality.identity.input != self.receipt.identity.identity.input
                        || quality.identity.space.kind != self.receipt.identity.identity.space.kind
                }))
        {
            return Err(serve_refused("model_activation_input_contract_mismatch"));
        }

        // All artifact, topology and fixed-policy admission precedes model
        // loading. The original physical root capability is retained even if
        // its pathname now names a copied root or no longer exists.
        let snapshot = selected.open_on_root(cx, &self.root, self.max_image_bytes);
        checkpoint(cx)?;
        let snapshot = Arc::new(
            snapshot.map_err(|error| bounded_error(error, "model_activation_admission_failed"))?,
        );
        if restores_rollback {
            let action = match snapshot.retained_snapshot().manifest() {
                ActivationManifest::V1(manifest) => manifest.action,
                ActivationManifest::V2(manifest) => manifest.action,
            };
            if action != GenerationAuthorityActionV1::Rollback {
                return Err(serve_refused("rollback_activation_requires_rollback"));
            }
        }
        let retain_primary = completes_quality
            || (restores_rollback
                && same_query_contract(&selected.identity, &self.receipt.identity));
        let retain_quality = restores_rollback
            && selected
                .quality_identity
                .as_ref()
                .zip(self.receipt.quality_identity.as_ref())
                .is_some_and(|(next, previous)| same_query_contract(next, previous));
        let options = match options {
            Some(options) => options,
            None => request.rollback_model_options(&selected, retain_primary, retain_quality)?,
        };
        if retain_primary {
            let admission =
                admit_serving_model(&selected.identity, self.models.models()?.fast.as_ref());
            checkpoint(cx)?;
            admission?;
        }
        if retain_quality {
            let quality = self
                .models
                .models()?
                .quality
                .as_ref()
                .ok_or_else(|| serve_refused("session_provider_topology_mismatch"))?;
            let admission = admit_serving_model(
                selected
                    .quality_identity
                    .as_ref()
                    .ok_or_else(|| serve_refused("session_provider_topology_mismatch"))?,
                quality.as_ref(),
            );
            checkpoint(cx)?;
            admission?;
        }
        checkpoint(cx)?;
        if retains_partial_quality {
            // A model change may replace either producer, but it cannot select
            // a different quality cohort. Equal counts do not prove membership:
            // compare IDs authenticated by both retained quality vector owners.
            let previous_ids = self
                .snapshot
                .partial_quality_document_ids(cx)
                .map_err(|error| {
                    bounded_error(error.into(), "model_activation_partial_coverage_mismatch")
                })?;
            let next_ids = snapshot.partial_quality_document_ids(cx).map_err(|error| {
                bounded_error(error.into(), "model_activation_partial_coverage_mismatch")
            })?;
            if next_ids != previous_ids {
                return Err(serve_refused("model_activation_partial_coverage_mismatch"));
            }
            // Partial model activation adopts a retained-source rebuild. Source
            // edits use the ordinary partial update path; quality expansion uses
            // explicit completion. Bind every source field and its exact cut
            // before preparing a scope or constructing replacement providers.
            let previous = self.snapshot.retained_snapshot().manifest();
            let next = snapshot.retained_snapshot().manifest();
            if next.source_checkpoint_sha256() != previous.source_checkpoint_sha256()
                || next.document_set_sha256() != previous.document_set_sha256()
                || next.components().lexical != previous.components().lexical
                || snapshot.metadata().commit_range != self.snapshot.metadata().commit_range
            {
                return Err(serve_refused("model_activation_partial_source_mismatch"));
            }
        }
        if completes_quality {
            // A matching count or ID set cannot prove that bodies, titles or
            // metadata stayed unchanged. The exact retained Quill image covers
            // all source fields, and both admitted manifests bind the same cut.
            let previous = self.snapshot.retained_snapshot().manifest();
            let next = snapshot.retained_snapshot().manifest();
            if next.source_checkpoint_sha256() != previous.source_checkpoint_sha256()
                || next.document_set_sha256() != previous.document_set_sha256()
                || next.components().lexical != previous.components().lexical
                || snapshot.metadata().commit_range != self.snapshot.metadata().commit_range
            {
                return Err(serve_refused("quality_completion_source_mismatch"));
            }
            if snapshot.retrieval_configuration() != self.snapshot.retrieval_configuration() {
                return Err(serve_refused("quality_completion_fast_retrieval_mismatch"));
            }
            if matches!(
                snapshot.topology(),
                NativeGenerationTopology::PartialQuality { .. }
            ) {
                // Counts are only an early refusal. Admitted owners must prove
                // that a larger partial cohort kept every previous quality ID.
                // A full successor already proves complete coverage of this
                // exact unchanged source through ordinary full admission above.
                let next_ids = snapshot.partial_quality_document_ids(cx).map_err(|error| {
                    bounded_error(error.into(), "quality_completion_coverage_mismatch")
                })?;
                let previous_ids = if self.receipt.quality_documents.is_some() {
                    self.snapshot
                        .partial_quality_document_ids(cx)
                        .map_err(|error| {
                            bounded_error(error.into(), "quality_completion_coverage_mismatch")
                        })?
                } else {
                    Vec::new()
                };
                if next_ids.len() <= previous_ids.len()
                    || previous_ids
                        .iter()
                        .any(|id| next_ids.binary_search(id).is_err())
                {
                    return Err(serve_refused("quality_completion_coverage_mismatch"));
                }
            }
        }
        let scope = prepare_scope(
            cx,
            &snapshot,
            self.fixed_filter.as_ref(),
            self.max_source_bytes,
        );
        checkpoint(cx)?;
        let scope = scope.map_err(|error| bounded_error(error, "model_activation_scope_failed"))?;
        let fast_options = models::QualityOptions::default();
        let primary_options = if selected.tier == EmbedderTierTag::Fast {
            &fast_options
        } else {
            &options
        };
        let model = if retain_primary {
            Arc::clone(&self.models.models()?.fast)
        } else {
            let loaded = load_model(
                cx,
                request.model_dir.as_deref(),
                selected.tier,
                primary_options,
            );
            checkpoint(cx)?;
            loaded.map_err(|error| bounded_error(error, "model_activation_load_failed"))?
        };
        let quality = if retain_quality {
            Some(Arc::clone(
                self.models
                    .models()?
                    .quality
                    .as_ref()
                    .ok_or_else(|| serve_refused("session_provider_topology_mismatch"))?,
            ))
        } else if selected.quality_identity.is_some() {
            let loaded = load_model(
                cx,
                request.model_dir.as_deref(),
                EmbedderTierTag::Quality,
                &options,
            );
            checkpoint(cx)?;
            Some(loaded.map_err(|error| bounded_error(error, "model_activation_load_failed"))?)
        } else {
            None
        };
        let loaded = Models {
            fast: model,
            quality,
        };
        let admission = admit_serving_models(&selected, &loaded);
        checkpoint(cx)?;
        admission?;
        let cached = if retain_quality && self.query_cache.enabled() {
            let previous = self.query_models().models()?;
            let fast = if retain_primary {
                Arc::clone(&previous.fast)
            } else {
                self.query_cache
                    .wrap_embedder(cx, Arc::clone(&loaded.fast))?
            };
            let quality =
                Some(Arc::clone(previous.quality.as_ref().ok_or_else(|| {
                    serve_refused("session_provider_topology_mismatch")
                })?));
            Some(AuthorityModels::Semantic(Models { fast, quality }))
        } else {
            self.cache_replacement_models(cx, &loaded, retain_primary)?
        };

        // Loading can be slow. A successor published meanwhile must not turn
        // this stale request into an implicit selection of that newer head.
        // Recheck the exact selected authority and pair after loading, while
        // keeping the originally admitted immutable reader and scope together.
        let current = selected.open_on_root(cx, &self.root, self.max_image_bytes);
        checkpoint(cx)?;
        let current =
            current.map_err(|error| bounded_error(error, "model_activation_head_changed"))?;
        if current.retained_snapshot().authority_pair()
            != snapshot.retained_snapshot().authority_pair()
        {
            return Err(serve_refused("model_activation_head_changed"));
        }
        drop(current);
        let admission = admit_serving_models(&selected, &loaded);
        checkpoint(cx)?;
        admission?;

        let activation = Activation {
            models_loaded: usize::from(!retain_primary)
                + usize::from(selected.quality_identity.is_some() && !retain_quality),
            primary_retained: retain_primary,
            quality_retained: retain_quality,
        };

        // The request loop exclusively owns all session fields. No fallible
        // operation or query can run between these assignments.
        self.snapshot = snapshot;
        self.scope = scope;
        self.receipt = selected;
        self.models = AuthorityModels::Semantic(loaded);
        self.cached_query_models = cached;
        Ok(activation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> serde_json::Value {
        serde_json::json!({
            "op": "activate_model",
            "id": "selected-model",
            "receipt": std::env::temp_dir().join("authority-model.json"),
            "expected_authority": [],
            "model_dir": std::env::temp_dir().join("authority-models"),
        })
    }

    #[test]
    fn model_activation_requests_have_explicit_operations_and_no_writer_fields() {
        for operation in [
            "activate_model",
            "activate_quality_completion",
            "activate_rollback",
        ] {
            let mut value = request();
            value["op"] = serde_json::json!(operation);
            let parsed: Request = serde_json::from_value(value.clone()).unwrap();
            assert_eq!(parsed.id.as_deref(), Some("selected-model"));
            assert_eq!(parsed.operation(), operation);
            assert_eq!(
                parsed.completes_quality(),
                operation == "activate_quality_completion"
            );
            assert_eq!(parsed.restores_rollback(), operation == "activate_rollback");
            assert!(matches!(
                serde_json::from_value::<super::super::AuthorityRequest>(value.clone()).unwrap(),
                super::super::AuthorityRequest::ModelActivation(_)
            ));
            for field in ["work_dir", "changes", "filter", "tier", "exact", "mode"] {
                let mut extra = value.clone();
                extra[field] = serde_json::json!("ungranted");
                assert!(serde_json::from_value::<Request>(extra).is_err(), "{field}");
            }
        }
        for operation in ["activate", "update", "fetch", "migrate", "ACTIVATE_MODEL"] {
            let mut value = request();
            value["op"] = serde_json::json!(operation);
            assert!(serde_json::from_value::<Request>(value).is_err());
        }
        for field in ["op", "receipt", "expected_authority"] {
            let mut value = request();
            value.as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<Request>(value).is_err(), "{field}");
        }
    }

    #[test]
    fn model_activation_requires_explicit_bounded_local_paths_for_each_tier() {
        let parsed: Request = serde_json::from_value(request()).unwrap();
        assert!(parsed.model_options(EmbedderTierTag::Fast, false).is_ok());
        assert!(
            parsed
                .model_options(EmbedderTierTag::Quality, false)
                .is_ok()
        );
        for field in ["receipt", "model_dir", "quality_model_dir"] {
            for path in [
                "relative/private-model".to_owned(),
                format!("/{}", "x".repeat(MAX_PATH_BYTES)),
                "/private/model\0secret".to_owned(),
            ] {
                let mut value = request();
                value[field] = serde_json::json!(path);
                let parsed: Request = serde_json::from_value(value).unwrap();
                let error = parsed
                    .model_options(EmbedderTierTag::Quality, false)
                    .unwrap_err();
                assert!(error.to_string().contains("model_activation_invalid_path"));
                assert!(!error.to_string().contains("secret"));
            }
        }
        let mut value = request();
        value.as_object_mut().unwrap().remove("model_dir");
        let parsed: Request = serde_json::from_value(value.clone()).unwrap();
        for tier in [EmbedderTierTag::Fast, EmbedderTierTag::Quality] {
            assert!(
                parsed
                    .model_options(tier, false)
                    .unwrap_err()
                    .to_string()
                    .contains("model_activation_model_path_required")
            );
        }
        value["quality_backend"] = serde_json::json!("native-int8");
        value["quality_model_dir"] =
            serde_json::json!(std::env::temp_dir().join("verified-native-model"));
        let parsed: Request = serde_json::from_value(value).unwrap();
        assert!(parsed.model_options(EmbedderTierTag::Fast, false).is_err());
        let options = parsed
            .model_options(EmbedderTierTag::Quality, false)
            .unwrap();
        assert_eq!(options.backend, Some(models::QualityBackend::NativeInt8));
        assert_eq!(
            options.directory,
            Some(std::env::temp_dir().join("verified-native-model"))
        );
        assert!(
            parsed
                .model_options(EmbedderTierTag::Fast, true)
                .unwrap_err()
                .to_string()
                .contains("model_activation_model_path_required")
        );
        let mut full = request();
        full["quality_backend"] = serde_json::json!("native-f32");
        full["quality_model_dir"] =
            serde_json::json!(std::env::temp_dir().join("verified-native-model"));
        let full: Request = serde_json::from_value(full).unwrap();
        assert!(full.model_options(EmbedderTierTag::Fast, true).is_ok());
    }

    #[test]
    fn model_activation_rejects_invalid_or_incomplete_backend_selection() {
        for backend in ["automatic", "hash", "fastembed", ""] {
            let mut value = request();
            value["quality_backend"] = serde_json::json!(backend);
            let parsed: Request = serde_json::from_value(value).unwrap();
            assert!(
                parsed
                    .model_options(EmbedderTierTag::Quality, false)
                    .is_err()
            );
        }
        let mut value = request();
        value["quality_backend"] = serde_json::json!("native-f32");
        let parsed: Request = serde_json::from_value(value.clone()).unwrap();
        assert!(
            parsed
                .model_options(EmbedderTierTag::Quality, false)
                .is_err()
        );
        value["quality_backend"] = serde_json::json!("onnx");
        value["quality_model_dir"] = serde_json::json!(std::env::temp_dir());
        let parsed: Request = serde_json::from_value(value).unwrap();
        assert!(
            parsed
                .model_options(EmbedderTierTag::Quality, false)
                .is_err()
        );
    }
}
