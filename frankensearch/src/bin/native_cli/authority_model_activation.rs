//! Explicit local-model activation of an already published authority.
//!
//! A separate startup grant permits changing the serving producer. This path
//! retains the original qualified root, admits the complete selected successor,
//! and replaces the model, reader and fixed scope together. It never rebuilds,
//! publishes, writes a receipt, downloads a model or changes the selected tier.

use super::{
    Arc, AuthorityModels, AuthorityRefV1, AuthoritySession, Cx, Deserialize, Embedder,
    EmbedderTierTag, Error, Models, Path, PathBuf, Receipt, Result, SearchError,
    admit_serving_models, checkpoint, models, prepare_scope, serve_refused,
};

const MAX_PATH_BYTES: usize = 4096;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Operation {
    ActivateModel,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    #[serde(rename = "op")]
    _operation: Operation,
    pub(super) id: Option<String>,
    receipt: PathBuf,
    expected_authority: Vec<u8>,
    model_dir: Option<PathBuf>,
    quality_backend: Option<String>,
    quality_model_dir: Option<PathBuf>,
}

impl Request {
    fn model_options(
        &self,
        tier: EmbedderTierTag,
        has_quality_component: bool,
    ) -> Result<models::QualityOptions> {
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
            .validate_usage(tier == EmbedderTierTag::Quality || has_quality_component)
            .map_err(|_| serve_refused("model_activation_invalid_model_options"))?;
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
}

fn validate_path(path: &Path) -> Result<()> {
    if !path.is_absolute()
        || path
            .to_str()
            .is_none_or(|value| value.len() > MAX_PATH_BYTES || value.contains('\0'))
    {
        return Err(serve_refused("model_activation_invalid_path"));
    }
    Ok(())
}

fn bounded_error(
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
    pub(super) fn activate_model(&mut self, cx: &Cx, request: &Request) -> Result<()> {
        self.activate_model_with_loader(cx, request, |cx, directory, tier, options| {
            models::load_single(cx, directory, tier, options, cx.blocking_pool_handle())
        })
    }

    pub(super) fn activate_model_with_loader<F>(
        &mut self,
        cx: &Cx,
        request: &Request,
        mut load_model: F,
    ) -> Result<()>
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
        let options =
            request.model_options(self.receipt.tier, self.receipt.quality_identity.is_some())?;
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
        if selected.topology() != self.receipt.topology() {
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
        let model = load_model(
            cx,
            request.model_dir.as_deref(),
            selected.tier,
            primary_options,
        );
        checkpoint(cx)?;
        let model = model.map_err(|error| bounded_error(error, "model_activation_load_failed"))?;
        let quality = if selected.quality_identity.is_some() {
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
        admit_serving_models(&selected, &loaded)?;
        checkpoint(cx)?;

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
        admit_serving_models(&selected, &loaded)?;
        checkpoint(cx)?;

        // The request loop exclusively owns all session fields. No fallible
        // operation or query can run between these assignments.
        self.snapshot = snapshot;
        self.scope = scope;
        self.receipt = selected;
        self.models = AuthorityModels::Semantic(loaded);
        Ok(())
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
    fn model_activation_request_has_one_strict_operation_and_no_writer_fields() {
        let parsed: Request = serde_json::from_value(request()).unwrap();
        assert_eq!(parsed.id.as_deref(), Some("selected-model"));
        for field in ["work_dir", "changes", "filter", "tier", "exact", "mode"] {
            let mut value = request();
            value[field] = serde_json::json!("ungranted");
            assert!(serde_json::from_value::<Request>(value).is_err(), "{field}");
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
