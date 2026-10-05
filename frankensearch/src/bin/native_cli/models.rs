//! Explicit local quality producers for the native executable. Choosing a
//! backend never grants permission to query another producer's stored vectors.
//! The selection and native descriptor still perform full identity admission.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use asupersync::runtime::blocking_pool::BlockingPoolHandle;
use frankensearch::{Cx, Embedder, SearchError};
use frankensearch_embed::{DetectOptions, EmbedderStack};

use super::{Models, Result, bad};

/// Stable across feature combinations: adding a compiled backend must not
/// silently change the model selected by the same command line.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum QualityBackend {
    #[default]
    Onnx,
    NativeInt8,
    NativeF32,
    NativeMultilingual,
}

impl QualityBackend {
    pub(super) fn parse(value: &str) -> Result<Self> {
        match value {
            "onnx" => Ok(Self::Onnx),
            "native-int8" => Ok(Self::NativeInt8),
            "native-f32" => Ok(Self::NativeF32),
            "native-multilingual" => Ok(Self::NativeMultilingual),
            _ => Err(bad(
                "quality backend must be onnx, native-int8, native-f32, or native-multilingual",
            )),
        }
    }

    #[cfg(any(feature = "native", feature = "rerank"))]
    fn native_profile(self) -> Option<frankensearch::NativeEmbeddingModel> {
        use frankensearch::NativeEmbeddingModel;
        match self {
            Self::Onnx => None,
            Self::NativeInt8 => Some(NativeEmbeddingModel::AllMiniLmL6V2),
            Self::NativeF32 => Some(NativeEmbeddingModel::AllMiniLmL6V2F32),
            Self::NativeMultilingual => {
                Some(NativeEmbeddingModel::ParaphraseMultilingualMiniLmL12V2)
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct QualityOptions {
    // None and explicit Onnx choose the same backend, but explicit options on
    // a fast-only selection are errors rather than silently ignored requests.
    pub(super) backend: Option<QualityBackend>,
    pub(super) directory: Option<PathBuf>,
}

impl QualityOptions {
    pub(super) fn validate_usage(&self, required: bool) -> Result<()> {
        if !required {
            if self.backend.is_some() || self.directory.is_some() {
                return Err(bad(
                    "quality options cannot be used with a fast-only build or selection",
                ));
            }
            return Ok(());
        }
        match self.backend.unwrap_or_default() {
            QualityBackend::Onnx => {
                if self.directory.is_some() {
                    return Err(bad(
                        "--quality-model-dir requires an explicit native --quality-backend; ONNX uses --model-dir",
                    ));
                }
            }
            _ => {
                let path = self.directory.as_deref().ok_or_else(|| {
                    bad("native quality requires --quality-model-dir naming its exact verified model directory")
                })?;
                if path
                    .to_str()
                    .is_none_or(|text| text.trim().is_empty() || text.contains('\0'))
                {
                    return Err(bad(
                        "quality model directory must be nonblank, UTF-8, and NUL-free",
                    ));
                }
            }
        }
        Ok(())
    }

    // Preflight runs before either model loader. An unavailable backend cannot
    // turn into an ONNX/hash fallback or hide behind a fast-model load error.
    fn preflight(&self, required: bool, has_pool: bool) -> Result<Option<QualityBackend>> {
        self.validate_usage(required)?;
        if !required {
            return Ok(None);
        }
        let backend = self.backend.unwrap_or_default();
        match backend {
            QualityBackend::Onnx if !cfg!(feature = "fastembed") => {
                return Err(bad(
                    "ONNX quality is not compiled in; explicitly select a native quality backend or build with fastembed",
                ));
            }
            QualityBackend::Onnx => {}
            _ => {
                if !cfg!(any(feature = "native", feature = "rerank")) {
                    return Err(bad(
                        "native quality is not compiled in; build with --features hybrid-native",
                    ));
                }
                if !has_pool {
                    return Err(bad("native quality requires the caller's blocking pool"));
                }
            }
        }
        Ok(Some(backend))
    }
}

fn checkpoint(cx: &Cx) -> Result<()> {
    cx.checkpoint().map_err(|error| {
        Box::new(SearchError::Cancelled {
            phase: "native_cli.models".to_owned(),
            reason: error.to_string(),
        }) as Box<dyn std::error::Error + Send + Sync>
    })
}

/// Local-only startup. The native constructor verifies registered artifacts and
/// the executing producer certificate; it is not an unchecked weights loader.
/// Its async inference retains the command-owned pool through updates and live
/// activation. Startup verification/loading itself remains synchronous.
pub fn load(
    cx: &Cx,
    root: Option<&Path>,
    required_quality: bool,
    options: &QualityOptions,
    pool: Option<BlockingPoolHandle>,
) -> Result<Models> {
    checkpoint(cx)?;
    let backend = options.preflight(required_quality, pool.is_some())?;
    let policy = DetectOptions {
        offline: Some(true),
    };
    let fast_result = EmbedderStack::auto_detect_fast_semantic_with_options(root, &policy);
    checkpoint(cx)?;
    let fast = fast_result?.fast_arc();
    let quality_result = backend
        .map(|backend| load_quality(backend, root, options, policy, pool))
        .transpose();
    checkpoint(cx)?;
    Ok(Models {
        fast,
        quality: quality_result?,
    })
}

fn load_quality(
    backend: QualityBackend,
    root: Option<&Path>,
    options: &QualityOptions,
    policy: DetectOptions,
    pool: Option<BlockingPoolHandle>,
) -> Result<Arc<dyn Embedder>> {
    if backend == QualityBackend::Onnx {
        return EmbedderStack::auto_detect_quality_with_options(root, &policy)?
            .ok_or_else(|| bad("the required local ONNX quality model is unavailable; no native or hash substitution is permitted"));
    }
    #[cfg(any(feature = "native", feature = "rerank"))]
    {
        let profile = backend
            .native_profile()
            .ok_or_else(|| bad("native quality profile is missing"))?;
        let directory = options
            .directory
            .as_deref()
            .ok_or_else(|| bad("native quality directory is missing"))?;
        let pool = pool.ok_or_else(|| bad("native quality requires the caller's blocking pool"))?;
        let model =
            frankensearch::NativeEmbedder::load_model(directory, profile)?.with_blocking_pool(pool);
        Ok(Arc::new(model))
    }
    #[cfg(not(any(feature = "native", feature = "rerank")))]
    {
        let _ = (options, pool);
        Err(bad(
            "native quality is not compiled in; build with --features hybrid-native",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Command, Options};
    use super::*;

    fn parse(args: &[&str]) -> Result<Option<Options>> {
        Options::parse(args.iter().map(|value| (*value).to_owned()))
    }

    #[test]
    fn backend_selection_is_explicit_and_feature_independent() {
        assert_eq!(QualityBackend::default(), QualityBackend::Onnx);
        for (name, expected) in [
            ("onnx", QualityBackend::Onnx),
            ("native-int8", QualityBackend::NativeInt8),
            ("native-f32", QualityBackend::NativeF32),
            ("native-multilingual", QualityBackend::NativeMultilingual),
        ] {
            assert_eq!(QualityBackend::parse(name).unwrap(), expected);
        }
        for name in ["", "auto", "native", "hash", "NATIVE-F32", "native-f32 "] {
            assert!(QualityBackend::parse(name).is_err(), "{name}");
        }
    }

    #[test]
    fn every_native_command_accepts_the_same_explicit_quality_policy() {
        for (command, expected) in [
            ("index", Command::Index),
            ("rebuild", Command::Rebuild),
            ("update", Command::Update),
            ("search", Command::Search),
            ("serve", Command::Serve),
        ] {
            let mut args = vec![
                command,
                "--receipt",
                "receipt.json",
                "--quality-backend",
                "native-multilingual",
                "--quality-model-dir",
                "multilingual",
            ];
            if matches!(
                expected,
                Command::Index | Command::Update | Command::Rebuild
            ) {
                args.extend(["--index-dir", "new-index"]);
            }
            if matches!(expected, Command::Update | Command::Rebuild) {
                args.extend(["--new-receipt", "new-receipt.json"]);
            }
            if expected == Command::Search {
                args.extend(["--query", "网络重试"]);
            }
            let options = parse(&args).unwrap().unwrap();
            assert_eq!(options.command, expected);
            assert_eq!(
                options.quality.backend,
                Some(QualityBackend::NativeMultilingual)
            );
            assert_eq!(
                options.quality.directory.as_deref(),
                Some(Path::new("multilingual"))
            );
        }
    }

    #[test]
    fn ambiguous_missing_and_inapplicable_quality_options_are_refused() {
        for tail in [
            vec!["--quality-backend", "native-int8"],
            vec!["--quality-model-dir", "native-model"],
            vec![
                "--quality-backend",
                "onnx",
                "--quality-model-dir",
                "native-model",
            ],
            vec![
                "--quality-backend",
                "native-f32",
                "--quality-model-dir",
                "bad\0path",
            ],
            vec!["--quality-backend", "onnx", "--quality-backend", "onnx"],
        ] {
            let mut args = vec!["search", "--receipt", "saved.json", "--query", "retry"];
            args.extend(tail);
            assert!(parse(&args).is_err());
        }
        for command in ["index", "rebuild"] {
            let mut args = vec![
                command,
                "--receipt",
                "old.json",
                "--index-dir",
                "new",
                "--fast-only",
                "--quality-backend",
                "native-f32",
                "--quality-model-dir",
                "native",
            ];
            if command == "rebuild" {
                args.extend(["--new-receipt", "next.json"]);
            }
            assert!(parse(&args).is_err());
        }
    }

    #[test]
    fn fast_only_selection_does_not_load_or_implicitly_drop_requested_quality() {
        let default = QualityOptions::default();
        assert_eq!(default.preflight(false, false).unwrap(), None);
        for backend in [QualityBackend::Onnx, QualityBackend::NativeInt8] {
            let selected = QualityOptions {
                backend: Some(backend),
                directory: None,
            };
            assert!(
                selected
                    .preflight(false, true)
                    .unwrap_err()
                    .to_string()
                    .contains("fast-only")
            );
        }
    }

    #[test]
    fn unavailable_quality_capabilities_fail_before_fast_model_discovery() {
        let root = tempfile::tempdir().unwrap();
        let options = QualityOptions {
            backend: Some(QualityBackend::NativeInt8),
            directory: Some(root.path().join("missing-native")),
        };
        let error = load(&Cx::for_testing(), Some(root.path()), true, &options, None)
            .err()
            .expect("no native pool or feature");
        let expected = if cfg!(any(feature = "native", feature = "rerank")) {
            "blocking pool"
        } else {
            "not compiled in"
        };
        assert!(error.to_string().contains(expected), "{error}");
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
        assert_eq!(
            QualityOptions::default().preflight(true, false).is_ok(),
            cfg!(feature = "fastembed")
        );
    }

    #[test]
    fn cancelled_model_loading_preserves_cancellation_before_any_discovery() {
        let cx = Cx::for_testing();
        let root = tempfile::tempdir().unwrap();
        cx.set_cancel_requested(true);
        let result = load(
            &cx,
            Some(root.path()),
            true,
            &QualityOptions::default(),
            None,
        );
        cx.set_cancel_requested(false);
        let error = result.err().expect("cancelled load");
        assert!(matches!(
            error.downcast_ref::<SearchError>(),
            Some(SearchError::Cancelled { .. })
        ));
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    }

    #[test]
    fn native_profile_does_not_require_the_onnx_bundle_to_build_the_binary() {
        // Manifest contract only, not a Cargo resolution or compilation claim.
        let manifest: toml::Value = toml::from_str(include_str!("../../../Cargo.toml")).unwrap();
        assert_eq!(
            manifest["features"]["hybrid-native"].as_array().unwrap(),
            &[
                toml::Value::String("model2vec".to_owned()),
                toml::Value::String("quill".to_owned()),
                toml::Value::String("native".to_owned()),
            ]
        );
        let binary = manifest["bin"]
            .as_array()
            .unwrap()
            .iter()
            .find(|binary| binary["name"].as_str() == Some("frankensearch-native"))
            .unwrap();
        assert_eq!(
            binary["required-features"].as_array().unwrap(),
            &[toml::Value::String("quill".to_owned()),]
        );
    }

    #[cfg(any(feature = "native", feature = "rerank"))]
    #[test]
    fn native_backend_variants_select_their_exact_registered_constructors() {
        use frankensearch::NativeEmbeddingModel;
        assert_eq!(QualityBackend::Onnx.native_profile(), None);
        assert_eq!(
            QualityBackend::NativeInt8.native_profile(),
            Some(NativeEmbeddingModel::AllMiniLmL6V2)
        );
        assert_eq!(
            QualityBackend::NativeF32.native_profile(),
            Some(NativeEmbeddingModel::AllMiniLmL6V2F32)
        );
        assert_eq!(
            QualityBackend::NativeMultilingual.native_profile(),
            Some(NativeEmbeddingModel::ParaphraseMultilingualMiniLmL12V2)
        );
    }
}

#[cfg(all(
    test,
    feature = "model2vec",
    any(feature = "native", feature = "rerank")
))]
mod real_model_tests {
    use super::super::{Options, Selection, execute, filter, query, serve};
    use super::*;

    fn required_directory(variable: &str) -> PathBuf {
        let path =
            PathBuf::from(std::env::var_os(variable).unwrap_or_else(|| {
                panic!("set {variable} to the installed verified model directory")
            }));
        assert!(path.is_dir(), "{variable} must name an existing directory");
        path
    }

    fn options(
        command: &str,
        backend: &str,
        root: &Path,
        quality: &Path,
        extra: &[(&str, &Path)],
    ) -> Options {
        let mut args = vec![
            command.to_owned(),
            "--model-dir".to_owned(),
            root.to_str().unwrap().to_owned(),
            "--quality-backend".to_owned(),
            backend.to_owned(),
            "--quality-model-dir".to_owned(),
            quality.to_str().unwrap().to_owned(),
        ];
        for (flag, path) in extra {
            args.push((*flag).to_owned());
            args.push(path.to_str().unwrap().to_owned());
        }
        Options::parse(args).unwrap().unwrap()
    }

    fn roundtrip(backend: &str, quality_variable: &str, expected_model: &str) {
        let model_root = required_directory("FRANKENSEARCH_NATIVE_CLI_MODEL_ROOT");
        let quality_dir = required_directory(quality_variable);
        let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
            .blocking_threads(0, 2)
            .build()
            .unwrap();
        let pool = runtime.blocking_handle().unwrap();
        runtime.block_on(async move {
            let cx = Cx::current().expect("caller runtime installs a context");
            let work = tempfile::tempdir().unwrap();
            let input = work.path().join("documents.jsonl");
            let receipt = work.path().join("g1.json");
            let directory = work.path().join("g1");
            std::fs::write(&input, concat!(
                "{\"id\":\"retry.rs\",\"content\":\"Retry failed network requests with exponential backoff.\",\"metadata\":{\"language\":\"rust\"}}\n",
                "{\"id\":\"garden.md\",\"content\":\"Tomatoes and flowers grow in the summer garden.\"}\n",
            )).unwrap();
            let create = options("index", backend, &model_root, &quality_dir, &[
                ("--receipt", &receipt), ("--index-dir", &directory), ("--input", &input),
            ]);
            let mut output = Vec::new();
            execute(&cx, create, &mut output, Some(pool.clone())).await.unwrap();
            let indexed: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(indexed["event"], "indexed");
            let old_bytes = std::fs::read(&receipt).unwrap();
            let selection = Selection::read(&receipt).unwrap();
            let policy = QualityOptions {
                backend: Some(QualityBackend::parse(backend).unwrap()),
                directory: Some(quality_dir.clone()),
            };
            let models = load(&cx, Some(&model_root), true, &policy, Some(pool.clone())).unwrap();
            let quality = models.quality.as_ref().unwrap();
            assert_eq!(quality.id(), expected_model);
            assert!(quality.is_semantic());
            assert_eq!(selection.quality_producer.as_deref(), Some(quality.identity().unwrap().fingerprint().as_str()));
            let reader = selection.open(&cx, models.clone()).await.unwrap();
            let text = "retry failed network requests";
            for mode in [super::super::Mode::Fast, super::super::Mode::Quality, super::super::Mode::Full] {
                let page = query::buffered(&reader, &cx, text, mode, 2, [None, None],
                    &query::Policy::default()).await.unwrap();
                assert!(!page["results"].as_array().unwrap().is_empty());
                assert_eq!(page["generation"], serde_json::to_value(selection.generation).unwrap());
            }
            let scoped = filter::Filter::parse("{\"metadata\":{\"language\":\"rust\"}}").unwrap();
            let request = serve::Request {
                id: Some("native-real".to_owned()), query: text.to_owned(), mode: None,
                limit: Some(2), filter: Some(scoped), timeout_ms: None,
            };
            output.clear();
            assert!(serve::stream_one(&reader, &cx, &request, 1,
                (super::super::Mode::Full, 2), &mut output, None,
                &query::Policy::default()).await.unwrap());
            let frames: Vec<serde_json::Value> = output.split(|byte| *byte == b'\n')
                .filter(|line| !line.is_empty())
                .map(|line| serde_json::from_slice(line).unwrap()).collect();
            for phase in ["initial", "refined"] {
                let page = frames.iter().find(|frame| frame["phase"] == phase).unwrap();
                let hits = page["results"].as_array().unwrap();
                assert_eq!(hits.len(), 1);
                assert_eq!(hits[0]["doc_id"], "retry.rs");
            }
            // Same 384 dimensions and the same weights do not license a switch
            // between the native INT8 and F32 producers on an existing receipt.
            if backend == "native-int8" {
                let foreign: Arc<dyn Embedder> = Arc::new(
                    frankensearch::NativeEmbedder::load_model(&quality_dir,
                        frankensearch::NativeEmbeddingModel::AllMiniLmL6V2F32)
                        .unwrap().with_blocking_pool(pool.clone()),
                );
                assert_eq!(foreign.dimension(), quality.dimension());
                assert_ne!(foreign.identity().unwrap().fingerprint(), quality.identity().unwrap().fingerprint());
                let error = selection.open(&cx, Models {
                    fast: Arc::clone(&models.fast), quality: Some(foreign),
                }).await.err().expect("changed producer must be refused");
                assert!(error.to_string().contains("producers differ"));
            }
            // Exercise the actual update/rebuild command loaders, not just a
            // manually supplied model in an isolated builder test.
            let edits = work.path().join("edits.jsonl");
            std::fs::write(&edits,
                "{\"op\":\"upsert\",\"id\":\"queue.rs\",\"content\":\"A bounded queue applies backpressure.\"}\n").unwrap();
            let next_dir = work.path().join("g2");
            let next_receipt = work.path().join("g2.json");
            let update = options("update", backend, &model_root, &quality_dir, &[
                ("--receipt", &receipt), ("--index-dir", &next_dir),
                ("--new-receipt", &next_receipt), ("--input", &edits),
            ]);
            output.clear();
            execute(&cx, update, &mut output, Some(pool.clone())).await.unwrap();
            let next = Selection::read(&next_receipt).unwrap();
            assert_eq!(next.documents, 3);
            assert_eq!(next.quality_producer, selection.quality_producer);
            assert!(next.generation.sequence > selection.generation.sequence);
            let rebuilt_dir = work.path().join("g3");
            let rebuilt_receipt = work.path().join("g3.json");
            let rebuild = options("rebuild", backend, &model_root, &quality_dir, &[
                ("--receipt", &next_receipt), ("--index-dir", &rebuilt_dir),
                ("--new-receipt", &rebuilt_receipt),
            ]);
            output.clear();
            execute(&cx, rebuild, &mut output, Some(pool.clone())).await.unwrap();
            let rebuilt = Selection::read(&rebuilt_receipt).unwrap();
            assert_eq!(rebuilt.documents, 3);
            assert_eq!(rebuilt.quality_producer, selection.quality_producer);
            let fresh = rebuilt.open(&cx, models).await.unwrap();
            assert_eq!(fresh.vectors().documents().len(), 3);
            assert!(fresh.vectors().document("queue.rs").is_some());
            assert_eq!(reader.vectors().documents().len(), 2);
            assert!(reader.vectors().document("queue.rs").is_none());
            assert_eq!(std::fs::read(&receipt).unwrap(), old_bytes);
            assert_eq!(reader.vectors().document("retry.rs").unwrap().content,
                "Retry failed network requests with exponential backoff.");
        });
    }

    #[test]
    #[ignore = "requires installed Potion and verified MINILM_FIXTURE_DIR; no download or skip fallback"]
    fn native_int8_real_index_query_update_rebuild_and_wrong_producer_refusal() {
        roundtrip("native-int8", "MINILM_FIXTURE_DIR", "minilm-384-native");
    }

    #[test]
    #[ignore = "requires installed Potion and verified MINILM_FIXTURE_DIR; no download or skip fallback"]
    fn native_f32_real_index_query_update_and_rebuild() {
        roundtrip("native-f32", "MINILM_FIXTURE_DIR", "minilm-384-native-f32");
    }

    #[test]
    #[ignore = "requires installed Potion and MULTILINGUAL_MINILM_FIXTURE_DIR; no download or skip fallback"]
    fn native_multilingual_real_index_query_update_and_rebuild() {
        roundtrip(
            "native-multilingual",
            "MULTILINGUAL_MINILM_FIXTURE_DIR",
            "paraphrase-multilingual-minilm-l12-v2-384-native",
        );
    }
}
