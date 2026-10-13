//! Explicit model replacement in a warm native stdio session.
//!
//! Read only the controller's original sealed selection, load explicitly named
//! local models, and delegate all source/identity/policy joins to the native
//! migration API. Ordinary activation still forbids model changes. This control
//! does not rebuild, write receipts, switch durable authority, or grant GC.

use asupersync::runtime::blocking_pool::BlockingPoolHandle;
use frankensearch::SearchError;
use frankensearch::native_ann::builder::NativeHybridReopenLimits;
use frankensearch::native_ann::builder::live::NativeHybridSnapshot;
use frankensearch::native_ann::builder::sharded::NativeShardedHybridReopenLimits;
use frankensearch::native_ann::builder::sharded::live::NativeShardedSnapshot;

use super::validate_id;
use crate::{
    ArtifactGenerationIdentityV1, Cx, Deserialize, GenerationComponentReceiptV1, HnswParams,
    MAX_DOCUMENTS, Models, NativeBuildPrecision, NativeBuildRetrieval, Path, PathBuf, Result,
    SCHEMA, SELECTION_SCHEMA, Selection, Write, bad, cohort, emit, live, models, query, sharded,
};

type Loader = dyn Fn(&Cx, &Path, bool, &models::QualityOptions) -> Result<Models> + Send + Sync;

/// Constructed only by a startup grant. The loader is the same local-only,
/// verified loader used by the executable's index/search/rebuild commands.
/// Keeping the seam here lets tests exercise real receipt admission and the
/// actual control loop with synthetic providers, not downloaded model assets.
pub(super) struct Authority {
    load: Box<Loader>,
}

impl Authority {
    pub(super) fn local(pool: Option<BlockingPoolHandle>) -> Self {
        Self {
            load: Box::new(move |cx, directory, quality, options| {
                models::load(cx, Some(directory), quality, options, pool.clone())
            }),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Request {
    ActivateModelMigration {
        id: Option<String>,
        expected_generation: ArtifactGenerationIdentityV1,
        receipt: PathBuf,
        model_dir: PathBuf,
        quality_backend: Option<String>,
        quality_model_dir: Option<PathBuf>,
        // Required, not an inferred interpretation of an on-disk descriptor.
        // Match the original CLI rebuild's --exact policy (F32 in both tiers).
        exact: bool,
    },
}

impl Request {
    fn id(&self) -> Option<&str> {
        let Self::ActivateModelMigration { id, .. } = self;
        id.as_deref()
    }
}

fn checkpoint(cx: &Cx) -> Result<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: "native_cli.model_migration".to_owned(),
        reason: error.to_string(),
    })?;
    Ok(())
}

fn validate_path(path: &Path) -> Result<()> {
    let value = path
        .to_str()
        .ok_or_else(|| bad("migration paths must be UTF-8"))?;
    if !path.is_absolute() || value.len() > 4096 || value.contains('\0') {
        return Err(bad(
            "migration paths must be absolute, NUL-free, and at most 4096 bytes",
        ));
    }
    Ok(())
}

struct Prepared {
    candidate: live::Candidate,
    selection: Selection,
}

struct Selected {
    selection: Selection,
    models: Models,
    retrieval: NativeBuildRetrieval,
}

// Both layouts share authority-independent preflight, receipt selection and
// model admission. Only the native typed migration/open operation differs.
// Never infer a layout by trying one opener and falling back to the other.
fn select(
    base: cohort::Index<'_>,
    cx: &Cx,
    request: &Request,
    authority: &Authority,
) -> Result<Selected> {
    let Request::ActivateModelMigration {
        expected_generation,
        receipt,
        model_dir,
        quality_backend,
        quality_model_dir,
        exact,
        ..
    } = request;
    checkpoint(cx)?;
    expected_generation.validate()?;
    if *expected_generation != base.generation() {
        return Err(bad(
            "expected_generation does not match the serving snapshot; no models or receipt read",
        ));
    }
    validate_path(receipt)?;
    validate_path(model_dir)?;
    if let Some(path) = quality_model_dir {
        validate_path(path)?;
    }
    let options = models::QualityOptions {
        backend: quality_backend
            .as_deref()
            .map(models::QualityBackend::parse)
            .transpose()?,
        directory: quality_model_dir.clone(),
    };
    let outcome = Selection::read(receipt);
    checkpoint(cx)?;
    let selection = outcome?;
    let schema = match base {
        cohort::Index::Single(_) => SELECTION_SCHEMA,
        cohort::Index::Sharded(_) => sharded::SELECTION_SCHEMA,
    };
    if selection.schema != schema
        || selection.documents != base.document_count()
        || selection.generation.sequence <= base.generation().sequence
    {
        return Err(bad(
            "migration requires a strictly newer same-layout selection of the entire retained source cohort",
        ));
    }
    let has_quality = selection.quality_producer.is_some();
    options.validate_usage(has_quality)?;
    let outcome = (authority.load)(cx, model_dir, has_quality, &options);
    checkpoint(cx)?;
    let models = outcome?;
    let admission = selection.admit_models(&models);
    checkpoint(cx)?;
    admission?;
    let retrieval = if *exact {
        NativeBuildRetrieval::Exact
    } else {
        NativeBuildRetrieval::Hnsw {
            params: HnswParams::default(),
            seed: 42,
        }
    };
    Ok(Selected {
        selection,
        models,
        retrieval,
    })
}

async fn prepare(
    base: &NativeHybridSnapshot,
    cx: &Cx,
    request: &Request,
    authority: &Authority,
) -> Result<Prepared> {
    let Selected {
        selection,
        models,
        retrieval,
    } = select(base.index().into(), cx, request, authority)?;
    let outcome = base.begin_model_migration(
        cx,
        &selection.directory,
        selection.generation,
        models.fast,
        models.quality,
    );
    checkpoint(cx)?;
    let mut migration = outcome?.with_fast_storage(NativeBuildPrecision::F32, retrieval);
    if selection.quality_producer.is_some() {
        migration = migration.with_quality_storage(NativeBuildPrecision::F32, retrieval)?;
    }
    let expected = GenerationComponentReceiptV1 {
        byte_len: selection.snapshot.byte_len,
        sha256: selection.snapshot.sha256,
    };
    let mut limits = NativeHybridReopenLimits::default();
    limits.vectors.max_documents = MAX_DOCUMENTS;
    // This single library operation verifies all source fields, both complete
    // producer identities, nonce, vector precision, graphs, and Quill. Never
    // merely open the candidate and create a new unrelated serving handle.
    let outcome = Box::pin(migration.open_selected(cx, &expected, limits)).await;
    checkpoint(cx)?;
    Ok(Prepared {
        candidate: live::Candidate::Single(outcome?),
        selection,
    })
}

async fn prepare_sharded(
    base: &NativeShardedSnapshot,
    cx: &Cx,
    request: &Request,
    authority: &Authority,
) -> Result<Prepared> {
    let Selected {
        selection,
        models,
        retrieval,
    } = select(base.index().into(), cx, request, authority)?;
    let outcome = base.begin_model_migration(
        cx,
        &selection.directory,
        selection.generation,
        models.fast,
        models.quality,
    );
    checkpoint(cx)?;
    let mut migration = outcome?.with_fast_storage(NativeBuildPrecision::F32, retrieval);
    if selection.quality_producer.is_some() {
        migration = migration.with_quality_storage(NativeBuildPrecision::F32, retrieval)?;
    }
    let expected = GenerationComponentReceiptV1 {
        byte_len: selection.snapshot.byte_len,
        sha256: selection.snapshot.sha256,
    };
    let mut limits = NativeShardedHybridReopenLimits::default();
    limits.vectors.max_documents = MAX_DOCUMENTS;
    // Admit every partition plus global Quill and exact source fields through
    // the existing migration gate. Its private candidate keeps THIS base pin.
    // The receipt chooses partition boundaries, not a new untrusted shard-size
    // parameter. A missing final artifact never becomes an accepted prefix.
    let outcome = Box::pin(migration.open_selected(cx, &expected, limits)).await;
    checkpoint(cx)?;
    Ok(Prepared {
        candidate: live::Candidate::Sharded(outcome?),
        selection,
    })
}

pub(super) async fn execute<W: Write>(
    live: live::Live<'_>,
    cx: &Cx,
    request: &Request,
    ordinal: u64,
    authority: Option<&Authority>,
    output: &mut W,
) -> Result<()> {
    let mut previous = None;
    let result: Result<serde_json::Value> = async {
        checkpoint(cx)?;
        validate_id(request.id())?;
        let authority = authority.ok_or_else(|| {
            bad("model migration is disabled; start serve with --allow-model-migration for a trusted controller")
        })?;
        let base = live.snapshot(cx).await?;
        previous = Some(base.generation());
        let prepared = match &base {
            live::Snapshot::Single(base) => prepare(base, cx, request, authority).await?,
            live::Snapshot::Sharded(base) => prepare_sharded(base, cx, request, authority).await?,
        };
        // Exact original predecessor Arc, not a new snapshot after loading.
        // All fallible model/receipt checks precede this atomic operation.
        let installed = live.install(cx, &prepared.candidate).await?;
        // No checkpoint or fallible provider call after the visible swap.
        let mut payload = serde_json::json!({
            "ok": true, "status": "complete", "selection_changed": true,
            "generation": installed.generation(), "previous_generation": base.generation(),
            "documents": prepared.selection.documents,
            "quality": prepared.selection.quality_producer.is_some(),
            "fast_producer": prepared.selection.fast_producer,
            "quality_producer": prepared.selection.quality_producer,
        });
        installed.index().annotate(&mut payload);
        Ok(payload)
    }
    .await;
    let mut payload = match result {
        Ok(payload) => payload,
        Err(error) => {
            let mut payload = query::failure(error.as_ref());
            payload["ok"] = serde_json::json!(false);
            payload["previous_generation"] = serde_json::json!(previous);
            // This request did not install. Another writer may have won while
            // admission was running; do not report that writer's work undone.
            payload["selection_changed"] = serde_json::json!(false);
            payload
        }
    };
    payload["schema"] = serde_json::json!(SCHEMA);
    payload["event"] = serde_json::json!("terminal");
    payload["operation"] = serde_json::json!("activate_model_migration");
    payload["request"] = serde_json::json!(ordinal);
    payload["id"] = serde_json::json!(request.id().filter(|id| validate_id(Some(id)).is_ok()));
    payload["seq"] = serde_json::json!(0);
    payload["partial_results"] = serde_json::json!(false);
    payload["activation_scope"] = serde_json::json!("process_local");
    // Delivery failure stops the stream; never append an error claiming a
    // completed swap was rolled back. The status control resolves lost acks.
    emit(output, &payload)
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
#[path = "model_migration_tests.rs"]
mod tests;

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
#[path = "model_migration_sharded_tests.rs"]
mod sharded_tests;
