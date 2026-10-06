//! Receipt-selected partitioned native builds and executable dispatch.
//! There is no filename discovery, model substitution or alternate search engine.

use frankensearch::native_ann::builder::sharded::{
    MAX_NATIVE_BUILD_SHARDS, NativeBuiltShardedHybridIndex, NativeShardedHybridReopenLimits,
};

use super::{
    Cx, GenerationComponentReceiptV1, IndexableDocument, MAX_DOCUMENTS, Models,
    NativeBuiltHybridIndex, Options, Path, Result, Selection, SnapshotReceipt, bad, cohort,
    configured_builder, new_generation,
};

pub const SELECTION_SCHEMA: &str = "frankensearch.native.sharded-selection.v1";

/// Receipt schema, not a guessed filename or failed ordinary open, selects this
/// enum. Boxing keeps the command future from containing both large owners.
pub enum Opened {
    Single(Box<NativeBuiltHybridIndex>),
    Sharded(Box<NativeBuiltShardedHybridIndex>),
}

impl Opened {
    pub async fn open(cx: &Cx, selection: &Selection, models: Models) -> Result<Self> {
        match selection.schema.as_str() {
            super::SELECTION_SCHEMA => {
                Ok(Self::Single(Box::new(selection.open(cx, models).await?)))
            }
            SELECTION_SCHEMA => Ok(Self::Sharded(Box::new(open(cx, selection, models).await?))),
            _ => Err(bad(
                "unknown selected layout; no discovery or fallback is permitted",
            )),
        }
    }

    pub fn borrow(&self) -> cohort::Index<'_> {
        match self {
            Self::Single(index) => cohort::Index::Single(index.as_ref()),
            Self::Sharded(index) => cohort::Index::Sharded(index.as_ref()),
        }
    }
}

pub fn validate_size(size: usize, documents: usize) -> Result<()> {
    if size == 0 || size > MAX_DOCUMENTS || documents > MAX_DOCUMENTS {
        return Err(bad(
            "shard size must be between 1 and 100000; corpus limit is 100000 documents",
        ));
    }
    if documents.div_ceil(size).max(1) > MAX_NATIVE_BUILD_SHARDS {
        return Err(bad(
            "the requested shard size would exceed the 1024-partition limit",
        ));
    }
    Ok(())
}

pub async fn build(
    cx: &Cx,
    options: &Options,
    directory: &Path,
    documents: Vec<IndexableDocument>,
    models: Models,
    size: usize,
) -> Result<(NativeBuiltShardedHybridIndex, Selection)> {
    validate_size(size, documents.len())?;
    let generation = new_generation(1)?;
    let fast_producer = models.fast.identity()?.fingerprint();
    let quality_producer = models
        .quality
        .as_ref()
        .map(|model| model.identity().map(|identity| identity.fingerprint()))
        .transpose()?;
    let index = configured_builder(options, directory, generation, documents, models)?
        .build_sharded_hybrid(cx, size)
        .await?;
    // Only the complete hybrid seal binds global Quill AND every partition.
    // A child/vector-only receipt is never promoted to hybrid authority.
    let snapshot = index.seal_for_reopen(cx)?;
    let selection = Selection {
        schema: SELECTION_SCHEMA.to_owned(),
        directory: index.vectors().directory().to_path_buf(),
        generation,
        snapshot: SnapshotReceipt {
            byte_len: snapshot.byte_len,
            sha256: snapshot.sha256,
        },
        documents: index.vectors().document_count(),
        fast_producer,
        quality_producer,
    };
    Ok((index, selection))
}

pub async fn open(
    cx: &Cx,
    selection: &Selection,
    models: Models,
) -> Result<NativeBuiltShardedHybridIndex> {
    if selection.schema != SELECTION_SCHEMA || selection.documents > MAX_DOCUMENTS {
        return Err(bad(
            "expected a bounded, explicit sharded selection receipt",
        ));
    }
    selection.admit_models(&models)?;
    let expected = GenerationComponentReceiptV1 {
        byte_len: selection.snapshot.byte_len,
        sha256: selection.snapshot.sha256,
    };
    let mut limits = NativeShardedHybridReopenLimits::default();
    limits.vectors.max_documents = MAX_DOCUMENTS;
    let index = NativeBuiltShardedHybridIndex::open_selected(
        cx,
        &selection.directory,
        &expected,
        models.fast,
        models.quality,
        limits,
    )
    .await?;
    if index.vectors().generation() != selection.generation
        || index.vectors().document_count() != selection.documents
    {
        return Err(bad(
            "reopened shard inventory differs from its trusted selection",
        ));
    }
    Ok(index)
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
#[path = "sharded_tests.rs"]
mod tests;
