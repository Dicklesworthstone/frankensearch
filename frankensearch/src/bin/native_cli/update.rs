//! Incremental inference into a new selected native hybrid cohort. The original
//! receipt, source text, vectors, graphs and Quill publication remain untouched.

use std::collections::BTreeMap;

use super::*;

type Edits = BTreeMap<String, Option<IndexableDocument>>;

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Mutation {
    Upsert {
        id: String,
        content: String,
        title: Option<String>,
        #[serde(default)]
        metadata: HashMap<String, String>,
    },
    Delete { id: String },
}

pub(super) fn read_edits(reader: &mut impl BufRead) -> Result<Edits> {
    let mut edits = BTreeMap::new();
    let mut line = Vec::new();
    let mut total = 0_usize;
    let mut ordinal = 0_usize;
    loop {
        line.clear();
        let count = (&mut *reader)
            .take(MAX_RECORD_BYTES as u64 + 1)
            .read_until(b'\n', &mut line)?;
        if count == 0 {
            return Ok(edits);
        }
        ordinal += 1;
        total = total.checked_add(count).ok_or_else(|| bad("update byte count overflow"))?;
        if line.len() > MAX_RECORD_BYTES || total > MAX_INPUT_BYTES {
            return Err(bad("update exceeds its record or total input byte limit"));
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let mutation: Mutation = serde_json::from_slice(&line).map_err(|error| {
            bad(&format!(
                "invalid update on line {ordinal} ({:?} at column {})",
                error.classify(), error.column(),
            ))
        })?;
        let (id, document) = match mutation {
            Mutation::Upsert { id, content, title, metadata } => {
                let mut document = IndexableDocument::new(id.clone(), content);
                document.title = title;
                document.metadata = metadata;
                (id, Some(document))
            }
            Mutation::Delete { id } => (id, None),
        };
        if id.trim().is_empty() || id.contains('\0') || id.len() > usize::from(u16::MAX) {
            return Err(bad("update IDs must be nonblank, NUL-free, and at most 65535 bytes"));
        }
        edits.insert(id, document);
        if edits.len() > MAX_DOCUMENTS {
            return Err(bad("update exceeds the distinct edited-document limit"));
        }
    }
}

// Bound the final retained source cohort, not just the delta. Repeated updates
// must not evade the CLI's source limits by adding a bounded batch every time.
fn validate_final_cohort(cx: &Cx, index: &NativeBuiltHybridIndex, edits: &Edits) -> Result<()> {
    let mut count = 0_usize;
    let mut bytes = 0_usize;
    let mut admit = |document: &IndexableDocument| -> Result<()> {
        cx.checkpoint().map_err(|_| bad("native update cancelled"))?;
        let encoded = encode(document, MAX_RECORD_BYTES)?;
        count += 1;
        bytes = bytes.checked_add(encoded.len()).ok_or_else(|| bad("source byte count overflow"))?;
        if count > MAX_DOCUMENTS || bytes > MAX_INPUT_BYTES {
            return Err(bad("the final source cohort exceeds the document or encoded-byte limit"));
        }
        Ok(())
    };
    for previous in index.vectors().documents() {
        match edits.get(&previous.id) {
            Some(Some(document)) => admit(document)?,
            Some(None) => {}
            None => admit(previous)?,
        }
    }
    for (id, document) in edits {
        if index.vectors().document(id).is_none()
            && let Some(document) = document
        {
            admit(document)?;
        }
    }
    Ok(())
}

pub(super) async fn apply(
    cx: &Cx,
    previous: &Selection,
    index: &NativeBuiltHybridIndex,
    directory: &Path,
    edits: Edits,
    batch_size: usize,
) -> Result<(NativeBuiltHybridIndex, Selection)> {
    if index.vectors().fast().index().owner_witness().generation != previous.generation {
        return Err(bad("update source does not match the selected predecessor"));
    }
    validate_final_cohort(cx, index, &edits)?;
    let sequence = previous.generation.sequence.checked_add(1)
        .ok_or_else(|| bad("generation sequence exhausted"))?;
    let generation = new_generation(sequence)?;
    let mut update = index.begin_update(cx, directory, generation)?
        .with_batch_size(batch_size)?
        .with_max_batch_input_bytes(MAX_RECORD_BYTES)?;
    for (id, document) in edits {
        update = match document {
            Some(document) => update.upsert_document(document),
            None => update.delete_document(id),
        };
    }
    // The library owns exact per-tier reuse and rebuilds the complete Quill
    // cohort. No raw-vector copy, partial-tier publication or new merge path.
    let successor = update.build_hybrid(cx).await?;
    let snapshot = successor.seal_for_reopen(cx)?;
    let selection = Selection {
        schema: SELECTION_SCHEMA.to_owned(),
        directory: successor.vectors().directory().to_path_buf(),
        generation,
        snapshot: SnapshotReceipt { byte_len: snapshot.byte_len, sha256: snapshot.sha256 },
        documents: successor.vectors().documents().len(),
        fast_producer: previous.fast_producer.clone(),
        quality_producer: previous.quality_producer.clone(),
    };
    Ok((successor, selection))
}

pub(super) async fn execute(cx: &Cx, options: &Options, output: &mut impl Write) -> Result<()> {
    let previous = Selection::read(&options.receipt)?;
    let old_root = fs::canonicalize(&previous.directory)?;
    let directory = new_path(options.directory.as_deref().ok_or_else(|| bad("missing new index directory"))?)?;
    let receipt = new_path(options.new_receipt.as_deref().ok_or_else(|| bad("missing --new-receipt"))?)?;
    if directory.starts_with(&old_root)
        || receipt.starts_with(&old_root)
        || directory.starts_with(&receipt)
        || receipt.starts_with(&directory)
    {
        return Err(bad("new destinations must be outside the predecessor and must not overlap"));
    }
    let edits = match &options.input {
        Some(path) => read_edits(&mut BufReader::new(File::open(path)?))?,
        None => read_edits(&mut io::stdin().lock())?,
    };
    let edited_ids = edits.len();
    let models = load_models(options.models.as_deref(), previous.quality_producer.is_some())?;
    let index = previous.open(cx, models).await?;
    let (_successor, selection) = apply(cx, &previous, &index, &directory, edits, options.batch_size).await?;
    save_selection(&selection, &receipt)?;
    emit(output, &serde_json::json!({
        "schema": SCHEMA, "ok": true, "event": "updated",
        "predecessor": previous.generation, "edited_ids": edited_ids,
        "receipt": receipt, "selection": selection,
    }))
}
