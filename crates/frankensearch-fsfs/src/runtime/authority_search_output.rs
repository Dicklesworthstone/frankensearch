//! Buffered display formats for the retained native result. Machine envelopes
//! and progressive stream frames use the ordinary fsfs emitters.

use std::io::Write;

use frankensearch::native_ann::NativeGenerationSnapshot;
use frankensearch_core::SearchResult;

use super::super::sha256_digest_hex;
use super::AuthorityPage;
use crate::adapters::format_emitter::{
    collect_query_terms, detect_terminal_width, focused_preview, terminal_safe, write_csv_row,
    write_err,
};

/// Authority references serialize as byte arrays on the wire. Their canonical
/// SHA-256 fingerprints identify the same selection compactly in display output.
struct Selection {
    sequence: u64,
    generation: u64,
    authority_sha256: String,
    head_sha256: Option<String>,
}

impl Selection {
    fn from_snapshot(snapshot: &NativeGenerationSnapshot) -> Self {
        let authority = snapshot.authority();
        let retained = snapshot.retained_snapshot();
        Self {
            sequence: authority.sequence,
            generation: retained.manifest().generation().sequence,
            authority_sha256: sha256_digest_hex(authority.fingerprint()),
            head_sha256: retained
                .selected_history_of()
                .map(|head| sha256_digest_hex(head.fingerprint())),
        }
    }
}

pub(super) fn write_table(
    page: &AuthorityPage,
    snapshot: &NativeGenerationSnapshot,
    elapsed_ms: u64,
    writer: &mut impl Write,
) -> SearchResult<()> {
    let selection = Selection::from_snapshot(snapshot);
    let query = page.evidence["query"].as_str().unwrap_or_default();
    let kind = page.evidence["selection_kind"].as_str().unwrap_or_default();
    let topology = page.evidence["topology"].as_str().unwrap_or_default();
    writeln!(
        writer,
        "{} results for \"{}\"  ({} ms)",
        page.results.len(),
        terminal_safe(query),
        elapsed_ms,
    )
    .map_err(write_err)?;
    writeln!(
        writer,
        "{} / {}  ·  {}  ·  {}  ·  generation {}",
        page.mode, page.phase, topology, kind, selection.generation,
    )
    .map_err(write_err)?;
    writeln!(
        writer,
        "Authority {}  {}",
        selection.sequence, selection.authority_sha256
    )
    .map_err(write_err)?;
    if let Some(head) = selection.head_sha256 {
        writeln!(writer, "Authenticated by {head}").map_err(write_err)?;
    }
    if let Some(eligible) = page.evidence["scope"]["eligible_documents"].as_u64() {
        writeln!(
            writer,
            "Eligible documents  {eligible} of {}",
            snapshot.metadata().total_documents
        )
        .map_err(write_err)?;
    }
    let terms = collect_query_terms(query);
    let width = detect_terminal_width().saturating_sub(8).max(32);
    for (index, hit) in page.results.iter().enumerate() {
        let scored = &hit.result;
        write!(
            writer,
            "{:>3}. {}  score={}  [{:?}]",
            index + 1,
            terminal_safe(scored.doc_id.as_str()),
            scored.score,
            scored.source,
        )
        .map_err(write_err)?;
        for (label, score) in [
            ("lexical", scored.lexical_score),
            ("fast", scored.fast_score),
            ("quality", scored.quality_score),
            ("rerank", scored.rerank_score),
        ] {
            if let Some(score) = score {
                write!(writer, "  {label}={score}").map_err(write_err)?;
            }
        }
        writeln!(writer).map_err(write_err)?;
        if let Some(title) = hit.title.as_deref().filter(|s| !s.is_empty()) {
            writeln!(writer, "     {}", terminal_safe(title)).map_err(write_err)?;
        }
        if let Some(path) = hit
            .source_path
            .as_deref()
            .filter(|s| *s != scored.doc_id.as_str())
        {
            writeln!(writer, "     Path  {}", terminal_safe(path)).map_err(write_err)?;
        }
        if !hit.snippet.is_empty() {
            let preview = focused_preview(&hit.snippet, &terms, width);
            writeln!(writer, "     {}", terminal_safe(&preview)).map_err(write_err)?;
        }
    }
    Ok(())
}

pub(super) fn write_csv(
    page: &AuthorityPage,
    snapshot: &NativeGenerationSnapshot,
    writer: &mut impl Write,
) -> SearchResult<()> {
    write_csv_row(
        writer,
        &[
            "query",
            "mode",
            "phase",
            "topology",
            "selection_kind",
            "generation_sequence",
            "authority_sequence",
            "authority_sha256",
            "selected_history_of_sha256",
            "rank",
            "doc_id",
            "title",
            "source_path",
            "score",
            "score_source",
            "lexical_score",
            "fast_score",
            "quality_score",
            "rerank_score",
            "snippet",
        ],
    )?;
    let selection = Selection::from_snapshot(snapshot);
    let generation = selection.generation.to_string();
    let sequence = selection.sequence.to_string();
    let text = |key: &str| page.evidence[key].as_str().unwrap_or_default();
    for (index, hit) in page.results.iter().enumerate() {
        let scored = &hit.result;
        let optional_score = |score: Option<f32>| score.map(|s| s.to_string()).unwrap_or_default();
        write_csv_row(
            writer,
            &[
                text("query"),
                page.mode,
                page.phase,
                text("topology"),
                text("selection_kind"),
                &generation,
                &sequence,
                &selection.authority_sha256,
                selection.head_sha256.as_deref().unwrap_or_default(),
                &(index + 1).to_string(),
                scored.doc_id.as_str(),
                hit.title.as_deref().unwrap_or_default(),
                hit.source_path.as_deref().unwrap_or_default(),
                &scored.score.to_string(),
                &format!("{:?}", scored.source),
                &optional_score(scored.lexical_score),
                &optional_score(scored.fast_score),
                &optional_score(scored.quality_score),
                &optional_score(scored.rerank_score),
                &hit.snippet,
            ],
        )?;
    }
    Ok(())
}
