//! Admission of the lexical records against the already-admitted source cohort.
//!
//! An exact file receipt proves which bytes were selected, not that independently
//! valid source/vector and lexical artifacts describe the same documents. Counts
//! and generation numbers are necessary but cannot establish that association.

use std::collections::BTreeMap;

use frankensearch_quill::{DEFAULT_SCHEMA, Query, QuillSearchIndex};

use crate::native_ann::{checkpoint, invalid};
use crate::{Cx, IndexableDocument, SearchResult};

/// The caller owns an unrefreshable shipping-schema reader and strictly ID-sorted
/// sources. Enumerate the complete live set, not a text-query sample: empty text
/// and punctuation-only documents are part of the cohort too. `Query::all` avoids
/// interpreting any source ID as query syntax. No writer or second reader opens.
///
/// Admission has full-cohort cost: the bounded match-all page plus a membership
/// bitmap are resident, while stored fields are compared one document at a time.
/// This uses ordinary Quill admission/collection limits; it does not disable fuel
/// or replace byte comparisons with the noncryptographic IDMAP content witness.
pub(super) fn validate(
    cx: &Cx,
    lexical: &QuillSearchIndex,
    documents: &[IndexableDocument],
) -> SearchResult<()> {
    checkpoint(cx, "native_ann.builder.lexical_source_start")?;
    let count = u64::try_from(documents.len()).map_err(|_| {
        invalid(
            "builder.lexical_source_join",
            "cardinality",
            "source count does not fit the lexical census",
        )
    })?;
    let page = lexical.search_preparsed_paginated(cx, &Query::all(), documents.len(), 0, true)?;
    checkpoint(cx, "native_ann.builder.lexical_source_census")?;
    if page.total_count != Some(count) || page.hits.len() != documents.len() {
        return Err(invalid(
            "builder.lexical_source_join",
            "cardinality",
            "the complete live lexical census must equal the source cohort",
        ));
    }
    let id_field = stored_field("id")?;
    let content_field = stored_field("content")?;
    let title_field = stored_field("title")?;
    let metadata_field = stored_field("metadata_json")?;
    let mut seen = vec![false; documents.len()];
    for hit in page.hits.iter() {
        checkpoint(cx, "native_ann.builder.lexical_source_document")?;
        let position = documents
            .binary_search_by(|document| document.id.as_str().cmp(hit.document_id.as_str()))
            .map_err(|_| {
                invalid(
                    "builder.lexical_source_join",
                    "document-id",
                    "a live lexical identity is absent from the source cohort",
                )
            })?;
        if std::mem::replace(&mut seen[position], true) {
            return Err(invalid(
                "builder.lexical_source_join",
                "duplicate-id",
                "each source identity must name exactly one live lexical document",
            ));
        }
        let document = &documents[position];
        // Quill's shipping writer serializes this ordered string map. Compare
        // exact canonical bytes, not a JSON parser that accepts duplicate keys.
        let ordered_metadata: BTreeMap<_, _> = document.metadata.iter().collect();
        let metadata = serde_json::to_vec(&ordered_metadata).map_err(|_| {
            invalid(
                "builder.lexical_source_join",
                "metadata-encoding",
                "could not encode the source document's canonical metadata",
            )
        })?;
        for (field, name, expected) in [
            (id_field, "id", document.id.as_bytes()),
            (content_field, "content", document.content.as_bytes()),
            // The shipping schema encodes absent and empty titles identically.
            // Keep the source's original Option; do not rewrite it to match.
            (
                title_field,
                "title",
                document.title.as_deref().unwrap_or("").as_bytes(),
            ),
            (metadata_field, "metadata", metadata.as_slice()),
        ] {
            checkpoint(cx, "native_ann.builder.lexical_source_field")?;
            let actual = lexical.stored_field_value(field, hit.global_docid)?;
            checkpoint(cx, "native_ann.builder.lexical_source_field_read")?;
            if actual.as_deref() != Some(expected) {
                return Err(invalid(
                    "builder.lexical_source_join",
                    name,
                    "stored lexical fields must match the exact retained source document",
                ));
            }
        }
    }
    // Unique membership plus equal cardinality establishes complete coverage,
    // irrespective of source order, FSVI physical rows or lexical global IDs.
    checkpoint(cx, "native_ann.builder.lexical_source_complete")
}

fn stored_field(name: &str) -> SearchResult<u16> {
    DEFAULT_SCHEMA
        .fields
        .iter()
        .find(|field| field.name == name && field.stored)
        .map(|field| field.id)
        .ok_or_else(|| {
            invalid(
                "builder.lexical_source_join",
                "schema",
                "the shipping lexical schema must retain every source field",
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::Arc;
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    use std::sync::atomic::Ordering;

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    use frankensearch_core::generation::GenerationComponentReceiptV1;
    use frankensearch_quill::{QuillConfig, QuillIndex};
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    use sha2::{Digest, Sha256};

    use super::super::NativeBuiltHybridIndex;
    use super::super::snapshot::LexicalSeal;
    use crate::native_ann::builder::tests::{Provider, Reply, generation};
    use crate::native_ann::builder::{NativeBuiltIndex, NativeIndexBuilder};
    use crate::{LexicalRead, LexicalWrite, SearchError};

    fn documents() -> Vec<IndexableDocument> {
        vec![
            IndexableDocument::new("z", "horizontal").with_title("original title"),
            IndexableDocument::new("b", "vertical")
                .with_metadata("zeta", "quote: \" and slash: \\")
                .with_metadata("alpha", "\u{000b}\nβ"),
            IndexableDocument::new("a: \"β\"", "").with_title(""),
            IndexableDocument::new("punctuation", "!!!\t—"),
        ]
    }

    async fn vectors(cx: &Cx, path: &Path, docs: Vec<IndexableDocument>) -> NativeBuiltIndex {
        NativeIndexBuilder::new(
            path,
            generation(),
            Arc::new(Provider::new("cohort-fast", 2, Reply::Correct)),
        )
        .unwrap()
        .add_documents(docs)
        .build(cx)
        .await
        .unwrap()
    }

    async fn lexical(
        cx: &Cx,
        path: &Path,
        docs: &[IndexableDocument],
    ) -> (QuillSearchIndex, LexicalSeal) {
        std::fs::create_dir(path).unwrap();
        let writer = Box::pin(QuillIndex::create(
            cx,
            path,
            QuillConfig {
                bulk_load_mode: true,
                deterministic_ingest: true,
                max_ingest_shards: 1,
                ..QuillConfig::default()
            },
        ))
        .await
        .unwrap();
        for document in docs {
            LexicalWrite::index_document(&writer, cx, document)
                .await
                .unwrap();
        }
        Box::pin(writer.finish_bulk_load(cx)).await.unwrap();
        let seal = LexicalSeal::capture(
            cx,
            path,
            writer.search_snapshot().unwrap().keeper_generation(),
        )
        .unwrap();
        let reader = Box::pin(QuillSearchIndex::open(cx, path, QuillConfig::default()))
            .await
            .unwrap();
        drop(writer);
        (reader, seal)
    }

    fn assert_join_error(error: SearchError, value: &str) {
        assert!(matches!(error, SearchError::InvalidConfig { ref field, value: ref actual, .. }
            if field == "native_ann.builder.lexical_source_join" && actual == value));
    }

    #[test]
    fn exact_cohort_admission_joins_ids_instead_of_ordinal_positions() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let source = documents();
            let vectors = vectors(&cx, &directory.path().join("vectors"), source.clone()).await;
            let mut different_order = source;
            different_order.reverse();
            // None and Some("") have the same actual Quill encoding, while the
            // source descriptor must keep the distinction for its consumer.
            different_order
                .iter_mut()
                .find(|doc| doc.id == "b")
                .unwrap()
                .title = Some(String::new());
            let (reader, seal) =
                lexical(&cx, &vectors.directory().join("lexical"), &different_order).await;
            let built = NativeBuiltHybridIndex::from_readers(&cx, vectors, reader, seal).unwrap();
            assert_eq!(built.lexical().doc_count().unwrap(), 4);
            assert!(built.vectors().document("b").unwrap().title.is_none());
            assert_eq!(
                built.search_refined(&cx, "vertical", 1).await.unwrap()[0].doc_id,
                "b"
            );
            assert_eq!(built.vectors().document("a: \"β\"").unwrap().content, "");
        });
    }

    #[test]
    fn equal_counts_and_valid_lexical_seals_do_not_admit_mixed_sources() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            type Change = fn(&mut IndexableDocument);
            let changes: &[(&str, Change)] = &[
                ("document-id", |doc| doc.id = "foreign-document".to_owned()),
                // Same analyzed terms, different original bytes.
                ("content", |doc| doc.content = "VERTICAL!".to_owned()),
                ("title", |doc| doc.title = Some("other title".to_owned())),
                ("metadata", |doc| {
                    doc.metadata.insert("alpha".to_owned(), "replacement".to_owned());
                }),
            ];
            for &(field, change) in changes {
                let directory = tempfile::tempdir().unwrap();
                let source = documents();
                let vectors = vectors(&cx, &directory.path().join("vectors"), source.clone()).await;
                let mut wrong = source;
                change(wrong.iter_mut().find(|doc| doc.id == "b").unwrap());
                let (reader, seal) = lexical(&cx, &vectors.directory().join("lexical"), &wrong).await;
                // The old admission conditions are all satisfied. Both vector
                // and lexical artifacts were produced by their real writers.
                assert_eq!(
                    LexicalRead::doc_count(&reader).unwrap(),
                    vectors.documents().len()
                );
                assert_eq!(reader.keeper_generation(), seal.generation());
                let result = NativeBuiltHybridIndex::from_readers(&cx, vectors, reader, seal);
                assert_join_error(result.err().expect("mixed source must be refused"), field);
            }
        });
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn public_build_and_selected_restart_accept_empty_and_full_source_records() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for docs in [Vec::new(), documents()] {
                let directory = tempfile::tempdir().unwrap();
                let fast = Arc::new(Provider::new("cohort-fast", 2, Reply::Correct));
                let quality = Arc::new(Provider::new("cohort-quality", 3, Reply::Correct));
                let path = directory.path().join("hybrid");
                let built = NativeIndexBuilder::new(&path, generation(), fast.clone())
                    .unwrap()
                    .with_quality_embedder(quality.clone())
                    .unwrap()
                    .add_documents(docs.clone())
                    .build_hybrid(&cx)
                    .await
                    .unwrap();
                let receipt = built.seal_for_reopen(&cx).unwrap();
                let reopened = NativeBuiltHybridIndex::open_selected(
                    &cx,
                    &path,
                    &receipt,
                    fast.clone(),
                    Some(quality.clone()),
                )
                .await
                .unwrap();
                assert_eq!(reopened.lexical().doc_count().unwrap(), docs.len());
                for document in docs {
                    let restored = reopened.vectors().document(&document.id).unwrap();
                    assert_eq!(restored.content, document.content);
                    assert_eq!(restored.title, document.title);
                    assert_eq!(restored.metadata, document.metadata);
                }
                assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
                assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
            }
        });
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn receipt(bytes: &[u8]) -> GenerationComponentReceiptV1 {
        GenerationComponentReceiptV1 {
            byte_len: u64::try_from(bytes.len()).unwrap(),
            sha256: Sha256::digest(bytes).into(),
        }
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn encoded_receipt(bytes: &[u8]) -> serde_json::Value {
        let receipt = receipt(bytes);
        serde_json::json!({"byte_len": receipt.byte_len, "sha256": receipt.sha256})
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn selected_restart_rejects_a_checksum_valid_but_mixed_source_descriptor() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("cohort-fast", 2, Reply::Correct));
            let path = directory.path().join("hybrid");
            let built = NativeIndexBuilder::new(&path, generation(), fast.clone())
                .unwrap()
                .add_documents(documents())
                .build_hybrid(&cx)
                .await
                .unwrap();
            built.seal_for_reopen(&cx).unwrap();
            // Model a caller-selected, internally checksum-valid descriptor
            // assembled from inconsistent artifacts. This is not a claim that
            // an attacker can forge the caller's original trusted receipt.
            let source = std::fs::read_to_string(path.join("native.source.jsonl")).unwrap();
            let mut rewritten = String::new();
            for (line, record) in source.lines().enumerate() {
                if line == 0 {
                    rewritten.push_str(record);
                } else {
                    let mut document: IndexableDocument = serde_json::from_str(record).unwrap();
                    if document.id == "b" {
                        "source text never indexed lexically".clone_into(&mut document.content);
                    }
                    rewritten.push_str(&serde_json::to_string(&document).unwrap());
                }
                rewritten.push('\n');
            }
            std::fs::write(path.join("native.source.jsonl"), &rewritten).unwrap();
            let vector_descriptor = path.join("native.snapshot.json");
            let mut vectors: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&vector_descriptor).unwrap()).unwrap();
            vectors["source"] = encoded_receipt(rewritten.as_bytes());
            let vectors = serde_json::to_vec(&vectors).unwrap();
            std::fs::write(vector_descriptor, &vectors).unwrap();
            let hybrid_descriptor = path.join("native.hybrid.json");
            let mut hybrid: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&hybrid_descriptor).unwrap()).unwrap();
            hybrid["vectors"] = encoded_receipt(&vectors);
            let hybrid = serde_json::to_vec(&hybrid).unwrap();
            std::fs::write(hybrid_descriptor, &hybrid).unwrap();
            // Prove that source hashes, vector ownership, producer identity and
            // source membership all pass their original independent admission.
            let vector_only = NativeBuiltIndex::open_selected(
                &cx,
                &path,
                &receipt(&vectors),
                fast.clone(),
                None,
            )
            .unwrap();
            assert_eq!(
                vector_only.document("b").unwrap().content,
                "source text never indexed lexically"
            );
            assert_eq!(
                vector_only.fast().index().owner_witness(),
                built.vectors().fast().index().owner_witness()
            );
            let result = NativeBuiltHybridIndex::open_selected(
                &cx,
                &path,
                &receipt(&hybrid),
                fast.clone(),
                None,
            )
            .await;
            assert_join_error(
                result.err().expect("mixed source must fail after hash admission"),
                "content",
            );
            assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
            assert_eq!(built.vectors().document("b").unwrap().content, "vertical");
            assert_eq!(built.search(&cx, "vertical", 1).await.unwrap()[0].doc_id, "b");
        });
    }

    #[test]
    fn cancelled_census_does_not_admit_a_partial_or_empty_cohort() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for mut docs in [Vec::new(), documents()] {
                let directory = tempfile::tempdir().unwrap();
                let (reader, _) = lexical(&cx, &directory.path().join("lexical"), &docs).await;
                docs.sort_by(|left, right| left.id.cmp(&right.id));
                cx.set_cancel_requested(true);
                assert!(matches!(
                    validate(&cx, &reader, &docs),
                    Err(SearchError::Cancelled { .. })
                ));
                cx.set_cancel_requested(false);
                validate(&cx, &reader, &docs).unwrap();
            }
        });
    }
}
