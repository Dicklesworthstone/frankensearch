//! Production-adapter regressions for resumable update ordering.

use asupersync::Cx;
use asupersync::test_utils::run_test_with_cx;
use frankensearch_quill::{QuillConfig, QuillIndex, SegmentStatsProvider};

use frankensearch_fsfs::config::IngestionClass;
use frankensearch_fsfs::{LexicalMutation, LexicalPipeline, QuillLexicalBackend};

fn index() -> QuillIndex {
    QuillIndex::in_memory(QuillConfig {
        max_ingest_shards: 1,
        deterministic_ingest: true,
        ..QuillConfig::default()
    })
    .expect("create deterministic Quill index")
}

fn upsert(id: &str, revision: u64, text: &str) -> LexicalMutation {
    LexicalMutation::upsert(
        id,
        revision,
        IngestionClass::FullSemanticLexical,
        text,
        "resumable ordering regression",
    )
}

async fn seed(cx: &Cx, index: &QuillIndex) {
    let mut pipeline = LexicalPipeline::new(QuillLexicalBackend::new(index));
    pipeline
        .apply_initial(&[upsert("doc", 1, "alpha original body")])
        .expect("plan seed");
    pipeline.backend_mut().flush(cx).await.expect("stage seed");
    index.commit(cx).await.expect("publish seed");
}

fn assert_restored(cx: &Cx, index: &QuillIndex) {
    assert_eq!(
        index
            .search_doc_ids(cx, "alpha", 10)
            .expect("query restored text")
            .iter()
            .map(|hit| hit.document_id.clone())
            .collect::<Vec<_>>(),
        ["doc"]
    );
    assert!(
        index
            .search_doc_ids(cx, "beta", 10)
            .expect("query superseded text")
            .is_empty()
    );
    assert_eq!(index.segment_stats().expect("live stats").live_docs, 1);
}

#[test]
fn same_batch_restore_must_not_be_skipped_against_old_published_hash() {
    run_test_with_cx(|cx| async move {
        let index = index();
        seed(&cx, &index).await;
        let original = index.document_witness("doc").expect("original witness");
        let mut pipeline = LexicalPipeline::new(QuillLexicalBackend::new(&index));
        pipeline
            .apply_incremental(&[
                upsert("doc", 2, "beta replacement body"),
                upsert("doc", 3, "alpha original body"),
            ])
            .expect("plan change followed by restoration");
        let stats = pipeline
            .backend_mut()
            .flush_resumable(&cx)
            .await
            .expect("flush ordered replacements");
        assert_eq!(stats.unchanged, 0, "restoration is not a no-op");
        assert_eq!(stats.changed, 2);
        assert_eq!(pipeline.backend().pending_len(), 0);
        index.commit(&cx).await.expect("publish restoration");
        assert_restored(&cx, &index);
        assert_eq!(
            index
                .document_witness("doc")
                .expect("restored witness")
                .expect("restored row")
                .content_hash,
            original.expect("original row").content_hash
        );
    });
}

#[test]
fn prior_uncommitted_flush_must_not_authorize_a_stale_hash_skip() {
    run_test_with_cx(|cx| async move {
        let index = index();
        seed(&cx, &index).await;
        let mut first = LexicalPipeline::new(QuillLexicalBackend::new(&index));
        first
            .apply_incremental(&[upsert("doc", 2, "beta replacement body")])
            .expect("plan uncommitted replacement");
        first
            .backend_mut()
            .flush(&cx)
            .await
            .expect("stage replacement");
        assert!(index.has_uncommitted_changes());
        drop(first);

        // A fresh adapter must also see that the index has pending writes;
        // tracking only this adapter's current batch is insufficient.
        let mut second = LexicalPipeline::new(QuillLexicalBackend::new(&index));
        second
            .apply_incremental(&[upsert("doc", 3, "alpha original body")])
            .expect("plan restoration through another adapter");
        let stats = second
            .backend_mut()
            .flush_resumable(&cx)
            .await
            .expect("stage restoration");
        assert_eq!(stats.unchanged, 0);
        assert_eq!(stats.changed, 1);
        index.commit(&cx).await.expect("publish restoration");
        assert_restored(&cx, &index);
    });
}

#[test]
fn delete_then_restore_preserves_the_final_upsert() {
    run_test_with_cx(|cx| async move {
        let index = index();
        seed(&cx, &index).await;
        let mut pipeline = LexicalPipeline::new(QuillLexicalBackend::new(&index));
        pipeline
            .apply_incremental(&[
                LexicalMutation::delete(
                    "doc",
                    2,
                    IngestionClass::FullSemanticLexical,
                    "delete barrier",
                ),
                upsert("doc", 3, "alpha original body"),
            ])
            .expect("plan delete and restoration");
        let stats = pipeline
            .backend_mut()
            .flush_resumable(&cx)
            .await
            .expect("flush delete and restoration");
        assert_eq!(stats.deleted, 1);
        assert_eq!(stats.unchanged, 0);
        index.commit(&cx).await.expect("publish restoration");
        assert_restored(&cx, &index);
    });
}

#[test]
fn clean_unchanged_document_still_preserves_its_published_docid() {
    run_test_with_cx(|cx| async move {
        let index = index();
        seed(&cx, &index).await;
        let before = index.document_witness("doc").expect("original witness");
        let mut pipeline = LexicalPipeline::new(QuillLexicalBackend::new(&index));
        pipeline
            .apply_incremental(&[upsert("doc", 2, "alpha original body")])
            .expect("plan equal content");
        let stats = pipeline
            .backend_mut()
            .flush_resumable(&cx)
            .await
            .expect("skip equal content");
        assert_eq!(stats.unchanged, 1);
        assert_eq!(stats.changed, 0);
        assert_eq!(stats.absent, 0);
        assert!(!index.has_uncommitted_changes());
        assert_eq!(
            index.document_witness("doc").expect("current witness"),
            before
        );
        assert_eq!(pipeline.backend().pending_len(), 0);
    });
}

#[test]
fn another_document_in_the_batch_does_not_disable_clean_resume() {
    run_test_with_cx(|cx| async move {
        let index = index();
        seed(&cx, &index).await;
        let before = index.document_witness("doc").expect("original witness");
        let mut pipeline = LexicalPipeline::new(QuillLexicalBackend::new(&index));
        pipeline
            .apply_incremental(&[
                upsert("other", 1, "gamma independent body"),
                upsert("doc", 2, "alpha original body"),
            ])
            .expect("plan independent change and equal row");
        let stats = pipeline
            .backend_mut()
            .flush_resumable(&cx)
            .await
            .expect("flush mixed independent rows");
        assert_eq!(stats.absent, 1);
        assert_eq!(stats.unchanged, 1);
        index.commit(&cx).await.expect("publish independent change");
        assert_eq!(
            index.document_witness("doc").expect("current witness"),
            before
        );
        assert_eq!(index.segment_stats().expect("live stats").live_docs, 2);
    });
}
