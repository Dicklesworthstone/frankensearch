use super::*;
use asupersync::test_utils::run_test_with_cx;

fn docs() -> Vec<IndexableDocument> {
    ["é", "猫", "", "ab", "x"]
        .into_iter()
        .enumerate()
        .map(|(n, text)| IndexableDocument::new(format!("doc-{n}"), text))
        .collect()
}

#[test]
fn byte_budget_partitions_utf8_without_truncation_and_combines_with_count_limit() {
    run_test_with_cx(|cx| async move {
        let docs = docs();
        let mut start = 0;
        let mut groups = Vec::new();
        while start < docs.len() {
            let end = batch_end(&cx, &docs, start, 3, Some(5)).unwrap();
            assert!(end > start);
            assert!(end - start <= 3);
            assert!(
                docs[start..end]
                    .iter()
                    .map(|doc| doc.content.len())
                    .sum::<usize>()
                    <= 5
            );
            groups.push(
                docs[start..end]
                    .iter()
                    .map(|doc| doc.content.as_str())
                    .collect::<Vec<_>>(),
            );
            start = end;
        }
        assert_eq!(groups, vec![vec!["é", "猫", ""], vec!["ab", "x"]]);
        assert_eq!(batch_end(&cx, &docs, 0, 2, None).unwrap(), 2);
        assert_eq!(batch_end(&cx, &[], 0, 1, Some(5)).unwrap(), 0);
    });
}

#[test]
fn invalid_limits_and_offsets_fail_closed() {
    run_test_with_cx(|cx| async move {
        let docs = docs();
        assert!(validate_document_size(6, Some(5)).is_err());
        assert!(validate_document_size(5, Some(5)).is_ok());
        assert!(validate_document_size(usize::MAX, None).is_ok());
        assert!(batch_end(&cx, &docs, 0, 0, None).is_err());
        assert!(batch_end(&cx, &docs, 0, 2, Some(0)).is_err());
        assert!(batch_end(&cx, &docs, usize::MAX, 2, Some(5)).is_err());
    });
}

#[test]
fn cancellation_precedes_partitioning() {
    run_test_with_cx(|cx| async move {
        let docs = docs();
        cx.set_cancel_requested(true);
        assert!(matches!(
            batch_end(&cx, &docs, 0, 3, Some(5)),
            Err(frankensearch_core::SearchError::Cancelled { .. })
        ));
    });
}
