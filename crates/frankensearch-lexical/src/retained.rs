//! Engine-neutral lexical reads bound to one Tantivy publication.

use std::collections::HashMap;
use std::sync::Arc;

use frankensearch_core::traits::{LexicalRead, SearchFuture};

use super::{
    BooleanQuery, Cx, DocId, DocSetCollector, Index, IndexRecordOption, Occur, Query, QueryParser,
    SchemaFields, ScoreSource, ScoredResult, SearchError, SearchResult, Searcher, TITLE_BOOST,
    TantivyDocument, TantivyIndex, Term, TermQuery, Value, execute_top_k, load_doc, repair_and_not,
    repair_boosted_group_negation, repair_negated_conjunction, search_guarded,
};

/// Parse with the same shipping repairs and field boosts as the mutable facade.
/// The gauntlet's deliberately unmodified oracle parser remains separate.
pub(super) fn parse_shipping_query(
    index: &Index,
    fields: SchemaFields,
    query: &str,
) -> Box<dyn Query> {
    let mut parser = QueryParser::for_index(index, vec![fields.content, fields.title]);
    parser.set_field_boost(fields.title, TITLE_BOOST);
    let conjunction = repair_negated_conjunction(query);
    let repaired = repair_and_not(&conjunction);
    let (parsed, errors) = parser.parse_query_lenient(&repair_boosted_group_negation(&repaired));
    if !errors.is_empty() {
        tracing::debug!(
            error_count = errors.len(),
            "lenient query parse produced warnings"
        );
    }
    parsed
}

fn checkpoint(cx: &Cx) -> SearchResult<()> {
    if cx.checkpoint().is_err() || cx.is_cancel_requested() {
        return Err(SearchError::Cancelled {
            phase: "tantivy.retained_read".to_owned(),
            reason: "operation cancelled".to_owned(),
        });
    }
    Ok(())
}

fn invalid_metadata(reason: &'static str) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "tantivy.metadata",
        source: reason.into(),
    }
}

fn document_id(document: &TantivyDocument, fields: SchemaFields) -> SearchResult<&str> {
    document
        .get_first(fields.id)
        .and_then(|value| value.as_str())
        .filter(|id| !id.is_empty())
        .ok_or_else(|| invalid_metadata("selected Tantivy document has no valid stored ID"))
}

fn document_metadata(
    document: &TantivyDocument,
    fields: SchemaFields,
) -> SearchResult<Option<Arc<serde_json::Value>>> {
    let Some(value) = document.get_first(fields.metadata_json) else {
        return Ok(None);
    };
    let raw = value
        .as_str()
        .ok_or_else(|| invalid_metadata("selected Tantivy metadata is not a stored string"))?;
    serde_json::from_str(raw)
        .map(Arc::new)
        .map(Some)
        .map_err(|_| invalid_metadata("selected Tantivy metadata is not valid JSON"))
}

/// Holds a fixed segment set without retaining the writer or a reload surface.
#[derive(Clone)]
pub(super) struct PinnedTantivyReader {
    index: Index,
    fields: SchemaFields,
    searcher: Searcher,
}

impl PinnedTantivyReader {
    pub(super) fn new(cx: &Cx, index: &TantivyIndex) -> SearchResult<Self> {
        checkpoint(cx)?;
        let reader = Self {
            index: index.index.clone(),
            fields: index.fields,
            searcher: index.reader.searcher(),
        };
        checkpoint(cx)?;
        Ok(reader)
    }

    pub(super) fn selected_metadata(
        &self,
        cx: &Cx,
        document_ids: &[&str],
    ) -> SearchResult<Vec<Option<Arc<serde_json::Value>>>> {
        checkpoint(cx)?;
        if document_ids.is_empty() {
            return Ok(Vec::new());
        }
        let clauses = document_ids
            .iter()
            .map(|id| {
                let term = Term::from_field_text(self.fields.id, id);
                (
                    Occur::Should,
                    Box::new(TermQuery::new(term, IndexRecordOption::Basic)) as Box<dyn Query>,
                )
            })
            .collect();
        let query = BooleanQuery::new(clauses);
        let addresses = search_guarded(&self.searcher, &query, &DocSetCollector)?;
        checkpoint(cx)?;
        let mut metadata: HashMap<DocId, Option<Arc<serde_json::Value>>> = HashMap::new();
        for address in addresses {
            checkpoint(cx)?;
            let document = load_doc(&self.searcher, address)?;
            let id = document_id(&document, self.fields)?;
            let value = document_metadata(&document, self.fields)?;
            if metadata.insert(id.into(), value).is_some() {
                return Err(invalid_metadata(
                    "selected Tantivy ID has multiple live documents",
                ));
            }
        }
        checkpoint(cx)?;
        Ok(document_ids
            .iter()
            .map(|id| metadata.get(*id).cloned().flatten())
            .collect())
    }
}

impl LexicalRead for PinnedTantivyReader {
    fn pin_search<'a>(&'a self, cx: &'a Cx) -> SearchFuture<'a, Option<Arc<dyn LexicalRead>>> {
        Box::pin(async move {
            checkpoint(cx)?;
            Ok(Some(Arc::new(self.clone()) as Arc<dyn LexicalRead>))
        })
    }

    fn metadata_for_documents<'a>(
        &'a self,
        cx: &'a Cx,
        document_ids: &'a [&str],
    ) -> SearchFuture<'a, Vec<Option<Arc<serde_json::Value>>>> {
        Box::pin(async move { self.selected_metadata(cx, document_ids) })
    }

    fn search<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            checkpoint(cx)?;
            let query = TantivyIndex::truncate_query(query);
            if query.trim().is_empty() || limit == 0 {
                return Ok(Vec::new());
            }
            let parsed = parse_shipping_query(&self.index, self.fields, query);
            let hits = execute_top_k(&self.searcher, parsed.as_ref(), limit, 0)?;
            checkpoint(cx)?;
            let mut results = Vec::with_capacity(hits.len());
            for hit in hits {
                checkpoint(cx)?;
                let document = load_doc(&self.searcher, hit.doc_address)?;
                results.push(ScoredResult {
                    doc_id: document_id(&document, self.fields)?.into(),
                    score: hit.bm25_score,
                    source: ScoreSource::Lexical,
                    index: None,
                    fast_score: None,
                    quality_score: None,
                    lexical_score: Some(hit.bm25_score),
                    rerank_score: None,
                    explanation: None,
                    metadata: document_metadata(&document, self.fields)?,
                });
            }
            checkpoint(cx)?;
            Ok(results)
        })
    }

    fn doc_count(&self) -> SearchResult<usize> {
        TantivyIndex::checked_searcher_doc_count(&self.searcher)
    }
}
