//! Conjunctive source scoping over the query's retained native hybrid snapshot.
//! Membership is evaluated before retrieval and fusion, never on a displayed
//! top-k page. An owner-supplied server scope cannot be widened by a request.

use frankensearch::native_ann::NativeProgressiveSearch;
use frankensearch::native_ann::builder::scope::NativeScopedHybridIndex;
use std::collections::{BTreeMap, BTreeSet};

use super::{
    Cx, Deserialize, IndexableDocument, Mode, NativeBuiltHybridIndex, Result, SCHEMA, Serialize,
    bad, validate_query,
};

const MAX_FILTER_BYTES: usize = 64 * 1024;
const MAX_FILTER_IDS: usize = 4096;
const MAX_METADATA_FIELDS: usize = 64;

/// Case-sensitive exact metadata/ID comparisons and a literal ID prefix.
/// Every present field must match. `ids: []` deliberately selects no documents.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Filter {
    ids: Option<BTreeSet<String>>,
    id_prefix: Option<String>,
    #[serde(default)]
    metadata: BTreeMap<String, String>,
}

impl Filter {
    pub(super) fn parse(text: &str) -> Result<Self> {
        if text.len() > MAX_FILTER_BYTES {
            return Err(bad("filter JSON exceeds 64 KiB"));
        }
        let filter: Self = serde_json::from_str(text).map_err(|_| {
            bad("invalid filter: expected ids, id_prefix and/or string-valued metadata")
        })?;
        filter.validate()?;
        Ok(filter)
    }

    pub(super) fn validate(&self) -> Result<()> {
        if self
            .ids
            .as_ref()
            .is_some_and(|ids| ids.len() > MAX_FILTER_IDS)
            || self.metadata.len() > MAX_METADATA_FIELDS
        {
            return Err(bad("filter exceeds 4096 IDs or 64 metadata fields"));
        }
        let mut remaining = MAX_FILTER_BYTES;
        for id in self.ids.iter().flatten().chain(self.id_prefix.iter()) {
            if id.trim().is_empty() || id.len() > usize::from(u16::MAX) {
                return Err(bad(
                    "filter IDs and prefixes must be nonblank and at most 65535 bytes",
                ));
            }
            charge(id, &mut remaining)?;
        }
        for (key, value) in &self.metadata {
            if key.is_empty() || key.len() > 4096 || value.len() > 4096 {
                return Err(bad(
                    "metadata keys must be nonempty; keys and values are limited to 4096 bytes",
                ));
            }
            charge(key, &mut remaining)?;
            charge(value, &mut remaining)?;
        }
        Ok(())
    }

    pub(super) fn matches(&self, document: &IndexableDocument) -> bool {
        self.ids
            .as_ref()
            .is_none_or(|ids| ids.contains(&document.id))
            && self
                .id_prefix
                .as_ref()
                .is_none_or(|prefix| document.id.starts_with(prefix.as_str()))
            && self
                .metadata
                .iter()
                .all(|(key, value)| document.metadata.get(key) == Some(value))
    }
}

fn charge(text: &str, remaining: &mut usize) -> Result<()> {
    if text.contains('\0') || text.len() > *remaining {
        return Err(bad(
            "filter text must be NUL-free and at most 64 KiB in total",
        ));
    }
    *remaining -= text.len();
    Ok(())
}

/// The server's fixed scope and the caller's per-request scope are both required.
pub type Filters<'a> = [Option<&'a Filter>; 2];

pub struct Query<'a> {
    index: &'a NativeBuiltHybridIndex,
    scope: Option<NativeScopedHybridIndex<'a>>,
}

impl<'a> Query<'a> {
    pub(super) fn prepare(
        index: &'a NativeBuiltHybridIndex,
        cx: &Cx,
        filters: Filters<'_>,
    ) -> Result<Self> {
        cx.checkpoint()
            .map_err(|_| bad("native query scoping cancelled"))?;
        for filter in filters.iter().flatten() {
            filter.validate()?;
        }
        let scope = if filters.iter().any(Option::is_some) {
            Some(index.scope(cx, |document| {
                Ok(filters
                    .iter()
                    .flatten()
                    .all(|filter| filter.matches(document)))
            })?)
        } else {
            // Keep ordinary unscoped searches O(1) at this layer: no source scan.
            None
        };
        Ok(Self { index, scope })
    }

    pub(super) fn annotate(&self, payload: &mut serde_json::Value) {
        if let Some(scope) = &self.scope {
            payload["scope"] = serde_json::json!({
                "filtered": true, "eligible_documents": scope.len(),
            });
        }
    }

    pub(super) fn progressive<'q>(
        &'q self,
        cx: &'q Cx,
        query: &'q str,
        limit: usize,
    ) -> frankensearch::SearchResult<NativeProgressiveSearch<'q>> {
        self.scope.as_ref().map_or_else(
            || self.index.progressive(cx, query, limit),
            |scope| scope.progressive(cx, query, limit),
        )
    }

    pub(super) async fn search(
        &self,
        cx: &Cx,
        query: &str,
        mode: Mode,
        limit: usize,
    ) -> Result<serde_json::Value> {
        validate_query(query)?;
        let full = mode == Mode::Full && self.index.vectors().quality().is_some();
        let results = match (&self.scope, mode, full) {
            (Some(scope), Mode::Quality, _) => scope.search_quality(cx, query, limit).await?,
            (Some(scope), _, true) => scope.search_refined(cx, query, limit).await?,
            (Some(scope), _, false) => scope.search(cx, query, limit).await?,
            (None, Mode::Quality, _) => self.index.search_quality(cx, query, limit).await?,
            (None, _, true) => self.index.search_refined(cx, query, limit).await?,
            (None, _, false) => self.index.search(cx, query, limit).await?,
        };
        let phase = if mode == Mode::Quality {
            "quality"
        } else if full {
            "refined"
        } else {
            "initial"
        };
        let mut payload = serde_json::json!({
            "schema": SCHEMA, "ok": true, "event": "results", "phase": phase,
            "generation": self.index.vectors().fast().index().owner_witness().generation,
            "results": results,
        });
        self.annotate(&mut payload);
        Ok(payload)
    }

    pub(super) fn progressive_with_reranker<'q>(
        &'q self,
        cx: &'q Cx,
        text: &'q str,
        limit: usize,
        reranker: &'q dyn frankensearch::Reranker,
        window: usize,
    ) -> frankensearch::SearchResult<NativeProgressiveSearch<'q>> {
        self.scope.as_ref().map_or_else(
            || {
                self.index
                    .progressive_with_reranker(cx, text, limit, reranker, window)
            },
            |scope| scope.progressive_with_reranker(cx, text, limit, reranker, window),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_fields_are_conjunctive_case_sensitive_and_empty_ids_deny_all() {
        let mut doc = IndexableDocument::new("src/東京.rs", "retained content");
        doc.metadata.insert("tenant".to_owned(), "A".to_owned());
        assert!(Filter::parse(r"{}").unwrap().matches(&doc));
        assert!(
            Filter::parse(r#"{"id_prefix":"src/","metadata":{"tenant":"A"}}"#)
                .unwrap()
                .matches(&doc)
        );
        assert!(
            Filter::parse(r#"{"ids":["src/東京.rs"]}"#)
                .unwrap()
                .matches(&doc)
        );
        for text in [
            r#"{"ids":[]}"#,
            r#"{"id_prefix":"SRC/"}"#,
            r#"{"ids":["missing"],"metadata":{"tenant":"A"}}"#,
            r#"{"metadata":{"tenant":"a"}}"#,
            r#"{"metadata":{"unknown":""}}"#,
        ] {
            assert!(!Filter::parse(text).unwrap().matches(&doc), "{text}");
        }
    }

    #[test]
    fn malformed_or_oversized_filters_do_not_become_unrestricted() {
        for text in [
            "null",
            "[]",
            r#"{"unknown":true}"#,
            r#"{"ids":[""]}"#,
            r#"{"ids":["x\u0000y"]}"#,
            r#"{"id_prefix":""}"#,
            r#"{"metadata":{"tenant":42}}"#,
            r#"{"metadata":{"":"A"}}"#,
            r#"{"ids":[],"ids":["widen"]}"#,
        ] {
            assert!(Filter::parse(text).is_err(), "{text}");
        }
        assert!(Filter::parse(&" ".repeat(MAX_FILTER_BYTES + 1)).is_err());
        let too_many = (0..=MAX_FILTER_IDS)
            .map(|id| id.to_string())
            .collect::<BTreeSet<_>>();
        assert!(
            Filter {
                ids: Some(too_many),
                ..Filter::default()
            }
            .validate()
            .is_err()
        );
        let too_long = BTreeMap::from([("tenant".to_owned(), "a".repeat(4097))]);
        assert!(
            Filter {
                metadata: too_long,
                ..Filter::default()
            }
            .validate()
            .is_err()
        );
    }
}
