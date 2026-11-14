//! Native lexical live queries over fsfs immutable complete generations.
//!
//! This connects the refresh/delta engine to real read-only Quill readers. The
//! host supplies its `Cx` and poll cadence; there is no private runtime, writer
//! lease, watcher thread, or event queue. Every query and its hydrated metadata
//! come from one admitted immutable bundle. This is lexical search, not a claim
//! that a semantic model is available or that hybrid reranking was performed.

use std::fs;
use std::io::{ErrorKind, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use asupersync::Cx;
use frankensearch_core::{ParsedQuery, SearchError, SearchResult};
use frankensearch_quill::{
    BlueGreenEngine, LexicalLayout, QuillConfig, QuillSearchIndex, inspect_lexical_layout,
};
use serde_json::Value;

use super::live_search::{LiveSearchConfig, LiveSearchFrame, LiveSearchHit, LiveSearchTracker};
use super::live_search_session::{
    CommittedLiveSearchSnapshot, LiveSearchRefreshConfig, LiveSearchSession, LiveSearchSource,
};
use crate::generation_store::{CompleteGenerationStore, PublishedGeneration};
use crate::runtime::{FsfsRuntime, SearchFilterExpr};

/// Live BM25 result payloads contain stored metadata, when the document has it.
/// Stable document identity, score, and rank are carried by the surrounding hit.
pub type QuillLiveSearchFrame = LiveSearchFrame<Option<Value>>;

/// A usable native live-query session for a complete-generation store.
///
/// The first poll admits the selected bundle and emits a full result snapshot.
/// Subsequent polls admit replacements, debounce re-querying, and publish deltas.
/// Reads use `QuillSearchIndex`, never the writer-capable `QuillIndex`.
#[derive(Debug)]
pub struct QuillLiveSearchSession {
    session: LiveSearchSession<QuillSource>,
}

impl QuillLiveSearchSession {
    /// Bind the query and existing store without starting work or creating files.
    ///
    /// # Errors
    /// Rejects invalid Quill, result-window, or refresh configuration.
    pub fn new(
        store: CompleteGenerationStore,
        query: impl Into<String>,
        limits: LiveSearchConfig,
        refresh: LiveSearchRefreshConfig,
        quill: QuillConfig,
    ) -> SearchResult<Self> {
        Self::new_filtered(store, query, None, limits, refresh, quill)
    }

    /// Bind a fixed scope using the ordinary fsfs path/type/lang/ext grammar.
    ///
    /// The scope survives query edits and generation changes. Selective filters
    /// widen the lexical candidate prefix until the eligible window is full or
    /// the admitted corpus is exhausted. Only selected results are hydrated.
    /// Construction performs no I/O and creates no files.
    ///
    /// # Errors
    /// Rejects invalid filter, Quill, result-window or refresh configuration.
    pub fn new_filtered(
        store: CompleteGenerationStore,
        query: impl Into<String>,
        filter: Option<&str>,
        limits: LiveSearchConfig,
        refresh: LiveSearchRefreshConfig,
        quill: QuillConfig,
    ) -> SearchResult<Self> {
        let filter = SearchFilterExpr::parse(filter.unwrap_or_default())?;
        quill.validate()?;
        let source = QuillSource {
            store,
            config: quill,
            filter,
            admitted: None,
        };
        Ok(Self {
            session: LiveSearchSession::new(query, limits, refresh, source)?,
        })
    }

    /// Last delivered result window and generation, unchanged by failed refreshes.
    #[must_use]
    pub const fn tracker(&self) -> &LiveSearchTracker<Option<Value>> {
        self.session.tracker()
    }

    /// Query belonging to the current logical subscription.
    #[must_use]
    pub fn query(&self) -> &str {
        self.session.query()
    }

    /// Change the query without discarding the admitted read-only Quill reader.
    ///
    /// Identical queries are no-ops. A changed query resets the logical stream:
    /// its next successful poll emits a sequence-one snapshot, including an
    /// empty snapshot when nothing matches, even if no index was published.
    /// The host must reset its consumer before accepting that new stream. This
    /// operation does no I/O or indexing; the next poll still checks selection
    /// and admits any replacement through the ordinary complete-bundle path.
    ///
    /// # Errors
    /// Cancellation or a blank query leaves the current subscription unchanged.
    pub fn set_query(&mut self, cx: &Cx, query: impl Into<String>) -> SearchResult<bool> {
        self.session.set_query(cx, query)
    }

    /// Admit any replacement and deliver a native result update to the host.
    ///
    /// Admission awaits Quill through the caller's runtime. The underlying
    /// filesystem work and synchronous query belong on the host's blocking lane.
    /// Unchanged polls perform only a bounded selection probe, not a bundle rehash
    /// or an index reopen.
    ///
    /// # Errors
    /// Returns original admission, query, cancellation, and validation errors.
    /// No failure advances the subscriber's published result baseline.
    pub async fn poll(
        &mut self,
        cx: &Cx,
        now: Instant,
    ) -> SearchResult<Option<QuillLiveSearchFrame>> {
        self.session.source_mut().refresh(cx).await?;
        self.session.poll(cx, now)
    }

    /// Admit, search, serialize, and flush a native NDJSON update atomically with
    /// respect to the session baseline. Stop the transport on any output error.
    ///
    /// # Errors
    /// Returns admission/search errors or the original writer I/O error. A
    /// partial write may already be visible, but the baseline is not advanced.
    pub async fn poll_ndjson<W: Write + Send>(
        &mut self,
        cx: &Cx,
        now: Instant,
        writer: &mut W,
    ) -> SearchResult<Option<QuillLiveSearchFrame>> {
        self.session.source_mut().refresh(cx).await?;
        self.session.poll_ndjson(cx, now, writer)
    }

    /// Deliver only the immutable generation named by an owned publisher's
    /// durable receipt. A competing selection must not be relabeled as that
    /// publication, including when it changes while the reader is opening.
    ///
    /// The receipt is checked against the admitted reader before searching or
    /// emitting bytes. The synchronous query retains that reader even if the
    /// store selection subsequently changes. As with `poll_ndjson`, only a
    /// successfully flushed frame advances the delivered baseline.
    ///
    /// # Errors
    /// Returns admission/output errors, or rejects an absent or different
    /// admitted generation without emitting a frame or advancing the baseline.
    pub async fn poll_receipted_ndjson<W: Write + Send>(
        &mut self,
        cx: &Cx,
        now: Instant,
        expected: &PublishedGeneration,
        writer: &mut W,
    ) -> SearchResult<Option<QuillLiveSearchFrame>> {
        let source = self.session.source_mut();
        source.refresh(cx).await?;
        if source.admitted.as_ref().is_none_or(|admitted| {
            admitted.generation.id() != expected.id()
                || admitted.generation.manifest_sha256() != expected.manifest_sha256()
        }) {
            return Err(SearchError::InvalidConfig {
                field: "live_search.watch".to_owned(),
                value: expected.id().to_owned(),
                reason: "query admission selected a different publication; the watched generation remains published"
                    .to_owned(),
            });
        }
        self.session.poll_ndjson(cx, now, writer)
    }
}

#[derive(Clone)]
struct AdmittedQuill {
    generation: PublishedGeneration,
    reader: Arc<QuillSearchIndex>,
}

struct QuillSource {
    store: CompleteGenerationStore,
    config: QuillConfig,
    filter: Option<SearchFilterExpr>,
    admitted: Option<AdmittedQuill>,
}

impl QuillSource {
    async fn refresh(&mut self, cx: &Cx) -> SearchResult<()> {
        checkpoint(cx)?;
        if let Some(admitted) = &self.admitted
            && self.store.is_selected(cx, &admitted.generation)?
        {
            return Ok(());
        }
        let Some(generation) = self.store.active(cx)? else {
            self.admitted = None;
            return Ok(());
        };
        // `active` authenticates the complete inventory. Opening under that
        // retained path cannot mix artifacts from a concurrent pointer switch.
        let engine = lexical_engine(&generation)?;
        let reader = QuillSearchIndex::open(cx, &engine, self.config.clone()).await?;
        checkpoint(cx)?;
        // Do not replace even the admitted candidate until engine admission and
        // cancellation checks succeed. The subscriber baseline is separate.
        self.admitted = Some(AdmittedQuill {
            generation,
            reader: Arc::new(reader),
        });
        Ok(())
    }
}

impl LiveSearchSource for QuillSource {
    type Snapshot = AdmittedQuill;
    type Item = Option<Value>;

    fn snapshot(
        &mut self,
        cx: &Cx,
    ) -> SearchResult<Option<CommittedLiveSearchSnapshot<AdmittedQuill>>> {
        checkpoint(cx)?;
        Ok(self
            .admitted
            .clone()
            .map(|snapshot| CommittedLiveSearchSnapshot {
                generation: format!(
                    "{}@{}",
                    snapshot.generation.id(),
                    snapshot.generation.manifest_sha256(),
                ),
                snapshot,
            }))
    }

    fn search(
        &mut self,
        cx: &Cx,
        snapshot: &AdmittedQuill,
        query: &str,
        limit: usize,
    ) -> SearchResult<Vec<LiveSearchHit<Option<Value>>>> {
        // Bound raw input before the exclusion scanner allocates its scalar
        // buffer. Rendering can expand it further and is checked below.
        if query
            .chars()
            .nth(frankensearch_quill::query::MAX_QUERY_LENGTH)
            .is_some()
        {
            return Err(SearchError::InvalidConfig {
                field: "cli.search_query".to_owned(),
                value: "query_limit".to_owned(),
                reason: "search query exceeds the lexical parser's query length bound".to_owned(),
            });
        }
        let parsed = ParsedQuery::parse(query);
        if self.filter.is_some() || parsed.has_negations() {
            let live_docs = usize::try_from(snapshot.reader.doc_count()?).unwrap_or(usize::MAX);
            let excluded =
                FsfsRuntime::query_exclusions(cx, Some(&snapshot.reader), &parsed, live_docs)?;
            let candidates = FsfsRuntime::gather_lexical_candidates(
                cx,
                &snapshot.reader,
                query,
                limit,
                self.filter.as_ref(),
                excluded.as_deref(),
                live_docs,
            )?;
            let selected = candidates
                .into_iter()
                .filter(|hit| {
                    self.filter
                        .as_ref()
                        .is_none_or(|filter| filter.matches_doc_id(&hit.doc_id))
                        && excluded
                            .as_deref()
                            .is_none_or(|excluded| !excluded.contains(&hit.doc_id))
                })
                .take(limit)
                .collect::<Vec<_>>();
            let ids = selected
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<Vec<_>>();
            // This source never refreshes an admitted reader in place. Both
            // candidate gathering and hydration retain this exact bundle.
            let metadata = snapshot.reader.metadata_for_documents(cx, &ids)?;
            return Ok(selected
                .into_iter()
                .zip(metadata)
                .map(|(hit, metadata)| LiveSearchHit {
                    doc_id: hit.doc_id,
                    score: f64::from(hit.score),
                    item: metadata.as_deref().cloned(),
                })
                .collect());
        }
        let query = crate::lexical_pipeline::lexical_query_text(query);
        if query.chars().count() > frankensearch_quill::query::MAX_QUERY_LENGTH {
            return Err(SearchError::InvalidConfig {
                field: "cli.search_query".to_owned(),
                value: "lexical_query_limit".to_owned(),
                reason: "rendered search query exceeds the lexical parser's query length bound"
                    .to_owned(),
            });
        }
        let results = snapshot.reader.search_results(cx, &query, limit)?;
        Ok(results
            .into_iter()
            .map(|hit| LiveSearchHit {
                doc_id: hit.doc_id.into(),
                score: f64::from(hit.score),
                item: hit.metadata.as_deref().cloned(),
            })
            .collect())
    }
}

fn lexical_engine(generation: &PublishedGeneration) -> SearchResult<PathBuf> {
    // fsfs accepts both a direct legacy root and the nested lexical arm. An
    // existing broken nested arm must fail, never fall back to another root.
    let nested = generation.path().join("lexical");
    let root = match fs::symlink_metadata(&nested) {
        Ok(metadata) if metadata.is_dir() => nested,
        Err(error) if error.kind() == ErrorKind::NotFound => generation.path().to_path_buf(),
        Err(error) => return Err(error.into()),
        Ok(_) => {
            return Err(SearchError::IndexCorrupted {
                path: nested,
                detail: "lexical arm is not a directory".to_owned(),
            });
        }
    };
    let layout = inspect_lexical_layout(&root).map_err(|source| SearchError::SubsystemError {
        subsystem: "fsfs.live_search.lexical_layout",
        source: Box::new(source),
    })?;
    match layout {
        LexicalLayout::DirectQuill => Ok(root),
        LexicalLayout::BlueGreen { pointer, .. } if pointer.engine() == BlueGreenEngine::Quill => {
            Ok(pointer.engine_dir(&root))
        }
        layout => Err(SearchError::InvalidConfig {
            field: "live_search.lexical_layout".to_owned(),
            value: root.display().to_string(),
            reason: format!(
                "native live search requires Quill; selected layout is {}",
                layout.label()
            ),
        }),
    }
}

fn checkpoint(cx: &Cx) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: "fsfs.live_search.quill".to_owned(),
        reason: error.to_string(),
    })
}

#[cfg(all(test, unix))]
mod tests {
    use std::io;
    use std::time::Duration;

    use asupersync::test_utils::run_test_with_cx;
    use frankensearch_core::{IndexableDocument, LexicalWrite};
    use frankensearch_quill::QuillIndex;

    use super::*;
    use crate::adapters::live_search::{LiveSearchChange, LiveSearchEvent};
    use crate::generation_store::{COMPLETE_GENERATION_POINTER, GenerationPublication};

    fn config() -> QuillConfig {
        QuillConfig {
            deterministic_ingest: true,
            ..QuillConfig::default()
        }
    }

    async fn publish(
        cx: &Cx,
        store: &CompleteGenerationStore,
        documents: &[IndexableDocument],
    ) -> PublishedGeneration {
        publish_at(cx, store, documents, true).await
    }

    async fn publish_at(
        cx: &Cx,
        store: &CompleteGenerationStore,
        documents: &[IndexableDocument],
        nested: bool,
    ) -> PublishedGeneration {
        let build = store.begin(cx).unwrap();
        let directory = if nested {
            build.path().join("lexical")
        } else {
            build.path().to_path_buf()
        };
        let writer = QuillIndex::create(cx, &directory, config()).await.unwrap();
        LexicalWrite::index_documents(&writer, cx, documents)
            .await
            .unwrap();
        LexicalWrite::commit(&writer, cx).await.unwrap();
        drop(writer);
        let GenerationPublication::Durable(generation) = build.publish(cx, |_, _| Ok(())).unwrap()
        else {
            panic!("test publication did not reach durability");
        };
        generation
    }

    fn session(store: CompleteGenerationStore) -> QuillLiveSearchSession {
        QuillLiveSearchSession::new(
            store,
            "alpha",
            LiveSearchConfig::default(),
            LiveSearchRefreshConfig {
                debounce: Duration::ZERO,
                max_wait: Duration::from_millis(100),
            },
            config(),
        )
        .unwrap()
    }

    #[test]
    fn native_committed_search_emits_real_additions_removals_and_stored_metadata() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            let first = publish(
                &cx,
                &store,
                &[
                    IndexableDocument::new("a", "alpha alpha").with_metadata("path", "old.rs"),
                    IndexableDocument::new("irrelevant", "beta"),
                ],
            )
            .await;
            let mut session = session(store.clone());
            let frame = session.poll(&cx, Instant::now()).await.unwrap().unwrap();
            assert!(frame.generation.starts_with(first.id()));
            assert!(matches!(frame.event, LiveSearchEvent::Snapshot { .. }));
            assert_eq!(session.tracker().results().len(), 1);
            let hit = &session.tracker().results()[0].hit;
            assert_eq!(hit.doc_id, "a");
            assert_eq!(hit.item.as_ref().unwrap()["path"], "old.rs");

            let second = publish(
                &cx,
                &store,
                &[IndexableDocument::new("b", "alpha").with_metadata("path", "new.rs")],
            )
            .await;
            let frame = session.poll(&cx, Instant::now()).await.unwrap().unwrap();
            assert_eq!(frame.sequence, 2);
            assert!(frame.generation.starts_with(second.id()));
            let LiveSearchEvent::Delta { changes } = frame.event else {
                panic!("successor must emit a delta");
            };
            assert!(changes.iter().any(|change| matches!(
                change, LiveSearchChange::Removed { doc_id, .. } if doc_id == "a"
            )));
            assert!(changes.iter().any(|change| matches!(
                change, LiveSearchChange::Added { result } if result.hit.doc_id == "b"
            )));
            assert_eq!(session.tracker().results()[0].hit.doc_id, "b");
        });
    }

    #[test]
    fn scoped_search_fills_the_tail_and_keeps_scope_through_edits_and_publications() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            let mut documents = (0..96)
                .map(|index| {
                    IndexableDocument::new(format!("src/excluded-{index:03}.txt"), "alpha beta")
                        .with_title("alpha beta alpha beta")
                })
                .collect::<Vec<_>>();
            let body = format!("alpha beta {}", "padding ".repeat(64));
            documents.extend([
                IndexableDocument::new("src/parser.rs", body.clone())
                    .with_metadata("revision", "first")
                    .with_metadata("path", "stored-display-path"),
                IndexableDocument::new("src/parser.md", "alpha long reference notes")
                    .with_metadata("revision", "markdown"),
                IndexableDocument::new("elsewhere/parser.rs", "alpha beta")
                    .with_title("alpha beta"),
                IndexableDocument::new("src/unrelated.rs", "alpha beta").with_title("alpha beta"),
                IndexableDocument::new("src/parser.py", "alpha beta").with_title("alpha beta"),
            ]);
            let first = publish(&cx, &store, &documents).await;
            let raw_filter = "path:SRC/ parser lang:rust ext:md";
            let mut session = QuillLiveSearchSession::new_filtered(
                store.clone(),
                "alpha",
                Some(raw_filter),
                LiveSearchConfig {
                    max_results: 2,
                    ..LiveSearchConfig::default()
                },
                LiveSearchRefreshConfig {
                    debounce: Duration::ZERO,
                    max_wait: Duration::from_millis(100),
                },
                config(),
            )
            .unwrap();
            let frame = session.poll(&cx, Instant::now()).await.unwrap().unwrap();
            assert!(frame.generation.starts_with(first.id()));
            let results = session.tracker().results();
            assert_eq!(results.len(), 2, "eligible tail must fill the window");
            let rust = results
                .iter()
                .find(|hit| hit.hit.doc_id == "src/parser.rs")
                .unwrap();
            assert_eq!(rust.hit.item.as_ref().unwrap()["revision"], "first");
            assert_eq!(
                rust.hit.item.as_ref().unwrap()["path"],
                "stored-display-path"
            );
            assert!(results.iter().any(|hit| hit.hit.doc_id == "src/parser.md"));

            let reader = Arc::clone(
                &session
                    .session
                    .source_mut()
                    .admitted
                    .as_ref()
                    .unwrap()
                    .reader,
            );
            let filter = SearchFilterExpr::parse(raw_filter).unwrap().unwrap();
            let prefix = reader.search_doc_ids(&cx, "alpha", 64).unwrap();
            assert_eq!(prefix.len(), 64);
            assert!(
                prefix
                    .iter()
                    .all(|hit| !filter.matches_doc_id(&hit.document_id)),
                "fixture must place eligible winners beyond a fixed oversampling prefix"
            );

            assert!(session.set_query(&cx, "beta").unwrap());
            let edited = session.poll(&cx, Instant::now()).await.unwrap().unwrap();
            assert_eq!(edited.sequence, 1);
            assert_eq!(edited.previous_generation, None);
            assert_eq!(session.tracker().results().len(), 1);
            assert_eq!(session.tracker().results()[0].hit.doc_id, "src/parser.rs");
            assert!(Arc::ptr_eq(
                &reader,
                &session
                    .session
                    .source_mut()
                    .admitted
                    .as_ref()
                    .unwrap()
                    .reader,
            ));

            let second = publish(
                &cx,
                &store,
                &[
                    IndexableDocument::new("src/parser.rs", body)
                        .with_metadata("revision", "second"),
                    IndexableDocument::new("src/parser-new.md", "beta")
                        .with_metadata("revision", "new"),
                    IndexableDocument::new("src/parser.py", "beta beta").with_title("beta beta"),
                ],
            )
            .await;
            let updated = session.poll(&cx, Instant::now()).await.unwrap().unwrap();
            assert_eq!(updated.sequence, 2);
            assert!(updated.generation.starts_with(second.id()));
            assert_eq!(
                updated.previous_generation.as_deref(),
                Some(edited.generation.as_str())
            );
            assert!(matches!(updated.event, LiveSearchEvent::Delta { .. }));
            assert_eq!(session.tracker().results().len(), 2);
            assert!(
                session
                    .tracker()
                    .results()
                    .iter()
                    .all(|hit| filter.matches_doc_id(&hit.hit.doc_id))
            );
            let rust = session
                .tracker()
                .results()
                .iter()
                .find(|hit| hit.hit.doc_id == "src/parser.rs")
                .unwrap();
            assert_eq!(rust.hit.item.as_ref().unwrap()["revision"], "second");

            publish(
                &cx,
                &store,
                &[IndexableDocument::new("src/parser.py", "beta")],
            )
            .await;
            let empty = session.poll(&cx, Instant::now()).await.unwrap().unwrap();
            assert_eq!(empty.sequence, 3);
            assert!(session.tracker().results().is_empty());
            let LiveSearchEvent::Delta { changes } = empty.event else {
                panic!("scope becoming empty must remove the delivered eligible window");
            };
            assert_eq!(changes.len(), 2);
            assert!(
                changes
                    .iter()
                    .all(|change| matches!(change, LiveSearchChange::Removed { .. }))
            );
        });
    }

    #[test]
    fn global_exclusions_fill_the_live_lexical_tail_with_and_without_a_scope() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            let mut documents = (0..96)
                .map(|index| {
                    IndexableDocument::new(
                        format!("src/blocked-{index:03}.rs"),
                        "alpha blocked AND OR NOT internal secret",
                    )
                    .with_title("alpha alpha")
                })
                .collect::<Vec<_>>();
            for id in [
                "src/allowed-a.rs",
                "src/allowed-b.rs",
                "docs/allowed-c.md",
                "docs/allowed-d.md",
            ] {
                documents.push(
                    IndexableDocument::new(id, format!("alpha {}", "padding ".repeat(64)))
                        .with_metadata("source", id),
                );
            }
            let generation = publish(&cx, &store, &documents).await;
            let open = |query: &str, filter: Option<&str>| {
                QuillLiveSearchSession::new_filtered(
                    store.clone(),
                    query,
                    filter,
                    LiveSearchConfig {
                        max_results: 2,
                        ..LiveSearchConfig::default()
                    },
                    LiveSearchRefreshConfig {
                        debounce: Duration::ZERO,
                        max_wait: Duration::from_millis(100),
                    },
                    config(),
                )
                .unwrap()
            };
            let mut positive = open("alpha", None);
            positive.poll(&cx, Instant::now()).await.unwrap().unwrap();
            let reader = &positive
                .session
                .source_mut()
                .admitted
                .as_ref()
                .unwrap()
                .reader;
            let prefix = reader.search_doc_ids(&cx, "alpha", 64).unwrap();
            assert_eq!(prefix.len(), 64);
            assert!(
                prefix
                    .iter()
                    .all(|hit| hit.document_id.contains("blocked-"))
            );
            for filter in [None, Some("path:src/ ext:rs")] {
                for exclusion in [
                    "-blocked",
                    "-AND",
                    "-OR",
                    "-NOT",
                    r#"NOT "internal secret""#,
                ] {
                    for query in [
                        format!("alpha OR missing AND {exclusion}"),
                        format!("{exclusion} -absent"),
                    ] {
                        let mut live = open(&query, filter);
                        let frame = live.poll(&cx, Instant::now()).await.unwrap().unwrap();
                        assert!(frame.generation.starts_with(generation.id()));
                        let results = live.tracker().results();
                        assert_eq!(results.len(), 2, "{query}, filter={filter:?}");
                        for hit in results {
                            assert!(hit.hit.doc_id.contains("allowed-"), "{query}");
                            if filter.is_some() {
                                assert!(hit.hit.doc_id.starts_with("src/"), "{query}");
                            }
                            assert_eq!(
                                hit.hit.item.as_ref().unwrap()["source"],
                                hit.hit.doc_id,
                                "{query}",
                            );
                        }
                    }
                }
            }
        });
    }

    #[test]
    fn quoted_field_exclusions_keep_nonadjacent_and_other_field_matches() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            let documents = [
                IndexableDocument::new("src/exact.rs", "needle").with_title("internal secret"),
                IndexableDocument::new("src/nonadjacent.rs", "needle")
                    .with_title("internal unrelated secret"),
                IndexableDocument::new("src/content.rs", "needle internal secret")
                    .with_title("public"),
                IndexableDocument::new("src/partial.rs", "needle").with_title("internal"),
                IndexableDocument::new("docs/plain.md", "needle").with_title("public"),
            ];
            publish(&cx, &store, &documents).await;
            for filter in [None, Some("path:src/ ext:rs")] {
                for exclusion in [
                    r#"-title:"internal secret""#,
                    "-title:'internal secret'",
                    r#"-title : "internal secret""#,
                    r#"-title: "internal secret""#,
                ] {
                    for positive in ["needle", "needle OR absent AND"] {
                        let query = format!("{positive} {exclusion} -absent");
                        let mut live = QuillLiveSearchSession::new_filtered(
                            store.clone(),
                            &query,
                            filter,
                            LiveSearchConfig {
                                max_results: documents.len(),
                                ..LiveSearchConfig::default()
                            },
                            LiveSearchRefreshConfig {
                                debounce: Duration::ZERO,
                                max_wait: Duration::from_millis(100),
                            },
                            config(),
                        )
                        .unwrap();
                        live.poll(&cx, Instant::now()).await.unwrap().unwrap();
                        let mut actual = live
                            .tracker()
                            .results()
                            .iter()
                            .map(|hit| hit.hit.doc_id.as_str())
                            .collect::<Vec<_>>();
                        actual.sort_unstable();
                        let mut expected =
                            vec!["src/content.rs", "src/nonadjacent.rs", "src/partial.rs"];
                        if filter.is_none() {
                            expected.insert(0, "docs/plain.md");
                        }
                        assert_eq!(actual, expected, "{query}, filter={filter:?}");
                    }
                }
            }
        });
    }

    #[test]
    fn live_query_rendering_refuses_overflow_and_preserves_query_recovery() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            publish(
                &cx,
                &store,
                &[
                    IndexableDocument::new("blocked", "alpha AND"),
                    IndexableDocument::new("allowed", "alpha"),
                ],
            )
            .await;
            for (query, expected, fits_raw_bound) in [
                (
                    format!("alpha {}", "-AND ".repeat(1_998)),
                    "exclusion_query_limit",
                    true,
                ),
                (
                    format!("-absent {}alpha", "x:: ".repeat(2_000)),
                    "lexical_query_limit",
                    true,
                ),
                (
                    "a".repeat(frankensearch_quill::query::MAX_QUERY_LENGTH + 1),
                    "query_limit",
                    false,
                ),
                (
                    format!("{}alpha", "x:: ".repeat(2_000)),
                    "lexical_query_limit",
                    true,
                ),
            ] {
                assert_eq!(
                    query.chars().count() <= frankensearch_quill::query::MAX_QUERY_LENGTH,
                    fits_raw_bound,
                );
                let mut live = QuillLiveSearchSession::new(
                    store.clone(),
                    query,
                    LiveSearchConfig::default(),
                    LiveSearchRefreshConfig {
                        debounce: Duration::ZERO,
                        max_wait: Duration::from_millis(100),
                    },
                    config(),
                )
                .unwrap();
                let error = live.poll(&cx, Instant::now()).await.unwrap_err();
                assert!(
                    matches!(&error, SearchError::InvalidConfig { value, .. } if value == expected),
                    "{error:?}",
                );
                assert_eq!(live.tracker().sequence(), 0);
                assert!(live.tracker().generation().is_none());
                assert!(live.tracker().results().is_empty());
                assert!(live.set_query(&cx, "alpha -AND").unwrap());
                live.poll(&cx, Instant::now()).await.unwrap().unwrap();
                assert_eq!(live.tracker().results().len(), 1);
                assert_eq!(live.tracker().results()[0].hit.doc_id, "allowed");
            }
            let unicode_query = format!(
                "alpha OR {}",
                "é".repeat(frankensearch_quill::query::MAX_QUERY_LENGTH / 2 + 1),
            );
            assert!(unicode_query.len() > frankensearch_quill::query::MAX_QUERY_LENGTH);
            assert!(unicode_query.chars().count() < frankensearch_quill::query::MAX_QUERY_LENGTH);
            let mut unicode = session(store.clone());
            assert!(unicode.set_query(&cx, unicode_query).unwrap());
            unicode.poll(&cx, Instant::now()).await.unwrap().unwrap();
            assert_eq!(unicode.tracker().results().len(), 2);
        });
    }

    #[test]
    fn invalid_scope_is_rejected_before_admitting_or_creating_a_generation() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            assert!(matches!(
                QuillLiveSearchSession::new_filtered(
                    store,
                    "alpha",
                    Some("owner:unknown"),
                    LiveSearchConfig::default(),
                    LiveSearchRefreshConfig::default(),
                    config(),
                ),
                Err(SearchError::InvalidConfig { field, .. }) if field == "cli.search.filter"
            ));
            assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        });
    }

    #[test]
    fn native_unchanged_poll_reuses_the_exact_read_only_reader() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            publish(&cx, &store, &[IndexableDocument::new("a", "alpha")]).await;
            let mut session = session(store);
            session.poll(&cx, Instant::now()).await.unwrap();
            let reader = Arc::clone(
                &session
                    .session
                    .source_mut()
                    .admitted
                    .as_ref()
                    .unwrap()
                    .reader,
            );
            assert!(session.poll(&cx, Instant::now()).await.unwrap().is_none());
            assert!(Arc::ptr_eq(
                &reader,
                &session
                    .session
                    .source_mut()
                    .admitted
                    .as_ref()
                    .unwrap()
                    .reader,
            ));
        });
    }

    #[test]
    fn native_direct_bundle_layout_is_admitted_without_a_nested_lexical_directory() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            publish_at(
                &cx,
                &store,
                &[IndexableDocument::new("direct", "alpha")],
                false,
            )
            .await;
            let mut session = session(store);
            let frame = session.poll(&cx, Instant::now()).await.unwrap().unwrap();
            assert_eq!(frame.result_count, 1);
            assert_eq!(session.tracker().results()[0].hit.doc_id, "direct");
        });
    }

    #[test]
    fn native_metadata_only_change_is_visible_with_unchanged_identity_and_text() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            publish(
                &cx,
                &store,
                &[IndexableDocument::new("a", "alpha").with_metadata("path", "before.rs")],
            )
            .await;
            let mut session = session(store.clone());
            session.poll(&cx, Instant::now()).await.unwrap();
            publish(
                &cx,
                &store,
                &[IndexableDocument::new("a", "alpha").with_metadata("path", "after.rs")],
            )
            .await;
            let frame = session.poll(&cx, Instant::now()).await.unwrap().unwrap();
            let LiveSearchEvent::Delta { changes } = frame.event else {
                panic!("expected a metadata delta");
            };
            assert!(matches!(
                &changes[..], [LiveSearchChange::Updated { result, .. }]
                    if result.hit.item.as_ref().unwrap()["path"] == "after.rs"
            ));
        });
    }

    #[test]
    fn native_pending_queries_retain_the_previous_immutable_reader() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            publish(&cx, &store, &[IndexableDocument::new("old", "alpha")]).await;
            let mut source = QuillSource {
                store: store.clone(),
                config: config(),
                filter: None,
                admitted: None,
            };
            source.refresh(&cx).await.unwrap();
            let pinned = source.snapshot(&cx).unwrap().unwrap();
            publish(&cx, &store, &[IndexableDocument::new("new", "alpha")]).await;
            source.refresh(&cx).await.unwrap();
            let hits = source.search(&cx, &pinned.snapshot, "alpha", 20).unwrap();
            assert_eq!(hits[0].doc_id, "old");
            let latest = source.snapshot(&cx).unwrap().unwrap();
            let hits = source.search(&cx, &latest.snapshot, "alpha", 20).unwrap();
            assert_eq!(hits[0].doc_id, "new");
        });
    }

    #[test]
    fn native_corrupt_selection_is_an_error_not_a_deletion_or_fallback() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            publish(&cx, &store, &[IndexableDocument::new("a", "alpha")]).await;
            let mut session = session(store.clone());
            session.poll(&cx, Instant::now()).await.unwrap();
            fs::write(store.root().join(COMPLETE_GENERATION_POINTER), b"corrupt").unwrap();
            assert!(session.poll(&cx, Instant::now()).await.is_err());
            assert_eq!(session.tracker().sequence(), 1);
            assert_eq!(session.tracker().results()[0].hit.doc_id, "a");
        });
    }

    #[test]
    fn native_nonmatching_successor_removes_every_previously_visible_hit() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            publish(&cx, &store, &[IndexableDocument::new("a", "alpha")]).await;
            let mut session = session(store.clone());
            session.poll(&cx, Instant::now()).await.unwrap();
            publish(&cx, &store, &[IndexableDocument::new("a", "beta")]).await;
            let frame = session.poll(&cx, Instant::now()).await.unwrap().unwrap();
            assert_eq!(frame.result_count, 0);
            let LiveSearchEvent::Delta { changes } = frame.event else {
                panic!("expected removals");
            };
            assert!(matches!(
                &changes[..], [LiveSearchChange::Removed { doc_id, .. }] if doc_id == "a"
            ));
        });
    }

    struct BrokenTransport;

    impl Write for BrokenTransport {
        fn write(&mut self, _bytes: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn native_ndjson_failures_preserve_state_and_successful_frames_round_trip() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            publish(&cx, &store, &[IndexableDocument::new("a", "alpha")]).await;
            let mut session = session(store);
            assert!(matches!(
                session
                    .poll_ndjson(&cx, Instant::now(), &mut BrokenTransport)
                    .await,
                Err(SearchError::Io(_))
            ));
            assert_eq!(session.tracker().sequence(), 0);
            let mut output = Vec::new();
            let frame = session
                .poll_ndjson(&cx, Instant::now(), &mut output)
                .await
                .unwrap()
                .unwrap();
            let decoded: QuillLiveSearchFrame = serde_json::from_slice(&output).unwrap();
            assert_eq!(frame, decoded);
            assert_eq!(session.tracker().sequence(), 1);
        });
    }

    #[test]
    fn native_receipted_poll_fences_reloads_before_output_and_preserves_failed_baselines() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            let first = publish(&cx, &store, &[IndexableDocument::new("a", "alpha")]).await;
            let mut session = session(store.clone());
            let mut initial_output = Vec::new();
            let initial = session
                .poll_receipted_ndjson(&cx, Instant::now(), &first, &mut initial_output)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(
                initial.generation,
                format!("{}@{}", first.id(), first.manifest_sha256()),
            );
            assert!(matches!(initial.event, LiveSearchEvent::Snapshot { .. }));

            let second = publish(&cx, &store, &[IndexableDocument::new("b", "alpha")]).await;
            let mut foreign_output = Vec::new();
            let error = session
                .poll_receipted_ndjson(&cx, Instant::now(), &first, &mut foreign_output)
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                if field == "live_search.watch"));
            assert!(foreign_output.is_empty());
            assert_eq!(session.tracker().sequence(), 1);
            assert_eq!(session.tracker().results()[0].hit.doc_id, "a");
            assert_eq!(store.active(&cx).unwrap(), Some(second.clone()));

            assert!(matches!(
                session
                    .poll_receipted_ndjson(&cx, Instant::now(), &second, &mut BrokenTransport)
                    .await,
                Err(SearchError::Io(_))
            ));
            assert_eq!(session.tracker().sequence(), 1);
            assert_eq!(session.tracker().results()[0].hit.doc_id, "a");

            let mut output = Vec::new();
            let successor = session
                .poll_receipted_ndjson(&cx, Instant::now(), &second, &mut output)
                .await
                .unwrap()
                .unwrap();
            let decoded: QuillLiveSearchFrame = serde_json::from_slice(&output).unwrap();
            assert_eq!(successor, decoded);
            assert_eq!(successor.sequence, 2);
            assert_eq!(successor.previous_generation, Some(initial.generation));
            assert_eq!(
                successor.generation,
                format!("{}@{}", second.id(), second.manifest_sha256()),
            );
            let LiveSearchEvent::Delta { changes } = successor.event else {
                panic!("a receipted successor must produce a delta");
            };
            assert!(changes.iter().any(|change| matches!(
                change, LiveSearchChange::Removed { doc_id, .. } if doc_id == "a"
            )));
            assert!(changes.iter().any(|change| matches!(
                change, LiveSearchChange::Added { result } if result.hit.doc_id == "b"
            )));
        });
    }

    #[test]
    fn native_query_changes_reuse_the_reader_and_emit_query_specific_snapshots() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            let generation = publish(
                &cx,
                &store,
                &[
                    IndexableDocument::new("a", "alpha").with_metadata("path", "alpha.rs"),
                    IndexableDocument::new("b", "beta").with_metadata("path", "beta.rs"),
                ],
            )
            .await;
            let pointer = fs::read(store.root().join(COMPLETE_GENERATION_POINTER)).unwrap();
            let mut session = session(store.clone());
            session.poll(&cx, Instant::now()).await.unwrap();
            let reader = Arc::clone(
                &session
                    .session
                    .source_mut()
                    .admitted
                    .as_ref()
                    .unwrap()
                    .reader,
            );
            assert_eq!(session.tracker().results()[0].hit.doc_id, "a");
            assert!(session.set_query(&cx, "beta").unwrap());
            assert_eq!(session.query(), "beta");
            let mut output = Vec::new();
            let frame = session
                .poll_ndjson(&cx, Instant::now(), &mut output)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(frame.query, "beta");
            assert_eq!(frame.sequence, 1);
            assert!(frame.previous_generation.is_none());
            assert!(frame.generation.starts_with(generation.id()));
            assert!(matches!(frame.event, LiveSearchEvent::Snapshot { .. }));
            assert_eq!(session.tracker().results()[0].hit.doc_id, "b");
            assert_eq!(
                session.tracker().results()[0].hit.item.as_ref().unwrap()["path"],
                "beta.rs"
            );
            let decoded: QuillLiveSearchFrame = serde_json::from_slice(&output).unwrap();
            assert_eq!(decoded.query, "beta");
            assert!(Arc::ptr_eq(
                &reader,
                &session
                    .session
                    .source_mut()
                    .admitted
                    .as_ref()
                    .unwrap()
                    .reader,
            ));
            session.set_query(&cx, "unmatchedword").unwrap();
            let empty = session.poll(&cx, Instant::now()).await.unwrap().unwrap();
            assert_eq!(empty.sequence, 1);
            assert_eq!(empty.result_count, 0);
            assert!(matches!(empty.event,
                LiveSearchEvent::Snapshot { results } if results.is_empty()));
            assert_eq!(
                fs::read(store.root().join(COMPLETE_GENERATION_POINTER)).unwrap(),
                pointer
            );
        });
    }

    #[test]
    fn native_retarget_rejection_keeps_the_original_subscription() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            publish(&cx, &store, &[IndexableDocument::new("a", "alpha")]).await;
            let mut session = session(store);
            session.poll(&cx, Instant::now()).await.unwrap();
            assert!(!session.set_query(&cx, "alpha").unwrap());
            assert!(session.set_query(&cx, "  ").is_err());
            cx.set_cancel_requested(true);
            let cancelled = session.set_query(&cx, "beta");
            cx.set_cancel_requested(false);
            assert!(matches!(cancelled, Err(SearchError::Cancelled { .. })));
            assert_eq!(session.query(), "alpha");
            assert_eq!(session.tracker().sequence(), 1);
            assert_eq!(session.tracker().results()[0].hit.doc_id, "a");
            assert!(session.poll(&cx, Instant::now()).await.unwrap().is_none());
        });
    }

    #[test]
    fn native_retarget_admits_a_successor_instead_of_searching_the_old_bundle() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            publish(&cx, &store, &[IndexableDocument::new("old", "alpha")]).await;
            let mut session = session(store.clone());
            session.poll(&cx, Instant::now()).await.unwrap();
            session.set_query(&cx, "beta").unwrap();
            let successor = publish(&cx, &store, &[IndexableDocument::new("new", "beta")]).await;
            let frame = session.poll(&cx, Instant::now()).await.unwrap().unwrap();
            assert_eq!(frame.query, "beta");
            assert_eq!(frame.sequence, 1);
            assert!(frame.generation.starts_with(successor.id()));
            assert_eq!(session.tracker().results()[0].hit.doc_id, "new");
        });
    }
}
