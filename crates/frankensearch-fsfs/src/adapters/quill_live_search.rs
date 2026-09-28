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
use frankensearch_core::{SearchError, SearchResult};
use frankensearch_quill::{
    BlueGreenEngine, LexicalLayout, QuillConfig, QuillSearchIndex, inspect_lexical_layout,
};
use serde_json::Value;

use super::live_search::{LiveSearchConfig, LiveSearchFrame, LiveSearchHit, LiveSearchTracker};
use super::live_search_session::{
    CommittedLiveSearchSnapshot, LiveSearchRefreshConfig, LiveSearchSession, LiveSearchSource,
};
use crate::generation_store::{CompleteGenerationStore, PublishedGeneration};

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
        quill.validate()?;
        let source = QuillSource {
            store,
            config: quill,
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
}

#[derive(Clone)]
struct AdmittedQuill {
    generation: PublishedGeneration,
    reader: Arc<QuillSearchIndex>,
}

struct QuillSource {
    store: CompleteGenerationStore,
    config: QuillConfig,
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
        let results = snapshot.reader.search_results(cx, query, limit)?;
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
}
