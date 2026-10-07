//! Publication-driven indexing and live queries in one owned command future.
//!
//! The existing complete-generation watcher owns discovery, source fencing,
//! reuse, publication and native registration. We poll that future only while
//! waiting for a receipt or an interactive query. During query/refinement/output
//! it is suspended, but its native callback keeps coalescing bounded change hints.
//! No second indexer, background task, channel backlog or runtime is introduced.

use std::future::{Future, poll_fn};
use std::io::Write;
use std::pin::{Pin, pin};
use std::sync::Mutex;
use std::task::Poll;
use std::time::{Duration, Instant};

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use frankensearch_fsfs::adapters::live_search_session::LiveSearchRefreshConfig;
use frankensearch_fsfs::adapters::retained_live_search::{
    RetainedLiveSearchFrame, RetainedLiveSearchSession,
};
use frankensearch_fsfs::generation_store::{CompleteGenerationStore, PublishedGeneration};
use frankensearch_fsfs::{CliCommand, CliInput, FsfsRuntime, OutputFormat};

use super::terminal::QueryControl;
use super::{Budget, GuardedOutput, Options, hybrid};

/// The watcher invokes its receipt sink synchronously, then yields to its timer.
/// A full slot is an error, never permission to overwrite an undelivered receipt.
/// No lock is held while polling a future or running user/index code.
#[derive(Default)]
struct ReceiptSlot(Mutex<Option<PublishedGeneration>>);

impl ReceiptSlot {
    fn offer(&self, generation: &PublishedGeneration) -> SearchResult<()> {
        let mut slot = self
            .0
            .lock()
            .map_err(|_| watch_error("receipt slot poisoned"))?;
        if slot.is_some() {
            return Err(watch_error("watcher produced a second undelivered receipt"));
        }
        *slot = Some(generation.clone());
        drop(slot);
        Ok(())
    }

    fn take(&self) -> SearchResult<Option<PublishedGeneration>> {
        Ok(self
            .0
            .lock()
            .map_err(|_| watch_error("receipt slot poisoned"))?
            .take())
    }
}

#[derive(Debug)]
enum Operation {
    Publication(PublishedGeneration),
    Query(String),
}

#[cfg(test)]
async fn next_publication<F>(
    cx: &Cx,
    budget: &Budget,
    watcher: Pin<&mut F>,
    receipts: &ReceiptSlot,
) -> SearchResult<PublishedGeneration>
where
    F: Future<Output = SearchResult<()>> + Send,
{
    match next_operation(cx, budget, watcher, receipts, &mut None, false).await? {
        Operation::Publication(generation) => Ok(generation),
        Operation::Query(_) => Err(watch_error("query reached a fixed-query subscription")),
    }
}

async fn next_operation<F>(
    cx: &Cx,
    budget: &Budget,
    mut watcher: Pin<&mut F>,
    receipts: &ReceiptSlot,
    control: &mut Option<&mut dyn QueryControl>,
    can_query: bool,
) -> SearchResult<Operation>
where
    F: Future<Output = SearchResult<()>> + Send,
{
    poll_fn(|task| {
        budget.check(cx)?;
        // A receipt already made visible takes precedence over queries against
        // the previous generation. Its query can use the latest submitted text.
        if let Some(generation) = receipts.take()? {
            return Poll::Ready(Ok(Operation::Publication(generation)));
        }
        if can_query
            && let Some(control) = control.as_deref_mut()
            && let Some(query) = control.take_query()?
        {
            return Poll::Ready(Ok(Operation::Query(query)));
        }
        match watcher.as_mut().poll(task) {
            Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
            Poll::Ready(Ok(())) => Poll::Ready(Err(watch_error(
                "watcher ended without a publication; no empty result is synthesized",
            ))),
            Poll::Pending => receipts.take()?.map_or_else(
                || Poll::Pending,
                |generation| Poll::Ready(Ok(Operation::Publication(generation))),
            ),
        }
    })
    .await
}

fn watch_error(reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "live_search.watch".to_owned(),
        value: String::new(),
        reason: reason.to_owned(),
    }
}

/// Never emit a different publisher's generation under this watcher's receipt.
/// A pointer switch during admission can select a different retained reader;
/// reject its first phase BEFORE any bytes, rather than relabeling its results.
fn emit_receipted_frame<W: Write>(
    expected: &PublishedGeneration,
    frame: &RetainedLiveSearchFrame,
    writer: &mut W,
) -> SearchResult<()> {
    if frame.generation_id != expected.id() || frame.manifest_sha256 != expected.manifest_sha256() {
        return Err(watch_error(
            "query admission selected a different publication; the watched generation remains published",
        ));
    }
    hybrid::emit_frame(frame, writer)
}

fn publication_runtime(runtime: &FsfsRuntime, options: &Options) -> SearchResult<FsfsRuntime> {
    let source = options
        .watch_source
        .as_ref()
        .ok_or_else(|| watch_error("an explicit --watch-source is required before indexing"))?;
    if !options.hybrid || !runtime.config().indexing.offline {
        return Err(watch_error(
            "source watching requires the explicit offline hybrid runtime",
        ));
    }
    if options.max_updates.is_none_or(|maximum| maximum == 0) {
        return Err(watch_error(
            "a finite positive update limit is required; generations are retained",
        ));
    }
    if runtime.config().search.shadow_mode {
        return Err(watch_error(
            "shadow mode cannot write observations into sealed watch generations",
        ));
    }
    // Preserve the configured discovery/privacy/pressure/model policies and
    // shared model slots. Only this explicit branch authorizes source indexing.
    Ok(runtime.clone().with_cli_input(CliInput {
        command: CliCommand::Index,
        target_path: Some(source.clone()),
        index_dir: Some(options.root.clone()),
        format: OutputFormat::Jsonl,
        quiet: true,
        no_color: true,
        ..CliInput::default()
    }))
}

pub(super) async fn execute<W: Write + Send>(
    cx: &Cx,
    budget: &Budget,
    options: &Options,
    writer: &mut W,
    runtime: Option<FsfsRuntime>,
) -> SearchResult<u64> {
    execute_controlled(cx, budget, options, writer, runtime, None).await
}

fn update_query(
    cx: &Cx,
    subscriber: &mut RetainedLiveSearchSession,
    mut requested: Option<String>,
    control: &mut Option<&mut dyn QueryControl>,
) -> SearchResult<bool> {
    if requested.is_none()
        && let Some(control) = control.as_deref_mut()
    {
        requested = control.take_query()?;
    }
    let Some(query) = requested else {
        return Ok(false);
    };
    let changed = subscriber.set_query(cx, &query)?;
    if changed
        && let Some(control) = control.as_deref_mut()
    {
        control.begin_query(&query)?;
    }
    Ok(changed)
}

/// Interactive queries reuse the one owned watcher and retained subscriber.
/// They do not create a generation, advance its publication limit, reset its
/// command timeout, or abandon an in-progress publisher/query future.
pub(super) async fn execute_controlled<W: Write + Send>(
    cx: &Cx,
    budget: &Budget,
    options: &Options,
    writer: &mut W,
    runtime: Option<FsfsRuntime>,
    mut control: Option<&mut dyn QueryControl>,
) -> SearchResult<u64> {
    budget.check(cx)?;
    let runtime = runtime.ok_or_else(|| watch_error("hybrid runtime was not initialized"))?;
    let publisher = publication_runtime(&runtime, options)?;
    let receipts = ReceiptSlot::default();
    // Registering, scanning and creating a store are done by this existing
    // watcher on first poll, including its disjoint-tree and privacy checks.
    let mut watcher = pin!(publisher.watch_retained_generations(
        cx,
        &options.root,
        |generation| receipts.offer(generation),
    ));
    let mut store: Option<CompleteGenerationStore> = None;
    let mut subscriber: Option<RetainedLiveSearchSession> = None;
    let mut current: Option<PublishedGeneration> = None;
    let mut delivered = 0_u64;
    loop {
        let operation = next_operation(
            cx, budget, watcher.as_mut(), &receipts, &mut control, current.is_some(),
        )
        .await?;
        let (generation, publication, requested) = match operation {
            Operation::Publication(generation) => {
                current = Some(generation.clone());
                (generation, true, None)
            }
            Operation::Query(query) => (
                current.clone().ok_or_else(|| watch_error("no published query generation"))?,
                false,
                Some(query),
            ),
        };
        budget.check(cx)?;
        // Only a durable publication reaches the existing watch sink. Do not
        // open/create the store before it: --once must also bootstrap a new one.
        if store.is_none() {
            store = Some(CompleteGenerationStore::open(cx, &options.root)?);
        }
        let store = store.as_ref().ok_or_else(|| watch_error("store was not admitted"))?;
        if !store.is_selected(cx, &generation)? {
            return Err(watch_error(
                "selection changed after watch publication; no foreign or stale generation is streamed",
            ));
        }
        if subscriber.is_none() {
            subscriber = Some(RetainedLiveSearchSession::new(
                runtime.clone(),
                store.clone(),
                options.query.clone(),
                options.limits,
                // The watcher has already debounced source publications.
                LiveSearchRefreshConfig {
                    debounce: Duration::ZERO,
                    max_wait: Duration::from_millis(1),
                },
            )?);
        }
        let subscriber = subscriber.as_mut().ok_or_else(|| watch_error("subscriber missing"))?;
        let changed = update_query(cx, subscriber, requested, &mut control)?;
        if !publication && !changed {
            continue;
        }
        let result = {
            let mut output = GuardedOutput { writer: &mut *writer, cx, budget };
            let mut sink = |frame: &RetainedLiveSearchFrame| {
                emit_receipted_frame(&generation, frame, &mut output)
            };
            subscriber.poll_with_sink(cx, Instant::now(), &mut sink).await
        };
        // Preserve typed cancellation and never replay a partial delivery.
        budget.check(cx)?;
        if result? == 0 {
            return Err(watch_error("a due publication or query produced no query phases"));
        }
        if publication {
            delivered = delivered
                .checked_add(1)
                .ok_or_else(|| watch_error("delivered generation counter exhausted"))?;
        }
        if options.max_updates.is_some_and(|maximum| delivered >= maximum) {
            return Ok(delivered);
        }
        // Only now resume the same writer future. Native hints received during
        // query/refinement remain in its existing bounded reconciliation state.
    }
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::task::{Context, Waker};

    use asupersync::test_utils::run_test_with_cx;
    use frankensearch_fsfs::FsfsConfig;
    use frankensearch_fsfs::generation_store::GenerationPublication;
    use frankensearch_fsfs::output_schema::SearchOutputPhase;
    use serde_json::json;

    use super::*;

    fn budget() -> Budget {
        Budget {
            started: Instant::now(),
            timeout: None,
        }
    }

    fn publish(cx: &Cx, store: &CompleteGenerationStore, value: &str) -> PublishedGeneration {
        let build = store.begin(cx).unwrap();
        std::fs::write(build.path().join("fixture"), value).unwrap();
        let GenerationPublication::Durable(generation) = build.publish(cx, |_, _| Ok(())).unwrap()
        else {
            panic!("fixture publication must be durable");
        };
        generation
    }

    fn frame(generation: &PublishedGeneration) -> RetainedLiveSearchFrame {
        RetainedLiveSearchFrame {
            generation_id: generation.id().to_owned(),
            manifest_sha256: generation.manifest_sha256().to_owned(),
            phase: SearchOutputPhase::RefinementFailed,
            annotations: json!({"skip_reason": "quality_timeout"}),
            update: None,
        }
    }

    fn options(source: &std::path::Path, store: &std::path::Path) -> Options {
        Options::parse(vec![
            "--hybrid".into(),
            "--watch-source".into(),
            source.as_os_str().to_owned(),
            "--index-dir".into(),
            store.as_os_str().to_owned(),
            "--query".into(),
            "alpha".into(),
            "--once".into(),
        ])
        .unwrap()
    }

    fn offline_runtime() -> FsfsRuntime {
        let mut config = FsfsConfig::default();
        config.indexing.offline = true;
        FsfsRuntime::new(config)
    }

    #[test]
    fn receipt_slot_never_overwrites_an_undelivered_publication() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let first = publish(&cx, &store, "first");
            let second = publish(&cx, &store, "second");
            let slot = ReceiptSlot::default();
            slot.offer(&first).unwrap();
            assert!(slot.offer(&second).is_err());
            assert_eq!(slot.take().unwrap(), Some(first));
            assert!(slot.take().unwrap().is_none());
        });
    }

    #[test]
    fn queued_receipt_is_delivered_without_polling_the_writer_again() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let generation = publish(&cx, &store, "one");
            let slot = ReceiptSlot::default();
            slot.offer(&generation).unwrap();
            let mut producer = pin!(poll_fn(|_| -> Poll<SearchResult<()>> {
                panic!("writer was polled despite output backpressure");
            }));
            let actual = next_publication(&cx, &budget(), producer.as_mut(), &slot)
                .await
                .unwrap();
            assert_eq!(actual, generation);
        });
    }

    #[test]
    fn a_receipt_created_during_poll_is_returned_in_the_same_turn() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let generation = publish(&cx, &store, "one");
            let slot = ReceiptSlot::default();
            let mut producer = pin!(poll_fn(|_| -> Poll<SearchResult<()>> {
                slot.offer(&generation).unwrap();
                Poll::Pending
            }));
            let actual = next_publication(&cx, &budget(), producer.as_mut(), &slot)
                .await
                .unwrap();
            assert_eq!(actual, generation);
        });
    }

    #[test]
    fn idle_writer_yields_instead_of_spinning_or_fabricating_results() {
        run_test_with_cx(|cx| async move {
            let slot = ReceiptSlot::default();
            let budget = budget();
            let mut producer = pin!(std::future::pending::<SearchResult<()>>());
            let mut next = pin!(next_publication(&cx, &budget, producer.as_mut(), &slot));
            let mut task = Context::from_waker(Waker::noop());
            assert!(next.as_mut().poll(&mut task).is_pending());
            assert!(slot.take().unwrap().is_none());
        });
    }

    #[test]
    fn producer_failure_preserves_its_original_error_without_a_snapshot() {
        run_test_with_cx(|cx| async move {
            let slot = ReceiptSlot::default();
            let mut producer = pin!(std::future::ready(Err(SearchError::Io(io::Error::from(
                io::ErrorKind::PermissionDenied
            ),))));
            let error = next_publication(&cx, &budget(), producer.as_mut(), &slot)
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::Io(error)
                if error.kind() == io::ErrorKind::PermissionDenied));
        });
    }

    #[test]
    fn cancelled_or_expired_command_cannot_poll_the_publisher() {
        run_test_with_cx(|cx| async move {
            let slot = ReceiptSlot::default();
            let mut producer = pin!(poll_fn(|_| -> Poll<SearchResult<()>> {
                panic!("cancelled or expired command reached publisher");
            }));
            let expired = Budget {
                started: Instant::now(),
                timeout: Some(Duration::ZERO),
            };
            assert!(matches!(
                next_publication(&cx, &expired, producer.as_mut(), &slot).await,
                Err(SearchError::SearchTimeout { .. })
            ));
            cx.set_cancel_requested(true);
            assert!(matches!(
                next_publication(&cx, &budget(), producer.as_mut(), &slot).await,
                Err(SearchError::Cancelled { .. })
            ));
        });
    }

    #[test]
    fn foreign_generation_or_digest_is_refused_before_output() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let generation = publish(&cx, &store, "one");
            for change_digest in [false, true] {
                let mut frame = frame(&generation);
                if change_digest {
                    frame.manifest_sha256.push('0');
                } else {
                    frame.generation_id.push('0');
                }
                let mut bytes = Vec::new();
                assert!(emit_receipted_frame(&generation, &frame, &mut bytes).is_err());
                assert!(bytes.is_empty());
            }
            let mut bytes = Vec::new();
            emit_receipted_frame(&generation, &frame(&generation), &mut bytes).unwrap();
            let decoded: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(decoded["generation_id"], generation.id());
            assert!(decoded["update"].is_null());
            assert_eq!(decoded["annotations"]["skip_reason"], "quality_timeout");
        });
    }

    #[test]
    fn watch_refuses_source_store_overlap_before_creating_a_candidate() {
        run_test_with_cx(|cx| async move {
            let source = tempfile::tempdir().unwrap();
            let store = source.path().join("must-not-be-created");
            let mut bytes = Vec::new();
            let error = execute(
                &cx,
                &budget(),
                &options(source.path(), &store),
                &mut bytes,
                Some(offline_runtime()),
            )
            .await
            .unwrap_err();
            assert!(error.to_string().contains("overlap"), "{error}");
            assert!(!store.exists());
            assert!(bytes.is_empty());
            assert_eq!(std::fs::read_dir(source.path()).unwrap().count(), 0);
        });
    }

    #[test]
    fn unsafe_runtime_policy_is_refused_before_source_or_store_access() {
        let source = std::path::Path::new("/not-opened-source");
        let store = std::path::Path::new("/not-created-store");
        let options = options(source, store);
        let mut config = FsfsConfig::default();
        config.indexing.offline = false;
        assert!(publication_runtime(&FsfsRuntime::new(config.clone()), &options).is_err());
        config.indexing.offline = true;
        config.search.shadow_mode = true;
        assert!(publication_runtime(&FsfsRuntime::new(config), &options).is_err());
    }

    #[derive(Default)]
    struct Queries {
        pending: Option<String>,
        resets: Vec<String>,
    }

    impl QueryControl for Queries {
        fn take_query(&mut self) -> SearchResult<Option<String>> {
            Ok(self.pending.take())
        }

        fn begin_query(&mut self, query: &str) -> SearchResult<()> {
            self.resets.push(query.to_owned());
            Ok(())
        }
    }

    #[test]
    fn interactive_query_does_not_poll_or_replace_the_owned_publisher() {
        run_test_with_cx(|cx| async move {
            let slot = ReceiptSlot::default();
            let mut queries = Queries { pending: Some("beta".to_owned()), ..Queries::default() };
            let mut control: Option<&mut dyn QueryControl> = Some(&mut queries);
            let mut producer = pin!(poll_fn(|_| -> Poll<SearchResult<()>> {
                panic!("query change must not start a publication");
            }));
            let operation = next_operation(
                &cx, &budget(), producer.as_mut(), &slot, &mut control, true,
            )
            .await
            .unwrap();
            assert!(matches!(operation, Operation::Query(query) if query == "beta"));
            assert!(slot.take().unwrap().is_none());
        });
    }

    #[test]
    fn publication_receipts_win_over_queries_and_initial_build_keeps_pending_text() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let generation = publish(&cx, &store, "one");
            let slot = ReceiptSlot::default();
            let mut queries = Queries { pending: Some("beta".to_owned()), ..Queries::default() };
            let mut control: Option<&mut dyn QueryControl> = Some(&mut queries);
            let mut producer = pin!(std::future::pending::<SearchResult<()>>());
            let budget = budget();
            {
                let mut operation = pin!(next_operation(
                    &cx, &budget, producer.as_mut(), &slot, &mut control, false,
                ));
                let mut task = Context::from_waker(Waker::noop());
                assert!(operation.as_mut().poll(&mut task).is_pending());
            }
            slot.offer(&generation).unwrap();
            let operation = next_operation(
                &cx, &budget, producer.as_mut(), &slot, &mut control, true,
            )
            .await
            .unwrap();
            assert!(matches!(operation, Operation::Publication(value) if value == generation));
            assert_eq!(control.as_mut().unwrap().take_query().unwrap().as_deref(), Some("beta"));
        });
    }

    #[test]
    fn query_transitions_reset_the_consumer_once_without_publishing_or_restarting_models() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let generation = publish(&cx, &store, "one");
            let mut subscriber = RetainedLiveSearchSession::new(
                offline_runtime(), store.clone(), "alpha",
                super::super::Options::parse(vec![
                    "--index-dir".into(), root.path().as_os_str().to_owned(),
                    "--query".into(), "alpha".into(),
                ]).unwrap().limits,
                LiveSearchRefreshConfig::default(),
            ).unwrap();
            let mut queries = Queries { pending: Some("beta".to_owned()), ..Queries::default() };
            {
                let mut control: Option<&mut dyn QueryControl> = Some(&mut queries);
                assert!(update_query(&cx, &mut subscriber, None, &mut control).unwrap());
                assert!(!update_query(
                    &cx, &mut subscriber, Some("beta".to_owned()), &mut control,
                ).unwrap());
                assert!(update_query(
                    &cx, &mut subscriber, Some("  ".to_owned()), &mut control,
                ).is_err());
            }
            assert_eq!(subscriber.query(), "beta");
            assert_eq!(queries.resets, ["beta"]);
            assert_eq!(store.active(&cx).unwrap(), Some(generation));
        });
    }

    #[test]
    fn interactive_requests_cannot_extend_expired_watch_budgets() {
        run_test_with_cx(|cx| async move {
            let slot = ReceiptSlot::default();
            let mut queries = Queries { pending: Some("beta".to_owned()), ..Queries::default() };
            {
                let mut control: Option<&mut dyn QueryControl> = Some(&mut queries);
                let expired = Budget { started: Instant::now(), timeout: Some(Duration::ZERO) };
                let mut producer = pin!(std::future::pending::<SearchResult<()>>());
                assert!(matches!(
                    next_operation(&cx, &expired, producer.as_mut(), &slot, &mut control, true).await,
                    Err(SearchError::SearchTimeout { .. })
                ));
            }
            assert_eq!(queries.pending.as_deref(), Some("beta"));
        });
    }
}
