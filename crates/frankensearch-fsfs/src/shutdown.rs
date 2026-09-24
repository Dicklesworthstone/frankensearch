#[cfg(not(windows))]
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
#[cfg(not(windows))]
use std::thread;
use std::time::Duration;
#[cfg(any(test, not(windows)))]
use std::time::Instant;

use asupersync::Cx;
use asupersync::types::CancelKind;
#[cfg(not(windows))]
use frankensearch_core::SearchError;
use frankensearch_core::SearchResult;
#[cfg(not(windows))]
use signal_hook::consts::signal::{SIGHUP, SIGINT, SIGQUIT, SIGTERM};
#[cfg(all(windows, test))]
use signal_hook::consts::signal::{SIGINT, SIGTERM};
#[cfg(not(windows))]
type SignalHandle = signal_hook::iterator::Handle;
#[cfg(any(test, not(windows)))]
use tracing::debug;
use tracing::{info, warn};

/// Time window where a second stop signal (`SIGINT` or `SIGTERM`) forces
/// immediate exit.
pub const FORCE_EXIT_WINDOW: Duration = Duration::from_secs(3);
const WAIT_POLL_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownState {
    Running,
    ShuttingDown,
    ForceExit,
}

impl ShutdownState {
    const fn as_u8(self) -> u8 {
        match self {
            Self::Running => 0,
            Self::ShuttingDown => 1,
            Self::ForceExit => 2,
        }
    }

    const fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::ShuttingDown,
            2 => Self::ForceExit,
            _ => Self::Running,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShutdownReason {
    Signal(i32),
    ConfigReload,
    Error(String),
    UserRequest,
}

/// Tracks lifecycle shutdown intent and signal-driven transitions.
#[derive(Debug)]
pub struct ShutdownCoordinator {
    shutdown_state: AtomicU8,
    shutdown_reason: Mutex<Option<ShutdownReason>>,
    #[cfg(any(test, not(windows)))]
    first_stop_signal_at: Mutex<Option<Instant>>,
    reload_requested: AtomicBool,
    diagnostics_dump_count: AtomicU64,
    signal_registration_active: AtomicBool,
    // Registration publishes two handles. Stop must not observe a partial
    // registration, nor may a restart overtake a listener that is still joining.
    signal_lifecycle: Mutex<()>,
    #[cfg(not(windows))]
    signal_handle: Mutex<Option<SignalHandle>>,
    #[cfg(not(windows))]
    signal_listener_thread: Mutex<Option<thread::JoinHandle<()>>>,
    cancellation_contexts: Mutex<Vec<Weak<Cx>>>,
}

impl Default for ShutdownCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl ShutdownCoordinator {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            shutdown_state: AtomicU8::new(ShutdownState::Running.as_u8()),
            shutdown_reason: Mutex::new(None),
            #[cfg(any(test, not(windows)))]
            first_stop_signal_at: Mutex::new(None),
            reload_requested: AtomicBool::new(false),
            diagnostics_dump_count: AtomicU64::new(0),
            signal_registration_active: AtomicBool::new(false),
            signal_lifecycle: Mutex::new(()),
            #[cfg(not(windows))]
            signal_handle: Mutex::new(None),
            #[cfg(not(windows))]
            signal_listener_thread: Mutex::new(None),
            cancellation_contexts: Mutex::new(Vec::new()),
        }
    }

    /// Register process signal listeners exactly once.
    ///
    /// # Errors
    ///
    /// Returns an error when signal handler registration fails.
    pub fn register_signals(self: &Arc<Self>) -> SearchResult<()> {
        let _lifecycle = lock_or_recover(&self.signal_lifecycle);
        if self
            .signal_registration_active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Ok(());
        }

        #[cfg(windows)]
        {
            warn!(
                "signal listener thread is not supported on windows; using shutdown requests only"
            );
        }

        #[cfg(not(windows))]
        {
            let mut signals =
                signal_hook::iterator::Signals::new([SIGINT, SIGTERM, SIGHUP, SIGQUIT]).map_err(
                    |error| {
                        self.signal_registration_active
                            .store(false, Ordering::Release);
                        SearchError::SubsystemError {
                            subsystem: "fsfs",
                            source: Box::new(io::Error::other(format!(
                                "failed to register signal listeners: {error}"
                            ))),
                        }
                    },
                )?;
            let handle = signals.handle();

            // A strong reference here would keep the coordinator and its
            // process-wide signal handlers alive forever after the owner exits.
            let coordinator = Arc::downgrade(self);
            let listener = thread::Builder::new()
                .name("fsfs-signal-listener".to_owned())
                .spawn(move || {
                    for signal in signals.forever() {
                        let Some(coordinator) = coordinator.upgrade() else {
                            break;
                        };
                        coordinator.handle_signal(signal);
                        if coordinator.is_force_exit_requested() {
                            std::process::exit(crate::exit_code::INTERRUPTED);
                        }
                    }
                })
                .map_err(|error| {
                    self.signal_registration_active
                        .store(false, Ordering::Release);
                    SearchError::SubsystemError {
                        subsystem: "fsfs",
                        source: Box::new(io::Error::other(format!(
                            "failed to start signal listener thread: {error}"
                        ))),
                    }
                })?;

            *lock_or_recover(&self.signal_handle) = Some(handle);
            *lock_or_recover(&self.signal_listener_thread) = Some(listener);
        }

        Ok(())
    }

    /// Stop the signal listener thread and clear registration state.
    pub fn stop_signal_listener(&self) {
        let _lifecycle = lock_or_recover(&self.signal_lifecycle);
        #[cfg(not(windows))]
        {
            let signal_handle = lock_or_recover(&self.signal_handle).take();
            if let Some(handle) = signal_handle {
                handle.close();
            }

            let listener_thread = lock_or_recover(&self.signal_listener_thread).take();
            // The listener can release the last temporary upgraded Arc after
            // handling a signal. Its Drop closes the iterator, but must not
            // attempt to join the thread currently running that destructor.
            if let Some(listener_thread) = listener_thread
                && listener_thread.thread().id() != thread::current().id()
                && let Err(error) = listener_thread.join()
            {
                warn!(
                    ?error,
                    "fsfs signal listener thread panicked while stopping"
                );
            }
        }

        self.signal_registration_active
            .store(false, Ordering::Release);
    }

    /// Wait until shutdown is requested (signal/user/error) and return the reason.
    pub async fn wait_for_shutdown(&self, cx: &Cx) -> ShutdownReason {
        loop {
            if let Some(reason) = self.current_reason()
                && self.is_shutting_down()
            {
                return reason;
            }

            if cx.checkpoint().is_err() || cx.is_cancel_requested() {
                return ShutdownReason::Error(
                    "operation cancelled while waiting for shutdown".to_owned(),
                );
            }

            asupersync::time::sleep(cx.now(), WAIT_POLL_INTERVAL).await;
        }
    }

    /// Request graceful shutdown from non-signal sources (user, internal error).
    pub fn request_shutdown(&self, reason: ShutdownReason) {
        if self
            .shutdown_state
            .compare_exchange(
                ShutdownState::Running.as_u8(),
                ShutdownState::ShuttingDown.as_u8(),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            self.set_reason(reason);
            let mut contexts = lock_or_recover(&self.cancellation_contexts);
            contexts.retain(|context| {
                context.upgrade().is_some_and(|cx| {
                    cx.cancel_fast(CancelKind::User);
                    true
                })
            });
            drop(contexts);
            info!(reason = ?self.current_reason(), "shutdown requested");
        }
    }

    /// Bind cancellation to shutdown for the lifetime of the returned guard.
    ///
    /// The state check and registration share a lock with notification so a
    /// shutdown racing registration cannot leave the operation running.
    pub fn cancellation_scope(&self, cx: &Cx) -> Arc<Cx> {
        let context = Arc::new(cx.clone());
        let mut contexts = lock_or_recover(&self.cancellation_contexts);
        contexts.retain(|context| context.strong_count() > 0);
        contexts.push(Arc::downgrade(&context));
        if self.is_shutting_down() {
            context.cancel_fast(CancelKind::User);
        }
        drop(contexts);
        context
    }

    /// Mark a pending config reload request (SIGHUP).
    pub fn request_config_reload(&self) {
        self.reload_requested.store(true, Ordering::Release);
        self.set_reason_if_missing(ShutdownReason::ConfigReload);
        info!("configuration reload requested");
    }

    #[must_use]
    pub fn take_reload_requested(&self) -> bool {
        self.reload_requested.swap(false, Ordering::AcqRel)
    }

    #[must_use]
    pub fn state(&self) -> ShutdownState {
        ShutdownState::from_u8(self.shutdown_state.load(Ordering::Acquire))
    }

    #[must_use]
    pub fn is_shutting_down(&self) -> bool {
        self.state() != ShutdownState::Running
    }

    #[must_use]
    pub fn is_force_exit_requested(&self) -> bool {
        self.state() == ShutdownState::ForceExit
    }

    #[must_use]
    pub fn current_reason(&self) -> Option<ShutdownReason> {
        lock_or_recover(&self.shutdown_reason).clone()
    }

    #[must_use]
    pub fn diagnostics_dump_count(&self) -> u64 {
        self.diagnostics_dump_count.load(Ordering::Acquire)
    }

    #[cfg(any(test, not(windows)))]
    fn handle_signal(&self, signal: i32) {
        match signal {
            // Both stop signals escalate. SIGTERM used to request a graceful
            // shutdown and nothing more, so repeating it did nothing and an
            // operator whose graceful path was wedged had no recourse short of
            // SIGKILL (#43). The escape hatch was reachable only by an
            // interactive Ctrl-C, while `kill`, systemd and every process
            // supervisor send SIGTERM — so it was missing from exactly the
            // context that needs it.
            SIGINT | SIGTERM => self.handle_stop_signal(signal),
            #[cfg(not(windows))]
            SIGHUP => {
                self.request_config_reload();
                info!("received SIGHUP, config reload queued");
            }
            #[cfg(not(windows))]
            SIGQUIT => {
                let dump_count = self.diagnostics_dump_count.fetch_add(1, Ordering::AcqRel) + 1;
                warn!(
                    diagnostics_dump_count = dump_count,
                    state = ?self.state(),
                    "received SIGQUIT, diagnostics dump requested"
                );
            }
            _ => {
                debug!(signal, "received unsupported signal");
            }
        }
    }

    /// Handle SIGINT or SIGTERM: the first requests a graceful shutdown, a
    /// second within [`FORCE_EXIT_WINDOW`] forces exit.
    ///
    /// The window is shared between the two rather than tracked per signal, so
    /// Ctrl-C followed by `kill` escalates as readily as either on its own —
    /// an operator reaching for the second signal means the same thing
    /// whichever they send.
    #[cfg(any(test, not(windows)))]
    fn handle_stop_signal(&self, signal: i32) {
        let now = Instant::now();
        match self.state() {
            ShutdownState::Running => {
                *lock_or_recover(&self.first_stop_signal_at) = Some(now);
                self.request_shutdown(ShutdownReason::Signal(signal));
                info!(
                    signal,
                    "received first stop signal, initiating graceful shutdown"
                );
            }
            ShutdownState::ShuttingDown => {
                let first_stop_signal_at = *lock_or_recover(&self.first_stop_signal_at);
                if let Some(first) = first_stop_signal_at
                    && now.saturating_duration_since(first) <= FORCE_EXIT_WINDOW
                {
                    self.promote_force_exit(signal);
                    return;
                }

                *lock_or_recover(&self.first_stop_signal_at) = Some(now);
                debug!(
                    signal,
                    "received stop signal outside force-exit window; remaining in graceful shutdown"
                );
            }
            ShutdownState::ForceExit => {}
        }
    }

    #[cfg(any(test, not(windows)))]
    fn promote_force_exit(&self, signal: i32) {
        self.shutdown_state
            .store(ShutdownState::ForceExit.as_u8(), Ordering::Release);
        self.set_reason(ShutdownReason::Signal(signal));
        warn!(
            signal,
            "received second stop signal within window, forcing immediate exit"
        );
    }

    fn set_reason(&self, reason: ShutdownReason) {
        *lock_or_recover(&self.shutdown_reason) = Some(reason);
    }

    fn set_reason_if_missing(&self, reason: ShutdownReason) {
        let mut guard = lock_or_recover(&self.shutdown_reason);
        if guard.is_none() {
            *guard = Some(reason);
        }
    }

    #[cfg(test)]
    pub(crate) fn process_signal_for_test(&self, signal: i32) {
        self.handle_signal(signal);
    }
}

impl Drop for ShutdownCoordinator {
    fn drop(&mut self) {
        self.stop_signal_listener();
    }
}

fn lock_or_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use asupersync::test_utils::run_test_with_cx;
    #[cfg(not(windows))]
    use signal_hook::consts::signal::{SIGHUP, SIGQUIT};
    use signal_hook::consts::signal::{SIGINT, SIGTERM};

    use super::{ShutdownCoordinator, ShutdownReason, ShutdownState};

    #[test]
    fn sigterm_transitions_to_graceful_shutdown() {
        let coordinator = ShutdownCoordinator::new();
        assert_eq!(coordinator.state(), ShutdownState::Running);

        coordinator.process_signal_for_test(SIGTERM);

        assert_eq!(coordinator.state(), ShutdownState::ShuttingDown);
        assert!(coordinator.is_shutting_down());
        assert_eq!(
            coordinator.current_reason(),
            Some(ShutdownReason::Signal(SIGTERM))
        );
    }

    #[test]
    fn second_sigterm_forces_exit_like_a_second_sigint() {
        // SIGTERM used to request a graceful shutdown and nothing more, so a
        // wedged shutdown left SIGKILL as the only recourse — and SIGTERM is
        // what `kill`, systemd and every process supervisor send, so the
        // escape hatch was missing from the context that needs it most (#43).
        let coordinator = ShutdownCoordinator::new();

        coordinator.process_signal_for_test(SIGTERM);
        assert_eq!(coordinator.state(), ShutdownState::ShuttingDown);
        assert!(!coordinator.is_force_exit_requested());

        coordinator.process_signal_for_test(SIGTERM);
        assert_eq!(coordinator.state(), ShutdownState::ForceExit);
        assert!(coordinator.is_force_exit_requested());
        assert_eq!(
            coordinator.current_reason(),
            Some(ShutdownReason::Signal(SIGTERM))
        );
    }

    #[test]
    fn a_stop_signal_escalates_whichever_one_arrives_second() {
        // The force-exit window is shared rather than tracked per signal: an
        // operator who presses Ctrl-C and then reaches for `kill` means the
        // same thing as one who sends either twice.
        for (first, second) in [(SIGINT, SIGTERM), (SIGTERM, SIGINT)] {
            let coordinator = ShutdownCoordinator::new();
            coordinator.process_signal_for_test(first);
            assert_eq!(coordinator.state(), ShutdownState::ShuttingDown);

            coordinator.process_signal_for_test(second);
            assert_eq!(
                coordinator.state(),
                ShutdownState::ForceExit,
                "{first} then {second} must force exit"
            );
            assert_eq!(
                coordinator.current_reason(),
                Some(ShutdownReason::Signal(second)),
                "the reason names the signal that forced the exit"
            );
        }
    }

    #[test]
    fn first_sigint_marks_shutting_down() {
        let coordinator = ShutdownCoordinator::new();
        coordinator.process_signal_for_test(SIGINT);

        assert!(coordinator.is_shutting_down());
        assert_eq!(coordinator.state(), ShutdownState::ShuttingDown);
    }

    #[test]
    fn gh43_shutdown_cancels_only_live_scopes() {
        run_test_with_cx(|cx| async move {
            let coordinator = ShutdownCoordinator::new();
            let scope = coordinator.cancellation_scope(&cx);
            coordinator.process_signal_for_test(SIGTERM);
            assert!(cx.checkpoint().is_err());
            drop(scope);
            cx.set_cancel_requested(false);

            let coordinator = ShutdownCoordinator::new();
            drop(coordinator.cancellation_scope(&cx));
            coordinator.process_signal_for_test(SIGTERM);
            assert!(
                cx.checkpoint().is_ok(),
                "dropped scope must not cancel watch cleanup"
            );
        });
    }

    #[test]
    fn gh43_shutdown_before_registration_cancels_new_scope() {
        run_test_with_cx(|cx| async move {
            let coordinator = ShutdownCoordinator::new();
            coordinator.request_shutdown(ShutdownReason::UserRequest);
            let _scope = coordinator.cancellation_scope(&cx);
            assert!(cx.checkpoint().is_err());
            cx.set_cancel_requested(false);
        });
    }

    #[test]
    fn gh43_concurrent_shutdown_and_registration_cannot_lose_cancellation() {
        run_test_with_cx(|cx| async move {
            for _ in 0..32 {
                let coordinator = Arc::new(ShutdownCoordinator::new());
                let trigger = Arc::clone(&coordinator);
                let barrier = Arc::new(std::sync::Barrier::new(2));
                let worker_barrier = Arc::clone(&barrier);
                let worker = thread::spawn(move || {
                    worker_barrier.wait();
                    trigger.request_shutdown(ShutdownReason::UserRequest);
                });
                barrier.wait();
                let scope = coordinator.cancellation_scope(&cx);
                worker.join().expect("shutdown thread");
                assert!(cx.checkpoint().is_err());
                drop(scope);
                cx.set_cancel_requested(false);
            }
        });
    }

    #[test]
    fn second_sigint_promotes_force_exit() {
        let coordinator = ShutdownCoordinator::new();
        coordinator.process_signal_for_test(SIGINT);
        coordinator.process_signal_for_test(SIGINT);

        assert_eq!(coordinator.state(), ShutdownState::ForceExit);
        assert!(coordinator.is_force_exit_requested());
    }

    #[test]
    #[cfg(not(windows))]
    fn sighup_requests_reload_without_shutdown() {
        let coordinator = ShutdownCoordinator::new();
        coordinator.process_signal_for_test(SIGHUP);

        assert_eq!(coordinator.state(), ShutdownState::Running);
        assert!(coordinator.take_reload_requested());
        assert!(!coordinator.take_reload_requested());
        assert_eq!(
            coordinator.current_reason(),
            Some(ShutdownReason::ConfigReload)
        );
    }

    #[test]
    #[cfg(not(windows))]
    fn sigquit_increments_diagnostics_counter_without_shutdown() {
        let coordinator = ShutdownCoordinator::new();
        coordinator.process_signal_for_test(SIGQUIT);
        coordinator.process_signal_for_test(SIGQUIT);

        assert_eq!(coordinator.state(), ShutdownState::Running);
        assert_eq!(coordinator.diagnostics_dump_count(), 2);
    }

    #[test]
    fn wait_for_shutdown_returns_requested_reason() {
        run_test_with_cx(|cx| async move {
            let coordinator = Arc::new(ShutdownCoordinator::new());
            let trigger = Arc::clone(&coordinator);
            let worker = thread::spawn(move || {
                thread::sleep(Duration::from_millis(30));
                trigger.request_shutdown(ShutdownReason::UserRequest);
            });

            let reason = coordinator.wait_for_shutdown(&cx).await;
            worker.join().expect("shutdown trigger thread join");

            assert_eq!(reason, ShutdownReason::UserRequest);
        });
    }

    #[test]
    fn shutdown_reason_overrides_prior_reload_reason() {
        let coordinator = ShutdownCoordinator::new();
        coordinator.request_config_reload();
        coordinator.process_signal_for_test(SIGTERM);

        assert!(coordinator.is_shutting_down());
        assert_eq!(
            coordinator.current_reason(),
            Some(ShutdownReason::Signal(SIGTERM))
        );
    }

    #[test]
    fn signal_listener_can_be_stopped_and_restarted() {
        let coordinator = Arc::new(ShutdownCoordinator::new());
        coordinator
            .register_signals()
            .expect("register signal handlers");
        coordinator.stop_signal_listener();

        coordinator
            .register_signals()
            .expect("register signal handlers again");
        coordinator.stop_signal_listener();
    }

    #[test]
    #[cfg(not(windows))]
    fn dropping_coordinator_releases_registered_listener() {
        let coordinator = Arc::new(ShutdownCoordinator::new());
        coordinator.register_signals().expect("register signals");
        let weak = Arc::downgrade(&coordinator);

        // No stop call: error paths and library callers rely on owner lifetime.
        drop(coordinator);
        assert!(
            weak.upgrade().is_none(),
            "listener must not retain its owner"
        );
    }

    #[test]
    #[cfg(not(windows))]
    fn repeated_registration_keeps_one_listener_without_retaining_owner() {
        let coordinator = Arc::new(ShutdownCoordinator::new());
        coordinator.register_signals().expect("register signals");
        let listener_id = super::lock_or_recover(&coordinator.signal_listener_thread)
            .as_ref()
            .expect("listener installed")
            .thread()
            .id();

        coordinator.register_signals().expect("register again");
        assert_eq!(Arc::strong_count(&coordinator), 1);
        assert_eq!(
            super::lock_or_recover(&coordinator.signal_listener_thread)
                .as_ref()
                .expect("listener remains installed")
                .thread()
                .id(),
            listener_id
        );
    }

    #[test]
    #[cfg(not(windows))]
    fn concurrent_registration_and_stop_leave_no_listener_installed() {
        let coordinator = Arc::new(ShutdownCoordinator::new());
        let barrier = Arc::new(std::sync::Barrier::new(5));
        let workers: Vec<_> = (0..4)
            .map(|_| {
                let coordinator = Arc::clone(&coordinator);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..8 {
                        coordinator.register_signals().expect("register signals");
                        coordinator.stop_signal_listener();
                    }
                })
            })
            .collect();
        barrier.wait();
        for worker in workers {
            worker.join().expect("lifecycle worker");
        }
        coordinator.stop_signal_listener();
        assert!(super::lock_or_recover(&coordinator.signal_handle).is_none());
        assert!(super::lock_or_recover(&coordinator.signal_listener_thread).is_none());
        assert!(
            !coordinator
                .signal_registration_active
                .load(std::sync::atomic::Ordering::Acquire)
        );
        let weak = Arc::downgrade(&coordinator);
        drop(coordinator);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    #[cfg(not(windows))]
    fn listener_thread_can_drop_its_own_coordinator_without_self_join() {
        let coordinator = Arc::new(ShutdownCoordinator::new());
        let weak = Arc::downgrade(&coordinator);
        let (owner_tx, owner_rx) = std::sync::mpsc::channel::<Arc<ShutdownCoordinator>>();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let listener = thread::spawn(move || {
            let owner = owner_rx.recv().expect("coordinator ownership");
            drop(owner);
            done_tx.send(()).expect("report completed destruction");
        });
        *super::lock_or_recover(&coordinator.signal_listener_thread) = Some(listener);
        owner_tx.send(coordinator).expect("transfer final owner");

        done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("destructor must not join its own thread");
        assert!(weak.upgrade().is_none());
    }
}
