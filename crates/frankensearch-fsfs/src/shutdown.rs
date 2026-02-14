use std::io;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use signal_hook::consts::signal::{SIGHUP, SIGINT, SIGQUIT, SIGTERM};
use signal_hook::iterator::{Handle as SignalHandle, Signals};
use tracing::{debug, info, warn};

/// Time window where a second `SIGINT` forces immediate exit.
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
    first_sigint_at: Mutex<Option<Instant>>,
    reload_requested: AtomicBool,
    diagnostics_dump_count: AtomicU64,
    signal_registration_active: AtomicBool,
    signal_handle: Mutex<Option<SignalHandle>>,
    signal_listener_thread: Mutex<Option<thread::JoinHandle<()>>>,
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
            first_sigint_at: Mutex::new(None),
            reload_requested: AtomicBool::new(false),
            diagnostics_dump_count: AtomicU64::new(0),
            signal_registration_active: AtomicBool::new(false),
            signal_handle: Mutex::new(None),
            signal_listener_thread: Mutex::new(None),
        }
    }

    /// Register process signal listeners exactly once.
    ///
    /// # Errors
    ///
    /// Returns an error when signal handler registration fails.
    pub fn register_signals(self: &Arc<Self>) -> SearchResult<()> {
        if self
            .signal_registration_active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Ok(());
        }

        let mut signals = Signals::new([SIGINT, SIGTERM, SIGHUP, SIGQUIT]).map_err(|error| {
            self.signal_registration_active
                .store(false, Ordering::Release);
            SearchError::SubsystemError {
                subsystem: "fsfs",
                source: Box::new(io::Error::other(format!(
                    "failed to register signal listeners: {error}"
                ))),
            }
        })?;
        let handle = signals.handle();

        let coordinator = Arc::clone(self);
        let listener = thread::Builder::new()
            .name("fsfs-signal-listener".to_owned())
            .spawn(move || {
                for signal in signals.forever() {
                    coordinator.handle_signal(signal);
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

        Ok(())
    }

    /// Stop the signal listener thread and clear registration state.
    pub fn stop_signal_listener(&self) {
        let signal_handle = lock_or_recover(&self.signal_handle).take();
        if let Some(handle) = signal_handle {
            handle.close();
        }

        let listener_thread = lock_or_recover(&self.signal_listener_thread).take();
        if let Some(listener_thread) = listener_thread
            && let Err(error) = listener_thread.join()
        {
            warn!(
                ?error,
                "fsfs signal listener thread panicked while stopping"
            );
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

            if cx.is_cancel_requested() {
                return ShutdownReason::Error(
                    "operation cancelled while waiting for shutdown".to_owned(),
                );
            }

            asupersync::time::sleep(asupersync::time::wall_now(), WAIT_POLL_INTERVAL).await;
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
            info!(reason = ?self.current_reason(), "shutdown requested");
        }
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

    fn handle_signal(&self, signal: i32) {
        match signal {
            SIGINT => self.handle_sigint(),
            SIGTERM => {
                self.request_shutdown(ShutdownReason::Signal(SIGTERM));
                info!("received SIGTERM, initiating graceful shutdown");
            }
            SIGHUP => {
                self.request_config_reload();
                info!("received SIGHUP, config reload queued");
            }
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

    fn handle_sigint(&self) {
        let now = Instant::now();
        match self.state() {
            ShutdownState::Running => {
                *lock_or_recover(&self.first_sigint_at) = Some(now);
                self.request_shutdown(ShutdownReason::Signal(SIGINT));
                info!("received first SIGINT, initiating graceful shutdown");
            }
            ShutdownState::ShuttingDown => {
                let first_sigint_at = *lock_or_recover(&self.first_sigint_at);
                if let Some(first) = first_sigint_at
                    && now.saturating_duration_since(first) <= FORCE_EXIT_WINDOW
                {
                    self.promote_force_exit();
                    return;
                }

                *lock_or_recover(&self.first_sigint_at) = Some(now);
                debug!("received SIGINT outside force-exit window; remaining in graceful shutdown");
            }
            ShutdownState::ForceExit => {}
        }
    }

    fn promote_force_exit(&self) {
        self.shutdown_state
            .store(ShutdownState::ForceExit.as_u8(), Ordering::Release);
        self.set_reason(ShutdownReason::Signal(SIGINT));
        warn!("received second SIGINT within window, forcing immediate exit");
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
    use signal_hook::consts::signal::{SIGHUP, SIGINT, SIGQUIT, SIGTERM};

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
    fn first_sigint_marks_shutting_down() {
        let coordinator = ShutdownCoordinator::new();
        coordinator.process_signal_for_test(SIGINT);

        assert!(coordinator.is_shutting_down());
        assert_eq!(coordinator.state(), ShutdownState::ShuttingDown);
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
        coordinator.process_signal_for_test(SIGHUP);
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
}
