use std::path::Path;

use asupersync::Cx;
use frankensearch_core::SearchResult;
use tracing::info;

use crate::config::{DiscoveryCandidate, DiscoveryDecision, FsfsConfig, RootDiscoveryDecision};
use crate::shutdown::{ShutdownCoordinator, ShutdownReason};

/// Supported fsfs interfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterfaceMode {
    Cli,
    Tui,
}

/// Shared runtime entrypoint used by interface adapters.
#[derive(Debug, Clone)]
pub struct FsfsRuntime {
    config: FsfsConfig,
}

impl FsfsRuntime {
    #[must_use]
    pub const fn new(config: FsfsConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub const fn config(&self) -> &FsfsConfig {
        &self.config
    }

    /// Produce deterministic root-scope decisions from the current discovery
    /// config. This is the first stage of corpus selection before filesystem
    /// walking starts.
    #[must_use]
    pub fn discovery_root_plan(&self) -> Vec<(String, RootDiscoveryDecision)> {
        self.config
            .discovery
            .roots
            .iter()
            .map(|root| {
                let decision = self.config.discovery.evaluate_root(Path::new(root), None);
                (root.clone(), decision)
            })
            .collect()
    }

    /// Expose the discovery policy evaluator for runtime callers.
    #[must_use]
    pub fn classify_discovery_candidate(
        &self,
        candidate: &DiscoveryCandidate<'_>,
    ) -> DiscoveryDecision {
        self.config.discovery.evaluate_candidate(candidate)
    }

    /// Dispatch by interface mode using the caller-provided `Cx`.
    ///
    /// # Errors
    ///
    /// Returns any surfaced `SearchError` from the selected runtime lane.
    pub async fn run_mode(&self, cx: &Cx, mode: InterfaceMode) -> SearchResult<()> {
        match mode {
            InterfaceMode::Cli => self.run_cli(cx).await,
            InterfaceMode::Tui => self.run_tui(cx).await,
        }
    }

    /// Dispatch by interface mode with shutdown/signal integration.
    ///
    /// This path is intended for long-lived runs (watch mode and TUI): it
    /// listens for shutdown requests while allowing config reload signals.
    ///
    /// # Errors
    ///
    /// Returns any surfaced `SearchError` from the selected runtime lane or
    /// graceful-shutdown finalization path.
    pub async fn run_mode_with_shutdown(
        &self,
        cx: &Cx,
        mode: InterfaceMode,
        shutdown: &ShutdownCoordinator,
    ) -> SearchResult<()> {
        match mode {
            InterfaceMode::Cli => self.run_cli_with_shutdown(cx, shutdown).await,
            InterfaceMode::Tui => self.run_tui_with_shutdown(cx, shutdown).await,
        }
    }

    /// CLI runtime lane scaffold.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when downstream CLI runtime logic fails.
    pub async fn run_cli(&self, _cx: &Cx) -> SearchResult<()> {
        std::future::ready(()).await;
        let root_plan = self.discovery_root_plan();
        let accepted_roots = root_plan
            .iter()
            .filter(|(_, decision)| decision.include())
            .count();

        for (root, decision) in &root_plan {
            info!(
                root,
                scope = ?decision.scope,
                reason_codes = ?decision.reason_codes,
                "fsfs discovery root policy evaluated"
            );
        }

        info!(
            profile = ?self.config.pressure.profile,
            total_roots = root_plan.len(),
            accepted_roots,
            rejected_roots = root_plan.len().saturating_sub(accepted_roots),
            "fsfs cli runtime scaffold invoked"
        );
        Ok(())
    }

    /// TUI runtime lane scaffold.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when downstream TUI runtime logic fails.
    pub async fn run_tui(&self, _cx: &Cx) -> SearchResult<()> {
        std::future::ready(()).await;
        info!(theme = ?self.config.tui.theme, "fsfs tui runtime scaffold invoked");
        Ok(())
    }

    async fn run_cli_with_shutdown(
        &self,
        cx: &Cx,
        shutdown: &ShutdownCoordinator,
    ) -> SearchResult<()> {
        self.run_cli(cx).await?;

        if self.config.indexing.watch_mode {
            let reason = self.await_shutdown(cx, shutdown).await;
            self.finalize_shutdown(cx, reason).await?;
        }

        Ok(())
    }

    async fn run_tui_with_shutdown(
        &self,
        cx: &Cx,
        shutdown: &ShutdownCoordinator,
    ) -> SearchResult<()> {
        self.run_tui(cx).await?;
        let reason = self.await_shutdown(cx, shutdown).await;
        self.finalize_shutdown(cx, reason).await
    }

    async fn await_shutdown(&self, cx: &Cx, shutdown: &ShutdownCoordinator) -> ShutdownReason {
        loop {
            if shutdown.take_reload_requested() {
                info!("fsfs runtime observed SIGHUP; config reload scaffold invoked");
            }

            if shutdown.is_shutting_down() {
                return shutdown
                    .current_reason()
                    .unwrap_or(ShutdownReason::UserRequest);
            }

            if cx.is_cancel_requested() {
                return ShutdownReason::Error(
                    "runtime cancelled while waiting for shutdown".to_owned(),
                );
            }

            asupersync::time::sleep(
                asupersync::time::wall_now(),
                std::time::Duration::from_millis(25),
            )
            .await;
        }
    }

    async fn finalize_shutdown(&self, _cx: &Cx, reason: ShutdownReason) -> SearchResult<()> {
        // Placeholder for fsync/WAL flush/index checkpoint once these subsystems
        // are wired into fsfs runtime lanes.
        std::future::ready(()).await;
        info!(reason = ?reason, "fsfs graceful shutdown finalization completed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use asupersync::test_utils::run_test_with_cx;

    use super::{FsfsRuntime, InterfaceMode};
    use crate::config::{DiscoveryCandidate, DiscoveryScopeDecision, FsfsConfig, IngestionClass};
    use crate::shutdown::{ShutdownCoordinator, ShutdownReason};

    #[test]
    fn runtime_modes_are_callable() {
        run_test_with_cx(|cx| async move {
            let runtime = FsfsRuntime::new(FsfsConfig::default());
            runtime
                .run_mode(&cx, InterfaceMode::Cli)
                .await
                .expect("cli mode");
            runtime
                .run_mode(&cx, InterfaceMode::Tui)
                .await
                .expect("tui mode");
        });
    }

    #[test]
    fn runtime_builds_root_discovery_plan() {
        let mut config = FsfsConfig::default();
        config.discovery.roots = vec!["/home/tester".into(), "/proc".into()];
        let runtime = FsfsRuntime::new(config);
        let plan = runtime.discovery_root_plan();

        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].0, "/home/tester");
        assert_eq!(plan[1].0, "/proc");
    }

    #[test]
    fn runtime_classifies_discovery_candidate() {
        let runtime = FsfsRuntime::new(FsfsConfig::default());
        let candidate = DiscoveryCandidate::new(Path::new("/home/tester/src/lib.rs"), 2_048);
        let decision = runtime.classify_discovery_candidate(&candidate);

        assert_eq!(decision.scope, DiscoveryScopeDecision::Include);
        assert_eq!(
            decision.ingestion_class,
            IngestionClass::FullSemanticLexical
        );
    }

    #[test]
    fn watch_mode_waits_for_shutdown_and_exits() {
        run_test_with_cx(|cx| async move {
            let mut config = FsfsConfig::default();
            config.indexing.watch_mode = true;
            let runtime = FsfsRuntime::new(config);
            let coordinator: Arc<ShutdownCoordinator> = Arc::new(ShutdownCoordinator::new());

            let trigger: Arc<ShutdownCoordinator> = Arc::clone(&coordinator);
            let worker = thread::spawn(move || {
                thread::sleep(Duration::from_millis(30));
                trigger.request_shutdown(ShutdownReason::UserRequest);
            });

            runtime
                .run_mode_with_shutdown(&cx, InterfaceMode::Cli, &coordinator)
                .await
                .expect("watch mode with shutdown");

            worker.join().expect("shutdown trigger thread join");
        });
    }

    #[test]
    fn shutdown_wait_observes_reload_then_user_shutdown() {
        run_test_with_cx(|cx| async move {
            let runtime = FsfsRuntime::new(FsfsConfig::default());
            let coordinator: Arc<ShutdownCoordinator> = Arc::new(ShutdownCoordinator::new());
            let trigger: Arc<ShutdownCoordinator> = Arc::clone(&coordinator);

            let worker = thread::spawn(move || {
                thread::sleep(Duration::from_millis(20));
                trigger.request_config_reload();
                thread::sleep(Duration::from_millis(20));
                trigger.request_shutdown(ShutdownReason::UserRequest);
            });

            runtime
                .run_mode_with_shutdown(&cx, InterfaceMode::Tui, &coordinator)
                .await
                .expect("tui mode with reload + shutdown");

            worker.join().expect("reload trigger thread join");
        });
    }
}
