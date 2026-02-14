//! Cross-epic contract sanity checks for telemetry schema and adapter lockstep.
//!
//! Verifies that all telemetry-producing components and host adapters use
//! compatible schema versions and redaction policies. This module implements
//! the validation workflow described in `bd-2ugv`: a single cross-epic
//! compatibility contract covering core, fsfs, and ops surfaces.
//!
//! # Contract Rules
//!
//! 1. **Schema version lockstep**: Every adapter's `telemetry_schema_version`
//!    must match [`TELEMETRY_SCHEMA_VERSION`](crate::collectors::TELEMETRY_SCHEMA_VERSION).
//! 2. **Redaction policy alignment**: Every adapter must declare the same
//!    `redaction_policy_version` as the harness expects.
//! 3. **Compatibility window**: Adapters may lag by at most
//!    [`MAX_SCHEMA_VERSION_LAG`] versions during rolling upgrades.
//! 4. **Deprecation**: Schema versions older than `current - MAX_SCHEMA_VERSION_LAG`
//!    are rejected outright with a deprecation violation.
//! 5. **Forward compatibility**: Adapters reporting a schema version *newer*
//!    than the core library must be rejected (core must be upgraded first).

use serde::{Deserialize, Serialize};

use crate::collectors::TELEMETRY_SCHEMA_VERSION;
use crate::host_adapter::{ConformanceHarness, ConformanceViolation, HostAdapter};

/// Maximum allowed schema version lag during rolling upgrades.
///
/// Adapters whose `telemetry_schema_version` is within
/// `[current - MAX_SCHEMA_VERSION_LAG, current]` are considered compatible
/// (with a warning). Adapters outside this window are rejected.
pub const MAX_SCHEMA_VERSION_LAG: u8 = 1;

const REPLAY_CONTRACT_SANITY_TESTS: &str =
    "cargo test -p frankensearch-core contract_sanity::tests -- --nocapture";
const REPLAY_ADAPTER_CONFORMANCE_TESTS: &str =
    "cargo test -p frankensearch-core host_adapter::tests -- --nocapture";

// ---------------------------------------------------------------------------
// Contract report
// ---------------------------------------------------------------------------

/// Result of a cross-epic contract sanity check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContractSanityReport {
    /// Current core schema version.
    pub core_schema_version: u8,
    /// Number of adapters checked.
    pub adapters_checked: usize,
    /// Number of adapters that passed all checks.
    pub adapters_passed: usize,
    /// Per-adapter results.
    pub adapter_results: Vec<AdapterContractResult>,
    /// Overall pass/fail.
    pub passed: bool,
}

/// Per-adapter contract check result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterContractResult {
    /// Adapter identifier.
    pub adapter_id: String,
    /// Host project name.
    pub host_project: String,
    /// Adapter's declared schema version.
    pub adapter_schema_version: u8,
    /// Whether this adapter passed all checks.
    pub passed: bool,
    /// Compatibility status.
    pub compatibility: CompatibilityStatus,
    /// Detailed violations (if any).
    pub violations: Vec<ConformanceViolation>,
}

/// Schema version compatibility status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompatibilityStatus {
    /// Adapter schema version matches core exactly.
    Exact,
    /// Adapter is within the compatibility window (lagging but acceptable).
    Compatible {
        /// How many versions behind.
        lag: u8,
    },
    /// Adapter schema version is too old (outside compatibility window).
    Deprecated {
        /// How many versions behind.
        lag: u8,
    },
    /// Adapter reports a newer schema than core (forward incompatible).
    TooNew {
        /// How many versions ahead.
        ahead: u8,
    },
}

/// Severity assigned to one contract violation diagnostic entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViolationSeverity {
    /// Informational/warning signal that should be addressed, but does not fail the run.
    Warning,
    /// Hard-failure signal that must be remediated before rollout.
    Error,
}

/// Deterministic diagnostic record for one contract violation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContractViolationDiagnostic {
    /// Adapter identifier that produced the violation.
    pub adapter_id: String,
    /// Host project associated with the adapter.
    pub host_project: String,
    /// Compatibility classification at the time of violation.
    pub compatibility: CompatibilityStatus,
    /// Stable reason code.
    pub reason_code: String,
    /// Field/path associated with the violation.
    pub field: String,
    /// Human-readable message.
    pub message: String,
    /// Derived severity used by rollout gates.
    pub severity: ViolationSeverity,
    /// Deterministic replay command for triage.
    pub replay_command: String,
}

impl ContractSanityReport {
    /// Expand report violations into deterministic triage diagnostics.
    #[must_use]
    pub fn diagnostics(&self) -> Vec<ContractViolationDiagnostic> {
        let mut diagnostics = Vec::new();
        for result in &self.adapter_results {
            for violation in &result.violations {
                diagnostics.push(ContractViolationDiagnostic {
                    adapter_id: result.adapter_id.clone(),
                    host_project: result.host_project.clone(),
                    compatibility: result.compatibility.clone(),
                    reason_code: violation.code.clone(),
                    field: violation.field.clone(),
                    message: violation.message.clone(),
                    severity: classify_violation_severity(&violation.code, &result.compatibility),
                    replay_command: replay_command_for_reason(&violation.code, &result.adapter_id),
                });
            }
        }

        diagnostics.sort_by(|left, right| {
            left.adapter_id
                .cmp(&right.adapter_id)
                .then(left.reason_code.cmp(&right.reason_code))
                .then(left.field.cmp(&right.field))
        });
        diagnostics
    }
}

// ---------------------------------------------------------------------------
// Checker
// ---------------------------------------------------------------------------

/// Cross-epic contract sanity checker.
///
/// Validates a set of host adapters against the current core schema version
/// and conformance harness configuration.
pub struct ContractSanityChecker {
    harness: ConformanceHarness,
}

impl Default for ContractSanityChecker {
    fn default() -> Self {
        Self::new(ConformanceHarness::default())
    }
}

impl ContractSanityChecker {
    /// Create a checker with an explicit conformance harness.
    #[must_use]
    pub const fn new(harness: ConformanceHarness) -> Self {
        Self { harness }
    }

    /// Check a single adapter's contract compliance.
    #[must_use]
    pub fn check_adapter(&self, adapter: &dyn HostAdapter) -> AdapterContractResult {
        let identity = adapter.identity();
        let mut violations = self.harness.validate_identity(&identity);
        let compatibility = classify_version_against(
            TELEMETRY_SCHEMA_VERSION,
            identity.telemetry_schema_version,
            MAX_SCHEMA_VERSION_LAG,
        );

        // Add compatibility-specific violations.
        match &compatibility {
            CompatibilityStatus::Deprecated { lag } => {
                violations.push(ConformanceViolation {
                    code: "contract.schema.deprecated".to_owned(),
                    field: "identity.telemetry_schema_version".to_owned(),
                    message: format!(
                        "schema v{} is {} version(s) behind current v{} \
                         (max lag: {MAX_SCHEMA_VERSION_LAG})",
                        identity.telemetry_schema_version, lag, TELEMETRY_SCHEMA_VERSION
                    ),
                });
            }
            CompatibilityStatus::TooNew { ahead } => {
                violations.push(ConformanceViolation {
                    code: "contract.schema.too_new".to_owned(),
                    field: "identity.telemetry_schema_version".to_owned(),
                    message: format!(
                        "adapter schema v{} is {} version(s) ahead of core v{} \
                         — core must be upgraded first",
                        identity.telemetry_schema_version, ahead, TELEMETRY_SCHEMA_VERSION
                    ),
                });
            }
            CompatibilityStatus::Compatible { lag } => {
                // Not a violation, but log as informational.
                // We still pass the adapter but record the lag.
                if *lag > 0 {
                    violations.push(ConformanceViolation {
                        code: "contract.schema.lagging".to_owned(),
                        field: "identity.telemetry_schema_version".to_owned(),
                        message: format!(
                            "adapter schema v{} lags core v{} by {} version(s) \
                             — within compatibility window but should be updated",
                            identity.telemetry_schema_version, TELEMETRY_SCHEMA_VERSION, lag
                        ),
                    });
                }
            }
            CompatibilityStatus::Exact => {}
        }

        // A lagging adapter within the window is a warning, not a failure.
        let compatible_lagging = matches!(compatibility, CompatibilityStatus::Compatible { .. });
        let has_hard_violations = violations.iter().any(|violation| {
            if compatible_lagging {
                !matches!(
                    violation.code.as_str(),
                    "contract.schema.lagging" | "adapter.identity.schema_version_mismatch"
                )
            } else {
                true
            }
        });

        let passed = !has_hard_violations;

        AdapterContractResult {
            adapter_id: identity.adapter_id,
            host_project: identity.host_project,
            adapter_schema_version: identity.telemetry_schema_version,
            passed,
            compatibility,
            violations,
        }
    }

    /// Check all adapters and produce a summary report.
    #[must_use]
    pub fn check_all(&self, adapters: &[&dyn HostAdapter]) -> ContractSanityReport {
        let adapter_results: Vec<_> = adapters
            .iter()
            .map(|adapter| self.check_adapter(*adapter))
            .collect();

        let adapters_passed = adapter_results.iter().filter(|r| r.passed).count();
        let passed = adapters_passed == adapter_results.len();

        ContractSanityReport {
            core_schema_version: TELEMETRY_SCHEMA_VERSION,
            adapters_checked: adapter_results.len(),
            adapters_passed,
            adapter_results,
            passed,
        }
    }
}

fn classify_violation_severity(
    reason_code: &str,
    compatibility: &CompatibilityStatus,
) -> ViolationSeverity {
    match reason_code {
        "contract.schema.lagging" => ViolationSeverity::Warning,
        "adapter.identity.schema_version_mismatch"
            if matches!(compatibility, CompatibilityStatus::Compatible { .. }) =>
        {
            ViolationSeverity::Warning
        }
        _ => ViolationSeverity::Error,
    }
}

/// Return a deterministic replay command for a contract violation reason code.
#[must_use]
pub fn replay_command_for_reason(reason_code: &str, adapter_id: &str) -> String {
    let adapter_prefix = format!("FRANKENSEARCH_HOST_ADAPTER={adapter_id}");
    if matches!(
        reason_code,
        "contract.schema.lagging"
            | "contract.schema.deprecated"
            | "contract.schema.too_new"
            | "adapter.identity.schema_version_mismatch"
    ) {
        format!("{adapter_prefix} {REPLAY_CONTRACT_SANITY_TESTS}")
    } else if reason_code.starts_with("adapter.") {
        format!("{adapter_prefix} {REPLAY_ADAPTER_CONFORMANCE_TESTS}")
    } else {
        format!("{adapter_prefix} cargo test -p frankensearch-core -- --nocapture")
    }
}

/// Classify an adapter's schema version against the current core version.
#[must_use]
pub const fn classify_version(adapter_version: u8) -> CompatibilityStatus {
    classify_version_against(
        TELEMETRY_SCHEMA_VERSION,
        adapter_version,
        MAX_SCHEMA_VERSION_LAG,
    )
}

/// Classify adapter schema compatibility against an explicit core version.
///
/// This is used for deterministic drift simulations in tests and rollout tooling.
#[must_use]
#[allow(clippy::comparison_chain)] // Can't use Ord::cmp() in const fn
pub const fn classify_version_against(
    core_schema_version: u8,
    adapter_version: u8,
    max_schema_version_lag: u8,
) -> CompatibilityStatus {
    if adapter_version == core_schema_version {
        CompatibilityStatus::Exact
    } else if adapter_version > core_schema_version {
        CompatibilityStatus::TooNew {
            ahead: adapter_version - core_schema_version,
        }
    } else {
        let lag = core_schema_version - adapter_version;
        if lag <= max_schema_version_lag {
            CompatibilityStatus::Compatible { lag }
        } else {
            CompatibilityStatus::Deprecated { lag }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SearchResult;
    use crate::host_adapter::{AdapterIdentity, AdapterLifecycleEvent};

    #[derive(Debug)]
    struct StubAdapter {
        identity: AdapterIdentity,
    }

    impl StubAdapter {
        fn with_version(version: u8) -> Self {
            Self::with_identity("test-adapter", "test-project", version)
        }

        fn with_identity(adapter_id: &str, host_project: &str, version: u8) -> Self {
            Self {
                identity: AdapterIdentity {
                    adapter_id: adapter_id.to_owned(),
                    adapter_version: "0.1.0".to_owned(),
                    host_project: host_project.to_owned(),
                    telemetry_schema_version: version,
                    redaction_policy_version: "v1".to_owned(),
                },
            }
        }
    }

    impl HostAdapter for StubAdapter {
        fn identity(&self) -> AdapterIdentity {
            self.identity.clone()
        }

        fn emit_telemetry(
            &self,
            _envelope: &crate::collectors::TelemetryEnvelope,
        ) -> SearchResult<()> {
            Ok(())
        }

        fn on_lifecycle_event(&self, _event: &AdapterLifecycleEvent) -> SearchResult<()> {
            Ok(())
        }
    }

    #[test]
    fn exact_version_passes() {
        let checker = ContractSanityChecker::default();
        let adapter = StubAdapter::with_version(TELEMETRY_SCHEMA_VERSION);
        let result = checker.check_adapter(&adapter);

        assert!(result.passed);
        assert_eq!(result.compatibility, CompatibilityStatus::Exact);
    }

    #[test]
    fn compatible_lagging_version_passes_with_warning() {
        // Only meaningful when TELEMETRY_SCHEMA_VERSION > 0
        if TELEMETRY_SCHEMA_VERSION == 0 {
            return;
        }
        let checker = ContractSanityChecker::default();
        let adapter = StubAdapter::with_version(TELEMETRY_SCHEMA_VERSION - 1);
        let result = checker.check_adapter(&adapter);

        // Should pass (within compatibility window) but have a lagging warning.
        assert!(result.passed);
        assert_eq!(
            result.compatibility,
            CompatibilityStatus::Compatible { lag: 1 }
        );
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "contract.schema.lagging")
        );
    }

    #[test]
    fn deprecated_version_fails() {
        if TELEMETRY_SCHEMA_VERSION <= MAX_SCHEMA_VERSION_LAG {
            // Can't test deprecated when version is too low.
            return;
        }
        let checker = ContractSanityChecker::default();
        let old_version = TELEMETRY_SCHEMA_VERSION - MAX_SCHEMA_VERSION_LAG - 1;
        let adapter = StubAdapter::with_version(old_version);
        let result = checker.check_adapter(&adapter);

        assert!(!result.passed);
        assert!(matches!(
            result.compatibility,
            CompatibilityStatus::Deprecated { .. }
        ));
    }

    #[test]
    fn too_new_version_fails() {
        let checker = ContractSanityChecker::default();
        let adapter = StubAdapter::with_version(TELEMETRY_SCHEMA_VERSION + 1);
        let result = checker.check_adapter(&adapter);

        assert!(!result.passed);
        assert_eq!(
            result.compatibility,
            CompatibilityStatus::TooNew { ahead: 1 }
        );
    }

    #[test]
    fn check_all_reports_summary() {
        let checker = ContractSanityChecker::default();
        let good = StubAdapter::with_version(TELEMETRY_SCHEMA_VERSION);
        let bad = StubAdapter::with_version(TELEMETRY_SCHEMA_VERSION + 5);
        let adapters: Vec<&dyn HostAdapter> = vec![&good, &bad];

        let report = checker.check_all(&adapters);
        assert_eq!(report.adapters_checked, 2);
        assert_eq!(report.adapters_passed, 1);
        assert!(!report.passed);
    }

    #[test]
    fn check_all_passes_when_all_exact() {
        let checker = ContractSanityChecker::default();
        let a1 = StubAdapter::with_version(TELEMETRY_SCHEMA_VERSION);
        let a2 = StubAdapter::with_version(TELEMETRY_SCHEMA_VERSION);
        let adapters: Vec<&dyn HostAdapter> = vec![&a1, &a2];

        let report = checker.check_all(&adapters);
        assert!(report.passed);
        assert_eq!(report.adapters_passed, 2);
    }

    #[test]
    fn classify_version_exact() {
        assert_eq!(
            classify_version(TELEMETRY_SCHEMA_VERSION),
            CompatibilityStatus::Exact
        );
    }

    #[test]
    fn classify_version_too_new() {
        assert_eq!(
            classify_version(TELEMETRY_SCHEMA_VERSION + 3),
            CompatibilityStatus::TooNew { ahead: 3 }
        );
    }

    #[test]
    fn classify_version_compatible() {
        if TELEMETRY_SCHEMA_VERSION == 0 {
            return;
        }
        assert_eq!(
            classify_version(TELEMETRY_SCHEMA_VERSION - 1),
            CompatibilityStatus::Compatible { lag: 1 }
        );
    }

    #[test]
    fn classify_version_against_supports_drift_simulation() {
        assert_eq!(
            classify_version_against(3, 3, 1),
            CompatibilityStatus::Exact
        );
        assert_eq!(
            classify_version_against(3, 2, 1),
            CompatibilityStatus::Compatible { lag: 1 }
        );
        assert_eq!(
            classify_version_against(3, 1, 1),
            CompatibilityStatus::Deprecated { lag: 2 }
        );
        assert_eq!(
            classify_version_against(3, 4, 1),
            CompatibilityStatus::TooNew { ahead: 1 }
        );
    }

    #[test]
    fn empty_adapter_id_fails_identity_check() {
        let checker = ContractSanityChecker::default();
        let adapter = StubAdapter {
            identity: AdapterIdentity {
                adapter_id: String::new(),
                adapter_version: "0.1.0".to_owned(),
                host_project: "test".to_owned(),
                telemetry_schema_version: TELEMETRY_SCHEMA_VERSION,
                redaction_policy_version: "v1".to_owned(),
            },
        };
        let result = checker.check_adapter(&adapter);
        assert!(!result.passed);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "adapter.identity.missing_adapter_id")
        );
    }

    #[test]
    fn wrong_redaction_policy_fails() {
        let checker = ContractSanityChecker::default();
        let adapter = StubAdapter {
            identity: AdapterIdentity {
                adapter_id: "test".to_owned(),
                adapter_version: "0.1.0".to_owned(),
                host_project: "test".to_owned(),
                telemetry_schema_version: TELEMETRY_SCHEMA_VERSION,
                redaction_policy_version: "v99".to_owned(),
            },
        };
        let result = checker.check_adapter(&adapter);
        assert!(!result.passed);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "adapter.identity.redaction_policy_mismatch")
        );
    }

    #[test]
    fn contract_report_serde_roundtrip() {
        let report = ContractSanityReport {
            core_schema_version: TELEMETRY_SCHEMA_VERSION,
            adapters_checked: 1,
            adapters_passed: 1,
            adapter_results: vec![AdapterContractResult {
                adapter_id: "test".to_owned(),
                host_project: "proj".to_owned(),
                adapter_schema_version: TELEMETRY_SCHEMA_VERSION,
                passed: true,
                compatibility: CompatibilityStatus::Exact,
                violations: vec![],
            }],
            passed: true,
        };
        let json = serde_json::to_string(&report).expect("serialize");
        let back: ContractSanityReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(report, back);
    }

    #[test]
    fn check_all_with_empty_list() {
        let checker = ContractSanityChecker::default();
        let adapters: Vec<&dyn HostAdapter> = vec![];
        let report = checker.check_all(&adapters);
        assert!(report.passed);
        assert_eq!(report.adapters_checked, 0);
    }

    #[test]
    fn diagnostics_are_deterministic_with_replay_commands() {
        let report = ContractSanityReport {
            core_schema_version: TELEMETRY_SCHEMA_VERSION,
            adapters_checked: 2,
            adapters_passed: 1,
            adapter_results: vec![
                AdapterContractResult {
                    adapter_id: "xf-host-adapter".to_owned(),
                    host_project: "xf".to_owned(),
                    adapter_schema_version: TELEMETRY_SCHEMA_VERSION + 1,
                    passed: false,
                    compatibility: CompatibilityStatus::TooNew { ahead: 1 },
                    violations: vec![ConformanceViolation {
                        code: "contract.schema.too_new".to_owned(),
                        field: "identity.telemetry_schema_version".to_owned(),
                        message: "too new".to_owned(),
                    }],
                },
                AdapterContractResult {
                    adapter_id: "cass-host-adapter".to_owned(),
                    host_project: "cass".to_owned(),
                    adapter_schema_version: TELEMETRY_SCHEMA_VERSION,
                    passed: false,
                    compatibility: CompatibilityStatus::Exact,
                    violations: vec![ConformanceViolation {
                        code: "adapter.identity.redaction_policy_mismatch".to_owned(),
                        field: "identity.redaction_policy_version".to_owned(),
                        message: "bad redaction policy".to_owned(),
                    }],
                },
            ],
            passed: false,
        };

        let diagnostics = report.diagnostics();
        assert_eq!(diagnostics.len(), 2);
        assert_eq!(diagnostics[0].adapter_id, "cass-host-adapter");
        assert_eq!(
            diagnostics[0].severity,
            ViolationSeverity::Error,
            "redaction mismatch is a hard violation"
        );
        assert!(
            diagnostics[0]
                .replay_command
                .contains("host_adapter::tests"),
            "adapter-level violations should replay host adapter conformance tests"
        );

        assert_eq!(diagnostics[1].adapter_id, "xf-host-adapter");
        assert_eq!(diagnostics[1].reason_code, "contract.schema.too_new");
        assert!(
            diagnostics[1]
                .replay_command
                .contains("contract_sanity::tests"),
            "schema drift violations should replay contract sanity tests"
        );
    }

    #[test]
    fn lagging_schema_mismatch_is_warning_in_diagnostics() {
        let report = ContractSanityReport {
            core_schema_version: TELEMETRY_SCHEMA_VERSION,
            adapters_checked: 1,
            adapters_passed: 1,
            adapter_results: vec![AdapterContractResult {
                adapter_id: "ops-host-adapter".to_owned(),
                host_project: "ops".to_owned(),
                adapter_schema_version: TELEMETRY_SCHEMA_VERSION.saturating_sub(1),
                passed: true,
                compatibility: CompatibilityStatus::Compatible { lag: 1 },
                violations: vec![
                    ConformanceViolation {
                        code: "adapter.identity.schema_version_mismatch".to_owned(),
                        field: "identity.telemetry_schema_version".to_owned(),
                        message: "expected schema mismatch warning".to_owned(),
                    },
                    ConformanceViolation {
                        code: "contract.schema.lagging".to_owned(),
                        field: "identity.telemetry_schema_version".to_owned(),
                        message: "lagging in window".to_owned(),
                    },
                ],
            }],
            passed: true,
        };

        let diagnostics = report.diagnostics();
        assert_eq!(diagnostics.len(), 2);
        assert!(
            diagnostics
                .iter()
                .all(|diag| diag.severity == ViolationSeverity::Warning)
        );
    }

    #[test]
    fn two_host_adapter_drift_scenario_emits_actionable_diagnostics() {
        let checker = ContractSanityChecker::default();
        let cass =
            StubAdapter::with_identity("cass-host-adapter", "cass", TELEMETRY_SCHEMA_VERSION);
        let xf = StubAdapter::with_identity("xf-host-adapter", "xf", TELEMETRY_SCHEMA_VERSION + 1);
        let adapters: Vec<&dyn HostAdapter> = vec![&cass, &xf];
        let report = checker.check_all(&adapters);

        assert!(!report.passed);
        let diagnostics = report.diagnostics();
        assert!(
            diagnostics
                .iter()
                .any(|diag| diag.reason_code == "contract.schema.too_new")
        );
        assert!(
            diagnostics
                .iter()
                .all(|diag| !diag.replay_command.is_empty())
        );
    }

    #[test]
    fn replay_command_mapping_uses_expected_harnesses() {
        let schema_cmd = replay_command_for_reason("contract.schema.too_new", "xf-host-adapter");
        assert!(schema_cmd.contains("contract_sanity::tests"));
        assert!(schema_cmd.contains("FRANKENSEARCH_HOST_ADAPTER=xf-host-adapter"));

        let adapter_cmd = replay_command_for_reason(
            "adapter.identity.redaction_policy_mismatch",
            "cass-host-adapter",
        );
        assert!(adapter_cmd.contains("host_adapter::tests"));
        assert!(adapter_cmd.contains("FRANKENSEARCH_HOST_ADAPTER=cass-host-adapter"));
    }
}
