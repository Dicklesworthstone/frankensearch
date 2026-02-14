//! Observability contract lint rules (bd-tn1o).
//!
//! Validates evidence records and streams against the canonical
//! evidence-ledger schema checklist. Used by CI gates and release
//! gate (bd-ehuk) to prevent semantic drift across adaptive components.

use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::decision_plane::{EvidenceEventType, EvidenceRecord, ReasonCode, Severity};

/// Lint rule identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LintRuleId {
    /// Evidence record missing required field.
    Obs001,
    /// Reason code format violation.
    Obs002,
    /// Reason code not in registry.
    Obs003,
    /// Component emits no evidence records.
    Obs004,
    /// Decision event missing `expected_loss`.
    Obs005,
    /// State transition without evidence context.
    Obs006,
    /// Reason human text too long.
    Obs007,
    /// Severity contradicts event type.
    Obs008,
    /// Duplicate `event_id` in stream.
    Obs009,
    /// Evidence stream not timestamp-ordered.
    Obs010,
}

impl LintRuleId {
    /// Machine-stable string for the rule ID.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Obs001 => "OBS-001",
            Self::Obs002 => "OBS-002",
            Self::Obs003 => "OBS-003",
            Self::Obs004 => "OBS-004",
            Self::Obs005 => "OBS-005",
            Self::Obs006 => "OBS-006",
            Self::Obs007 => "OBS-007",
            Self::Obs008 => "OBS-008",
            Self::Obs009 => "OBS-009",
            Self::Obs010 => "OBS-010",
        }
    }
}

impl fmt::Display for LintRuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Severity of a lint finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LintSeverity {
    Info,
    Warning,
    Error,
}

impl fmt::Display for LintSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => f.write_str("info"),
            Self::Warning => f.write_str("warning"),
            Self::Error => f.write_str("error"),
        }
    }
}

/// A single lint finding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintFinding {
    pub rule_id: LintRuleId,
    pub severity: LintSeverity,
    pub message: String,
    /// Source component or context.
    pub source: String,
}

impl fmt::Display for LintFinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} ({}): {}",
            self.rule_id, self.severity, self.source, self.message
        )
    }
}

/// Aggregated lint report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintReport {
    pub findings: Vec<LintFinding>,
}

impl LintReport {
    /// Create an empty report.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            findings: Vec::new(),
        }
    }

    /// Whether the report has any error-level findings.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.findings
            .iter()
            .any(|f| f.severity == LintSeverity::Error)
    }

    /// Count of findings at or above a given severity.
    #[must_use]
    pub fn count_at_severity(&self, min_severity: LintSeverity) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity >= min_severity)
            .count()
    }

    /// All findings pass (no errors).
    #[must_use]
    pub fn all_pass(&self) -> bool {
        !self.has_errors()
    }

    fn add(&mut self, finding: LintFinding) {
        self.findings.push(finding);
    }
}

impl Default for LintReport {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for LintReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.findings.is_empty() {
            return write!(f, "Observability lint: 0 findings (PASS)");
        }
        let errors = self.count_at_severity(LintSeverity::Error);
        let warnings = self.count_at_severity(LintSeverity::Warning) - errors;
        let status = if self.has_errors() { "FAIL" } else { "PASS" };
        writeln!(
            f,
            "Observability lint: {} findings ({errors} errors, {warnings} warnings) ({status})",
            self.findings.len()
        )?;
        for finding in &self.findings {
            writeln!(f, "  {finding}")?;
        }
        Ok(())
    }
}

/// Set of all known reason codes from the registry.
fn registered_reason_codes() -> HashSet<&'static str> {
    HashSet::from([
        // Decision Plane
        ReasonCode::DECISION_SKIP_FAST_ONLY,
        ReasonCode::DECISION_SKIP_CIRCUIT_OPEN,
        ReasonCode::DECISION_SKIP_BUDGET_EXHAUSTED,
        ReasonCode::DECISION_SKIP_HIGH_LOSS,
        ReasonCode::DECISION_SKIP_EMPTY_QUERY,
        ReasonCode::DECISION_REFINE_NOMINAL,
        ReasonCode::DECISION_PROBE_SENT,
        ReasonCode::DECISION_PROBE_SUCCESS,
        ReasonCode::DECISION_PROBE_FAILURE,
        // Circuit Breaker
        ReasonCode::CIRCUIT_OPEN_FAILURES,
        ReasonCode::CIRCUIT_OPEN_LATENCY,
        ReasonCode::CIRCUIT_CLOSE_RECOVERY,
        // Calibration
        ReasonCode::CALIBRATION_FALLBACK_DATA,
        ReasonCode::CALIBRATION_FALLBACK_DRIFT,
        ReasonCode::CALIBRATION_FALLBACK_ERROR,
        ReasonCode::CALIBRATION_FALLBACK_MODEL,
        ReasonCode::CALIBRATION_TRAINED,
        ReasonCode::CALIBRATION_RESET,
        // Adaptive Fusion
        ReasonCode::FUSION_BLEND_ADJUSTED,
        ReasonCode::FUSION_RRF_K_ADJUSTED,
        ReasonCode::FUSION_FALLBACK_DEFAULT,
        // Sequential Testing
        ReasonCode::TESTING_REJECT,
        ReasonCode::TESTING_CONTINUE,
        ReasonCode::TESTING_RESET,
        // Conformal
        ReasonCode::CONFORMAL_VALID,
        ReasonCode::CONFORMAL_VIOLATION,
        ReasonCode::CONFORMAL_UPDATE,
        // Feedback
        ReasonCode::FEEDBACK_BOOST_UPDATED,
        ReasonCode::FEEDBACK_BOOST_DECAYED,
    ])
}

/// Validate the format of a reason code string.
fn validate_reason_code_format(code: &str) -> bool {
    let parts: Vec<&str> = code.split('.').collect();
    if parts.len() != 3 {
        return false;
    }
    parts.iter().all(|p| {
        !p.is_empty()
            && p.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
    })
}

/// Minimum severity required for a given event type (OBS-008).
const fn minimum_severity_for(event_type: EvidenceEventType) -> Severity {
    match event_type {
        EvidenceEventType::Degradation | EvidenceEventType::Alert => Severity::Warn,
        EvidenceEventType::Decision
        | EvidenceEventType::Transition
        | EvidenceEventType::ReplayMarker => Severity::Info,
    }
}

/// Check if a severity meets or exceeds the minimum.
fn severity_meets_minimum(actual: Severity, minimum: Severity) -> bool {
    let rank = |s: Severity| match s {
        Severity::Info => 0,
        Severity::Warn => 1,
        Severity::Error => 2,
    };
    rank(actual) >= rank(minimum)
}

/// Maximum allowed length for `reason_human` text (OBS-007).
const MAX_REASON_HUMAN_LEN: usize = 200;

/// Lint a single evidence record.
#[must_use]
pub fn lint_record(record: &EvidenceRecord, source: &str) -> Vec<LintFinding> {
    let mut findings = Vec::new();

    // OBS-001: Required fields check
    if record.source_component.is_empty() {
        findings.push(LintFinding {
            rule_id: LintRuleId::Obs001,
            severity: LintSeverity::Error,
            message: "source_component is empty".to_owned(),
            source: source.to_owned(),
        });
    }
    if record.reason_human.is_empty() {
        findings.push(LintFinding {
            rule_id: LintRuleId::Obs001,
            severity: LintSeverity::Error,
            message: "reason_human is empty".to_owned(),
            source: source.to_owned(),
        });
    }

    // OBS-002: Reason code format
    if !validate_reason_code_format(&record.reason_code.0) {
        findings.push(LintFinding {
            rule_id: LintRuleId::Obs002,
            severity: LintSeverity::Error,
            message: format!(
                "reason code '{}' does not match namespace.subject.detail pattern",
                record.reason_code.0
            ),
            source: source.to_owned(),
        });
    }

    // OBS-003: Reason code in registry
    let registry = registered_reason_codes();
    if !registry.contains(record.reason_code.0.as_str()) {
        findings.push(LintFinding {
            rule_id: LintRuleId::Obs003,
            severity: LintSeverity::Error,
            message: format!(
                "reason code '{}' not found in ReasonCode registry",
                record.reason_code.0
            ),
            source: source.to_owned(),
        });
    }

    // OBS-005: Decision without expected_loss
    if record.event_type == EvidenceEventType::Decision && record.expected_loss.is_none() {
        findings.push(LintFinding {
            rule_id: LintRuleId::Obs005,
            severity: LintSeverity::Warning,
            message: "decision event without expected_loss field".to_owned(),
            source: source.to_owned(),
        });
    }

    // OBS-007: Long reason text
    if record.reason_human.len() > MAX_REASON_HUMAN_LEN {
        findings.push(LintFinding {
            rule_id: LintRuleId::Obs007,
            severity: LintSeverity::Info,
            message: format!(
                "reason_human text is {} chars (max {})",
                record.reason_human.len(),
                MAX_REASON_HUMAN_LEN
            ),
            source: source.to_owned(),
        });
    }

    // OBS-008: Severity consistency
    let min_severity = minimum_severity_for(record.event_type);
    if !severity_meets_minimum(record.severity, min_severity) {
        findings.push(LintFinding {
            rule_id: LintRuleId::Obs008,
            severity: LintSeverity::Error,
            message: format!(
                "{} event must have severity >= {min_severity}, got {}",
                record.event_type, record.severity
            ),
            source: source.to_owned(),
        });
    }

    findings
}

/// Lint a stream of evidence records.
#[must_use]
pub fn lint_stream(records: &[(String, EvidenceRecord)]) -> LintReport {
    let mut report = LintReport::new();
    let mut seen_event_ids: HashSet<String> = HashSet::new();

    for (event_id, record) in records {
        // OBS-009: Duplicate event_id
        if !seen_event_ids.insert(event_id.clone()) {
            report.add(LintFinding {
                rule_id: LintRuleId::Obs009,
                severity: LintSeverity::Error,
                message: format!("duplicate event_id: {event_id}"),
                source: record.source_component.clone(),
            });
        }

        for finding in lint_record(record, &record.source_component) {
            report.add(finding);
        }
    }

    report
}

/// Check that a set of expected components all appear in the evidence stream.
#[must_use]
pub fn lint_component_coverage(
    records: &[(String, EvidenceRecord)],
    expected_components: &[&str],
) -> LintReport {
    let mut report = LintReport::new();
    let seen: HashSet<&str> = records
        .iter()
        .map(|(_, r)| r.source_component.as_str())
        .collect();

    for component in expected_components {
        if !seen.contains(component) {
            report.add(LintFinding {
                rule_id: LintRuleId::Obs004,
                severity: LintSeverity::Warning,
                message: format!("component '{component}' emitted no evidence records"),
                source: (*component).to_owned(),
            });
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use crate::decision_plane::{LossVector, PipelineState};

    use super::*;

    fn make_record(
        event_type: EvidenceEventType,
        reason_code: &str,
        severity: Severity,
        source: &str,
    ) -> EvidenceRecord {
        EvidenceRecord {
            event_type,
            reason_code: ReasonCode::from(reason_code),
            reason_human: "test reason".to_owned(),
            severity,
            pipeline_state: PipelineState::Nominal,
            action: None,
            expected_loss: None,
            query_class: None,
            source_component: source.to_owned(),
        }
    }

    #[test]
    fn obs001_empty_source_component() {
        let record = make_record(
            EvidenceEventType::Decision,
            ReasonCode::DECISION_REFINE_NOMINAL,
            Severity::Info,
            "",
        );
        let findings = lint_record(&record, "test");
        assert!(findings.iter().any(|f| f.rule_id == LintRuleId::Obs001));
    }

    #[test]
    fn obs001_empty_reason_human() {
        let mut record = make_record(
            EvidenceEventType::Decision,
            ReasonCode::DECISION_REFINE_NOMINAL,
            Severity::Info,
            "decision_plane",
        );
        record.reason_human = String::new();
        let findings = lint_record(&record, "test");
        assert!(findings.iter().any(|f| f.rule_id == LintRuleId::Obs001));
    }

    #[test]
    fn obs002_invalid_format() {
        let record = make_record(
            EvidenceEventType::Decision,
            "bad_format",
            Severity::Info,
            "decision_plane",
        );
        let findings = lint_record(&record, "test");
        assert!(findings.iter().any(|f| f.rule_id == LintRuleId::Obs002));
    }

    #[test]
    fn obs002_valid_format_passes() {
        let record = make_record(
            EvidenceEventType::Decision,
            ReasonCode::DECISION_REFINE_NOMINAL,
            Severity::Info,
            "decision_plane",
        );
        let findings = lint_record(&record, "test");
        assert!(!findings.iter().any(|f| f.rule_id == LintRuleId::Obs002));
    }

    #[test]
    fn obs003_unregistered_code() {
        let record = make_record(
            EvidenceEventType::Decision,
            "custom.made_up.code",
            Severity::Info,
            "decision_plane",
        );
        let findings = lint_record(&record, "test");
        assert!(findings.iter().any(|f| f.rule_id == LintRuleId::Obs003));
    }

    #[test]
    fn obs003_registered_code_passes() {
        let record = make_record(
            EvidenceEventType::Decision,
            ReasonCode::CIRCUIT_OPEN_FAILURES,
            Severity::Info,
            "circuit_breaker",
        );
        let findings = lint_record(&record, "test");
        assert!(!findings.iter().any(|f| f.rule_id == LintRuleId::Obs003));
    }

    #[test]
    fn obs004_missing_component_coverage() {
        let records = vec![(
            "evt1".to_owned(),
            make_record(
                EvidenceEventType::Decision,
                ReasonCode::DECISION_REFINE_NOMINAL,
                Severity::Info,
                "decision_plane",
            ),
        )];
        let report = lint_component_coverage(
            &records,
            &["decision_plane", "circuit_breaker", "calibrator"],
        );
        let obs004: Vec<_> = report
            .findings
            .iter()
            .filter(|f| f.rule_id == LintRuleId::Obs004)
            .collect();
        assert_eq!(obs004.len(), 2);
        assert!(obs004.iter().any(|f| f.source == "circuit_breaker"));
        assert!(obs004.iter().any(|f| f.source == "calibrator"));
    }

    #[test]
    fn obs005_decision_without_expected_loss() {
        let record = make_record(
            EvidenceEventType::Decision,
            ReasonCode::DECISION_REFINE_NOMINAL,
            Severity::Info,
            "decision_plane",
        );
        let findings = lint_record(&record, "test");
        assert!(findings.iter().any(|f| f.rule_id == LintRuleId::Obs005));
    }

    #[test]
    fn obs005_decision_with_expected_loss_passes() {
        let mut record = make_record(
            EvidenceEventType::Decision,
            ReasonCode::DECISION_REFINE_NOMINAL,
            Severity::Info,
            "decision_plane",
        );
        record.expected_loss = Some(LossVector {
            quality: 0.1,
            latency: 0.2,
            resource: 0.3,
        });
        let findings = lint_record(&record, "test");
        assert!(!findings.iter().any(|f| f.rule_id == LintRuleId::Obs005));
    }

    #[test]
    fn obs007_long_reason_text() {
        let mut record = make_record(
            EvidenceEventType::Decision,
            ReasonCode::DECISION_REFINE_NOMINAL,
            Severity::Info,
            "decision_plane",
        );
        record.reason_human = "a".repeat(201);
        let findings = lint_record(&record, "test");
        assert!(findings.iter().any(|f| f.rule_id == LintRuleId::Obs007));
    }

    #[test]
    fn obs007_max_length_passes() {
        let mut record = make_record(
            EvidenceEventType::Decision,
            ReasonCode::DECISION_REFINE_NOMINAL,
            Severity::Info,
            "decision_plane",
        );
        record.reason_human = "a".repeat(200);
        let findings = lint_record(&record, "test");
        assert!(!findings.iter().any(|f| f.rule_id == LintRuleId::Obs007));
    }

    #[test]
    fn obs008_degradation_with_info_severity() {
        let record = make_record(
            EvidenceEventType::Degradation,
            ReasonCode::DECISION_SKIP_CIRCUIT_OPEN,
            Severity::Info,
            "decision_plane",
        );
        let findings = lint_record(&record, "test");
        assert!(findings.iter().any(|f| f.rule_id == LintRuleId::Obs008));
    }

    #[test]
    fn obs008_degradation_with_warn_passes() {
        let record = make_record(
            EvidenceEventType::Degradation,
            ReasonCode::DECISION_SKIP_CIRCUIT_OPEN,
            Severity::Warn,
            "decision_plane",
        );
        let findings = lint_record(&record, "test");
        assert!(!findings.iter().any(|f| f.rule_id == LintRuleId::Obs008));
    }

    #[test]
    fn obs008_alert_with_error_passes() {
        let record = make_record(
            EvidenceEventType::Alert,
            ReasonCode::CIRCUIT_OPEN_FAILURES,
            Severity::Error,
            "circuit_breaker",
        );
        let findings = lint_record(&record, "test");
        assert!(!findings.iter().any(|f| f.rule_id == LintRuleId::Obs008));
    }

    #[test]
    fn obs009_duplicate_event_ids() {
        let records = vec![
            (
                "dup123".to_owned(),
                make_record(
                    EvidenceEventType::Decision,
                    ReasonCode::DECISION_REFINE_NOMINAL,
                    Severity::Info,
                    "decision_plane",
                ),
            ),
            (
                "dup123".to_owned(),
                make_record(
                    EvidenceEventType::Transition,
                    ReasonCode::CIRCUIT_CLOSE_RECOVERY,
                    Severity::Info,
                    "circuit_breaker",
                ),
            ),
        ];
        let report = lint_stream(&records);
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.rule_id == LintRuleId::Obs009)
        );
    }

    #[test]
    fn obs009_unique_event_ids_pass() {
        let records = vec![
            (
                "evt1".to_owned(),
                make_record(
                    EvidenceEventType::Decision,
                    ReasonCode::DECISION_REFINE_NOMINAL,
                    Severity::Info,
                    "decision_plane",
                ),
            ),
            (
                "evt2".to_owned(),
                make_record(
                    EvidenceEventType::Transition,
                    ReasonCode::CIRCUIT_CLOSE_RECOVERY,
                    Severity::Info,
                    "circuit_breaker",
                ),
            ),
        ];
        let report = lint_stream(&records);
        assert!(
            !report
                .findings
                .iter()
                .any(|f| f.rule_id == LintRuleId::Obs009)
        );
    }

    #[test]
    fn lint_report_display_format() {
        let mut report = LintReport::new();
        report.add(LintFinding {
            rule_id: LintRuleId::Obs001,
            severity: LintSeverity::Error,
            message: "test error".to_owned(),
            source: "test".to_owned(),
        });
        report.add(LintFinding {
            rule_id: LintRuleId::Obs007,
            severity: LintSeverity::Info,
            message: "test info".to_owned(),
            source: "test".to_owned(),
        });
        let output = format!("{report}");
        assert!(output.contains("FAIL"));
        assert!(output.contains("OBS-001"));
        assert!(output.contains("OBS-007"));
    }

    #[test]
    fn empty_report_displays_pass() {
        let report = LintReport::new();
        let output = format!("{report}");
        assert!(output.contains("PASS"));
        assert!(output.contains("0 findings"));
    }

    #[test]
    fn lint_report_serialization_roundtrip() {
        let mut report = LintReport::new();
        report.add(LintFinding {
            rule_id: LintRuleId::Obs002,
            severity: LintSeverity::Error,
            message: "bad format".to_owned(),
            source: "test".to_owned(),
        });
        let json = serde_json::to_string(&report).unwrap();
        let decoded: LintReport = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.findings.len(), 1);
        assert_eq!(decoded.findings[0].rule_id, LintRuleId::Obs002);
    }

    #[test]
    fn valid_record_produces_no_errors() {
        let mut record = make_record(
            EvidenceEventType::Decision,
            ReasonCode::DECISION_REFINE_NOMINAL,
            Severity::Info,
            "decision_plane",
        );
        record.expected_loss = Some(LossVector {
            quality: 0.0,
            latency: 0.0,
            resource: 0.0,
        });
        let findings = lint_record(&record, "test");
        let errors: Vec<_> = findings
            .iter()
            .filter(|f| f.severity == LintSeverity::Error)
            .collect();
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    #[test]
    fn all_registered_codes_have_valid_format() {
        for code in registered_reason_codes() {
            assert!(
                validate_reason_code_format(code),
                "registered code '{code}' fails format validation"
            );
        }
    }

    #[test]
    fn reason_code_format_rejects_bad_inputs() {
        assert!(!validate_reason_code_format(""));
        assert!(!validate_reason_code_format("single"));
        assert!(!validate_reason_code_format("two.parts"));
        assert!(!validate_reason_code_format("four.parts.too.many"));
        assert!(!validate_reason_code_format("UPPER.case.bad"));
        assert!(!validate_reason_code_format("has space.in.code"));
    }

    #[test]
    fn lint_stream_aggregates_findings() {
        let records = vec![
            (
                "evt1".to_owned(),
                make_record(
                    EvidenceEventType::Degradation,
                    ReasonCode::DECISION_SKIP_CIRCUIT_OPEN,
                    Severity::Info,
                    "decision_plane",
                ),
            ),
            (
                "evt2".to_owned(),
                make_record(
                    EvidenceEventType::Decision,
                    "invalid_code",
                    Severity::Info,
                    "",
                ),
            ),
        ];
        let report = lint_stream(&records);
        assert!(report.has_errors());
        assert!(report.findings.len() >= 3);
    }

    #[test]
    fn count_at_severity_works() {
        let mut report = LintReport::new();
        report.add(LintFinding {
            rule_id: LintRuleId::Obs001,
            severity: LintSeverity::Error,
            message: "err".to_owned(),
            source: "t".to_owned(),
        });
        report.add(LintFinding {
            rule_id: LintRuleId::Obs005,
            severity: LintSeverity::Warning,
            message: "warn".to_owned(),
            source: "t".to_owned(),
        });
        report.add(LintFinding {
            rule_id: LintRuleId::Obs007,
            severity: LintSeverity::Info,
            message: "info".to_owned(),
            source: "t".to_owned(),
        });
        assert_eq!(report.count_at_severity(LintSeverity::Error), 1);
        assert_eq!(report.count_at_severity(LintSeverity::Warning), 2);
        assert_eq!(report.count_at_severity(LintSeverity::Info), 3);
    }
}
