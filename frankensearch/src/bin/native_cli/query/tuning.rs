//! Explicit full-mode CLI policy, applied to the original retained query.
//! Parsing is bounded and precedes model loading. Opted-in request overrides
//! derive a private query policy within startup limits, never mutate the server
//! policy, and cannot change source scope, models or the session's time budget.

use frankensearch_fusion::RrfConfig;
use serde::{Deserialize, Serialize};

use super::{Cx, Mode, Policy, Result, bad, cohort};

const MAX_POLICY_BYTES: usize = 8 * 1024;
const MAX_MULTIPLIER: usize = 100;
const MAX_BEAM: usize = 1_000_000;

/// Settings for full-mode queries only. Omitted fields retain native defaults.
/// Private fields ensure production callers use validated parsing. Serde is
/// used only for command input/output, never generation identity or disk state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FullQuery {
    candidate_multiplier: usize,
    fast_ef: Option<usize>,
    quality_ef: Option<usize>,
    rrf_k: f64,
    lexical_weight: f64,
    semantic_weight: f64,
    quality_weight: f32,
    /// Startup-only grant. Requests cannot grant this permission themselves.
    #[serde(skip_serializing_if = "is_false")]
    request_overrides: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// A partial full-mode request policy. Missing/null fields inherit the server;
/// in particular null cannot clear an explicit server beam or candidate limit.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FullQueryOverride {
    candidate_multiplier: Option<usize>,
    fast_ef: Option<usize>,
    quality_ef: Option<usize>,
    rrf_k: Option<f64>,
    lexical_weight: Option<f64>,
    semantic_weight: Option<f64>,
    quality_weight: Option<f32>,
    /// None inherits; false skips only the optional cross-encoder stage.
    rerank: Option<bool>,
    /// A positive window at most the window of the already loaded reranker.
    rerank_window: Option<usize>,
}

/// Require an object without first decoding through Value (which would erase
/// duplicate keys). Derived field decoding still rejects duplicates/unknowns.
pub(crate) fn deserialize_query_override<'de, D>(
    deserializer: D,
) -> std::result::Result<FullQueryOverride, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct Object;
    impl<'de> serde::de::Visitor<'de> for Object {
        type Value = FullQueryOverride;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a full-query override object")
        }

        fn visit_map<M>(self, map: M) -> std::result::Result<Self::Value, M::Error>
        where
            M: serde::de::MapAccess<'de>,
        {
            FullQueryOverride::deserialize(serde::de::value::MapAccessDeserializer::new(map))
        }
    }
    deserializer.deserialize_map(Object)
}

impl Default for FullQuery {
    fn default() -> Self {
        let rrf = RrfConfig::default();
        Self {
            candidate_multiplier: 3,
            fast_ef: None,
            quality_ef: None,
            rrf_k: rrf.k,
            lexical_weight: rrf.lexical_weight,
            semantic_weight: rrf.semantic_weight,
            quality_weight: 0.7,
            request_overrides: false,
        }
    }
}

impl FullQuery {
    pub(crate) fn parse(text: &str) -> Result<Self> {
        if text.len() > MAX_POLICY_BYTES {
            return Err(bad("full-query policy exceeds 8 KiB"));
        }
        if !text.trim_start().starts_with('{') {
            return Err(bad("full-query policy must be a JSON object"));
        }
        let policy: Self = serde_json::from_str(text)
            .map_err(|_| bad("invalid full-query policy JSON or unknown/duplicate field"))?;
        policy.validate()?;
        Ok(policy)
    }

    fn validate(&self) -> Result<()> {
        if !(1..=MAX_MULTIPLIER).contains(&self.candidate_multiplier) {
            return Err(bad("candidate_multiplier must be between 1 and 100"));
        }
        if [self.fast_ef, self.quality_ef]
            .into_iter()
            .flatten()
            .any(|beam| !(1..=MAX_BEAM).contains(&beam))
        {
            return Err(bad("graph beams must be null or between 1 and 1000000"));
        }
        if !self.quality_weight.is_finite() || !(0.0..=1.0).contains(&self.quality_weight) {
            return Err(bad(
                "quality_weight must be finite and between zero and one",
            ));
        }
        if !self.rrf_k.is_finite()
            || self.rrf_k < 0.0
            || !self.lexical_weight.is_finite()
            || self.lexical_weight <= 0.0
            || !self.semantic_weight.is_finite()
            || self.semantic_weight <= 0.0
        {
            return Err(bad(
                "RRF K must be finite and nonnegative; source weights finite and positive",
            ));
        }
        let first_rank = 1.0 / (self.rrf_k + 1.0);
        let maximum = first_rank * self.lexical_weight + first_rank * self.semantic_weight;
        if !maximum.is_finite() || maximum > f64::from(f32::MAX) {
            return Err(bad("full-query maximum fused score exceeds f32 precision"));
        }
        Ok(())
    }

    fn apply<'a>(&self, query: cohort::Progressive<'a>) -> Result<cohort::Progressive<'a>> {
        // Recheck at use as well: test/deserialized construction must not bypass
        // CLI bounds. Native setters also enforce their own state/cancel gates.
        self.validate()?;
        let rrf = RrfConfig {
            k: self.rrf_k,
            lexical_weight: self.lexical_weight,
            semantic_weight: self.semantic_weight,
            ..RrfConfig::default()
        };
        Ok(match query {
            cohort::Progressive::Single(query) => cohort::Progressive::Single(Box::new(
                (*query)
                    .with_fusion(rrf, self.quality_weight)?
                    .with_candidate_multiplier(self.candidate_multiplier)?
                    .with_beam_widths(self.fast_ef, self.quality_ef)?,
            )),
            cohort::Progressive::Sharded(query) => cohort::Progressive::Sharded(Box::new(
                (*query)
                    .with_fusion(rrf, self.quality_weight)?
                    .with_candidate_multiplier(self.candidate_multiplier)?
                    .with_beam_widths(self.fast_ef, self.quality_ef)?,
            )),
        })
    }
}

impl Policy {
    pub(crate) fn query_overrides_enabled(&self) -> bool {
        self.full_query
            .as_ref()
            .is_some_and(|policy| policy.request_overrides)
    }

    /// Derive one request's policy. The original timer, timeout ceiling and
    /// retained reranker Arc are preserved. A request may skip
    /// optional reranking or reduce its window within the server's maximum.
    /// Limits concern requested hits/beams, not RSS, graph visits or latency.
    pub(crate) fn for_query_override(
        &self,
        cx: &Cx,
        requested: &FullQueryOverride,
    ) -> Result<Self> {
        cx.checkpoint().map_err(|_| super::SearchError::Cancelled {
            phase: "native_cli.query.override".to_owned(),
            reason: "query tuning cancelled".to_owned(),
        })?;
        let base = self
            .full_query
            .as_ref()
            .filter(|policy| policy.request_overrides)
            .ok_or_else(|| bad("request tuning is disabled by the server full-query policy"))?;
        base.validate()?;
        let mut effective = base.clone();
        effective.candidate_multiplier = requested
            .candidate_multiplier
            .unwrap_or(base.candidate_multiplier);
        if effective.candidate_multiplier > base.candidate_multiplier {
            return Err(bad("request candidate multiplier exceeds the server limit"));
        }
        for (value, maximum, target) in [
            (requested.fast_ef, base.fast_ef, &mut effective.fast_ef),
            (
                requested.quality_ef,
                base.quality_ef,
                &mut effective.quality_ef,
            ),
        ] {
            if let Some(value) = value {
                if maximum.is_none_or(|maximum| value > maximum) {
                    return Err(bad(
                        "request graph beam requires and must not exceed a server beam limit",
                    ));
                }
                *target = Some(value);
            }
        }
        effective.rrf_k = requested.rrf_k.unwrap_or(base.rrf_k);
        effective.lexical_weight = requested.lexical_weight.unwrap_or(base.lexical_weight);
        effective.semantic_weight = requested.semantic_weight.unwrap_or(base.semantic_weight);
        effective.quality_weight = requested.quality_weight.unwrap_or(base.quality_weight);
        effective.validate()?;
        let rerank = self.request_reranker(requested)?;
        // The derived query is not itself a grant for another override.
        effective.request_overrides = false;
        Ok(Self {
            full_query: Some(effective),
            rerank,
            ..self.clone()
        })
    }

    fn request_reranker(&self, requested: &FullQueryOverride) -> Result<Option<super::Rerank>> {
        if requested.rerank == Some(false) {
            if requested.rerank_window.is_some() {
                return Err(bad("rerank_window cannot be combined with rerank=false"));
            }
            return Ok(None);
        }
        let Some(base) = &self.rerank else {
            if requested.rerank == Some(true) || requested.rerank_window.is_some() {
                return Err(bad("request reranking requires a server-loaded reranker"));
            }
            return Ok(None);
        };
        let window = requested.rerank_window.unwrap_or(base.window);
        if !(1..=1_000).contains(&base.window) || window == 0 || window > base.window {
            return Err(bad(
                "request rerank window must be positive and within the server limit",
            ));
        }
        Ok(Some(super::Rerank {
            model: base.model.clone(),
            window,
        }))
    }

    /// Prepare the same full-mode query for buffered and streaming execution.
    /// Attach retained-source reranking first so candidate tuning accounts for
    /// its larger window. The complete scope and all original owners survive.
    pub(crate) fn progressive<'a>(
        &'a self,
        scoped: &'a cohort::Prepared<'_>,
        cx: &'a Cx,
        text: &'a str,
        limit: usize,
    ) -> Result<cohort::Progressive<'a>> {
        let query = match &self.rerank {
            Some(rerank) => scoped.progressive_with_reranker_on_pool(
                cx,
                text,
                limit,
                rerank.model.as_ref(),
                rerank.window,
                self.query_pool(),
            )?,
            None => scoped.progressive_on_pool(cx, text, limit, self.query_pool())?,
        };
        match &self.full_query {
            Some(policy) => policy.apply(query),
            None => Ok(query),
        }
    }

    pub(crate) fn annotate_full_query(&self, mode: Mode, payload: &mut serde_json::Value) {
        if mode == Mode::Full
            && let Some(policy) = &self.full_query
        {
            // These are requested settings, not measured work/recall or model
            // provenance. Native retrieval retains its beam floors and caps.
            payload["full_query"] = serde_json::json!(policy);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn server() -> Policy {
        Policy {
            full_query: Some(FullQuery::parse(
                r#"{"request_overrides":true,"candidate_multiplier":6,"fast_ef":128,"quality_ef":256,"quality_weight":0.4}"#,
            ).unwrap()),
            ..Policy::default()
        }
    }

    #[test]
    fn partial_overrides_inherit_without_mutating_or_delegating_the_server_policy() {
        let cx = Cx::for_testing();
        let base = server();
        let original = base.full_query.clone();
        let request: FullQueryOverride = serde_json::from_str(
            r#"{"candidate_multiplier":2,"fast_ef":64,"quality_ef":null,"lexical_weight":2.5}"#,
        )
        .unwrap();
        let effective = base.for_query_override(&cx, &request).unwrap();
        let values = effective.full_query.as_ref().unwrap();
        assert_eq!(values.candidate_multiplier, 2);
        assert_eq!(values.fast_ef, Some(64));
        assert_eq!(values.quality_ef, Some(256));
        assert_eq!(values.quality_weight, 0.4);
        assert_eq!(values.lexical_weight, 2.5);
        assert_eq!(base.full_query, original);
        assert!(!effective.query_overrides_enabled());
        assert!(
            effective
                .for_query_override(&cx, &FullQueryOverride::default())
                .is_err()
        );
        let inherited = base
            .for_query_override(&cx, &FullQueryOverride::default())
            .unwrap();
        let mut expected = original.unwrap();
        expected.request_overrides = false;
        assert_eq!(inherited.full_query, Some(expected));
    }

    #[test]
    fn request_limits_and_numeric_validation_cannot_be_bypassed() {
        let cx = Cx::for_testing();
        let base = server();
        for text in [
            r#"{"candidate_multiplier":0}"#,
            r#"{"candidate_multiplier":7}"#,
            r#"{"fast_ef":0}"#,
            r#"{"fast_ef":129}"#,
            r#"{"quality_ef":257}"#,
            r#"{"quality_weight":1.1}"#,
            r#"{"rrf_k":-1}"#,
            r#"{"lexical_weight":0}"#,
            r#"{"semantic_weight":1e300}"#,
        ] {
            let request: FullQueryOverride = serde_json::from_str(text).unwrap();
            assert!(base.for_query_override(&cx, &request).is_err(), "{text}");
        }
        for request in [
            FullQueryOverride {
                quality_weight: Some(f32::NAN),
                ..Default::default()
            },
            FullQueryOverride {
                rrf_k: Some(f64::INFINITY),
                ..Default::default()
            },
            FullQueryOverride {
                lexical_weight: Some(f64::NAN),
                ..Default::default()
            },
        ] {
            assert!(base.for_query_override(&cx, &request).is_err());
        }
        let no_beam = Policy {
            full_query: Some(FullQuery::parse(r#"{"request_overrides":true}"#).unwrap()),
            ..Policy::default()
        };
        assert!(
            no_beam
                .for_query_override(
                    &cx,
                    &FullQueryOverride {
                        fast_ef: Some(1),
                        ..Default::default()
                    }
                )
                .is_err()
        );
        for quality_weight in [0.0, 1.0] {
            assert!(
                base.for_query_override(
                    &cx,
                    &FullQueryOverride {
                        quality_weight: Some(quality_weight),
                        ..Default::default()
                    }
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn overrides_preserve_the_original_timer_and_timeout_ceiling() {
        use asupersync::time::{TimerDriverHandle, VirtualClock};
        use std::future::Future;
        use std::sync::Arc;
        use std::task::{Context, Poll, Waker};

        let cx = Cx::for_testing();
        let clock = Arc::new(VirtualClock::new());
        let base = Policy {
            maximum_ms: Some(10),
            timer: Some(TimerDriverHandle::with_virtual_clock(Arc::clone(&clock))),
            ..server()
        };
        let effective = base
            .for_query_override(&cx, &FullQueryOverride::default())
            .unwrap();
        assert_eq!(effective.effective_ms(Some(1000)).unwrap(), Some(10));
        assert_eq!(effective.effective_ms(Some(5)).unwrap(), Some(5));
        let deadline = effective.start(&cx, Some(1000)).unwrap().unwrap();
        clock.advance(10_000_000);
        let mut pending = Box::pin(super::super::within(&cx, Some(&deadline), async { Ok(()) }));
        let Poll::Ready(Err(error)) = pending
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
        else {
            panic!("original clock expiry must be observed");
        };
        assert_eq!(super::super::failure(error.as_ref())["status"], "timed_out");
        assert!(!cx.is_cancel_requested());
    }

    #[test]
    fn missing_permission_is_not_implicit_and_cancellation_keeps_its_type() {
        let cx = Cx::for_testing();
        let request = FullQueryOverride::default();
        for base in [
            Policy::default(),
            Policy {
                full_query: Some(FullQuery::default()),
                ..Policy::default()
            },
        ] {
            assert!(!base.query_overrides_enabled());
            assert!(base.for_query_override(&cx, &request).is_err());
            cx.set_cancel_requested(true);
            let error = base.for_query_override(&cx, &request).err().unwrap();
            assert_eq!(super::super::failure(error.as_ref())["status"], "cancelled");
            cx.set_cancel_requested(false);
        }
        for text in ["null", "[]", "[3,null,null,60,1,1,0.7,true]"] {
            assert!(FullQuery::parse(text).is_err());
        }
    }
}
