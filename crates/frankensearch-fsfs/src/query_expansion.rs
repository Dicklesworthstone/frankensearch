//! LLM-powered query expansion for fsfs search.
//!
//! When the `--expand` flag is set, this module generates additional query
//! variants using an LLM backend (Anthropic Claude or OpenAI) and returns
//! them for parallel search execution. Results from all queries are later
//! merged via reciprocal rank fusion in the main search pipeline.
//!
//! Expansion strategies:
//! - **Keyword**: extract key terms and add synonyms/related terms
//! - **Semantic**: rephrase the query for semantic matching
//! - **HyDE**: generate a hypothetical answer snippet
//!
//! The feature is entirely optional and off by default.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Timeout for LLM API calls.
const LLM_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// Maximum tokens for expansion responses.
const MAX_EXPANSION_TOKENS: u32 = 256;

/// Supported LLM backend providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmBackend {
    Anthropic,
    OpenAi,
}

/// A single expanded query variant with metadata about its origin.
#[derive(Debug, Clone, Serialize)]
pub struct ExpandedQuery {
    pub text: String,
    pub strategy: ExpansionStrategy,
}

/// The expansion strategy that produced a query variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ExpansionStrategy {
    /// Original user query (always included).
    Original,
    /// Keyword extraction with synonyms and related terms.
    Keyword,
    /// Semantic rephrasing for embedding-based retrieval.
    Semantic,
    /// Hypothetical Document Embedding -- a synthetic answer snippet.
    HyDE,
}

impl ExpansionStrategy {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::Keyword => "keyword",
            Self::Semantic => "semantic",
            Self::HyDE => "hyde",
        }
    }
}

/// Result of the query expansion process.
#[derive(Debug, Clone)]
pub struct ExpansionResult {
    /// All queries to search (original + expansions).
    pub queries: Vec<ExpandedQuery>,
    /// Which backend was used, if any.
    pub backend_used: Option<LlmBackend>,
    /// Wall-clock time spent on expansion.
    pub elapsed_ms: u64,
}

/// Detect available LLM backend from environment variables.
#[must_use]
fn detect_backend(env_map: &HashMap<String, String>) -> Option<(LlmBackend, String)> {
    if let Some(key) = env_map.get("ANTHROPIC_API_KEY") {
        if !key.trim().is_empty() {
            return Some((LlmBackend::Anthropic, key.clone()));
        }
    }
    if let Some(key) = env_map.get("OPENAI_API_KEY") {
        if !key.trim().is_empty() {
            return Some((LlmBackend::OpenAi, key.clone()));
        }
    }
    None
}

/// Expand a search query using an LLM backend.
///
/// Returns the original query plus up to 3 additional variants. If no API key
/// is available or the LLM call fails, returns only the original query with
/// a warning.
#[must_use]
pub fn expand_query(query: &str, env_map: &HashMap<String, String>) -> ExpansionResult {
    let start = Instant::now();
    let original = ExpandedQuery {
        text: query.to_owned(),
        strategy: ExpansionStrategy::Original,
    };

    let Some((backend, api_key)) = detect_backend(env_map) else {
        warn!(
            "fsfs --expand: no ANTHROPIC_API_KEY or OPENAI_API_KEY found; \
             continuing with original query only"
        );
        return ExpansionResult {
            queries: vec![original],
            backend_used: None,
            elapsed_ms: elapsed_ms_since(start),
        };
    };

    info!(
        backend = ?backend,
        query = query,
        "fsfs query expansion: calling LLM for query variants"
    );

    let prompt = build_expansion_prompt(query);

    let result = match backend {
        LlmBackend::Anthropic => call_anthropic(&api_key, &prompt),
        LlmBackend::OpenAi => call_openai(&api_key, &prompt),
    };

    match result {
        Ok(raw_response) => {
            let mut queries = vec![original];
            let parsed = parse_expansion_response(&raw_response);
            queries.extend(parsed);
            let elapsed = elapsed_ms_since(start);
            info!(
                backend = ?backend,
                expansion_count = queries.len() - 1,
                elapsed_ms = elapsed,
                "fsfs query expansion completed"
            );
            ExpansionResult {
                queries,
                backend_used: Some(backend),
                elapsed_ms: elapsed,
            }
        }
        Err(error) => {
            warn!(
                backend = ?backend,
                error = %error,
                "fsfs --expand: LLM call failed; continuing with original query only"
            );
            ExpansionResult {
                queries: vec![original],
                backend_used: Some(backend),
                elapsed_ms: elapsed_ms_since(start),
            }
        }
    }
}

/// Build the system+user prompt for query expansion.
fn build_expansion_prompt(query: &str) -> String {
    format!(
        "You are a search query expansion assistant. Given a user's search query, \
         generate exactly 3 alternative query formulations to improve search recall.\n\n\
         For each, output ONE line in the exact format shown (no numbering, no extra text):\n\
         KEYWORD: <extracted key terms plus synonyms and related terms>\n\
         SEMANTIC: <rephrased query optimized for semantic/embedding search>\n\
         HYDE: <a short hypothetical document snippet that would answer this query>\n\n\
         Rules:\n\
         - Each line must start with KEYWORD:, SEMANTIC:, or HYDE: exactly\n\
         - Keep each expansion under 100 words\n\
         - Do not repeat the original query verbatim\n\
         - Output ONLY the 3 lines, nothing else\n\n\
         User query: {query}"
    )
}

/// Build a `ureq::Agent` with the configured timeout.
fn make_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(LLM_REQUEST_TIMEOUT))
        .build()
        .new_agent()
}

/// Call the Anthropic Messages API with Claude Haiku.
fn call_anthropic(api_key: &str, prompt: &str) -> Result<String, String> {
    let body = serde_json::json!({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": MAX_EXPANSION_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    });

    let agent = make_agent();
    let mut response: ureq::Body = agent
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .send_json(&body)
        .map_err(|e| format!("Anthropic API request failed: {e}"))?
        .into_body();

    let parsed: AnthropicResponse = serde_json::from_reader(response.as_reader())
        .map_err(|e| format!("Failed to parse Anthropic response: {e}"))?;

    parsed
        .content
        .into_iter()
        .find(|block| block.block_type == "text")
        .map(|block| block.text)
        .ok_or_else(|| "Anthropic response contained no text content".to_owned())
}

/// Call the OpenAI Chat Completions API with gpt-4o-mini.
fn call_openai(api_key: &str, prompt: &str) -> Result<String, String> {
    let body = serde_json::json!({
        "model": "gpt-4o-mini",
        "max_tokens": MAX_EXPANSION_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    });

    let agent = make_agent();
    let mut response: ureq::Body = agent
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", &format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .send_json(&body)
        .map_err(|e| format!("OpenAI API request failed: {e}"))?
        .into_body();

    let parsed: OpenAiResponse = serde_json::from_reader(response.as_reader())
        .map_err(|e| format!("Failed to parse OpenAI response: {e}"))?;

    parsed
        .choices
        .into_iter()
        .next()
        .map(|choice| choice.message.content)
        .ok_or_else(|| "OpenAI response contained no choices".to_owned())
}

/// Parse the raw LLM response into structured expanded queries.
fn parse_expansion_response(raw: &str) -> Vec<ExpandedQuery> {
    let mut results = Vec::with_capacity(3);
    for line in raw.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("KEYWORD:") {
            let text = rest.trim();
            if !text.is_empty() {
                results.push(ExpandedQuery {
                    text: text.to_owned(),
                    strategy: ExpansionStrategy::Keyword,
                });
            }
        } else if let Some(rest) = trimmed.strip_prefix("SEMANTIC:") {
            let text = rest.trim();
            if !text.is_empty() {
                results.push(ExpandedQuery {
                    text: text.to_owned(),
                    strategy: ExpansionStrategy::Semantic,
                });
            }
        } else if let Some(rest) = trimmed.strip_prefix("HYDE:") {
            let text = rest.trim();
            if !text.is_empty() {
                results.push(ExpandedQuery {
                    text: text.to_owned(),
                    strategy: ExpansionStrategy::HyDE,
                });
            }
        }
    }
    results
}

fn elapsed_ms_since(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

// ─── API response types ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAiMessage {
    content: String,
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_expansion_response_extracts_all_strategies() {
        let raw = "\
KEYWORD: rust async runtime executor scheduler concurrency
SEMANTIC: What are the best approaches for building an async runtime in Rust?
HYDE: An async runtime in Rust typically uses an executor that polls futures to completion, \
combined with a reactor for I/O events. Popular implementations include tokio and smol.";

        let expanded = parse_expansion_response(raw);
        assert_eq!(expanded.len(), 3);
        assert_eq!(expanded[0].strategy, ExpansionStrategy::Keyword);
        assert_eq!(expanded[1].strategy, ExpansionStrategy::Semantic);
        assert_eq!(expanded[2].strategy, ExpansionStrategy::HyDE);
        assert!(expanded[0].text.contains("rust"));
        assert!(expanded[1].text.contains("async runtime"));
        assert!(expanded[2].text.contains("executor"));
    }

    #[test]
    fn parse_expansion_response_handles_partial_output() {
        let raw = "KEYWORD: search indexing bm25\nsome garbage line\n";
        let expanded = parse_expansion_response(raw);
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].strategy, ExpansionStrategy::Keyword);
    }

    #[test]
    fn parse_expansion_response_handles_empty_values() {
        let raw = "KEYWORD: \nSEMANTIC:\nHYDE: actual content";
        let expanded = parse_expansion_response(raw);
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].strategy, ExpansionStrategy::HyDE);
    }

    #[test]
    fn detect_backend_prefers_anthropic() {
        let env = HashMap::from([
            ("ANTHROPIC_API_KEY".to_owned(), "sk-ant-test".to_owned()),
            ("OPENAI_API_KEY".to_owned(), "sk-test".to_owned()),
        ]);
        let (backend, _) = detect_backend(&env).unwrap();
        assert_eq!(backend, LlmBackend::Anthropic);
    }

    #[test]
    fn detect_backend_falls_back_to_openai() {
        let env = HashMap::from([("OPENAI_API_KEY".to_owned(), "sk-test".to_owned())]);
        let (backend, _) = detect_backend(&env).unwrap();
        assert_eq!(backend, LlmBackend::OpenAi);
    }

    #[test]
    fn detect_backend_returns_none_without_keys() {
        let env = HashMap::new();
        assert!(detect_backend(&env).is_none());
    }

    #[test]
    fn detect_backend_ignores_empty_keys() {
        let env = HashMap::from([
            ("ANTHROPIC_API_KEY".to_owned(), "  ".to_owned()),
            ("OPENAI_API_KEY".to_owned(), String::new()),
        ]);
        assert!(detect_backend(&env).is_none());
    }

    #[test]
    fn expand_query_without_keys_returns_original_only() {
        let env = HashMap::new();
        let result = expand_query("test query", &env);
        assert_eq!(result.queries.len(), 1);
        assert_eq!(result.queries[0].strategy, ExpansionStrategy::Original);
        assert_eq!(result.queries[0].text, "test query");
        assert!(result.backend_used.is_none());
    }

    #[test]
    fn expansion_prompt_contains_query() {
        let prompt = build_expansion_prompt("rust async");
        assert!(prompt.contains("rust async"));
        assert!(prompt.contains("KEYWORD:"));
        assert!(prompt.contains("SEMANTIC:"));
        assert!(prompt.contains("HYDE:"));
    }
}
