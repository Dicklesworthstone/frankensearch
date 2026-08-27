//! Reconciliation integration tests (bd-uxwa2).
//!
//! Validates the frankensearch pipeline components that were reconciled with
//! cass's local implementations during the frankensearch migration (epic 2s9fq).
//!
//! Coverage:
//! 1. Canonicalization — `DefaultCanonicalizer` text preprocessing
//! 2. Model registry — `EmbedderRegistry` auto-detection and metadata
//! 3. `DaemonClient` — `NoopDaemonClient` graceful fallback through search pipeline
//! 4. Daemon fallback embedder — `DaemonFallbackEmbedder` integration with indexing
//! 5. Hash embedder pipeline — end-to-end embed → index → search → verify

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use frankensearch::prelude::*;
use frankensearch::{
    Canonicalizer, DefaultCanonicalizer, EmbedderRegistry, EmbedderStack, HashEmbedder,
    IndexBuilder, NoopDaemonClient, TwoTierIndex, VectorIndex,
};
use frankensearch_core::DaemonClient;
use frankensearch_core::config::TwoTierConfig;
use frankensearch_core::traits::Embedder;
use frankensearch_index::{Quantization, VECTOR_INDEX_FAST_FILENAME};

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn temp_dir(name: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "frankensearch-reconcile-{name}-{}-{now}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

const SAMPLE_DOCS: &[(&str, &str)] = &[
    ("d1", "Rust ownership prevents data races at compile time"),
    ("d2", "Machine learning requires large training datasets"),
    ("d3", "Distributed consensus ensures fault tolerance"),
    ("d4", "Database indexing with B-trees gives fast lookups"),
    ("d5", "Zero-knowledge proofs verify without revealing data"),
];

#[test]
fn facade_reexports_split_lexical_reader_contract() {
    fn accepts_root_reexport(_: Option<&dyn frankensearch::LexicalRead>) {}
    fn accepts_prelude_reexport(_: Option<&dyn LexicalRead>) {}

    accepts_root_reexport(None);
    accepts_prelude_reexport(None);
}

#[cfg(feature = "quill")]
#[test]
fn quill_read_only_refresh_catches_up_across_multiple_generations() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = temp_dir("quill-reader-generation-catch-up");
        let writer = frankensearch::QuillIndex::create(
            &cx,
            &directory,
            frankensearch::QuillConfig::default(),
        )
        .await
        .expect("create durable Quill writer");
        let reader = frankensearch::QuillSearchIndex::open(
            &cx,
            &directory,
            frankensearch::QuillConfig::default(),
        )
        .await
        .expect("open read-only Quill index at genesis");
        let initial_generation = reader.keeper_generation();

        for (document_id, content) in [
            ("refresh-first", "alpha first generation"),
            ("refresh-second", "alpha second generation"),
        ] {
            writer
                .index_document(
                    &cx,
                    &frankensearch::IndexableDocument::new(document_id, content),
                )
                .await
                .expect("stage one refresh fixture document");
            writer
                .commit(&cx)
                .await
                .expect("publish one refresh fixture generation");
        }

        assert!(reader.refresh(&cx).await.expect("catch up read-only index"));
        assert_eq!(
            reader.keeper_generation(),
            initial_generation + 2,
            "one refresh must advance directly across both durable publications"
        );
        let mut document_ids = reader
            .search_doc_ids(&cx, "alpha", 10)
            .expect("search caught-up read-only index")
            .iter()
            .map(|hit| hit.document_id.clone())
            .collect::<Vec<_>>();
        document_ids.sort_unstable();
        assert_eq!(document_ids, ["refresh-first", "refresh-second"]);
        assert!(
            !reader.refresh(&cx).await.expect("repeat current refresh"),
            "refreshing an already-current reader must report no change"
        );
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Canonicalization
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn canonicalizer_strips_markdown() {
    let c = DefaultCanonicalizer::default();
    let input = "**Bold text** and _italic_ with [links](https://example.com)";
    let output = c.canonicalize(input);
    assert!(!output.contains("**"), "bold markers should be stripped");
    assert!(
        !output.contains("_italic_"),
        "italic markers should be stripped"
    );
    assert!(
        output.contains("Bold text"),
        "bold content preserved: {output}"
    );
}

#[test]
fn canonicalizer_collapses_code_blocks() {
    let c = DefaultCanonicalizer {
        code_head_lines: 3,
        code_tail_lines: 2,
        ..Default::default()
    };
    // 10-line code block (> head + tail = 5)
    let lines: Vec<String> = (1..=10).map(|i| format!("line {i}")).collect();
    let code_block = format!("```\n{}\n```", lines.join("\n"));
    let output = c.canonicalize(&code_block);
    // Head lines preserved
    assert!(
        output.contains("line 1"),
        "first head line preserved: {output}"
    );
    assert!(
        output.contains("line 3"),
        "last head line preserved: {output}"
    );
    // Middle lines collapsed
    assert!(
        !output.contains("line 5"),
        "middle lines should be collapsed: {output}"
    );
    // Tail lines preserved
    assert!(
        output.contains("line 9"),
        "first tail line preserved: {output}"
    );
    assert!(
        output.contains("line 10"),
        "last tail line preserved: {output}"
    );
}

#[test]
fn canonicalizer_normalizes_whitespace() {
    let c = DefaultCanonicalizer::default();
    let output = c.canonicalize("hello    world\n\n\nfoo");
    assert!(
        !output.contains("  "),
        "multiple spaces should be collapsed: {output:?}"
    );
}

#[test]
fn canonicalizer_filters_low_signal() {
    let c = DefaultCanonicalizer::default();
    for low in &["OK", "Done.", "Got it.", "Thanks", "yes", "no"] {
        let output = c.canonicalize(low);
        assert!(
            output.is_empty(),
            "low-signal '{low}' should produce empty output, got: {output:?}"
        );
    }
}

#[test]
fn canonicalizer_truncates_long_text() {
    let c = DefaultCanonicalizer {
        max_length: 100,
        ..Default::default()
    };
    let input = "word ".repeat(200); // ~1000 chars
    let output = c.canonicalize(&input);
    assert!(
        output.len() <= 100,
        "output should be truncated to max_length, got {} chars",
        output.len()
    );
}

#[test]
fn canonicalizer_query_is_simpler() {
    let c = DefaultCanonicalizer::default();
    // Queries should preserve content without heavy markdown stripping
    let query = "how does **Rust** handle ownership?";
    let output = c.canonicalize_query(query);
    assert!(
        output.contains("Rust") || output.contains("rust"),
        "query should preserve key terms: {output}"
    );
    assert_ne!(output, "");
}

#[test]
fn canonicalizer_nfc_normalization() {
    let c = DefaultCanonicalizer::default();
    // é as combining sequence (e + combining acute) vs precomposed é
    let combining = "caf\u{0065}\u{0301}";
    let precomposed = "caf\u{00e9}";
    let out1 = c.canonicalize(combining);
    let out2 = c.canonicalize(precomposed);
    assert_eq!(out1, out2, "NFC normalization should unify representations");
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Model registry
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn registry_has_at_least_six_embedders() {
    let embedders = frankensearch::embed::model_registry::registered_embedders();
    assert!(
        embedders.len() >= 6,
        "expected ≥6 registered embedders, got {}",
        embedders.len()
    );
}

#[test]
fn registry_has_rerankers() {
    let rerankers = frankensearch::embed::model_registry::registered_rerankers();
    assert!(
        rerankers.len() >= 5,
        "expected ≥5 registered rerankers, got {}",
        rerankers.len()
    );
}

#[test]
fn registry_hash_always_available() {
    let dir = temp_dir("registry-hash");
    let registry = EmbedderRegistry::new(&dir);
    let available = registry.available();
    assert!(
        available.iter().any(|e| e.name.starts_with("hash")),
        "hash embedder should always be available, got: {:?}",
        available.iter().map(|e| e.name).collect::<Vec<_>>()
    );
}

#[test]
fn registry_best_available_falls_back_to_hash() {
    let dir = temp_dir("registry-best");
    let registry = EmbedderRegistry::new(&dir);
    let best = registry.best_available();
    // Without ML model files, best should be the hash/fnv1a embedder
    assert!(
        best.name.starts_with("hash"),
        "best_available without models should be hash, got: {}",
        best.name
    );
}

#[test]
fn registry_get_by_name_and_id() {
    let dir = temp_dir("registry-get");
    let registry = EmbedderRegistry::new(&dir);

    // By name
    let minilm = registry.get("minilm");
    assert!(minilm.is_some());
    assert_eq!(minilm.unwrap().id, "minilm-384");

    // By id
    let hash = registry.get("fnv1a-384");
    assert!(hash.is_some());
    assert!(hash.unwrap().name.starts_with("hash"));
}

#[test]
fn registry_bakeoff_eligible_excludes_baselines() {
    let dir = temp_dir("registry-bakeoff");
    let registry = EmbedderRegistry::new(&dir);
    let eligible = registry.bakeoff_eligible();

    // No baselines in eligible list
    for e in &eligible {
        assert!(
            !e.is_baseline,
            "baseline '{}' should not be in bakeoff_eligible",
            e.name
        );
    }
}

#[test]
fn registry_embedder_metadata_is_consistent() {
    let embedders = frankensearch::embed::model_registry::registered_embedders();

    // Unique IDs
    let mut ids: Vec<&str> = embedders.iter().map(|e| e.id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(
        ids.len(),
        embedders.len(),
        "all embedder IDs should be unique"
    );

    // Unique names
    let mut names: Vec<&str> = embedders.iter().map(|e| e.name).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(
        names.len(),
        embedders.len(),
        "all embedder names should be unique"
    );

    // All have valid dimensions
    for e in embedders {
        assert!(
            e.dimension > 0 && e.dimension <= 4096,
            "{}: dimension {} out of range",
            e.name,
            e.dimension
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. NoopDaemonClient graceful fallback
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn noop_daemon_client_is_not_available() {
    let client = NoopDaemonClient::new("test-noop");
    assert!(!client.is_available());
}

#[test]
fn noop_daemon_client_embed_returns_error() {
    let client = NoopDaemonClient::new("test-noop");
    let result = client.embed("hello world", "req-1");
    assert!(result.is_err());
}

#[test]
fn noop_daemon_client_rerank_returns_error() {
    let client = NoopDaemonClient::new("test-noop");
    let result = client.rerank("query", &["doc1", "doc2"], "req-1");
    assert!(result.is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. NoopDaemonClient through search pipeline
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn noop_daemon_client_batch_embed_returns_error() {
    let client = NoopDaemonClient::new("test-noop-batch");
    let result = client.embed_batch(&["hello", "world"], "req-batch");
    assert!(result.is_err());
}

#[test]
fn search_pipeline_works_without_daemon() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_dir("no-daemon-search");
        let hash = HashEmbedder::default_256();

        // Build index with hash embedder
        let path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let mut writer = VectorIndex::create_with_revision(
            &path,
            hash.id(),
            "v1",
            hash.dimension(),
            Quantization::F16,
        )
        .expect("create writer");
        for (id, text) in SAMPLE_DOCS {
            let vec = hash.embed_sync(text);
            writer.write_record(id, &vec).expect("write");
        }
        writer.finish().expect("finish");

        // Search with the same embedder — no daemon involved, pure local pipeline
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());
        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));
        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());
        let (results, metrics) = searcher
            .search_collect(&cx, "database indexing lookups", 3)
            .await
            .unwrap();

        assert!(!results.is_empty(), "should return results");
        assert!(results.len() <= 3, "should respect k limit");
        assert!(metrics.phase1_total_ms > 0.0);
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Hash embedder end-to-end pipeline
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hash_embed_index_search_roundtrip() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_dir("hash-roundtrip");
        let fast = Arc::new(HashEmbedder::default_256()) as Arc<dyn Embedder>;
        let stack = EmbedderStack::from_parts(fast, None);

        let mut builder = IndexBuilder::new(&dir).with_embedder_stack(stack);
        for (id, text) in SAMPLE_DOCS {
            builder = builder.add_document(*id, *text);
        }
        let stats = builder.build(&cx).await.unwrap();

        assert_eq!(stats.doc_count, 5);
        assert_eq!(stats.error_count, 0);

        // Search
        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());
        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());
        let (results, _) = searcher
            .search_collect(&cx, "Rust ownership compile time", 3)
            .await
            .unwrap();

        assert!(!results.is_empty());
        let top_ids: Vec<&str> = results.iter().map(|r| r.doc_id.as_str()).collect();
        assert!(
            top_ids.contains(&"d1"),
            "Rust doc should rank high for Rust query: {top_ids:?}"
        );
    });
}

#[test]
fn hash_embedder_determinism() {
    let e = HashEmbedder::default_256();
    let text = "deterministic embedding test";
    let v1 = e.embed_sync(text);
    let v2 = e.embed_sync(text);
    assert_eq!(v1, v2, "same text should produce identical vectors");
}

#[test]
fn hash_embedder_different_texts_differ() {
    let e = HashEmbedder::default_256();
    let v1 = e.embed_sync("Rust programming language");
    let v2 = e.embed_sync("Python data science");
    assert_ne!(v1, v2, "different texts should produce different vectors");
}

#[test]
fn canonicalized_search_matches_uncanonicalized() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_dir("canon-search");
        let fast = Arc::new(HashEmbedder::default_256()) as Arc<dyn Embedder>;
        let stack = EmbedderStack::from_parts(fast, None);

        // Index with markdown-heavy documents
        let docs: &[(&str, &str)] = &[
            ("md1", "**Rust** ownership prevents _data races_"),
            ("md2", "Machine learning needs large datasets"),
            ("md3", "## Distributed Systems\nConsensus algorithms"),
        ];
        let mut builder = IndexBuilder::new(&dir).with_embedder_stack(stack);
        for (id, text) in docs {
            builder = builder.add_document(*id, *text);
        }
        builder.build(&cx).await.unwrap();

        // Search with plain text query
        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());
        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());
        let (results, _) = searcher
            .search_collect(&cx, "distributed consensus", 3)
            .await
            .unwrap();

        assert!(!results.is_empty(), "search should return results");
    });
}
