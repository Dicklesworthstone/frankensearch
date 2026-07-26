//! End-to-end retrieval proof over a real book.
//!
//! Every other corpus in this workspace is synthetic, and
//! `tests/fixtures/README.md` says plainly what that costs us:
//!
//! > For hash-based embeddings, these fixtures primarily validate pipeline
//! > correctness and deterministic behavior.
//!
//! Pipeline-correctness tests pass just as happily when the embedder has
//! silently degraded to non-semantic FNV-1a hashing. That is exactly the
//! failure that made semantic search useless for downstream consumers: with no
//! model on disk the stack falls back to a hash embedder, the fallback is
//! reported at `info!` level, `TwoTierSearcher` then treats a hash fast-embedder
//! as a reason to skip the vector arm entirely, and the caller receives a
//! perfectly ordinary-looking `Ok` containing lexical-only results.
//!
//! These tests exist to make that failure impossible to ship again:
//!
//! * [`lexical`] proves BM25 finds exact rare surface forms — character names,
//!   the ship, the stockade — and returns *nothing* for anachronisms.
//! * [`semantic`] proves concept queries retrieve the right chapter **and that a
//!   hash embedder cannot do the same**. The assertion is the *gap* between the
//!   two, so the test fails loudly the moment semantic quietly degrades.
//! * [`hybrid`] proves RRF actually fuses both arms rather than silently
//!   running on one.
//!
//! Run the semantic lanes with a real model present:
//!
//! ```text
//! FRANKENSEARCH_MODEL_DIR=~/.cache/frankensearch/models \
//!   cargo test --release -p frankensearch --features quill,native \
//!   --test treasure_island_e2e
//! ```
//!
//! **Use `--release`.** These lanes run a real transformer over the whole book
//! three times (semantic corpus, hash-control corpus, hybrid corpus) — roughly
//! 1,000 `MiniLM` forwards. In an unoptimized build the int8 GEMM is slow enough
//! to push the run past half an hour, which is how a lane like this ends up
//! disabled instead of maintained. Optimized, it is minutes.
//!
//! With no model installed the semantic lanes **skip**, printing what is
//! missing and how to get it. Set `FRANKENSEARCH_REQUIRE_SEMANTIC_E2E=1` to turn
//! that skip into a hard failure — CI should do this so "semantic is absent"
//! can never be mistaken for "semantic is fine".

#![allow(clippy::items_after_statements)]

use std::collections::BTreeSet;
#[cfg(feature = "native")]
use std::path::PathBuf;

/// Claim `tracing`'s global default subscriber before anything else can.
///
/// `asupersync::test_utils::run_test_with_cx` installs a TRACE-level subscriber
/// and ignores `RUST_LOG`. `tracing`'s global default is first-wins and applies
/// process-wide, so once any test in this binary calls it, every later test
/// inherits TRACE — including the ones that tokenize. The `HuggingFace` tokenizer
/// inside `NativeEmbedder` logs *five lines per character*: one measured run
/// produced 1,976,875 lines and burned over ten CPU-hours across 66 threads
/// without finishing, while the same tests in a process that never reaches
/// `run_test_with_cx` produce four lines and complete normally.
///
/// So every test here calls this first. It is a workaround, not the fix —
/// `bd-irb1` tracks making `run_test_with_cx` honour `RUST_LOG` — but without it
/// the hybrid lane (the only one needing both `Cx` and the embedder) is
/// unrunnable.
///
/// Honours `RUST_LOG` when set, so deliberate debugging still works.
fn quiet_logging() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
        // Failure means someone else already claimed the global default; that
        // is exactly the race this guards against, and losing it is not fatal.
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .try_init();
    });
}

// ─── Fixture loading and chunking ────────────────────────────────────────────

/// One indexed passage of the book.
#[derive(Debug, Clone)]
pub struct Passage {
    /// Stable identifier, `ch{chapter:02}-{seq:03}`.
    pub id: String,
    /// Sequential chapter index, 1..=34.
    pub chapter: u32,
    /// Chapter title as printed.
    pub title: String,
    /// Passage prose.
    pub text: String,
}

const BOOK: &str = include_str!("../../tests/fixtures/treasure_island/treasure_island.txt");
/// Consumed only by the Quill-backed lexical lane.
#[cfg(feature = "quill")]
const LEXICAL_QUERIES: &str =
    include_str!("../../tests/fixtures/treasure_island/lexical_queries.json");
const SEMANTIC_QUERIES: &str =
    include_str!("../../tests/fixtures/treasure_island/semantic_queries.json");

/// Target passage size in bytes. Chosen so a passage is a few paragraphs of
/// narrative — large enough to carry a scene's meaning into a sentence
/// embedding, small enough that a typical passage fits `MiniLM`'s 512-token
/// window.
///
/// This is a floor, not a cap: paragraphs are never split, so a single long
/// paragraph produces an oversized passage. Over this fixture the result is 339
/// passages averaging ~1.1 KB, with the longest around 2.2 KB (~550 tokens).
/// The embedder truncates those few at 512 tokens, which costs the tail of a
/// long paragraph and is deliberately accepted — the alternative is splitting
/// mid-paragraph, which would damage the meaning the embedding is supposed to
/// capture.
const TARGET_PASSAGE_BYTES: usize = 900;

/// Split the book into chapter-attributed passages.
///
/// Paragraphs are accumulated in order until the buffer reaches
/// [`TARGET_PASSAGE_BYTES`], then flushed. Chapter boundaries always force a
/// flush, so a passage never straddles two chapters — that is what lets the
/// tests use "did the expected chapter come back?" as ground truth.
pub fn chunk_book(raw: &str) -> Vec<Passage> {
    let mut passages = Vec::new();
    let mut chapter = 0_u32;
    let mut title = String::new();
    let mut seq = 0_u32;
    let mut buf = String::new();

    fn flush(
        passages: &mut Vec<Passage>,
        buf: &mut String,
        chapter: u32,
        title: &str,
        seq: &mut u32,
    ) {
        let text = buf.trim();
        if text.is_empty() || chapter == 0 {
            buf.clear();
            return;
        }
        *seq += 1;
        passages.push(Passage {
            id: format!("ch{chapter:02}-{seq:03}"),
            chapter,
            title: title.to_owned(),
            text: text.to_owned(),
        });
        buf.clear();
    }

    for block in raw.split("\n\n") {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }

        if let Some(rest) = block.strip_prefix("== CHAPTER ") {
            flush(&mut passages, &mut buf, chapter, &title, &mut seq);
            // `== CHAPTER 11 :: XI :: What I Heard in the Apple-Barrel ==`
            let body = rest.trim_end_matches(" ==");
            let mut parts = body.split(" :: ");
            chapter = parts
                .next()
                .and_then(|n| n.trim().parse().ok())
                .unwrap_or_else(|| panic!("unparseable chapter heading: {block}"));
            let _roman = parts.next();
            parts
                .next()
                .unwrap_or_default()
                .trim()
                .clone_into(&mut title);
            seq = 0;
            continue;
        }
        if block.starts_with("== PART ") {
            continue;
        }

        // Paragraphs arrive hard-wrapped from the source; unwrap them so the
        // embedder sees ordinary sentences rather than ragged line fragments.
        let paragraph = block
            .lines()
            .map(str::trim)
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_owned();
        if paragraph.is_empty() {
            continue;
        }

        if !buf.is_empty() {
            buf.push_str("\n\n");
        }
        buf.push_str(&paragraph);

        if buf.len() >= TARGET_PASSAGE_BYTES {
            flush(&mut passages, &mut buf, chapter, &title, &mut seq);
        }
    }
    flush(&mut passages, &mut buf, chapter, &title, &mut seq);
    passages
}

fn corpus() -> Vec<Passage> {
    let passages = chunk_book(BOOK);
    assert!(
        passages.len() > 200,
        "fixture should chunk into a realistic corpus, got {}",
        passages.len()
    );
    passages
}

/// Chapters represented by the given doc ids.
///
/// Only the retrieval lanes need this; the fixture-shape tests work on the
/// passages directly.
#[cfg(any(feature = "quill", feature = "native"))]
fn chapters_of(passages: &[Passage], ids: &[String]) -> BTreeSet<u32> {
    ids.iter()
        .filter_map(|id| passages.iter().find(|p| &p.id == id))
        .map(|p| p.chapter)
        .collect()
}

// ─── Model resolution ────────────────────────────────────────────────────────

/// Where a real MiniLM sentence-embedder may live.
///
/// Mirrors the library's own precedence: `FRANKENSEARCH_MODEL_DIR` first (both
/// as a model root and as a direct model directory), then the default per-user
/// cache.
#[cfg(feature = "native")]
fn model_dir_candidates() -> Vec<PathBuf> {
    const MODEL_NAME: &str = "all-MiniLM-L6-v2";
    let mut out = Vec::new();
    if let Some(root) = std::env::var_os("FRANKENSEARCH_MODEL_DIR") {
        let root = PathBuf::from(root);
        out.push(root.join(MODEL_NAME));
        out.push(root);
    }
    if let Some(home) = std::env::var_os("HOME") {
        out.push(
            PathBuf::from(home)
                .join(".cache/frankensearch/models")
                .join(MODEL_NAME),
        );
    }
    out
}

/// A resolved model directory, or `None` with the reason already reported.
///
/// Honours `FRANKENSEARCH_REQUIRE_SEMANTIC_E2E=1` by panicking instead of
/// skipping, so a CI lane can assert that semantic search was genuinely
/// exercised rather than quietly stepped over.
#[cfg(feature = "native")]
fn resolve_model_dir(lane: &str) -> Option<PathBuf> {
    // Ask the frozen manifest what the loader will actually demand rather than
    // keeping a second list here. `NativeEmbedder::load` verifies every declared
    // artifact by size and SHA-256, so any list maintained separately from the
    // manifest is a list that can drift out of date — and did: this guard used
    // to check only `tokenizer.json` plus a weights file, which let a directory
    // missing `config.json`, `special_tokens_map.json` and `tokenizer_config.json`
    // pass as "model present". The lane then died inside `NativeEmbedder::load`
    // naming one missing file at a time instead of skipping with a full list.
    let manifest =
        frankensearch_embed::model_manifest::ModelArtifactManifestV1::minilm_native_frankentorch()
            .expect("registered native MiniLM manifest");
    let required: Vec<&str> = manifest
        .artifacts
        .iter()
        .map(|a| a.relative_path.as_str())
        .collect();

    let candidates = model_dir_candidates();
    // Remember the closest near-miss so the diagnostic names what a
    // half-provisioned directory is actually missing.
    let mut closest: Option<(&PathBuf, Vec<&str>)> = None;
    for dir in &candidates {
        let missing: Vec<&str> = required
            .iter()
            .copied()
            .filter(|rel| !dir.join(rel).is_file())
            .collect();
        if missing.is_empty() {
            return Some(dir.clone());
        }
        if closest
            .as_ref()
            .is_none_or(|(_, seen)| missing.len() < seen.len())
        {
            closest = Some((dir, missing));
        }
    }

    let searched = candidates
        .iter()
        .map(|p| format!("  - {}", p.display()))
        .collect::<Vec<_>>()
        .join("\n");
    let detail = match &closest {
        Some((dir, missing)) => {
            let lines = missing
                .iter()
                .map(|rel| {
                    let url = manifest
                        .artifacts
                        .iter()
                        .find(|a| a.relative_path == *rel)
                        .map_or("<no pinned url>", |a| a.upstream_url.as_str());
                    format!("  - {rel}\n      {url}")
                })
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "Closest directory {} is missing {} of {} pinned artifacts:\n{lines}\n",
                dir.display(),
                missing.len(),
                required.len()
            )
        }
        None => String::new(),
    };
    let message = format!(
        "SKIPPING {lane}: no complete all-MiniLM-L6-v2 model found.\n\
         Searched:\n{searched}\n\
         {detail}\
         Every artifact the frozen manifest declares must be present; \
         `NativeEmbedder::load` verifies each by size and SHA-256 at revision \
         {revision}.\n\
         Set FRANKENSEARCH_REQUIRE_SEMANTIC_E2E=1 to make this a hard failure.",
        revision = manifest.upstream_revision
    );

    assert!(
        std::env::var("FRANKENSEARCH_REQUIRE_SEMANTIC_E2E").as_deref() != Ok("1"),
        "{message}"
    );
    eprintln!("{message}");
    None
}

// ─── Corpus shape ────────────────────────────────────────────────────────────

#[test]
fn fixture_chunks_into_all_thirty_four_chapters() {
    quiet_logging();
    let passages = corpus();
    let chapters: BTreeSet<u32> = passages.iter().map(|p| p.chapter).collect();

    assert_eq!(
        chapters.len(),
        34,
        "expected all 34 chapters to survive chunking, saw {chapters:?}"
    );
    assert_eq!(*chapters.iter().next().expect("first"), 1);
    assert_eq!(*chapters.iter().next_back().expect("last"), 34);

    // Passage ids must be unique — duplicates would silently collapse documents
    // in any index we build from them.
    let ids: BTreeSet<&str> = passages.iter().map(|p| p.id.as_str()).collect();
    assert_eq!(ids.len(), passages.len(), "passage ids must be unique");

    // No passage may straddle a chapter, and none may be empty.
    for p in &passages {
        assert!(!p.text.trim().is_empty(), "empty passage {}", p.id);
        assert!(
            !p.title.is_empty(),
            "passage {} lost its chapter title",
            p.id
        );
    }

    // Sanity-check that the text is really the book.
    let ch11 = passages
        .iter()
        .find(|p| p.chapter == 11)
        .expect("chapter 11");
    assert_eq!(ch11.title, "What I Heard in the Apple-Barrel");
}

#[test]
fn semantic_queries_do_not_leak_the_answer_vocabulary() {
    quiet_logging();
    // A future maintainer facing a red semantic test could "fix" it by sneaking
    // the target passage's distinctive words into the query. That would turn a
    // meaning test into a keyword test and re-open the exact hole this file
    // exists to close. Enforce the fixture's own `avoid_terms` contract.
    let spec: serde_json::Value =
        serde_json::from_str(SEMANTIC_QUERIES).expect("semantic_queries.json parses");
    let queries = spec["queries"].as_array().expect("queries array");
    assert!(!queries.is_empty());

    for q in queries {
        let text = q["query"].as_str().expect("query text").to_lowercase();
        let name = q["name"].as_str().unwrap_or("<unnamed>");
        for term in q["avoid_terms"].as_array().expect("avoid_terms") {
            let term = term.as_str().expect("avoid term").to_lowercase();
            assert!(
                !text.contains(&term),
                "semantic query `{name}` leaks answer vocabulary `{term}`: {text:?}"
            );
        }
    }
}

// ─── Lexical ─────────────────────────────────────────────────────────────────

#[cfg(feature = "quill")]
mod lexical {
    use super::{LEXICAL_QUERIES, Passage, chapters_of, corpus};
    use frankensearch::{IndexableDocument, QuillConfig, QuillIndex};

    async fn build_index(
        cx: &frankensearch::Cx,
        dir: &std::path::Path,
        passages: &[Passage],
    ) -> QuillIndex {
        let index = QuillIndex::create(cx, dir, QuillConfig::default())
            .await
            .expect("create quill index");
        let docs: Vec<IndexableDocument> = passages
            .iter()
            .map(|p| {
                IndexableDocument::new(p.id.clone(), p.text.clone()).with_title(p.title.clone())
            })
            .collect();
        index
            .index_documents(cx, &docs)
            .await
            .expect("index passages");
        index.commit(cx).await.expect("commit");
        index
    }

    #[test]
    fn exact_terms_retrieve_the_right_chapters_and_nothing_else() {
        super::quiet_logging();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let passages = corpus();
            let tmp = tempfile::tempdir().expect("tempdir");
            let index = build_index(&cx, tmp.path(), &passages).await;

            let spec: serde_json::Value =
                serde_json::from_str(LEXICAL_QUERIES).expect("lexical_queries.json parses");
            let search_limit =
                usize::try_from(spec["search_limit"].as_u64().expect("search_limit"))
                    .expect("usize");
            let recall_limit =
                usize::try_from(spec["recall_limit"].as_u64().expect("recall_limit"))
                    .expect("usize");

            for q in spec["queries"].as_array().expect("queries") {
                let name = q["name"].as_str().expect("name");
                let term = q["term"].as_str().expect("term");
                let needle = q["must_contain"]
                    .as_str()
                    .expect("must_contain")
                    .to_lowercase();

                let hits = index
                    .search_results(&cx, term, search_limit)
                    .expect("lexical search");
                assert!(
                    !hits.is_empty(),
                    "lexical query `{name}` ({term}) returned no hits"
                );

                // Precision: every hit must genuinely contain the term. We look
                // the text up in our own corpus rather than trusting stored
                // content, so this checks retrieval, not storage.
                for hit in &hits {
                    let passage = passages
                        .iter()
                        .find(|p| p.id == hit.doc_id.as_str())
                        .unwrap_or_else(|| panic!("hit {} is not a corpus passage", hit.doc_id));
                    let hay = format!("{} {}", passage.title, passage.text).to_lowercase();
                    assert!(
                        hay.contains(&needle),
                        "lexical query `{name}` returned {} which does not contain `{needle}`",
                        hit.doc_id
                    );
                }

                // Recall, where the fixture judges it honest to assert one. An
                // empty `expect_chapters` means the term recurs throughout the
                // novel, so which chapters surface is a term-density accident
                // rather than a retrieval property worth pinning.
                let expected: Vec<u32> = q["expect_chapters"]
                    .as_array()
                    .expect("expect_chapters")
                    .iter()
                    .map(|c| u32::try_from(c.as_u64().expect("chapter")).expect("u32"))
                    .collect();
                if expected.is_empty() {
                    continue;
                }

                let deep = index
                    .search_results(&cx, term, recall_limit)
                    .expect("lexical search");
                let ids: Vec<String> = deep.iter().map(|h| h.doc_id.to_string()).collect();
                let got = chapters_of(&passages, &ids);
                assert!(
                    expected.iter().any(|c| got.contains(c)),
                    "lexical query `{name}` ({term}) found none of chapters \
                     {expected:?} within {recall_limit} hits; got {got:?}"
                );
            }

            // Anachronisms and non-words must return nothing at all. A search
            // engine that always returns *something* is indistinguishable from
            // one that returns noise.
            for q in spec["must_return_nothing"]
                .as_array()
                .expect("must_return_nothing")
            {
                let term = q["term"].as_str().expect("term");
                let hits = index.search_results(&cx, term, 10).expect("lexical search");
                assert!(
                    hits.is_empty(),
                    "`{term}` should match nothing in a 19th-century novel, got {} hits",
                    hits.len()
                );
            }
        });
    }
}

// ─── Semantic ────────────────────────────────────────────────────────────────

#[cfg(feature = "native")]
mod semantic {
    use super::{Passage, SEMANTIC_QUERIES, chapters_of, corpus, resolve_model_dir};
    use frankensearch::{
        Embedder, HashAlgorithm, HashEmbedder, InMemoryVectorIndex, NativeEmbedder, SyncEmbed,
    };

    /// Embed every passage and build a brute-force vector index over it.
    ///
    /// Takes the embedding step as a closure rather than a trait bound: the
    /// real embedder implements [`SyncEmbed`] while the hash control implements
    /// [`Embedder`] with inherent sync helpers, and forcing them into one bound
    /// would need an adapter that obscures what is being compared.
    fn index_with(
        embed_batch: impl Fn(&[&str]) -> Vec<Vec<f32>>,
        dimension: usize,
        passages: &[Passage],
    ) -> InMemoryVectorIndex {
        let texts: Vec<&str> = passages.iter().map(|p| p.text.as_str()).collect();
        let vectors = embed_batch(&texts);
        let ids: Vec<String> = passages.iter().map(|p| p.id.clone()).collect();
        InMemoryVectorIndex::from_vectors(ids, vectors, dimension).expect("build vector index")
    }

    /// How many of the concept queries retrieved their expected chapter.
    /// Per-query outcomes, in fixture order.
    ///
    /// Returns the whole vector rather than a count because the assertion is a
    /// PAIRED comparison: what matters is whether the real embedder succeeds on
    /// the same query the degraded one fails, not the two totals.
    fn hit_vector(
        embed_one: impl Fn(&str) -> Vec<f32>,
        index: &InMemoryVectorIndex,
        passages: &[Passage],
        spec: &serde_json::Value,
        label: &str,
    ) -> Vec<bool> {
        let top_k = usize::try_from(spec["top_k"].as_u64().expect("top_k")).expect("usize");
        let mut outcomes = Vec::new();

        for q in spec["queries"].as_array().expect("queries") {
            let name = q["name"].as_str().expect("name");
            let text = q["query"].as_str().expect("query");
            let expected: Vec<u32> = q["expect_chapters"]
                .as_array()
                .expect("expect_chapters")
                .iter()
                .map(|c| u32::try_from(c.as_u64().expect("chapter")).expect("u32"))
                .collect();

            let qvec = embed_one(text);
            let results = index
                .search_top_k(&qvec, top_k, None)
                .expect("vector search");
            let ids: Vec<String> = results.iter().map(|h| h.doc_id.to_string()).collect();
            let got = chapters_of(passages, &ids);

            let hit = expected.iter().any(|c| got.contains(c));
            outcomes.push(hit);
            eprintln!(
                "  [{label}] {name}: expected any of {expected:?}, got {got:?} -> {}",
                if hit { "HIT" } else { "miss" }
            );
        }
        outcomes
    }

    #[test]
    fn concept_queries_find_the_right_chapters_and_hashing_cannot() {
        super::quiet_logging();
        let Some(model_dir) = resolve_model_dir("semantic concept-retrieval lane") else {
            return;
        };

        let passages = corpus();
        let spec: serde_json::Value =
            serde_json::from_str(SEMANTIC_QUERIES).expect("semantic_queries.json parses");
        let total = spec["queries"].as_array().expect("queries").len();
        let min_semantic =
            usize::try_from(spec["min_semantic_hits"].as_u64().expect("min")).expect("usize");
        let max_hash_only =
            usize::try_from(spec["max_hash_only_hits"].as_u64().expect("max_hash_only"))
                .expect("usize");
        let min_semantic_only = usize::try_from(
            spec["min_semantic_only_hits"]
                .as_u64()
                .expect("min_semantic_only"),
        )
        .expect("usize");

        let semantic = NativeEmbedder::load(&model_dir).expect("load native MiniLM embedder");
        assert!(
            semantic.is_semantic(),
            "NativeEmbedder must report itself semantic"
        );

        let semantic_id = semantic.id().to_owned();
        eprintln!(
            "embedding {} passages with {semantic_id}...",
            passages.len()
        );
        let semantic_index = index_with(
            |texts| semantic.embed_batch_sync(texts).expect("embed corpus"),
            semantic.dimension(),
            &passages,
        );
        let semantic_hits = hit_vector(
            |text| semantic.embed_sync(text).expect("embed query"),
            &semantic_index,
            &passages,
            &spec,
            &semantic_id,
        );

        // The control. This is precisely what the stack silently falls back to
        // when no model is installed, so it is the right thing to measure
        // against: if these two scores are close, semantic search is not
        // actually working, whatever the logs say.
        let hash = HashEmbedder::new(semantic.dimension(), HashAlgorithm::FnvModular);
        assert!(
            !hash.is_semantic(),
            "HashEmbedder must report itself non-semantic"
        );
        let hash_id = hash.id().to_owned();
        let hash_index = index_with(
            |texts| texts.iter().map(|t| hash.embed_sync(t)).collect(),
            hash.dimension(),
            &passages,
        );
        let hash_hits = hit_vector(
            |text| hash.embed_sync(text),
            &hash_index,
            &passages,
            &spec,
            &hash_id,
        );

        // Paired analysis. Totals alone are the wrong statistic: `HashEmbedder`
        // hashes *tokens* into dimensions, so it behaves as a degenerate
        // bag-of-words matcher and will win any query whose wording happens to
        // overlap its answer. What distinguishes real retrieval is succeeding on
        // the same query where the degraded embedder fails, so count discordant
        // pairs and apply a sign test. See `_criterion` in the fixture.
        let semantic_total = semantic_hits.iter().filter(|h| **h).count();
        let hash_total = hash_hits.iter().filter(|h| **h).count();
        let semantic_only = semantic_hits
            .iter()
            .zip(&hash_hits)
            .filter(|(s, h)| **s && !**h)
            .count();
        let hash_only = semantic_hits
            .iter()
            .zip(&hash_hits)
            .filter(|(s, h)| !**s && **h)
            .count();

        eprintln!(
            "semantic {semantic_total}/{total} vs hash {hash_total}/{total}; \
             discordant: semantic-only {semantic_only}, hash-only {hash_only} \
             (need semantic >= {min_semantic}, semantic-only >= \
             {min_semantic_only}, hash-only <= {max_hash_only})"
        );

        assert!(
            semantic_total >= min_semantic,
            "real sentence embeddings retrieved the expected chapter for only \
             {semantic_total}/{total} concept queries (need {min_semantic}). \
             Semantic retrieval is not working."
        );
        assert!(
            semantic_only >= min_semantic_only,
            "only {semantic_only} queries were answered by the real embedder and \
             missed by the hash control (need {min_semantic_only}). Without \
             enough discordant pairs this comparison has no statistical power, \
             so it cannot demonstrate that meaning is being used."
        );
        assert!(
            hash_only <= max_hash_only,
            "the non-semantic hash control beat the real embedder on {hash_only} \
             queries (at most {max_hash_only} tolerated). Either the real \
             embedder is degraded, or those queries are answerable by surface \
             form and belong in the lexical fixture instead."
        );
    }

    #[test]
    fn a_hash_embedder_is_never_mistaken_for_a_semantic_one() {
        super::quiet_logging();
        // The degradation is only detectable if these two disagree about
        // themselves. Guard the property directly — it is load-bearing for the
        // skip logic, for `TwoTierAvailability`, and for the fusion searcher's
        // decision to run the vector arm at all.
        let hash = HashEmbedder::default_384();
        assert!(!hash.is_semantic());
        assert!(hash.id().starts_with("fnv1a-") || hash.id().starts_with("jl-"));

        let Some(model_dir) = resolve_model_dir("embedder self-report lane") else {
            return;
        };
        let semantic = NativeEmbedder::load(&model_dir).expect("load native MiniLM embedder");
        assert!(semantic.is_semantic());
        assert_ne!(semantic.id(), hash.id());
        assert_eq!(semantic.dimension(), 384);
    }

    #[test]
    fn semantically_close_passages_score_higher_than_unrelated_ones() {
        super::quiet_logging();
        // A direct, corpus-independent check that the vectors carry meaning:
        // two descriptions of the same idea must be closer than two
        // descriptions of different ideas.
        let Some(model_dir) = resolve_model_dir("embedding geometry lane") else {
            return;
        };
        let embedder = NativeEmbedder::load(&model_dir).expect("load native MiniLM embedder");

        let anchor = embedder
            .embed_sync("pirates searching for gold buried on a remote island")
            .expect("embed anchor");
        let near = embedder
            .embed_sync("buccaneers hunting for treasure hidden on a distant shore")
            .expect("embed near");
        let far = embedder
            .embed_sync("a recipe for baking sourdough bread at home")
            .expect("embed far");

        let sim = frankensearch::cosine_similarity;
        let near_score = sim(&anchor, &near);
        let far_score = sim(&anchor, &far);

        assert!(
            near_score > far_score,
            "paraphrase similarity {near_score} should exceed unrelated \
             similarity {far_score}"
        );
        assert!(
            near_score - far_score > 0.15,
            "the gap between paraphrase ({near_score}) and unrelated \
             ({far_score}) is only {}, too small to call these embeddings \
             semantic",
            near_score - far_score
        );
    }
}

// ─── Hybrid ──────────────────────────────────────────────────────────────────

#[cfg(all(feature = "quill", feature = "native"))]
mod hybrid {
    use super::{Passage, chapters_of, corpus, resolve_model_dir};
    use frankensearch::{
        InMemoryVectorIndex, IndexableDocument, NativeEmbedder, QuillConfig, QuillIndex, RrfConfig,
        SyncEmbed, rrf_fuse,
    };

    #[test]
    fn rrf_fuses_both_arms_over_real_prose() {
        super::quiet_logging();
        let Some(model_dir) = resolve_model_dir("hybrid fusion lane") else {
            return;
        };

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let passages: Vec<Passage> = corpus();
            let embedder = NativeEmbedder::load(&model_dir).expect("load native MiniLM embedder");

            // Lexical arm.
            let tmp = tempfile::tempdir().expect("tempdir");
            let lexical = QuillIndex::create(&cx, tmp.path(), QuillConfig::default())
                .await
                .expect("create quill index");
            let docs: Vec<IndexableDocument> = passages
                .iter()
                .map(|p| {
                    IndexableDocument::new(p.id.clone(), p.text.clone()).with_title(p.title.clone())
                })
                .collect();
            lexical.index_documents(&cx, &docs).await.expect("index");
            lexical.commit(&cx).await.expect("commit");

            // Semantic arm.
            let texts: Vec<&str> = passages.iter().map(|p| p.text.as_str()).collect();
            let vectors = embedder.embed_batch_sync(&texts).expect("embed corpus");
            let ids: Vec<String> = passages.iter().map(|p| p.id.clone()).collect();
            let vector_index =
                InMemoryVectorIndex::from_vectors(ids, vectors, embedder.dimension())
                    .expect("vector index");

            // A query with both an exact name and a concept: the name is what
            // BM25 is good at, the situation is what embeddings are good at.
            let query = "Ben Gunn the man left alone on the island for years";
            let lexical_hits = lexical
                .search_results(&cx, query, 30)
                .expect("lexical search");
            let qvec = embedder.embed_sync(query).expect("embed query");
            let semantic_hits = vector_index
                .search_top_k(&qvec, 30, None)
                .expect("vector search");

            assert!(!lexical_hits.is_empty(), "lexical arm produced nothing");
            assert!(!semantic_hits.is_empty(), "semantic arm produced nothing");

            let fused = rrf_fuse(&lexical_hits, &semantic_hits, 10, 0, &RrfConfig::default());
            assert!(!fused.is_empty(), "fusion produced nothing");
            assert!(fused.len() <= 10, "fusion must respect the limit");

            // Fusion is only worth its complexity if both arms actually feed
            // it. If every fused hit came from one source, we are shipping a
            // single-arm engine wearing a fusion costume.
            let both = fused.iter().filter(|h| h.in_both_sources).count();
            assert!(
                both > 0,
                "no fused hit was found by BOTH arms — fusion is running on one \
                 source only"
            );

            // `both > 0` alone is weak evidence here: this query contains the
            // literal string "Ben Gunn", so BM25 and the embedder can agree for
            // purely lexical reasons and the semantic arm could be contributing
            // nothing of its own. Re-run with a query that has no proper nouns
            // at all and require the semantic arm to put something into the
            // fused top-k that lexical never found.
            let concept = "someone who has lived completely alone in the wild for \
                           years and has half forgotten how to speak";
            let concept_lexical = lexical
                .search_results(&cx, concept, 30)
                .expect("lexical search");
            let concept_vec = embedder.embed_sync(concept).expect("embed query");
            let concept_semantic = vector_index
                .search_top_k(&concept_vec, 30, None)
                .expect("vector search");
            let concept_fused = rrf_fuse(
                &concept_lexical,
                &concept_semantic,
                10,
                0,
                &RrfConfig::default(),
            );

            let semantic_only = concept_fused
                .iter()
                .filter(|h| h.semantic_rank.is_some() && h.lexical_rank.is_none())
                .count();
            assert!(
                semantic_only > 0,
                "on a proper-noun-free concept query, every fused hit was also \
                 found lexically — the semantic arm contributed nothing unique, \
                 so fusion is not actually adding meaning-based recall"
            );

            // Scores must be ranked descending, and the target chapter present.
            for pair in fused.windows(2) {
                assert!(
                    pair[0].rrf_score >= pair[1].rrf_score,
                    "fused results must be ordered by descending RRF score"
                );
            }
            let fused_ids: Vec<String> = fused.iter().map(|h| h.doc_id.to_string()).collect();
            let chapters = chapters_of(&passages, &fused_ids);
            assert!(
                chapters.contains(&15),
                "hybrid search for Ben Gunn missed chapter 15; got {chapters:?}"
            );
        });
    }
}
