use super::*;
use asupersync::test_utils::run_test_with_cx;
use frankensearch::native_ann::builder::{
    NativeBuildPrecision, NativeBuildRetrieval, NativeIndexBuilder,
};
use frankensearch::{Embedder, ModelCategory, SearchFuture, SearchResult};
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use std::io::{self, Cursor};
use std::path::{Path, PathBuf};
use std::sync::Arc;

// Used only to construct real sealed fixtures. The keyword serving entry point
// accepts neither this provider nor any other model, including after fetch.
struct Provider(EmbeddingIdentityBundleV1);
impl Embedder for Provider {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.0)
    }
    fn dimension(&self) -> usize {
        2
    }
    fn id(&self) -> &str {
        &self.0.space.logical_model_id
    }
    fn model_name(&self) -> &str {
        self.id()
    }
    fn is_semantic(&self) -> bool {
        false
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async { Ok(vec![1.0, 0.0]) })
    }
}

fn generation(sequence: u64) -> ArtifactGenerationIdentityV1 {
    ArtifactGenerationIdentityV1::new(sequence, [37; 16]).unwrap()
}

async fn fixture(cx: &Cx, partitioned: bool) -> (tempfile::TempDir, Selection, PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let model = Arc::new(Provider(EmbeddingIdentityBundleV1::explicit_test_model(
        "keyword-source",
        2,
    )));
    let builder = NativeIndexBuilder::new(root.path().join("index"), generation(1), model)
        .unwrap()
        .with_fast_storage(
            NativeBuildPrecision::F32,
            NativeBuildRetrieval::Hnsw {
                params: crate::HnswParams::default(),
                seed: 37,
            },
        )
        .add_documents([
            IndexableDocument::new("a", "retry café🦀 retained source")
                .with_title("original")
                .with_metadata("tenant", "A"),
            IndexableDocument::new("z", "retry CANARY_PRIVATE").with_metadata("tenant", "B"),
        ]);
    let opened = if partitioned {
        sharded::Opened::Sharded(Box::new(builder.build_sharded_hybrid(cx, 1).await.unwrap()))
    } else {
        sharded::Opened::Single(Box::new(builder.build_hybrid(cx).await.unwrap()))
    };
    let vector = match &opened {
        sharded::Opened::Single(index) => index.vectors().fast().vector_path().to_path_buf(),
        sharded::Opened::Sharded(index) => index
            .vectors()
            .partitions()
            .last()
            .unwrap()
            .fast()
            .vector_path()
            .to_path_buf(),
    };
    let selection = crate::update::seal_selection(cx, opened.borrow()).unwrap();
    drop(opened);
    (root, selection, vector)
}

fn options(receipt: &Path) -> Options {
    Options::parse([
        "serve".to_owned(),
        "--receipt".to_owned(),
        receipt.to_str().unwrap().to_owned(),
        "--lexical-only".to_owned(),
        "--filter".to_owned(),
        r#"{"metadata":{"tenant":"A"}}"#.to_owned(),
    ])
    .unwrap()
    .unwrap()
}

fn fetch(id: &str, sequence: u64) -> serde_json::Value {
    serde_json::json!({
        "op":"fetch", "id":id, "expected_generation":generation(sequence),
        "documents":[{"doc_id":"a", "include_metadata":true}, {"doc_id":"z"}, {"doc_id":"/not/a/source/path"}],
    })
}

fn input(messages: &[serde_json::Value]) -> Cursor<Vec<u8>> {
    let mut bytes = Vec::new();
    for message in messages {
        serde_json::to_writer(&mut bytes, message).unwrap();
        bytes.push(b'\n');
    }
    Cursor::new(bytes)
}

fn frames(bytes: &[u8]) -> Vec<serde_json::Value> {
    bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect()
}

#[test]
fn model_free_keyword_search_and_fetch_share_scopes_even_with_damaged_vectors() {
    run_test_with_cx(|cx| async move {
        for partitioned in [false, true] {
            let (root, selection, vector) = fixture(&cx, partitioned).await;
            let receipt = root.path().join("selection.json");
            crate::save_selection(&selection, &receipt).unwrap();
            // Deliberately invalidate a semantic artifact after dropping its
            // owner. This must not be read/repaired by the explicit lexical lane.
            crate::fs::write(&vector, b"invalid semantic artifact").unwrap();
            let mut narrowed = fetch("narrowed", 1);
            narrowed["filter"] = serde_json::json!({"ids":["z"]});
            let mut reader = input(&[
                serde_json::json!({"id":"search","query":"retry"}),
                fetch("source", 1),
                narrowed,
                fetch("stale", 2),
                serde_json::json!({"op":"status","id":"status"}),
            ]);
            let mut output = Vec::new();
            serve_with_io(&cx, &options(&receipt), &mut reader, &mut output)
                .await
                .unwrap();
            let replies = frames(&output);
            assert_eq!(replies[0]["source_fetch_supported"], true);
            assert_eq!(replies[0]["semantic_components_verified"], false);
            assert_eq!(
                replies[0]["source_layout"],
                if partitioned { "sharded" } else { "single" }
            );
            let page = replies
                .iter()
                .find(|frame| frame["event"] == "results")
                .unwrap();
            assert_eq!(page["results"][0]["doc_id"], "a");
            let source = replies
                .iter()
                .find(|frame| frame["id"] == "source")
                .unwrap();
            assert_eq!(source["request"], 2);
            assert_eq!(source["seq"], 0);
            assert_eq!(source["generation"], page["generation"]);
            assert_eq!(
                source["documents"][0]["content"],
                "retry café🦀 retained source"
            );
            assert_eq!(source["documents"][0]["title"], "original");
            assert_eq!(source["documents"][1]["status"], "unavailable");
            assert_eq!(source["documents"][2]["status"], "unavailable");
            let narrowed = replies
                .iter()
                .find(|frame| frame["id"] == "narrowed")
                .unwrap();
            assert!(
                narrowed["documents"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|doc| doc["status"] == "unavailable")
            );
            let stale = replies.iter().find(|frame| frame["id"] == "stale").unwrap();
            assert_eq!(stale["ok"], false);
            assert!(stale.get("documents").is_none());
            assert_eq!(replies.last().unwrap()["operation"], "status");
            assert_eq!(replies.last().unwrap()["ok"], true);
            assert!(!String::from_utf8_lossy(&output).contains("CANARY_PRIVATE"));
            assert_eq!(
                crate::fs::read(vector).unwrap(),
                b"invalid semantic artifact"
            );
        }
    });
}

#[test]
fn fetch_never_bypasses_the_selected_source_receipt_on_keyword_startup() {
    run_test_with_cx(|cx| async move {
        for partitioned in [false, true] {
            let (root, mut selection, _) = fixture(&cx, partitioned).await;
            selection.snapshot.sha256[0] ^= 1;
            let receipt = root.path().join("wrong-receipt.json");
            crate::save_selection(&selection, &receipt).unwrap();
            let mut reader = input(&[fetch("not-admitted", 1)]);
            let mut output = Vec::new();
            assert!(
                serve_with_io(&cx, &options(&receipt), &mut reader, &mut output)
                    .await
                    .is_err()
            );
            assert!(
                output.is_empty(),
                "no ready or source payload before admission"
            );
            assert_eq!(reader.position(), 0);
        }
    });
}

#[test]
fn keyword_source_flush_failure_ends_the_session_without_a_followup_frame() {
    struct Broken {
        writes: usize,
        flushes: usize,
    }
    impl Write for Broken {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            self.flushes += 1;
            if self.flushes == 2 {
                Err(io::Error::other("fetch flush failed"))
            } else {
                Ok(())
            }
        }
    }
    run_test_with_cx(|cx| async move {
        let (root, selection, _) = fixture(&cx, true).await;
        let receipt = root.path().join("selection.json");
        crate::save_selection(&selection, &receipt).unwrap();
        let mut reader = input(&[fetch("first", 1), fetch("unread", 1)]);
        let mut output = Broken {
            writes: 0,
            flushes: 0,
        };
        let error = serve_with_io(&cx, &options(&receipt), &mut reader, &mut output)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("fetch flush failed"));
        assert_eq!(output.writes, 2);
        assert_eq!(output.flushes, 2);
        assert!(reader.position() < u64::try_from(reader.get_ref().len()).unwrap());
    });
}
