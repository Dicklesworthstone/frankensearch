//! A `tokenizers` Unigram model with a compact prefix trie (GH #46).
//!
//! `tokenizers` 0.23 keeps a Unigram vocabulary in a trie whose every internal
//! node owns a hash map. For the 500 353 pieces of potion-multilingual-128M
//! that is about 1.1 million maps, roughly 400 MB, built and dropped on every
//! process start. This model stores the same trie as sorted arrays and ports
//! the library's `encode_optimized` Viterbi and `tokenize` unchanged, so it
//! yields the same pieces and ids; everything around the model (added tokens,
//! normalizer, pre-tokenizer, post-processor) is the library's own
//! `TokenizerImpl`, deserialized from the same `tokenizer.json`.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Deserializer};
use tokenizers::models::unigram::{UnigramError, UnigramTrainer};
use tokenizers::{
    DecoderWrapper, Model, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper, Token,
    TokenizerImpl,
};

/// Score penalty for an unknown character (`K_UNK_PENALTY` in `tokenizers`).
const UNK_PENALTY: f64 = 10.0;

/// A complete tokenizer whose model is [`CompactUnigram`].
pub(super) type CompactUnigramTokenizer = TokenizerImpl<
    CompactUnigram,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

/// Prefix trie over piece bytes in compressed-sparse-row form. Node `n` has
/// the sorted child labels `labels[first[n]..first[n + 1]]` with the matching
/// `targets`, and `piece[n]` is one more than the id of the piece ending at
/// `n` (0 when no piece ends there).
struct PieceTrie {
    first: Vec<u32>,
    labels: Vec<u8>,
    targets: Vec<u32>,
    piece: Vec<u32>,
}

impl PieceTrie {
    fn build(vocab: &[(String, f64)]) -> Self {
        let mut keys = vocab
            .iter()
            .zip(0_u32..)
            .map(|((piece, _), id)| (piece.as_bytes(), id))
            .collect::<Vec<_>>();
        keys.sort_unstable();
        // A repeated piece keeps its last id, as the library's `token_to_ids`
        // map does when it is filled in vocabulary order.
        keys.dedup_by(|later, earlier| {
            let same = later.0 == earlier.0;
            if same {
                earlier.1 = later.1;
            }
            same
        });

        let mut first = vec![0_u32];
        let mut labels = Vec::new();
        let mut targets = Vec::new();
        let mut piece = Vec::new();
        let mut next_node = 1_u32;
        // Breadth-first, so every node's children get consecutive ids and
        // nodes are finished in id order.
        let mut pending = VecDeque::from([(0, keys.len(), 0)]);
        while let Some((mut low, high, depth)) = pending.pop_front() {
            // The key equal to this node's prefix sorts first in its range.
            let mut ends_here = 0;
            if low < high && keys[low].0.len() == depth {
                ends_here = keys[low].1 + 1;
                low += 1;
            }
            piece.push(ends_here);
            while low < high {
                let label = keys[low].0[depth];
                let mut end = low + 1;
                while end < high && keys[end].0[depth] == label {
                    end += 1;
                }
                labels.push(label);
                targets.push(next_node);
                next_node += 1;
                pending.push_back((low, end, depth + 1));
                low = end;
            }
            first.push(u32::try_from(labels.len()).unwrap_or(u32::MAX));
        }
        Self {
            first,
            labels,
            targets,
            piece,
        }
    }

    fn child(&self, node: u32, label: u8) -> Option<u32> {
        let low = self.first[node as usize] as usize;
        let high = self.first[node as usize + 1] as usize;
        self.labels[low..high]
            .binary_search(&label)
            .ok()
            .map(|offset| self.targets[low + offset])
    }

    fn piece_id(&self, node: u32) -> Option<u32> {
        self.piece[node as usize].checked_sub(1)
    }

    fn exact(&self, bytes: &[u8]) -> Option<u32> {
        let mut node = 0;
        for &label in bytes {
            node = self.child(node, label)?;
        }
        self.piece_id(node)
    }
}

/// Load-only Unigram model with the library's exact segmentation.
pub(super) struct CompactUnigram {
    vocab: Vec<(String, f64)>,
    trie: PieceTrie,
    min_score: f64,
    unk_id: Option<usize>,
    byte_fallback: bool,
}

impl CompactUnigram {
    fn new(
        vocab: Vec<(String, f64)>,
        unk_id: Option<usize>,
        byte_fallback: bool,
    ) -> Result<Self, String> {
        if let Some(unk_id) = unk_id {
            if vocab.is_empty() {
                return Err(UnigramError::EmptyVocabulary.to_string());
            }
            if unk_id >= vocab.len() {
                return Err(UnigramError::UnkIdNotInVocabulary.to_string());
            }
        }
        if u32::try_from(vocab.len()).is_err() {
            return Err("vocabulary does not fit 32-bit piece ids".to_owned());
        }
        let mut min_score = f64::INFINITY;
        for (_, score) in &vocab {
            if *score < min_score {
                min_score = *score;
            }
        }
        Ok(Self {
            trie: PieceTrie::build(&vocab),
            vocab,
            min_score,
            unk_id,
            byte_fallback,
        })
    }

    /// Port of `Unigram::encode_optimized` (the library's default path).
    fn encode(&self, sentence: &str) -> tokenizers::Result<Vec<String>> {
        #[derive(Clone, Default)]
        struct BestPathNode {
            id: usize,
            best_path_score: f64,
            starts_at: Option<usize>,
        }

        if sentence.is_empty() {
            return Ok(Vec::new());
        }
        let size = sentence.len();
        let bytes = sentence.as_bytes();
        let unk_score = self.min_score - UNK_PENALTY;
        let mut best_path_ends_at = vec![BestPathNode::default(); size + 1];
        let mut starts_at = 0;
        while starts_at < size {
            let best_path_score_till_here = best_path_ends_at[starts_at].best_path_score;
            let mut has_single_node = false;
            let mblen = sentence[starts_at..]
                .chars()
                .next()
                .map_or(1, char::len_utf8);
            // Pieces that prefix the rest of the sentence, shortest first,
            // exactly as the library's trie search yields them.
            let mut node = 0;
            for (length, &label) in (1..).zip(&bytes[starts_at..]) {
                let Some(child) = self.trie.child(node, label) else {
                    break;
                };
                node = child;
                let Some(id) = self.trie.piece_id(node) else {
                    continue;
                };
                let id = id as usize;
                let target_node = &mut best_path_ends_at[starts_at + length];
                let candidate_best_path_score = self.vocab[id].1 + best_path_score_till_here;
                if target_node.starts_at.is_none()
                    || candidate_best_path_score > target_node.best_path_score
                {
                    target_node.best_path_score = candidate_best_path_score;
                    target_node.starts_at = Some(starts_at);
                    target_node.id = id;
                }
                if !has_single_node && length == mblen {
                    has_single_node = true;
                }
            }
            if !has_single_node {
                let target_node = &mut best_path_ends_at[starts_at + mblen];
                let candidate_best_path_score = unk_score + best_path_score_till_here;
                if target_node.starts_at.is_none()
                    || candidate_best_path_score > target_node.best_path_score
                {
                    target_node.best_path_score = candidate_best_path_score;
                    target_node.starts_at = Some(starts_at);
                    target_node.id = self.unk_id.ok_or(UnigramError::MissingUnkId)?;
                }
            }
            starts_at += mblen;
        }

        // Backtrack, fusing consecutive unknown pieces (`fuse_unk` is always
        // on for a deserialized library Unigram).
        let mut ends_at = size;
        let mut results = Vec::new();
        let mut token = Vec::new();
        while ends_at > 0 {
            let node = &best_path_ends_at[ends_at];
            let starts_at = node.starts_at.ok_or(UnigramError::MissingUnkId)?;
            if Some(node.id) == self.unk_id {
                token.push(sentence[starts_at..ends_at].to_owned());
            } else {
                if !token.is_empty() {
                    token.reverse();
                    results.push(token.concat());
                    token.clear();
                }
                results.push(sentence[starts_at..ends_at].to_owned());
            }
            ends_at = starts_at;
        }
        if !token.is_empty() {
            token.reverse();
            results.push(token.concat());
        }
        results.reverse();
        Ok(results)
    }
}

impl Model for CompactUnigram {
    type Trainer = UnigramTrainer;

    /// Port of the library `Unigram::tokenize`.
    fn tokenize(&self, sentence: &str) -> tokenizers::Result<Vec<Token>> {
        let str_tokens = self.encode(sentence)?;
        let mut offset = 0;
        let mut tokens = Vec::with_capacity(str_tokens.len());
        for string in str_tokens {
            let len = string.len();
            let offsets = (offset, offset + len);
            let id = if let Some(id) = self.token_to_id(&string) {
                id
            } else {
                if self.byte_fallback {
                    let byte_tokens = string
                        .bytes()
                        .map(|byte| {
                            let byte_string = format!("<0x{byte:02X}>");
                            self.token_to_id(&byte_string)
                                .map(|id| Token::new(id, byte_string, (offset, offset + len)))
                        })
                        .collect::<Option<Vec<_>>>();
                    if let Some(byte_tokens) = byte_tokens {
                        tokens.extend(byte_tokens);
                        offset += len;
                        continue;
                    }
                }
                let unk_id = self.unk_id.ok_or(UnigramError::MissingUnkId)?;
                u32::try_from(unk_id)?
            };
            offset += len;
            tokens.push(Token::new(id, string, offsets));
        }
        Ok(tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.trie.exact(token.as_bytes())
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.get(id as usize).map(|(piece, _)| piece.clone())
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        // Later ids overwrite earlier ones for a repeated piece.
        self.vocab
            .iter()
            .zip(0_u32..)
            .map(|((piece, _), id)| (piece.clone(), id))
            .collect()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn save(&self, _folder: &Path, _prefix: Option<&str>) -> tokenizers::Result<Vec<PathBuf>> {
        Err("the compact Unigram model is load-only".into())
    }

    fn get_trainer(&self) -> UnigramTrainer {
        UnigramTrainer::default()
    }
}

impl<'de> Deserialize<'de> for CompactUnigram {
    /// Accepts exactly what the library `Unigram` deserializer accepts.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Fields {
            #[serde(rename = "type", default)]
            kind: Option<String>,
            #[serde(default)]
            vocab: Option<Vec<(String, f64)>>,
            #[serde(default)]
            unk_id: Option<usize>,
            #[serde(default)]
            byte_fallback: bool,
        }

        let fields = Fields::deserialize(deserializer)?;
        if let Some(kind) = fields.kind.as_deref()
            && kind != "Unigram"
        {
            return Err(serde::de::Error::invalid_value(
                serde::de::Unexpected::Str(kind),
                &"Unigram",
            ));
        }
        let vocab = fields
            .vocab
            .ok_or_else(|| serde::de::Error::custom("Missing vocab"))?;
        Self::new(vocab, fields.unk_id, fields.byte_fallback)
            .map_err(|error| serde::de::Error::custom(format!("Unable to load vocab {error}")))
    }
}

/// Whether a `tokenizer.json` document's model is a Unigram, without
/// materializing its vocabulary.
pub(super) fn is_unigram_tokenizer(content: &str) -> bool {
    #[derive(Deserialize)]
    struct Document {
        model: ModelTag,
    }
    #[derive(Deserialize)]
    struct ModelTag {
        #[serde(rename = "type")]
        kind: Option<String>,
    }
    serde_json::from_str::<Document>(content)
        .is_ok_and(|document| document.model.kind.as_deref() == Some("Unigram"))
}

#[cfg(test)]
mod tests {
    use super::{CompactUnigram, CompactUnigramTokenizer, is_unigram_tokenizer};
    use std::str::FromStr;
    use tokenizers::models::unigram::Unigram;
    use tokenizers::{Model, Tokenizer};

    /// Deterministic generator for reproducible adversarial inputs.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }

        fn below(&mut self, bound: usize) -> usize {
            usize::try_from(self.next()).unwrap_or(0) % bound
        }
    }

    /// A vocabulary that exercises ties (several scores coincide, so the
    /// strict `>` and the shortest-first search order decide), multi-byte
    /// pieces, a repeated piece, pieces that only a fused unknown run can
    /// spell, and byte-fallback pieces.
    fn vocab() -> Vec<(String, f64)> {
        let mut vocab = vec![("<unk>".to_owned(), 0.0)];
        let pieces = [
            ("a", -1.0),
            ("b", -1.0),
            ("ab", -2.0),
            ("abc", -2.0),
            ("bc", -1.5),
            ("c", -2.0),
            ("▁", -0.5),
            ("▁a", -1.0),
            ("▁ab", -2.5),
            ("é", -3.0),
            ("éa", -3.0),
            ("日本", -4.0),
            ("日", -5.0),
            ("本語", -4.0),
            ("語", -5.0),
            ("🦀", -6.0),
            ("xy", -1.0),
            ("ab", -3.5),
            ("q", -7.0),
            ("<0x7A>", -8.0),
            ("<0xC3>", -8.0),
            ("<0xB1>", -8.0),
        ];
        vocab.extend(
            pieces
                .iter()
                .map(|(piece, score)| ((*piece).to_owned(), *score)),
        );
        vocab
    }

    fn random_sentence(rng: &mut Lcg) -> String {
        const ALPHABET: [&str; 14] = [
            "a", "b", "c", "▁", "é", "日", "本", "語", "🦀", "x", "y", "z", "ñ", "q",
        ];
        let length = rng.below(24);
        (0..length)
            .map(|_| ALPHABET[rng.below(ALPHABET.len())])
            .collect()
    }

    #[test]
    fn segmentation_matches_the_library_unigram() {
        for byte_fallback in [false, true] {
            let library = Unigram::from(vocab(), Some(0), byte_fallback).unwrap();
            let compact = CompactUnigram::new(vocab(), Some(0), byte_fallback).unwrap();
            let mut rng = Lcg(0x5eed);
            for _ in 0..20_000 {
                let sentence = random_sentence(&mut rng);
                assert_eq!(
                    compact.tokenize(&sentence).unwrap(),
                    library.tokenize(&sentence).unwrap(),
                    "sentence {sentence:?}, byte_fallback {byte_fallback}"
                );
            }
            for (piece, _) in vocab() {
                assert_eq!(compact.token_to_id(&piece), library.token_to_id(&piece));
            }
            assert_eq!(compact.token_to_id("abcd"), None);
            assert_eq!(compact.get_vocab(), library.get_vocab());
        }
    }

    #[test]
    fn repeated_piece_keeps_its_last_id() {
        let compact = CompactUnigram::new(vocab(), Some(0), false).unwrap();
        // "ab" is listed at ids 3 and 18.
        assert_eq!(compact.token_to_id("ab"), Some(18));
        let library = Unigram::from(vocab(), Some(0), false).unwrap();
        assert_eq!(library.token_to_id("ab"), Some(18));
    }

    #[test]
    fn construction_refuses_what_the_library_refuses() {
        assert!(CompactUnigram::new(Vec::new(), Some(0), false).is_err());
        assert!(CompactUnigram::new(vocab(), Some(99), false).is_err());
        let without_unk = CompactUnigram::new(vec![("a".to_owned(), -1.0)], None, false).unwrap();
        assert!(without_unk.tokenize("a").is_ok());
        assert!(
            without_unk.tokenize("b").is_err(),
            "unknown text needs an unk id"
        );
    }

    fn tokenizer_json(model_type: &str) -> String {
        let vocab = vocab()
            .into_iter()
            .map(|(piece, score)| serde_json::json!([piece, score]))
            .collect::<Vec<_>>();
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [{
                "id": 0, "content": "<unk>", "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true
            }],
            "normalizer": {"type": "NFKC"},
            "pre_tokenizer": {
                "type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": false
            },
            "post_processor": null,
            "decoder": null,
            "model": {"type": model_type, "unk_id": 0, "vocab": vocab, "byte_fallback": false}
        })
        .to_string()
    }

    #[test]
    fn whole_tokenizer_matches_the_library_pipeline() {
        let json = tokenizer_json("Unigram");
        assert!(is_unigram_tokenizer(&json));
        let library = Tokenizer::from_str(&json).unwrap();
        let compact: CompactUnigramTokenizer = serde_json::from_str(&json).unwrap();
        let mut rng = Lcg(0xfeed);
        for round in 0..5_000 {
            let mut text = random_sentence(&mut rng).replace('▁', " ");
            if round % 7 == 0 {
                text.push_str(" <unk> ab");
            }
            for add_special_tokens in [false, true] {
                let expected = library
                    .encode_fast(text.as_str(), add_special_tokens)
                    .unwrap();
                let observed = compact
                    .encode_fast(text.as_str(), add_special_tokens)
                    .unwrap();
                assert_eq!(observed.get_ids(), expected.get_ids(), "text {text:?}");
                let expected = library.encode(text.as_str(), add_special_tokens).unwrap();
                let observed = compact.encode(text.as_str(), add_special_tokens).unwrap();
                assert_eq!(observed.get_ids(), expected.get_ids(), "text {text:?}");
                assert_eq!(
                    observed.get_tokens(),
                    expected.get_tokens(),
                    "text {text:?}"
                );
                assert_eq!(
                    observed.get_offsets(),
                    expected.get_offsets(),
                    "text {text:?}"
                );
            }
        }
    }

    /// The real multilingual vocabulary: identical ids for a spread of
    /// scripts and hostile inputs, plus every line of the file named by
    /// `COMPACT_UNIGRAM_PARITY_TEXTS` when it is set.
    #[test]
    #[ignore = "requires the potion-multilingual-128M tokenizer via POTION_FIXTURE_DIR"]
    fn real_potion_tokenizer_matches_the_library() {
        let dir = std::env::var("POTION_FIXTURE_DIR")
            .expect("POTION_FIXTURE_DIR must name the potion-multilingual-128M directory");
        let content =
            std::fs::read_to_string(std::path::Path::new(&dir).join("tokenizer.json")).unwrap();
        assert!(is_unigram_tokenizer(&content));
        let library = Tokenizer::from_str(&content).unwrap();
        let compact: CompactUnigramTokenizer = serde_json::from_str(&content).unwrap();
        let mut texts = [
            "hello world",
            "semantic search finds related ideas",
            "naïve café Zürich — Straße, déjà vu",
            "Съешь же ещё этих мягких французских булок",
            "Ζεῦ πάτερ, ἄγγελος",
            "مرحبا بالعالم، هذا اختبار",
            "שלום עולם",
            "नमस्ते दुनिया",
            "東京都の天気は晴れです。日本語の文章",
            "我们使用多语言模型进行检索",
            "안녕하세요 세계",
            "สวัสดีชาวโลก",
            "🦀🔥 emoji 👩‍👩‍👧 zero\u{200d}width and e\u{301} combining",
            "fn main() { println!(\"{}\", x_y_z); } // fsvi_v2 0xDEADBEEF",
            "https://example.com/a/b?c=d&e=f#frag user@example.org",
            "\t tabs\nnewlines\r\n  multiple   spaces  ",
            "[UNK] [PAD] <unk> literal special tokens",
            "𐍈𐌰𐌿𐍃 ᚠᚢᚦ ⅷ ① ℃ ﬁ",
        ]
        .map(str::to_owned)
        .to_vec();
        texts.push("ꙮ".repeat(300));
        if let Ok(path) = std::env::var("COMPACT_UNIGRAM_PARITY_TEXTS") {
            texts.extend(
                std::fs::read_to_string(path)
                    .unwrap()
                    .lines()
                    .map(str::to_owned),
            );
        }
        for text in &texts {
            let expected = library.encode_fast(text.as_str(), false).unwrap();
            let observed = compact.encode_fast(text.as_str(), false).unwrap();
            assert_eq!(observed.get_ids(), expected.get_ids(), "text {text:?}");
        }
        println!(
            "compact Unigram matched the library on {} texts",
            texts.len()
        );
    }

    #[test]
    fn only_unigram_documents_take_the_compact_route() {
        assert!(!is_unigram_tokenizer(&tokenizer_json("WordPiece")));
        assert!(!is_unigram_tokenizer("{\"model\": {}}"));
        assert!(!is_unigram_tokenizer("not json"));
        assert!(serde_json::from_str::<CompactUnigramTokenizer>(&tokenizer_json("BPE")).is_err());
    }
}
