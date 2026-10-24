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

#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};
use tokenizers::models::unigram::{UnigramError, UnigramTrainer};
use tokenizers::{
    DecoderWrapper, Model, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper, Token,
    TokenizerImpl,
};

/// Score penalty for an unknown character (`K_UNK_PENALTY` in `tokenizers`).
const UNK_PENALTY: f64 = 10.0;

/// Filename of the persisted CSR `PieceTrie` sidecar next to `tokenizer.json`.
pub(super) const UNIGRAM_TRIE_SIDECAR_FILENAME: &str = ".tokenizer.unigram-trie.v1";

const TRIE_SIDECAR_MAGIC: &[u8; 8] = b"FSUTRIE1";
const TRIE_SIDECAR_VERSION: u32 = 1;
const TRIE_SIDECAR_HEADER_BYTES: usize = 88;

thread_local! {
    static ACTIVE_SIDECAR_PATH: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
    #[cfg(test)]
    static SIDECAR_HITS: Cell<usize> = const { Cell::new(0) };
    #[cfg(test)]
    static SIDECAR_BUILDS: Cell<usize> = const { Cell::new(0) };
}

struct SidecarPathGuard {
    previous: Option<PathBuf>,
}

impl SidecarPathGuard {
    fn set(path: Option<PathBuf>) -> Self {
        let previous = ACTIVE_SIDECAR_PATH.with(|slot| slot.replace(path));
        Self { previous }
    }
}

impl Drop for SidecarPathGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        ACTIVE_SIDECAR_PATH.with(|slot| {
            slot.replace(previous);
        });
    }
}

/// Deserialize a [`CompactUnigramTokenizer`] from `content`, consulting and
/// populating `sidecar_path` when supplied.
pub(super) fn deserialize_unigram_tokenizer(
    content: &str,
    sidecar_path: Option<&Path>,
) -> tokenizers::Result<CompactUnigramTokenizer> {
    let _guard = SidecarPathGuard::set(sidecar_path.map(Path::to_path_buf));
    Ok(serde_json::from_str(content)?)
}

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
#[derive(Debug, PartialEq, Eq)]
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

    fn vocab_digest(vocab: &[(String, f64)]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.compact_unigram.vocab.v1\0");
        hasher.update(u64::try_from(vocab.len()).unwrap_or(u64::MAX).to_le_bytes());
        for (piece, _) in vocab {
            hasher.update(u32::try_from(piece.len()).unwrap_or(u32::MAX).to_le_bytes());
            hasher.update(piece.as_bytes());
        }
        hasher.finalize().into()
    }

    fn encode_sidecar(&self, vocab_digest: &[u8; 32], vocab_len: u32) -> Option<Vec<u8>> {
        let node_count = u32::try_from(self.piece.len()).ok()?;
        if self.first.len() != self.piece.len().checked_add(1)?
            || self.labels.len() != self.targets.len()
        {
            return None;
        }
        let edge_count = u32::try_from(self.labels.len()).ok()?;
        let payload_len = self
            .first
            .len()
            .checked_mul(4)?
            .checked_add(self.labels.len())?
            .checked_add(self.targets.len().checked_mul(4)?)?
            .checked_add(self.piece.len().checked_mul(4)?)?;
        let total_len = TRIE_SIDECAR_HEADER_BYTES.checked_add(payload_len)?;
        let mut out = Vec::with_capacity(total_len);
        out.resize(TRIE_SIDECAR_HEADER_BYTES, 0);
        for &value in &self.first {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&self.labels);
        for &value in &self.targets {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for &value in &self.piece {
            out.extend_from_slice(&value.to_le_bytes());
        }
        let payload_sha256: [u8; 32] = Sha256::digest(&out[TRIE_SIDECAR_HEADER_BYTES..]).into();
        out[0..8].copy_from_slice(TRIE_SIDECAR_MAGIC);
        out[8..12].copy_from_slice(&TRIE_SIDECAR_VERSION.to_le_bytes());
        out[12..16].copy_from_slice(&vocab_len.to_le_bytes());
        out[16..20].copy_from_slice(&node_count.to_le_bytes());
        out[20..24].copy_from_slice(&edge_count.to_le_bytes());
        out[24..56].copy_from_slice(vocab_digest);
        out[56..88].copy_from_slice(&payload_sha256);
        Some(out)
    }

    fn decode_sidecar(
        bytes: &[u8],
        expected_vocab_digest: &[u8; 32],
        expected_vocab_len: u32,
    ) -> Option<Self> {
        if bytes.len() < TRIE_SIDECAR_HEADER_BYTES || &bytes[0..8] != TRIE_SIDECAR_MAGIC {
            return None;
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
        if version != TRIE_SIDECAR_VERSION {
            return None;
        }
        let vocab_len = u32::from_le_bytes(bytes[12..16].try_into().ok()?);
        if vocab_len != expected_vocab_len {
            return None;
        }
        let node_count = u32::from_le_bytes(bytes[16..20].try_into().ok()?);
        if node_count == 0 {
            return None;
        }
        let edge_count = u32::from_le_bytes(bytes[20..24].try_into().ok()?);
        // The builder numbers each newly discovered child consecutively.
        // Require a tree, not merely a forward-only DAG with aliased paths.
        if edge_count != node_count.checked_sub(1)? {
            return None;
        }
        if &bytes[24..56] != expected_vocab_digest.as_slice() {
            return None;
        }
        let node_len = usize::try_from(node_count).ok()?;
        let first_len = node_len.checked_add(1)?;
        let edge_len = usize::try_from(edge_count).ok()?;
        let first_bytes = first_len.checked_mul(4)?;
        let labels_bytes = edge_len;
        let targets_bytes = edge_len.checked_mul(4)?;
        let piece_bytes = node_len.checked_mul(4)?;
        let payload_len = first_bytes
            .checked_add(labels_bytes)?
            .checked_add(targets_bytes)?
            .checked_add(piece_bytes)?;
        let total_len = TRIE_SIDECAR_HEADER_BYTES.checked_add(payload_len)?;
        if bytes.len() != total_len {
            return None;
        }
        let payload = &bytes[TRIE_SIDECAR_HEADER_BYTES..];
        let actual_payload_sha256: [u8; 32] = Sha256::digest(payload).into();
        if &bytes[56..88] != actual_payload_sha256.as_slice() {
            return None;
        }
        let (first_raw, rest) = payload.split_at(first_bytes);
        let (labels_raw, rest) = rest.split_at(labels_bytes);
        let (targets_raw, piece_raw) = rest.split_at(targets_bytes);
        let first = first_raw
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        let labels = labels_raw.to_vec();
        let targets = targets_raw
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        let piece = piece_raw
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();

        if !targets.iter().copied().eq(1..node_count) {
            return None;
        }
        if *first.first()? != 0 || *first.last()? != edge_count {
            return None;
        }
        for node_idx in 0..node_len {
            let low = first[node_idx];
            let high = first[node_idx + 1];
            if low > high || high > edge_count || piece[node_idx] > expected_vocab_len {
                return None;
            }
            if node_idx != 0 && low == high && piece[node_idx] == 0 {
                return None;
            }
            let low_usize = usize::try_from(low).ok()?;
            let high_usize = usize::try_from(high).ok()?;
            if labels[low_usize..high_usize]
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                return None;
            }
            let node_u32 = u32::try_from(node_idx).ok()?;
            if targets[low_usize..high_usize]
                .iter()
                .any(|&target| target <= node_u32 || target >= node_count)
            {
                return None;
            }
        }
        Some(Self {
            first,
            labels,
            targets,
            piece,
        })
    }

    fn load_or_build(vocab: &[(String, f64)], sidecar_path: Option<&Path>) -> Self {
        let Some(path) = sidecar_path else {
            return Self::build(vocab);
        };
        let Ok(vocab_len) = u32::try_from(vocab.len()) else {
            return Self::build(vocab);
        };
        let vocab_digest = Self::vocab_digest(vocab);
        if let Some(max_bytes) = Self::max_sidecar_bytes(vocab)
            && let Some(bytes) = Self::read_sidecar(path, max_bytes)
            && let Some(trie) = Self::decode_sidecar(&bytes, &vocab_digest, vocab_len)
            && trie.matches_vocab(vocab)
        {
            #[cfg(test)]
            SIDECAR_HITS.with(|hits| hits.set(hits.get() + 1));
            return trie;
        }
        #[cfg(test)]
        SIDECAR_BUILDS.with(|builds| builds.set(builds.get() + 1));
        let trie = Self::build(vocab);
        if let Some(encoded) = trie.encode_sidecar(&vocab_digest, vocab_len) {
            let _ = Self::persist_sidecar(path, &encoded);
        }
        trie
    }

    /// Checksums detect damaged bytes, not a cache which consistently asserts
    /// the wrong trie for a vocabulary. Prove both directions without building
    /// another trie or allocating a vocabulary map. Structural admission has
    /// already established one path per node and no nonterminal leaves.
    fn matches_vocab(&self, vocab: &[(String, f64)]) -> bool {
        for (id, (text, _)) in vocab.iter().enumerate() {
            let Some(actual) = self.exact(text.as_bytes()) else {
                return false;
            };
            let Ok(actual) = usize::try_from(actual) else {
                return false;
            };
            // Repeated pieces keep the LAST vocabulary id. Requiring >= id
            // on every occurrence, plus equality of the referenced bytes,
            // establishes that contract without a duplicate-tracking map.
            if actual < id || !vocab.get(actual).is_some_and(|(value, _)| value == text) {
                return false;
            }
        }
        for (node, &stored) in self.piece.iter().enumerate() {
            let Some(id) = stored.checked_sub(1) else {
                continue;
            };
            let Ok(id) = usize::try_from(id) else {
                return false;
            };
            let Some((text, _)) = vocab.get(id) else {
                return false;
            };
            // Reject invented prefixes/aliases too, including terminals inside
            // a UTF-8 code point. Checking only real vocabulary lookups would
            // miss these additional tokenization candidates.
            if self
                .node_for(text.as_bytes())
                .and_then(|n| usize::try_from(n).ok())
                != Some(node)
            {
                return false;
            }
        }
        true
    }

    /// Each vocabulary byte can add at most one node and one edge. A root
    /// alone occupies 12 payload bytes; every added node/edge occupies 13.
    /// Derive the read ceiling from authoritative input, not cache headers.
    fn max_sidecar_bytes(vocab: &[(String, f64)]) -> Option<usize> {
        vocab.iter().try_fold(
            TRIE_SIDECAR_HEADER_BYTES.checked_add(12)?,
            |size, (piece, _)| size.checked_add(piece.len().checked_mul(13)?),
        )
    }

    fn read_sidecar(path: &Path, max_bytes: usize) -> Option<Vec<u8>> {
        let limit = u64::try_from(max_bytes).ok()?;
        let metadata = std::fs::symlink_metadata(path).ok()?;
        if !metadata.is_file() || metadata.len() > limit {
            return None;
        }
        let file = std::fs::File::open(path).ok()?;
        let metadata = file.metadata().ok()?;
        if !metadata.is_file() || metadata.len() > limit {
            return None;
        }
        // Limit the read itself too: a regular file can grow after metadata.
        let mut bytes = Vec::new();
        file.take(limit.checked_add(1)?)
            .read_to_end(&mut bytes)
            .ok()?;
        (bytes.len() <= max_bytes).then_some(bytes)
    }

    /// Publish a complete cache file by replacing its directory entry, never
    /// by truncating an existing inode (which may be hard-linked or in use).
    /// Keep ordinary symlinks and special entries untouched. The model
    /// directory is caller-controlled; this is not a hostile-directory lease.
    /// A cache write failure never prevents using the freshly built trie.
    fn persist_sidecar(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        match std::fs::symlink_metadata(path) {
            Ok(metadata) if !metadata.is_file() => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "tokenizer cache destination is not a regular file",
                ));
            }
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(error),
            _ => {}
        }
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let mut staged = tempfile::Builder::new()
            .prefix(".unigram-trie-")
            .tempfile_in(parent)?;
        staged.write_all(bytes)?;
        staged.as_file().sync_all()?;
        staged.persist(path).map_err(|error| error.error)?;
        // The directory is not synced: this disposable cache can disappear
        // after a crash. No model artifact or durable publication depends on it.
        Ok(())
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

    fn node_for(&self, bytes: &[u8]) -> Option<u32> {
        let mut node = 0;
        for &label in bytes {
            node = self.child(node, label)?;
        }
        Some(node)
    }

    fn exact(&self, bytes: &[u8]) -> Option<u32> {
        self.piece_id(self.node_for(bytes)?)
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
        let sidecar_path = ACTIVE_SIDECAR_PATH.with(|slot| slot.borrow().clone());
        Ok(Self {
            trie: PieceTrie::load_or_build(&vocab, sidecar_path.as_deref()),
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

    #[test]
    fn piece_trie_sidecar_round_trip_and_tamper_invalidation() {
        use super::{PieceTrie, TRIE_SIDECAR_HEADER_BYTES};
        use sha2::{Digest, Sha256};

        let sample_vocab = vocab();
        let vocab_len = u32::try_from(sample_vocab.len()).unwrap();
        let vocab_digest = PieceTrie::vocab_digest(&sample_vocab);
        let built = PieceTrie::build(&sample_vocab);
        let encoded = built
            .encode_sidecar(&vocab_digest, vocab_len)
            .expect("sidecar should encode");

        let decoded = PieceTrie::decode_sidecar(&encoded, &vocab_digest, vocab_len)
            .expect("valid sidecar should decode");
        assert_eq!(decoded, built);

        // Wrong vocabulary digest or length is rejected.
        let mut wrong_digest = vocab_digest;
        wrong_digest[0] ^= 0xFF;
        assert!(PieceTrie::decode_sidecar(&encoded, &wrong_digest, vocab_len).is_none());
        assert!(PieceTrie::decode_sidecar(&encoded, &vocab_digest, vocab_len + 1).is_none());

        // Truncated or magic-tampered bytes are rejected.
        assert!(
            PieceTrie::decode_sidecar(&encoded[..encoded.len() - 1], &vocab_digest, vocab_len)
                .is_none()
        );
        let mut bad_magic = encoded.clone();
        bad_magic[0] ^= 0x01;
        assert!(PieceTrie::decode_sidecar(&bad_magic, &vocab_digest, vocab_len).is_none());

        // Flipping a payload byte without updating payload_sha256 is rejected.
        let mut flipped_payload = encoded;
        let last_idx = flipped_payload.len() - 1;
        flipped_payload[last_idx] ^= 0x01;
        assert!(PieceTrie::decode_sidecar(&flipped_payload, &vocab_digest, vocab_len).is_none());

        // Structural invariant gate: even with a valid payload_sha256, an out-of-bounds piece id
        // or cyclic target is rejected by decode_sidecar.
        let mut invalid_trie = PieceTrie::build(&sample_vocab);
        invalid_trie.piece[0] = vocab_len + 1;
        let mut invalid_encoded = invalid_trie
            .encode_sidecar(&vocab_digest, vocab_len)
            .unwrap();
        let recomputed_sha: [u8; 32] =
            Sha256::digest(&invalid_encoded[TRIE_SIDECAR_HEADER_BYTES..]).into();
        invalid_encoded[56..88].copy_from_slice(&recomputed_sha);
        assert!(PieceTrie::decode_sidecar(&invalid_encoded, &vocab_digest, vocab_len).is_none());
    }

    #[test]
    fn deserialize_unigram_tokenizer_persists_and_reuses_sidecar_and_recovers_from_corruption() {
        use super::{
            SIDECAR_BUILDS, SIDECAR_HITS, UNIGRAM_TRIE_SIDECAR_FILENAME,
            deserialize_unigram_tokenizer,
        };

        let dir = tempfile::tempdir().unwrap();
        let sidecar_path = dir.path().join(UNIGRAM_TRIE_SIDECAR_FILENAME);
        let json = tokenizer_json("Unigram");

        let hits_before = SIDECAR_HITS.with(std::cell::Cell::get);
        let builds_before = SIDECAR_BUILDS.with(std::cell::Cell::get);

        let first = deserialize_unigram_tokenizer(&json, Some(&sidecar_path)).unwrap();
        assert!(sidecar_path.exists(), "first load should write sidecar");
        assert_eq!(SIDECAR_HITS.with(std::cell::Cell::get), hits_before);
        assert_eq!(SIDECAR_BUILDS.with(std::cell::Cell::get), builds_before + 1);

        let second = deserialize_unigram_tokenizer(&json, Some(&sidecar_path)).unwrap();
        assert_eq!(SIDECAR_HITS.with(std::cell::Cell::get), hits_before + 1);
        assert_eq!(SIDECAR_BUILDS.with(std::cell::Cell::get), builds_before + 1);

        let probe = "ab ▁ab 日本語 🦀 unknown_word";
        assert_eq!(
            first.encode_fast(probe, false).unwrap().get_ids(),
            second.encode_fast(probe, false).unwrap().get_ids()
        );

        // Corrupt the sidecar on disk: load must fall back to PieceTrie::build and repair it.
        std::fs::write(&sidecar_path, b"corrupted-sidecar-bytes").unwrap();
        let third = deserialize_unigram_tokenizer(&json, Some(&sidecar_path)).unwrap();
        assert_eq!(SIDECAR_HITS.with(std::cell::Cell::get), hits_before + 1);
        assert_eq!(SIDECAR_BUILDS.with(std::cell::Cell::get), builds_before + 2);
        assert_eq!(
            first.encode_fast(probe, false).unwrap().get_ids(),
            third.encode_fast(probe, false).unwrap().get_ids()
        );

        // Subsequent load hits the repaired sidecar.
        let fourth = deserialize_unigram_tokenizer(&json, Some(&sidecar_path)).unwrap();
        assert_eq!(SIDECAR_HITS.with(std::cell::Cell::get), hits_before + 2);
        assert_eq!(
            first.encode_fast(probe, false).unwrap().get_ids(),
            fourth.encode_fast(probe, false).unwrap().get_ids()
        );
    }

    #[test]
    fn sidecar_read_ceiling_comes_from_vocabulary_bytes() {
        use super::PieceTrie;

        assert_eq!(PieceTrie::max_sidecar_bytes(&[]), Some(100));
        let sample = vec![("é\0中".to_owned(), -1.0)];
        // Six UTF-8 bytes, not three Unicode characters.
        assert_eq!(PieceTrie::max_sidecar_bytes(&sample), Some(178));
        let digest = PieceTrie::vocab_digest(&sample);
        let encoded = PieceTrie::build(&sample)
            .encode_sidecar(&digest, 1)
            .unwrap();
        assert_eq!(encoded.len(), 178);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache");
        std::fs::write(&path, &encoded).unwrap();
        assert_eq!(PieceTrie::read_sidecar(&path, 178), Some(encoded));
        assert!(PieceTrie::read_sidecar(&path, 177).is_none());
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(179)
            .unwrap();
        assert!(PieceTrie::read_sidecar(&path, 178).is_none());
    }

    #[test]
    fn unusable_sidecar_destinations_do_not_prevent_tokenizer_loading() {
        use super::{PieceTrie, deserialize_unigram_tokenizer};

        let dir = tempfile::tempdir().unwrap();
        let directory = dir.path().join("directory");
        std::fs::create_dir(&directory).unwrap();
        let missing = dir.path().join("missing-parent").join("cache");
        let json = tokenizer_json("Unigram");
        let expected: CompactUnigramTokenizer = serde_json::from_str(&json).unwrap();
        for path in [&directory, &missing] {
            assert!(PieceTrie::read_sidecar(path, 1024).is_none());
            assert!(PieceTrie::persist_sidecar(path, b"cache").is_err());
            let loaded = deserialize_unigram_tokenizer(&json, Some(path)).unwrap();
            assert_eq!(
                loaded.encode("ab 日本語", false).unwrap().get_ids(),
                expected.encode("ab 日本語", false).unwrap().get_ids()
            );
        }
        assert!(directory.is_dir());
        assert!(!missing.parent().unwrap().exists());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn sidecar_publication_preserves_existing_readers_and_hard_link_targets() {
        use super::PieceTrie;
        use std::io::Read;

        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.safetensors");
        let cache = dir.path().join("cache");
        let original = b"model artifact must not be truncated";
        std::fs::write(&model, original).unwrap();
        std::fs::hard_link(&model, &cache).unwrap();
        let mut retained = std::fs::File::open(&cache).unwrap();
        let sample = vocab();
        let trie = PieceTrie::load_or_build(&sample, Some(&cache));
        assert_eq!(trie, PieceTrie::build(&sample));
        assert_eq!(std::fs::read(&model).unwrap(), original);
        let mut old_bytes = Vec::new();
        retained.read_to_end(&mut old_bytes).unwrap();
        assert_eq!(old_bytes, original);
        let decoded = PieceTrie::decode_sidecar(
            &std::fs::read(&cache).unwrap(),
            &PieceTrie::vocab_digest(&sample),
            u32::try_from(sample.len()).unwrap(),
        )
        .unwrap();
        assert_eq!(decoded, trie);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_sidecars_neither_modify_nor_create_their_targets() {
        use super::PieceTrie;
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let existing = dir.path().join("model.safetensors");
        let absent = dir.path().join("absent-model");
        std::fs::write(&existing, b"authoritative model bytes").unwrap();
        for (name, target) in [("existing-link", &existing), ("dangling-link", &absent)] {
            let cache = dir.path().join(name);
            symlink(target, &cache).unwrap();
            assert!(PieceTrie::read_sidecar(&cache, 1024).is_none());
            assert_eq!(
                PieceTrie::load_or_build(&vocab(), Some(&cache)),
                PieceTrie::build(&vocab())
            );
            assert!(
                std::fs::symlink_metadata(&cache)
                    .unwrap()
                    .file_type()
                    .is_symlink()
            );
            assert_eq!(std::fs::read_link(&cache).unwrap(), *target);
        }
        assert_eq!(
            std::fs::read(&existing).unwrap(),
            b"authoritative model bytes"
        );
        assert!(!absent.exists());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 3);
    }

    #[test]
    fn vocabulary_admission_preserves_duplicates_empty_pieces_and_score_changes() {
        use super::PieceTrie;

        let mut sample = vocab();
        sample.push((String::new(), -3.0));
        sample.push(("ab".to_owned(), -4.0));
        for mut sample in [Vec::new(), vec![(String::new(), 0.0)], sample] {
            let built = PieceTrie::build(&sample);
            let digest = PieceTrie::vocab_digest(&sample);
            let len = u32::try_from(sample.len()).unwrap();
            let encoded = built.encode_sidecar(&digest, len).unwrap();
            let decoded = PieceTrie::decode_sidecar(&encoded, &digest, len).unwrap();
            assert!(decoded.matches_vocab(&sample));
            assert_eq!(decoded, built);
            // Scores are read from authoritative JSON, not from trie storage.
            for (_, score) in &mut sample {
                *score -= 7.0;
            }
            assert_eq!(PieceTrie::vocab_digest(&sample), digest);
            assert!(decoded.matches_vocab(&sample));
        }
    }

    #[test]
    fn checksummed_wrong_piece_ids_are_rebuilt_before_tokenization() {
        use super::{PieceTrie, SIDECAR_BUILDS, SIDECAR_HITS, deserialize_unigram_tokenizer};

        let sample = vocab();
        let digest = PieceTrie::vocab_digest(&sample);
        let len = u32::try_from(sample.len()).unwrap();
        let mut forged = PieceTrie::build(&sample);
        let node = usize::try_from(forged.node_for(b"a").unwrap()).unwrap();
        forged.piece[node] = forged.exact(b"b").unwrap() + 1;
        // Re-encoding recomputes a VALID payload hash and preserves the correct
        // vocabulary digest. Structural bounds alone accept this wrong mapping.
        let encoded = forged.encode_sidecar(&digest, len).unwrap();
        let decoded = PieceTrie::decode_sidecar(&encoded, &digest, len).unwrap();
        assert!(!decoded.matches_vocab(&sample));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache");
        std::fs::write(&path, encoded).unwrap();
        let hits = SIDECAR_HITS.with(std::cell::Cell::get);
        let builds = SIDECAR_BUILDS.with(std::cell::Cell::get);
        let json = tokenizer_json("Unigram");
        let loaded = deserialize_unigram_tokenizer(&json, Some(&path)).unwrap();
        assert_eq!(SIDECAR_HITS.with(std::cell::Cell::get), hits);
        assert_eq!(SIDECAR_BUILDS.with(std::cell::Cell::get), builds + 1);
        let library = Tokenizer::from_str(&json).unwrap();
        for text in ["a", "ab", "éa 日本語 🦀", "unknown_word"] {
            assert_eq!(
                loaded.encode(text, false).unwrap().get_ids(),
                library.encode(text, false).unwrap().get_ids()
            );
        }
        let repaired =
            PieceTrie::decode_sidecar(&std::fs::read(&path).unwrap(), &digest, len).unwrap();
        assert!(repaired.matches_vocab(&sample));
        assert_eq!(repaired, PieceTrie::build(&sample));
    }

    #[test]
    fn invented_terminals_cannot_hide_behind_correct_vocabulary_lookups() {
        use super::PieceTrie;

        let sample = vec![("ab".to_owned(), -1.0), ("é".to_owned(), -2.0)];
        let digest = PieceTrie::vocab_digest(&sample);
        for prefix in [b"".as_slice(), b"a".as_slice(), &[0xc3][..]] {
            let mut forged = PieceTrie::build(&sample);
            let node = usize::try_from(forged.node_for(prefix).unwrap()).unwrap();
            forged.piece[node] = 1;
            assert_eq!(forged.exact(b"ab"), Some(0));
            assert_eq!(forged.exact("é".as_bytes()), Some(1));
            let encoded = forged.encode_sidecar(&digest, 2).unwrap();
            let decoded = PieceTrie::decode_sidecar(&encoded, &digest, 2).unwrap();
            assert!(!decoded.matches_vocab(&sample));
        }
    }

    #[test]
    fn sidecar_cannot_restore_an_earlier_duplicate_piece_id() {
        use super::PieceTrie;

        let sample = vec![("ab".to_owned(), -1.0), ("ab".to_owned(), -7.0)];
        let digest = PieceTrie::vocab_digest(&sample);
        let mut forged = PieceTrie::build(&sample);
        assert_eq!(forged.exact(b"ab"), Some(1));
        let node = usize::try_from(forged.node_for(b"ab").unwrap()).unwrap();
        forged.piece[node] = 1;
        let encoded = forged.encode_sidecar(&digest, 2).unwrap();
        let decoded = PieceTrie::decode_sidecar(&encoded, &digest, 2).unwrap();
        // The bytes still match the referenced vocabulary entry and its path,
        // but choosing the older id would change both scoring and token ids.
        assert_eq!(decoded.exact(b"ab"), Some(0));
        assert!(!decoded.matches_vocab(&sample));
    }

    #[test]
    fn sidecar_structure_rejects_aliased_children_and_unreachable_or_dead_nodes() {
        use super::PieceTrie;

        let sample = vec![("a".to_owned(), -1.0), ("b".to_owned(), -2.0)];
        let digest = PieceTrie::vocab_digest(&sample);
        for corruption in ["alias", "unreachable", "dead", "swapped"] {
            let mut forged = PieceTrie::build(&sample);
            match corruption {
                "alias" => forged.targets[1] = forged.targets[0],
                "unreachable" => {
                    forged.piece.push(1);
                    forged.first.push(2);
                }
                "dead" => forged.piece[1] = 0,
                "swapped" => forged.targets.swap(0, 1),
                _ => unreachable!(),
            }
            let encoded = forged.encode_sidecar(&digest, 2).unwrap();
            assert!(
                PieceTrie::decode_sidecar(&encoded, &digest, 2).is_none(),
                "{corruption}"
            );
        }
    }

    #[test]
    fn admitted_warm_sidecar_matches_the_library_tokenizer_pipeline() {
        use super::{SIDECAR_HITS, deserialize_unigram_tokenizer};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache");
        let json = tokenizer_json("Unigram");
        let _cold = deserialize_unigram_tokenizer(&json, Some(&path)).unwrap();
        let hits = SIDECAR_HITS.with(std::cell::Cell::get);
        let warm = deserialize_unigram_tokenizer(&json, Some(&path)).unwrap();
        assert_eq!(SIDECAR_HITS.with(std::cell::Cell::get), hits + 1);
        let library = Tokenizer::from_str(&json).unwrap();
        let mut rng = Lcg(0x51decaf);
        for _ in 0..1_000 {
            let text = random_sentence(&mut rng);
            let expected = library.encode(text.as_str(), false).unwrap();
            let actual = warm.encode(text.as_str(), false).unwrap();
            assert_eq!(actual.get_ids(), expected.get_ids(), "{text:?}");
            assert_eq!(actual.get_tokens(), expected.get_tokens(), "{text:?}");
            assert_eq!(actual.get_offsets(), expected.get_offsets(), "{text:?}");
        }
    }
}
