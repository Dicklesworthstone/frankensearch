//! One control protocol over the existing single and sharded live selectors.
//! These variants never install parts independently or implement another lock,
//! publisher, source transaction, model loader or rollback authority.

use frankensearch::native_ann::builder::NativeHybridReopenLimits;
use frankensearch::native_ann::builder::live::{
    NativeHybridCandidate, NativeHybridSnapshot, NativeLiveHybridIndex,
};
use frankensearch::native_ann::builder::sharded::NativeShardedHybridReopenLimits;
use frankensearch::native_ann::builder::sharded::live::{
    NativeLiveShardedHybridIndex, NativeShardedCandidate, NativeShardedSnapshot,
};

use super::{
    ArtifactGenerationIdentityV1, Cx, GenerationComponentReceiptV1, MAX_DOCUMENTS, Result,
    Selection, bad, cohort, sharded,
};

pub enum Serving {
    Single(NativeLiveHybridIndex),
    Sharded(NativeLiveShardedHybridIndex),
}

impl Serving {
    pub fn new(cx: &Cx, index: sharded::Opened) -> Result<Self> {
        match index {
            sharded::Opened::Single(index) => {
                Ok(Self::Single(NativeLiveHybridIndex::new(cx, *index)?))
            }
            sharded::Opened::Sharded(index) => Ok(Self::Sharded(
                NativeLiveShardedHybridIndex::new(cx, *index)?,
            )),
        }
    }

    pub fn borrow(&self) -> Live<'_> {
        match self {
            Self::Single(live) => Live::Single(live),
            Self::Sharded(live) => Live::Sharded(live),
        }
    }
}

#[derive(Clone, Copy)]
pub enum Live<'a> {
    Single(&'a NativeLiveHybridIndex),
    Sharded(&'a NativeLiveShardedHybridIndex),
}

impl<'a> From<&'a NativeLiveHybridIndex> for Live<'a> {
    fn from(live: &'a NativeLiveHybridIndex) -> Self {
        Self::Single(live)
    }
}

impl<'a> From<&'a NativeLiveShardedHybridIndex> for Live<'a> {
    fn from(live: &'a NativeLiveShardedHybridIndex) -> Self {
        Self::Sharded(live)
    }
}

impl Live<'_> {
    pub async fn snapshot(self, cx: &Cx) -> Result<Snapshot> {
        match self {
            Self::Single(live) => Ok(Snapshot::Single(live.snapshot(cx).await?)),
            Self::Sharded(live) => Ok(Snapshot::Sharded(live.snapshot(cx).await?)),
        }
    }

    pub async fn install(self, cx: &Cx, candidate: &Candidate) -> Result<Snapshot> {
        match (self, candidate) {
            (Self::Single(live), Candidate::Single(candidate)) => {
                Ok(Snapshot::Single(live.install(cx, candidate).await?))
            }
            (Self::Sharded(live), Candidate::Sharded(candidate)) => {
                Ok(Snapshot::Sharded(live.install(cx, candidate).await?))
            }
            _ => Err(bad(
                "activation cannot change the live handle's selected layout",
            )),
        }
    }
}

#[derive(Clone)]
pub enum Snapshot {
    Single(NativeHybridSnapshot),
    Sharded(NativeShardedSnapshot),
}

impl Snapshot {
    pub fn index(&self) -> cohort::Index<'_> {
        match self {
            Self::Single(snapshot) => snapshot.index().into(),
            Self::Sharded(snapshot) => snapshot.index().into(),
        }
    }

    pub fn generation(&self) -> ArtifactGenerationIdentityV1 {
        self.index().generation()
    }

    /// Reuse the native selected-reopen API's original retained models and
    /// exact-predecessor pin. No newly loaded model or invented expected digest.
    pub async fn prepare_selected(&self, cx: &Cx, selection: &Selection) -> Result<Candidate> {
        let expected = GenerationComponentReceiptV1 {
            byte_len: selection.snapshot.byte_len,
            sha256: selection.snapshot.sha256,
        };
        let candidate = match self {
            Self::Single(snapshot) if selection.schema == super::SELECTION_SCHEMA => {
                let mut limits = NativeHybridReopenLimits::default();
                limits.vectors.max_documents = MAX_DOCUMENTS;
                Candidate::Single(
                    snapshot
                        .prepare_selected(cx, &selection.directory, &expected, limits)
                        .await?,
                )
            }
            Self::Sharded(snapshot) if selection.schema == sharded::SELECTION_SCHEMA => {
                let mut limits = NativeShardedHybridReopenLimits::default();
                limits.vectors.max_documents = MAX_DOCUMENTS;
                Candidate::Sharded(
                    snapshot
                        .prepare_selected(cx, &selection.directory, &expected, limits)
                        .await?,
                )
            }
            _ => {
                return Err(bad(
                    "activation requires a receipt for the same selected layout",
                ));
            }
        };
        let admitted = candidate.index();
        if admitted.generation() != selection.generation
            || admitted.document_count() != selection.documents
        {
            return Err(bad(
                "admitted successor differs from the trusted receipt; serving selection unchanged",
            ));
        }
        Ok(candidate)
    }
}

pub enum Candidate {
    Single(NativeHybridCandidate),
    Sharded(NativeShardedCandidate),
}

impl Candidate {
    pub fn index(&self) -> cohort::Index<'_> {
        match self {
            Self::Single(candidate) => candidate.index().into(),
            Self::Sharded(candidate) => candidate.index().into(),
        }
    }
}
