//! Bounded inference groups for mixed retained updates. Inputs borrow the owned
//! prepared bodies; only a completed, fully validated group enters staged output.

use frankensearch_core::IdentityBoundEmbedding;

use super::{
    Cx, MAX_VECTOR_BYTES, Producer, SearchError, SearchResult, charge_vector, invalid,
    retained_search_checkpoint,
};

const MAX_ROWS: usize = 64;
const MAX_TEXT_BYTES: usize = 512 * 1024;

type VectorRows = Vec<(String, Vec<f32>)>;

pub(super) struct InferenceBatch<'a> {
    producer: &'a Producer,
    native: bool,
    ids: Vec<String>,
    texts: Vec<&'a str>,
    text_bytes: usize,
    output: VectorRows,
}

impl<'a> InferenceBatch<'a> {
    pub(super) fn new(producer: &'a Producer) -> Self {
        Self {
            producer,
            native: producer.embedder.bound_batch_is_native(),
            ids: Vec::with_capacity(MAX_ROWS),
            texts: Vec::with_capacity(MAX_ROWS),
            text_bytes: 0,
            output: Vec::new(),
        }
    }

    fn check(&self, cx: &Cx) -> SearchResult<()> {
        retained_search_checkpoint(cx)?;
        let admission = self.producer.recheck();
        retained_search_checkpoint(cx)?;
        admission?;
        let native = self.producer.embedder.bound_batch_is_native();
        retained_search_checkpoint(cx)?;
        if native != self.native {
            return Err(SearchError::UnverifiableRemoteSpace {
                producer: self.producer.role.to_owned(),
                reason: "bound batching contract changed during a retained update".to_owned(),
            });
        }
        Ok(())
    }

    pub(super) async fn push(
        &mut self,
        cx: &Cx,
        id: String,
        text: &'a str,
        vector_bytes: &mut usize,
    ) -> SearchResult<()> {
        self.check(cx)?;
        if self.ids.len() == MAX_ROWS
            || (!self.ids.is_empty() && self.text_bytes.saturating_add(text.len()) > MAX_TEXT_BYTES)
        {
            self.flush(cx, vector_bytes).await?;
        }
        // A single larger input is allowed under the parent's existing 8 MiB
        // document cap, but is never combined with another input in one group.
        self.text_bytes = self
            .text_bytes
            .checked_add(text.len())
            .ok_or_else(|| invalid("inference input length overflow"))?;
        self.ids.push(id);
        self.texts.push(text);
        Ok(())
    }

    pub(super) async fn finish(
        mut self,
        cx: &Cx,
        vector_bytes: &mut usize,
    ) -> SearchResult<VectorRows> {
        self.flush(cx, vector_bytes).await?;
        self.check(cx)?;
        Ok(self.output)
    }

    fn admit_response(&self, response: &IdentityBoundEmbedding) -> SearchResult<()> {
        if response.identity != self.producer.identity {
            return Err(SearchError::UnverifiableRemoteSpace {
                producer: self.producer.role.to_owned(),
                reason: "batch response does not carry the admitted producer identity".to_owned(),
            });
        }
        response.validate()
    }

    async fn flush(&mut self, cx: &Cx, vector_bytes: &mut usize) -> SearchResult<()> {
        self.check(cx)?;
        if self.ids.is_empty() {
            return Ok(());
        }
        // Refuse known output cost BEFORE inference. The other tier shares this
        // counter, including completed groups; do not invoke a model merely to
        // discover that an exactly dimensioned response exceeds the budget.
        let expected_bytes = usize::try_from(self.producer.identity.space.dimension)
            .ok()
            .and_then(|width| width.checked_mul(std::mem::size_of::<f32>()))
            .and_then(|bytes| bytes.checked_mul(self.ids.len()))
            .and_then(|bytes| vector_bytes.checked_add(bytes))
            .filter(|bytes| *bytes <= MAX_VECTOR_BYTES)
            .ok_or_else(|| invalid("retained batch vector payload exceeds 128 MiB"))?;
        let values = if self.native {
            let outcome = self
                .producer
                .embedder
                .embed_batch_bound(cx, &self.texts)
                .await;
            retained_search_checkpoint(cx)?;
            let outcome = match outcome {
                Err(error @ SearchError::Cancelled { .. }) => return Err(error),
                outcome => outcome,
            };
            // Drift during ordinary failure must not become a retryable error.
            self.check(cx)?;
            let response = outcome?;
            if response.len() != self.ids.len() {
                return Err(invalid("one bound batch output is required per input row"));
            }
            for row in &response {
                retained_search_checkpoint(cx)?;
                self.admit_response(row)?;
            }
            response
                .into_iter()
                .map(|row| row.values)
                .collect::<Vec<_>>()
        } else {
            let mut response = Vec::with_capacity(self.ids.len());
            for text in &self.texts {
                self.check(cx)?;
                response.push(self.producer.infer(cx, text).await?);
                self.check(cx)?;
            }
            response
        };
        self.check(cx)?;
        let mut charged = *vector_bytes;
        for vector in &values {
            charge_vector(&mut charged, vector)?;
        }
        if charged != expected_bytes {
            return Err(invalid(
                "inference group output cost disagrees with its admitted shape",
            ));
        }
        // Commit accounting and all rows together. On any earlier error neither
        // the group's IDs nor any prefix of its output has been staged.
        *vector_bytes = charged;
        self.output.extend(self.ids.drain(..).zip(values));
        self.texts.clear();
        self.text_bytes = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
