//! Bounded, all-or-error JSONL admission for retained-generation append.
//!
//! Input is consumed once in fixed-size chunks. Only the current record and
//! the final body for each ID survive; there is no whole-input String or Vec
//! of lines. Every record is validated, duplicates remain last-write-wins,
//! and canonicalization runs once per final body before publication ownership
//! is acquired. These are payload/work limits, not a process-wide RSS cap.

use std::collections::BTreeMap;
use std::io::{self, Read};

use asupersync::Cx;
use asupersync::io::{AsyncRead, AsyncReadExt};
use frankensearch_core::{Canonicalizer, DefaultCanonicalizer, SearchError, SearchResult};
use serde::Deserialize;

use super::{FsfsRuntime, retained_search_checkpoint};

const READ_CHUNK_BYTES: usize = 64 * 1024;
const INPUT_LIMITS: InputLimits = InputLimits {
    wire_bytes: 256 * 1024 * 1024,
    record_bytes: 16 * 1024 * 1024,
    retained_bytes: 64 * 1024 * 1024,
    documents: 100_000,
};

#[derive(Clone, Copy)]
struct InputLimits {
    // Includes delimiters, ignored fields, blank lines and superseded records.
    wire_bytes: usize,
    // Encoded record bytes, excluding LF but including an optional CR.
    record_bytes: usize,
    // Sum of ID/body lengths, checked before and after canonicalization.
    retained_bytes: usize,
    // Distinct IDs, not the number of updates received for those IDs.
    documents: usize,
}

fn input_error(location: impl Into<String>, reason: impl Into<String>) -> SearchError {
    SearchError::InvalidConfig {
        field: "append_batch.input".to_owned(),
        value: location.into(),
        reason: reason.into(),
    }
}

fn limit_error(location: impl Into<String>, name: &str, limit: usize) -> SearchError {
    input_error(
        location,
        format!("append input exceeds the {name} limit ({limit}); the batch was not applied"),
    )
}

/// Read explicit bodies without interpreting IDs as source paths. Both file
/// and stdin inputs obey the same admission limits and atomicity contract.
/// File reads use Asupersync's async I/O. Stdin retains the owning-lane blocking
/// contract: cancellation is checked around each read, not promised to interrupt
/// an idle blocked OS read. No task is detached to bypass that limitation.
pub(in crate::runtime) async fn read_append_documents(
    cx: &Cx,
    runtime: &FsfsRuntime,
) -> SearchResult<BTreeMap<String, String>> {
    retained_search_checkpoint(cx)?;
    if let Some(path) = runtime.cli_input.input_file.as_ref() {
        let mut file = asupersync::fs::File::open(path).await?;
        read_async(cx, &mut file, INPUT_LIMITS).await
    } else {
        read_sync(cx, &mut io::stdin().lock(), INPUT_LIMITS)
    }
}

async fn read_async<R: AsyncRead + Unpin>(
    cx: &Cx,
    reader: &mut R,
    limits: InputLimits,
) -> SearchResult<BTreeMap<String, String>> {
    retained_search_checkpoint(cx)?;
    let mut input = AppendInput::new(limits);
    let mut buffer = vec![0_u8; READ_CHUNK_BYTES];
    loop {
        retained_search_checkpoint(cx)?;
        let read = AsyncReadExt::read(reader, &mut buffer).await;
        // A successful (including EOF) read must not publish cancelled input.
        retained_search_checkpoint(cx)?;
        let count = match read {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            other => other?,
        };
        if count == 0 {
            return input.finish(cx);
        }
        input.push(cx, &buffer[..count])?;
    }
}

fn read_sync<R: Read>(
    cx: &Cx,
    reader: &mut R,
    limits: InputLimits,
) -> SearchResult<BTreeMap<String, String>> {
    retained_search_checkpoint(cx)?;
    let mut input = AppendInput::new(limits);
    let mut buffer = vec![0_u8; READ_CHUNK_BYTES];
    loop {
        retained_search_checkpoint(cx)?;
        let read = reader.read(&mut buffer);
        retained_search_checkpoint(cx)?;
        let count = match read {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            other => other?,
        };
        if count == 0 {
            return input.finish(cx);
        }
        input.push(cx, &buffer[..count])?;
    }
}

struct AppendInput {
    limits: InputLimits,
    line: Vec<u8>,
    line_number: usize,
    wire_bytes: usize,
    retained_bytes: usize,
    documents: BTreeMap<String, String>,
}

impl AppendInput {
    const fn new(limits: InputLimits) -> Self {
        Self {
            limits,
            line: Vec::new(),
            line_number: 1,
            wire_bytes: 0,
            retained_bytes: 0,
            documents: BTreeMap::new(),
        }
    }

    fn push(&mut self, cx: &Cx, bytes: &[u8]) -> SearchResult<()> {
        retained_search_checkpoint(cx)?;
        self.wire_bytes = self
            .wire_bytes
            .checked_add(bytes.len())
            .filter(|&total| total <= self.limits.wire_bytes)
            .ok_or_else(|| limit_error("stream", "encoded-byte", self.limits.wire_bytes))?;
        for part in bytes.split_inclusive(|&byte| byte == b'\n') {
            retained_search_checkpoint(cx)?;
            let complete = part.last() == Some(&b'\n');
            let content = if complete {
                &part[..part.len() - 1]
            } else {
                part
            };
            let required = self
                .line
                .len()
                .checked_add(content.len())
                .filter(|&total| total <= self.limits.record_bytes)
                .ok_or_else(|| {
                    limit_error(
                        format!("line {}", self.line_number),
                        "record-byte",
                        self.limits.record_bytes,
                    )
                })?;
            if required > self.line.capacity() {
                // Geometric growth with a capped reservation request: an
                // unterminated record cannot request unbounded buffering.
                let capacity = required
                    .max(self.line.capacity().saturating_mul(2).max(READ_CHUNK_BYTES))
                    .min(self.limits.record_bytes);
                self.line
                    .try_reserve_exact(capacity - self.line.len())
                    .map_err(|_| input_error("record", "cannot reserve append record buffer"))?;
            }
            self.line.extend_from_slice(content);
            if complete {
                self.accept_line(cx)?;
                self.line.clear();
                self.line_number = self
                    .line_number
                    .checked_add(1)
                    .ok_or_else(|| input_error("stream", "append line number overflow"))?;
            }
        }
        retained_search_checkpoint(cx)
    }

    fn accept_line(&mut self, cx: &Cx) -> SearchResult<()> {
        retained_search_checkpoint(cx)?;
        let location = || format!("line {}", self.line_number);
        let line = std::str::from_utf8(&self.line)
            .map_err(|_| input_error(location(), "append input must be valid UTF-8"))?;
        if line.trim().is_empty() {
            return Ok(());
        }
        #[derive(Deserialize)]
        struct Document {
            id: String,
            text: String,
        }
        let document: Document = serde_json::from_str(line).map_err(|error| {
            // serde's Display can quote a private body or ID on a type error.
            // Position and category diagnose malformed input without echoing it.
            input_error(
                location(),
                format!(
                    "expected a JSON object with string id and text fields ({:?} at column {})",
                    error.classify(),
                    error.column(),
                ),
            )
        })?;
        retained_search_checkpoint(cx)?;
        if document.id.trim().is_empty() || document.id.len() > usize::from(u16::MAX) {
            return Err(input_error(
                location(),
                "document IDs must be nonblank and fit the vector record's 65535-byte limit",
            ));
        }
        let previous = self.documents.get(&document.id);
        if previous.is_none() && self.documents.len() >= self.limits.documents {
            return Err(limit_error(
                location(),
                "distinct-document",
                self.limits.documents,
            ));
        }
        // Replacement releases the superseded body, rather than charging its
        // size forever. Encoded-byte accounting still bounds repeated updates.
        let old_bytes = previous.map_or(0, String::len);
        let new_key_bytes = if previous.is_none() { document.id.len() } else { 0 };
        let retained_bytes = self
            .retained_bytes
            .checked_sub(old_bytes)
            .and_then(|total| total.checked_add(new_key_bytes))
            .and_then(|total| total.checked_add(document.text.len()))
            .filter(|&total| total <= self.limits.retained_bytes)
            .ok_or_else(|| {
                limit_error(
                    location(),
                    "retained-payload-byte",
                    self.limits.retained_bytes,
                )
            })?;
        self.documents.insert(document.id, document.text);
        self.retained_bytes = retained_bytes;
        Ok(())
    }

    fn finish(mut self, cx: &Cx) -> SearchResult<BTreeMap<String, String>> {
        retained_search_checkpoint(cx)?;
        if !self.line.is_empty() {
            self.accept_line(cx)?;
        }
        // Release the (possibly large) encoded record before canonicalizing.
        drop(self.line);
        let canonicalizer = DefaultCanonicalizer::default();
        for text in self.documents.values_mut() {
            retained_search_checkpoint(cx)?;
            let canonical = canonicalizer.canonicalize(text);
            retained_search_checkpoint(cx)?;
            if canonical.trim().is_empty() {
                return Err(input_error(
                    "empty_canonical_text",
                    "every final document body must contain canonical text; the batch was not applied",
                ));
            }
            if canonical.len() > self.limits.record_bytes {
                return Err(limit_error(
                    "canonical_document",
                    "record-byte",
                    self.limits.record_bytes,
                ));
            }
            self.retained_bytes = self
                .retained_bytes
                .checked_sub(text.len())
                .and_then(|total| total.checked_add(canonical.len()))
                .filter(|&total| total <= self.limits.retained_bytes)
                .ok_or_else(|| {
                    limit_error(
                        "canonical_batch",
                        "retained-payload-byte",
                        self.limits.retained_bytes,
                    )
                })?;
            *text = canonical;
        }
        retained_search_checkpoint(cx)?;
        Ok(self.documents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asupersync::io::ReadBuf;
    use asupersync::test_utils::run_test_with_cx;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    fn record(id: &str, text: &str) -> Vec<u8> {
        let mut bytes = serde_json::to_vec(&serde_json::json!({"id": id, "text": text})).unwrap();
        bytes.push(b'\n');
        bytes
    }

    fn parse(
        cx: &Cx,
        bytes: &[u8],
        chunk: usize,
        limits: InputLimits,
    ) -> SearchResult<BTreeMap<String, String>> {
        let mut parser = AppendInput::new(limits);
        for part in bytes.chunks(chunk) {
            parser.push(cx, part)?;
        }
        parser.finish(cx)
    }

    fn assert_limit(error: SearchError, name: &str) {
        assert!(
            matches!(error, SearchError::InvalidConfig { field, reason, .. }
                if field == "append_batch.input" && reason.contains(name))
        );
    }

    #[test]
    fn all_chunk_boundaries_preserve_unicode_crlf_and_last_write_wins() {
        run_test_with_cx(|cx| async move {
            let mut bytes = "\u{2003}\r\n".as_bytes().to_vec();
            bytes.extend(record("東京", "obsolete"));
            bytes.extend(record("alpha", "hello"));
            bytes.extend(record("東京", "café 🦀"));
            // Final record deliberately has no trailing newline.
            bytes.pop();
            let expected = BTreeMap::from([
                ("alpha".to_owned(), "hello".to_owned()),
                (
                    "東京".to_owned(),
                    DefaultCanonicalizer::default().canonicalize("café 🦀"),
                ),
            ]);
            for chunk in 1..=bytes.len() {
                assert_eq!(parse(&cx, &bytes, chunk, INPUT_LIMITS).unwrap(), expected);
            }
            let crlf = b"{\"id\":\"x\",\"text\":\"hello\"}\r\n";
            assert_eq!(parse(&cx, crlf, 1, INPUT_LIMITS).unwrap()["x"], "hello");
        });
    }

    #[test]
    fn record_and_wire_limits_include_exact_boundaries() {
        run_test_with_cx(|cx| async move {
            let bytes = record("x", "hello");
            let exact = InputLimits {
                wire_bytes: bytes.len(),
                record_bytes: bytes.len() - 1,
                ..INPUT_LIMITS
            };
            assert_eq!(parse(&cx, &bytes, 1, exact).unwrap()["x"], "hello");
            assert_limit(
                parse(
                    &cx,
                    &bytes,
                    1,
                    InputLimits { wire_bytes: bytes.len() - 1, ..exact },
                )
                .unwrap_err(),
                "encoded-byte",
            );
            assert_limit(
                parse(
                    &cx,
                    &bytes,
                    1,
                    InputLimits { record_bytes: bytes.len() - 2, ..exact },
                )
                .unwrap_err(),
                "record-byte",
            );
            let mut blanks = bytes;
            blanks.extend_from_slice(b"\n\n");
            assert_limit(parse(&cx, &blanks, 1, exact).unwrap_err(), "encoded-byte");
        });
    }

    #[test]
    fn unterminated_record_is_bounded_before_growth() {
        run_test_with_cx(|cx| async move {
            let limits = InputLimits {
                record_bytes: 128,
                ..INPUT_LIMITS
            };
            let mut input = AppendInput::new(limits);
            input.push(&cx, &[b' '; 128]).unwrap();
            let capacity = input.line.capacity();
            assert_limit(input.push(&cx, b"x").unwrap_err(), "record-byte");
            assert_eq!(input.line.len(), 128);
            assert_eq!(input.line.capacity(), capacity);
        });
    }

    #[test]
    fn duplicate_updates_release_old_payload_and_do_not_consume_new_id_slots() {
        run_test_with_cx(|cx| async move {
            let limits = InputLimits {
                retained_bytes: 10,
                documents: 1,
                ..INPUT_LIMITS
            };
            let mut parser = AppendInput::new(limits);
            for _ in 0..1_000 {
                parser.push(&cx, &record("x", "123456789")).unwrap();
                assert_eq!(parser.retained_bytes, 10);
                parser.push(&cx, &record("x", "a")).unwrap();
                assert_eq!(parser.retained_bytes, 2);
                assert_eq!(parser.documents.len(), 1);
            }
            assert_eq!(parser.finish(&cx).unwrap()["x"], "a");
            let mut parser = AppendInput::new(limits);
            parser.push(&cx, &record("x", "a")).unwrap();
            assert_limit(
                parser.push(&cx, &record("y", "b")).unwrap_err(),
                "distinct-document",
            );
        });
    }

    #[test]
    fn retained_payload_counts_id_and_decoded_body_bytes() {
        run_test_with_cx(|cx| async move {
            let bytes = record("ab", "é");
            let limits = InputLimits {
                retained_bytes: 4,
                ..INPUT_LIMITS
            };
            assert!(parse(&cx, &bytes, 2, limits).is_ok());
            assert_limit(
                parse(
                    &cx,
                    &bytes,
                    2,
                    InputLimits { retained_bytes: 3, ..limits },
                )
                .unwrap_err(),
                "retained-payload-byte",
            );
            let mut parser = AppendInput::new(limits);
            parser.push(&cx, &record("x", "a")).unwrap();
            assert_limit(
                parser.push(&cx, &record("y", "abc")).unwrap_err(),
                "retained-payload-byte",
            );
        });
    }

    #[test]
    fn canonicalization_only_applies_to_final_bodies() {
        run_test_with_cx(|cx| async move {
            let mut bytes = record("x", "");
            bytes.extend(record("x", "hello"));
            assert_eq!(parse(&cx, &bytes, 3, INPUT_LIMITS).unwrap()["x"], "hello");
            bytes.extend(record("x", "  "));
            assert!(
                matches!(parse(&cx, &bytes, 3, INPUT_LIMITS),
                    Err(SearchError::InvalidConfig { value, .. }) if value == "empty_canonical_text")
            );
        });
    }

    #[test]
    fn invalid_input_is_not_silently_skipped_or_echoed_in_diagnostics() {
        run_test_with_cx(|cx| async move {
            let mut bytes = record("good", "hello");
            bytes.extend_from_slice(b"{\"id\":\"private-id\",\"text\":[\"private-body\"]}\n");
            let error = parse(&cx, &bytes, 4, INPUT_LIMITS).unwrap_err();
            let display = error.to_string();
            assert!(!display.contains("private-id"));
            assert!(!display.contains("private-body"));
            assert!(
                matches!(error, SearchError::InvalidConfig { value, .. } if value == "line 2")
            );
            for invalid in [
                b"\xff\n".as_slice(),
                b"{\"id\":\"\",\"text\":\"hello\"}\n",
                b"[]\n",
            ] {
                assert!(parse(&cx, invalid, 1, INPUT_LIMITS).is_err());
            }
            let id = "x".repeat(usize::from(u16::MAX) + 1);
            assert!(parse(&cx, &record(&id, "hello"), 1024, INPUT_LIMITS).is_err());
        });
    }

    #[test]
    fn empty_input_and_only_blank_lines_are_empty_batches() {
        run_test_with_cx(|cx| async move {
            for bytes in [b"".as_slice(), b"\n\r\n\t  "] {
                assert!(parse(&cx, bytes, 1, INPUT_LIMITS).unwrap().is_empty());
            }
        });
    }

    #[test]
    fn wire_accounting_overflow_fails_closed() {
        run_test_with_cx(|cx| async move {
            let mut input = AppendInput::new(InputLimits {
                wire_bytes: usize::MAX,
                ..INPUT_LIMITS
            });
            input.wire_bytes = usize::MAX;
            assert_limit(input.push(&cx, b"x").unwrap_err(), "encoded-byte");
            assert!(input.line.is_empty());
            assert!(input.documents.is_empty());
        });
    }

    struct CancelReader<'a> {
        cx: &'a Cx,
        eof: bool,
        reads: usize,
    }

    impl Read for CancelReader<'_> {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            self.reads += 1;
            self.cx.set_cancel_requested(true);
            if self.eof {
                return Ok(0);
            }
            let bytes = record("x", "hello");
            buffer[..bytes.len()].copy_from_slice(&bytes);
            Ok(bytes.len())
        }
    }

    impl AsyncRead for CancelReader<'_> {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _: &mut Context<'_>,
            buffer: &mut ReadBuf<'_>,
        ) -> Poll<io::Result<()>> {
            self.reads += 1;
            self.cx.set_cancel_requested(true);
            if !self.eof {
                buffer.put_slice(&record("x", "hello"));
            }
            Poll::Ready(Ok(()))
        }
    }

    #[test]
    fn sync_cancellation_is_observed_before_reads_after_data_and_at_eof() {
        run_test_with_cx(|cx| async move {
            let mut reader = CancelReader {
                cx: &cx,
                eof: false,
                reads: 0,
            };
            cx.set_cancel_requested(true);
            let result = read_sync(&cx, &mut reader, INPUT_LIMITS);
            cx.set_cancel_requested(false);
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            assert_eq!(reader.reads, 0);
            for eof in [false, true] {
                reader.eof = eof;
                let result = read_sync(&cx, &mut reader, INPUT_LIMITS);
                cx.set_cancel_requested(false);
                assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            }
            assert_eq!(reader.reads, 2);
        });
    }

    #[test]
    fn async_cancellation_is_observed_before_reads_after_data_and_at_eof() {
        run_test_with_cx(|cx| async move {
            let mut reader = CancelReader {
                cx: &cx,
                eof: false,
                reads: 0,
            };
            cx.set_cancel_requested(true);
            let result = read_async(&cx, &mut reader, INPUT_LIMITS).await;
            cx.set_cancel_requested(false);
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            assert_eq!(reader.reads, 0);
            for eof in [false, true] {
                reader.eof = eof;
                let result = read_async(&cx, &mut reader, INPUT_LIMITS).await;
                cx.set_cancel_requested(false);
                assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            }
            assert_eq!(reader.reads, 2);
        });
    }

    struct FailingReader {
        reads: usize,
    }

    impl Read for FailingReader {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            self.reads += 1;
            match self.reads {
                1 => Err(io::ErrorKind::Interrupted.into()),
                2 => {
                    let bytes = record("x", "hello");
                    buffer[..bytes.len()].copy_from_slice(&bytes);
                    Ok(bytes.len())
                }
                _ => Err(io::ErrorKind::PermissionDenied.into()),
            }
        }
    }

    #[test]
    fn io_failure_after_valid_records_returns_no_partial_batch() {
        run_test_with_cx(|cx| async move {
            let mut reader = FailingReader { reads: 0 };
            assert!(
                matches!(read_sync(&cx, &mut reader, INPUT_LIMITS),
                    Err(SearchError::Io(error)) if error.kind() == io::ErrorKind::PermissionDenied)
            );
            assert_eq!(reader.reads, 3);
        });
    }

    #[test]
    fn async_and_sync_transports_match_the_streaming_state_machine() {
        run_test_with_cx(|cx| async move {
            let mut bytes = record("x", &"hello ".repeat(20_000));
            bytes.extend(record("x", "replacement"));
            bytes.extend(record("y", "other"));
            let expected = parse(&cx, &bytes, 7, INPUT_LIMITS).unwrap();
            assert_eq!(
                read_sync(&cx, &mut bytes.as_slice(), INPUT_LIMITS).unwrap(),
                expected,
            );
            assert_eq!(
                read_async(&cx, &mut bytes.as_slice(), INPUT_LIMITS).await.unwrap(),
                expected,
            );
        });
    }

    #[test]
    fn production_file_path_admits_before_creating_any_store() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let path = parent.path().join("input.jsonl");
            let root = parent.path().join("store");
            let bytes = record("not-a-source-path", "hello");
            std::fs::write(&path, &bytes).unwrap();
            let runtime = FsfsRuntime::new(crate::FsfsConfig::default())
                .with_cli_input(crate::CliInput {
                    command: crate::CliCommand::AppendBatch,
                    input_file: Some(path.clone()),
                    index_dir: Some(root.clone()),
                    ..crate::CliInput::default()
                });
            assert_eq!(
                read_append_documents(&cx, &runtime).await.unwrap()["not-a-source-path"],
                "hello",
            );
            assert_eq!(std::fs::read(&path).unwrap(), bytes);
            assert!(!root.exists());
            std::fs::write(&path, b"{broken\n").unwrap();
            assert!(read_append_documents(&cx, &runtime).await.is_err());
            assert!(!root.exists());
        });
    }
}
