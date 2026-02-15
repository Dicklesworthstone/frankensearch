//! Model download system with progress reporting and atomic installation.
//!
//! Downloads model files from `HuggingFace` with SHA-256 verification,
//! atomic installation (rename-over), and progress callbacks.
//!
//! Gated behind the `download` feature flag to keep the core crate network-free.

use std::fmt;
use std::future::poll_fn;
use std::io::ErrorKind;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use asupersync::bytes::Buf;
use asupersync::http::body::{Body, Frame};
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use asupersync::http::h1::{ClientError, HttpClient, HttpClientConfig, Method, RedirectPolicy};
use frankensearch_core::error::{SearchError, SearchResult};

use crate::model_manifest::{ModelFile, ModelLifecycle, ModelManifest};

static STAGING_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for model downloads.
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    /// Maximum retries per file on transient failure.
    pub max_retries: u32,
    /// Base delay for exponential backoff between retries.
    pub retry_base_delay: Duration,
    /// User-Agent header value.
    pub user_agent: String,
    /// Maximum redirects to follow.
    pub max_redirects: u32,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_base_delay: Duration::from_secs(1),
            user_agent: format!("frankensearch/{}", env!("CARGO_PKG_VERSION")),
            max_redirects: 5,
        }
    }
}

// ─── Progress ───────────────────────────────────────────────────────────────

/// Progress information for an in-flight model download.
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Name of the file currently being downloaded.
    pub file_name: String,
    /// Bytes downloaded so far (current file).
    pub bytes_downloaded: u64,
    /// Total bytes expected (current file), if known.
    pub total_bytes: Option<u64>,
    /// Number of files completed so far.
    pub files_completed: usize,
    /// Total number of files to download.
    pub files_total: usize,
    /// Estimated download speed in bytes per second.
    pub speed_bytes_per_sec: f64,
    /// Estimated time remaining in seconds, if calculable.
    pub eta_seconds: Option<f64>,
}

impl fmt::Display for DownloadProgress {
    #[allow(clippy::cast_precision_loss)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pct = self
            .total_bytes
            .filter(|&t| t > 0)
            .map(|t| self.bytes_downloaded as f64 / t as f64 * 100.0);

        if let Some(pct) = pct {
            write!(
                f,
                "[{}/{}] {} {:.0}% ({}/{})",
                self.files_completed + 1,
                self.files_total,
                self.file_name,
                pct,
                format_bytes(self.bytes_downloaded),
                format_bytes(self.total_bytes.unwrap_or(0)),
            )
        } else {
            write!(
                f,
                "[{}/{}] {} {}",
                self.files_completed + 1,
                self.files_total,
                self.file_name,
                format_bytes(self.bytes_downloaded),
            )
        }
    }
}

// ─── Downloader ─────────────────────────────────────────────────────────────

/// Downloads model files from `HuggingFace` with verification and progress reporting.
pub struct ModelDownloader {
    config: DownloadConfig,
    client: HttpClient,
}

impl ModelDownloader {
    /// Create a new downloader with the given configuration.
    #[must_use]
    pub fn new(config: DownloadConfig) -> Self {
        let client_config = HttpClientConfig {
            redirect_policy: RedirectPolicy::Limited(config.max_redirects),
            user_agent: Some(config.user_agent.clone()),
            ..HttpClientConfig::default()
        };
        Self {
            config,
            client: HttpClient::with_config(client_config),
        }
    }

    /// Create a downloader with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(DownloadConfig::default())
    }

    /// Download all files for a model manifest into a staging directory.
    ///
    /// A unique staging directory is created under `{dest_dir}` for each call
    /// (for example, `.download-<pid>-<counter>`), and files are placed there
    /// during download. After all files are verified, the
    /// caller should use [`ModelManifest::promote_verified_installation`] to
    /// atomically install the model.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` on network failure, hash mismatch, or I/O error.
    pub async fn download_model(
        &self,
        manifest: &ModelManifest,
        dest_dir: &Path,
        lifecycle: &mut ModelLifecycle,
        on_progress: impl Fn(&DownloadProgress) + Send + Sync,
    ) -> SearchResult<PathBuf> {
        let total_bytes = manifest.total_size_bytes();
        lifecycle.begin_download(total_bytes.max(1))?;

        if !manifest.is_production_ready() {
            let reason = format!(
                "manifest for '{}' must be production-ready (pinned revision + verified checksums) before download",
                manifest.id
            );
            lifecycle.fail_verification(reason.clone());
            return Err(SearchError::InvalidConfig {
                field: "manifest".to_owned(),
                value: manifest.id.clone(),
                reason,
            });
        }

        let staging_dir = match create_unique_staging_dir(dest_dir) {
            Ok(path) => path,
            Err(err) => {
                lifecycle.fail_verification(format!(
                    "failed to create staging directory for '{}': {err}",
                    manifest.id
                ));
                return Err(err);
            }
        };

        let files_total = manifest.files.len();
        let mut cumulative_bytes: u64 = 0;

        for (idx, file) in manifest.files.iter().enumerate() {
            let url = file.download_url(&manifest.repo, &manifest.revision);
            let file_dest = staging_dir.join(&file.name);

            // Create parent directories for nested paths (e.g., "onnx/model.onnx").
            if let Some(parent) = file_dest.parent()
                && let Err(err) = std::fs::create_dir_all(parent).map_err(SearchError::from)
            {
                lifecycle.fail_verification(format!(
                    "failed to create parent directory for '{}': {err}",
                    file.name
                ));
                return Err(err);
            }

            info!(
                file = %file.name,
                size = file.size,
                url = %url,
                "downloading model file"
            );

            if let Err(err) = self
                .download_file_with_retry(&url, &file_dest, file, idx, files_total, &on_progress)
                .await
            {
                lifecycle.fail_verification(format!("download failed for '{}': {err}", file.name));
                return Err(err);
            }

            cumulative_bytes = cumulative_bytes.saturating_add(file.size);
            if let Err(err) = lifecycle.update_download_progress(cumulative_bytes) {
                lifecycle.fail_verification(format!("failed to update download progress: {err}"));
                return Err(err);
            }
        }

        // Verify all files.
        if let Err(err) = lifecycle.begin_verification() {
            lifecycle.fail_verification(format!(
                "failed to transition model lifecycle into verification: {err}"
            ));
            return Err(err);
        }
        info!(model = %manifest.id, "verifying downloaded files");
        match manifest.verify_dir(&staging_dir) {
            Ok(()) => {
                lifecycle.mark_ready();
                info!(model = %manifest.id, "model download complete and verified");
                Ok(staging_dir)
            }
            Err(e) => {
                lifecycle.fail_verification(e.to_string());
                Err(e)
            }
        }
    }

    /// Download a single file with retry logic.
    async fn download_file_with_retry(
        &self,
        url: &str,
        dest: &Path,
        file: &ModelFile,
        file_idx: usize,
        files_total: usize,
        on_progress: &(impl Fn(&DownloadProgress) + Send + Sync),
    ) -> SearchResult<()> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay = self.config.retry_base_delay * 2_u32.saturating_pow(attempt - 1);
                warn!(
                    file = %file.name,
                    attempt,
                    delay_ms = delay.as_millis(),
                    "retrying download after failure"
                );
                asupersync::time::sleep(asupersync::time::wall_now(), delay).await;
            }

            match self
                .download_single_file(url, dest, file, file_idx, files_total, on_progress)
                .await
            {
                Ok(()) => return Ok(()),
                Err(e) => {
                    warn!(
                        file = %file.name,
                        attempt,
                        error = %e,
                        "download attempt failed"
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| SearchError::ModelLoadFailed {
            path: dest.to_path_buf(),
            source: "download failed after all retries".into(),
        }))
    }

    /// Download a single file (one attempt).
    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    async fn download_single_file(
        &self,
        url: &str,
        dest: &Path,
        file: &ModelFile,
        file_idx: usize,
        files_total: usize,
        on_progress: &(impl Fn(&DownloadProgress) + Send + Sync),
    ) -> SearchResult<()> {
        let start = Instant::now();

        // Report start.
        on_progress(&DownloadProgress {
            file_name: file.name.clone(),
            bytes_downloaded: 0,
            total_bytes: if file.size > 0 { Some(file.size) } else { None },
            files_completed: file_idx,
            files_total,
            speed_bytes_per_sec: 0.0,
            eta_seconds: None,
        });

        // Stream directly into a temp file to keep memory bounded.
        let mut response = self
            .client
            .request_streaming(Method::Get, url, Vec::new(), Vec::new())
            .await
            .map_err(|e| client_error_to_search(e, url))?;

        // Check HTTP status.
        if response.head.status < 200 || response.head.status >= 300 {
            return Err(SearchError::ModelLoadFailed {
                path: dest.to_path_buf(),
                source: format!(
                    "HTTP {} {} for {url}",
                    response.head.status, response.head.reason
                )
                .into(),
            });
        }

        let tmp_path = dest.with_extension("tmp");
        let mut tmp_guard = TempFileGuard::new(tmp_path.clone());
        let mut tmp_file = std::fs::File::create(&tmp_path).map_err(SearchError::from)?;

        let total_bytes = if file.size > 0 {
            Some(file.size)
        } else {
            response_content_length(&response.head.headers)
        };
        let mut hasher = Sha256::new();
        let mut bytes_downloaded: u64 = 0;

        while let Some(frame) = poll_fn(|cx| Pin::new(&mut response.body).poll_frame(cx)).await {
            let frame = frame.map_err(|e| SearchError::ModelLoadFailed {
                path: dest.to_path_buf(),
                source: format!("stream read failed for {url}: {e}").into(),
            })?;
            if let Frame::Data(mut chunk) = frame {
                while chunk.has_remaining() {
                    let bytes = chunk.chunk();
                    if bytes.is_empty() {
                        break;
                    }
                    tmp_file.write_all(bytes).map_err(SearchError::from)?;
                    hasher.update(bytes);
                    bytes_downloaded = bytes_downloaded
                        .saturating_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
                    chunk.advance(bytes.len());
                }

                let elapsed = start.elapsed();
                let speed = if elapsed.as_secs_f64() > 0.0 {
                    bytes_downloaded as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };
                let eta_seconds = total_bytes.and_then(|total| {
                    if speed <= f64::EPSILON || bytes_downloaded >= total {
                        None
                    } else {
                        Some((total.saturating_sub(bytes_downloaded)) as f64 / speed)
                    }
                });
                on_progress(&DownloadProgress {
                    file_name: file.name.clone(),
                    bytes_downloaded,
                    total_bytes,
                    files_completed: file_idx,
                    files_total,
                    speed_bytes_per_sec: speed,
                    eta_seconds,
                });
            }
        }

        if file.size > 0 && bytes_downloaded != file.size {
            return Err(SearchError::HashMismatch {
                path: dest.to_path_buf(),
                expected: format!("size={}", file.size),
                actual: format!("size={bytes_downloaded}"),
            });
        }

        // Verify SHA-256 only when the manifest provides a concrete checksum.
        if file.has_verified_checksum() {
            let actual_hash = sha256_digest_hex(hasher.finalize().as_slice());
            if actual_hash != file.sha256 {
                return Err(SearchError::HashMismatch {
                    path: dest.to_path_buf(),
                    expected: format!("sha256={},size={}", file.sha256, file.size),
                    actual: format!("sha256={actual_hash},size={bytes_downloaded}"),
                });
            }
        }

        tmp_file.flush().map_err(SearchError::from)?;
        tmp_file.sync_all().map_err(SearchError::from)?;
        drop(tmp_file);
        std::fs::rename(&tmp_path, dest).map_err(SearchError::from)?;
        tmp_guard.disarm();

        let elapsed = start.elapsed();
        let speed = if elapsed.as_secs_f64() > 0.0 {
            bytes_downloaded as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        on_progress(&DownloadProgress {
            file_name: file.name.clone(),
            bytes_downloaded,
            total_bytes: if file.size > 0 {
                Some(file.size)
            } else {
                total_bytes
            },
            files_completed: file_idx,
            files_total,
            speed_bytes_per_sec: speed,
            eta_seconds: Some(0.0),
        });

        info!(
            file = %file.name,
            bytes = bytes_downloaded,
            elapsed_ms = elapsed.as_millis(),
            "file saved"
        );

        Ok(())
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Build a `HuggingFace` CDN URL for a model file.
#[cfg(test)]
fn huggingface_url(repo: &str, revision: &str, file_name: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/{revision}/{file_name}")
}

fn create_unique_staging_dir(dest_dir: &Path) -> SearchResult<PathBuf> {
    std::fs::create_dir_all(dest_dir).map_err(SearchError::from)?;

    let pid = std::process::id();
    for _ in 0..64 {
        let counter = STAGING_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let candidate = dest_dir.join(format!(".download-{pid}-{counter:016x}"));
        match std::fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(err) if err.kind() == ErrorKind::AlreadyExists => {}
            Err(err) => return Err(SearchError::from(err)),
        }
    }

    Err(SearchError::ModelLoadFailed {
        path: dest_dir.to_path_buf(),
        source: "failed to allocate unique staging directory".into(),
    })
}

fn response_content_length(headers: &[(String, String)]) -> Option<u64> {
    headers.iter().find_map(|(name, value)| {
        name.eq_ignore_ascii_case("content-length")
            .then(|| value.trim().parse::<u64>().ok())
            .flatten()
    })
}

fn sha256_digest_hex(data: &[u8]) -> String {
    let mut out = String::with_capacity(64);
    for byte in data {
        use std::fmt::Write;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

/// Compute lowercase hex SHA-256 of a byte slice.
#[cfg(test)]
fn sha256_hex(data: &[u8]) -> String {
    sha256_digest_hex(Sha256::digest(data).as_slice())
}

struct TempFileGuard {
    path: PathBuf,
    armed: bool,
}

impl TempFileGuard {
    const fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    const fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if self.armed
            && let Err(e) = std::fs::remove_file(&self.path)
        {
            tracing::warn!(path = %self.path.display(), error = %e, "failed to clean up temp file");
        }
    }
}

/// Format bytes as a human-readable string.
#[allow(clippy::cast_precision_loss)]
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Convert asupersync `ClientError` to `SearchError`.
fn client_error_to_search(error: ClientError, url: &str) -> SearchError {
    SearchError::ModelLoadFailed {
        path: PathBuf::from(url),
        source: Box::new(error),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_manifest::{PLACEHOLDER_PINNED_REVISION, PLACEHOLDER_VERIFY_AFTER_DOWNLOAD};
    use std::collections::VecDeque;
    use std::io::{Read, Write};
    use std::net::{Shutdown, TcpListener, TcpStream};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;

    use asupersync::test_utils::run_test_with_cx;

    #[derive(Debug, Clone)]
    struct TestHttpResponse {
        status: u16,
        reason: &'static str,
        body: Vec<u8>,
    }

    fn spawn_test_http_server(
        responses: Vec<TestHttpResponse>,
    ) -> (String, Arc<AtomicUsize>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let queue = Arc::new(Mutex::new(VecDeque::from(responses)));
        let served = Arc::new(AtomicUsize::new(0));
        let served_for_thread = Arc::clone(&served);
        let queue_for_thread = Arc::clone(&queue);

        let handle = thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                if read_http_headers(&mut stream).is_err() {
                    break;
                }

                let response = {
                    let mut guard = queue_for_thread.lock().unwrap();
                    guard.pop_front()
                };
                let Some(response) = response else {
                    break;
                };

                served_for_thread.fetch_add(1, Ordering::SeqCst);
                if write_http_response(&mut stream, &response).is_err() {
                    break;
                }
                let _ = stream.shutdown(Shutdown::Both);
                if queue_for_thread.lock().unwrap().is_empty() {
                    break;
                }
            }
        });

        (format!("http://{addr}"), served, handle)
    }

    fn read_http_headers(stream: &mut TcpStream) -> std::io::Result<()> {
        let mut buf = [0_u8; 1024];
        let mut request = Vec::new();
        loop {
            let read = stream.read(&mut buf)?;
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buf[..read]);
            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
            if request.len() > 64 * 1024 {
                break;
            }
        }
        Ok(())
    }

    fn write_http_response(
        stream: &mut TcpStream,
        response: &TestHttpResponse,
    ) -> std::io::Result<()> {
        write!(
            stream,
            "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            response.status,
            response.reason,
            response.body.len()
        )?;
        stream.write_all(&response.body)?;
        stream.flush()?;
        Ok(())
    }

    #[test]
    fn huggingface_url_format() {
        let url = huggingface_url(
            "sentence-transformers/all-MiniLM-L6-v2",
            "abc123",
            "onnx/model.onnx",
        );
        assert_eq!(
            url,
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/abc123/onnx/model.onnx"
        );
    }

    #[test]
    fn sha256_hex_known_value() {
        let hash = sha256_hex(b"hello world");
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn format_bytes_units() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1_048_576), "1.0 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.0 GB");
    }

    #[test]
    fn download_config_defaults() {
        let config = DownloadConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_base_delay, Duration::from_secs(1));
        assert_eq!(config.max_redirects, 5);
        assert!(config.user_agent.starts_with("frankensearch/"));
    }

    #[test]
    fn download_progress_display_with_total() {
        let progress = DownloadProgress {
            file_name: "model.onnx".to_owned(),
            bytes_downloaded: 524_288,
            total_bytes: Some(1_048_576),
            files_completed: 0,
            files_total: 3,
            speed_bytes_per_sec: 1_048_576.0,
            eta_seconds: Some(0.5),
        };
        let display = progress.to_string();
        assert!(display.contains("[1/3]"));
        assert!(display.contains("model.onnx"));
        assert!(display.contains("50%"));
    }

    #[test]
    fn download_progress_display_without_total() {
        let progress = DownloadProgress {
            file_name: "config.json".to_owned(),
            bytes_downloaded: 1024,
            total_bytes: None,
            files_completed: 2,
            files_total: 3,
            speed_bytes_per_sec: 0.0,
            eta_seconds: None,
        };
        let display = progress.to_string();
        assert!(display.contains("[3/3]"));
        assert!(display.contains("config.json"));
        assert!(display.contains("1.0 KB"));
    }

    #[test]
    fn client_error_converts_to_search_error() {
        let err = client_error_to_search(
            ClientError::InvalidUrl("bad".to_owned()),
            "https://example.com",
        );
        assert!(matches!(err, SearchError::ModelLoadFailed { .. }));
    }

    #[test]
    fn download_single_file_success_writes_file_and_reports_progress() {
        let body = b"hello-model".to_vec();
        let file = ModelFile {
            name: "model.onnx".to_owned(),
            sha256: sha256_hex(&body),
            size: u64::try_from(body.len()).unwrap(),
            url: None,
        };

        let (base_url, served, handle) = spawn_test_http_server(vec![TestHttpResponse {
            status: 200,
            reason: "OK",
            body: body.clone(),
        }]);
        let url = format!("{base_url}/model.onnx");
        let dest_dir = tempfile::tempdir().unwrap();
        let dest = dest_dir.path().join("model.onnx");
        let dest_for_task = dest.clone();
        let progress = Arc::new(Mutex::new(Vec::<DownloadProgress>::new()));
        let progress_for_cb = Arc::clone(&progress);
        let downloader = ModelDownloader::new(DownloadConfig {
            max_retries: 0,
            retry_base_delay: Duration::from_millis(1),
            user_agent: "frankensearch-test".to_owned(),
            max_redirects: 0,
        });

        run_test_with_cx(|_cx| async move {
            downloader
                .download_single_file(&url, &dest_for_task, &file, 0, 1, &|p| {
                    progress_for_cb.lock().unwrap().push(p.clone());
                })
                .await
                .unwrap();
        });

        handle.join().unwrap();
        assert_eq!(served.load(Ordering::SeqCst), 1);
        assert_eq!(std::fs::read(dest).unwrap(), body);

        let events = progress.lock().unwrap();
        assert!(events.len() >= 2);
        assert_eq!(events[0].bytes_downloaded, 0);
        assert_eq!(events[0].file_name, "model.onnx");
        let last = events.last().unwrap();
        let expected_size = u64::try_from(body.len()).unwrap();
        assert_eq!(last.bytes_downloaded, expected_size);
        assert_eq!(last.total_bytes, Some(expected_size));
        drop(events);
    }

    #[test]
    fn download_file_with_retry_succeeds_after_transient_http_error() {
        let body = b"retry-success".to_vec();
        let file = ModelFile {
            name: "model.onnx".to_owned(),
            sha256: sha256_hex(&body),
            size: u64::try_from(body.len()).unwrap(),
            url: None,
        };

        let (base_url, served, handle) = spawn_test_http_server(vec![
            TestHttpResponse {
                status: 500,
                reason: "Internal Server Error",
                body: b"server error".to_vec(),
            },
            TestHttpResponse {
                status: 200,
                reason: "OK",
                body: body.clone(),
            },
        ]);
        let url = format!("{base_url}/model.onnx");
        let dest_dir = tempfile::tempdir().unwrap();
        let dest = dest_dir.path().join("model.onnx");
        let dest_for_task = dest.clone();
        let downloader = ModelDownloader::new(DownloadConfig {
            max_retries: 1,
            retry_base_delay: Duration::from_millis(1),
            user_agent: "frankensearch-test".to_owned(),
            max_redirects: 0,
        });

        run_test_with_cx(|_cx| async move {
            downloader
                .download_file_with_retry(&url, &dest_for_task, &file, 0, 1, &|_| {})
                .await
                .unwrap();
        });

        handle.join().unwrap();
        assert_eq!(served.load(Ordering::SeqCst), 2);
        assert_eq!(std::fs::read(dest).unwrap(), body);
    }

    #[test]
    fn download_file_with_retry_returns_error_after_max_attempts() {
        let file = ModelFile {
            name: "model.onnx".to_owned(),
            sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            size: 4,
            url: None,
        };

        let (base_url, served, handle) = spawn_test_http_server(vec![
            TestHttpResponse {
                status: 500,
                reason: "Internal Server Error",
                body: b"error".to_vec(),
            },
            TestHttpResponse {
                status: 500,
                reason: "Internal Server Error",
                body: b"error".to_vec(),
            },
        ]);
        let url = format!("{base_url}/model.onnx");
        let dest_dir = tempfile::tempdir().unwrap();
        let dest = dest_dir.path().join("model.onnx");
        let dest_for_task = dest.clone();
        let downloader = ModelDownloader::new(DownloadConfig {
            max_retries: 1,
            retry_base_delay: Duration::from_millis(1),
            user_agent: "frankensearch-test".to_owned(),
            max_redirects: 0,
        });

        run_test_with_cx(|_cx| async move {
            let err = downloader
                .download_file_with_retry(&url, &dest_for_task, &file, 0, 1, &|_| {})
                .await
                .unwrap_err();
            assert!(matches!(err, SearchError::ModelLoadFailed { .. }));
            assert!(err.to_string().contains("HTTP 500"));
        });

        handle.join().unwrap();
        assert_eq!(served.load(Ordering::SeqCst), 2);
        assert!(!dest.exists());
    }

    #[test]
    fn download_single_file_hash_mismatch_does_not_write_destination() {
        let expected = b"expected-content".to_vec();
        let file = ModelFile {
            name: "model.onnx".to_owned(),
            sha256: sha256_hex(&expected),
            size: u64::try_from(expected.len()).unwrap(),
            url: None,
        };

        let (base_url, served, handle) = spawn_test_http_server(vec![TestHttpResponse {
            status: 200,
            reason: "OK",
            body: b"different-content".to_vec(),
        }]);
        let url = format!("{base_url}/model.onnx");
        let dest_dir = tempfile::tempdir().unwrap();
        let dest = dest_dir.path().join("model.onnx");
        let dest_for_task = dest.clone();
        let downloader = ModelDownloader::new(DownloadConfig {
            max_retries: 0,
            retry_base_delay: Duration::from_millis(1),
            user_agent: "frankensearch-test".to_owned(),
            max_redirects: 0,
        });

        run_test_with_cx(|_cx| async move {
            let err = downloader
                .download_single_file(&url, &dest_for_task, &file, 0, 1, &|_| {})
                .await
                .unwrap_err();
            assert!(matches!(err, SearchError::HashMismatch { .. }));
        });

        handle.join().unwrap();
        assert_eq!(served.load(Ordering::SeqCst), 1);
        assert!(!dest.exists());
        assert!(!dest.with_extension("tmp").exists());
    }

    #[test]
    fn download_model_failure_transitions_lifecycle_to_verification_failed() {
        let manifest = ModelManifest {
            id: "test-model".to_owned(),
            version: "test-v1".to_owned(),
            display_name: None,
            description: None,
            repo: "owner/repo".to_owned(),
            revision: "deadbeef".to_owned(),
            files: vec![ModelFile {
                // Invalid URL path segment (space) forces an immediate client error.
                name: "bad file.bin".to_owned(),
                sha256: "0".repeat(64),
                size: 1,
                url: None,
            }],
            license: "Apache-2.0".to_owned(),
            dimension: None,
            tier: None,
            download_size_bytes: 0,
        };
        let consent = crate::model_manifest::DownloadConsent::granted(
            crate::model_manifest::ConsentSource::Environment,
        );
        let mut lifecycle = ModelLifecycle::new(manifest.clone(), consent);
        let dest = tempfile::tempdir().unwrap();
        let downloader = ModelDownloader::new(DownloadConfig {
            max_retries: 0,
            retry_base_delay: Duration::from_millis(1),
            user_agent: "frankensearch-test".to_owned(),
            max_redirects: 0,
        });

        run_test_with_cx(|_cx| async move {
            let err = downloader
                .download_model(&manifest, dest.path(), &mut lifecycle, |_| {})
                .await
                .unwrap_err();
            assert!(matches!(err, SearchError::ModelLoadFailed { .. }));
            assert!(matches!(
                lifecycle.state(),
                crate::model_manifest::ModelState::VerificationFailed { .. }
            ));
            assert!(lifecycle.begin_download(1).is_ok());
        });
    }

    #[test]
    fn create_unique_staging_dir_returns_distinct_paths() {
        let temp = tempfile::tempdir().unwrap();
        let first = create_unique_staging_dir(temp.path()).expect("first staging dir");
        let second = create_unique_staging_dir(temp.path()).expect("second staging dir");

        assert_ne!(first, second);
        assert!(first.is_dir());
        assert!(second.is_dir());
    }

    // ─── bd-r476 tests begin ───

    #[test]
    fn response_content_length_found() {
        let headers = vec![("Content-Length".to_owned(), "42".to_owned())];
        assert_eq!(response_content_length(&headers), Some(42));
    }

    #[test]
    fn response_content_length_missing() {
        let headers = vec![("X-Custom".to_owned(), "value".to_owned())];
        assert_eq!(response_content_length(&headers), None);
    }

    #[test]
    fn response_content_length_invalid_value() {
        let headers = vec![("Content-Length".to_owned(), "not-a-number".to_owned())];
        assert_eq!(response_content_length(&headers), None);
    }

    #[test]
    fn response_content_length_case_insensitive() {
        let headers = vec![("content-length".to_owned(), "100".to_owned())];
        assert_eq!(response_content_length(&headers), Some(100));

        let headers_upper = vec![("CONTENT-LENGTH".to_owned(), "200".to_owned())];
        assert_eq!(response_content_length(&headers_upper), Some(200));
    }

    #[test]
    fn response_content_length_trims_whitespace() {
        let headers = vec![("Content-Length".to_owned(), "  300  ".to_owned())];
        assert_eq!(response_content_length(&headers), Some(300));
    }

    #[test]
    fn sha256_digest_hex_known_empty() {
        // SHA-256 of empty data
        let hash = sha256_hex(b"");
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_digest_hex_always_64_chars() {
        let hash = sha256_hex(b"test");
        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn sha256_digest_hex_lowercase() {
        let hash = sha256_hex(b"ABC");
        // Hex should be lowercase
        assert_eq!(hash, hash.to_lowercase());
    }

    #[test]
    fn format_bytes_boundary_values() {
        // Just below KB
        assert_eq!(format_bytes(1023), "1023 B");
        // Exactly KB
        assert_eq!(format_bytes(1024), "1.0 KB");
        // Just below MB
        assert!(format_bytes(1024 * 1024 - 1).contains("KB"));
        // Exactly MB
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        // Just below GB
        assert!(format_bytes(1024 * 1024 * 1024 - 1).contains("MB"));
        // Exactly GB
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn format_bytes_large_values() {
        let ten_gb = 10 * 1024 * 1024 * 1024_u64;
        let result = format_bytes(ten_gb);
        assert!(result.contains("GB"));
        assert!(result.contains("10.0"));
    }

    #[test]
    fn temp_file_guard_armed_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("temp.bin");
        std::fs::write(&path, b"data").unwrap();
        assert!(path.exists());
        {
            let _guard = TempFileGuard::new(path.clone());
            // guard drops here, armed=true
        }
        assert!(!path.exists(), "armed guard should remove file on drop");
    }

    #[test]
    fn temp_file_guard_disarmed_does_not_clean_up() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keep.bin");
        std::fs::write(&path, b"data").unwrap();
        assert!(path.exists());
        {
            let mut guard = TempFileGuard::new(path.clone());
            guard.disarm();
            // guard drops here, armed=false
        }
        assert!(path.exists(), "disarmed guard should leave file intact");
    }

    #[test]
    fn temp_file_guard_armed_nonexistent_file_does_not_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonexistent.bin");
        {
            let _guard = TempFileGuard::new(path);
            // guard drops on nonexistent file — should not panic
        }
    }

    #[test]
    fn download_progress_debug() {
        let progress = DownloadProgress {
            file_name: "model.onnx".to_owned(),
            bytes_downloaded: 0,
            total_bytes: Some(100),
            files_completed: 0,
            files_total: 1,
            speed_bytes_per_sec: 0.0,
            eta_seconds: None,
        };
        let debug = format!("{progress:?}");
        assert!(debug.contains("DownloadProgress"));
        assert!(debug.contains("model.onnx"));
    }

    #[test]
    fn download_progress_clone() {
        let progress = DownloadProgress {
            file_name: "file.bin".to_owned(),
            bytes_downloaded: 42,
            total_bytes: Some(100),
            files_completed: 1,
            files_total: 2,
            speed_bytes_per_sec: 1000.0,
            eta_seconds: Some(0.058),
        };
        #[allow(clippy::redundant_clone)]
        let cloned = progress.clone();
        assert_eq!(cloned.file_name, "file.bin");
        assert_eq!(cloned.bytes_downloaded, 42);
        assert_eq!(cloned.total_bytes, Some(100));
        assert_eq!(cloned.files_completed, 1);
        assert_eq!(cloned.files_total, 2);
    }

    #[test]
    fn download_config_clone_and_debug() {
        let config = DownloadConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.max_retries, config.max_retries);
        assert_eq!(cloned.user_agent, config.user_agent);
        let debug = format!("{config:?}");
        assert!(debug.contains("DownloadConfig"));
        assert!(debug.contains("max_retries"));
    }

    #[test]
    fn model_downloader_with_defaults_creates_valid_instance() {
        let _downloader = ModelDownloader::with_defaults();
    }

    #[test]
    fn download_progress_display_zero_total_bytes_no_percent() {
        let progress = DownloadProgress {
            file_name: "test.bin".to_owned(),
            bytes_downloaded: 500,
            total_bytes: Some(0),
            files_completed: 0,
            files_total: 1,
            speed_bytes_per_sec: 0.0,
            eta_seconds: None,
        };
        let display = progress.to_string();
        // total_bytes=0 is filtered out, so no percentage
        assert!(!display.contains('%'));
    }

    #[test]
    fn create_unique_staging_dir_creates_parent_if_needed() {
        let temp = tempfile::tempdir().unwrap();
        let nested = temp.path().join("deeply").join("nested").join("dir");
        let result = create_unique_staging_dir(&nested).expect("should create nested dir");
        assert!(result.is_dir());
        assert!(nested.is_dir());
    }

    // ─── bd-r476 tests end ───

    #[test]
    fn download_model_rejects_non_production_ready_manifest() {
        let mut manifest = ModelManifest::minilm_v2();
        manifest.revision = PLACEHOLDER_PINNED_REVISION.to_owned();
        manifest.files[0].sha256 = PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned();
        manifest.files[0].size = 0;
        manifest.files[0].url = None;
        let consent = crate::model_manifest::DownloadConsent::granted(
            crate::model_manifest::ConsentSource::Environment,
        );
        let mut lifecycle = ModelLifecycle::new(manifest.clone(), consent);
        let dest = tempfile::tempdir().unwrap();
        let downloader = ModelDownloader::new(DownloadConfig {
            max_retries: 0,
            retry_base_delay: Duration::from_millis(1),
            user_agent: "frankensearch-test".to_owned(),
            max_redirects: 0,
        });

        run_test_with_cx(|_cx| async move {
            let err = downloader
                .download_model(&manifest, dest.path(), &mut lifecycle, |_| {})
                .await
                .unwrap_err();
            assert!(matches!(err, SearchError::InvalidConfig { .. }));
            assert!(err.to_string().contains("production-ready"));
            assert!(matches!(
                lifecycle.state(),
                crate::model_manifest::ModelState::VerificationFailed { .. }
            ));
        });
    }
}
