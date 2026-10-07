//! Native qualification of the receipt store used by the pending reader policy.
//!
//! Import the real source rather than a copy or a fake protocol worker. Cargo's
//! ordinary integration-test discovery now compiles its embedded adversarial
//! tests without waiting for the separate Keeper and reader integration.
//! This target does not enable receipt reuse in production.
#![cfg(target_os = "linux")]

// The store binds receipts to the actual engine and segment format versions.
// These aliases preserve its crate-relative references in this test crate.
mod keeper {
    pub use frankensearch_quill::CURRENT_ENGINE_VERSION;
}

mod segment {
    pub use frankensearch_quill::FSLX_FORMAT_VERSION;
}

#[path = "../src/read_open_receipts.rs"]
mod read_open_receipts;

#[test]
fn receipt_storage_accepts_tmpfs_and_rejects_procfs() -> Result<(), Box<dyn std::error::Error>> {
    // Positive checks must exercise supported storage, not silently turn into
    // strict-fallback tests when the checkout itself lives on overlayfs.
    let local = tempfile::tempfile_in("/dev/shm")?;
    assert!(
        read_open_receipts::supported(&local),
        "the positive receipt fixture requires supported local /dev/shm storage"
    );
    let proc = std::fs::File::open("/proc/self/status")?;
    assert!(
        !read_open_receipts::supported(&proc),
        "a readable descriptor on an unsupported filesystem must not admit receipts"
    );
    Ok(())
}
