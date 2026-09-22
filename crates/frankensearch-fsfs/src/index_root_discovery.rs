//! Locate an existing index without interpreting or following its control files.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub fn find_ancestor_index(
    directory: &Path,
    configured: &Path,
    markers: &[&str],
) -> io::Result<PathBuf> {
    for ancestor in directory.ancestors() {
        let candidate = ancestor.join(configured);
        for marker in markers {
            match fs::symlink_metadata(candidate.join(marker)) {
                Ok(_) => return Ok(candidate),
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
        }
    }
    Ok(directory.join(configured))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "fsfs-root-discovery-{}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, Ordering::Relaxed),
            ));
            fs::create_dir(&root).unwrap();
            Self(root)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            // Only this test's freshly allocated directory is owned here.
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    const MARKERS: &[&str] = &["FSFS-CURRENT", "generations", "index.sentinel.json"];

    #[test]
    fn discovers_complete_or_legacy_parent_without_creating_a_child_index() {
        for marker in MARKERS {
            let fixture = Fixture::new();
            let child = fixture.0.join("project/src/module");
            let index = fixture.0.join("project/.frankensearch");
            fs::create_dir_all(&child).unwrap();
            fs::create_dir(&index).unwrap();
            if *marker == "generations" {
                fs::create_dir(index.join(marker)).unwrap();
            } else {
                // Discovery must select even a corrupt control for admission
                // to reject, rather than silently searching a different root.
                fs::write(index.join(marker), "corrupt control").unwrap();
            }
            assert_eq!(
                find_ancestor_index(&child, Path::new(".frankensearch"), MARKERS).unwrap(),
                index,
            );
            assert!(!child.join(".frankensearch").exists());
        }
    }

    #[test]
    fn nearest_index_wins_and_an_absent_index_uses_the_requested_directory() {
        let fixture = Fixture::new();
        let child = fixture.0.join("project/src");
        fs::create_dir_all(&child).unwrap();
        assert_eq!(
            find_ancestor_index(&child, Path::new(".frankensearch"), MARKERS).unwrap(),
            child.join(".frankensearch"),
        );
        let outer = fixture.0.join(".frankensearch");
        let inner = fixture.0.join("project/.frankensearch");
        for index in [&outer, &inner] {
            fs::create_dir(index).unwrap();
            fs::write(index.join("FSFS-CURRENT"), "control").unwrap();
        }
        assert_eq!(
            find_ancestor_index(&child, Path::new(".frankensearch"), MARKERS).unwrap(),
            inner,
        );
    }

    #[cfg(unix)]
    #[test]
    fn dangling_control_is_not_permission_to_fall_back_to_an_outer_index() {
        let fixture = Fixture::new();
        let child = fixture.0.join("project/src");
        let outer = fixture.0.join(".frankensearch");
        let inner = fixture.0.join("project/.frankensearch");
        fs::create_dir_all(&child).unwrap();
        fs::create_dir(&outer).unwrap();
        fs::create_dir(&inner).unwrap();
        fs::write(outer.join("index.sentinel.json"), "legacy").unwrap();
        std::os::unix::fs::symlink("missing-target", inner.join("FSFS-CURRENT")).unwrap();
        assert_eq!(
            find_ancestor_index(&child, Path::new(".frankensearch"), MARKERS).unwrap(),
            inner,
        );
    }

    #[test]
    fn invalid_index_directory_is_an_error_instead_of_an_ancestor_fallback() {
        let fixture = Fixture::new();
        let child = fixture.0.join("project/src");
        fs::create_dir_all(&child).unwrap();
        fs::write(child.join(".frankensearch"), "not a directory").unwrap();
        assert!(find_ancestor_index(&child, Path::new(".frankensearch"), MARKERS).is_err());
    }
}
