//! Platform base-directory lookups without the `dirs` dependency.
//!
//! In-tree replacement for the three `dirs` crate calls frankensearch
//! actually made (`home_dir`, `data_local_dir`, `cache_dir`). The `dirs`
//! crate resolves these through `dirs-sys` (and, transitively, platform
//! shims like `option-ext`/`redox_users`/`windows-sys`) to cover exotic
//! configurations frankensearch does not run in; what our call sites need
//! is the documented environment-first contract below, implemented
//! directly and testably.
//!
//! Contract:
//!
//! * Environment values that are empty or relative are ignored. For the
//!   XDG variables this is the XDG Base Directory specification's own
//!   rule (a relative `XDG_CACHE_HOME` must be treated as unset, not
//!   joined onto an accidental working directory); for `HOME` /
//!   `USERPROFILE` it is the same principle — a relative home directory
//!   is never a meaningful answer to the "durable per-user location"
//!   question these lookups exist to answer.
//! * [`home_dir`] is environment-based and returns `None` when unset
//!   rather than guessing through passwd lookups; every frankensearch
//!   call site already carries a fallback for the `None` arm.
//! * The resolution logic is pure over an injected environment reader
//!   (the `*_from` functions), so tests exercise every branch without
//!   mutating process-global state.

use std::env;
use std::ffi::OsString;
use std::path::PathBuf;

/// Environment reader: `name -> value`, `None` when unset.
type EnvReader<'lookup> = &'lookup dyn Fn(&str) -> Option<OsString>;

fn process_env(name: &str) -> Option<OsString> {
    env::var_os(name)
}

/// A non-empty, absolute environment value as a path; `None` otherwise.
fn absolute_env_path(env_reader: EnvReader<'_>, variable: &str) -> Option<PathBuf> {
    let value = env_reader(variable)?;
    if value.is_empty() {
        return None;
    }
    let path = PathBuf::from(value);
    path.is_absolute().then_some(path)
}

fn home_dir_from(env_reader: EnvReader<'_>) -> Option<PathBuf> {
    let variable = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    absolute_env_path(env_reader, variable)
}

fn data_local_dir_from(env_reader: EnvReader<'_>) -> Option<PathBuf> {
    if cfg!(windows) {
        absolute_env_path(env_reader, "LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        home_dir_from(env_reader).map(|home| home.join("Library").join("Application Support"))
    } else {
        absolute_env_path(env_reader, "XDG_DATA_HOME")
            .or_else(|| home_dir_from(env_reader).map(|home| home.join(".local").join("share")))
    }
}

fn cache_dir_from(env_reader: EnvReader<'_>) -> Option<PathBuf> {
    if cfg!(windows) {
        absolute_env_path(env_reader, "LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        home_dir_from(env_reader).map(|home| home.join("Library").join("Caches"))
    } else {
        absolute_env_path(env_reader, "XDG_CACHE_HOME")
            .or_else(|| home_dir_from(env_reader).map(|home| home.join(".cache")))
    }
}

fn data_dir_from(env_reader: EnvReader<'_>) -> Option<PathBuf> {
    if cfg!(windows) {
        absolute_env_path(env_reader, "APPDATA")
    } else {
        // Roaming and local data coincide on Unix platforms.
        data_local_dir_from(env_reader)
    }
}

fn config_dir_from(env_reader: EnvReader<'_>) -> Option<PathBuf> {
    if cfg!(windows) {
        absolute_env_path(env_reader, "APPDATA")
    } else if cfg!(target_os = "macos") {
        home_dir_from(env_reader).map(|home| home.join("Library").join("Application Support"))
    } else {
        absolute_env_path(env_reader, "XDG_CONFIG_HOME")
            .or_else(|| home_dir_from(env_reader).map(|home| home.join(".config")))
    }
}

fn runtime_dir_from(env_reader: EnvReader<'_>) -> Option<PathBuf> {
    if cfg!(all(unix, not(target_os = "macos"))) {
        // XDG_RUNTIME_DIR has no defined fallback: absent means absent,
        // and callers already chain their own .or_else fallbacks.
        absolute_env_path(env_reader, "XDG_RUNTIME_DIR")
    } else {
        None
    }
}

/// The current user's home directory (`HOME` on Unix, `USERPROFILE` on
/// Windows), or `None` when the environment does not define one.
#[must_use]
pub fn home_dir() -> Option<PathBuf> {
    home_dir_from(&process_env)
}

/// The per-user local (non-roaming) data directory.
///
/// Linux and other non-Apple Unix: `$XDG_DATA_HOME`, else
/// `~/.local/share`. macOS: `~/Library/Application Support`. Windows:
/// `%LOCALAPPDATA%`.
#[must_use]
pub fn data_local_dir() -> Option<PathBuf> {
    data_local_dir_from(&process_env)
}

/// The per-user cache directory.
///
/// Linux and other non-Apple Unix: `$XDG_CACHE_HOME`, else `~/.cache`.
/// macOS: `~/Library/Caches`. Windows: `%LOCALAPPDATA%`.
#[must_use]
pub fn cache_dir() -> Option<PathBuf> {
    cache_dir_from(&process_env)
}

/// The per-user (roaming, on Windows) data directory.
///
/// Linux and other non-Apple Unix: `$XDG_DATA_HOME`, else
/// `~/.local/share`. macOS: `~/Library/Application Support`. Windows:
/// `%APPDATA%`.
#[must_use]
pub fn data_dir() -> Option<PathBuf> {
    data_dir_from(&process_env)
}

/// The per-user configuration directory.
///
/// Linux and other non-Apple Unix: `$XDG_CONFIG_HOME`, else
/// `~/.config`. macOS: `~/Library/Application Support`. Windows:
/// `%APPDATA%`.
#[must_use]
pub fn config_dir() -> Option<PathBuf> {
    config_dir_from(&process_env)
}

/// The per-user runtime directory (`$XDG_RUNTIME_DIR`).
///
/// `None` when unset and on platforms without the concept (macOS,
/// Windows) — exactly the `dirs` behavior our call sites chain
/// `.or_else` fallbacks onto.
#[must_use]
pub fn runtime_dir() -> Option<PathBuf> {
    runtime_dir_from(&process_env)
}

#[cfg(test)]
mod tests {
    use super::{
        absolute_env_path, cache_dir_from, config_dir_from, data_dir_from, data_local_dir_from,
        home_dir_from, runtime_dir_from,
    };
    use std::collections::HashMap;
    use std::ffi::OsString;
    use std::path::PathBuf;

    fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> {
        let map: HashMap<String, OsString> = pairs
            .iter()
            .map(|(name, value)| ((*name).to_string(), OsString::from(value)))
            .collect();
        move |name: &str| map.get(name).cloned()
    }

    #[test]
    fn empty_and_relative_env_values_are_ignored() {
        let empty = env_of(&[("XDG_CACHE_HOME", "")]);
        assert_eq!(absolute_env_path(&empty, "XDG_CACHE_HOME"), None);

        let relative = env_of(&[("XDG_CACHE_HOME", "relative/cache")]);
        assert_eq!(
            absolute_env_path(&relative, "XDG_CACHE_HOME"),
            None,
            "relative XDG values must be treated as unset per the spec"
        );

        let unset = env_of(&[]);
        assert_eq!(absolute_env_path(&unset, "XDG_CACHE_HOME"), None);
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    #[test]
    fn xdg_overrides_win_then_home_then_none() {
        let with_xdg = env_of(&[
            ("HOME", "/test-home"),
            ("XDG_DATA_HOME", "/xdg-data"),
            ("XDG_CACHE_HOME", "/xdg-cache"),
        ]);
        assert_eq!(
            data_local_dir_from(&with_xdg),
            Some(PathBuf::from("/xdg-data"))
        );
        assert_eq!(cache_dir_from(&with_xdg), Some(PathBuf::from("/xdg-cache")));

        let home_only = env_of(&[("HOME", "/test-home")]);
        assert_eq!(
            data_local_dir_from(&home_only),
            Some(PathBuf::from("/test-home/.local/share"))
        );
        assert_eq!(
            cache_dir_from(&home_only),
            Some(PathBuf::from("/test-home/.cache"))
        );

        // A relative XDG value must fall through to the HOME-derived
        // default, not shadow it.
        let relative_xdg = env_of(&[("HOME", "/test-home"), ("XDG_CACHE_HOME", "rel")]);
        assert_eq!(
            cache_dir_from(&relative_xdg),
            Some(PathBuf::from("/test-home/.cache"))
        );

        let bare = env_of(&[]);
        assert_eq!(home_dir_from(&bare), None);
        assert_eq!(data_local_dir_from(&bare), None);
        assert_eq!(cache_dir_from(&bare), None);
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    #[test]
    fn config_data_and_runtime_follow_xdg_rules() {
        let env = env_of(&[
            ("HOME", "/test-home"),
            ("XDG_CONFIG_HOME", "/xdg-config"),
            ("XDG_RUNTIME_DIR", "/run/user/1000"),
        ]);
        assert_eq!(config_dir_from(&env), Some(PathBuf::from("/xdg-config")));
        assert_eq!(
            runtime_dir_from(&env),
            Some(PathBuf::from("/run/user/1000"))
        );
        // Roaming data coincides with local data off Windows.
        assert_eq!(
            data_dir_from(&env),
            Some(PathBuf::from("/test-home/.local/share"))
        );

        let home_only = env_of(&[("HOME", "/test-home")]);
        assert_eq!(
            config_dir_from(&home_only),
            Some(PathBuf::from("/test-home/.config"))
        );
        assert_eq!(
            runtime_dir_from(&home_only),
            None,
            "XDG_RUNTIME_DIR has no fallback by specification"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_uses_library_locations() {
        let env = env_of(&[("HOME", "/Users/test")]);
        assert_eq!(home_dir_from(&env), Some(PathBuf::from("/Users/test")));
        assert_eq!(
            data_local_dir_from(&env),
            Some(PathBuf::from("/Users/test/Library/Application Support"))
        );
        assert_eq!(
            data_dir_from(&env),
            Some(PathBuf::from("/Users/test/Library/Application Support"))
        );
        assert_eq!(
            config_dir_from(&env),
            Some(PathBuf::from("/Users/test/Library/Application Support"))
        );
        assert_eq!(
            cache_dir_from(&env),
            Some(PathBuf::from("/Users/test/Library/Caches"))
        );
        assert_eq!(runtime_dir_from(&env), None);
    }

    #[cfg(windows)]
    #[test]
    fn windows_uses_profile_and_local_appdata() {
        let env = env_of(&[
            ("USERPROFILE", r"C:\Users\test"),
            ("LOCALAPPDATA", r"C:\Users\test\AppData\Local"),
        ]);
        assert_eq!(home_dir_from(&env), Some(PathBuf::from(r"C:\Users\test")));
        assert_eq!(
            data_local_dir_from(&env),
            Some(PathBuf::from(r"C:\Users\test\AppData\Local"))
        );
        assert_eq!(
            cache_dir_from(&env),
            Some(PathBuf::from(r"C:\Users\test\AppData\Local"))
        );
    }
}
