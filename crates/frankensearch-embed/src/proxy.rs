//! Outbound HTTP proxy selection for model downloads and hosted embedding APIs.
//!
//! `asupersync`'s HTTP client supports `http://`, `https://` and `socks5://` proxies but
//! reads no environment variables; choosing a proxy is left to the application. This
//! module applies the conventions curl and most command-line tools share, so `fsfs`
//! works on networks whose only route out is a proxy.

use std::fmt;

/// How outbound HTTP requests reach the network.
#[derive(Clone, Default, PartialEq, Eq)]
pub enum ProxySetting {
    /// Use the conventional proxy environment variables, as curl does:
    /// `https_proxy` for `https://` URLs and `http_proxy` for `http://` URLs, falling back
    /// to `all_proxy`. Each is read lower case first, then upper case, except
    /// `http_proxy`, which is lower case only (upper-case `HTTP_PROXY` can be set from a
    /// request header under CGI, the "httpoxy" issue). Hosts matched by `no_proxy`
    /// connect directly. Unset or empty variables mean a direct connection.
    #[default]
    FromEnvironment,
    /// Always connect directly.
    Disabled,
    /// Send every request through this proxy (`http://`, `https://`, `socks5://` or
    /// `socks5h://`), ignoring the environment, `no_proxy` included.
    Url(String),
}

impl fmt::Debug for ProxySetting {
    // A proxy URL can carry credentials (`http://user:pass@proxy:3128`), and configs
    // holding this type are logged with `{:?}`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FromEnvironment => f.write_str("FromEnvironment"),
            Self::Disabled => f.write_str("Disabled"),
            Self::Url(_) => f.write_str("Url(<redacted>)"),
        }
    }
}

/// The proxy URL to use for a request to `url`, or `None` to connect directly.
///
/// `lookup` reads an environment variable; production callers pass
/// `|name| std::env::var(name).ok()`. `no_proxy` is matched against `url`'s host only:
/// a redirect followed within the same request stays on the route chosen for its first
/// URL, because the HTTP client chooses its proxy once per client.
pub(crate) fn proxy_for_url(
    setting: &ProxySetting,
    url: &str,
    lookup: impl Fn(&str) -> Option<String>,
) -> Option<String> {
    let proxy = match setting {
        ProxySetting::Disabled => return None,
        ProxySetting::Url(proxy) => proxy.clone(),
        ProxySetting::FromEnvironment => {
            let (scheme, host) = scheme_and_host(url)?;
            if let Some(no_proxy) = env_value(&lookup, "no_proxy", true) {
                if no_proxy_matches(&host, &no_proxy) {
                    return None;
                }
            }
            let by_scheme = match scheme.as_str() {
                "https" => env_value(&lookup, "https_proxy", true),
                "http" => env_value(&lookup, "http_proxy", false),
                _ => return None,
            };
            by_scheme.or_else(|| env_value(&lookup, "all_proxy", true))?
        }
    };
    Some(normalize_proxy_url(&proxy))
}

/// The trimmed, non-empty value of `name` (lower case), then of its upper-case form.
fn env_value(
    lookup: &impl Fn(&str) -> Option<String>,
    name: &str,
    allow_upper: bool,
) -> Option<String> {
    let upper = name.to_ascii_uppercase();
    let names: &[&str] = if allow_upper {
        &[name, &upper]
    } else {
        &[name]
    };
    names
        .iter()
        .filter_map(|candidate| lookup(candidate))
        .map(|value| value.trim().to_owned())
        .find(|value| !value.is_empty())
}

/// curl's `socks5h://` (resolve names at the proxy) is what asupersync's `socks5://`
/// already does: it always sends host names in the CONNECT request.
fn normalize_proxy_url(proxy: &str) -> String {
    proxy
        .strip_prefix("socks5h://")
        .map_or_else(|| proxy.to_owned(), |rest| format!("socks5://{rest}"))
}

/// The lower-cased scheme and host of an absolute URL (no brackets around IPv6).
fn scheme_and_host(url: &str) -> Option<(String, String)> {
    let (scheme, rest) = url.split_once("://")?;
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    let host_port = authority.rsplit_once('@').map_or(authority, |(_, hp)| hp);
    Some((
        scheme.to_ascii_lowercase(),
        strip_port(host_port).to_ascii_lowercase(),
    ))
}

/// `host[:port]` or `[ipv6][:port]` without the port and brackets.
fn strip_port(host_port: &str) -> &str {
    if let Some(rest) = host_port.strip_prefix('[') {
        return rest.split(']').next().unwrap_or_default();
    }
    match host_port.rsplit_once(':') {
        // A bare IPv6 literal has several colons and no port to strip.
        Some((host, _)) if !host.contains(':') => host,
        _ => host_port,
    }
}

/// curl's rules: comma-separated entries; `*` matches every host; otherwise an entry
/// matches the host itself or any subdomain of it. A leading dot and a port are ignored,
/// and matching is case-insensitive.
fn no_proxy_matches(host: &str, no_proxy: &str) -> bool {
    no_proxy
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .any(|entry| {
            if entry == "*" {
                return true;
            }
            let entry = strip_port(entry.trim_start_matches('.')).to_ascii_lowercase();
            !entry.is_empty()
                && (host == entry
                    || host
                        .strip_suffix(entry.as_str())
                        .is_some_and(|prefix| prefix.ends_with('.')))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned())
        }
    }

    #[test]
    fn selects_the_variable_by_scheme_and_falls_back_to_all_proxy() {
        let setting = ProxySetting::FromEnvironment;
        let env = env_of(&[
            ("https_proxy", "http://secure:3128"),
            ("http_proxy", "http://plain:3128"),
            ("all_proxy", "http://any:3128"),
        ]);
        assert_eq!(
            proxy_for_url(&setting, "https://huggingface.co/x", &env).as_deref(),
            Some("http://secure:3128")
        );
        assert_eq!(
            proxy_for_url(&setting, "http://mirror.local/x", &env).as_deref(),
            Some("http://plain:3128")
        );
        let only_all = env_of(&[("ALL_PROXY", "http://any:3128")]);
        assert_eq!(
            proxy_for_url(&setting, "https://huggingface.co/x", &only_all).as_deref(),
            Some("http://any:3128")
        );
        assert_eq!(
            proxy_for_url(&setting, "https://huggingface.co/x", env_of(&[])),
            None
        );
        assert_eq!(
            proxy_for_url(&setting, "ftp://huggingface.co/x", &env),
            None,
            "only http and https targets are proxied"
        );
    }

    #[test]
    fn reads_lower_case_first_skips_empty_values_and_ignores_upper_http_proxy() {
        let setting = ProxySetting::FromEnvironment;
        let both = env_of(&[
            ("https_proxy", "http://lower:1"),
            ("HTTPS_PROXY", "http://upper:1"),
        ]);
        assert_eq!(
            proxy_for_url(&setting, "https://huggingface.co/x", &both).as_deref(),
            Some("http://lower:1")
        );
        let empty_lower = env_of(&[("https_proxy", "  "), ("HTTPS_PROXY", "http://upper:1")]);
        assert_eq!(
            proxy_for_url(&setting, "https://huggingface.co/x", &empty_lower).as_deref(),
            Some("http://upper:1")
        );
        let upper_http = env_of(&[("HTTP_PROXY", "http://attacker:1")]);
        assert_eq!(
            proxy_for_url(&setting, "http://mirror.local/x", &upper_http),
            None,
            "upper-case HTTP_PROXY is ignored (httpoxy)"
        );
    }

    #[test]
    fn honors_no_proxy_like_curl() {
        let setting = ProxySetting::FromEnvironment;
        let direct = |no_proxy: &str, url: &str| {
            let pairs = [("https_proxy", "http://p:1"), ("no_proxy", no_proxy)];
            proxy_for_url(&setting, url, env_of(&pairs)).is_none()
        };
        assert!(
            direct("huggingface.co", "https://huggingface.co/x"),
            "exact host"
        );
        assert!(
            direct("huggingface.co", "https://cdn.huggingface.co/x"),
            "subdomain"
        );
        assert!(
            direct(".huggingface.co", "https://huggingface.co/x"),
            "leading dot"
        );
        assert!(
            direct("HuggingFace.co:443", "https://huggingface.co:443/x"),
            "case and port"
        );
        assert!(
            direct("localhost, huggingface.co", "https://huggingface.co/x"),
            "list"
        );
        assert!(direct("*", "https://huggingface.co/x"), "wildcard");
        assert!(direct("::1", "https://[::1]:8443/x"), "ipv6 literal");
        assert!(
            direct("huggingface.co", "https://user:pw@huggingface.co/x"),
            "userinfo"
        );
        assert!(
            !direct("face.co", "https://huggingface.co/x"),
            "suffix needs a dot boundary"
        );
        assert!(
            !direct("example.com", "https://huggingface.co/x"),
            "other host"
        );
        let upper = [
            ("HTTPS_PROXY", "http://p:1"),
            ("NO_PROXY", "huggingface.co"),
        ];
        assert!(proxy_for_url(&setting, "https://huggingface.co/x", env_of(&upper)).is_none());
    }

    #[test]
    fn maps_socks5h_to_asupersync_socks5() {
        let env = env_of(&[("all_proxy", "socks5h://127.0.0.1:1080")]);
        assert_eq!(
            proxy_for_url(
                &ProxySetting::FromEnvironment,
                "https://huggingface.co/x",
                &env
            )
            .as_deref(),
            Some("socks5://127.0.0.1:1080")
        );
        let explicit = ProxySetting::Url("socks5h://user:pw@proxy:1080".to_owned());
        assert_eq!(
            proxy_for_url(&explicit, "https://huggingface.co/x", env_of(&[])).as_deref(),
            Some("socks5://user:pw@proxy:1080")
        );
    }

    #[test]
    fn explicit_settings_ignore_the_environment() {
        let env = env_of(&[("https_proxy", "http://env:1"), ("no_proxy", "*")]);
        assert_eq!(
            proxy_for_url(&ProxySetting::Disabled, "https://huggingface.co/x", &env),
            None
        );
        let explicit = ProxySetting::Url("http://chosen:3128".to_owned());
        assert_eq!(
            proxy_for_url(&explicit, "https://huggingface.co/x", &env).as_deref(),
            Some("http://chosen:3128")
        );
    }

    #[test]
    fn debug_redacts_the_proxy_url() {
        let setting = ProxySetting::Url("http://user:hunter2@proxy.corp:3128".to_owned());
        let rendered = format!("{setting:?}");
        assert_eq!(rendered, "Url(<redacted>)");
    }
}
