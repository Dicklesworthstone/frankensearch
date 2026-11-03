//! Outbound HTTP proxy selection for model downloads and hosted embedding APIs.
//!
//! `asupersync`'s HTTP client supports `http://`, `https://` and `socks5://` proxies but
//! reads no environment variables; choosing a proxy is left to the application. This
//! module applies the conventions curl and most command-line tools share, so `fsfs`
//! works on networks whose only route out is a proxy.
//!
//! Based on the resolver contributed by `tkfor21` in reference PR #71.

use std::fmt;
use std::net::{IpAddr, Ipv6Addr};

use asupersync::http::h1::{ParsedUrl, Scheme};
use frankensearch_core::error::{SearchError, SearchResult};

/// How outbound HTTP requests reach the network.
#[derive(Clone, Default, PartialEq, Eq)]
pub enum ProxySetting {
    /// Use the conventional proxy environment variables, as curl does:
    /// `https_proxy` for `https://` URLs and `http_proxy` for `http://` URLs, falling back
    /// to `all_proxy`. Each is read lower case first, then upper case, except
    /// `http_proxy`, which is lower case only (upper-case `HTTP_PROXY` can be set from a
    /// request header under CGI, the "httpoxy" issue). Hosts matched by `no_proxy`
    /// connect directly. Unset or empty variables mean a direct connection.
    /// A selected non-Unicode value is an invalid configuration; it never
    /// authorizes a direct connection or a lower-priority proxy fallback.
    ///
    /// Bypass entries are comma-separated DNS names, IP addresses, IP networks in
    /// CIDR notation, or `*` for every host. DNS matching includes subdomains at
    /// dot boundaries and ignores ASCII case and a trailing root dot. An optional
    /// `:port` restricts a host entry to that destination port; IPv6 entries with a
    /// port must use `[address]:port`. CIDR matching applies only to literal IP
    /// destinations and never triggers a local DNS lookup.
    #[default]
    FromEnvironment,
    /// Always connect directly.
    Disabled,
    /// Send every request through this proxy (`http://`, `https://`, `socks5://` or
    /// `socks5h://`), ignoring the environment, `no_proxy` included.
    ///
    /// A scheme is required. Both SOCKS5 spellings resolve destination names at
    /// the proxy, following `asupersync`'s transport behavior. Proxy URL credentials
    /// are passed unchanged: `asupersync` 0.5.0 does not percent-decode userinfo, so
    /// percent-encoded usernames/passwords are not supported. Invalid proxy URLs
    /// fail the request rather than silently falling back to a direct connection.
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

/// Resolve one request against the process environment with the same httpoxy
/// protection on every platform, including Windows' case-insensitive lookup.
pub(crate) fn proxy_from_environment(
    setting: &ProxySetting,
    url: &str,
) -> SearchResult<Option<String>> {
    proxy_for_url(setting, url, |name| {
        #[cfg(windows)]
        if name == "http_proxy" {
            // `var("http_proxy")` can return an uppercase HTTP_PROXY on Windows.
            // Inspect the stored spelling so a CGI Proxy header cannot select a
            // proxy merely because the OS ignores environment-variable case.
            return exact_environment_value(name, std::env::vars_os());
        }
        environment_value(name, std::env::var_os(name))
    })
}

/// Preserve the distinction between an absent setting and one that cannot be
/// decoded. Never retain the rejected value in an error: it may contain proxy
/// credentials. The caller's lazy lookup order decides whether it is relevant.
fn environment_value(
    name: &str,
    value: Option<std::ffi::OsString>,
) -> SearchResult<Option<String>> {
    value
        .map(std::ffi::OsString::into_string)
        .transpose()
        .map_err(|_| SearchError::InvalidConfig {
            field: format!("proxy.{name}"),
            value: "<non-Unicode>".to_owned(),
            reason: "proxy environment variables must contain valid Unicode".to_owned(),
        })
}

#[cfg(any(windows, test))]
fn exact_environment_value(
    name: &str,
    variables: impl IntoIterator<Item = (std::ffi::OsString, std::ffi::OsString)>,
) -> SearchResult<Option<String>> {
    let value = variables
        .into_iter()
        .find_map(|(key, value)| (key == std::ffi::OsStr::new(name)).then_some(value));
    environment_value(name, value)
}

/// The proxy URL to use for a request to `url`, or `None` to connect directly.
///
/// `lookup` reads an environment variable; production callers use
/// [`proxy_from_environment`]. Callers must resolve again for each allowed
/// redirect destination, because the transport fixes its proxy at construction.
/// A selected but malformed proxy remains `Some` so the transport rejects it;
/// configuration errors must never silently select a direct connection.
/// Errors reading a selected environment value propagate before any transport
/// is constructed; unselected fallback variables are not read.
pub(crate) fn proxy_for_url(
    setting: &ProxySetting,
    url: &str,
    lookup: impl Fn(&str) -> SearchResult<Option<String>>,
) -> SearchResult<Option<String>> {
    let proxy = match setting {
        ProxySetting::Disabled => return Ok(None),
        ProxySetting::Url(proxy) => proxy.clone(),
        ProxySetting::FromEnvironment => {
            let Some(target) = parse_target(url) else {
                return Ok(None);
            };
            if let Some(no_proxy) = env_value(&lookup, "no_proxy", true)? {
                if no_proxy_matches(&target, &no_proxy) {
                    return Ok(None);
                }
            }
            let by_scheme = match target.scheme {
                Scheme::Https => env_value(&lookup, "https_proxy", true),
                Scheme::Http => env_value(&lookup, "http_proxy", false),
            }?;
            let selected = if by_scheme.is_some() {
                by_scheme
            } else {
                env_value(&lookup, "all_proxy", true)?
            };
            let Some(proxy) = selected else {
                return Ok(None);
            };
            proxy
        }
    };
    Ok(Some(normalize_proxy_url(&proxy)))
}

/// The trimmed, non-empty value of `name` (lower case), then of its upper-case form.
fn env_value(
    lookup: &impl Fn(&str) -> SearchResult<Option<String>>,
    name: &str,
    allow_upper: bool,
) -> SearchResult<Option<String>> {
    let upper = name.to_ascii_uppercase();
    let names: &[&str] = if allow_upper {
        &[name, &upper]
    } else {
        &[name]
    };
    for candidate in names {
        if let Some(value) = lookup(candidate)? {
            let value = value.trim();
            if !value.is_empty() {
                return Ok(Some(value.to_owned()));
            }
        }
    }
    Ok(None)
}

/// curl's `socks5h://` (resolve names at the proxy) is what asupersync's `socks5://`
/// already does: it sends host names in the CONNECT request without resolving them
/// locally. Normalize only recognized schemes; never decode or log credentials.
fn normalize_proxy_url(proxy: &str) -> String {
    let proxy = proxy.trim();
    let Some((scheme, rest)) = proxy.split_once("://") else {
        return proxy.to_owned();
    };
    let normalized = if scheme.eq_ignore_ascii_case("http") {
        "http"
    } else if scheme.eq_ignore_ascii_case("https") {
        "https"
    } else if scheme.eq_ignore_ascii_case("socks5") || scheme.eq_ignore_ascii_case("socks5h") {
        "socks5"
    } else {
        return proxy.to_owned();
    };
    format!("{normalized}://{rest}")
}

/// Use the transport's authority parser so bypass and transport agree on the
/// destination. Scheme names are case-insensitive for proxy selection.
fn parse_target(url: &str) -> Option<ParsedUrl> {
    let (scheme, rest) = url.split_once("://")?;
    if scheme == "http" || scheme == "https" {
        return ParsedUrl::parse(url).ok();
    }
    let normalized = if scheme.eq_ignore_ascii_case("http") {
        "http"
    } else if scheme.eq_ignore_ascii_case("https") {
        "https"
    } else {
        return None;
    };
    ParsedUrl::parse(&format!("{normalized}://{rest}")).ok()
}

/// Parse a bypass host and optional port. A bare IPv6 literal has no port;
/// bracketed IPv6 is required when restricting a bypass to a specific port.
fn bypass_host_and_port(entry: &str) -> Option<(&str, Option<u16>)> {
    if let Some(rest) = entry.strip_prefix('[') {
        let (host, suffix) = rest.split_once(']')?;
        host.parse::<Ipv6Addr>().ok()?;
        let port = if suffix.is_empty() {
            None
        } else {
            Some(parse_port(suffix.strip_prefix(':')?)?)
        };
        return Some((host, port));
    }
    match entry.rsplit_once(':') {
        Some((host, port)) if !host.contains(':') => Some((host, Some(parse_port(port)?))),
        Some(_) => {
            entry.parse::<Ipv6Addr>().ok()?;
            Some((entry, None))
        }
        None => Some((entry, None)),
    }
}

fn parse_port(port: &str) -> Option<u16> {
    if port.is_empty() || !port.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    port.parse().ok()
}

fn canonical_host(host: &str) -> String {
    let host = host
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host);
    host.strip_suffix('.').unwrap_or(host).to_ascii_lowercase()
}

/// Host/domain bypass conventions with explicit port restrictions and IP/CIDR
/// support. Port-qualified rules are our defined behavior, not a claim that every
/// curl version implements them. Malformed rules cannot widen a bypass.
fn no_proxy_matches(target: &ParsedUrl, no_proxy: &str) -> bool {
    let host = canonical_host(&target.host);
    let host_ip = host.parse::<IpAddr>().ok();
    no_proxy
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .any(|entry| {
            if entry == "*" {
                return true;
            }
            if entry.contains('/') {
                return host_ip.is_some_and(|ip| cidr_matches(ip, entry));
            }
            let Some((entry, port)) = bypass_host_and_port(entry) else {
                return false;
            };
            if port.is_some_and(|port| port != target.port) {
                return false;
            }
            let entry = canonical_host(entry.strip_prefix('.').unwrap_or(entry));
            match (host_ip, entry.parse::<IpAddr>().ok()) {
                (Some(host), Some(entry)) => host == entry,
                // IP rules match addresses, never domain suffixes. DNS names
                // are not resolved locally just to evaluate bypass rules.
                (Some(_), None) | (None, Some(_)) => false,
                (None, None) => {
                    valid_bypass_domain(&entry)
                        && (host == entry
                            || host
                                .strip_suffix(entry.as_str())
                                .is_some_and(|prefix| prefix.ends_with('.')))
                }
            }
        })
}

fn valid_bypass_domain(domain: &str) -> bool {
    !domain.is_empty()
        && !domain.chars().any(|ch| {
            ch.is_control()
                || ch.is_whitespace()
                || matches!(ch, '/' | '?' | '#' | '@' | ':' | '[' | ']' | '*')
        })
        && domain.split('.').all(|label| !label.is_empty())
}

fn cidr_matches(host: IpAddr, entry: &str) -> bool {
    let Some((network, prefix)) = entry.split_once('/') else {
        return false;
    };
    if prefix.is_empty() || !prefix.bytes().all(|byte| byte.is_ascii_digit()) {
        return false;
    }
    let (Ok(network), Ok(prefix)) = (network.parse::<IpAddr>(), prefix.parse::<u32>()) else {
        return false;
    };
    match (host, network) {
        (IpAddr::V4(host), IpAddr::V4(network)) if prefix <= 32 => {
            let shift = 32 - prefix;
            u32::from(host).checked_shr(shift).unwrap_or(0)
                == u32::from(network).checked_shr(shift).unwrap_or(0)
        }
        (IpAddr::V6(host), IpAddr::V6(network)) if prefix <= 128 => {
            let shift = 128 - prefix;
            u128::from(host).checked_shr(shift).unwrap_or(0)
                == u128::from(network).checked_shr(shift).unwrap_or(0)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Keep the infallible string fixtures concise while exercising the same
    // fallible resolver used by real environment lookups.
    fn proxy_for_url(
        setting: &ProxySetting,
        url: &str,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Option<String> {
        super::proxy_for_url(setting, url, |name| Ok(lookup(name)))
            .expect("infallible environment lookup")
    }

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
    fn honors_host_domain_and_wildcard_bypasses() {
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
    fn lower_case_and_scheme_specific_values_win_before_fallbacks() {
        let setting = ProxySetting::FromEnvironment;
        let pairs = [
            ("https_proxy", " http://secure-lower:3128 "),
            ("HTTPS_PROXY", "http://secure-upper:3128"),
            ("http_proxy", "http://plain-lower:3128"),
            ("HTTP_PROXY", "http://ignored:3128"),
            ("all_proxy", "socks5h://fallback-lower:1080"),
            ("ALL_PROXY", "http://fallback-upper:3128"),
        ];
        assert_eq!(
            proxy_for_url(&setting, "https://remote.invalid/x", env_of(&pairs)).as_deref(),
            Some("http://secure-lower:3128")
        );
        assert_eq!(
            proxy_for_url(&setting, "http://remote.invalid/x", env_of(&pairs)).as_deref(),
            Some("http://plain-lower:3128")
        );
        assert_eq!(
            proxy_for_url(&setting, "https://remote.invalid/x", env_of(&pairs[1..])).as_deref(),
            Some("http://secure-upper:3128")
        );
        for url in ["http://remote.invalid/x", "https://remote.invalid/x"] {
            assert_eq!(
                proxy_for_url(&setting, url, env_of(&pairs[3..])).as_deref(),
                Some("socks5://fallback-lower:1080")
            );
            assert_eq!(
                proxy_for_url(&setting, url, env_of(&pairs[5..])).as_deref(),
                Some("http://fallback-upper:3128")
            );
        }
    }

    #[test]
    fn ignores_upper_http_proxy_even_in_cgi_and_for_https_targets() {
        let setting = ProxySetting::FromEnvironment;
        let pairs = [
            ("REQUEST_METHOD", "GET"),
            ("HTTP_PROXY", "http://request-header-attacker:8080"),
        ];
        for url in ["http://remote.invalid/x", "https://remote.invalid/x"] {
            assert_eq!(proxy_for_url(&setting, url, env_of(&pairs)), None);
        }
    }

    #[test]
    fn windows_http_proxy_lookup_checks_the_stored_key_spelling() {
        use std::ffi::OsString;

        let upper = [(
            OsString::from("HTTP_PROXY"),
            OsString::from("http://attacker:8080"),
        )];
        assert_eq!(exact_environment_value("http_proxy", upper).unwrap(), None);
        let mixed = [(
            OsString::from("Http_Proxy"),
            OsString::from("http://attacker:8080"),
        )];
        assert_eq!(exact_environment_value("http_proxy", mixed).unwrap(), None);
        let lower = [(
            OsString::from("http_proxy"),
            OsString::from("http://configured:3128"),
        )];
        assert_eq!(
            exact_environment_value("http_proxy", lower)
                .unwrap()
                .as_deref(),
            Some("http://configured:3128")
        );
    }

    #[test]
    fn lower_case_no_proxy_takes_precedence_and_empty_values_fall_back() {
        let setting = ProxySetting::FromEnvironment;
        let env = env_of(&[
            ("ALL_PROXY", "http://proxy:3128"),
            ("no_proxy", "another.invalid"),
            ("NO_PROXY", "*"),
        ]);
        assert_eq!(
            proxy_for_url(&setting, "https://remote.invalid/x", env).as_deref(),
            Some("http://proxy:3128"),
            "a populated lower-case bypass list owns the whole setting"
        );
        let env = env_of(&[
            ("ALL_PROXY", "http://proxy:3128"),
            ("no_proxy", " \t "),
            ("NO_PROXY", "remote.invalid"),
        ]);
        assert!(proxy_for_url(&setting, "https://remote.invalid/x", env).is_none());
        let env = env_of(&[
            ("https_proxy", " "),
            ("HTTPS_PROXY", "\t"),
            ("all_proxy", "\n"),
            ("ALL_PROXY", " http://fallback:3128 "),
        ]);
        assert_eq!(
            proxy_for_url(&setting, "https://remote.invalid/x", env).as_deref(),
            Some("http://fallback:3128")
        );
    }

    fn bypasses(url: &str, no_proxy: &str) -> bool {
        let pairs = [("ALL_PROXY", "http://proxy:3128"), ("NO_PROXY", no_proxy)];
        proxy_for_url(&ProxySetting::FromEnvironment, url, env_of(&pairs)).is_none()
    }

    #[test]
    fn normalizes_dns_case_and_root_dots_without_widening_domains() {
        for (url, rule) in [
            ("https://huggingface.co./x", "huggingface.co"),
            ("https://huggingface.co/x", ".HuggingFace.CO."),
            ("https://CDN.HUGGINGFACE.CO./x", "huggingface.co."),
        ] {
            assert!(bypasses(url, rule), "{url} with {rule}");
        }
        for url in [
            "https://nothuggingface.co/x",
            "https://huggingface.co.attacker.invalid/x",
        ] {
            assert!(!bypasses(url, "huggingface.co"), "{url}");
        }
    }

    #[test]
    fn compares_ip_literals_as_addresses_never_domain_suffixes() {
        for (url, rule) in [
            ("https://127.0.0.1/x", "127.0.0.1"),
            ("https://[0:0:0:0:0:0:0:1]/x", "::1"),
            ("https://[2001:db8::1]/x", "2001:0DB8:0:0:0:0:0:1"),
            ("https://[::1]/x", "[0:0:0:0:0:0:0:1]"),
        ] {
            assert!(bypasses(url, rule), "{url} with {rule}");
        }
        for (url, rule) in [
            ("https://127.0.0.1/x", "0.1"),
            ("https://127.0.0.1/x", "0.0.1"),
            ("https://127.0.0.1/x", "127.0.0.2"),
            ("https://subdomain.127.0.0.1/x", "127.0.0.1"),
            ("https://localhost/x", "127.0.0.1"),
            ("https://[::1]/x", "127.0.0.1"),
        ] {
            assert!(!bypasses(url, rule), "{url} with {rule}");
        }
    }

    #[test]
    fn restricts_port_qualified_bypasses_to_the_destination_port() {
        for (url, rule) in [
            ("http://mirror.invalid/x", "mirror.invalid:80"),
            ("https://mirror.invalid/x", "mirror.invalid:443"),
            ("https://mirror.invalid:8443/x", ".mirror.invalid:8443"),
            ("https://mirror.invalid:8443/x", "mirror.invalid"),
            ("https://[::1]:8443/x", "[::1]:8443"),
            ("https://[::1]:8443/x", "::1"),
            ("http://127.0.0.1:8080/x", "127.0.0.1:8080"),
        ] {
            assert!(bypasses(url, rule), "{url} with {rule}");
        }
        for (url, rule) in [
            ("http://mirror.invalid/x", "mirror.invalid:443"),
            ("https://mirror.invalid:8443/x", "mirror.invalid:443"),
            ("https://[::1]:8443/x", "[::1]:443"),
            ("http://127.0.0.1:8080/x", "127.0.0.1:80"),
        ] {
            assert!(!bypasses(url, rule), "{url} with {rule}");
        }
    }

    #[test]
    fn matches_ipv4_and_ipv6_cidr_boundaries_without_local_dns() {
        for (url, rule) in [
            ("https://192.0.2.0/x", "192.0.2.0/24"),
            ("https://192.0.2.255/x", "192.0.2.0/24"),
            ("https://192.0.2.17/x", "192.0.2.17/32"),
            ("https://192.0.2.17/x", "0.0.0.0/0"),
            ("https://[2001:db8::]/x", "2001:db8::/32"),
            (
                "https://[2001:db8:ffff:ffff:ffff:ffff:ffff:ffff]/x",
                "2001:db8::/32",
            ),
            ("https://[2001:db8::1]/x", "2001:db8::1/128"),
            ("https://[::1]/x", "::/0"),
        ] {
            assert!(bypasses(url, rule), "{url} with {rule}");
        }
        for (url, rule) in [
            ("https://192.0.1.255/x", "192.0.2.0/24"),
            ("https://192.0.3.0/x", "192.0.2.0/24"),
            ("https://192.0.2.18/x", "192.0.2.17/32"),
            ("https://[2001:db9::]/x", "2001:db8::/32"),
            ("https://[2001:db8::2]/x", "2001:db8::1/128"),
            ("https://[::1]/x", "0.0.0.0/0"),
            ("https://127.0.0.1/x", "::/0"),
            ("https://localhost/x", "127.0.0.0/8"),
            ("https://model-host.invalid/x", "0.0.0.0/0,::/0"),
        ] {
            assert!(!bypasses(url, rule), "{url} with {rule}");
        }
    }

    #[test]
    fn malformed_bypass_entries_cannot_broaden_a_direct_route() {
        for rule in [
            "mirror.invalid:",
            "mirror.invalid:bad",
            "mirror.invalid:+443",
            "mirror.invalid:65536",
            "..mirror.invalid",
            "mirror.invalid..",
            "https://mirror.invalid",
            "mirror.invalid/path",
            "mirror.invalid@attacker.invalid",
            "*.invalid",
            "*:443",
            "mirror .invalid",
            ",, ,",
            ".",
        ] {
            assert!(!bypasses("https://mirror.invalid/x", rule), "{rule:?}");
        }
        assert!(!bypasses("https://mirror.invalid/x", "mirror.\r\ninvalid"));
        for rule in [
            "127.0.0.1/33",
            "127.0.0.1/-1",
            "127.0.0.1/+0",
            "0.0.0.0/",
            "0.0.0.0/0:443",
        ] {
            assert!(!bypasses("https://127.0.0.1/x", rule), "{rule}");
        }
        for rule in [
            "[::1]:bad",
            "[::1]suffix",
            "[::1",
            "[::1]:",
            "::/129",
            "::/-1",
            "::/*",
            "[::1]/*",
            "[::1*]",
        ] {
            assert!(!bypasses("https://[::1]/x", rule), "{rule}");
        }
    }

    #[test]
    fn malformed_or_unsupported_targets_are_left_to_transport_validation() {
        let env = env_of(&[("ALL_PROXY", "http://proxy:3128")]);
        for url in [
            "https://user:pw@huggingface.co/x",
            "https:///x",
            "https://mirror.invalid:bad/x",
            "https://mirror.invalid:\r\n443/x",
            "https://[::1/x",
            "https://[::1]suffix/x",
            "file:///model.onnx",
        ] {
            assert!(proxy_for_url(&ProxySetting::FromEnvironment, url, &env).is_none());
        }
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
    fn normalizes_known_proxy_schemes_without_changing_credentials() {
        for (proxy, expected) in [
            ("HTTP://proxy:3128", "http://proxy:3128"),
            (" HTTPS://proxy:3128 ", "https://proxy:3128"),
            ("SOCKS5://proxy:1080", "socks5://proxy:1080"),
            (
                "SoCkS5H://User:P%40ss@Proxy:1080",
                "socks5://User:P%40ss@Proxy:1080",
            ),
        ] {
            let pairs = [("https_proxy", proxy)];
            let env = env_of(&pairs);
            assert_eq!(
                proxy_for_url(
                    &ProxySetting::FromEnvironment,
                    "HtTpS://remote.invalid/x",
                    env
                )
                .as_deref(),
                Some(expected)
            );
            assert_eq!(
                proxy_for_url(
                    &ProxySetting::Url(proxy.to_owned()),
                    "https://remote.invalid/x",
                    env_of(&[])
                )
                .as_deref(),
                Some(expected)
            );
        }
    }

    #[test]
    fn malformed_selected_proxy_does_not_fall_back_to_direct_or_another_proxy() {
        for proxy in [
            "ftp://user:secret@proxy",
            "http://proxy:bad",
            "socks6://proxy",
            "proxy:3128",
        ] {
            let pairs = [
                ("https_proxy", proxy),
                ("ALL_PROXY", "http://fallback:3128"),
            ];
            let env = env_of(&pairs);
            assert_eq!(
                proxy_for_url(
                    &ProxySetting::FromEnvironment,
                    "https://remote.invalid/x",
                    env
                )
                .as_deref(),
                Some(proxy)
            );
        }
        assert_eq!(
            proxy_for_url(
                &ProxySetting::Url(String::new()),
                "https://remote.invalid/x",
                env_of(&[])
            ),
            Some(String::new()),
            "an explicitly empty URL is invalid, not permission to connect directly"
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
        let never_read_environment = |_name: &str| -> Option<String> {
            panic!("explicit proxy settings must not consult the environment")
        };
        assert!(
            proxy_for_url(
                &ProxySetting::Disabled,
                "https://huggingface.co/x",
                never_read_environment
            )
            .is_none()
        );
        assert_eq!(
            proxy_for_url(
                &explicit,
                "https://huggingface.co/x",
                never_read_environment
            )
            .as_deref(),
            Some("http://chosen:3128")
        );
    }

    #[cfg(any(unix, windows))]
    fn non_unicode_environment_value() -> std::ffi::OsString {
        const SECRET: &str = "proxy-credential-canary";
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;

            let mut bytes = SECRET.as_bytes().to_vec();
            bytes.push(0xff);
            std::ffi::OsString::from_vec(bytes)
        }
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStringExt;

            let mut units: Vec<_> = SECRET.encode_utf16().collect();
            units.push(0xd800);
            std::ffi::OsString::from_wide(&units)
        }
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn non_unicode_environment_errors_never_retain_credentials() {
        for name in ["http_proxy", "HTTPS_PROXY", "ALL_PROXY", "no_proxy"] {
            let error = environment_value(name, Some(non_unicode_environment_value()))
                .expect_err("a present non-Unicode value is not an absent setting");
            match &error {
                SearchError::InvalidConfig { field, value, .. } => {
                    assert_eq!(field, &format!("proxy.{name}"));
                    assert_eq!(value, "<non-Unicode>");
                }
                other => panic!("unexpected error: {other:?}"),
            }
            let rendered = format!("{error:?} {error}");
            assert!(!rendered.contains("proxy-credential-canary"));
        }
        assert!(environment_value("http_proxy", None).unwrap().is_none());
        assert_eq!(
            environment_value("http_proxy", Some(std::ffi::OsString::new()))
                .unwrap()
                .as_deref(),
            Some("")
        );
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn non_unicode_selected_variables_stop_before_lower_priority_lookups() {
        for (url, selected) in [
            ("http://remote.invalid/model", "http_proxy"),
            ("https://remote.invalid/model", "https_proxy"),
            ("https://remote.invalid/model", "HTTPS_PROXY"),
            ("http://remote.invalid/model", "all_proxy"),
            ("https://remote.invalid/model", "ALL_PROXY"),
            ("http://remote.invalid/model", "no_proxy"),
            ("https://remote.invalid/model", "NO_PROXY"),
        ] {
            let read = std::cell::RefCell::new(Vec::new());
            let error = super::proxy_for_url(&ProxySetting::FromEnvironment, url, |name| {
                read.borrow_mut().push(name.to_owned());
                if name == selected {
                    environment_value(name, Some(non_unicode_environment_value()))
                } else {
                    Ok(None)
                }
            })
            .expect_err("the selected malformed setting must terminate resolution");
            assert!(matches!(error, SearchError::InvalidConfig { .. }));
            assert_eq!(read.borrow().last().map(String::as_str), Some(selected));
        }
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn unused_malformed_environment_values_do_not_change_selected_policy() {
        let url = "https://remote.invalid/model";
        let bypass = super::proxy_for_url(&ProxySetting::FromEnvironment, url, |name| {
            if name == "no_proxy" {
                Ok(Some("remote.invalid".to_owned()))
            } else {
                environment_value(name, Some(non_unicode_environment_value()))
            }
        })
        .unwrap();
        assert!(
            bypass.is_none(),
            "an intentional bypass never reads a proxy"
        );

        let selected =
            super::proxy_for_url(&ProxySetting::FromEnvironment, url, |name| match name {
                "no_proxy" | "NO_PROXY" => Ok(None),
                "https_proxy" => Ok(Some("http://chosen:3128".to_owned())),
                _ => environment_value(name, Some(non_unicode_environment_value())),
            })
            .unwrap();
        assert_eq!(selected.as_deref(), Some("http://chosen:3128"));

        for setting in [
            ProxySetting::Disabled,
            ProxySetting::Url("http://chosen:3128".to_owned()),
        ] {
            let route = super::proxy_for_url(&setting, url, |name| {
                environment_value(name, Some(non_unicode_environment_value()))
            })
            .unwrap();
            match setting {
                ProxySetting::Disabled => assert!(route.is_none()),
                ProxySetting::Url(_) => assert_eq!(route.as_deref(), Some("http://chosen:3128")),
                ProxySetting::FromEnvironment => unreachable!(),
            }
        }
        let ignored_upper_http = super::proxy_for_url(
            &ProxySetting::FromEnvironment,
            "http://remote.invalid/model",
            |name| {
                if name == "HTTP_PROXY" {
                    environment_value(name, Some(non_unicode_environment_value()))
                } else {
                    Ok(None)
                }
            },
        )
        .unwrap();
        assert!(ignored_upper_http.is_none());
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn exact_http_proxy_lookup_preserves_invalid_selected_values() {
        let upper = [(
            std::ffi::OsString::from("HTTP_PROXY"),
            non_unicode_environment_value(),
        )];
        assert!(
            exact_environment_value("http_proxy", upper)
                .unwrap()
                .is_none()
        );
        let lower = [(
            std::ffi::OsString::from("http_proxy"),
            non_unicode_environment_value(),
        )];
        assert!(matches!(
            exact_environment_value("http_proxy", lower),
            Err(SearchError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn debug_redacts_the_proxy_url() {
        let setting = ProxySetting::Url("http://user:hunter2@proxy.corp:3128".to_owned());
        let rendered = format!("{setting:?}");
        assert_eq!(rendered, "Url(<redacted>)");
    }
}
