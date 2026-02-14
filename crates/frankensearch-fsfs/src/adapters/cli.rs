use std::path::PathBuf;
use std::str::FromStr;

use frankensearch_core::{SearchError, SearchResult};

use crate::config::{CliOverrides, PressureProfile, TuiTheme};

// ─── Exit Codes ──────────────────────────────────────────────────────────────

/// Standardized exit codes for the fsfs CLI.
pub mod exit_code {
    /// Success.
    pub const OK: i32 = 0;
    /// Runtime error (search, index, config error).
    pub const RUNTIME_ERROR: i32 = 1;
    /// Usage error (invalid args, unknown command).
    pub const USAGE_ERROR: i32 = 2;
    /// Interrupted by signal (SIGINT).
    pub const INTERRUPTED: i32 = 130;
}

// ─── Output Format ───────────────────────────────────────────────────────────

/// Output format for CLI results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// Human-readable table (default for TTY).
    #[default]
    Table,
    /// Machine-readable JSON.
    Json,
    /// CSV for spreadsheet/pipeline use.
    Csv,
    /// Newline-delimited JSON (streaming).
    Jsonl,
    /// TOON (Token-Oriented Object Notation) machine output.
    Toon,
}

impl FromStr for OutputFormat {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "table" => Ok(Self::Table),
            "json" => Ok(Self::Json),
            "csv" => Ok(Self::Csv),
            "jsonl" => Ok(Self::Jsonl),
            "toon" => Ok(Self::Toon),
            _ => Err(()),
        }
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Table => write!(f, "table"),
            Self::Json => write!(f, "json"),
            Self::Csv => write!(f, "csv"),
            Self::Jsonl => write!(f, "jsonl"),
            Self::Toon => write!(f, "toon"),
        }
    }
}

// ─── CLI Command ─────────────────────────────────────────────────────────────

/// Top-level fsfs command entrypoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CliCommand {
    /// Progressive search with Phase 0 → Phase 1 rendering.
    Search,
    /// Index/re-index corpus.
    Index,
    /// Show index health and status.
    #[default]
    Status,
    /// Show score decomposition for a document+query pair.
    Explain,
    /// Manage configuration (get/set/list/reset).
    Config,
    /// Download embedding models.
    Download,
    /// Run self-diagnostics.
    Doctor,
    /// Launch the deluxe TUI interface.
    Tui,
    /// Show version and build info.
    Version,
}

impl CliCommand {
    /// All valid command names for help text.
    pub const ALL_NAMES: &'static [&'static str] = &[
        "search", "index", "status", "explain", "config", "download", "doctor", "tui", "version",
    ];
}

/// Whether the command token was explicitly provided or defaulted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CommandSource {
    /// No command token provided; parser used the default command.
    #[default]
    ImplicitDefault,
    /// User explicitly provided a command token.
    Explicit,
}

/// Parsed CLI input including command and high-priority overrides.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CliInput {
    /// The selected command.
    pub command: CliCommand,
    /// Whether the command token was explicitly provided by the user.
    pub command_source: CommandSource,
    /// Configuration overrides from flags.
    pub overrides: CliOverrides,
    /// Output format (default: table).
    pub format: OutputFormat,
    /// Search query text (for search command).
    pub query: Option<String>,
    /// Filter expression (for search command).
    pub filter: Option<String>,
    /// Whether `--stream` was requested.
    pub stream: bool,
    /// Whether `--watch` was requested (for index command).
    pub watch: bool,
    /// Whether `--full` was requested (for index command).
    pub full_reindex: bool,
    /// Target model name (for download command).
    pub model_name: Option<String>,
    /// Config subcommand (get/set/list/reset).
    pub config_action: Option<ConfigAction>,
}

/// Config subcommand actions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigAction {
    /// Get a config value.
    Get { key: String },
    /// Set a config value.
    Set { key: String, value: String },
    /// List all config values with sources.
    List,
    /// Reset config to defaults.
    Reset,
}

/// Detect interface mode based on command and terminal state.
///
/// - No subcommand + TTY → launch TUI
/// - No subcommand + pipe → show help (return None)
/// - Explicit `tui` subcommand → launch TUI
/// - Any other subcommand → CLI mode
#[must_use]
pub const fn detect_auto_mode(
    command: CliCommand,
    is_tty: bool,
    command_source: CommandSource,
) -> Option<CliCommand> {
    match (command, command_source, is_tty) {
        (CliCommand::Status, CommandSource::ImplicitDefault, true) => Some(CliCommand::Tui),
        (CliCommand::Status, CommandSource::ImplicitDefault, false) => {
            None // Pipe without subcommand — show help.
        }
        _ => Some(command),
    }
}

/// Parse fsfs CLI args.
///
/// The first non-flag token selects the command. Supported flags implement
/// the precedence contract from `bd-2hz.13`.
///
/// # Errors
///
/// Returns `SearchError::InvalidConfig` if the command/flags are malformed.
#[allow(clippy::too_many_lines)]
pub fn parse_cli_args<I, S>(args: I) -> SearchResult<CliInput>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let tokens: Vec<String> = args.into_iter().map(Into::into).collect();
    let (command, mut idx, command_source) = extract_command(&tokens)?;
    let mut input = CliInput {
        command,
        command_source,
        ..CliInput::default()
    };

    // For search command, capture an explicit query token before flags.
    if command == CliCommand::Search && idx < tokens.len() {
        if tokens[idx] == "--" {
            let query = tokens
                .get(idx + 1)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "cli.search_query".into(),
                    value: "--".into(),
                    reason: "missing query after '--'".into(),
                })?;
            input.query = Some(query.clone());
            idx += 2;
        } else if !is_known_cli_flag(tokens[idx].as_str()) {
            input.query = Some(tokens[idx].clone());
            idx += 1;
        }
    }

    // For download command, the next non-flag token is the model name.
    if command == CliCommand::Download && idx < tokens.len() && !tokens[idx].starts_with('-') {
        input.model_name = Some(tokens[idx].clone());
        idx += 1;
    }

    // For config command, parse subcommand.
    if command == CliCommand::Config && idx < tokens.len() && !tokens[idx].starts_with('-') {
        let action = parse_config_action(&tokens, &mut idx)?;
        input.config_action = Some(action);
    }

    while idx < tokens.len() {
        let flag = tokens[idx].as_str();
        match flag {
            "--roots" => {
                let value = expect_value(&tokens, idx, "--roots")?;
                input.overrides.roots = Some(split_csv(value)?);
                idx += 2;
            }
            "--exclude" => {
                let value = expect_value(&tokens, idx, "--exclude")?;
                input.overrides.exclude_patterns = Some(split_csv(value)?);
                idx += 2;
            }
            "--limit" | "-l" => {
                let value = expect_value(&tokens, idx, "--limit")?;
                input.overrides.limit = Some(parse_usize(value, "search.default_limit")?);
                idx += 2;
            }
            "--format" | "-f" => {
                let value = expect_value(&tokens, idx, "--format")?;
                input.format =
                    OutputFormat::from_str(value).map_err(|()| SearchError::InvalidConfig {
                        field: "cli.format".into(),
                        value: value.into(),
                        reason: "expected table|json|csv|jsonl|toon".into(),
                    })?;
                idx += 2;
            }
            "--fast-only" => {
                input.overrides.fast_only = Some(true);
                idx += 1;
            }
            "--no-fast-only" => {
                input.overrides.fast_only = Some(false);
                idx += 1;
            }
            "--watch-mode" => {
                input.overrides.allow_background_indexing = Some(true);
                idx += 1;
            }
            "--no-watch-mode" => {
                input.overrides.allow_background_indexing = Some(false);
                idx += 1;
            }
            "--explain" | "-e" => {
                input.overrides.explain = Some(true);
                idx += 1;
            }
            "--stream" => {
                input.stream = true;
                idx += 1;
            }
            "--watch" => {
                input.watch = true;
                idx += 1;
            }
            "--full" => {
                input.full_reindex = true;
                idx += 1;
            }
            "--filter" => {
                let value = expect_value(&tokens, idx, "--filter")?;
                input.filter = Some(value.to_string());
                idx += 2;
            }
            "--profile" => {
                let value = expect_value(&tokens, idx, "--profile")?;
                input.overrides.profile = Some(PressureProfile::from_str(value).map_err(|()| {
                    SearchError::InvalidConfig {
                        field: "pressure.profile".into(),
                        value: value.into(),
                        reason: "expected strict|performance|degraded".into(),
                    }
                })?);
                idx += 2;
            }
            "--theme" => {
                let value = expect_value(&tokens, idx, "--theme")?;
                input.overrides.theme =
                    Some(
                        TuiTheme::from_str(value).map_err(|()| SearchError::InvalidConfig {
                            field: "tui.theme".into(),
                            value: value.into(),
                            reason: "expected auto|light|dark".into(),
                        })?,
                    );
                idx += 2;
            }
            "--config" => {
                let value = expect_value(&tokens, idx, "--config")?;
                input.overrides.config_path = Some(PathBuf::from(value));
                idx += 2;
            }
            _ => {
                return Err(SearchError::InvalidConfig {
                    field: "cli.flag".into(),
                    value: flag.into(),
                    reason: format!(
                        "unknown flag; valid commands: {}",
                        CliCommand::ALL_NAMES.join("|")
                    ),
                });
            }
        }
    }

    normalize_stream_settings(&mut input)?;

    Ok(input)
}

fn normalize_stream_settings(input: &mut CliInput) -> SearchResult<()> {
    if !input.stream {
        return Ok(());
    }

    // Streaming defaults to NDJSON unless explicitly set to TOON.
    if input.format == OutputFormat::Table {
        input.format = OutputFormat::Jsonl;
    }

    if input.command != CliCommand::Search {
        return Err(SearchError::InvalidConfig {
            field: "cli.stream.command".into(),
            value: format!("{:?}", input.command).to_lowercase(),
            reason: "stream mode is only supported for the search command".into(),
        });
    }

    if !matches!(input.format, OutputFormat::Jsonl | OutputFormat::Toon) {
        return Err(SearchError::InvalidConfig {
            field: "cli.stream.format".into(),
            value: input.format.to_string(),
            reason: "stream mode requires --format jsonl or --format toon".into(),
        });
    }

    Ok(())
}

fn extract_command(tokens: &[String]) -> SearchResult<(CliCommand, usize, CommandSource)> {
    if let Some(token) = tokens.first()
        && !token.starts_with('-')
    {
        return Ok((parse_command(token)?, 1, CommandSource::Explicit));
    }
    Ok((CliCommand::default(), 0, CommandSource::ImplicitDefault))
}

fn parse_command(token: &str) -> SearchResult<CliCommand> {
    match token {
        "search" | "s" => Ok(CliCommand::Search),
        "index" | "idx" => Ok(CliCommand::Index),
        "status" | "st" => Ok(CliCommand::Status),
        "explain" | "ex" => Ok(CliCommand::Explain),
        "config" | "cfg" => Ok(CliCommand::Config),
        "download" | "dl" => Ok(CliCommand::Download),
        "doctor" | "doc" => Ok(CliCommand::Doctor),
        "tui" => Ok(CliCommand::Tui),
        "version" | "ver" => Ok(CliCommand::Version),
        _ => Err(SearchError::InvalidConfig {
            field: "cli.command".into(),
            value: token.into(),
            reason: format!(
                "unknown command; expected: {}",
                CliCommand::ALL_NAMES.join("|")
            ),
        }),
    }
}

fn parse_config_action(tokens: &[String], idx: &mut usize) -> SearchResult<ConfigAction> {
    let action = tokens[*idx].as_str();
    *idx += 1;
    match action {
        "list" | "ls" => Ok(ConfigAction::List),
        "reset" => Ok(ConfigAction::Reset),
        "get" => {
            let key = tokens.get(*idx).ok_or_else(|| SearchError::InvalidConfig {
                field: "config.get".into(),
                value: String::new(),
                reason: "missing key for config get".into(),
            })?;
            *idx += 1;
            Ok(ConfigAction::Get { key: key.clone() })
        }
        "set" => {
            let key = tokens.get(*idx).ok_or_else(|| SearchError::InvalidConfig {
                field: "config.set".into(),
                value: String::new(),
                reason: "missing key for config set".into(),
            })?;
            *idx += 1;
            let value = tokens.get(*idx).ok_or_else(|| SearchError::InvalidConfig {
                field: "config.set".into(),
                value: key.clone(),
                reason: "missing value for config set".into(),
            })?;
            *idx += 1;
            Ok(ConfigAction::Set {
                key: key.clone(),
                value: value.clone(),
            })
        }
        _ => Err(SearchError::InvalidConfig {
            field: "config.action".into(),
            value: action.into(),
            reason: "expected get|set|list|reset".into(),
        }),
    }
}

fn expect_value<'a>(tokens: &'a [String], idx: usize, flag: &str) -> SearchResult<&'a str> {
    tokens
        .get(idx + 1)
        .map(String::as_str)
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "cli.flag".into(),
            value: flag.into(),
            reason: "missing value".into(),
        })
}

fn split_csv(value: &str) -> SearchResult<Vec<String>> {
    let parts: Vec<String> = value
        .split(',')
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .map(str::to_string)
        .collect();

    if parts.is_empty() {
        return Err(SearchError::InvalidConfig {
            field: "cli.csv".into(),
            value: value.into(),
            reason: "expected at least one comma-separated value".into(),
        });
    }

    Ok(parts)
}

fn parse_usize(value: &str, field: &str) -> SearchResult<usize> {
    value
        .parse::<usize>()
        .map_err(|_| SearchError::InvalidConfig {
            field: field.into(),
            value: value.into(),
            reason: "expected unsigned integer".into(),
        })
}

#[must_use]
fn is_known_cli_flag(token: &str) -> bool {
    matches!(
        token,
        "--roots"
            | "--exclude"
            | "--limit"
            | "-l"
            | "--format"
            | "-f"
            | "--fast-only"
            | "--no-fast-only"
            | "--watch-mode"
            | "--no-watch-mode"
            | "--explain"
            | "-e"
            | "--stream"
            | "--watch"
            | "--full"
            | "--filter"
            | "--profile"
            | "--theme"
            | "--config"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_command_and_overrides() {
        let input = parse_cli_args([
            "search",
            "hello world",
            "--roots",
            "/repo,/notes",
            "--exclude",
            "target,node_modules",
            "--limit",
            "25",
            "--fast-only",
            "--watch-mode",
            "--explain",
            "--profile",
            "degraded",
            "--theme",
            "light",
        ])
        .expect("parse");

        assert_eq!(input.command, CliCommand::Search);
        assert_eq!(input.query.as_deref(), Some("hello world"));
        assert_eq!(input.overrides.limit, Some(25));
        assert_eq!(input.overrides.fast_only, Some(true));
        assert_eq!(input.overrides.allow_background_indexing, Some(true));
        assert_eq!(input.overrides.explain, Some(true));
        assert_eq!(
            input.overrides.roots.expect("roots"),
            vec!["/repo".to_string(), "/notes".to_string()]
        );
    }

    #[test]
    fn unknown_flag_returns_error() {
        let err = parse_cli_args(["status", "--wat"]).expect_err("must fail");
        let msg = err.to_string();
        assert!(msg.contains("unknown flag"));
    }

    #[test]
    fn default_command_is_status() {
        let input = parse_cli_args(["--limit", "10"]).expect("parse");
        assert_eq!(input.command, CliCommand::Status);
        assert_eq!(input.command_source, CommandSource::ImplicitDefault);
    }

    #[test]
    fn parse_new_commands() {
        assert_eq!(
            parse_cli_args(["download"]).unwrap().command,
            CliCommand::Download
        );
        assert_eq!(
            parse_cli_args(["doctor"]).unwrap().command,
            CliCommand::Doctor
        );
        assert_eq!(parse_cli_args(["tui"]).unwrap().command, CliCommand::Tui);
        assert_eq!(
            parse_cli_args(["version"]).unwrap().command,
            CliCommand::Version
        );
    }

    #[test]
    fn parse_command_aliases() {
        assert_eq!(parse_cli_args(["s"]).unwrap().command, CliCommand::Search);
        assert_eq!(parse_cli_args(["idx"]).unwrap().command, CliCommand::Index);
        assert_eq!(parse_cli_args(["st"]).unwrap().command, CliCommand::Status);
        assert_eq!(parse_cli_args(["ex"]).unwrap().command, CliCommand::Explain);
        assert_eq!(parse_cli_args(["cfg"]).unwrap().command, CliCommand::Config);
        assert_eq!(
            parse_cli_args(["dl"]).unwrap().command,
            CliCommand::Download
        );
        assert_eq!(parse_cli_args(["doc"]).unwrap().command, CliCommand::Doctor);
        assert_eq!(
            parse_cli_args(["ver"]).unwrap().command,
            CliCommand::Version
        );
    }

    #[test]
    fn parse_output_format() {
        let input = parse_cli_args(["status", "--format", "json"]).unwrap();
        assert_eq!(input.format, OutputFormat::Json);

        let toon = parse_cli_args(["status", "--format", "toon"]).unwrap();
        assert_eq!(toon.format, OutputFormat::Toon);
    }

    #[test]
    fn parse_short_flags() {
        let input = parse_cli_args(["search", "q", "-l", "5", "-f", "csv", "-e"]).unwrap();
        assert_eq!(input.overrides.limit, Some(5));
        assert_eq!(input.format, OutputFormat::Csv);
        assert_eq!(input.overrides.explain, Some(true));
    }

    #[test]
    fn parse_stream_flag() {
        let input = parse_cli_args(["search", "test", "--stream"]).unwrap();
        assert!(input.stream);
        assert_eq!(input.format, OutputFormat::Jsonl);
    }

    #[test]
    fn parse_stream_accepts_toon_format() {
        let input = parse_cli_args(["search", "test", "--stream", "--format", "toon"]).unwrap();
        assert!(input.stream);
        assert_eq!(input.format, OutputFormat::Toon);
    }

    #[test]
    fn parse_stream_rejects_non_search_command() {
        let err = parse_cli_args(["status", "--stream"]).expect_err("must fail");
        assert!(
            err.to_string()
                .contains("stream mode is only supported for the search command")
        );
    }

    #[test]
    fn parse_stream_rejects_non_stream_format() {
        let err = parse_cli_args(["search", "test", "--stream", "--format", "json"])
            .expect_err("must fail");
        assert!(
            err.to_string()
                .contains("stream mode requires --format jsonl or --format toon")
        );
    }

    #[test]
    fn parse_watch_mode_override_flags() {
        let input = parse_cli_args(["status", "--no-watch-mode"]).unwrap();
        assert_eq!(input.overrides.allow_background_indexing, Some(false));
    }

    #[test]
    fn parse_index_flags() {
        let input = parse_cli_args(["index", "--full", "--watch"]).unwrap();
        assert_eq!(input.command, CliCommand::Index);
        assert!(input.full_reindex);
        assert!(input.watch);
    }

    #[test]
    fn parse_download_with_model() {
        let input = parse_cli_args(["download", "potion-multilingual-128M"]).unwrap();
        assert_eq!(input.command, CliCommand::Download);
        assert_eq!(
            input.model_name.as_deref(),
            Some("potion-multilingual-128M")
        );
    }

    #[test]
    fn parse_download_without_model() {
        let input = parse_cli_args(["download"]).unwrap();
        assert_eq!(input.command, CliCommand::Download);
        assert!(input.model_name.is_none());
    }

    #[test]
    fn parse_filter_flag() {
        let input = parse_cli_args(["search", "query", "--filter", "type:rs"]).unwrap();
        assert_eq!(input.filter.as_deref(), Some("type:rs"));
    }

    #[test]
    fn parse_search_query_starting_with_dash() {
        let input = parse_cli_args(["search", "-secret-token"]).unwrap();
        assert_eq!(input.command, CliCommand::Search);
        assert_eq!(input.query.as_deref(), Some("-secret-token"));
    }

    #[test]
    fn parse_search_query_after_double_dash() {
        let input = parse_cli_args(["search", "--", "--limit"]).unwrap();
        assert_eq!(input.command, CliCommand::Search);
        assert_eq!(input.query.as_deref(), Some("--limit"));
    }

    #[test]
    fn parse_search_double_dash_requires_query() {
        let err = parse_cli_args(["search", "--"]).expect_err("must fail");
        assert!(err.to_string().contains("missing query after '--'"));
    }

    #[test]
    fn parse_config_list() {
        let input = parse_cli_args(["config", "list"]).unwrap();
        assert_eq!(input.command, CliCommand::Config);
        assert_eq!(input.config_action, Some(ConfigAction::List));
    }

    #[test]
    fn parse_config_get() {
        let input = parse_cli_args(["config", "get", "search.default_limit"]).unwrap();
        assert_eq!(
            input.config_action,
            Some(ConfigAction::Get {
                key: "search.default_limit".to_string()
            })
        );
    }

    #[test]
    fn parse_config_set() {
        let input = parse_cli_args(["config", "set", "search.default_limit", "50"]).unwrap();
        assert_eq!(
            input.config_action,
            Some(ConfigAction::Set {
                key: "search.default_limit".to_string(),
                value: "50".to_string()
            })
        );
    }

    #[test]
    fn parse_config_reset() {
        let input = parse_cli_args(["config", "reset"]).unwrap();
        assert_eq!(input.config_action, Some(ConfigAction::Reset));
    }

    #[test]
    fn parse_config_alias() {
        let input = parse_cli_args(["cfg", "ls"]).unwrap();
        assert_eq!(input.command, CliCommand::Config);
        assert_eq!(input.config_action, Some(ConfigAction::List));
    }

    #[test]
    fn invalid_format_returns_error() {
        let err = parse_cli_args(["status", "--format", "xml"]).expect_err("must fail");
        let msg = err.to_string();
        assert!(msg.contains("expected table|json|csv|jsonl|toon"));
    }

    #[test]
    fn invalid_config_action_returns_error() {
        let err = parse_cli_args(["config", "drop"]).expect_err("must fail");
        let msg = err.to_string();
        assert!(msg.contains("expected get|set|list|reset"));
    }

    #[test]
    fn auto_mode_tty_no_command_launches_tui() {
        assert_eq!(
            detect_auto_mode(CliCommand::Status, true, CommandSource::ImplicitDefault),
            Some(CliCommand::Tui)
        );
    }

    #[test]
    fn auto_mode_pipe_no_command_returns_none() {
        assert_eq!(
            detect_auto_mode(CliCommand::Status, false, CommandSource::ImplicitDefault),
            None
        );
    }

    #[test]
    fn auto_mode_explicit_status_stays_cli() {
        assert_eq!(
            detect_auto_mode(CliCommand::Status, true, CommandSource::Explicit),
            Some(CliCommand::Status)
        );
    }

    #[test]
    fn auto_mode_explicit_command_always_works() {
        assert_eq!(
            detect_auto_mode(CliCommand::Search, false, CommandSource::Explicit),
            Some(CliCommand::Search)
        );
        assert_eq!(
            detect_auto_mode(CliCommand::Tui, false, CommandSource::Explicit),
            Some(CliCommand::Tui)
        );
    }

    #[test]
    fn output_format_display() {
        assert_eq!(OutputFormat::Table.to_string(), "table");
        assert_eq!(OutputFormat::Json.to_string(), "json");
        assert_eq!(OutputFormat::Csv.to_string(), "csv");
        assert_eq!(OutputFormat::Jsonl.to_string(), "jsonl");
        assert_eq!(OutputFormat::Toon.to_string(), "toon");
    }

    #[test]
    fn output_format_roundtrip() {
        for fmt_str in ["table", "json", "csv", "jsonl", "toon"] {
            let fmt = OutputFormat::from_str(fmt_str).unwrap();
            assert_eq!(fmt.to_string(), fmt_str);
        }
    }

    #[test]
    fn all_command_names_parseable() {
        for name in CliCommand::ALL_NAMES {
            let _input = parse_cli_args([*name]).expect(name);
        }
    }

    #[test]
    fn exit_codes_standard() {
        assert_eq!(exit_code::OK, 0);
        assert_eq!(exit_code::RUNTIME_ERROR, 1);
        assert_eq!(exit_code::USAGE_ERROR, 2);
        assert_eq!(exit_code::INTERRUPTED, 130);
    }

    #[test]
    fn empty_args_defaults() {
        let input = parse_cli_args::<[&str; 0], _>([]).unwrap();
        assert_eq!(input.command, CliCommand::Status);
        assert_eq!(input.command_source, CommandSource::ImplicitDefault);
        assert_eq!(input.format, OutputFormat::Table);
        assert!(input.query.is_none());
        assert!(!input.stream);
    }
}
