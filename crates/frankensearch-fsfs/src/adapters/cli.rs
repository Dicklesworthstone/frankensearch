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
    /// Required model unavailable (no models cached and download blocked/failed).
    ///
    /// Uses sysexits.h `EX_CONFIG` (78) to indicate a configuration/resource issue
    /// that the user can resolve by downloading models or setting `FRANKENSEARCH_MODEL_DIR`.
    pub const MODEL_UNAVAILABLE: i32 = 78;
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

/// Supported shell targets for completion script generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionShell {
    Bash,
    Zsh,
    Fish,
    PowerShell,
}

impl FromStr for CompletionShell {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "bash" => Ok(Self::Bash),
            "zsh" => Ok(Self::Zsh),
            "fish" => Ok(Self::Fish),
            "powershell" | "pwsh" => Ok(Self::PowerShell),
            _ => Err(()),
        }
    }
}

impl std::fmt::Display for CompletionShell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bash => write!(f, "bash"),
            Self::Zsh => write!(f, "zsh"),
            Self::Fish => write!(f, "fish"),
            Self::PowerShell => write!(f, "powershell"),
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
    /// Watch mode alias (`index --watch`).
    Watch,
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
    /// Self-update the fsfs binary.
    Update,
    /// Generate shell completion scripts.
    Completions,
    /// Remove local fsfs installation artifacts.
    Uninstall,
    /// Show command usage and examples.
    Help,
    /// Launch the deluxe TUI interface.
    Tui,
    /// Show version and build info.
    Version,
    /// Run long-lived query server mode over stdin/stdout.
    Serve,
}

impl CliCommand {
    /// All valid command names for help text.
    pub const ALL_NAMES: &'static [&'static str] = &[
        "search",
        "serve",
        "index",
        "watch",
        "status",
        "explain",
        "config",
        "download-models",
        "doctor",
        "update",
        "completions",
        "uninstall",
        "help",
        "tui",
        "version",
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
#[allow(clippy::struct_excessive_bools)]
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
    /// Whether output format was explicitly set by `--format`/`-f`.
    pub format_explicit: bool,
    /// Search query text (for search command).
    pub query: Option<String>,
    /// Optional target path for index/watch.
    pub target_path: Option<PathBuf>,
    /// Optional explicit index directory override for index/status commands.
    pub index_dir: Option<PathBuf>,
    /// Explain result identifier.
    pub result_id: Option<String>,
    /// Filter expression (for search command).
    pub filter: Option<String>,
    /// Route search requests through a long-lived daemon transport.
    pub daemon: bool,
    /// Optional daemon socket override (search/serve commands).
    pub daemon_socket: Option<PathBuf>,
    /// Whether `--stream` was requested.
    pub stream: bool,
    /// Whether `--watch` was requested (for index command).
    pub watch: bool,
    /// Whether `--full` was requested (for index command).
    pub full_reindex: bool,
    /// Target model name (for download command).
    pub model_name: Option<String>,
    /// Whether `--list` was requested (for download command).
    pub download_list: bool,
    /// Whether `--verify` was requested (for download command).
    pub download_verify: bool,
    /// Optional output/cache root override (for download command).
    pub download_output_dir: Option<PathBuf>,
    /// Config subcommand (get/set/list/reset).
    pub config_action: Option<ConfigAction>,
    /// Shell target for completions command.
    pub completion_shell: Option<CompletionShell>,
    /// Whether update should only check for availability.
    pub update_check_only: bool,
    /// Whether update should rollback to a previous version.
    pub update_rollback: bool,
    /// Specific version to rollback to (e.g. "v0.1.0").
    pub update_rollback_version: Option<String>,
    /// Whether uninstall should purge cache/config artifacts.
    pub uninstall_purge: bool,
    /// Whether uninstall execution is explicitly confirmed.
    pub uninstall_yes: bool,
    /// Whether uninstall should only print planned removals.
    pub uninstall_dry_run: bool,
    /// Verbose diagnostics requested.
    pub verbose: bool,
    /// Quiet output requested.
    pub quiet: bool,
    /// Disable ANSI colors in terminal output.
    pub no_color: bool,
}

/// Config subcommand actions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigAction {
    /// Show merged config values with source annotations.
    Show,
    /// Get a config value.
    Get { key: String },
    /// Set a config value.
    Set { key: String, value: String },
    /// Initialize a starter config file.
    Init,
    /// Validate configuration resolution and report warnings.
    Validate,
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

/// Resolve the effective output format using terminal awareness.
///
/// Contract:
/// - explicit `--format` always wins
/// - implicit default on TTY stays `table`
/// - implicit default on non-TTY switches to `jsonl`
#[must_use]
pub const fn resolve_output_format(
    format: OutputFormat,
    format_explicit: bool,
    is_tty: bool,
) -> OutputFormat {
    if !format_explicit && matches!(format, OutputFormat::Table) && !is_tty {
        OutputFormat::Jsonl
    } else {
        format
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
    let mut daemon_preference: Option<bool> = None;

    if command == CliCommand::Help {
        return Ok(input);
    }

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

    if matches!(command, CliCommand::Index | CliCommand::Watch)
        && idx < tokens.len()
        && !tokens[idx].starts_with('-')
    {
        input.target_path = Some(PathBuf::from(tokens[idx].clone()));
        idx += 1;
    }

    if command == CliCommand::Explain && idx < tokens.len() && !tokens[idx].starts_with('-') {
        input.result_id = Some(tokens[idx].clone());
        idx += 1;
    }

    // For download command, the next non-flag token is the model name.
    if command == CliCommand::Download && idx < tokens.len() && !tokens[idx].starts_with('-') {
        input.model_name = Some(tokens[idx].clone());
        idx += 1;
    }

    if command == CliCommand::Completions && idx < tokens.len() && !tokens[idx].starts_with('-') {
        let shell = CompletionShell::from_str(tokens[idx].as_str()).map_err(|()| {
            SearchError::InvalidConfig {
                field: "cli.completions.shell".into(),
                value: tokens[idx].clone(),
                reason: "expected bash|zsh|fish|powershell".into(),
            }
        })?;
        input.completion_shell = Some(shell);
        idx += 1;
    }

    // For config command, parse subcommand.
    if command == CliCommand::Config && idx < tokens.len() && !tokens[idx].starts_with('-') {
        let action = parse_config_action(&tokens, &mut idx)?;
        input.config_action = Some(action);
    }

    if command == CliCommand::Watch {
        input.watch = true;
        input.overrides.allow_background_indexing = Some(true);
    }

    while idx < tokens.len() {
        let flag = tokens[idx].as_str();
        match flag {
            "--help" | "-h" => {
                input.command = CliCommand::Help;
                input.command_source = CommandSource::Explicit;
                return Ok(input);
            }
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
                input.overrides.limit = Some(parse_search_limit(value)?);
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
                input.format_explicit = true;
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
            "--daemon" => {
                if !matches!(command, CliCommand::Search | CliCommand::Serve) {
                    return Err(SearchError::InvalidConfig {
                        field: "cli.flag".into(),
                        value: "--daemon".into(),
                        reason: "--daemon is only valid for search or serve commands".into(),
                    });
                }
                daemon_preference = Some(true);
                idx += 1;
            }
            "--no-daemon" => {
                if !matches!(command, CliCommand::Search | CliCommand::Serve) {
                    return Err(SearchError::InvalidConfig {
                        field: "cli.flag".into(),
                        value: "--no-daemon".into(),
                        reason: "--no-daemon is only valid for search or serve commands".into(),
                    });
                }
                daemon_preference = Some(false);
                idx += 1;
            }
            "--daemon-socket" => {
                if !matches!(command, CliCommand::Search | CliCommand::Serve) {
                    return Err(SearchError::InvalidConfig {
                        field: "cli.flag".into(),
                        value: "--daemon-socket".into(),
                        reason: "--daemon-socket is only valid for search or serve commands".into(),
                    });
                }
                let value = expect_value(&tokens, idx, "--daemon-socket")?;
                input.daemon_socket = Some(PathBuf::from(value));
                idx += 2;
            }
            "--watch" => {
                input.watch = true;
                idx += 1;
            }
            "--model" => {
                if command != CliCommand::Download {
                    return Err(SearchError::InvalidConfig {
                        field: "cli.flag".into(),
                        value: "--model".into(),
                        reason: "--model is only valid for the download command".into(),
                    });
                }
                let value = expect_value(&tokens, idx, "--model")?;
                input.model_name = Some(value.to_owned());
                idx += 2;
            }
            "--list" => {
                if command != CliCommand::Download {
                    return Err(SearchError::InvalidConfig {
                        field: "cli.flag".into(),
                        value: "--list".into(),
                        reason: "--list is only valid for the download command".into(),
                    });
                }
                input.download_list = true;
                idx += 1;
            }
            "--verify" => {
                if command != CliCommand::Download {
                    return Err(SearchError::InvalidConfig {
                        field: "cli.flag".into(),
                        value: "--verify".into(),
                        reason: "--verify is only valid for the download command".into(),
                    });
                }
                input.download_verify = true;
                idx += 1;
            }
            "--output" => {
                if command != CliCommand::Download {
                    return Err(SearchError::InvalidConfig {
                        field: "cli.flag".into(),
                        value: "--output".into(),
                        reason: "--output is only valid for the download command".into(),
                    });
                }
                let value = expect_value(&tokens, idx, "--output")?;
                input.download_output_dir = Some(PathBuf::from(value));
                idx += 2;
            }
            "--force" | "--full" => {
                input.full_reindex = true;
                idx += 1;
            }
            "--check" => {
                if command != CliCommand::Update {
                    return Err(SearchError::InvalidConfig {
                        field: "cli.flag".into(),
                        value: "--check".into(),
                        reason: "--check is only valid for the update command".into(),
                    });
                }
                input.update_check_only = true;
                idx += 1;
            }
            "--rollback" => {
                if command != CliCommand::Update {
                    return Err(SearchError::InvalidConfig {
                        field: "cli.flag".into(),
                        value: "--rollback".into(),
                        reason: "--rollback is only valid for the update command".into(),
                    });
                }
                input.update_rollback = true;
                // Optional version argument (e.g. --rollback v0.1.0).
                if let Some(next) = tokens.get(idx + 1)
                    && !next.starts_with('-')
                {
                    input.update_rollback_version = Some(next.clone());
                    idx += 1;
                }
                idx += 1;
            }
            "--purge" => {
                if command != CliCommand::Uninstall {
                    return Err(SearchError::InvalidConfig {
                        field: "cli.flag".into(),
                        value: "--purge".into(),
                        reason: "--purge is only valid for the uninstall command".into(),
                    });
                }
                input.uninstall_purge = true;
                idx += 1;
            }
            "--yes" => {
                if command != CliCommand::Uninstall {
                    return Err(SearchError::InvalidConfig {
                        field: "cli.flag".into(),
                        value: "--yes".into(),
                        reason: "--yes is only valid for the uninstall command".into(),
                    });
                }
                input.uninstall_yes = true;
                idx += 1;
            }
            "--dry-run" => {
                if command != CliCommand::Uninstall {
                    return Err(SearchError::InvalidConfig {
                        field: "cli.flag".into(),
                        value: "--dry-run".into(),
                        reason: "--dry-run is only valid for the uninstall command".into(),
                    });
                }
                input.uninstall_dry_run = true;
                idx += 1;
            }
            "--verbose" | "-v" => {
                input.verbose = true;
                idx += 1;
            }
            "--quiet" | "-q" => {
                input.quiet = true;
                idx += 1;
            }
            "--no-color" => {
                input.no_color = true;
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
            "--index-dir" => {
                let value = expect_value(&tokens, idx, "--index-dir")?;
                input.index_dir = Some(PathBuf::from(value));
                idx += 2;
            }
            _ => {
                if command == CliCommand::Search && input.query.is_none() {
                    if flag == "--" {
                        let query =
                            tokens
                                .get(idx + 1)
                                .ok_or_else(|| SearchError::InvalidConfig {
                                    field: "cli.search_query".into(),
                                    value: "--".into(),
                                    reason: "missing query after '--'".into(),
                                })?;
                        input.query = Some(query.clone());
                        idx += 2;
                        continue;
                    }

                    if !flag.starts_with('-') {
                        input.query = Some(tokens[idx].clone());
                        idx += 1;
                        continue;
                    }
                }

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
    apply_daemon_settings(&mut input, daemon_preference);
    validate_required_args(&input)?;

    Ok(input)
}

fn apply_daemon_settings(input: &mut CliInput, daemon_preference: Option<bool>) {
    if let Some(explicit_preference) = daemon_preference {
        input.daemon = explicit_preference;
        return;
    }

    #[cfg(unix)]
    {
        // Default search path uses daemon transport to amortize startup/index warmup.
        input.daemon = input.command == CliCommand::Search && !input.stream;
    }

    #[cfg(not(unix))]
    {
        input.daemon = false;
    }
}

fn validate_required_args(input: &CliInput) -> SearchResult<()> {
    if input.command == CliCommand::Search && input.query.is_none() {
        return Err(SearchError::InvalidConfig {
            field: "cli.search_query".into(),
            value: String::new(),
            reason: "missing search query argument".into(),
        });
    }
    if input.command == CliCommand::Explain && input.result_id.is_none() {
        return Err(SearchError::InvalidConfig {
            field: "cli.explain.result_id".into(),
            value: String::new(),
            reason: "missing result identifier argument".into(),
        });
    }
    if input.command == CliCommand::Completions && input.completion_shell.is_none() {
        return Err(SearchError::InvalidConfig {
            field: "cli.completions.shell".into(),
            value: String::new(),
            reason: "missing shell argument (bash|zsh|fish|powershell)".into(),
        });
    }
    if input.command == CliCommand::Download && input.download_list && input.download_verify {
        return Err(SearchError::InvalidConfig {
            field: "cli.download.mode".into(),
            value: "--list --verify".into(),
            reason: "use either --list or --verify, not both".into(),
        });
    }
    if input.command == CliCommand::Search && input.stream && input.daemon {
        return Err(SearchError::InvalidConfig {
            field: "cli.search.mode".into(),
            value: "--stream --daemon".into(),
            reason: "stream mode is not supported with daemon transport".into(),
        });
    }
    Ok(())
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
    if let Some(token) = tokens.first() {
        if is_help_flag(token) {
            return Ok((CliCommand::Help, 1, CommandSource::Explicit));
        }
        if !token.starts_with('-') {
            return Ok((parse_command(token)?, 1, CommandSource::Explicit));
        }
    }
    Ok((CliCommand::default(), 0, CommandSource::ImplicitDefault))
}

fn parse_command(token: &str) -> SearchResult<CliCommand> {
    match token {
        "search" | "s" => Ok(CliCommand::Search),
        "serve" | "srv" => Ok(CliCommand::Serve),
        "index" | "idx" => Ok(CliCommand::Index),
        "watch" | "w" => Ok(CliCommand::Watch),
        "status" | "st" => Ok(CliCommand::Status),
        "explain" | "ex" => Ok(CliCommand::Explain),
        "config" | "cfg" => Ok(CliCommand::Config),
        "download-models" | "download" | "dl" => Ok(CliCommand::Download),
        "doctor" | "doc" => Ok(CliCommand::Doctor),
        "update" | "up" => Ok(CliCommand::Update),
        "completions" | "comp" => Ok(CliCommand::Completions),
        "uninstall" | "rm" => Ok(CliCommand::Uninstall),
        "help" | "h" => Ok(CliCommand::Help),
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
        "show" | "list" | "ls" => Ok(ConfigAction::Show),
        "init" => Ok(ConfigAction::Init),
        "validate" | "check" => Ok(ConfigAction::Validate),
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
            reason: "expected show|get|set|init|validate|reset".into(),
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

fn parse_search_limit(value: &str) -> SearchResult<usize> {
    if value.eq_ignore_ascii_case("all") {
        return Ok(usize::MAX);
    }
    parse_usize(value, "search.default_limit")
}

#[must_use]
fn is_help_flag(token: &str) -> bool {
    matches!(token, "--help" | "-h")
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
            | "--daemon"
            | "--no-daemon"
            | "--daemon-socket"
            | "--watch"
            | "--model"
            | "--list"
            | "--verify"
            | "--output"
            | "--force"
            | "--full"
            | "--check"
            | "--purge"
            | "--yes"
            | "--dry-run"
            | "--verbose"
            | "-v"
            | "--quiet"
            | "-q"
            | "--no-color"
            | "--filter"
            | "--profile"
            | "--theme"
            | "--config"
            | "--index-dir"
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
            parse_cli_args(["serve"]).unwrap().command,
            CliCommand::Serve
        );
        assert_eq!(
            parse_cli_args(["doctor"]).unwrap().command,
            CliCommand::Doctor
        );
        assert_eq!(
            parse_cli_args(["watch"]).unwrap().command,
            CliCommand::Watch
        );
        assert_eq!(
            parse_cli_args(["update"]).unwrap().command,
            CliCommand::Update
        );
        assert_eq!(
            parse_cli_args(["completions", "bash"]).unwrap().command,
            CliCommand::Completions
        );
        assert_eq!(
            parse_cli_args(["uninstall"]).unwrap().command,
            CliCommand::Uninstall
        );
        assert_eq!(parse_cli_args(["help"]).unwrap().command, CliCommand::Help);
        assert_eq!(parse_cli_args(["tui"]).unwrap().command, CliCommand::Tui);
        assert_eq!(
            parse_cli_args(["version"]).unwrap().command,
            CliCommand::Version
        );
    }

    #[test]
    fn parse_command_aliases() {
        assert_eq!(
            parse_cli_args(["s", "q"]).unwrap().command,
            CliCommand::Search
        );
        assert_eq!(parse_cli_args(["srv"]).unwrap().command, CliCommand::Serve);
        assert_eq!(parse_cli_args(["idx"]).unwrap().command, CliCommand::Index);
        assert_eq!(parse_cli_args(["w"]).unwrap().command, CliCommand::Watch);
        assert_eq!(parse_cli_args(["st"]).unwrap().command, CliCommand::Status);
        assert_eq!(
            parse_cli_args(["ex", "r0"]).unwrap().command,
            CliCommand::Explain
        );
        assert_eq!(parse_cli_args(["cfg"]).unwrap().command, CliCommand::Config);
        assert_eq!(
            parse_cli_args(["dl"]).unwrap().command,
            CliCommand::Download
        );
        assert_eq!(parse_cli_args(["doc"]).unwrap().command, CliCommand::Doctor);
        assert_eq!(parse_cli_args(["up"]).unwrap().command, CliCommand::Update);
        assert_eq!(
            parse_cli_args(["comp", "zsh"]).unwrap().command,
            CliCommand::Completions
        );
        assert_eq!(
            parse_cli_args(["rm"]).unwrap().command,
            CliCommand::Uninstall
        );
        assert_eq!(parse_cli_args(["h"]).unwrap().command, CliCommand::Help);
        assert_eq!(
            parse_cli_args(["ver"]).unwrap().command,
            CliCommand::Version
        );
    }

    #[test]
    fn parse_help_flag() {
        let input = parse_cli_args(["--help"]).expect("parse");
        assert_eq!(input.command, CliCommand::Help);
        assert_eq!(input.command_source, CommandSource::Explicit);
    }

    #[test]
    fn parse_help_flag_after_command() {
        let input = parse_cli_args(["status", "--help"]).expect("parse");
        assert_eq!(input.command, CliCommand::Help);
        assert_eq!(input.command_source, CommandSource::Explicit);
    }

    #[test]
    fn parse_output_format() {
        let input = parse_cli_args(["status", "--format", "json"]).unwrap();
        assert_eq!(input.format, OutputFormat::Json);
        assert!(input.format_explicit);

        let toon = parse_cli_args(["status", "--format", "toon"]).unwrap();
        assert_eq!(toon.format, OutputFormat::Toon);
        assert!(toon.format_explicit);
    }

    #[test]
    fn parse_short_flags() {
        let input = parse_cli_args(["search", "q", "-l", "5", "-f", "csv", "-e"]).unwrap();
        assert_eq!(input.overrides.limit, Some(5));
        assert_eq!(input.format, OutputFormat::Csv);
        assert_eq!(input.overrides.explain, Some(true));
    }

    #[test]
    fn parse_limit_all_alias() {
        let input = parse_cli_args(["search", "query", "--limit", "all"]).unwrap();
        assert_eq!(input.overrides.limit, Some(usize::MAX));
    }

    #[test]
    fn parse_daemon_flags_for_search() {
        let input = parse_cli_args([
            "search",
            "query",
            "--daemon",
            "--daemon-socket",
            "/tmp/fsfs-query.sock",
        ])
        .unwrap();
        assert!(input.daemon);
        assert_eq!(
            input.daemon_socket,
            Some(PathBuf::from("/tmp/fsfs-query.sock"))
        );
    }

    #[test]
    fn parse_serve_accepts_daemon_flag() {
        let input = parse_cli_args(["serve", "--daemon"]).expect("parse");
        assert_eq!(input.command, CliCommand::Serve);
        assert!(input.daemon);
    }

    #[test]
    fn parse_search_defaults_to_daemon_on_unix() {
        let input = parse_cli_args(["search", "query"]).expect("parse");
        #[cfg(unix)]
        assert!(input.daemon);
        #[cfg(not(unix))]
        assert!(!input.daemon);
    }

    #[test]
    fn parse_no_daemon_disables_auto_daemon_transport() {
        let input = parse_cli_args(["search", "query", "--no-daemon"]).expect("parse");
        assert!(!input.daemon);
    }

    #[test]
    fn parse_stream_flag() {
        let input = parse_cli_args(["search", "test", "--stream"]).unwrap();
        assert!(input.stream);
        assert!(!input.daemon);
        assert_eq!(input.format, OutputFormat::Jsonl);
        assert!(!input.format_explicit);
    }

    #[test]
    fn parse_stream_accepts_toon_format() {
        let input = parse_cli_args(["search", "test", "--stream", "--format", "toon"]).unwrap();
        assert!(input.stream);
        assert_eq!(input.format, OutputFormat::Toon);
        assert!(input.format_explicit);
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
    fn parse_daemon_rejects_non_search_or_serve_command() {
        let err = parse_cli_args(["status", "--daemon"]).expect_err("must fail");
        assert!(
            err.to_string()
                .contains("--daemon is only valid for search or serve commands")
        );
    }

    #[test]
    fn parse_no_daemon_rejects_non_search_or_serve_command() {
        let err = parse_cli_args(["status", "--no-daemon"]).expect_err("must fail");
        assert!(
            err.to_string()
                .contains("--no-daemon is only valid for search or serve commands")
        );
    }

    #[test]
    fn parse_stream_and_daemon_together_rejected() {
        let err =
            parse_cli_args(["search", "query", "--stream", "--daemon"]).expect_err("must fail");
        assert!(
            err.to_string()
                .contains("stream mode is not supported with daemon transport")
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
    fn parse_index_dir_override() {
        let input = parse_cli_args(["status", "--index-dir", "/tmp/fsfs-index"]).unwrap();
        assert_eq!(input.index_dir, Some(PathBuf::from("/tmp/fsfs-index")));
    }

    #[test]
    fn parse_index_flags() {
        let input = parse_cli_args(["index", "--full", "--watch"]).unwrap();
        assert_eq!(input.command, CliCommand::Index);
        assert!(input.full_reindex);
        assert!(input.watch);
    }

    #[test]
    fn parse_watch_command_sets_watch_mode_and_path() {
        let input = parse_cli_args(["watch", "/tmp/corpus"]).unwrap();
        assert_eq!(input.command, CliCommand::Watch);
        assert!(input.watch);
        assert_eq!(input.target_path, Some(PathBuf::from("/tmp/corpus")));
        assert_eq!(input.overrides.allow_background_indexing, Some(true));
    }

    #[test]
    fn parse_force_alias_for_full_reindex() {
        let input = parse_cli_args(["index", "--force"]).unwrap();
        assert!(input.full_reindex);
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
    fn parse_download_models_alias() {
        let input = parse_cli_args(["download-models"]).unwrap();
        assert_eq!(input.command, CliCommand::Download);
    }

    #[test]
    fn parse_download_flags() {
        let input = parse_cli_args([
            "download-models",
            "--model",
            "all-MiniLM-L6-v2",
            "--list",
            "--output",
            "/tmp/models",
            "--force",
        ])
        .unwrap();
        assert_eq!(input.command, CliCommand::Download);
        assert_eq!(input.model_name.as_deref(), Some("all-MiniLM-L6-v2"));
        assert!(input.download_list);
        assert!(!input.download_verify);
        assert_eq!(
            input.download_output_dir,
            Some(PathBuf::from("/tmp/models"))
        );
        assert!(input.full_reindex);
    }

    #[test]
    fn parse_download_rejects_list_and_verify_together() {
        let err = parse_cli_args(["download-models", "--list", "--verify"]).expect_err("must fail");
        assert!(
            err.to_string()
                .contains("use either --list or --verify, not both")
        );
    }

    #[test]
    fn parse_filter_flag() {
        let input = parse_cli_args(["search", "query", "--filter", "type:rs"]).unwrap();
        assert_eq!(input.filter.as_deref(), Some("type:rs"));
    }

    #[test]
    fn parse_search_requires_query() {
        let err = parse_cli_args(["search"]).expect_err("must fail");
        assert!(err.to_string().contains("missing search query argument"));
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
    fn parse_search_query_after_flags() {
        let input = parse_cli_args(["search", "--format", "json", "latency"]).unwrap();
        assert_eq!(input.command, CliCommand::Search);
        assert_eq!(input.format, OutputFormat::Json);
        assert_eq!(input.query.as_deref(), Some("latency"));
    }

    #[test]
    fn parse_search_query_after_flags_with_double_dash() {
        let input = parse_cli_args(["search", "--format", "json", "--", "-secret-token"]).unwrap();
        assert_eq!(input.command, CliCommand::Search);
        assert_eq!(input.format, OutputFormat::Json);
        assert_eq!(input.query.as_deref(), Some("-secret-token"));
    }

    #[test]
    fn parse_search_double_dash_requires_query() {
        let err = parse_cli_args(["search", "--"]).expect_err("must fail");
        assert!(err.to_string().contains("missing query after '--'"));
    }

    #[test]
    fn parse_explain_requires_result_id() {
        let err = parse_cli_args(["explain"]).expect_err("must fail");
        assert!(
            err.to_string()
                .contains("missing result identifier argument")
        );
    }

    #[test]
    fn parse_explain_result_id() {
        let input = parse_cli_args(["explain", "hit_123"]).unwrap();
        assert_eq!(input.command, CliCommand::Explain);
        assert_eq!(input.result_id.as_deref(), Some("hit_123"));
    }

    #[test]
    fn parse_config_list() {
        let input = parse_cli_args(["config", "list"]).unwrap();
        assert_eq!(input.command, CliCommand::Config);
        assert_eq!(input.config_action, Some(ConfigAction::Show));
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
        assert_eq!(input.config_action, Some(ConfigAction::Show));
    }

    #[test]
    fn parse_config_show_init_validate() {
        let show = parse_cli_args(["config", "show"]).unwrap();
        assert_eq!(show.config_action, Some(ConfigAction::Show));

        let init = parse_cli_args(["config", "init"]).unwrap();
        assert_eq!(init.config_action, Some(ConfigAction::Init));

        let validate = parse_cli_args(["config", "validate"]).unwrap();
        assert_eq!(validate.config_action, Some(ConfigAction::Validate));
    }

    #[test]
    fn parse_completions_requires_shell() {
        let err = parse_cli_args(["completions"]).expect_err("must fail");
        assert!(err.to_string().contains("missing shell argument"));
    }

    #[test]
    fn parse_completions_shell() {
        let input = parse_cli_args(["completions", "fish"]).unwrap();
        assert_eq!(input.command, CliCommand::Completions);
        assert_eq!(input.completion_shell, Some(CompletionShell::Fish));
    }

    #[test]
    fn parse_update_check_flag() {
        let input = parse_cli_args(["update", "--check"]).unwrap();
        assert_eq!(input.command, CliCommand::Update);
        assert!(input.update_check_only);
    }

    #[test]
    fn parse_update_rollback_flag() {
        let input = parse_cli_args(["update", "--rollback"]).unwrap();
        assert_eq!(input.command, CliCommand::Update);
        assert!(input.update_rollback);
        assert!(input.update_rollback_version.is_none());
    }

    #[test]
    fn parse_update_rollback_with_version() {
        let input = parse_cli_args(["update", "--rollback", "v0.1.0"]).unwrap();
        assert!(input.update_rollback);
        assert_eq!(input.update_rollback_version.as_deref(), Some("v0.1.0"));
    }

    #[test]
    fn parse_rollback_rejects_non_update_commands() {
        let err = parse_cli_args(["status", "--rollback"]).expect_err("must fail");
        assert!(
            err.to_string()
                .contains("only valid for the update command"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn parse_check_rejects_non_update_commands() {
        let err = parse_cli_args(["status", "--check"]).expect_err("must fail");
        assert!(
            err.to_string()
                .contains("only valid for the update command")
        );
    }

    #[test]
    fn parse_uninstall_flags() {
        let input = parse_cli_args(["uninstall", "--purge", "--yes", "--dry-run"]).unwrap();
        assert_eq!(input.command, CliCommand::Uninstall);
        assert!(input.uninstall_purge);
        assert!(input.uninstall_yes);
        assert!(input.uninstall_dry_run);
    }

    #[test]
    fn parse_uninstall_flags_reject_non_uninstall_commands() {
        for flag in ["--purge", "--yes", "--dry-run"] {
            let err = parse_cli_args(["status", flag]).expect_err("must fail");
            assert!(
                err.to_string()
                    .contains("only valid for the uninstall command"),
                "unexpected error for {flag}: {err}"
            );
        }
    }

    #[test]
    fn parse_global_output_flags() {
        let input = parse_cli_args(["status", "--verbose", "--quiet", "--no-color"]).unwrap();
        assert!(input.verbose);
        assert!(input.quiet);
        assert!(input.no_color);
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
        assert!(msg.contains("expected show|get|set|init|validate|reset"));
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
    fn resolve_output_format_defaults_to_jsonl_on_non_tty() {
        assert_eq!(
            resolve_output_format(OutputFormat::Table, false, false),
            OutputFormat::Jsonl
        );
    }

    #[test]
    fn resolve_output_format_keeps_table_on_tty() {
        assert_eq!(
            resolve_output_format(OutputFormat::Table, false, true),
            OutputFormat::Table
        );
    }

    #[test]
    fn resolve_output_format_respects_explicit_override() {
        assert_eq!(
            resolve_output_format(OutputFormat::Table, true, false),
            OutputFormat::Table
        );
        assert_eq!(
            resolve_output_format(OutputFormat::Json, true, false),
            OutputFormat::Json
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
            let parsed = match *name {
                "search" => parse_cli_args(["search", "q"]),
                "explain" => parse_cli_args(["explain", "result-id"]),
                "completions" => parse_cli_args(["completions", "bash"]),
                _ => parse_cli_args([*name]),
            };
            let _input = parsed.expect(name);
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
        assert!(!input.format_explicit);
        assert!(input.query.is_none());
        assert!(!input.stream);
    }
}
