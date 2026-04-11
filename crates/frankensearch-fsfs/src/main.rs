use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use asupersync::Cx;
use asupersync::runtime::RuntimeBuilder;
use frankensearch_core::{SearchError, SearchResult};
use frankensearch_fsfs::{
    CliCommand, CliInput, CliOverrides, ConfigAction, ConfigLoadResult, ConfigWarning, FsfsConfig,
    FsfsRuntime, InterfaceMode, OutputEnvelope, OutputFormat, PathExpansion, ShutdownCoordinator,
    ShutdownReason, Verbosity, default_project_config_file_path, default_user_config_file_path,
    detect_auto_mode, emit_config_loaded, emit_envelope, exit_code, init_subscriber,
    is_cache_valid, load_from_layered_sources, load_from_sources, load_from_str,
    maybe_print_update_notice, meta_for_format, parse_cli_args, read_version_cache,
    resolve_output_format, spawn_version_cache_refresh,
};
use serde::Serialize;
use serde_json::Value;
use tracing::info;

const CONFIG_SUBSYSTEM: &str = "fsfs.config";

#[derive(Debug, Clone, Serialize)]
struct ConfigFileSummary {
    explicit: Option<String>,
    project: Option<String>,
    user: Option<String>,
    active: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ConfigShowPayload {
    effective: FsfsConfig,
    effective_toml: String,
    value_sources: BTreeMap<String, String>,
    source_precedence: Vec<String>,
    files: ConfigFileSummary,
    warnings: Vec<ConfigWarning>,
    path_expansions: Vec<PathExpansion>,
}

#[derive(Debug, Clone, Serialize)]
struct ConfigGetPayload {
    key: String,
    value: Value,
    source: String,
}

#[derive(Debug, Clone, Serialize)]
struct ConfigValidatePayload {
    valid: bool,
    warning_count: usize,
    source_precedence: Vec<String>,
    files: ConfigFileSummary,
    warnings: Vec<ConfigWarning>,
}

#[derive(Debug, Clone, Serialize)]
struct ConfigInitPayload {
    created: bool,
    path: String,
    template: String,
}

#[derive(Debug, Clone)]
struct ConfigStageSnapshot {
    name: &'static str,
    values: BTreeMap<String, Value>,
    changed_keys: HashSet<String>,
}

fn allow_missing_explicit_config(command: CliCommand, action: Option<&ConfigAction>) -> bool {
    if command != CliCommand::Config {
        return false;
    }
    matches!(
        action,
        Some(ConfigAction::Init | ConfigAction::Reset | ConfigAction::Set { .. })
    )
}

#[allow(clippy::too_many_lines)]
fn main() -> SearchResult<()> {
    let mut cli_input = parse_cli_args(std::env::args().skip(1))?;
    let env_map: HashMap<String, String> = std::env::vars().collect();
    apply_cli_env_overrides(&mut cli_input, &env_map)?;

    // Version is handled immediately, before config loading.
    if cli_input.command == CliCommand::Version {
        println!(
            "fsfs {} (frankensearch {})",
            env!("CARGO_PKG_VERSION"),
            env!("CARGO_PKG_VERSION"),
        );
        std::process::exit(exit_code::OK);
    }

    // Initialize tracing subscriber before anything else that emits events.
    // CLI flags are already parsed, so we can derive verbosity.
    let verbosity = Verbosity::from_flags(cli_input.verbose, cli_input.quiet);
    init_subscriber(verbosity, cli_input.no_color);

    let home_dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"));
    let mut explicit_config_path = None;
    let mut project_config_path = None;
    let mut user_config_path = None;
    let allow_missing_config =
        allow_missing_explicit_config(cli_input.command, cli_input.config_action.as_ref());
    let loaded = if let Some(path) = cli_input.overrides.config_path.as_deref() {
        let config_path = expand_cli_config_path(path, &home_dir);
        if !config_path.exists() && !allow_missing_config {
            return Err(SearchError::InvalidConfig {
                field: "config_file".to_owned(),
                value: config_path.display().to_string(),
                reason: "explicitly provided --config path does not exist".to_owned(),
            });
        }
        explicit_config_path = Some(config_path.clone());
        load_from_sources(
            Some(config_path.as_path()),
            &env_map,
            &cli_input.overrides,
            &home_dir,
        )?
    } else {
        let cwd = std::env::current_dir().map_err(SearchError::Io)?;
        project_config_path = Some(default_project_config_file_path(&cwd));
        user_config_path = Some(default_user_config_file_path(&home_dir));
        load_from_layered_sources(
            project_config_path.as_deref(),
            user_config_path.as_deref(),
            &env_map,
            &cli_input.overrides,
            &home_dir,
        )?
    };

    let event = loaded.to_loaded_event();
    emit_config_loaded(&event);

    let is_tty = std::io::IsTerminal::is_terminal(&std::io::stdout());
    let Some(command) = detect_auto_mode(cli_input.command, is_tty, cli_input.command_source)
    else {
        eprintln!("usage: fsfs <command> [flags]");
        eprintln!("commands: {}", CliCommand::ALL_NAMES.join(", "));
        std::process::exit(exit_code::USAGE_ERROR);
    };

    let mut runtime_cli_input = cli_input;
    runtime_cli_input.command = command;
    runtime_cli_input.format = resolve_output_format(
        runtime_cli_input.format,
        runtime_cli_input.format_explicit,
        is_tty,
    );
    if command == CliCommand::Config {
        return run_config_command(
            &loaded,
            &runtime_cli_input,
            &env_map,
            &home_dir,
            explicit_config_path.as_deref(),
            project_config_path.as_deref(),
            user_config_path.as_deref(),
        );
    }

    let mut resolved_config = loaded.config;
    if runtime_cli_input.watch {
        resolved_config.indexing.watch_mode = true;
    }
    let cli_quiet = runtime_cli_input.quiet;
    let app_runtime = FsfsRuntime::new(resolved_config).with_cli_input(runtime_cli_input);
    let interface_mode = match command {
        CliCommand::Tui => InterfaceMode::Tui,
        _ => InterfaceMode::Cli,
    };

    info!(
        command = ?command,
        interface_mode = ?interface_mode,
        pressure_profile = ?app_runtime.config().pressure.profile,
        "fsfs command parsed and runtime wired"
    );

    let scheduler =
        RuntimeBuilder::current_thread()
            .build()
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "fsfs",
                source: Box::new(io::Error::other(format!(
                    "failed to initialize asupersync runtime: {error}"
                ))),
            })?;
    let shutdown = Arc::new(ShutdownCoordinator::new());
    shutdown.register_signals()?;
    let cx = Cx::for_request();
    let run_with_shutdown = matches!(interface_mode, InterfaceMode::Tui)
        || app_runtime.config().indexing.watch_mode
        || command == CliCommand::Daemon;
    // ── Startup version check (non-blocking) ──────────────────────────
    // Print a one-line update notice from cache. If the cache is expired
    // or missing, spawn a background thread to refresh it for next time.
    // Disabled when FRANKENSEARCH_CHECK_UPDATES=0 or in quiet mode.
    let updates_disabled = env_map
        .get("FRANKENSEARCH_CHECK_UPDATES")
        .or_else(|| env_map.get("FSFS_CHECK_UPDATES"))
        .is_some_and(|v| {
            v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("no")
        });

    if !updates_disabled && !cli_quiet && command != CliCommand::Update {
        let _ = maybe_print_update_notice(false);
        // If cache is expired or missing, refresh in background.
        let needs_refresh = read_version_cache()
            .as_ref()
            .is_none_or(|c| !is_cache_valid(c));
        if needs_refresh {
            spawn_version_cache_refresh();
        }
    }
    // ─────────────────────────────────────────────────────────────────────

    let shutdown_for_run = Arc::clone(&shutdown);

    let run_result = scheduler.block_on(async move {
        let run_result = if run_with_shutdown {
            app_runtime
                .run_mode_with_shutdown(&cx, interface_mode, shutdown_for_run.as_ref())
                .await
        } else {
            app_runtime.run_mode(&cx, interface_mode).await
        };

        if let Err(error) = &run_result {
            shutdown_for_run.request_shutdown(ShutdownReason::Error(error.to_string()));
        }

        run_result
    });
    shutdown.stop_signal_listener();

    if shutdown.is_force_exit_requested() {
        std::process::exit(exit_code::INTERRUPTED);
    }

    if let Some(reason) = shutdown.current_reason()
        && shutdown.is_shutting_down()
    {
        info!(reason = ?reason, "fsfs shutdown completed at process boundary");
    }

    run_result
}

fn expand_cli_config_path(path: &std::path::Path, home_dir: &std::path::Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if raw == "~" {
        home_dir.to_path_buf()
    } else if let Some(rest) = raw.strip_prefix("~/") {
        home_dir.join(rest)
    } else {
        path.to_path_buf()
    }
}

fn run_config_command(
    loaded: &ConfigLoadResult,
    cli_input: &CliInput,
    env_map: &HashMap<String, String>,
    home_dir: &Path,
    explicit_config_path: Option<&Path>,
    project_config_path: Option<&Path>,
    user_config_path: Option<&Path>,
) -> SearchResult<()> {
    let action = cli_input
        .config_action
        .clone()
        .unwrap_or(ConfigAction::Show);
    match action {
        ConfigAction::Show => run_config_show_command(
            loaded,
            cli_input,
            env_map,
            home_dir,
            explicit_config_path,
            project_config_path,
            user_config_path,
        ),
        ConfigAction::Get { key } => run_config_get_command(
            loaded,
            cli_input,
            env_map,
            home_dir,
            explicit_config_path,
            project_config_path,
            user_config_path,
            &key,
        ),
        ConfigAction::Validate => {
            let payload = ConfigValidatePayload {
                valid: true,
                warning_count: loaded.warnings.len(),
                source_precedence: configured_precedence(explicit_config_path.is_some()),
                files: config_file_summary(
                    loaded,
                    explicit_config_path,
                    project_config_path,
                    user_config_path,
                ),
                warnings: loaded.warnings.clone(),
            };
            if cli_input.format == OutputFormat::Table {
                print!("config is valid");
                if payload.warning_count > 0 {
                    println!(" (warnings: {})", payload.warning_count);
                    for warning in &payload.warnings {
                        println!(
                            "- [{}] {} ({})",
                            warning.reason_code, warning.field, warning.message
                        );
                    }
                } else {
                    println!();
                }
                return Ok(());
            }
            emit_success_payload("config", &payload, cli_input.format)
        }
        ConfigAction::Init => run_config_init_command(
            cli_input,
            explicit_config_path,
            project_config_path,
            user_config_path,
        ),
        ConfigAction::Set { key, value } => run_config_set_command(
            cli_input,
            explicit_config_path,
            project_config_path,
            user_config_path,
            &key,
            &value,
        ),
        ConfigAction::Reset => run_config_reset_command(
            cli_input,
            explicit_config_path,
            project_config_path,
            user_config_path,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_config_get_command(
    loaded: &ConfigLoadResult,
    cli_input: &CliInput,
    env_map: &HashMap<String, String>,
    home_dir: &Path,
    explicit_config_path: Option<&Path>,
    project_config_path: Option<&Path>,
    user_config_path: Option<&Path>,
    key: &str,
) -> SearchResult<()> {
    let stage_snapshots = build_stage_snapshots(
        env_map,
        &cli_input.overrides,
        home_dir,
        explicit_config_path,
        project_config_path,
        user_config_path,
    )?;
    let final_values = flatten_config_values(&loaded.config)?;
    let value = final_values
        .get(key)
        .cloned()
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "config.get".to_owned(),
            value: key.to_owned(),
            reason: "unknown configuration key".to_owned(),
        })?;
    let source = source_for_key(key, &value, &stage_snapshots).to_owned();
    if cli_input.format == OutputFormat::Table {
        println!("{key} = {} ({source})", format_value_for_table(&value));
        return Ok(());
    }
    let payload = ConfigGetPayload {
        key: key.to_owned(),
        value,
        source,
    };
    emit_success_payload("config", &payload, cli_input.format)
}

fn run_config_show_command(
    loaded: &ConfigLoadResult,
    cli_input: &CliInput,
    env_map: &HashMap<String, String>,
    home_dir: &Path,
    explicit_config_path: Option<&Path>,
    project_config_path: Option<&Path>,
    user_config_path: Option<&Path>,
) -> SearchResult<()> {
    let stage_snapshots = build_stage_snapshots(
        env_map,
        &cli_input.overrides,
        home_dir,
        explicit_config_path,
        project_config_path,
        user_config_path,
    )?;
    let final_values = flatten_config_values(&loaded.config)?;
    let value_sources = final_values
        .iter()
        .map(|(key, value)| {
            (
                key.clone(),
                source_for_key(key, value, &stage_snapshots).to_owned(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    if cli_input.format == OutputFormat::Table {
        for (key, value) in &final_values {
            let source = value_sources.get(key).map_or("defaults", String::as_str);
            println!("{key} = {} ({source})", format_value_for_table(value));
        }
        if !loaded.warnings.is_empty() {
            println!();
            println!("warnings:");
            for warning in &loaded.warnings {
                println!(
                    "- [{}] {} ({})",
                    warning.reason_code, warning.field, warning.message
                );
            }
        }
        return Ok(());
    }
    let payload = ConfigShowPayload {
        effective: loaded.config.clone(),
        effective_toml: toml::to_string_pretty(&loaded.config).map_err(|source| {
            SearchError::SubsystemError {
                subsystem: CONFIG_SUBSYSTEM,
                source: Box::new(io::Error::other(format!(
                    "failed to encode effective config as TOML: {source}"
                ))),
            }
        })?,
        value_sources,
        source_precedence: configured_precedence(explicit_config_path.is_some()),
        files: config_file_summary(
            loaded,
            explicit_config_path,
            project_config_path,
            user_config_path,
        ),
        warnings: loaded.warnings.clone(),
        path_expansions: loaded.path_expansions.clone(),
    };
    emit_success_payload("config", &payload, cli_input.format)
}

fn run_config_init_command(
    cli_input: &CliInput,
    explicit_config_path: Option<&Path>,
    project_config_path: Option<&Path>,
    user_config_path: Option<&Path>,
) -> SearchResult<()> {
    let target = explicit_config_path
        .or(project_config_path)
        .or(user_config_path)
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "config.init.path".to_owned(),
            value: String::new(),
            reason: "unable to determine config path".to_owned(),
        })?;
    let path = target.to_path_buf();
    reject_directory_path(&path, "config.init.path")?;
    let template = toml::to_string_pretty(&FsfsConfig::default()).map_err(|source| {
        SearchError::SubsystemError {
            subsystem: CONFIG_SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "failed to encode default config template: {source}"
            ))),
        }
    })?;
    let created = if path.exists() {
        false
    } else {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, &template)?;
        true
    };
    if cli_input.format == OutputFormat::Table {
        if created {
            println!("created {}", path.display());
        } else {
            println!("exists {}", path.display());
        }
        return Ok(());
    }
    let payload = ConfigInitPayload {
        created,
        path: path.display().to_string(),
        template,
    };
    emit_success_payload("config", &payload, cli_input.format)
}

#[derive(Debug, Clone, Serialize)]
struct ConfigSetPayload {
    path: String,
    key: String,
    value: String,
}

#[derive(Debug, Clone, Serialize)]
struct ConfigResetPayload {
    path: String,
}

fn resolve_writable_config_path(
    explicit: Option<&Path>,
    project: Option<&Path>,
    user: Option<&Path>,
) -> SearchResult<PathBuf> {
    // Prefer: explicit > project > user. At least one must exist or be derivable.
    explicit
        .or(project)
        .or(user)
        .map(Path::to_path_buf)
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "config.path".to_owned(),
            value: String::new(),
            reason: "unable to determine config path; use --config to specify one".to_owned(),
        })
        .and_then(|path| {
            reject_directory_path(&path, "config.path")?;
            Ok(path)
        })
}

fn reject_directory_path(path: &Path, field: &'static str) -> SearchResult<()> {
    if path.exists() && path.is_dir() {
        return Err(SearchError::InvalidConfig {
            field: field.to_owned(),
            value: path.display().to_string(),
            reason: "config path points to a directory".to_owned(),
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_config_set_command(
    cli_input: &CliInput,
    explicit_config_path: Option<&Path>,
    project_config_path: Option<&Path>,
    user_config_path: Option<&Path>,
    key: &str,
    value: &str,
) -> SearchResult<()> {
    let path =
        resolve_writable_config_path(explicit_config_path, project_config_path, user_config_path)?;

    // Load the existing TOML file, or start from an empty table if it doesn't exist.
    let existing_text = if path.exists() {
        std::fs::read_to_string(&path)?
    } else {
        String::new()
    };
    let mut doc: toml::Value = if existing_text.trim().is_empty() {
        toml::Value::Table(toml::map::Map::new())
    } else {
        existing_text
            .parse()
            .map_err(|source: toml::de::Error| SearchError::SubsystemError {
                subsystem: CONFIG_SUBSYSTEM,
                source: Box::new(io::Error::other(format!(
                    "failed to parse existing config at {}: {source}",
                    path.display()
                ))),
            })?
    };

    // Navigate the dotted key path and set the value.
    let segments: Vec<&str> = key.split('.').collect();
    if segments.is_empty() || segments.iter().any(|s| s.is_empty()) {
        return Err(SearchError::InvalidConfig {
            field: "config.set.key".to_owned(),
            value: key.to_owned(),
            reason: "key must be a non-empty dot-separated path (e.g. 'search.rrf_k')".to_owned(),
        });
    }

    // Descend to the parent table, creating intermediate tables as needed.
    let mut cursor = &mut doc;
    for &segment in &segments[..segments.len() - 1] {
        if !cursor.is_table() {
            *cursor = toml::Value::Table(toml::map::Map::new());
        }
        cursor = cursor
            .as_table_mut()
            .ok_or_else(|| SearchError::SubsystemError {
                subsystem: CONFIG_SUBSYSTEM,
                source: Box::new(io::Error::other(format!(
                    "config.set path segment '{segment}' did not resolve to a table"
                ))),
            })?
            .entry(segment)
            .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
    }
    let leaf_key = segments[segments.len() - 1];

    // Parse the value: try bool, then integer, then float, then string.
    let parsed_value = if value.eq_ignore_ascii_case("true") {
        toml::Value::Boolean(true)
    } else if value.eq_ignore_ascii_case("false") {
        toml::Value::Boolean(false)
    } else if let Ok(i) = value.parse::<i64>() {
        toml::Value::Integer(i)
    } else if let Ok(f) = value.parse::<f64>() {
        if f.is_finite() {
            toml::Value::Float(f)
        } else {
            // TOML does not support inf/nan — treat as a literal string.
            toml::Value::String(value.to_owned())
        }
    } else {
        toml::Value::String(value.to_owned())
    };

    if !cursor.is_table() {
        *cursor = toml::Value::Table(toml::map::Map::new());
    }
    cursor
        .as_table_mut()
        .ok_or_else(|| SearchError::SubsystemError {
            subsystem: CONFIG_SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "config.set path '{key}' did not resolve to a table"
            ))),
        })?
        .insert(leaf_key.to_owned(), parsed_value);

    // Write back.
    let output = toml::to_string_pretty(&doc).map_err(|source| SearchError::SubsystemError {
        subsystem: CONFIG_SUBSYSTEM,
        source: Box::new(io::Error::other(format!(
            "failed to encode config: {source}"
        ))),
    })?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, &output)?;

    if cli_input.format == OutputFormat::Table {
        println!("set {key}={value} in {}", path.display());
        return Ok(());
    }
    let payload = ConfigSetPayload {
        path: path.display().to_string(),
        key: key.to_owned(),
        value: value.to_owned(),
    };
    emit_success_payload("config", &payload, cli_input.format)
}

fn run_config_reset_command(
    cli_input: &CliInput,
    explicit_config_path: Option<&Path>,
    project_config_path: Option<&Path>,
    user_config_path: Option<&Path>,
) -> SearchResult<()> {
    let path =
        resolve_writable_config_path(explicit_config_path, project_config_path, user_config_path)?;
    let template = toml::to_string_pretty(&FsfsConfig::default()).map_err(|source| {
        SearchError::SubsystemError {
            subsystem: CONFIG_SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "failed to encode default config: {source}"
            ))),
        }
    })?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, &template)?;

    if cli_input.format == OutputFormat::Table {
        println!("reset config to defaults at {}", path.display());
        return Ok(());
    }
    let payload = ConfigResetPayload {
        path: path.display().to_string(),
    };
    emit_success_payload("config", &payload, cli_input.format)
}

#[allow(clippy::too_many_arguments)]
fn build_stage_snapshots(
    env_map: &HashMap<String, String>,
    cli_overrides: &CliOverrides,
    home_dir: &Path,
    explicit_config_path: Option<&Path>,
    project_config_path: Option<&Path>,
    user_config_path: Option<&Path>,
) -> SearchResult<Vec<ConfigStageSnapshot>> {
    let empty_env = HashMap::new();
    let empty_cli = CliOverrides::default();
    let defaults = load_from_str(None, None, &empty_env, &empty_cli, home_dir)?;

    let mut named_results: Vec<(&'static str, ConfigLoadResult)> = vec![("defaults", defaults)];
    if let Some(path) = explicit_config_path {
        named_results.push((
            "config",
            load_from_sources(Some(path), &empty_env, &empty_cli, home_dir)?,
        ));
        named_results.push((
            "env",
            load_from_sources(Some(path), env_map, &empty_cli, home_dir)?,
        ));
        named_results.push((
            "cli",
            load_from_sources(Some(path), env_map, cli_overrides, home_dir)?,
        ));
    } else {
        named_results.push((
            "user",
            load_from_sources(user_config_path, &empty_env, &empty_cli, home_dir)?,
        ));
        named_results.push((
            "project",
            load_from_layered_sources(
                project_config_path,
                user_config_path,
                &empty_env,
                &empty_cli,
                home_dir,
            )?,
        ));
        named_results.push((
            "env",
            load_from_layered_sources(
                project_config_path,
                user_config_path,
                env_map,
                &empty_cli,
                home_dir,
            )?,
        ));
        named_results.push((
            "cli",
            load_from_layered_sources(
                project_config_path,
                user_config_path,
                env_map,
                cli_overrides,
                home_dir,
            )?,
        ));
    }

    let mut snapshots: Vec<ConfigStageSnapshot> = Vec::with_capacity(named_results.len());
    for (idx, (name, result)) in named_results.into_iter().enumerate() {
        let values = flatten_config_values(&result.config)?;
        let changed_keys = if idx == 0 {
            values.keys().cloned().collect()
        } else {
            changed_keys(&snapshots[idx - 1].values, &values)
        };
        snapshots.push(ConfigStageSnapshot {
            name,
            values,
            changed_keys,
        });
    }

    Ok(snapshots)
}

fn flatten_config_values(config: &FsfsConfig) -> SearchResult<BTreeMap<String, Value>> {
    let projected = serde_json::to_value(config).map_err(|source| SearchError::SubsystemError {
        subsystem: CONFIG_SUBSYSTEM,
        source: Box::new(io::Error::other(format!(
            "failed to project config to JSON value map: {source}"
        ))),
    })?;
    let mut out = BTreeMap::new();
    flatten_json_paths("", &projected, &mut out);
    Ok(out)
}

fn flatten_json_paths(prefix: &str, value: &Value, out: &mut BTreeMap<String, Value>) {
    if let Value::Object(map) = value {
        for (key, child) in map {
            let next_prefix = if prefix.is_empty() {
                key.clone()
            } else {
                format!("{prefix}.{key}")
            };
            if child.is_object() {
                flatten_json_paths(&next_prefix, child, out);
            } else {
                out.insert(next_prefix, child.clone());
            }
        }
        return;
    }
    if !prefix.is_empty() {
        out.insert(prefix.to_owned(), value.clone());
    }
}

fn changed_keys(
    previous: &BTreeMap<String, Value>,
    current: &BTreeMap<String, Value>,
) -> HashSet<String> {
    current
        .iter()
        .filter_map(|(key, value)| {
            if previous.get(key) == Some(value) {
                None
            } else {
                Some(key.clone())
            }
        })
        .collect()
}

fn source_for_key<'a>(
    key: &str,
    final_value: &Value,
    snapshots: &'a [ConfigStageSnapshot],
) -> &'a str {
    for snapshot in snapshots.iter().rev() {
        if snapshot.changed_keys.contains(key)
            && snapshot
                .values
                .get(key)
                .is_some_and(|value| value == final_value)
        {
            return snapshot.name;
        }
    }
    "defaults"
}

fn configured_precedence(has_explicit_config: bool) -> Vec<String> {
    if has_explicit_config {
        vec![
            "cli".into(),
            "env".into(),
            "config".into(),
            "defaults".into(),
        ]
    } else {
        vec![
            "cli".into(),
            "env".into(),
            "project".into(),
            "user".into(),
            "defaults".into(),
        ]
    }
}

fn config_file_summary(
    loaded: &ConfigLoadResult,
    explicit_config_path: Option<&Path>,
    project_config_path: Option<&Path>,
    user_config_path: Option<&Path>,
) -> ConfigFileSummary {
    ConfigFileSummary {
        explicit: explicit_config_path.map(|path| path.display().to_string()),
        project: project_config_path.map(|path| path.display().to_string()),
        user: user_config_path.map(|path| path.display().to_string()),
        active: loaded
            .config_file_used
            .as_ref()
            .map(|path| path.display().to_string()),
    }
}

fn emit_success_payload<T: Serialize>(
    command: &str,
    payload: &T,
    format: OutputFormat,
) -> SearchResult<()> {
    let envelope = OutputEnvelope::success(
        payload,
        meta_for_format(command, format),
        iso_timestamp_now(),
    );
    let mut stdout = std::io::stdout();
    emit_envelope(&envelope, format, &mut stdout)?;
    if format != OutputFormat::Jsonl {
        stdout
            .write_all(b"\n")
            .map_err(|source| SearchError::SubsystemError {
                subsystem: CONFIG_SUBSYSTEM,
                source: Box::new(source),
            })?;
    }
    Ok(())
}

fn apply_cli_env_overrides(
    cli_input: &mut CliInput,
    env_map: &HashMap<String, String>,
) -> SearchResult<()> {
    if !cli_input.no_color
        && let Some(no_color) = parse_env_bool(env_map, "FRANKENSEARCH_NO_COLOR", "FSFS_NO_COLOR")?
    {
        cli_input.no_color = no_color;
    }

    if !cli_input.verbose
        && !cli_input.quiet
        && let Some(verbose) = parse_env_bool(env_map, "FRANKENSEARCH_VERBOSE", "FSFS_VERBOSE")?
    {
        cli_input.verbose = verbose;
    }

    Ok(())
}

fn parse_env_bool(
    env_map: &HashMap<String, String>,
    canonical: &'static str,
    legacy: &'static str,
) -> SearchResult<Option<bool>> {
    let Some((name, value)) = env_map
        .get(canonical)
        .map(|value| (canonical, value))
        .or_else(|| env_map.get(legacy).map(|value| (legacy, value)))
    else {
        return Ok(None);
    };
    let parsed = parse_bool_token(value).ok_or_else(|| SearchError::InvalidConfig {
        field: name.to_owned(),
        value: value.clone(),
        reason: "expected boolean (true/false/1/0/yes/no/on/off)".to_owned(),
    })?;
    Ok(Some(parsed))
}

fn parse_bool_token(value: &str) -> Option<bool> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn format_value_for_table(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_owned())
}

fn iso_timestamp_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format_epoch_secs_utc(secs)
}

fn format_epoch_secs_utc(secs: u64) -> String {
    let days_since_epoch = secs / 86_400;
    let time_of_day = secs % 86_400;
    let hours = time_of_day / 3_600;
    let minutes = (time_of_day % 3_600) / 60;
    let seconds = time_of_day % 60;
    let (year, month, day) = epoch_days_to_ymd(days_since_epoch);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

const fn epoch_days_to_ymd(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::{
        CliCommand, ConfigAction, SearchError, allow_missing_explicit_config,
        apply_cli_env_overrides, expand_cli_config_path, parse_bool_token, run_config_init_command,
        run_config_reset_command, run_config_set_command,
    };
    use frankensearch_fsfs::CliInput;
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;
    use toml::Value;

    #[test]
    fn expand_cli_config_path_expands_tilde_prefix() {
        let expanded =
            expand_cli_config_path(Path::new("~/cfg/fsfs.toml"), Path::new("/home/alex"));
        assert_eq!(expanded, PathBuf::from("/home/alex/cfg/fsfs.toml"));
    }

    #[test]
    fn expand_cli_config_path_keeps_absolute_paths() {
        let expanded = expand_cli_config_path(Path::new("/tmp/fsfs.toml"), Path::new("/home/alex"));
        assert_eq!(expanded, PathBuf::from("/tmp/fsfs.toml"));
    }

    #[test]
    fn expand_cli_config_path_bare_tilde() {
        let expanded = expand_cli_config_path(Path::new("~"), Path::new("/home/alex"));
        assert_eq!(expanded, PathBuf::from("/home/alex"));
    }

    #[test]
    fn expand_cli_config_path_keeps_relative_paths() {
        let expanded =
            expand_cli_config_path(Path::new("relative/config.toml"), Path::new("/home/alex"));
        assert_eq!(expanded, PathBuf::from("relative/config.toml"));
    }

    #[test]
    fn parse_bool_token_accepts_standard_values() {
        assert_eq!(parse_bool_token("true"), Some(true));
        assert_eq!(parse_bool_token("on"), Some(true));
        assert_eq!(parse_bool_token("0"), Some(false));
        assert_eq!(parse_bool_token("No"), Some(false));
        assert_eq!(parse_bool_token("bad"), None);
    }

    #[test]
    fn cli_env_overrides_apply_when_flags_absent() {
        let mut input = CliInput::default();
        let env = HashMap::from([
            ("FRANKENSEARCH_NO_COLOR".to_owned(), "1".to_owned()),
            ("FRANKENSEARCH_VERBOSE".to_owned(), "true".to_owned()),
        ]);
        apply_cli_env_overrides(&mut input, &env).expect("apply env");
        assert!(input.no_color);
        assert!(input.verbose);
    }

    #[test]
    fn config_set_creates_new_file_when_missing() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("fsfs.toml");
        let cli = CliInput {
            format: frankensearch_fsfs::OutputFormat::Json,
            ..CliInput::default()
        };

        run_config_set_command(&cli, Some(path.as_path()), None, None, "search.rrf_k", "75")
            .expect("config set");

        let contents = std::fs::read_to_string(&path).expect("read config");
        let parsed: Value = toml::from_str(&contents).expect("parse toml");
        assert_eq!(
            parsed
                .get("search")
                .and_then(|v| v.get("rrf_k"))
                .and_then(Value::as_integer),
            Some(75)
        );
    }

    #[test]
    fn config_set_accepts_whitespace_only_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("fsfs.toml");
        std::fs::write(&path, "  \n\t").expect("write whitespace file");
        let cli = CliInput {
            format: frankensearch_fsfs::OutputFormat::Json,
            ..CliInput::default()
        };

        run_config_set_command(
            &cli,
            Some(path.as_path()),
            None,
            None,
            "search.fast_only",
            "true",
        )
        .expect("config set");

        let contents = std::fs::read_to_string(&path).expect("read config");
        let parsed: Value = toml::from_str(&contents).expect("parse toml");
        assert_eq!(
            parsed
                .get("search")
                .and_then(|v| v.get("fast_only"))
                .and_then(Value::as_bool),
            Some(true)
        );
    }

    #[test]
    fn config_init_rejects_directory_path() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("config-dir");
        std::fs::create_dir_all(&path).expect("mkdir config-dir");
        let cli = CliInput {
            format: frankensearch_fsfs::OutputFormat::Json,
            ..CliInput::default()
        };

        let err = run_config_init_command(&cli, Some(path.as_path()), None, None)
            .expect_err("init should reject directory path");
        assert!(
            matches!(
                err,
                SearchError::InvalidConfig {
                    ref field,
                    ..
                } if field == "config.init.path"
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn config_set_rejects_directory_path() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("config-dir");
        std::fs::create_dir_all(&path).expect("mkdir config-dir");
        let cli = CliInput {
            format: frankensearch_fsfs::OutputFormat::Json,
            ..CliInput::default()
        };

        let err = run_config_set_command(
            &cli,
            Some(path.as_path()),
            None,
            None,
            "search.fast_only",
            "true",
        )
        .expect_err("set should reject directory path");
        assert!(
            matches!(
                err,
                SearchError::InvalidConfig { ref field, .. } if field == "config.path"
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn config_reset_rejects_directory_path() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("config-dir");
        std::fs::create_dir_all(&path).expect("mkdir config-dir");
        let cli = CliInput {
            format: frankensearch_fsfs::OutputFormat::Json,
            ..CliInput::default()
        };

        let err = run_config_reset_command(&cli, Some(path.as_path()), None, None)
            .expect_err("reset should reject directory path");
        assert!(
            matches!(
                err,
                SearchError::InvalidConfig { ref field, .. } if field == "config.path"
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn allow_missing_explicit_config_for_mutating_actions() {
        assert!(allow_missing_explicit_config(
            CliCommand::Config,
            Some(&ConfigAction::Init)
        ));
        assert!(allow_missing_explicit_config(
            CliCommand::Config,
            Some(&ConfigAction::Reset)
        ));
        assert!(allow_missing_explicit_config(
            CliCommand::Config,
            Some(&ConfigAction::Set {
                key: "search.rrf_k".to_owned(),
                value: "75".to_owned(),
            })
        ));
    }

    #[test]
    fn allow_missing_explicit_config_rejects_read_actions() {
        assert!(!allow_missing_explicit_config(CliCommand::Config, None));
        assert!(!allow_missing_explicit_config(
            CliCommand::Config,
            Some(&ConfigAction::Show)
        ));
        assert!(!allow_missing_explicit_config(
            CliCommand::Config,
            Some(&ConfigAction::Get {
                key: "search.rrf_k".to_owned(),
            })
        ));
        assert!(!allow_missing_explicit_config(
            CliCommand::Config,
            Some(&ConfigAction::Validate)
        ));
        assert!(!allow_missing_explicit_config(
            CliCommand::Search,
            Some(&ConfigAction::Init)
        ));
    }
}
