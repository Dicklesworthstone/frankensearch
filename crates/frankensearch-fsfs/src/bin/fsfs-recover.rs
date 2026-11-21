//! Standalone entry point for the shared receipt-directed recovery command.
#![recursion_limit = "512"]
#![deny(unsafe_code)]

#[path = "../recovery_command.rs"]
mod recovery_command;

fn main() -> std::process::ExitCode {
    std::process::ExitCode::from(recovery_command::entry(
        std::env::args_os().skip(1).collect(),
    ))
}
