//! Interactive, read-only live-search terminal frontend.
#![forbid(unsafe_code)]
#![recursion_limit = "512"]

#[cfg(unix)]
#[path = "../live_search_tui.rs"]
mod live_search_tui;

#[cfg(unix)]
fn main() {
    live_search_tui::entry();
}

#[cfg(not(unix))]
fn main() {
    eprintln!("fsfs-live requires the native Unix terminal backend");
    std::process::exit(2);
}
