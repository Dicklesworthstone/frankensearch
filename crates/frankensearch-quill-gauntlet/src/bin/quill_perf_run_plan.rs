//! `quill-perf-run-plan` — render the human operator run plan from the
//! normative performance manifest and the canonical cell matrix
//! (bd-quill-e8-perf-doctrine-x4e4.15).
//!
//! Prints the generated Markdown to stdout. The committed document
//! (`docs/contracts/quill-perf-gates.run-plan.md`) is kept honest by the
//! `perf_run_plan_document_matches_the_manifest` drift test; regenerate it
//! deliberately with `QUILL_PERF_RUN_PLAN_UPDATE=1 cargo test -p
//! frankensearch-quill-gauntlet perf_run_plan_document_matches_the_manifest`.

use std::io::Write as _;

fn main() {
    let rendered = match frankensearch_quill_gauntlet::render_perf_run_plan_markdown() {
        Ok(rendered) => rendered,
        Err(error) => {
            eprintln!("could not render the run plan from the manifest: {error}");
            std::process::exit(2);
        }
    };
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    if handle.write_all(rendered.as_bytes()).is_err() {
        std::process::exit(1);
    }
}
