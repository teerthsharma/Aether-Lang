//! Every source file in the workspace must be reachable from a module tree.
//!
//! `crates/aegis-core/src/ml/autograd.rs` is 298 lines that Cargo has never
//! compiled: `lib.rs` declares `memory` and nothing else, so no `mod` reaches it.
//! It is not compiled, not linted, not tested, and not covered by any gate here —
//! and it did worse than sit idle. An `use alloc::` inside it was read as evidence
//! that an `extern crate alloc` declaration was load-bearing, which produced a
//! wrong claim in a commit message and a comment asserting the opposite of the
//! truth. A file the compiler never reads still looks like code to a grep.
//!
//! Nothing else in this repository can see this. The compiler cannot warn about a
//! file it is never given, clippy lints what compiles, and a test suite exercises
//! what links. Orphaned sources are invisible to every check that works by
//! building.
//!
//! # What "reachable" means here
//!
//! Rust's real rule is a tree: the crate root declares modules, those declare
//! more, and a file is compiled when a chain of `mod` declarations reaches it.
//! Reproducing that faithfully means resolving `#[path]`, `cfg`-gated modules and
//! inline `mod x { }` blocks, which is most of a front end.
//!
//! This checks something weaker and stated: that some file in the crate contains a
//! `mod <stem>` declaration for each source file's stem. A file that is declared
//! but only inside a module that is itself orphaned would pass, and so would one
//! declared under `#[path]` with a different name. It catches a file nothing
//! mentions, which is the case that occurred and the one that hides best.

use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

#[test]
fn every_source_file_is_declared_as_a_module() {
    let root = workspace_root();
    let crates_dir = root.join("crates");

    let mut orphans = Vec::new();
    let mut checked = 0;

    let entries = fs::read_dir(&crates_dir).expect("crates/ is missing");
    for entry in entries.flatten() {
        let src = entry.path().join("src");
        if !src.is_dir() {
            continue;
        }

        let mut files = Vec::new();
        rust_files(&src, &mut files);

        // Every declaration anywhere in the crate, as one haystack. Cheap, and
        // enough for "does anything mention this file at all".
        let all: String = files
            .iter()
            .filter_map(|p| fs::read_to_string(p).ok())
            .collect::<Vec<_>>()
            .join("\n");

        for file in &files {
            let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("");

            // Crate roots and directory roots are reached by Cargo and by their
            // parent directory's declaration, not by a `mod` naming them.
            if matches!(stem, "lib" | "main" | "mod") {
                continue;
            }
            checked += 1;

            let declared = all.contains(&format!("mod {stem};"))
                || all.contains(&format!("mod {stem} "))
                || all.contains(&format!("mod {stem}\n"));

            if !declared {
                let shown = file
                    .strip_prefix(&root)
                    .unwrap_or(file)
                    .display()
                    .to_string()
                    .replace('\\', "/");
                let lines = fs::read_to_string(file)
                    .map(|s| s.lines().count())
                    .unwrap_or(0);
                orphans.push(format!("{shown} ({lines} lines)"));
            }
        }
    }

    assert!(
        checked >= 20,
        "only {checked} source files were examined, which is fewer than this \
         workspace holds — the crates/ layout has changed and this check is \
         looking at a fraction of it"
    );

    assert!(
        orphans.is_empty(),
        "these source files are not declared as modules anywhere, so Cargo never \
         compiles them:\n  {}\nA file outside the module tree is not built, not \
         linted, not tested, and still readable by a grep — which is how one of \
         them was once cited as evidence for a claim about code that does not \
         run.",
        orphans.join("\n  ")
    );
}
