//! The `missing_docs` migration only moves forwards.
//!
//! `aether-core` had 215 undocumented public items when the sweep started, which
//! is too many to write in one pass: comments produced that way restate the
//! signature, and this repository has already had to delete one of those — a doc
//! asserting a GPU kernel was where offloading mattered most, written before the
//! benchmark that measured it as never worth doing and left standing after.
//!
//! So the lint goes on per module, behind finished work. That leaves two gaps
//! nothing else covers: a module can lose its attribute in a refactor and the
//! build stays green, and a new module arrives unguarded by default with no
//! record that it was never done rather than deliberately skipped.
//!
//! This is the ratchet. It fails when the count of enforced modules drops, so
//! finished work cannot regress, and it names the modules still outstanding so
//! the remaining job is a list rather than a rediscovery.

use std::fs;
use std::path::PathBuf;

/// Modules with `#![warn(missing_docs)]` today.
///
/// Raising this is a deliberate act, which is the point: the number moves when
/// someone documents a module, and never because a lint attribute was dropped
/// while tidying.
const ENFORCED: usize = 7;

fn source_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Every `.rs` under `src`, recursively, since `ml/` is a subdirectory.
fn source_files(dir: &PathBuf, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("readable source directory") {
        let path = entry.expect("readable entry").path();
        if path.is_dir() {
            source_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

#[test]
fn the_documented_module_count_never_falls() {
    let mut files = Vec::new();
    source_files(&source_dir(), &mut files);
    files.sort();

    let mut enforced = Vec::new();
    let mut outstanding = Vec::new();

    for path in &files {
        let src = fs::read_to_string(path).expect("readable source");
        let name = path
            .strip_prefix(source_dir())
            .unwrap_or(path)
            .display()
            .to_string();

        if src.contains("#![warn(missing_docs)]") {
            enforced.push(name);
        } else {
            outstanding.push(name);
        }
    }

    assert!(
        enforced.len() >= ENFORCED,
        "{} modules enforce missing_docs, down from {ENFORCED}. A module lost its \
         attribute, which the build cannot notice because the lint it removed is \
         the thing that would have complained.\nEnforced: {enforced:?}",
        enforced.len()
    );

    // Not an equality check. Documenting a module and forgetting to raise the
    // constant should not fail — that is progress, and a test that punished it
    // would be an argument against doing the work.
    assert!(
        !outstanding.is_empty() || enforced.len() == files.len(),
        "bookkeeping error: no modules outstanding but the count disagrees"
    );

    println!(
        "  missing_docs enforced in {} of {} modules",
        enforced.len(),
        files.len()
    );
    for name in &outstanding {
        println!("    outstanding: {name}");
    }
}
