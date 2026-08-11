//! The two clippy steps in CI must lint the same code.
//!
//! One denies correctness and suspicious plus every rustc warning; the other
//! prints the style, complexity and perf groups the first silences, and never
//! fails. They exist as a pair because `-D warnings` promotes anything that warns
//! into an error, so a denied unused import and a printed `needless_range_loop`
//! cannot come from one invocation.
//!
//! Being a pair is exactly the fragility. A crate added to the gate and not to
//! the advisory is unlinted for style with nothing to say so, and the failure is
//! silent in the direction that matters: the gate stays green, the advisory keeps
//! printing, and the new crate is simply absent from a list nobody counts.
//!
//! This binds the two scopes together. It does not check that the scope is
//! *right* — only that one step cannot quietly stop covering what the other does.

use std::fs;
use std::path::PathBuf;

fn workflow() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.github/workflows/ci.yml");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// The package selection of every `cargo clippy` invocation in the workflow.
///
/// The command is written as a folded YAML scalar spread over several lines, so
/// the flags are gathered by continuing until a line that starts a new key or
/// list item. Reconstructing it from the `cargo clippy` line alone is how a
/// previous attempt to run CI's command locally produced a stricter invocation
/// than CI runs and a false report that the job was failing — the continuation
/// lines are the command.
fn clippy_scopes(doc: &str) -> Vec<String> {
    let mut scopes = Vec::new();
    let lines: Vec<&str> = doc.lines().collect();

    for (i, line) in lines.iter().enumerate() {
        if !line.contains("cargo clippy") {
            continue;
        }

        let mut selection: Vec<String> = Vec::new();
        for part in lines[i..].iter() {
            let t = part.trim();
            // A new step or key ends the folded scalar.
            if !selection.is_empty() && (t.starts_with("- ") || t.starts_with('#') || t.is_empty())
            {
                break;
            }
            for token in t.split_whitespace() {
                if token == "--workspace" || token == "--all-targets" {
                    selection.push(token.to_string());
                }
                if token.starts_with("--exclude") {
                    selection.push(token.to_string());
                }
            }
            // `--exclude foo` is two tokens; capture the value.
            if let Some(rest) = t.split("--exclude ").nth(1) {
                if let Some(pkg) = rest.split_whitespace().next() {
                    selection.push(format!("exclude:{pkg}"));
                }
            }
        }

        selection.sort();
        selection.dedup();
        scopes.push(selection.join(" "));
    }

    scopes
}

#[test]
fn both_clippy_steps_lint_the_same_packages() {
    let doc = workflow();
    let scopes = clippy_scopes(&doc);

    assert_eq!(
        scopes.len(),
        2,
        "expected two cargo clippy invocations in ci.yml, found {}. The gate and \
         the advisory step are a pair; a third would need its scope considered \
         here, and a missing one means the advisory was dropped and style lints \
         are silenced with nothing printing them.",
        scopes.len()
    );

    assert_eq!(
        scopes[0], scopes[1],
        "the two clippy steps in ci.yml lint different packages:\n  gate:     \
         {}\n  advisory: {}\nA crate covered by one and not the other is \
         unlinted for style with nothing to say so.",
        scopes[0], scopes[1]
    );

    assert!(
        !scopes[0].is_empty(),
        "neither clippy step names a package selection, so this test compares two \
         empty strings and passes by checking nothing — the flags moved out of the \
         folded scalar this parses"
    );
}

/// The advisory step must be unable to fail the build.
///
/// Its whole value is printing without gating; a lint promoted to
/// deny-by-default in some future nightly would otherwise turn it into a second
/// gate, and one nobody chose. `continue-on-error` says that in the workflow
/// rather than relying on clippy continuing to exit zero on warnings.
#[test]
fn the_advisory_clippy_step_cannot_fail_the_build() {
    let doc = workflow();

    // The step's `name:`, not the first line mentioning it. The gate's own
    // comment refers to "the advisory step below", so matching the word alone
    // finds prose four paragraphs above the thing being checked and reports it
    // missing — which is how this test failed the first time it ran.
    let advisory = doc
        .lines()
        .position(|l| {
            let t = l.trim();
            t.starts_with("- name:") && t.contains("advisory")
        })
        .expect("no step in ci.yml has a name marking it advisory");

    let window: String = doc
        .lines()
        .skip(advisory)
        .take(4)
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        window.contains("continue-on-error: true"),
        "the advisory clippy step does not set continue-on-error: true. Without \
         it the step gates on style lints, which is the opposite of what it is \
         for, and the change would show up as CI failing on a lint nobody chose \
         to enforce.\n{window}"
    );
}
