//! Checks that FEATURES.md's headline counts match the repository.
//!
//! The summary at the top of that file states a test count and a mutant count.
//! Both were accurate when written and neither is bound to anything, so the
//! first change that adds a test or a mutant makes the document quietly wrong —
//! which is the failure mode this repository documents at length in every other
//! context and had not guarded here.
//!
//! Only the mechanically checkable claims are covered. "The backend works and
//! nothing uses it" is a statement about intent that no assertion can verify,
//! and pretending otherwise would be worse than leaving it unchecked.
//!
//! These tests need no GPU: they read files.

use std::fs;
use std::path::PathBuf;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn features_md() -> String {
    fs::read_to_string(crate_root().join("FEATURES.md")).expect("FEATURES.md is missing")
}

/// Count `#[test]` across every integration test in the crate, this file
/// included.
///
/// An earlier version excluded this file, which made the documented count
/// three lower than the crate's actual test count -- a document check that
/// itself needed a footnote to be true.
///
/// Counting attributes rather than running the suite keeps this cheap and,
/// more importantly, makes it work without an adapter — the hardware tests are
/// `#[ignore]`d in that case and would not appear in a run's totals.
fn actual_test_count() -> usize {
    let dir = crate_root().join("tests");
    let mut total = 0;

    for entry in fs::read_dir(&dir).expect("tests/ is missing") {
        let path = entry.expect("readable entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let src = fs::read_to_string(&path).expect("readable test file");
        total += src
            .lines()
            .filter(|l| l.trim_start().starts_with("#[test]"))
            .count();
    }

    total
}

/// Count the mutants declared in the harness.
fn actual_mutant_count() -> usize {
    let src = fs::read_to_string(crate_root().join("mutants.sh")).expect("mutants.sh is missing");

    // Entries are `"name|perl expression"` lines inside the `mutants=(...)`
    // array. Counting quoted lines containing a pipe is specific enough to
    // avoid the comments and the shell around them.
    src.lines()
        .map(str::trim)
        .filter(|l| l.starts_with('"') && l.contains('|') && l.ends_with('"'))
        .count()
}

#[test]
fn the_documented_test_count_matches_the_tests_that_exist() {
    let doc = features_md();
    let actual = actual_test_count();

    let claimed = doc
        .split_whitespace()
        .collect::<Vec<_>>()
        .windows(2)
        .find_map(|w| {
            if w[1].starts_with("tests") {
                w[0].trim_end_matches(',').parse::<usize>().ok()
            } else {
                None
            }
        })
        .expect("FEATURES.md states no test count");

    assert_eq!(
        claimed, actual,
        "FEATURES.md claims {claimed} tests; {actual} exist. Update the summary, \
         or the document is wrong in exactly the way this repository spends its \
         README warning about."
    );
}

/// The documented mutant denominator must match the harness.
///
/// This binds the count and deliberately not the result. An earlier version
/// asserted the literal string "0 of N mutants", which made the document's only
/// legal statement the flattering one: adding a mutant and honestly recording
/// that it escaped would fail the test, and the cheapest way to green would be
/// to claim a clean sweep that had not been run. A guard that can only be
/// satisfied by a particular answer is not checking the answer.
///
/// What must hold is that the denominator describes the harness that exists, so
/// a mutant added without re-running cannot hide behind a stale total.
#[test]
fn the_documented_mutant_count_matches_the_harness() {
    let doc = features_md();
    let actual = actual_mutant_count();

    let denominator = doc
        .split_whitespace()
        .collect::<Vec<_>>()
        .windows(2)
        .find_map(|w| {
            if w[1].starts_with("mutants") {
                w[0].parse::<usize>().ok()
            } else {
                None
            }
        })
        .expect("FEATURES.md states no mutant count");

    assert_eq!(
        denominator, actual,
        "FEATURES.md reports against {denominator} mutants; mutants.sh declares \
         {actual}. A mutant was added without re-running the harness, or the \
         summary was not updated after it was."
    );
}

/// Every command the provenance table offers must name a binary or test target
/// that exists.
///
/// The table's whole purpose is telling a reader which figures are self-checking
/// and which need re-running, so a row pointing at a deleted example is worse
/// than no row: it promises reproducibility that is not there. The commands
/// themselves are not executed — several need an adapter and one needs thirty
/// runs — but the targets they name are checked to exist.
#[test]
fn the_provenance_table_points_at_targets_that_exist() {
    let doc = features_md();

    assert!(
        doc.contains("Which numbers here are checked, and which are snapshots"),
        "the provenance table is gone from FEATURES.md"
    );

    let examples = crate_root().join("examples");
    let tests = crate_root().join("tests");

    for named in ["gpu_bench", "tensor_crossover", "teardown_repro"] {
        assert!(
            doc.contains(named),
            "FEATURES.md no longer references the {named} example"
        );
        assert!(
            examples.join(format!("{named}.rs")).exists(),
            "FEATURES.md points at examples/{named}.rs, which does not exist"
        );
    }

    for named in ["features_doc", "gradcheck", "f32_topology"] {
        assert!(
            tests.join(format!("{named}.rs")).exists(),
            "FEATURES.md points at tests/{named}.rs, which does not exist"
        );
    }
}

/// The withdrawn-claims table is the part most likely to rot, since it is
/// maintained by hand and only grows when someone remembers.
///
/// This cannot verify that the table is *complete* — no test knows what was
/// retracted and not written down. It verifies that the table exists and still
/// has entries, which catches the failure where a later edit removes it while
/// tidying, and is honest about being nothing more than that.
#[test]
fn the_withdrawn_claims_table_is_still_present() {
    let doc = features_md();

    assert!(
        doc.contains("Conclusions this document reached and then withdrew"),
        "the withdrawn-claims section is gone from FEATURES.md"
    );

    let withdrawn = doc.matches("**withdrawn**").count();
    assert!(
        withdrawn >= 4,
        "only {withdrawn} withdrawn claims are listed; the table has been \
         trimmed, and a corrections log that loses its corrections is worse \
         than none"
    );
}
