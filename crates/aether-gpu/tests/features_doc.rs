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

/// Count the compute entry points in the shader.
///
/// Added because the documented figure was wrong. It read 18 against an actual
/// 20, having been updated by adding four to a remembered fourteen when the
/// number at the time was sixteen — arithmetic on a count instead of a count.
/// Unlike the test and mutant totals beside it, nothing had ever checked it.
fn actual_kernel_count() -> usize {
    let src = fs::read_to_string(crate_root().join("src/shaders.wgsl")).expect("shaders.wgsl");
    src.lines().filter(|l| l.starts_with("@compute")).count()
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

/// The documented kernel count must match the shader.
#[test]
fn the_documented_kernel_count_matches_the_shader() {
    let doc = features_md();
    let actual = actual_kernel_count();

    let claimed = doc
        .split_whitespace()
        .collect::<Vec<_>>()
        .windows(2)
        .find_map(|w| {
            if w[1] == "WGSL" {
                w[0].parse::<usize>().ok()
            } else {
                None
            }
        })
        .expect("FEATURES.md states no kernel count");

    assert_eq!(
        claimed, actual,
        "FEATURES.md claims {claimed} WGSL kernels; {actual} @compute entry \
         points exist. This one drifted by being incremented rather than \
         counted, which is why it is now counted."
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

/// Every kernel count in the repository README must match the shader too.
///
/// The previous change corrected that number in four places and recorded that
/// binding it would mean parsing prose in a 2,500-line document. That is true of
/// the *test* counts, which appear in sentences with no reliable shape. It is not
/// true of this one: "N WGSL kernels" is a fixed phrase that occurs exactly where
/// the claim is made, so a narrow pattern binds it without reading anything else.
///
/// Worth binding because it is the number that drifted twice — once in
/// FEATURES.md by being incremented rather than counted, and once in the README
/// by being left behind entirely. Every occurrence is checked rather than the
/// first, since the README stated it in two places and updating one of them is
/// the obvious way to fix this badly.
///
/// Reaching out of the crate for a repository file is a coupling, and a
/// deliberate one: the alternative is the front page disagreeing with the code it
/// describes, which is the more expensive failure.
#[test]
fn every_kernel_count_in_the_readme_matches_the_shader() {
    let readme = crate_root().join("../../README.md");
    let doc = fs::read_to_string(&readme)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", readme.display()));
    let actual = actual_kernel_count();

    let words: Vec<&str> = doc.split_whitespace().collect();
    let mut found = 0;
    for pair in words.windows(2) {
        if pair[1] != "WGSL" {
            continue;
        }
        if let Ok(claimed) = pair[0].parse::<usize>() {
            found += 1;
            assert_eq!(
                claimed, actual,
                "README.md claims {claimed} WGSL kernels; {actual} @compute entry \
                 points exist"
            );
        }
    }

    assert!(
        found > 0,
        "README.md states no kernel count, so this test now passes by checking \
         nothing — the phrase it binds has been reworded and the guard needs \
         following"
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
