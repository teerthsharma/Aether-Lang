//! The README's per-suite counts must match the suites.
//!
//! The repository README carries a line like ``` `diagram_distance.rs` — 17
//! tests, 422 lines ``` for each test file in this crate, once in a section
//! heading and once in a file tree. Nothing checked either, and running the
//! commands the README offers found every one of the five line counts stale and
//! one test count wrong — 607 against 740, 1,124 against 1,255, 576 against 643,
//! 381 against 422, 206 against 231, and eleven persistence invariants against
//! twelve.
//!
//! All five drifted in the same direction, because files grow and nobody
//! recounts them. That is what a guard is for, and the correction was made by
//! hand instead, which is the process that produced the drift.
//!
//! # Two different bindings for two different quantities
//!
//! The test count is bound **exactly**. It moves a handful of times a year, it
//! moves for a reason, and a document claiming eleven tests where twelve exist is
//! wrong in a way a reader would act on.
//!
//! The line count is bound to **5%**. It moves on every commit that touches the
//! file, and an exact assertion would fail on almost every change — a test that
//! fails constantly gets its number pasted over without being read, or deleted.
//! 5% of the smallest file here is about 12 lines; the smallest drift this was
//! written to catch was 41 and the largest 133, so the band catches what it was
//! built for with a wide margin and tolerates ordinary editing.

use std::fs;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `#[test]` attributes, counted the way the other guards in this workspace
/// count them: lines that *are* the attribute, not lines that mention it.
///
/// A counter that matches the string anywhere finds its own documentation. That
/// has now happened twice in this repository — once to a `#[test]` counter and
/// once to a counter of `cfg_attr` ignore gates — so it is written the careful
/// way here the first time.
fn test_count(src: &str) -> usize {
    src.lines()
        .filter(|l| l.trim_start().starts_with("#[test]"))
        .count()
}

/// Every ``` `name.rs` — N tests, M lines ``` claim in the README.
///
/// `M` is parsed with thousands separators stripped. The scripted pass that made
/// the corrections by hand missed `attention_contracts.rs` precisely because its
/// count is written `1,255` and the pattern wanted bare digits — a rule over
/// formatted numbers skips the largest ones, which are the likeliest to have
/// moved.
fn claims(doc: &str) -> Vec<(String, usize, usize)> {
    let mut found = Vec::new();

    for line in doc.lines() {
        let mut rest = line;
        while let Some(start) = rest.find('`') {
            let after = &rest[start + 1..];
            let Some(end) = after.find('`') else { break };
            let name = &after[..end];
            rest = &after[end + 1..];

            if !name.ends_with(".rs") {
                continue;
            }

            // " — N tests, M lines" must follow immediately.
            let tail = rest.trim_start();
            let Some(tail) = tail.strip_prefix("— ") else {
                continue;
            };
            let Some((count, tail)) = tail.split_once(" tests, ") else {
                continue;
            };
            let Some((lines, _)) = tail.split_once(" lines") else {
                continue;
            };

            let parse = |s: &str| s.replace(',', "").parse::<usize>().ok();
            if let (Some(t), Some(l)) = (parse(count), parse(lines)) {
                found.push((name.to_string(), t, l));
            }
        }
    }

    found
}

#[test]
fn the_readme_suite_counts_match_the_suites() {
    let root = repo_root();
    let doc = fs::read_to_string(root.join("README.md")).expect("README.md");
    let tests_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");

    let mut checked = 0;

    for (name, claimed_tests, claimed_lines) in claims(&doc) {
        let path = tests_dir.join(&name);
        if !path.exists() {
            // The README describes files in other crates too. Those are not this
            // crate's to bind, and skipping them silently is right — but only
            // because the count of what *was* checked is asserted below.
            continue;
        }

        let src = fs::read_to_string(&path).expect("readable suite");
        let actual_tests = test_count(&src);
        let actual_lines = src.lines().count();

        assert_eq!(
            claimed_tests, actual_tests,
            "README.md says tests/{name} has {claimed_tests} tests; it has \
             {actual_tests}. This count appears in a section heading and in the \
             file tree, so both need the change."
        );

        let tolerance = actual_lines / 20;
        let drift = claimed_lines.abs_diff(actual_lines);
        assert!(
            drift <= tolerance,
            "README.md says tests/{name} is {claimed_lines} lines; it is \
             {actual_lines}, a drift of {drift} against a {tolerance}-line \
             tolerance. Update every occurrence, including any written with a \
             thousands separator."
        );

        checked += 1;
    }

    assert!(
        checked >= 5,
        "only {checked} suite claims were checked against this crate; the README \
         describes five of its test files, so a lower number means the phrasing \
         changed and this guard is now checking less than it appears to"
    );
}
