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

use std::collections::BTreeMap;
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
        // The file-tree form, which states the same two numbers in the other
        // order and without backticks:
        //
        //     ├── persistence_invariants.rs        740   12 tests
        //
        // Reading only the heading form left this free to drift on its own, and
        // a figure stated twice with one occurrence checked is worse than one
        // stated once: the checked copy makes the unchecked one look defended.
        let stripped = line
            .trim_start_matches(['├', '└', '│', '─', ' '])
            .trim_start();
        if let Some((name, tail)) = stripped.split_once(".rs") {
            let mut fields = tail.split_whitespace();
            if let (Some(lines), Some(count), Some("tests")) =
                (fields.next(), fields.next(), fields.next())
            {
                let parse = |s: &str| s.replace(',', "").parse::<usize>().ok();
                if let (Some(l), Some(t)) = (parse(lines), parse(count)) {
                    found.push((format!("{name}.rs"), t, l));
                }
            }
        }

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

/// The README says nothing in `aether-core` or `aether-lang` calls the GPU
/// backend. That must stay true, or stop being said.
///
/// It is the most prominent limitation on the front page and the one a reader is
/// most likely to act on: it decides whether the topology engine is a CPU library
/// or a GPU-accelerated one. Every other claim this file binds is a count; this
/// one is about the architecture, and it would go false the moment somebody wired
/// the backend in — silently, because the wiring is the interesting part and the
/// sentence three thousand words away is not.
///
/// # Why it reads the manifests rather than grepping
///
/// `grep -rn aether.gpu crates/aether-core/src` returns five hits today, all of
/// them doc comments naming a reproduction command, and
/// `crates/aether-lang/Cargo.toml` contains the string in a comment recording
/// that its phantom `wgpu` dependency was deleted. A text search over this
/// repository finds prose, which is how a check for this was briefly reported as
/// failing when it is not.
///
/// A dependency edge is the thing that decides it, so the dependency sections are
/// what get read.
#[test]
fn no_cpu_crate_depends_on_the_gpu_backend() {
    let root = repo_root();
    let mut offenders = Vec::new();
    let mut checked = 0;

    // Derived from the workspace, not a list of names.
    //
    // Four were named here at first — the crates that existed and mattered when
    // it was written — which leaves any crate added afterwards outside the check
    // until somebody remembers to add it, and the whole point is catching a
    // dependency nobody announced. `aegis-core` and `aegis-cli` were already
    // outside it on the day it was written; adding the edge to `aegis-core` now
    // fails this test and did not before.
    //
    // The count assertion below covers a member that exists but does not live
    // under `crates/`. A member named in the workspace with no manifest at all is
    // not this test's to catch — Cargo refuses to load the workspace and nothing
    // here runs, which was checked rather than assumed.
    let workspace = fs::read_to_string(root.join("Cargo.toml")).expect("workspace Cargo.toml");
    let members: Vec<String> = workspace
        .lines()
        .skip_while(|l| !l.trim_start().starts_with("members"))
        .skip(1)
        .take_while(|l| !l.contains(']'))
        .filter_map(|l| {
            l.trim()
                .trim_matches(|c| c == '"' || c == ',')
                .rsplit('/')
                .next()
        })
        .map(str::to_string)
        .filter(|name| name != "aether-gpu")
        .collect();

    assert!(
        members.len() >= 5,
        "parsed only {} workspace members, so the member list is not being read \
         and this check covers less than the workspace",
        members.len()
    );

    for crate_name in &members {
        let manifest = root.join("crates").join(crate_name).join("Cargo.toml");
        let Ok(src) = fs::read_to_string(&manifest) else {
            continue;
        };
        checked += 1;

        let mut in_dependency_section = false;
        for line in src.lines() {
            let t = line.trim();

            if t.starts_with('[') {
                in_dependency_section = t.contains("dependencies");
                continue;
            }
            // Comments are not dependencies, which is the entire point.
            if !in_dependency_section || t.starts_with('#') || t.is_empty() {
                continue;
            }
            if t.split('#').next().unwrap_or(t).contains("aether-gpu") {
                offenders.push(format!("{crate_name}: {t}"));
            }
        }
    }

    assert!(
        checked >= 3,
        "only {checked} manifests were read, so this check is looking at less of \
         the workspace than it claims"
    );

    assert!(
        offenders.is_empty(),
        "these crates now depend on aether-gpu:\n  {}\nThe README states that \
         nothing in aether-core or aether-lang calls the GPU backend, in the \
         section a reader reaches first. If the integration was made, that \
         sentence and the status table beside it are now false and need changing \
         with it.",
        offenders.join("\n  ")
    );
}

#[test]
fn the_readme_suite_counts_match_the_suites() {
    let root = repo_root();
    let doc = fs::read_to_string(root.join("README.md")).expect("README.md");
    let tests_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");

    let mut checked = 0;
    let mut seen: Vec<String> = Vec::new();

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
        seen.push(name.clone());
    }

    // Every documented suite must be found twice, and the requirement is derived
    // rather than stated.
    //
    // This was a hardcoded floor of ten — five suites, two forms each — which is
    // right only while there are five. A sixth documented file would raise what
    // should be checked without raising what is required, and the guard would
    // pass with the new file's tree row unread. Deriving the count per file
    // scales with the document instead of with a number someone remembered to
    // update.
    //
    // Counting files rather than claims is also why this cannot demand two per
    // *test file*: this crate has ten test files and the README describes five of
    // them, so a requirement keyed on the directory would fail on the five it
    // deliberately does not document.
    let mut per_file: BTreeMap<String, usize> = BTreeMap::new();
    for name in &seen {
        *per_file.entry(name.clone()).or_default() += 1;
    }

    assert!(
        !per_file.is_empty(),
        "no suite claims matched this crate at all, so this guard passes by \
         checking nothing"
    );

    for (name, count) in &per_file {
        assert!(
            *count >= 2,
            "README.md states tests/{name} {count} time(s). Each documented suite \
             appears as a section heading and as a file-tree row, so one \
             occurrence means the other has been reworded and is no longer read — \
             which leaves a number that looks defended and is not."
        );
    }

    assert_eq!(
        checked,
        seen.len(),
        "internal: every matched claim should have been recorded"
    );
}
