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
use std::path::{Path, PathBuf};

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

/// Total lines under `dir` in files with the given extension, skipping build
/// output.
///
/// `target` and `.lake` are skipped so the figure does not depend on whether the
/// tree has been built. Neither currently contains a source file of either
/// extension — `crates/` holds no nested `target` at all — so this changes no
/// count today and stops one changing under someone later.
fn count_lines(dir: &Path, ext: &str) -> usize {
    let mut total = 0;

    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => panic!("cannot read {}: {e}", dir.display()),
    };

    for entry in entries {
        let path = entry.expect("readable entry").path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        if path.is_dir() {
            if name == "target" || name == ".lake" {
                continue;
            }
            total += count_lines(&path, ext);
        } else if path.extension().and_then(|e| e.to_str()) == Some(ext) {
            total += fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
                .lines()
                .count();
        }
    }

    total
}

/// Every count the README states in the form `N Rust lines` or `N lines of Rust`,
/// as the parsed numbers.
///
/// Both phrasings are in use and binding only one leaves the other free to rot,
/// which is the failure this guards: the figure was corrected in one place and
/// left stale in two others.
///
/// The trailing `lines` is required rather than matching on the subject alone.
/// Without it the sentence "Windows 11, Rust nightly" parses as a claim of eleven
/// Rust lines, because the comma strip that makes `30,352` a number also makes
/// `11,` one. A guard that reads version strings as measurements fails for
/// reasons that have nothing to do with what it guards.
fn readme_counts(doc: &str, subject: &str) -> Vec<usize> {
    let parse = |s: &str| s.replace(',', "").parse::<usize>().ok();
    let mut found = Vec::new();

    // Tokenised per line, not across the whole document. The status dashboard
    // puts its numbers in a right-aligned column:
    //
    //     Rust lines (crates/)    34,456
    //     Lean lines (Aether/)    11,637
    //
    // so a document-wide token stream reads "34,456 Lean lines" across the line
    // break and binds the Rust figure to the Lean label. Every phrase this looks
    // for is written on one line, so refusing to match across one costs nothing
    // and removes a whole class of coincidence.
    for line in doc.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();

        for w in words.windows(3) {
            if w[1] == subject && w[2].starts_with("lines") {
                found.extend(parse(w[0]));
            }
        }
        for w in words.windows(4) {
            if w[1] == "lines" && w[2] == "of" && w[3] == subject {
                found.extend(parse(w[0]));
            }
        }
    }

    found
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

/// The README's line-count claims must stay within 5% of the tree they describe.
///
/// These are the only numbers on the front page with a documented reproduction
/// command and nothing checking them, and they rotted the furthest: the Rust
/// figure read 24,180 against an actual 30,352, off by 6,172 lines — 20% — while
/// the Lean figure beside it was still exact. Both were audited by hand, which is
/// how they got that far apart in the first place.
///
/// **Bound to a tolerance rather than to equality, deliberately.** Every other
/// guard in this file asserts a count exactly, and that is right for kernels,
/// mutants and tests, which change a handful of times a year. A line count
/// changes on every commit that adds a line. An exact assertion on one would fail
/// on almost every change, and a test that fails constantly is fixed by pasting a
/// new number in without looking or by deleting the test — both of which leave
/// the document less trustworthy than no guard at all.
///
/// 5% of the current figure is about 1,500 lines: wide enough that ordinary work
/// does not trip it, and narrow enough that the drift that prompted it would have
/// been caught roughly four times over. The tolerance is a limit on how stale the
/// claim may get, not a claim that the number is approximate — the exact figure
/// is still stated and still reproduced by the command in the claims table.
#[test]
fn the_readme_line_counts_have_not_rotted() {
    let repo = crate_root().join("../../");
    let doc = fs::read_to_string(repo.join("README.md")).expect("README.md");

    for (subject, dir, ext) in [
        ("Rust", repo.join("crates"), "rs"),
        ("Lean", repo.join("Aether"), "lean"),
    ] {
        let actual = count_lines(&dir, ext);
        let claims = readme_counts(&doc, subject);

        assert!(
            !claims.is_empty(),
            "README.md states no {subject} line count, so this test now passes by \
             checking nothing — the phrases it binds have been reworded and the \
             guard needs following"
        );

        let tolerance = actual / 20;
        for claimed in claims {
            let drift = claimed.abs_diff(actual);
            assert!(
                drift <= tolerance,
                "README.md claims {claimed} {subject} lines; {actual} exist, a \
                 drift of {drift} against a {tolerance}-line tolerance. Re-run the \
                 command in the claims table and update every occurrence, not the \
                 first one found."
            );
        }
    }
}

/// Every ignored-test count in the README must equal the number of gated tests.
///
/// The README states this figure in four places — the badge, the status
/// dashboard, the claims table, and the paragraph explaining what hardware-gated
/// means. All four drifted, twice, within one session: 38 became 23 became 76,
/// and each correction was made by running a command and pasting the result,
/// which is the process that produced the stale numbers in the first place.
///
/// Unlike the passed count this one is derivable. A test is ignored exactly when
/// it carries `#[cfg_attr(not(feature = "gpu"), ignore)]` and the feature is off,
/// so counting the attribute gives the number a run will report — checked against
/// both `cargo test -p aether-gpu` and the workspace run, which agree at 76
/// because every gated test lives in this crate.
///
/// The passed count is deliberately not bound. 301 `#[test]` attributes exist
/// outside `aether-kernel` against 291 reported, because some are behind `cfg`
/// gates that keep them out of the build entirely, and a guard that has to model
/// which ones would be a second implementation of Cargo's feature resolution.
#[test]
fn the_readme_ignored_count_matches_the_gated_tests() {
    let readme = crate_root().join("../../README.md");
    let doc = fs::read_to_string(&readme)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", readme.display()));

    let gated = {
        let dir = crate_root();
        let mut total = 0;
        for sub in ["tests", "src"] {
            let Ok(entries) = fs::read_dir(dir.join(sub)) else {
                continue;
            };
            for entry in entries {
                let path = entry.expect("readable entry").path();
                if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                    continue;
                }
                // Lines that *are* the attribute, not lines that mention it.
                //
                // The first version counted every occurrence of the string and
                // reported 78 against 76, because this test's own doc comment and
                // its failure message each contain one. That is the same defect
                // as a `#[test]` counter that counts the two `#[test]`s inside the
                // guard counting them, found in this file an hour earlier and
                // repeated here — a counter written by pattern-matching on the
                // thing it counts will always find itself.
                total += fs::read_to_string(&path)
                    .expect("readable source")
                    .lines()
                    .filter(|l| {
                        l.trim_start()
                            .starts_with(r#"#[cfg_attr(not(feature = "gpu"), ignore"#)
                    })
                    .count();
            }
        }
        total
    };

    assert!(
        gated > 0,
        "no gated tests found, so this guard passes by counting nothing — the \
         attribute's spelling has changed"
    );

    // Plain occurrences: "76 ignored".
    let words: Vec<&str> = doc.split_whitespace().collect();
    let mut found = 0;
    for pair in words.windows(2) {
        if !pair[1].starts_with("ignored") {
            continue;
        }
        // Skip the badge's percent-encoded form, handled separately below: its
        // digits run together with the encoding and parse to a different number.
        let Ok(claimed) = pair[0].trim_start_matches('`').parse::<usize>() else {
            continue;
        };
        found += 1;
        assert_eq!(
            claimed, gated,
            "README.md claims {claimed} ignored tests; {gated} carry \
             #[cfg_attr(not(feature = \"gpu\"), ignore)]. Re-run the gate and \
             update every occurrence, not the one that was noticed."
        );
    }

    assert!(
        found >= 3,
        "only {found} plain ignored-counts found in README.md; the figure appears \
         in the dashboard, the claims table and the hardware-gated paragraph, so \
         fewer than three means the phrasing changed and this guard is now \
         checking less than it reads as checking"
    );

    // The badge encodes its comma and space, so it is matched literally.
    assert!(
        doc.contains(&format!("%20{gated}%20ignored")),
        "the test badge does not encode {gated} ignored; it is the one place the \
         count appears that a reader sees without scrolling"
    );
}

/// The f32 decision brief must point at sections and tests that exist.
///
/// The brief consolidates evidence made in five other places so the integration
/// question can be decided from one table instead of five. Consolidating means
/// pointing, and a pointer is the thing in this document most able to rot without
/// looking wrong: a renamed heading leaves the brief reading exactly as it did,
/// with a reference that resolves to nothing.
///
/// This is the same check the provenance table gets, applied to the newer table
/// for the same reason — that one was added after a row was found naming a
/// deleted example.
#[test]
fn the_f32_decision_brief_points_at_sections_and_tests_that_exist() {
    let doc = features_md();

    assert!(
        doc.contains("The decision this leaves open, and the evidence for it in one place"),
        "the f32 decision brief is gone from FEATURES.md"
    );

    // Headings the brief sends a reader to. Each must still be a heading, not
    // merely a string that survives somewhere in the prose.
    for section in [
        "The f32 backward drives the same training outcome",
        "f32 gradient error grows about linearly",
        "Is f32 good enough for the topology",
    ] {
        let is_heading = doc
            .lines()
            .any(|l| l.trim_start().starts_with('#') && l.contains(section));
        assert!(
            is_heading,
            "the brief points at a section titled {section:?}, which is no longer \
             a heading in this file. The brief still reads correctly, which is why \
             this is checked rather than noticed."
        );
    }

    // Tests it cites by name, which carry the two numbers it quotes.
    let parity = fs::read_to_string(crate_root().join("tests/gpu_parity.rs"))
        .expect("gpu_parity.rs is missing");

    for test in [
        "per_entry_error_stays_inside_the_condition_number_bound",
        "the_condition_number_of_every_entry_costs_one_extra_matmul",
    ] {
        assert!(
            doc.contains(test),
            "the brief no longer cites {test}, so the row quoting its measurement \
             has lost its provenance"
        );
        assert!(
            parity.contains(&format!("fn {test}")),
            "the brief cites {test}, which no longer exists in tests/gpu_parity.rs"
        );
    }

    assert!(
        crate_root().join("tests/f32_topology.rs").exists(),
        "the brief names tests/f32_topology.rs as the source of the Betti result, \
         and it does not exist"
    );
}

/// The precision budget must quote tolerances the suites actually assert.
///
/// That table is the only place the crate says how much numerical error is
/// acceptable, and it decides whether a measured difference is rounding or a
/// defect — three mutation verdicts rest on it. It was assembled by reading
/// literals out of three test files, which is exactly the process that produced
/// every other stale number this file guards.
///
/// Each constant is checked to still exist in the file the table names. Line
/// numbers are deliberately not checked: they move whenever anything above them
/// is edited, so binding them would make this fail constantly for reasons that
/// have nothing to do with the tolerances, and a test that cries wolf gets its
/// numbers pasted over without reading.
///
/// This cannot verify a tolerance is *appropriate*, only that the document is
/// quoting the code rather than a memory of it.
#[test]
fn the_precision_budget_quotes_tolerances_that_exist() {
    let doc = features_md();

    assert!(
        doc.contains("The line between rounding and defect, written down"),
        "the precision budget is gone from FEATURES.md"
    );

    // (documented tolerance, file that must contain it)
    let claims = [
        ("const EPSILONS_ALLOWED: f32 = 8.0;", "gpu_parity.rs"),
        ("const TOL: f64 = 2e-4;", "attention_parity.rs"),
        ("worst <= 1e-4", "attention_parity.rs"),
        ("worst_vs_cpu < 1e-6", "f32_topology.rs"),
        ("worst_rel < 1e-5", "f32_topology.rs"),
    ];

    for (needle, file) in claims {
        let src = fs::read_to_string(crate_root().join("tests").join(file))
            .unwrap_or_else(|e| panic!("cannot read tests/{file}: {e}"));
        assert!(
            src.contains(needle),
            "FEATURES.md's precision budget quotes `{needle}` from tests/{file}, \
             which no longer contains it. Either the tolerance changed and the \
             budget is now describing a requirement nothing enforces, or it was \
             reworded and this guard needs following."
        );
    }
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

    // Each row's status must match the kind of command beside it.
    //
    // The table has three statuses and they are not interchangeable. **checked**
    // and **bounded** both mean a test decides the figure, so the row must offer
    // a way to run one; **snapshot** means nothing decides it, so the row offers
    // a program to re-measure with. A snapshot labelled bounded reads as stronger
    // than it is, and that is the mislabelling this catches — the statuses are
    // the only thing telling a reader which figures are defended.
    // `same` is the table's shorthand for the command in the row above, and the
    // first version of this check read it as a row offering no command at all —
    // failing on a clean tree against the mutant-count row. Reshaping the document
    // to suit the parser would have been the wrong repair; the shorthand is
    // legitimate and carrying it forward is what reading the table means.
    let mut rows = 0;
    let mut previous_command_ran_a_test = false;

    for line in doc.lines() {
        let t = line.trim();
        if !t.starts_with('|') || !t.contains("re-derive") && t.matches('|').count() < 4 {
            continue;
        }
        let has_status = ["**checked**", "**bounded**", "**snapshot**"]
            .iter()
            .find(|s| t.contains(**s));
        let Some(status) = has_status else { continue };
        rows += 1;

        let inherits = t.contains("| same |");
        let asserts = if inherits {
            previous_command_ran_a_test
        } else {
            t.contains("cargo test")
        };
        previous_command_ran_a_test = asserts;
        match *status {
            "**checked**" | "**bounded**" => assert!(
                asserts,
                "a row marked {status} offers no `cargo test` command, so nothing \
                 a reader can run decides the figure it claims is decided:\n  {t}"
            ),
            "**snapshot**" => assert!(
                !asserts || t.contains("cargo run"),
                "a row marked **snapshot** offers a `cargo test` command, which \
                 means a test does decide it and the row understates itself:\n  {t}"
            ),
            _ => unreachable!(),
        }
    }

    assert!(
        rows >= 8,
        "only {rows} provenance rows carried a recognised status, so this check \
         read fewer than the table holds and the statuses have been reworded"
    );
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
