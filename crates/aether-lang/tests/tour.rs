//! The program that opens the README, run as written.
//!
//! The README quotes `examples/tour.aegis` and its output. This binds both the
//! quoted source (it must be the file, byte for byte) and the quoted results,
//! so the first thing a reader sees cannot drift from what the language does.

use aether_lang::interpreter::Value;
use aether_lang::{Interpreter, Parser};
use std::fs;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn num(interp: &Interpreter, name: &str) -> f64 {
    match &interp.variables[name] {
        Value::Num(n) => *n,
        other => panic!("{name} is not a number: {other:?}"),
    }
}

#[test]
fn the_readme_tour_runs_and_says_what_the_readme_says() {
    let root = repo_root();
    let src = fs::read_to_string(root.join("examples/tour.aegis")).expect("examples/tour.aegis");
    let readme = fs::read_to_string(root.join("README.md")).expect("README.md");
    assert!(
        readme
            .replace("\r\n", "\n")
            .contains(&src.replace("\r\n", "\n")),
        "the README's opening program is no longer examples/tour.aegis verbatim"
    );

    let program = Parser::new(&src).parse().expect("tour parses");
    let mut interp = Interpreter::new();
    interp.execute(&program).expect("tour runs");

    // Shape: the delay-embedded sine is one piece with one loop at radius 1.
    assert_eq!(num(&interp, "r"), 1.0);
    // Certainty: the hoop is certified through the square.
    match &interp.variables["link"] {
        Value::Record(fields) => {
            assert!(matches!(&fields["verdict"], Value::Str(v) if v == "linked"));
            assert!(matches!(fields["lk"], Value::Num(lk) if lk == 1.0));
        }
        other => panic!("link is not a record: {other:?}"),
    }
    // Proof as a stopping rule: undetermined at 2, 1 and 0.5, proven at 0.25.
    assert_eq!(num(&interp, "radius"), 0.25);
    for line in [
        "[one piece at radius, 1, betti, [1, 1, 0]]",
        "[linked?, linked, lk, 1, error bound, 7.072564457345712e-14]",
        "[proven at radius, 0.25]",
    ] {
        assert!(readme.contains(line), "README no longer quotes `{line}`");
    }
}
