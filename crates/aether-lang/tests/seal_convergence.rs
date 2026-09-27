//! `seal until convergence(eps)`, the loop the language was designed around,
//! and the two pieces of surface it needed to run as written: scientific
//! literals (`1e-6`) and parenthesised grouping.

use aether_lang::interpreter::Value;
use aether_lang::{Interpreter, Parser};

fn run(src: &str) -> Result<Interpreter, String> {
    let program = Parser::new(src).parse().map_err(|e| e.to_string())?;
    let mut interp = Interpreter::new();
    interp.execute(&program)?;
    Ok(interp)
}

fn num(interp: &Interpreter, name: &str) -> f64 {
    match &interp.variables[name] {
        Value::Num(n) => *n,
        other => panic!("{name} is not a number: {other:?}"),
    }
}

/// Newton's iteration for √2 from 1 reaches a fixed point to 1e-6 on its fifth
/// pass: the fourth moves by 2.1e-6, the fifth by 1.6e-12. The loop must stop
/// there, not one pass early (a bound checked before the body ran) and not at
/// the 1,000-pass ceiling (a condition never read).
#[test]
fn newton_sqrt2_seals_on_the_fifth_pass() {
    let interp = run("let x = 1~\n\
         let n = 0~\n\
         🦭 until convergence(1e-6) {\n\
             n = n + 1~\n\
             x = (x + 2 / x) / 2~\n\
             x~\n\
         }\n")
    .unwrap();
    assert_eq!(num(&interp, "n"), 5.0);
    assert!((num(&interp, "x") - 2f64.sqrt()).abs() < 1e-12);
}

/// A tolerance of zero demands an exact repeat, which the same iteration
/// reaches in finitely many passes because floating point has a fixed point:
/// 1.414213562373095, one ulp below the correctly rounded √2.
#[test]
fn zero_tolerance_waits_for_an_exact_repeat() {
    let interp = run("let x = 1~\n\
         let n = 0~\n\
         seal until convergence(0) {\n\
             n = n + 1~\n\
             x = (x + 2 / x) / 2~\n\
             x~\n\
         }\n")
    .unwrap();
    assert!(num(&interp, "n") > 5.0 && num(&interp, "n") < 1000.0);
    assert!((num(&interp, "x") - 2f64.sqrt()).abs() <= f64::EPSILON * 2f64.sqrt());
}

/// Lists converge in the max norm: one coordinate still moving keeps the loop
/// running even when the other has stopped.
#[test]
fn list_values_converge_in_the_max_norm() {
    let interp = run("let a = 0~\n\
         let b = 1~\n\
         let n = 0~\n\
         seal until convergence(0.001) {\n\
             n = n + 1~\n\
             b = b / 2~\n\
             let v = [a, b]~\n\
             v~\n\
         }\n")
    .unwrap();
    // b halves each pass: the change after pass n is 2^-n, first <= 0.001 at n = 10.
    assert_eq!(num(&interp, "n"), 10.0);
}

/// A body whose value has no distance is refused by name, never read as zero.
#[test]
fn a_non_numeric_body_is_refused() {
    let err = run("seal until convergence(0.1) {\n    let s = \"text\"~\n    s~\n}\n")
        .err()
        .expect("a string body has no distance");
    assert!(
        err.contains("convergence() needs a numeric body value"),
        "{err}"
    );
}

#[test]
fn a_negative_tolerance_is_refused() {
    let err = run("let x = 1~\nseal until convergence(-1) {\n    x~\n}\n")
        .err()
        .expect("negative tolerance");
    assert!(err.contains("non-negative tolerance"), "{err}");
}

/// Exponents are applied exactly in the literal's six-decimal fixed point.
#[test]
fn scientific_literals_scale_exactly() {
    let interp = run("let a = 1e-6~\nlet b = 2.5e3~\nlet c = 1.5E-2~\nlet d = 3e+2~\n").unwrap();
    assert_eq!(num(&interp, "a"), 0.000001);
    assert_eq!(num(&interp, "b"), 2500.0);
    assert_eq!(num(&interp, "c"), 0.015);
    assert_eq!(num(&interp, "d"), 300.0);
}

/// `1e-7` would round to zero in six decimals and turn a tolerance into "never
/// converge"; the lexer refuses it instead of rounding.
#[test]
fn a_literal_below_six_decimals_is_a_lexer_error() {
    let err = run("let a = 1e-7~\n")
        .err()
        .expect("unrepresentable literal");
    assert!(err.contains("not representable in six decimals"), "{err}");
}

/// Parentheses group; without them precedence applies.
#[test]
fn parentheses_group_and_precedence_still_holds() {
    let interp = run("let a = (1 + 2) * 3~\nlet b = 1 + 2 * 3~\nlet c = -(2 - 5)~\n").unwrap();
    assert_eq!(num(&interp, "a"), 9.0);
    assert_eq!(num(&interp, "b"), 7.0);
    assert_eq!(num(&interp, "c"), 3.0);
}
