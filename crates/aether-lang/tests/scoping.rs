//! Language rules both engines implement: a function sees the globals
//! read-only plus its own frame, calling an unbound name is an error, and how
//! a seal loop ends is decided by the parser, not by what a program names its
//! functions.

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

/// `inner` reads `secret`, a local of its caller. Under dynamic scope it would
/// read 42; it sees only the globals and its own frame, where `secret` is
/// unbound and reads as unit.
#[test]
fn a_callee_cannot_see_its_callers_locals() {
    let interp = run("fn inner() { return secret~ }
         fn outer() { let secret = 42~ return inner()~ }
         let seen = outer()~")
    .unwrap();
    let seen = &interp.variables["seen"];
    assert!(matches!(seen, Value::Unit), "{seen:?}");
}

/// The function reads the global `count`, writes a local of the same name,
/// and returns it; the global is 0 after two calls, so both calls return 1.
/// A method mutating a global list mutates a local copy the same way.
#[test]
fn a_global_write_inside_a_function_is_discarded() {
    let interp = run("let count = 0~
         let xs = [1]~
         fn bump() { count = count + 1~ xs.push(2)~ return count~ }
         let first = bump()~
         let second = bump()~")
    .unwrap();
    assert_eq!(num(&interp, "first"), 1.0);
    assert_eq!(num(&interp, "second"), 1.0);
    assert_eq!(num(&interp, "count"), 0.0);
    let xs = &interp.variables["xs"];
    assert!(
        matches!(xs, Value::List(items) if items.len() == 1),
        "{xs:?}"
    );
}

/// A recursive function reaches itself through the globals, and each call's
/// frame is restored when the callee returns.
#[test]
fn recursion_resolves_through_the_globals() {
    let interp = run(
        "fn fact(n) { if n <= 1 { return 1~ } return n * fact(n - 1)~ }
         let f = fact(5)~",
    )
    .unwrap();
    assert_eq!(num(&interp, "f"), 120.0);
}

#[test]
fn calling_an_unbound_name_is_an_error() {
    let err = run("let x = 1~ nope(x)~").err().expect("unbound call");
    assert_eq!(err, "undefined function 'nope'");
    // Reading an unbound variable still yields unit.
    let interp = run("let y = missing~").unwrap();
    assert!(matches!(interp.variables["y"], Value::Unit));
}

/// k climbs 0, 1, 2, 3 and the loop seals before the fifth pass, the one
/// after a pass left k at 3. Were the program's `stable` called, it would
/// return true before the first pass and k would stay 0.
#[test]
fn a_fn_named_stable_does_not_change_a_stable_loop() {
    let interp = run("fn stable(x) { return true~ }
         let k = 0~
         let passes = 0~
         seal until stable(k) {
             passes = passes + 1~
             if k < 3 { k = k + 1~ }
         }
         let direct = stable(1)~")
    .unwrap();
    assert_eq!(num(&interp, "k"), 3.0);
    assert_eq!(num(&interp, "passes"), 4.0);
    // The function is still an ordinary function outside the loop form.
    assert!(matches!(interp.variables["direct"], Value::Bool(true)));
}
