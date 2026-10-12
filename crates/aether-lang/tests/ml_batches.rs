//! `mlp.train` and `mlp.forward` on a batch of rows, and the shape checks that
//! turn an `aether-core` shape assertion into a refusal the program can see.

use aether_lang::{Interpreter, Parser};

fn run(src: &str) -> Result<Interpreter, String> {
    let program = Parser::new(src).parse().map_err(|e| e.to_string())?;
    let mut interp = Interpreter::new();
    interp.execute(&program)?;
    Ok(interp)
}

const NET: &str = "import Ml~
let mlp = Ml.MLP(0.1)~
mlp.add_layer(2, 4, \"relu\")~
mlp.add_layer(4, 1, \"sigmoid\")~
";

/// A `[2, 2]` batch trains as two samples: this ended the process with a
/// shape assertion when the whole batch was passed as one sample.
#[test]
fn a_batch_of_rows_trains_and_predicts_one_output_per_row() {
    let src = format!(
        "{NET}let x = Ml.load_weights([[0.0, 0.0], [1.0, 1.0]])~
let y = Ml.load_weights([[0.0], [1.0]])~
let loss = mlp.train(x, y, 10)~
let pred = mlp.forward(x)~
"
    );
    let interp = run(&src).expect("batch training runs");
    match &interp.variables["loss"] {
        aether_lang::interpreter::Value::Num(l) => assert!(l.is_finite()),
        other => panic!("loss is not a number: {other:?}"),
    }
    match &interp.variables["pred"] {
        aether_lang::interpreter::Value::Tensor(t) => assert_eq!(t.shape, vec![2, 1]),
        other => panic!("prediction is not a tensor: {other:?}"),
    }
}

#[test]
fn rows_of_the_wrong_width_are_refused_not_asserted() {
    let src = format!(
        "{NET}let x = Ml.load_weights([[0.0, 0.0, 0.0]])~
let y = Ml.load_weights([[1.0]])~
mlp.train(x, y, 1)~
"
    );
    let err = run(&src).err().expect("width 3 into a width-2 layer");
    assert!(err.contains("expected rows of 2 values"), "{err}");
}

#[test]
fn a_network_without_layers_is_refused() {
    let err = run("import Ml~\nlet mlp = Ml.MLP(0.1)~\nmlp.forward([1.0])~\n")
        .err()
        .expect("no layers");
    assert!(err.contains("MLP has no layers"), "{err}");
}
