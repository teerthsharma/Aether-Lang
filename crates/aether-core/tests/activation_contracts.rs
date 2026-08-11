//! Contracts for the activation derivatives in `ml::neural`.
//!
//! Written because a doc comment was added asserting that ReLU's derivative is
//! zero at exactly zero, matching the GPU kernel — and nothing in this crate
//! checked it. The parity test that does live in `aether-gpu`, so `aether-core`
//! could change its behaviour at the kink and only a different crate's suite
//! would notice. A claim in a comment whose only evidence is elsewhere is the
//! shape of defect this repository has spent most of its history removing, and
//! it was introduced here by the change that wrote the comment.
//!
//! The kink cases matter more than they look. A derivative at a single point is
//! measure-zero for random inputs and decisive for the ones that arrive in
//! practice: a dead ReLU sits at exactly zero, and whether it can revive depends
//! on which side of the branch that point falls.

use aether_core::ml::neural::Activation;
use aether_core::ml::tensor::Tensor;

fn derivative_at(activation: Activation, x: f64) -> f64 {
    let input = Tensor::new(&[x], &[1, 1]);
    activation.derivative(&input).data.borrow()[0]
}

/// ReLU's derivative at exactly zero is zero, not one.
///
/// Both conventions are defensible — the function has no derivative there — and
/// the only thing that matters is that the two implementations agree, because a
/// GPU result differing from the CPU reference at one point is a parity failure
/// that no random fixture will reproduce.
#[test]
fn relu_takes_the_lower_branch_at_exactly_zero() {
    assert_eq!(derivative_at(Activation::ReLU, 0.0), 0.0);
    assert_eq!(derivative_at(Activation::ReLU, -0.0), 0.0);

    // The neighbours, so the assertion above is about the kink and not about a
    // derivative that is zero everywhere.
    assert_eq!(derivative_at(Activation::ReLU, f64::MIN_POSITIVE), 1.0);
    assert_eq!(derivative_at(Activation::ReLU, -f64::MIN_POSITIVE), 0.0);
}

/// LeakyReLU takes the same branch at zero, which is its whole point.
///
/// If it took the upper branch it would agree with ReLU exactly where ReLU is
/// dead, and the leak would do nothing for the case it exists to fix.
#[test]
fn leaky_relu_leaks_at_exactly_zero() {
    assert_eq!(derivative_at(Activation::LeakyReLU, 0.0), 0.01);
    assert_eq!(derivative_at(Activation::LeakyReLU, f64::MIN_POSITIVE), 1.0);
}

/// Away from the kinks, every elementwise derivative matches a central
/// difference of the activation itself.
///
/// The kink tests pin a convention; this pins the arithmetic. Softmax is
/// excluded because it is not elementwise — see below.
#[test]
fn derivatives_match_central_differences_away_from_kinks() {
    let h = 1e-6;

    for activation in [
        Activation::ReLU,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::Linear,
        Activation::LeakyReLU,
    ] {
        for &x in &[-3.0, -1.25, -0.5, 0.5, 1.25, 3.0] {
            let plus = activation.apply(&Tensor::new(&[x + h], &[1, 1]));
            let minus = activation.apply(&Tensor::new(&[x - h], &[1, 1]));
            let numerical = (plus.data.borrow()[0] - minus.data.borrow()[0]) / (2.0 * h);
            let analytic = derivative_at(activation, x);

            assert!(
                (analytic - numerical).abs() < 1e-6,
                "{activation:?} at {x}: analytic {analytic}, numerical {numerical}"
            );
        }
    }
}

/// `Activation::Softmax::derivative` returns zeros, and nothing may consume them.
///
/// The previous version of this comment called the zeros deliberate, on the
/// strength of a source comment reading "Handled specially". Nothing handled
/// them: `DenseLayer::backward` multiplied by them like any other activation,
/// which zeroed the layer's gradient and everything upstream of it.
/// `a_softmax_output_layer_learns` below is what established that, and the
/// backward pass now computes the exact Jacobian-vector product instead.
///
/// The zeros stay, because there is no elementwise derivative for this variant
/// to return and a plausible-looking non-zero would be worse. This pins them so
/// that a later change returning ones — the value the private scalar path uses,
/// and correct only under a fusion this crate cannot perform — fails here rather
/// than in a training run.
#[test]
fn softmax_returns_a_zero_derivative_that_nothing_consumes() {
    let logits = Tensor::new(&[0.5, -1.0, 2.0], &[3, 1]);
    let derivative = Activation::Softmax.derivative(&logits);

    assert_eq!(derivative.shape, logits.shape);
    for (i, &value) in derivative.data.borrow().iter().enumerate() {
        assert_eq!(
            value, 0.0,
            "component {i} is {value}, not zero: softmax has no elementwise \
             derivative, and returning a non-zero one would produce a wrong \
             gradient where this produces an absent one"
        );
    }
}

/// A network with a softmax output layer must actually train.
///
/// The variant's doc says the training loop is expected to fuse softmax with
/// cross-entropy, where the combined gradient is `p - y` and the elementwise
/// derivative is never used. This checks that the loop does what the doc says.
///
/// If it does not, the failure is silent and total. `DenseLayer::backward`
/// computes `delta = grad_output * activation.derivative(z)`, and that
/// derivative is a zero tensor for softmax — so the output layer's gradients are
/// zero, and the zero it propagates backwards kills every layer before it too. A
/// network like this does not train slowly; it does not train at all, while the
/// loss stays finite and nothing errors.
#[test]
fn a_softmax_output_layer_learns() {
    use aether_core::ml::linalg::LossConfig;
    use aether_core::ml::neural::{OptimizerConfig, MLP};

    let mut mlp = MLP::new(
        OptimizerConfig::SGD {
            learning_rate: 0.5,
            momentum: 0.0,
        },
        LossConfig::BinaryCrossEntropy,
    );
    mlp.add_layer(3, 4, Activation::ReLU, Some(1));
    mlp.add_layer(4, 2, Activation::Softmax, Some(2));

    let x = vec![
        Tensor::new(&[1.0, 0.0, 0.0], &[3, 1]),
        Tensor::new(&[0.0, 1.0, 0.0], &[3, 1]),
    ];
    let y = vec![
        Tensor::new(&[1.0, 0.0], &[2, 1]),
        Tensor::new(&[0.0, 1.0], &[2, 1]),
    ];

    let before = mlp.layers[1].weights.data.borrow().clone();
    let result = mlp.fit(&x, &y, 40);
    let after = mlp.layers[1].weights.data.borrow().clone();

    let moved = before
        .iter()
        .zip(&after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    assert!(
        moved > 1e-9,
        "the softmax layer's weights did not move at all after 40 epochs \
         (largest change {moved:.3e}); its gradient is being multiplied by the \
         zero tensor that Activation::Softmax::derivative returns"
    );
    assert!(
        result.final_loss < result.loss_history[0],
        "loss did not fall: {} to {}",
        result.loss_history[0],
        result.final_loss
    );
}

/// The softmax layer's gradient must match a central difference of the loss.
///
/// The regression test above asserts the weights move and the loss falls, which
/// a wrong-but-non-zero gradient also satisfies. That was the exact hazard the
/// fix it guards was chosen to avoid — substituting a plausible gradient for an
/// absent one — so verifying only that something happens leaves the interesting
/// half unchecked.
///
/// The Jacobian is not diagonal here, so the elementwise finite-difference check
/// in this file cannot reach it. This differences the *loss* against a weight
/// instead, which needs no assumption about the Jacobian's shape.
///
/// The analytic gradient is recovered from one SGD step at a known learning rate
/// with no momentum, where `dW = (before - after) / lr` exactly. That reads the
/// gradient the backward pass actually applied rather than one recomputed for
/// the test, which is the only version worth checking.
#[test]
fn the_softmax_layer_gradient_matches_finite_differences() {
    use aether_core::ml::linalg::LossConfig;
    use aether_core::ml::neural::{OptimizerConfig, MLP};

    let lr = 1e-3;
    let h = 1e-6;
    let x = Tensor::new(&[0.7, -0.2, 0.4], &[3, 1]);
    // Three classes, so a formula that happens to be right for a pair does not
    // pass — with two, several wrong expressions coincide with the correct one.
    let y = Tensor::new(&[0.0, 1.0, 0.0], &[3, 1]);

    let build = || {
        let mut mlp = MLP::new(
            OptimizerConfig::SGD {
                learning_rate: lr,
                momentum: 0.0,
            },
            LossConfig::BinaryCrossEntropy,
        );
        mlp.add_layer(3, 4, Activation::Tanh, Some(7));
        mlp.add_layer(4, 3, Activation::Softmax, Some(11));
        mlp
    };

    // Both layers. The hidden one is the point: the fix changed `delta`, which
    // feeds the softmax layer's own weight gradients *and* the gradient it hands
    // backwards, and the second is what made the original bug total rather than
    // local -- a zero there killed every layer before it. Checking only the
    // output layer verifies the half that was never dangerous.
    // Weights and biases both. The softmax fix changed `delta`, which is the
    // common factor in both gradients, so a bias gradient that was wrong in the
    // same way would have passed everything here.
    //
    // Both halves are demonstrated, not assumed. Substituting pass-through for
    // the Jacobian-vector product fails at `layer 1 weights[7]`; doubling
    // `grad_b` in `DenseLayer::backward`, which corrupts the bias path and
    // leaves the weights correct, fails at `layer 1 biases[1]` with a worst
    // disagreement of 3.382e-1. The second probe exists because the first aborts
    // before reaching the bias comparison, so it could not have shown it
    // rejecting anything.
    for layer in [1usize, 0] {
        for parameter in ["weights", "biases"] {
            let read = |mlp: &MLP| -> Vec<f64> {
                if parameter == "weights" {
                    mlp.layers[layer].weights.data.borrow().clone()
                } else {
                    mlp.layers[layer].biases.data.borrow().clone()
                }
            };

            // One step, then read back what the optimiser applied.
            let mut trained = build();
            let before = read(&trained);
            trained.train_step(&x, &y);
            let after = read(&trained);

            let analytic: Vec<f64> = before
                .iter()
                .zip(&after)
                .map(|(b, a)| (b - a) / lr)
                .collect();

            let loss_with = |perturbed: &[f64]| -> f64 {
                let mut probe = build();
                if parameter == "weights" {
                    probe.layers[layer]
                        .weights
                        .data
                        .borrow_mut()
                        .copy_from_slice(perturbed);
                } else {
                    probe.layers[layer]
                        .biases
                        .data
                        .borrow_mut()
                        .copy_from_slice(perturbed);
                }
                let output = probe.predict(&x);
                probe.loss.compute(&y, &output)
            };

            let mut worst = 0.0f64;
            let mut worst_at = 0usize;
            for i in 0..before.len() {
                let mut plus = before.clone();
                let mut minus = before.clone();
                plus[i] += h;
                minus[i] -= h;

                let numerical = (loss_with(&plus) - loss_with(&minus)) / (2.0 * h);
                let error = (analytic[i] - numerical).abs();
                if error > worst {
                    worst = error;
                    worst_at = i;
                }
            }

            assert!(
                worst <= 1e-5,
                "layer {layer} {parameter}[{worst_at}]: analytic {}, worst \
                 disagreement {worst:.3e}. The backward pass is applying a \
                 gradient that is not the derivative of the loss it minimises.",
                analytic[worst_at]
            );

            // The control. Every assertion above holds trivially if the gradient
            // is zero and the loss is flat, which is the bug this file guards.
            let magnitude = analytic.iter().fold(0.0f64, |m, g| m.max(g.abs()));
            assert!(
                magnitude > 1e-6,
                "layer {layer} {parameter}: recovered gradient {magnitude:.3e} is \
                 indistinguishable from zero, so agreeing with a finite \
                 difference says nothing"
            );
        }
    }
}

/// One optimiser step is homogeneous of degree one in the learning rate.
///
/// Written to defend the gradcheck above, which recovers the analytic gradient
/// as `(before - after) / lr` and therefore looked like it depended on the
/// optimiser applying the gradient unmodified. Two things came out of writing
/// it, and both correct that reasoning.
///
/// First, this test does **not** detect momentum, and momentum is not a threat.
/// It was verified by setting `momentum: 0.9` here, expecting a failure, and
/// getting a pass — because on a first step from a fresh optimiser there is no
/// accumulated velocity, so `v = g` and the update is identical to plain
/// descent. Momentum diverges from step two onwards, and every recovery in this
/// file takes exactly one step from a freshly built network.
///
/// Second, the gradcheck was never as exposed as its own limits claimed. An
/// optimiser that distorted the gradient would make the recovered value disagree
/// with the finite difference, and the comparison would fail — the agreement is
/// itself the validation. The failure mode that worried me cannot be silent.
///
/// What this leaves is a narrower and still real property: the update scales
/// exactly with the learning rate. That rules out a clip applied after scaling,
/// which is not homogeneous. It does not rule out weight decay or an adaptive
/// denominator, both of which are homogeneous in `lr` and would be caught by the
/// gradcheck instead.
#[test]
fn one_optimiser_step_scales_with_the_learning_rate() {
    use aether_core::ml::linalg::LossConfig;
    use aether_core::ml::neural::{OptimizerConfig, MLP};

    let x = Tensor::new(&[0.7, -0.2, 0.4], &[3, 1]);
    let y = Tensor::new(&[0.0, 1.0, 0.0], &[3, 1]);

    let implied_gradient = |lr: f64| -> Vec<f64> {
        let mut mlp = MLP::new(
            OptimizerConfig::SGD {
                learning_rate: lr,
                momentum: 0.0,
            },
            LossConfig::BinaryCrossEntropy,
        );
        mlp.add_layer(3, 4, Activation::Tanh, Some(7));
        mlp.add_layer(4, 3, Activation::Softmax, Some(11));

        let before: Vec<f64> = mlp.layers[1].weights.data.borrow().clone();
        mlp.train_step(&x, &y);
        let after: Vec<f64> = mlp.layers[1].weights.data.borrow().clone();
        before
            .iter()
            .zip(&after)
            .map(|(b, a)| (b - a) / lr)
            .collect()
    };

    let slow = implied_gradient(1e-4);
    let fast = implied_gradient(1e-2);

    let mut worst = 0.0f64;
    for (a, b) in slow.iter().zip(&fast) {
        worst = worst.max((a - b).abs());
    }

    assert!(
        worst <= 1e-9,
        "the implied gradient changed with the learning rate by {worst:.3e}, so \
         one step is not minus lr times the gradient. Every finite-difference \
         check in this file recovers the gradient that way and is measuring \
         something else."
    );

    // The control. Two identical zero vectors agree perfectly and would satisfy
    // the assertion above without the optimiser having done anything.
    let magnitude = slow.iter().fold(0.0f64, |m, g| m.max(g.abs()));
    assert!(
        magnitude > 1e-6,
        "the implied gradient is {magnitude:.3e}, indistinguishable from zero"
    );
}
