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
