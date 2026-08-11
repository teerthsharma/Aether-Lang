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

/// `Activation::Softmax::derivative` returns zeros, and a caller must not use it.
///
/// This pins a trap rather than a property. Softmax is not elementwise — every
/// output depends on every logit in its row — so there is no per-element
/// derivative to return, and the implementation returns a zero tensor with the
/// comment "Handled specially". The training loop is expected to fuse softmax
/// with cross-entropy, where the combined gradient is `p - y` and this function
/// is never called.
///
/// A caller who does call it gets zeros, which is a silently dead gradient: the
/// layer stops learning, the loss stops falling, and nothing errors. That is
/// worth a test not because zeros are correct but because they are *deliberate*,
/// and a later change that made this return ones — the value the private scalar
/// path uses — would look like a fix and would produce a wrong gradient instead
/// of an absent one.
#[test]
fn softmax_returns_a_zero_derivative_and_that_is_deliberate() {
    let logits = Tensor::new(&[0.5, -1.0, 2.0], &[1, 3]);
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
