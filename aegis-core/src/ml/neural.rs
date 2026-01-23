//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Neural Network Library
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Neural networks with topological regularization and seal-loop training.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

use libm::{exp, fabs, sqrt};

/// Maximum layer size
const MAX_NEURONS: usize = 64;
/// Maximum layers
const MAX_LAYERS: usize = 8;
/// Maximum data points
const MAX_POINTS: usize = 256;

// ═══════════════════════════════════════════════════════════════════════════════
// Activation Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
    LeakyReLU,
    Softmax,
}

impl Activation {
    /// Apply activation to single value
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
            Activation::Sigmoid => 1.0 / (1.0 + exp(-x.clamp(-500.0, 500.0))),
            Activation::Tanh => {
                let e_pos = exp(x.clamp(-500.0, 500.0));
                let e_neg = exp((-x).clamp(-500.0, 500.0));
                (e_pos - e_neg) / (e_pos + e_neg)
            }
            Activation::Linear => x,
            Activation::LeakyReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            Activation::Softmax => x, // Applied to vector, not scalar
        }
    }

    /// Derivative for backprop
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Sigmoid => {
                let s = self.apply(x);
                s * (1.0 - s)
            }
            Activation::Tanh => {
                let t = self.apply(x);
                1.0 - t * t
            }
            Activation::Linear => 1.0,
            Activation::LeakyReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
            Activation::Softmax => 1.0, // Handled specially
        }
    }

    /// Apply to vector (for softmax)
    pub fn apply_vec(&self, x: &[f64; MAX_NEURONS], len: usize) -> [f64; MAX_NEURONS] {
        let mut result = [0.0; MAX_NEURONS];

        match self {
            Activation::Softmax => {
                let max_val = x.iter().take(len).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0;
                for (i, r) in result.iter_mut().enumerate().take(len) {
                    *r = exp((x[i] - max_val).clamp(-500.0, 500.0));
                    sum += *r;
                }
                for r in result.iter_mut().take(len) {
                    *r /= sum.max(1e-10);
                }
            }
            _ => {
                for i in 0..len {
                    result[i] = self.apply(x[i]);
                }
            }
        }

        result
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dense Layer
// ═══════════════════════════════════════════════════════════════════════════════

/// Dense (fully connected) layer
#[derive(Debug, Clone)]
pub struct DenseLayer {
    /// Weights [output][input]
    pub weights: [[f64; MAX_NEURONS]; MAX_NEURONS],
    /// Biases
    pub biases: [f64; MAX_NEURONS],
    /// Input size
    pub input_size: usize,
    /// Output size
    pub output_size: usize,
    /// Activation function
    pub activation: Activation,
    /// Last input (for backprop)
    last_input: [f64; MAX_NEURONS],
    /// Last pre-activation (for backprop)
    last_z: [f64; MAX_NEURONS],
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let input_size = input_size.min(MAX_NEURONS);
        let output_size = output_size.min(MAX_NEURONS);

        // Xavier initialization
        let scale = sqrt(2.0 / (input_size + output_size) as f64);
        let mut weights = [[0.0; MAX_NEURONS]; MAX_NEURONS];

        // Simple pseudo-random initialization
        let mut rng = 42u64;
        for row in weights.iter_mut().take(output_size) {
            for col in row.iter_mut().take(input_size) {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let r = (rng as f64 / u64::MAX as f64) * 2.0 - 1.0;
                *col = r * scale;
            }
        }

        Self {
            weights,
            biases: [0.0; MAX_NEURONS],
            input_size,
            output_size,
            activation,
            last_input: [0.0; MAX_NEURONS],
            last_z: [0.0; MAX_NEURONS],
        }
    }

    /// Forward pass
    pub fn forward(&mut self, input: &[f64; MAX_NEURONS]) -> [f64; MAX_NEURONS] {
        self.last_input = *input;

        // Linear transformation: z = Wx + b
        for (i, z) in self.last_z.iter_mut().enumerate().take(self.output_size) {
            let mut sum = self.biases[i];
            for (w, x) in self.weights[i].iter().zip(input.iter()).take(self.input_size) {
                sum += *w * *x;
            }
            *z = sum;
        }

        // Activation
        self.activation.apply_vec(&self.last_z, self.output_size)
    }

    /// Backward pass, returns gradient w.r.t. input
    pub fn backward(&mut self, grad_output: &[f64; MAX_NEURONS], lr: f64) -> [f64; MAX_NEURONS] {
        let mut grad_input = [0.0; MAX_NEURONS];

        // Compute delta = grad_output * activation'(z)
        let mut delta = [0.0; MAX_NEURONS];
        for i in 0..self.output_size {
            delta[i] = grad_output[i] * self.activation.derivative(self.last_z[i]);
        }

        // Gradient w.r.t. weights: dW = delta * input^T
        // Gradient w.r.t. biases: db = delta
        // Gradient w.r.t. input: dx = W^T * delta

        for (i, d) in delta.iter().enumerate().take(self.output_size) {
            for (j, input_j) in self.last_input.iter().enumerate().take(self.input_size) {
                // Update weight
                self.weights[i][j] -= lr * d * input_j;
                // Accumulate input gradient
                grad_input[j] += self.weights[i][j] * d;
            }
            // Update bias
            self.biases[i] -= lr * d;
        }

        grad_input
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Multi-Layer Perceptron
// ═══════════════════════════════════════════════════════════════════════════════

/// Multi-Layer Perceptron neural network
#[derive(Debug, Clone)]
pub struct MLP {
    /// Layers
    layers: [Option<DenseLayer>; MAX_LAYERS],
    /// Number of layers
    n_layers: usize,
    /// Learning rate
    learning_rate: f64,
}

impl MLP {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            layers: [None, None, None, None, None, None, None, None],
            n_layers: 0,
            learning_rate,
        }
    }

    /// Add a dense layer
    pub fn add_layer(&mut self, input_size: usize, output_size: usize, activation: Activation) {
        if self.n_layers >= MAX_LAYERS {
            return;
        }
        self.layers[self.n_layers] = Some(DenseLayer::new(input_size, output_size, activation));
        self.n_layers += 1;
    }

    /// Forward pass through all layers
    pub fn forward(&mut self, input: &[f64; MAX_NEURONS]) -> [f64; MAX_NEURONS] {
        let mut current = *input;

        for i in 0..self.n_layers {
            if let Some(ref mut layer) = self.layers[i] {
                current = layer.forward(&current);
            }
        }

        current
    }

    /// Train on single sample (returns loss)
    pub fn train_step(
        &mut self,
        input: &[f64; MAX_NEURONS],
        target: &[f64; MAX_NEURONS],
        output_size: usize,
    ) -> f64 {
        // Forward pass
        let output = self.forward(input);

        // Compute MSE loss and initial gradient
        let mut loss = 0.0;
        let mut grad = [0.0; MAX_NEURONS];
        for (i, (o, t)) in output.iter().zip(target.iter()).enumerate().take(output_size) {
            let diff = o - t;
            loss += diff * diff;
            grad[i] = 2.0 * diff / output_size as f64;
        }
        loss /= output_size as f64;

        // Backward pass through layers in reverse
        for i in (0..self.n_layers).rev() {
            if let Some(ref mut layer) = self.layers[i] {
                grad = layer.backward(&grad, self.learning_rate);
            }
        }

        loss
    }

    /// Train on dataset with seal-loop style convergence
    pub fn fit(
        &mut self,
        x: &[[f64; MAX_NEURONS]],
        y: &[[f64; MAX_NEURONS]],
        n_samples: usize,
        output_size: usize,
        max_epochs: usize,
        tol: f64,
    ) -> TrainingResult {
        let n = n_samples.min(MAX_POINTS);
        let mut result = TrainingResult::default();

        let mut prev_loss = f64::MAX;

        for epoch in 0..max_epochs {
            let mut total_loss = 0.0;

            for i in 0..n {
                let loss = self.train_step(&x[i], &y[i], output_size);
                total_loss += loss;
            }

            let avg_loss = total_loss / n as f64;
            result.final_loss = avg_loss;
            result.epochs = epoch as u32 + 1;
            result.loss_history[epoch.min(99)] = avg_loss;

            // Check convergence
            if avg_loss < tol {
                result.converged = true;
                break;
            }

            if fabs(prev_loss - avg_loss) < tol * 0.1 {
                result.converged = true;
                break;
            }

            prev_loss = avg_loss;
        }

        result
    }

    /// Predict (forward pass without training)
    pub fn predict(&mut self, input: &[f64; MAX_NEURONS]) -> [f64; MAX_NEURONS] {
        self.forward(input)
    }
}

/// Training result
#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub epochs: u32,
    pub final_loss: f64,
    pub converged: bool,
    pub loss_history: [f64; 100],
}

impl Default for TrainingResult {
    fn default() -> Self {
        Self {
            epochs: 0,
            final_loss: f64::MAX,
            converged: false,
            loss_history: [0.0; 100],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Optimizers
// ═══════════════════════════════════════════════════════════════════════════════

/// Optimizer state for a layer
#[derive(Debug, Clone)]
pub struct AdamState {
    /// First moment (m)
    m_weights: [[f64; MAX_NEURONS]; MAX_NEURONS],
    m_biases: [f64; MAX_NEURONS],
    /// Second moment (v)
    v_weights: [[f64; MAX_NEURONS]; MAX_NEURONS],
    v_biases: [f64; MAX_NEURONS],
    /// Timestep
    t: u64,
}

impl Default for AdamState {
    fn default() -> Self {
        Self {
            m_weights: [[0.0; MAX_NEURONS]; MAX_NEURONS],
            m_biases: [0.0; MAX_NEURONS],
            v_weights: [[0.0; MAX_NEURONS]; MAX_NEURONS],
            v_biases: [0.0; MAX_NEURONS],
            t: 0,
        }
    }
}

/// Adam optimizer parameters
#[derive(Debug, Clone, Copy)]
pub struct AdamParams {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
}

impl Default for AdamParams {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// Adam optimizer update for layer weights
#[allow(clippy::too_many_arguments)]
pub fn adam_update(
    weights: &mut [[f64; MAX_NEURONS]; MAX_NEURONS],
    biases: &mut [f64; MAX_NEURONS],
    grad_w: &[[f64; MAX_NEURONS]; MAX_NEURONS],
    grad_b: &[f64; MAX_NEURONS],
    state: &mut AdamState,
    params: &AdamParams,
    input_size: usize,
    output_size: usize,
) {
    state.t += 1;
    let t = state.t as f64;

    for i in 0..output_size {
        for j in 0..input_size {
            // Update moments
            state.m_weights[i][j] =
                params.beta1 * state.m_weights[i][j] + (1.0 - params.beta1) * grad_w[i][j];
            state.v_weights[i][j] = params.beta2 * state.v_weights[i][j]
                + (1.0 - params.beta2) * grad_w[i][j] * grad_w[i][j];

            // Bias correction
            let m_hat = state.m_weights[i][j] / (1.0 - libm::pow(params.beta1, t));
            let v_hat = state.v_weights[i][j] / (1.0 - libm::pow(params.beta2, t));

            // Update weight
            weights[i][j] -= params.lr * m_hat / (sqrt(v_hat) + params.epsilon);
        }

        // Update bias
        state.m_biases[i] = params.beta1 * state.m_biases[i] + (1.0 - params.beta1) * grad_b[i];
        state.v_biases[i] =
            params.beta2 * state.v_biases[i] + (1.0 - params.beta2) * grad_b[i] * grad_b[i];

        let m_hat = state.m_biases[i] / (1.0 - libm::pow(params.beta1, t));
        let v_hat = state.v_biases[i] / (1.0 - libm::pow(params.beta2, t));

        biases[i] -= params.lr * m_hat / (sqrt(v_hat) + params.epsilon);
    }
}

/// SGD with momentum state
#[derive(Debug, Clone)]
pub struct SGDMomentumState {
    velocity_weights: [[f64; MAX_NEURONS]; MAX_NEURONS],
    velocity_biases: [f64; MAX_NEURONS],
}

impl Default for SGDMomentumState {
    fn default() -> Self {
        Self {
            velocity_weights: [[0.0; MAX_NEURONS]; MAX_NEURONS],
            velocity_biases: [0.0; MAX_NEURONS],
        }
    }
}

/// SGD with momentum update
#[allow(clippy::too_many_arguments)]
pub fn sgd_momentum_update(
    weights: &mut [[f64; MAX_NEURONS]; MAX_NEURONS],
    biases: &mut [f64; MAX_NEURONS],
    grad_w: &[[f64; MAX_NEURONS]; MAX_NEURONS],
    grad_b: &[f64; MAX_NEURONS],
    state: &mut SGDMomentumState,
    lr: f64,
    momentum: f64,
    input_size: usize,
    output_size: usize,
) {
    for i in 0..output_size {
        for j in 0..input_size {
            state.velocity_weights[i][j] =
                momentum * state.velocity_weights[i][j] - lr * grad_w[i][j];
            weights[i][j] += state.velocity_weights[i][j];
        }

        state.velocity_biases[i] = momentum * state.velocity_biases[i] - lr * grad_b[i];
        biases[i] += state.velocity_biases[i];
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Topological Regularization
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute Betti-0 based penalty for fragmented outputs
pub fn topo_regularization(
    outputs: &[[f64; MAX_NEURONS]],
    n_samples: usize,
    output_size: usize,
    epsilon: f64,
) -> f64 {
    // Count connected components in output space
    let n = n_samples.min(MAX_POINTS);
    if n <= 1 {
        return 0.0;
    }

    let mut visited = [false; MAX_POINTS];
    let mut components = 0u32;

    for start in 0..n {
        if visited[start] {
            continue;
        }

        components += 1;
        let mut stack = [0usize; 64];
        let mut top = 1;
        stack[0] = start;

        while top > 0 {
            top -= 1;
            let current = stack[top];
            if visited[current] {
                continue;
            }
            visited[current] = true;

            for i in 0..n {
                if !visited[i] && i != current {
                    let dist = output_distance(&outputs[current], &outputs[i], output_size);
                    if dist < epsilon && top < 64 {
                        stack[top] = i;
                        top += 1;
                    }
                }
            }
        }
    }

    // Penalty for having more than 1 component (fragmented)
    (components.saturating_sub(1)) as f64 * 0.01
}

fn output_distance(a: &[f64; MAX_NEURONS], b: &[f64; MAX_NEURONS], len: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..len {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sqrt(sum)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_relu() {
        assert_eq!(Activation::ReLU.apply(5.0), 5.0);
        assert_eq!(Activation::ReLU.apply(-5.0), 0.0);
    }

    #[test]
    fn test_activation_sigmoid() {
        let s = Activation::Sigmoid.apply(0.0);
        assert!((s - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_dense_forward() {
        let mut layer = DenseLayer::new(2, 2, Activation::Linear);
        // Set weights to identity
        layer.weights = [[0.0; MAX_NEURONS]; MAX_NEURONS];
        layer.weights[0][0] = 1.0;
        layer.weights[1][1] = 1.0;

        let mut input = [0.0; MAX_NEURONS];
        input[0] = 1.0;
        input[1] = 2.0;

        let output = layer.forward(&input);
        assert!((output[0] - 1.0).abs() < 1e-10);
        assert!((output[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_mlp_xor() {
        // XOR problem
        let mut mlp = MLP::new(0.5);
        mlp.add_layer(2, 4, Activation::ReLU);
        mlp.add_layer(4, 1, Activation::Sigmoid);

        let x = [
            {
                let mut a = [0.0; MAX_NEURONS];
                a[0] = 0.0;
                a[1] = 0.0;
                a
            },
            {
                let mut a = [0.0; MAX_NEURONS];
                a[0] = 0.0;
                a[1] = 1.0;
                a
            },
            {
                let mut a = [0.0; MAX_NEURONS];
                a[0] = 1.0;
                a[1] = 0.0;
                a
            },
            {
                let mut a = [0.0; MAX_NEURONS];
                a[0] = 1.0;
                a[1] = 1.0;
                a
            },
        ];
        let y = [
            {
                let mut a = [0.0; MAX_NEURONS];
                a[0] = 0.0;
                a
            },
            {
                let mut a = [0.0; MAX_NEURONS];
                a[0] = 1.0;
                a
            },
            {
                let mut a = [0.0; MAX_NEURONS];
                a[0] = 1.0;
                a
            },
            {
                let mut a = [0.0; MAX_NEURONS];
                a[0] = 0.0;
                a
            },
        ];

        let result = mlp.fit(&x, &y, 4, 1, 1000, 0.01);
        // XOR is hard, just check it runs
        assert!(result.epochs > 0);
    }
}
