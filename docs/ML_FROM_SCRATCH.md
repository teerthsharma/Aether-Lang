# AEGIS ML From Scratch

> **Building Complete Machine Learning Algorithms from Zero in AEGIS**

This guide proves AEGIS can implement ML algorithms entirely from first principles, using only basic operations and the `~` statement terminator. No external libraries. No pre-built functions. Pure AEGIS.

---

## Table of Contents

1. [Linear Algebra Foundation](#linear-algebra-foundation)
2. [Loss Functions](#loss-functions)
3. [Gradient Computation](#gradient-computation)
4. [Optimizers](#optimizers)
5. [Complete Models](#complete-models)
6. [Neural Networks](#neural-networks)
7. [Topological Extensions](#topological-extensions)

---

## Linear Algebra Foundation

### Basic Operations

```aegis
// ═══════════════════════════════════════════════════
// Vector Operations from Scratch
// ═══════════════════════════════════════════════════

// Vector addition
fn vadd(a, b) {
    let result = zeros(len(a))~
    for i in 0..len(a) {
        result[i] = a[i] + b[i]~
    }
    return result~
}

// Scalar multiplication
fn vscale(v, s) {
    let result = zeros(len(v))~
    for i in 0..len(v) {
        result[i] = v[i] * s~
    }
    return result~
}

// Dot product
fn dot(a, b) {
    let sum = 0.0~
    for i in 0..len(a) {
        sum = sum + a[i] * b[i]~
    }
    return sum~
}

// Euclidean norm
fn norm(v) {
    return sqrt(dot(v, v))~
}

// Normalize vector
fn normalize(v) {
    let n = norm(v)~
    if n < 1e-10 {
        return v~
    }
    return vscale(v, 1.0 / n)~
}

// L2 distance
fn distance(a, b) {
    let sum = 0.0~
    for i in 0..len(a) {
        let diff = a[i] - b[i]~
        sum = sum + diff * diff~
    }
    return sqrt(sum)~
}

// Cosine similarity
fn cosine(a, b) {
    let na = norm(a)~
    let nb = norm(b)~
    if na < 1e-10 or nb < 1e-10 {
        return 0.0~
    }
    return dot(a, b) / (na * nb)~
}
```

### Matrix Operations

```aegis
// ═══════════════════════════════════════════════════
// Matrix Operations from Scratch
// ═══════════════════════════════════════════════════

// Create zero matrix
fn zeros_matrix(rows, cols) {
    let M = []~
    for i in 0..rows {
        M[i] = zeros(cols)~
    }
    return M~
}

// Matrix-vector multiply
fn matvec(A, v) {
    let rows = len(A)~
    let result = zeros(rows)~
    for i in 0..rows {
        result[i] = dot(A[i], v)~
    }
    return result~
}

// Matrix-matrix multiply
fn matmul(A, B) {
    let rows_a = len(A)~
    let cols_a = len(A[0])~
    let cols_b = len(B[0])~
    let C = zeros_matrix(rows_a, cols_b)~
    
    for i in 0..rows_a {
        for j in 0..cols_b {
            let sum = 0.0~
            for k in 0..cols_a {
                sum = sum + A[i][k] * B[k][j]~
            }
            C[i][j] = sum~
        }
    }
    return C~
}

// Transpose
fn transpose(A) {
    let rows = len(A)~
    let cols = len(A[0])~
    let T = zeros_matrix(cols, rows)~
    for i in 0..rows {
        for j in 0..cols {
            T[j][i] = A[i][j]~
        }
    }
    return T~
}

// Outer product
fn outer(a, b) {
    let m = len(a)~
    let n = len(b)~
    let O = zeros_matrix(m, n)~
    for i in 0..m {
        for j in 0..n {
            O[i][j] = a[i] * b[j]~
        }
    }
    return O~
}
```

### Statistical Functions

```aegis
// ═══════════════════════════════════════════════════
// Statistics from Scratch
// ═══════════════════════════════════════════════════

// Mean
fn mean(data) {
    let sum = 0.0~
    for x in data {
        sum = sum + x~
    }
    return sum / len(data)~
}

// Variance
fn variance(data) {
    let m = mean(data)~
    let sum = 0.0~
    for x in data {
        let diff = x - m~
        sum = sum + diff * diff~
    }
    return sum / len(data)~
}

// Standard deviation
fn std(data) {
    return sqrt(variance(data))~
}

// Covariance
fn covariance(x, y) {
    let mx = mean(x)~
    let my = mean(y)~
    let sum = 0.0~
    for i in 0..len(x) {
        sum = sum + (x[i] - mx) * (y[i] - my)~
    }
    return sum / len(x)~
}

// Correlation
fn correlation(x, y) {
    return covariance(x, y) / (std(x) * std(y))~
}
```

---

## Loss Functions

```aegis
// ═══════════════════════════════════════════════════
// Loss Functions from Scratch
// ═══════════════════════════════════════════════════

// Mean Squared Error
fn mse(y_true, y_pred) {
    let sum = 0.0~
    for i in 0..len(y_true) {
        let diff = y_true[i] - y_pred[i]~
        sum = sum + diff * diff~
    }
    return sum / len(y_true)~
}

// Mean Absolute Error
fn mae(y_true, y_pred) {
    let sum = 0.0~
    for i in 0..len(y_true) {
        sum = sum + abs(y_true[i] - y_pred[i])~
    }
    return sum / len(y_true)~
}

// Root Mean Squared Error
fn rmse(y_true, y_pred) {
    return sqrt(mse(y_true, y_pred))~
}

// Binary Cross-Entropy
fn binary_cross_entropy(y_true, y_pred) {
    let sum = 0.0~
    for i in 0..len(y_true) {
        let p = clip(y_pred[i], 1e-7, 1.0 - 1e-7)~
        sum = sum - (y_true[i] * log(p) + (1 - y_true[i]) * log(1 - p))~
    }
    return sum / len(y_true)~
}

// Categorical Cross-Entropy
fn categorical_cross_entropy(y_true, y_pred) {
    let sum = 0.0~
    for i in 0..len(y_true) {
        for k in 0..len(y_true[i]) {
            if y_true[i][k] > 0 {
                let p = clip(y_pred[i][k], 1e-7, 1.0)~
                sum = sum - y_true[i][k] * log(p)~
            }
        }
    }
    return sum / len(y_true)~
}

// Hinge Loss (SVM)
fn hinge_loss(y_true, y_pred) {
    let sum = 0.0~
    for i in 0..len(y_true) {
        let margin = 1 - y_true[i] * y_pred[i]~
        sum = sum + max(0, margin)~
    }
    return sum / len(y_true)~
}

// Huber Loss (robust to outliers)
fn huber_loss(y_true, y_pred, delta) {
    let sum = 0.0~
    for i in 0..len(y_true) {
        let diff = abs(y_true[i] - y_pred[i])~
        if diff <= delta {
            sum = sum + 0.5 * diff * diff~
        } else {
            sum = sum + delta * (diff - 0.5 * delta)~
        }
    }
    return sum / len(y_true)~
}
```

---

## Gradient Computation

```aegis
// ═══════════════════════════════════════════════════
// Gradient Computation from Scratch
// ═══════════════════════════════════════════════════

// Numerical gradient (finite differences)
fn numerical_gradient(f, x, epsilon) {
    let grad = zeros(len(x))~
    for i in 0..len(x) {
        let x_plus = copy(x)~
        let x_minus = copy(x)~
        x_plus[i] = x_plus[i] + epsilon~
        x_minus[i] = x_minus[i] - epsilon~
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)~
    }
    return grad~
}

// MSE gradient (analytical)
fn mse_gradient(X, y, weights) {
    let n = len(y)~
    let d = len(weights)~
    let grad = zeros(d)~
    
    for i in 0..n {
        let pred = dot(X[i], weights)~
        let error = pred - y[i]~
        for j in 0..d {
            grad[j] = grad[j] + 2 * error * X[i][j] / n~
        }
    }
    
    return grad~
}

// Binary cross-entropy gradient
fn bce_gradient(X, y, weights) {
    let n = len(y)~
    let d = len(weights)~
    let grad = zeros(d)~
    
    for i in 0..n {
        let z = dot(X[i], weights)~
        let pred = sigmoid(z)~
        let error = pred - y[i]~
        for j in 0..d {
            grad[j] = grad[j] + error * X[i][j] / n~
        }
    }
    
    return grad~
}
```

---

## Optimizers

```aegis
// ═══════════════════════════════════════════════════
// Optimizers from Scratch
// ═══════════════════════════════════════════════════

// Vanilla Gradient Descent
fn gradient_descent(f, x0, lr, epsilon) {
    let x = copy(x0)~
    
    🦭 until convergence(epsilon) {
        let grad = numerical_gradient(f, x, 1e-5)~
        for i in 0..len(x) {
            x[i] = x[i] - lr * grad[i]~
        }
    }
    
    return x~
}

// Stochastic Gradient Descent
fn sgd(X, y, batch_size, lr, epsilon) {
    let weights = zeros(len(X[0]))~
    let n = len(X)~
    
    🦭 until convergence(epsilon) {
        // Random batch
        let indices = sample_indices(n, batch_size)~
        let X_batch = select(X, indices)~
        let y_batch = select(y, indices)~
        
        // Gradient on batch
        let grad = mse_gradient(X_batch, y_batch, weights)~
        
        // Update
        weights = vadd(weights, vscale(grad, -lr))~
    }
    
    return weights~
}

// SGD with Momentum
fn sgd_momentum(X, y, lr, momentum, epsilon) {
    let weights = zeros(len(X[0]))~
    let velocity = zeros(len(X[0]))~
    
    🦭 until convergence(epsilon) {
        let grad = mse_gradient(X, y, weights)~
        
        // Update velocity
        velocity = vadd(vscale(velocity, momentum), grad)~
        
        // Update weights
        weights = vadd(weights, vscale(velocity, -lr))~
    }
    
    return weights~
}

// Adam Optimizer
fn adam(X, y, lr, beta1, beta2, epsilon) {
    let weights = zeros(len(X[0]))~
    let m = zeros(len(X[0]))~  // First moment
    let v = zeros(len(X[0]))~  // Second moment
    let t = 0~
    
    🦭 until convergence(epsilon) {
        t = t + 1~
        let grad = mse_gradient(X, y, weights)~
        
        // Update biased moments
        for i in 0..len(weights) {
            m[i] = beta1 * m[i] + (1 - beta1) * grad[i]~
            v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i]~
        }
        
        // Bias correction
        let m_hat = vscale(m, 1 / (1 - pow(beta1, t)))~
        let v_hat = vscale(v, 1 / (1 - pow(beta2, t)))~
        
        // Update weights
        for i in 0..len(weights) {
            weights[i] = weights[i] - lr * m_hat[i] / (sqrt(v_hat[i]) + 1e-8)~
        }
    }
    
    return weights~
}

// RMSprop
fn rmsprop(X, y, lr, decay, epsilon) {
    let weights = zeros(len(X[0]))~
    let cache = zeros(len(X[0]))~
    
    🦭 until convergence(epsilon) {
        let grad = mse_gradient(X, y, weights)~
        
        // Update cache
        for i in 0..len(weights) {
            cache[i] = decay * cache[i] + (1 - decay) * grad[i] * grad[i]~
            weights[i] = weights[i] - lr * grad[i] / (sqrt(cache[i]) + 1e-8)~
        }
    }
    
    return weights~
}
```

---

## Complete Models

### Linear Regression

```aegis
// Complete Linear Regression from Scratch
fn linear_regression(X, y) {
    let n = len(X)~
    let d = len(X[0])~
    
    // Add bias column
    let X_bias = zeros_matrix(n, d + 1)~
    for i in 0..n {
        X_bias[i][0] = 1.0~  // Bias term
        for j in 0..d {
            X_bias[i][j + 1] = X[i][j]~
        }
    }
    
    // Solve via gradient descent
    let weights = zeros(d + 1)~
    let lr = 0.01~
    
    🦭 until convergence(1e-6) {
        let grad = mse_gradient(X_bias, y, weights)~
        weights = vadd(weights, vscale(grad, -lr))~
    }
    
    return {
        bias: weights[0],
        coefficients: weights[1:]
    }~
}
```

### Logistic Regression

```aegis
// Logistic Regression from Scratch
fn sigmoid(z) {
    return 1.0 / (1.0 + exp(-z))~
}

fn logistic_regression(X, y, lr) {
    let weights = zeros(len(X[0]))~
    let bias = 0.0~
    
    🦭 until convergence(1e-6) {
        let total_grad_w = zeros(len(weights))~
        let total_grad_b = 0.0~
        
        for i in 0..len(X) {
            let z = dot(X[i], weights) + bias~
            let pred = sigmoid(z)~
            let error = pred - y[i]~
            
            for j in 0..len(weights) {
                total_grad_w[j] = total_grad_w[j] + error * X[i][j]~
            }
            total_grad_b = total_grad_b + error~
        }
        
        // Update
        let n = len(X)~
        for j in 0..len(weights) {
            weights[j] = weights[j] - lr * total_grad_w[j] / n~
        }
        bias = bias - lr * total_grad_b / n~
    }
    
    return { weights: weights, bias: bias }~
}

fn logistic_predict(model, x) {
    let z = dot(x, model.weights) + model.bias~
    return if sigmoid(z) >= 0.5 { 1 } else { 0 }~
}
```

### K-Means Clustering

```aegis
// K-Means from Scratch
fn kmeans(X, k) {
    let n = len(X)~
    let d = len(X[0])~
    
    // Initialize centroids randomly
    let indices = sample_indices(n, k)~
    let centroids = []~
    for i in 0..k {
        centroids[i] = copy(X[indices[i]])~
    }
    
    let labels = zeros(n)~
    
    🦭 until convergence(1e-6) {
        let changed = 0~
        
        // Assignment step
        for i in 0..n {
            let best_cluster = 0~
            let best_dist = 1e10~
            
            for j in 0..k {
                let d = distance(X[i], centroids[j])~
                if d < best_dist {
                    best_dist = d~
                    best_cluster = j~
                }
            }
            
            if labels[i] != best_cluster {
                labels[i] = best_cluster~
                changed = changed + 1~
            }
        }
        
        // Update step
        for j in 0..k {
            let sum = zeros(d)~
            let count = 0~
            
            for i in 0..n {
                if labels[i] == j {
                    sum = vadd(sum, X[i])~
                    count = count + 1~
                }
            }
            
            if count > 0 {
                centroids[j] = vscale(sum, 1.0 / count)~
            }
        }
        
        if changed == 0 {
            break~
        }
    }
    
    return { centroids: centroids, labels: labels }~
}
```

### Decision Stump (for boosting)

```aegis
// Decision Stump from Scratch
fn decision_stump(X, y, weights) {
    let best_feature = 0~
    let best_threshold = 0~
    let best_error = 1e10~
    let best_direction = 1~
    
    for f in 0..len(X[0]) {
        // Get unique values for this feature
        let values = unique(column(X, f))~
        
        for v in values {
            for direction in [-1, 1] {
                let error = 0.0~
                
                for i in 0..len(X) {
                    let pred = if direction * X[i][f] < direction * v { 1 } else { -1 }~
                    if pred != y[i] {
                        error = error + weights[i]~
                    }
                }
                
                if error < best_error {
                    best_error = error~
                    best_feature = f~
                    best_threshold = v~
                    best_direction = direction~
                }
            }
        }
    }
    
    return {
        feature: best_feature,
        threshold: best_threshold,
        direction: best_direction,
        error: best_error
    }~
}
```

---

## Neural Networks

### Activation Functions

```aegis
// ═══════════════════════════════════════════════════
// Activation Functions from Scratch
// ═══════════════════════════════════════════════════

fn relu(x) {
    let result = zeros(len(x))~
    for i in 0..len(x) {
        result[i] = if x[i] > 0 { x[i] } else { 0 }~
    }
    return result~
}

fn relu_derivative(x) {
    let result = zeros(len(x))~
    for i in 0..len(x) {
        result[i] = if x[i] > 0 { 1 } else { 0 }~
    }
    return result~
}

fn sigmoid_vec(x) {
    let result = zeros(len(x))~
    for i in 0..len(x) {
        result[i] = 1.0 / (1.0 + exp(-x[i]))~
    }
    return result~
}

fn sigmoid_derivative(x) {
    let s = sigmoid_vec(x)~
    let result = zeros(len(x))~
    for i in 0..len(x) {
        result[i] = s[i] * (1 - s[i])~
    }
    return result~
}

fn tanh_vec(x) {
    let result = zeros(len(x))~
    for i in 0..len(x) {
        let e_pos = exp(x[i])~
        let e_neg = exp(-x[i])~
        result[i] = (e_pos - e_neg) / (e_pos + e_neg)~
    }
    return result~
}

fn softmax(x) {
    let max_x = max(x)~
    let exp_x = zeros(len(x))~
    let sum_exp = 0.0~
    
    for i in 0..len(x) {
        exp_x[i] = exp(x[i] - max_x)~
        sum_exp = sum_exp + exp_x[i]~
    }
    
    for i in 0..len(x) {
        exp_x[i] = exp_x[i] / sum_exp~
    }
    
    return exp_x~
}
```

### Multi-Layer Perceptron

```aegis
// Complete MLP from Scratch
fn mlp_train(X, y, hidden_sizes, lr) {
    // Initialize layers
    let layers = []~
    let prev_size = len(X[0])~
    
    for size in hidden_sizes {
        layers.push({
            weights: random_matrix(prev_size, size, -0.5, 0.5),
            bias: zeros(size)
        })~
        prev_size = size~
    }
    // Output layer
    layers.push({
        weights: random_matrix(prev_size, 1, -0.5, 0.5),
        bias: zeros(1)
    })~
    
    🦭 until convergence(1e-4) {
        let total_loss = 0.0~
        
        for i in 0..len(X) {
            // ═══ Forward Pass ═══
            let activations = [X[i]]~
            let z_values = []~
            
            for l in 0..len(layers) {
                let z = vadd(matvec(transpose(layers[l].weights), activations[-1]), layers[l].bias)~
                z_values.push(z)~
                
                if l < len(layers) - 1 {
                    activations.push(relu(z))~
                } else {
                    activations.push(z)~  // Linear output
                }
            }
            
            let output = activations[-1][0]~
            let error = output - y[i]~
            total_loss = total_loss + error * error~
            
            // ═══ Backward Pass ═══
            let delta = [error]~
            
            for l in reverse(0..len(layers)) {
                // Gradient for this layer
                if l < len(layers) - 1 {
                    delta = hadamard(delta, relu_derivative(z_values[l]))~
                }
                
                // Update weights
                let grad_w = outer(activations[l], delta)~
                layers[l].weights = matrix_sub(layers[l].weights, matrix_scale(grad_w, lr))~
                layers[l].bias = vadd(layers[l].bias, vscale(delta, -lr))~
                
                // Propagate delta
                if l > 0 {
                    delta = matvec(layers[l].weights, delta)~
                }
            }
        }
        
        if total_loss / len(X) < 1e-6 {
            break~
        }
    }
    
    return layers~
}

fn mlp_predict(layers, x) {
    let current = x~
    for l in 0..len(layers) {
        let z = vadd(matvec(transpose(layers[l].weights), current), layers[l].bias)~
        if l < len(layers) - 1 {
            current = relu(z)~
        } else {
            current = z~
        }
    }
    return current[0]~
}
```

---

## Topological Extensions

### Betti-Regularized Training

```aegis
// Neural network with topological regularization
fn topo_train(X, y, hidden_sizes, lr, topo_weight) {
    let layers = init_layers(X, hidden_sizes)~
    
    🦭 until convergence(1e-5) {
        // Standard forward pass
        let predictions = []~
        for i in 0..len(X) {
            predictions.push(mlp_forward(layers, X[i]))~
        }
        
        // Standard MSE loss
        let mse_loss = mse(y, predictions)~
        
        // ═══ Topological Regularization ═══
        // Embed predictions into manifold
        manifold M = embed(predictions, dim=3, tau=1)~
        let (beta_0, beta_1) = M.shape()~
        
        // Penalty for fragmented predictions (beta_0 > 1)
        let topo_loss = topo_weight * max(0, beta_0 - 1)~
        
        // Total loss
        let total_loss = mse_loss + topo_loss~
        
        // Backprop with modified gradient
        layers = backprop(layers, total_loss, lr)~
    }
    
    return layers~
}
```

### Seal-Loop Hyperparameter Tuning

```aegis
// Hyperparameter tuning via topological convergence
fn seal_tune(X, y, param_grid) {
    let best_params = {}~
    let best_score = -1e10~
    
    🦭 until convergence(0.001) {
        // Sample random hyperparameters
        let params = {
            lr: random(param_grid.lr_min, param_grid.lr_max),
            hidden_size: random_int(param_grid.hidden_min, param_grid.hidden_max),
            batch_size: random_choice(param_grid.batch_sizes)
        }~
        
        // Train with these params
        let model = train_with_params(X, y, params)~
        let score = cross_validate(model, X, y, 5)~
        
        if score > best_score {
            best_score = score~
            best_params = params~
        }
        
        // Narrow search space around best
        param_grid = narrow_grid(param_grid, best_params)~
    }
    
    return best_params~
}
```

---

## Summary

This document proves AEGIS can implement **any ML algorithm from scratch**:

| Category | Algorithms |
|----------|------------|
| **Linear Algebra** | Dot, norm, matmul, transpose, outer |
| **Statistics** | Mean, variance, covariance, correlation |
| **Loss Functions** | MSE, MAE, BCE, CCE, Hinge, Huber |
| **Gradients** | Numerical, MSE, Binary CE |
| **Optimizers** | GD, SGD, Momentum, Adam, RMSprop |
| **Models** | Linear/Logistic Regression, K-Means, Stump |
| **Neural Nets** | Activations, MLP, Backprop |
| **AEGIS Exclusive** | Betti regularization, Seal-loop tuning |

All using `~` terminators and `🦭` seal loops. **From scratch. In AEGIS.** 🦭

---

## See Also

- [ML Library Reference](ML_LIBRARY.md)
- [ML Tasks Encyclopedia](ML_TASKS.md)
- [AETHER Integration](ML_AETHER.md)
