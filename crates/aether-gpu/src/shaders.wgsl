// ═══════════════════════════════════════════════════════════════════════════════
// AETHER GPU compute kernels
//
// WGSL has no f64. aether-core computes in f64, so everything here is f32 and
// every parity assertion against the CPU path uses an f32 tolerance.
//
// Matrices are row-major and dense, matching aether_core::ml::tensor::Tensor
// for the contiguous case.
// ═══════════════════════════════════════════════════════════════════════════════

struct Dims {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>          dims: Dims;

// C[m,n] = A[m,k] * B[k,n]
//
// One invocation per output element. The k-loop accumulates in a function-local
// f32, so the summation order is fixed by the loop rather than by a tree
// reduction -- this is what makes the kernel bitwise deterministic across runs.
// A split-k or atomic reduction would be faster and would break that.
@compute @workgroup_size(16, 16, 1)
fn matmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;

    if (row >= dims.m || col >= dims.n) {
        return;
    }

    var sum = 0.0;
    for (var i: u32 = 0u; i < dims.k; i = i + 1u) {
        sum = sum + a[row * dims.k + i] * b[i * dims.n + col];
    }
    c[row * dims.n + col] = sum;
}

// Elementwise C = A + B, with B broadcast over rows when it holds a single row.
// Bias addition in a dense layer is the broadcast case.
@compute @workgroup_size(256, 1, 1)
fn add_broadcast_row(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = dims.m * dims.n;

    if (idx >= total) {
        return;
    }

    let col = idx % dims.n;
    c[idx] = a[idx] + b[col];
}

// ReLU in place over the first m*n elements.
@compute @workgroup_size(256, 1, 1)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = dims.m * dims.n;

    if (idx >= total) {
        return;
    }

    c[idx] = max(a[idx], 0.0);
}

// Derivative of ReLU evaluated at the pre-activation, multiplied into an
// incoming gradient: c = grad * (pre > 0).
//
// The convention at exactly zero is 0, matching
// aether_core::ml::neural::Activation::derivative. Backward passes that
// disagree with their forward pass at the boundary train to a different
// optimum, which loss curves do not reveal.
@compute @workgroup_size(256, 1, 1)
fn relu_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = dims.m * dims.n;

    if (idx >= total) {
        return;
    }

    if (a[idx] > 0.0) {
        c[idx] = b[idx];
    } else {
        c[idx] = 0.0;
    }
}
