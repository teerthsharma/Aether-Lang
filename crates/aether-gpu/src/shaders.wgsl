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

// Tiled matmul using workgroup shared memory.
//
// The naive kernel above re-reads A and B from global memory once per k step.
// This one stages 16x16 tiles into workgroup memory, so each loaded element is
// reused 16 times. Same result, different summation order: the accumulation is
// now per-tile, so f32 rounding differs from the naive kernel and from the CPU
// reference. It is still a fixed order, so it is still bitwise deterministic.
//
// Both barriers sit in uniform control flow. Bounds checks write 0.0 into the
// tile rather than returning early, because an invocation that returns before
// a workgroupBarrier hangs the ones that do not.
const TILE: u32 = 16u;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn matmul_tiled(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x * TILE + lid.x;
    let col = wid.y * TILE + lid.y;

    var sum = 0.0;
    let num_tiles = (dims.k + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE + lid.y;
        let b_row = t * TILE + lid.x;

        if (row < dims.m && a_col < dims.k) {
            tile_a[lid.x][lid.y] = a[row * dims.k + a_col];
        } else {
            tile_a[lid.x][lid.y] = 0.0;
        }

        if (b_row < dims.k && col < dims.n) {
            tile_b[lid.x][lid.y] = b[b_row * dims.n + col];
        } else {
            tile_b[lid.x][lid.y] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < TILE; i = i + 1u) {
            sum = sum + tile_a[lid.x][i] * tile_b[i][lid.y];
        }

        workgroupBarrier();
    }

    if (row < dims.m && col < dims.n) {
        c[row * dims.n + col] = sum;
    }
}

// Squared Euclidean distance matrix: D[i,j] = ||x_i - x_j||^2 over an [m, k]
// row-major cloud. Output is [m, m].
//
// This is the O(n^2 d) term that dominates every Vietoris-Rips filtration in
// aether-core, so it is the operation where a GPU actually matters to this
// project rather than to a generic ML benchmark.
//
// Squared, not the root: the persistence engine sorts edges by distance, and
// sqrt is monotonic, so the ordering is identical and the root can be taken
// once on whatever survives rather than n^2 times here.
@compute @workgroup_size(16, 16, 1)
fn pairwise_sqdist(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;

    if (i >= dims.m || j >= dims.m) {
        return;
    }

    var sum = 0.0;
    for (var d: u32 = 0u; d < dims.k; d = d + 1u) {
        let delta = a[i * dims.k + d] - a[j * dims.k + d];
        sum = sum + delta * delta;
    }

    c[i * dims.m + j] = sum;
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
