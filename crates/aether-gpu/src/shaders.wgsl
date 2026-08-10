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

// Logistic sigmoid, elementwise.
//
// Computed in the branch form rather than as 1/(1+exp(-x)) directly: exp of a
// large positive magnitude overflows f32 at around 88, and the naive form hits
// that for x <= -88 while the algebraically identical exp(x)/(1+exp(x)) is
// stable there. Each branch uses whichever form keeps the exponent negative.
@compute @workgroup_size(256, 1, 1)
fn sigmoid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = dims.m * dims.n;

    if (idx >= total) {
        return;
    }

    let x = a[idx];
    if (x >= 0.0) {
        c[idx] = 1.0 / (1.0 + exp(-x));
    } else {
        let e = exp(x);
        c[idx] = e / (1.0 + e);
    }
}

// Fused gradient of binary cross-entropy through a sigmoid: (sigmoid(z) - y)/m.
//
// Fused because the composition collapses. Computing sigmoid, then the BCE
// derivative, then the sigmoid derivative separately would evaluate
// p*(1-p) explicitly, which underflows to zero once p saturates and silently
// kills the gradient. The collapsed form never forms that product.
@compute @workgroup_size(256, 1, 1)
fn sigmoid_bce_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = dims.m * dims.n;

    if (idx >= total) {
        return;
    }

    let z = a[idx];
    var p: f32;
    if (z >= 0.0) {
        p = 1.0 / (1.0 + exp(-z));
    } else {
        let e = exp(z);
        p = e / (1.0 + e);
    }

    c[idx] = (p - b[idx]) / f32(dims.m);
}

// Row-wise softmax over an [m, n] matrix of logits.
//
// One invocation per row, three passes over the row: maximum, exponentiate and
// accumulate, normalise. A per-element invocation would need a cross-invocation
// reduction for the max and the sum, and at the class counts this is used for
// (single digits) the row is shorter than a workgroup.
//
// The maximum is subtracted before exponentiating. softmax is invariant to a
// constant shift in its input, so this changes nothing mathematically, and it
// bounds every exponent at or below zero so exp cannot overflow. Without it a
// logit above about 88 produces inf, and inf/inf produces NaN.
@compute @workgroup_size(64, 1, 1)
fn softmax_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;

    if (row >= dims.m) {
        return;
    }

    let base = row * dims.n;

    var mx = a[base];
    for (var j: u32 = 1u; j < dims.n; j = j + 1u) {
        mx = max(mx, a[base + j]);
    }

    var sum = 0.0;
    for (var j: u32 = 0u; j < dims.n; j = j + 1u) {
        let e = exp(a[base + j] - mx);
        c[base + j] = e;
        sum = sum + e;
    }

    for (var j: u32 = 0u; j < dims.n; j = j + 1u) {
        c[base + j] = c[base + j] / sum;
    }
}

// Fused gradient of categorical cross-entropy through a softmax:
// (softmax(z) - y) / m, with y one-hot and m the batch size.
//
// Fused for the same reason as the binary case. Composed, the softmax Jacobian
// is an n-by-n matrix per row and the product with the cross-entropy derivative
// collapses to this difference; forming the Jacobian explicitly is both slower
// and numerically worse, because its diagonal p_i*(1-p_i) underflows once the
// softmax saturates.
@compute @workgroup_size(64, 1, 1)
fn softmax_xent_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;

    if (row >= dims.m) {
        return;
    }

    let base = row * dims.n;

    var mx = a[base];
    for (var j: u32 = 1u; j < dims.n; j = j + 1u) {
        mx = max(mx, a[base + j]);
    }

    var sum = 0.0;
    for (var j: u32 = 0u; j < dims.n; j = j + 1u) {
        sum = sum + exp(a[base + j] - mx);
    }

    for (var j: u32 = 0u; j < dims.n; j = j + 1u) {
        let p = exp(a[base + j] - mx) / sum;
        c[base + j] = (p - b[base + j]) / f32(dims.m);
    }
}

// C = transpose(A), where A is [m, n] and C is [n, m].
//
// A transpose is pure data movement, so on its own it is not worth a dispatch.
// It earns one here only because the alternative inside a training step is a
// readback: the backward pass needs A^T for a matmul whose operands are already
// resident, and doing the transpose on the host would drag the whole tensor
// across the bus and back to avoid a few microseconds of arithmetic.
@compute @workgroup_size(16, 16, 1)
fn transpose(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;

    if (row >= dims.m || col >= dims.n) {
        return;
    }

    c[col * dims.m + row] = a[row * dims.n + col];
}

// Column sums of an [m, n] matrix, one invocation per column. Used for the bias
// gradient, which is the sum of the incoming gradient down the batch axis.
@compute @workgroup_size(256, 1, 1)
fn column_sums(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;

    if (col >= dims.n) {
        return;
    }

    var sum = 0.0;
    for (var r: u32 = 0u; r < dims.m; r = r + 1u) {
        sum = sum + a[r * dims.n + col];
    }
    c[col] = sum;
}

// SGD update in place: c = a - lr * b, with lr passed in the unused dims slot
// reinterpreted as a float. Keeps the parameter update on the device so weights
// never leave it across an epoch.
@compute @workgroup_size(256, 1, 1)
fn sgd_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = dims.m * dims.n;

    if (idx >= total) {
        return;
    }

    c[idx] = a[idx] - bitcast<f32>(dims._pad) * b[idx];
}

// ── Adam ──────────────────────────────────────────────────────────────────────
//
// Two dispatches, because the bind group carries three storage buffers and Adam
// needs parameters, gradients, and two moment estimates. The moments are packed
// into a single state tensor of 2N floats -- first moment in [0, N), second in
// [N, 2N) -- so each dispatch stays within three buffers.
//
// Beta and epsilon are compile-time constants at their standard values. The
// learning rate arrives bitcast through the unused dims slot, and dims.k
// carries the step count, which the bias correction needs.

const ADAM_B1: f32 = 0.9;
const ADAM_B2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;

// state_out = [b1*m + (1-b1)*g , b2*v + (1-b2)*g^2]
//
// a: state in, 2N. b: gradient, N. c: state out, 2N. dims.n = N.
@compute @workgroup_size(256, 1, 1)
fn adam_moments(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = dims.n;

    if (idx >= n) {
        return;
    }

    let g = b[idx];
    c[idx] = ADAM_B1 * a[idx] + (1.0 - ADAM_B1) * g;
    c[n + idx] = ADAM_B2 * a[n + idx] + (1.0 - ADAM_B2) * g * g;
}

// param_out = param - lr * mhat / (sqrt(vhat) + eps)
//
// a: parameters, N. b: state, 2N. c: parameters out, N. dims.n = N,
// dims.k = step (1-based), dims._pad = learning rate as f32 bits.
//
// The bias correction is the part that is easy to omit and hard to notice. At
// step 1 the first moment is 0.1*g, so without correction the first update is
// an order of magnitude too small, and the model trains -- just slower, from a
// worse start. Dividing by (1 - b1^t) restores the scale, and the correction
// decays to nothing as t grows, which is why a late-training comparison cannot
// see whether it is there.
@compute @workgroup_size(256, 1, 1)
fn adam_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = dims.n;

    if (idx >= n) {
        return;
    }

    let t = f32(dims.k);
    let lr = bitcast<f32>(dims._pad);

    let mhat = b[idx] / (1.0 - pow(ADAM_B1, t));
    let vhat = b[n + idx] / (1.0 - pow(ADAM_B2, t));

    // Epsilon is added to sqrt(vhat), not inside the root. Inside, it would
    // change the effective step for every parameter rather than only guarding
    // the division, and the two are not the same optimiser.
    c[idx] = a[idx] - lr * mhat / (sqrt(vhat) + ADAM_EPS);
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

// ═══════════════════════════════════════════════════════════════════════════════
// Topology-scheduled attention
//
// The port of `aether_core::scheduled::scheduled_attention`: flash-style online
// softmax over a CSR block schedule, causally masked. This is the repository's
// headline mechanism, and until now it ran only on the CPU while the GPU did
// generic MLP work.
//
// Packing. Every kernel in this file shares one four-binding layout, and
// attention needs six arrays. Rather than introduce a second layout for one
// kernel, the operands are concatenated: `a` holds q ‖ k ‖ v and `b` holds
// offsets ‖ indices. The four `Dims` slots hold exactly the four sizes needed,
// so nothing is stolen from the uniform either.
//
// Schedule indices travel as f32. They are block numbers bounded by num_blocks,
// and every integer below 2^24 is exact in f32, so the conversion is lossless
// for any schedule that fits in memory by a wide margin.
//
// One invocation per query row. The row owns its running max, denominator and
// accumulator, so there is no cross-thread reduction and therefore no atomics --
// the same reason `matmul` is deterministic applies here.
// ═══════════════════════════════════════════════════════════════════════════════

// Bounds on the private scratch arrays. A kernel cannot allocate dynamically, so
// these are ceilings the host must check rather than negotiate.
const MAX_HEAD_DIM: u32 = 128u;
const MAX_BLOCK: u32 = 128u;

// f32::MIN as the empty-tile sentinel. A real score reaching it would need a dot
// product at the edge of the type, which the scale factor makes unreachable.
const NEG_SENTINEL: f32 = -3.4028235e38;

@compute @workgroup_size(64, 1, 1)
fn scheduled_attention(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let seq = dims.m;
    let head_dim = dims.k;
    let block_size = dims.n;
    let num_blocks = dims._pad;

    if (row >= seq) {
        return;
    }

    let q_base = 0u;
    let k_base = seq * head_dim;
    let v_base = 2u * seq * head_dim;
    let idx_base = num_blocks + 1u;

    let q_block = row / block_size;
    let scale = 1.0 / sqrt(f32(head_dim));

    var acc: array<f32, 128>;
    var scores: array<f32, 128>;
    for (var d = 0u; d < head_dim; d = d + 1u) {
        acc[d] = 0.0;
    }
    var running_max = NEG_SENTINEL;
    var denom = 0.0;

    let start = u32(b[q_block]);
    let end = u32(b[q_block + 1u]);

    for (var e = start; e < end; e = e + 1u) {
        let k_block = u32(b[idx_base + e]);

        // Score tile for this (query row, key block), causally masked.
        var tile_max = NEG_SENTINEL;
        for (var n = 0u; n < block_size; n = n + 1u) {
            let col = k_block * block_size + n;
            if (col > row) {
                scores[n] = NEG_SENTINEL;
                continue;
            }
            var dot = 0.0;
            for (var d = 0u; d < head_dim; d = d + 1u) {
                dot = dot + a[q_base + row * head_dim + d] * a[k_base + col * head_dim + d];
            }
            scores[n] = dot * scale;
            if (scores[n] > tile_max) {
                tile_max = scores[n];
            }
        }

        // Entirely in the future. Folding it in would rescale by
        // exp(-inf - -inf), which is NaN rather than the zero it should be.
        if (tile_max == NEG_SENTINEL) {
            continue;
        }

        let previous_max = running_max;
        let new_max = max(previous_max, tile_max);
        // exp(sentinel - finite) underflows to zero anyway; the branch states
        // the intent for the first block a row ever sees.
        var alpha = 0.0;
        if (previous_max != NEG_SENTINEL) {
            alpha = exp(previous_max - new_max);
        }

        denom = denom * alpha;
        for (var d = 0u; d < head_dim; d = d + 1u) {
            acc[d] = acc[d] * alpha;
        }

        for (var n = 0u; n < block_size; n = n + 1u) {
            if (scores[n] == NEG_SENTINEL) {
                continue;
            }
            let weight = exp(scores[n] - new_max);
            denom = denom + weight;
            let col = k_block * block_size + n;
            for (var d = 0u; d < head_dim; d = d + 1u) {
                acc[d] = acc[d] + weight * a[v_base + col * head_dim + d];
            }
        }
        running_max = new_max;
    }

    // `denom` is positive for every row: a causal schedule always includes the
    // diagonal block, so a row always attends to at least itself.
    for (var d = 0u; d < head_dim; d = d + 1u) {
        c[row * head_dim + d] = acc[d] / denom;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scheduled attention, reverse mode
//
// The port of `aether_core::scheduled::scheduled_attention_backward`. Four entry
// points, because the shared layout carries one output buffer and there are three
// gradients plus the per-row statistics they all need.
//
// No atomics anywhere, which is a design constraint rather than an optimisation.
// `dq` accumulates over the columns a query row sees, so one invocation per query
// row owns its whole result. `dk` and `dv` accumulate over the query rows that see
// a column, so those run one invocation per *key* row instead. Both directions
// keep every accumulation thread-local, which is what makes these deterministic
// for the same reason `matmul` is.
//
// The shorter alternative -- one thread per query row writing into `dk` through
// atomics -- would make the result depend on scheduling order. This crate has
// already recorded that a non-deterministic kernel invalidates every A/B
// comparison made with it, because part of the measured difference becomes the
// kernel disagreeing with itself.
//
// `attention_row_stats` exists so `dk` and `dv` need not rebuild a query row's
// softmax to touch one of its columns. Without it each (row, column) pair would
// cost a full sweep of that row's scheduled blocks, making the kernels quadratic
// in the sequence where the CPU reference is linear in the scheduled work. With
// it every entry point costs O(head_dim) per pair, matching the reference.
//
// Packing: `a` holds q ‖ k ‖ v ‖ dOut ‖ stats, where stats is three floats per
// row -- running maximum, log-sum-exp, and the delta term. The stats kernel
// writes them and does not read them.
// ═══════════════════════════════════════════════════════════════════════════════

// Row maximum, log-sum-exp, and delta = sum_j p_ij (dOut_i . V_j).
//
// Delta is what makes the softmax Jacobian a rank-one correction rather than a
// dense matrix. It is a per-row scalar, so computing it once here is what lets
// three kernels share it.
//
// **This kernel reads and writes the same buffer, and that is the point.** It
// takes the packed operands as `c` rather than `a`, reads q, k, v and dOut from
// the front of it, and writes the statistics into the tail that the upload left
// reserved. The three gradient kernels then bind that identical buffer as `a`
// and find everything already in place.
//
// The alternative, and what this replaces, was writing the statistics to their
// own output, reading them back to the host, and re-uploading them concatenated
// onto the operands -- a full round trip in the middle of a backward pass, purely
// because the shared four-binding layout has no free binding to leave them in.
// A `read_write` binding is readable, so the round trip was never necessary.
//
// The aliasing is safe because each invocation owns one row: it reads only the
// operand region, which nothing writes, and writes only its own three floats in
// the tail, which nothing reads until the next dispatch.
@compute @workgroup_size(64, 1, 1)
fn attention_row_stats(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let seq = dims.m;
    let head_dim = dims.k;
    let block_size = dims.n;
    let num_blocks = dims._pad;

    if (row >= seq) {
        return;
    }

    let k_base = seq * head_dim;
    let v_base = 2u * seq * head_dim;
    let d_base = 3u * seq * head_dim;
    let s_base = 4u * seq * head_dim;
    let idx_base = num_blocks + 1u;

    let q_block = row / block_size;
    let scale = 1.0 / sqrt(f32(head_dim));
    let start = u32(b[q_block]);
    let end = u32(b[q_block + 1u]);

    var mx = NEG_SENTINEL;
    for (var e = start; e < end; e = e + 1u) {
        let k_block = u32(b[idx_base + e]);
        for (var n = 0u; n < block_size; n = n + 1u) {
            let col = k_block * block_size + n;
            if (col > row) {
                continue;
            }
            var dot = 0.0;
            for (var d = 0u; d < head_dim; d = d + 1u) {
                dot = dot + c[row * head_dim + d] * c[k_base + col * head_dim + d];
            }
            let s = dot * scale;
            if (s > mx) {
                mx = s;
            }
        }
    }

    var denom = 0.0;
    var weighted = 0.0;
    for (var e = start; e < end; e = e + 1u) {
        let k_block = u32(b[idx_base + e]);
        for (var n = 0u; n < block_size; n = n + 1u) {
            let col = k_block * block_size + n;
            if (col > row) {
                continue;
            }
            var dot = 0.0;
            var dp = 0.0;
            for (var d = 0u; d < head_dim; d = d + 1u) {
                dot = dot + c[row * head_dim + d] * c[k_base + col * head_dim + d];
                dp = dp + c[d_base + row * head_dim + d] * c[v_base + col * head_dim + d];
            }
            let w = exp(dot * scale - mx);
            denom = denom + w;
            weighted = weighted + w * dp;
        }
    }

    c[s_base + row * 3u + 0u] = mx;
    c[s_base + row * 3u + 1u] = mx + log(denom);
    c[s_base + row * 3u + 2u] = weighted / denom;
}

// dQ_i = sum_j p_ij (dOut_i . V_j - delta_i) * scale * K_j
@compute @workgroup_size(64, 1, 1)
fn attention_dq(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let seq = dims.m;
    let head_dim = dims.k;
    let block_size = dims.n;
    let num_blocks = dims._pad;

    if (row >= seq) {
        return;
    }

    let k_base = seq * head_dim;
    let v_base = 2u * seq * head_dim;
    let d_base = 3u * seq * head_dim;
    let s_base = 4u * seq * head_dim;
    let idx_base = num_blocks + 1u;

    let q_block = row / block_size;
    let scale = 1.0 / sqrt(f32(head_dim));
    let lse = a[s_base + row * 3u + 1u];
    let delta = a[s_base + row * 3u + 2u];

    var acc: array<f32, 128>;
    for (var d = 0u; d < head_dim; d = d + 1u) {
        acc[d] = 0.0;
    }

    let start = u32(b[q_block]);
    let end = u32(b[q_block + 1u]);
    for (var e = start; e < end; e = e + 1u) {
        let k_block = u32(b[idx_base + e]);
        for (var n = 0u; n < block_size; n = n + 1u) {
            let col = k_block * block_size + n;
            if (col > row) {
                continue;
            }
            var dot = 0.0;
            var dp = 0.0;
            for (var d = 0u; d < head_dim; d = d + 1u) {
                dot = dot + a[row * head_dim + d] * a[k_base + col * head_dim + d];
                dp = dp + a[d_base + row * head_dim + d] * a[v_base + col * head_dim + d];
            }
            let p = exp(dot * scale - lse);
            let ds = p * (dp - delta) * scale;
            for (var d = 0u; d < head_dim; d = d + 1u) {
                acc[d] = acc[d] + ds * a[k_base + col * head_dim + d];
            }
        }
    }

    for (var d = 0u; d < head_dim; d = d + 1u) {
        c[row * head_dim + d] = acc[d];
    }
}

// dK_j = sum_i p_ij (dOut_i . V_j - delta_i) * scale * Q_i
//
// One invocation per key row, walking the query blocks that could see it. The
// membership test runs once per query block rather than once per query row, since
// every row in a block shares that block's schedule.
@compute @workgroup_size(64, 1, 1)
fn attention_dk(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let seq = dims.m;
    let head_dim = dims.k;
    let block_size = dims.n;
    let num_blocks = dims._pad;

    if (col >= seq) {
        return;
    }

    let k_base = seq * head_dim;
    let v_base = 2u * seq * head_dim;
    let d_base = 3u * seq * head_dim;
    let s_base = 4u * seq * head_dim;
    let idx_base = num_blocks + 1u;

    let k_block = col / block_size;
    let scale = 1.0 / sqrt(f32(head_dim));

    var acc: array<f32, 128>;
    for (var d = 0u; d < head_dim; d = d + 1u) {
        acc[d] = 0.0;
    }

    for (var q_block = k_block; q_block < num_blocks; q_block = q_block + 1u) {
        var present = false;
        let start = u32(b[q_block]);
        let end = u32(b[q_block + 1u]);
        for (var e = start; e < end; e = e + 1u) {
            if (u32(b[idx_base + e]) == k_block) {
                present = true;
            }
        }
        if (!present) {
            continue;
        }

        for (var m = 0u; m < block_size; m = m + 1u) {
            let row = q_block * block_size + m;
            if (row >= seq || col > row) {
                continue;
            }
            let lse = a[s_base + row * 3u + 1u];
            let delta = a[s_base + row * 3u + 2u];

            var dot = 0.0;
            var dp = 0.0;
            for (var d = 0u; d < head_dim; d = d + 1u) {
                dot = dot + a[row * head_dim + d] * a[k_base + col * head_dim + d];
                dp = dp + a[d_base + row * head_dim + d] * a[v_base + col * head_dim + d];
            }
            let p = exp(dot * scale - lse);
            let ds = p * (dp - delta) * scale;
            for (var d = 0u; d < head_dim; d = d + 1u) {
                acc[d] = acc[d] + ds * a[row * head_dim + d];
            }
        }
    }

    for (var d = 0u; d < head_dim; d = d + 1u) {
        c[col * head_dim + d] = acc[d];
    }
}

// dV_j = sum_i p_ij dOut_i
//
// The output is linear in V, so no softmax correction appears and delta is never
// read. That asymmetry is worth noticing: a bug in delta shows in dq and dk and
// leaves dv exactly right, which is why the tests check the three separately.
@compute @workgroup_size(64, 1, 1)
fn attention_dv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let seq = dims.m;
    let head_dim = dims.k;
    let block_size = dims.n;
    let num_blocks = dims._pad;

    if (col >= seq) {
        return;
    }

    let k_base = seq * head_dim;
    let d_base = 3u * seq * head_dim;
    let s_base = 4u * seq * head_dim;
    let idx_base = num_blocks + 1u;

    let k_block = col / block_size;
    let scale = 1.0 / sqrt(f32(head_dim));

    var acc: array<f32, 128>;
    for (var d = 0u; d < head_dim; d = d + 1u) {
        acc[d] = 0.0;
    }

    for (var q_block = k_block; q_block < num_blocks; q_block = q_block + 1u) {
        var present = false;
        let start = u32(b[q_block]);
        let end = u32(b[q_block + 1u]);
        for (var e = start; e < end; e = e + 1u) {
            if (u32(b[idx_base + e]) == k_block) {
                present = true;
            }
        }
        if (!present) {
            continue;
        }

        for (var m = 0u; m < block_size; m = m + 1u) {
            let row = q_block * block_size + m;
            if (row >= seq || col > row) {
                continue;
            }
            let lse = a[s_base + row * 3u + 1u];

            var dot = 0.0;
            for (var d = 0u; d < head_dim; d = d + 1u) {
                dot = dot + a[row * head_dim + d] * a[k_base + col * head_dim + d];
            }
            let p = exp(dot * scale - lse);
            for (var d = 0u; d < head_dim; d = d + 1u) {
                acc[d] = acc[d] + p * a[d_base + row * head_dim + d];
            }
        }
    }

    for (var d = 0u; d < head_dim; d = d + 1u) {
        c[col * head_dim + d] = acc[d];
    }
}
