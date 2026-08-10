//! ═══════════════════════════════════════════════════════════════════════════════
//! AETHER GPU — a wgpu compute backend for `aether-core` tensor operations
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! # What this is
//!
//! A real GPU execution path for the dense linear algebra underneath
//! `aether_core::ml`. It dispatches WGSL compute shaders through `wgpu`, so it
//! runs on Vulkan, DX12, Metal, and GL depending on what the host offers.
//!
//! # Precision
//!
//! **WGSL has no `f64`.** `aether-core` computes in `f64` throughout. Every
//! kernel here is `f32`, so results are compared against the CPU path at an
//! `f32` tolerance and never at the `1e-12` used elsewhere in this workspace.
//! This is a property of the target, not a shortcut: a GPU backend that claimed
//! `f64` parity would be claiming something the shading language cannot express.
//!
//! # Why a separate crate
//!
//! `aether-core` is `no_std` and builds for `thumbv7m-none-eabi`. `wgpu` needs
//! `std`, an allocator, and a driver stack. Keeping the GPU path in its own
//! crate means the embedded build cannot silently acquire a dependency on it.

pub mod datasets;

use std::borrow::Cow;
use std::cell::RefCell;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Commands recorded but not yet submitted.
///
/// Every resident operation used to build its own encoder and submit it, so a
/// training step cost one queue submission per operation. Submission is not
/// free: it is a driver transition, and at this network's size the step was
/// dominated by making twenty of them rather than by arithmetic.
///
/// Work now accumulates into a single encoder and is submitted once, at the
/// next [`GpuContext::read`] or [`GpuContext::flush`].
///
/// `bind_groups` and `dims_buffers` are held until submission. The recorded
/// commands reference them, and dropping them early would free resources the
/// queue is about to read.
#[derive(Default)]
struct Pending {
    encoder: Option<wgpu::CommandEncoder>,
    bind_groups: Vec<wgpu::BindGroup>,
    dims_buffers: Vec<wgpu::Buffer>,
    recorded: usize,
}

/// Uniform block matching `struct Dims` in `shaders.wgsl`.
///
/// `repr(C)` and the explicit pad keep the Rust and WGSL layouts identical;
/// a mismatch here reads as garbage dimensions rather than as a compile error.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Dims {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

/// Everything that failed before a kernel could run.
#[derive(Debug)]
pub enum GpuError {
    /// No adapter satisfied the request. On a headless box with no drivers this
    /// is the expected outcome, and callers that must not fail should fall back
    /// to the CPU path rather than treating it as fatal.
    NoAdapter,
    /// An adapter existed but a device could not be created from it.
    NoDevice(String),
    /// Shape arguments that no dispatch could satisfy.
    ShapeMismatch(String),
    /// The readback buffer could not be mapped.
    Readback(String),
}

impl core::fmt::Display for GpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GpuError::NoAdapter => write!(f, "no wgpu adapter available"),
            GpuError::NoDevice(e) => write!(f, "could not create wgpu device: {e}"),
            GpuError::ShapeMismatch(e) => write!(f, "shape mismatch: {e}"),
            GpuError::Readback(e) => write!(f, "buffer readback failed: {e}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// Which physical device the kernels are running on.
///
/// Recorded and reported rather than assumed: a benchmark that does not name
/// its adapter is a benchmark that cannot be reproduced, and `wgpu` will
/// silently hand back a software rasterizer if that is all it can find.
#[derive(Clone, Debug)]
pub struct AdapterInfo {
    pub name: String,
    pub backend: String,
    pub device_type: String,
}

impl AdapterInfo {
    /// Whether this is real hardware rather than a CPU implementation.
    ///
    /// `wgpu` exposes `DeviceType::Cpu` for lavapipe/WARP-style software
    /// adapters. A "GPU speedup" measured against one of those is measuring
    /// two CPU implementations, so tests that assert hardware execution check
    /// this rather than trusting that an adapter was found.
    pub fn is_hardware(&self) -> bool {
        self.device_type != "Cpu"
    }
}

/// An initialised GPU context: adapter, device, queue, and compiled pipelines.
///
/// Construction is expensive (driver enumeration, shader compilation) and the
/// result is reusable, so build one and keep it. Creating a context per
/// operation would make every measurement a measurement of driver startup.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: AdapterInfo,
    matmul: wgpu::ComputePipeline,
    matmul_tiled: wgpu::ComputePipeline,
    pairwise_sqdist: wgpu::ComputePipeline,
    add_broadcast_row: wgpu::ComputePipeline,
    relu: wgpu::ComputePipeline,
    relu_backward: wgpu::ComputePipeline,
    transpose: wgpu::ComputePipeline,
    column_sums: wgpu::ComputePipeline,
    sgd_update: wgpu::ComputePipeline,
    sigmoid: wgpu::ComputePipeline,
    sigmoid_bce_grad: wgpu::ComputePipeline,
    softmax_rows: wgpu::ComputePipeline,
    softmax_xent_grad: wgpu::ComputePipeline,
    adam_moments: wgpu::ComputePipeline,
    adam_update: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    pending: RefCell<Pending>,
}

/// Adam's per-parameter state: both moment estimates in one tensor.
///
/// Packed rather than held as two tensors because the bind group carries three
/// storage buffers, and Adam touches parameters, gradients and both moments.
/// First moment occupies `[0, n)`, second `[n, 2n)`.
pub struct AdamState {
    moments: GpuTensor,
    step: u32,
}

impl AdamState {
    /// How many updates have been applied. Bias correction depends on it, so it
    /// is exposed rather than hidden.
    pub fn step(&self) -> u32 {
        self.step
    }
}

/// A tensor that lives in GPU memory across operations.
///
/// The slice-taking methods on [`GpuContext`] upload their inputs and read the
/// result back on every call. That is the right shape for a one-shot kernel and
/// the wrong one for a training loop: a three-layer forward pass through them
/// pays twelve PCIe crossings per step to move intermediates the GPU produced
/// and the GPU is about to consume.
///
/// A `GpuTensor` stays resident. Chain operations on it and nothing crosses the
/// bus until [`GpuContext::read`] is called.
pub struct GpuTensor {
    buffer: wgpu::Buffer,
    rows: usize,
    cols: usize,
}

impl GpuTensor {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl GpuContext {
    /// Enumerate adapters, take the highest-performance one, compile the shaders.
    ///
    /// Returns `Err(GpuError::NoAdapter)` rather than panicking when no device
    /// exists, so a caller can fall back to the CPU path.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, GpuError> {
        // Honour WGPU_BACKEND so a caller can pin Vulkan, DX12, Metal or GL.
        //
        // `InstanceDescriptor::default()` does not read it, which is worth
        // knowing: setting the variable and assuming it took effect gives a
        // measurement of whatever backend was chosen anyway. Backend selection
        // matters here because the teardown fault in FEATURES.md needs to be
        // characterised as backend-specific or not before it is worth reporting
        // upstream.
        let backends = wgpu::Backends::from_env().unwrap_or_default();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|_| GpuError::NoAdapter)?;

        let raw = adapter.get_info();
        let info = AdapterInfo {
            name: raw.name.clone(),
            backend: format!("{:?}", raw.backend),
            device_type: format!("{:?}", raw.device_type),
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("aether-gpu device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuError::NoDevice(e.to_string()))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aether-gpu kernels"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders.wgsl"))),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aether-gpu bind group layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aether-gpu pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let build = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Ok(Self {
            matmul: build("matmul"),
            matmul_tiled: build("matmul_tiled"),
            pairwise_sqdist: build("pairwise_sqdist"),
            add_broadcast_row: build("add_broadcast_row"),
            relu: build("relu"),
            relu_backward: build("relu_backward"),
            transpose: build("transpose"),
            column_sums: build("column_sums"),
            sgd_update: build("sgd_update"),
            sigmoid: build("sigmoid"),
            sigmoid_bce_grad: build("sigmoid_bce_grad"),
            softmax_rows: build("softmax_rows"),
            softmax_xent_grad: build("softmax_xent_grad"),
            adam_moments: build("adam_moments"),
            adam_update: build("adam_update"),
            device,
            queue,
            info,
            layout,
            pending: RefCell::new(Pending::default()),
        })
    }

    /// Submit everything recorded so far.
    ///
    /// Called automatically by [`GpuContext::read`]. Call it directly when
    /// device-side state must land without reading anything back -- a sequence
    /// of parameter updates whose result is not inspected until later, for
    /// instance. Recorded work that is never flushed is never executed.
    pub fn flush(&self) {
        let mut pending = self.pending.borrow_mut();
        if let Some(encoder) = pending.encoder.take() {
            self.queue.submit(Some(encoder.finish()));
        }
        pending.bind_groups.clear();
        pending.dims_buffers.clear();
        pending.recorded = 0;
    }

    /// How many dispatches are recorded and not yet submitted.
    ///
    /// Exposed so a test can assert that batching actually batches, rather than
    /// inferring it from a timing difference that a faster kernel would also
    /// produce.
    pub fn pending_dispatches(&self) -> usize {
        self.pending.borrow().recorded
    }

    // ── Resident tensors ──────────────────────────────────────────────────────

    /// Upload a host slice and keep it on the device.
    pub fn upload(&self, data: &[f32], rows: usize, cols: usize) -> Result<GpuTensor, GpuError> {
        if data.len() != rows * cols {
            return Err(GpuError::ShapeMismatch(format!(
                "data has {} elements, expected {rows}*{cols} = {}",
                data.len(),
                rows * cols
            )));
        }

        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("resident tensor"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        Ok(GpuTensor { buffer, rows, cols })
    }

    /// Allocate an uninitialised device tensor.
    fn alloc(&self, rows: usize, cols: usize) -> GpuTensor {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident tensor"),
            size: (rows * cols * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        GpuTensor { buffer, rows, cols }
    }

    /// Copy a resident tensor back to the host. The only bus crossing in a
    /// chain of resident operations.
    pub fn read(&self, t: &GpuTensor) -> Result<Vec<f32>, GpuError> {
        let bytes = (t.len() * std::mem::size_of::<f32>()) as u64;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Record the copy into whatever batch is open, so the readback rides
        // the same submission as the work that produced the data rather than
        // forcing a second one.
        {
            let mut pending = self.pending.borrow_mut();
            if pending.encoder.is_none() {
                pending.encoder = Some(self.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("aether-gpu batch"),
                    },
                ));
            }
            pending
                .encoder
                .as_mut()
                .expect("encoder just created")
                .copy_buffer_to_buffer(&t.buffer, 0, &staging, 0, bytes);
        }

        self.flush();
        self.map_and_read(&staging)
    }

    /// `C = A * B` with both operands already resident, result left resident.
    ///
    /// Uses the tiled kernel: it stages 16x16 blocks into workgroup memory so
    /// each loaded element is reused sixteen times instead of being re-read
    /// from global memory on every k step.
    pub fn matmul_resident(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, GpuError> {
        if a.cols != b.rows {
            return Err(GpuError::ShapeMismatch(format!(
                "a is {}x{}, b is {}x{}; inner dimensions must agree",
                a.rows, a.cols, b.rows, b.cols
            )));
        }

        let (m, k, n) = (a.rows, a.cols, b.cols);
        let out = self.alloc(m, n);

        self.dispatch_resident(
            &self.matmul_tiled,
            &a.buffer,
            &b.buffer,
            &out.buffer,
            Dims {
                m: m as u32,
                k: k as u32,
                n: n as u32,
                _pad: 0,
            },
            (
                m.div_ceil(16).max(1) as u32,
                n.div_ceil(16).max(1) as u32,
                1,
            ),
        );

        Ok(out)
    }

    /// Squared Euclidean distance matrix of a resident `[n, d]` cloud.
    ///
    /// This is the O(n^2 d) term underneath every Vietoris-Rips filtration in
    /// `aether-core`, which is what makes it the operation where a GPU matters
    /// to this project specifically.
    pub fn pairwise_sqdist_resident(&self, points: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let n = points.rows;
        let d = points.cols;
        let out = self.alloc(n, n);

        self.dispatch_resident(
            &self.pairwise_sqdist,
            &points.buffer,
            &points.buffer,
            &out.buffer,
            Dims {
                m: n as u32,
                k: d as u32,
                n: n as u32,
                _pad: 0,
            },
            (
                n.div_ceil(16).max(1) as u32,
                n.div_ceil(16).max(1) as u32,
                1,
            ),
        );

        Ok(out)
    }

    /// `C[m,n] = A[m,n] + bias[n]`, both resident, result resident.
    pub fn add_bias_resident(
        &self,
        a: &GpuTensor,
        bias: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        if bias.len() != a.cols {
            return Err(GpuError::ShapeMismatch(format!(
                "bias has {} elements, expected {} columns",
                bias.len(),
                a.cols
            )));
        }

        let out = self.alloc(a.rows, a.cols);
        self.dispatch_resident(
            &self.add_broadcast_row,
            &a.buffer,
            &bias.buffer,
            &out.buffer,
            self.dims_of(a),
            (a.len().div_ceil(256).max(1) as u32, 1, 1),
        );
        Ok(out)
    }

    /// Elementwise ReLU, resident.
    pub fn relu_resident(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let out = self.alloc(a.rows, a.cols);
        self.dispatch_resident(
            &self.relu,
            &a.buffer,
            &a.buffer,
            &out.buffer,
            self.dims_of(a),
            (a.len().div_ceil(256).max(1) as u32, 1, 1),
        );
        Ok(out)
    }

    /// `grad * (pre > 0)`, resident. Zero at exactly zero, matching the
    /// forward pass and `aether_core::ml::neural::Activation::derivative`.
    pub fn relu_backward_resident(
        &self,
        pre: &GpuTensor,
        grad: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        if pre.len() != grad.len() {
            return Err(GpuError::ShapeMismatch(format!(
                "pre has {} elements, grad has {}",
                pre.len(),
                grad.len()
            )));
        }

        let out = self.alloc(pre.rows, pre.cols);
        self.dispatch_resident(
            &self.relu_backward,
            &pre.buffer,
            &grad.buffer,
            &out.buffer,
            self.dims_of(pre),
            (pre.len().div_ceil(256).max(1) as u32, 1, 1),
        );
        Ok(out)
    }

    /// `[m, n]` to `[n, m]`, resident.
    pub fn transpose_resident(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let out = self.alloc(a.cols, a.rows);
        self.dispatch_resident(
            &self.transpose,
            &a.buffer,
            &a.buffer,
            &out.buffer,
            self.dims_of(a),
            (
                a.rows.div_ceil(16).max(1) as u32,
                a.cols.div_ceil(16).max(1) as u32,
                1,
            ),
        );
        Ok(out)
    }

    /// Column sums of an `[m, n]` tensor, giving `[1, n]`. The bias gradient.
    pub fn column_sums_resident(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let out = self.alloc(1, a.cols);
        self.dispatch_resident(
            &self.column_sums,
            &a.buffer,
            &a.buffer,
            &out.buffer,
            self.dims_of(a),
            (a.cols.div_ceil(256).max(1) as u32, 1, 1),
        );
        Ok(out)
    }

    /// `param - lr * grad`, resident. Keeps weights on the device across epochs.
    pub fn sgd_update_resident(
        &self,
        param: &GpuTensor,
        grad: &GpuTensor,
        lr: f32,
    ) -> Result<GpuTensor, GpuError> {
        if param.len() != grad.len() {
            return Err(GpuError::ShapeMismatch(format!(
                "param has {} elements, grad has {}",
                param.len(),
                grad.len()
            )));
        }

        let out = self.alloc(param.rows, param.cols);
        self.dispatch_resident(
            &self.sgd_update,
            &param.buffer,
            &grad.buffer,
            &out.buffer,
            Dims {
                m: param.rows as u32,
                k: 0,
                n: param.cols as u32,
                // The shader bitcasts this slot back to f32. Passing the rate
                // through the existing uniform avoids a second binding for one
                // scalar; the bitcast is the price.
                _pad: lr.to_bits(),
            },
            (param.len().div_ceil(256).max(1) as u32, 1, 1),
        );
        Ok(out)
    }

    /// Logistic sigmoid, resident.
    pub fn sigmoid_resident(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let out = self.alloc(a.rows, a.cols);
        self.dispatch_resident(
            &self.sigmoid,
            &a.buffer,
            &a.buffer,
            &out.buffer,
            self.dims_of(a),
            (a.len().div_ceil(256).max(1) as u32, 1, 1),
        );
        Ok(out)
    }

    /// `(sigmoid(logits) - targets) / rows`, resident.
    ///
    /// The fused form of the binary-cross-entropy gradient through a sigmoid.
    /// Fused because the composition collapses algebraically, and computing the
    /// pieces separately forms `p*(1-p)`, which underflows to zero once the
    /// sigmoid saturates and silently stops the network learning.
    pub fn sigmoid_bce_grad_resident(
        &self,
        logits: &GpuTensor,
        targets: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        if logits.len() != targets.len() {
            return Err(GpuError::ShapeMismatch(format!(
                "logits has {} elements, targets has {}",
                logits.len(),
                targets.len()
            )));
        }

        let out = self.alloc(logits.rows, logits.cols);
        self.dispatch_resident(
            &self.sigmoid_bce_grad,
            &logits.buffer,
            &targets.buffer,
            &out.buffer,
            Dims {
                m: logits.rows as u32,
                k: logits.cols as u32,
                n: logits.cols as u32,
                _pad: 0,
            },
            (logits.len().div_ceil(256).max(1) as u32, 1, 1),
        );
        Ok(out)
    }

    /// Row-wise softmax over `[m, classes]` logits, resident.
    ///
    /// Subtracts the row maximum before exponentiating, which leaves the result
    /// unchanged and makes overflow impossible.
    pub fn softmax_resident(&self, logits: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let out = self.alloc(logits.rows, logits.cols);
        self.dispatch_resident(
            &self.softmax_rows,
            &logits.buffer,
            &logits.buffer,
            &out.buffer,
            self.dims_of(logits),
            (logits.rows.div_ceil(64).max(1) as u32, 1, 1),
        );
        Ok(out)
    }

    /// `(softmax(logits) - one_hot) / rows`, resident.
    ///
    /// The fused categorical cross-entropy gradient. Composed, this would form
    /// the softmax Jacobian per row; fused, the product collapses to a
    /// difference that neither allocates the Jacobian nor underflows when the
    /// softmax saturates.
    pub fn softmax_xent_grad_resident(
        &self,
        logits: &GpuTensor,
        one_hot: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        if logits.rows != one_hot.rows || logits.cols != one_hot.cols {
            return Err(GpuError::ShapeMismatch(format!(
                "logits are {}x{}, targets are {}x{}",
                logits.rows, logits.cols, one_hot.rows, one_hot.cols
            )));
        }

        let out = self.alloc(logits.rows, logits.cols);
        self.dispatch_resident(
            &self.softmax_xent_grad,
            &logits.buffer,
            &one_hot.buffer,
            &out.buffer,
            self.dims_of(logits),
            (logits.rows.div_ceil(64).max(1) as u32, 1, 1),
        );
        Ok(out)
    }

    /// Zeroed Adam state sized for `param`.
    ///
    /// Both moments start at zero, which is what makes the bias correction
    /// necessary: at step 1 the first moment is only `(1 - b1)` of the
    /// gradient.
    pub fn adam_state(&self, param: &GpuTensor) -> Result<AdamState, GpuError> {
        let zeros = vec![0.0f32; param.len() * 2];
        Ok(AdamState {
            moments: self.upload(&zeros, 1, param.len() * 2)?,
            step: 0,
        })
    }

    /// One Adam step. Returns the updated parameters; `state` is advanced.
    ///
    /// Two dispatches: the moment update, then the parameter update. Both are
    /// recorded into the open batch like any other resident operation.
    pub fn adam_update_resident(
        &self,
        param: &GpuTensor,
        grad: &GpuTensor,
        state: &mut AdamState,
        lr: f32,
    ) -> Result<GpuTensor, GpuError> {
        let n = param.len();

        if grad.len() != n {
            return Err(GpuError::ShapeMismatch(format!(
                "param has {n} elements, grad has {}",
                grad.len()
            )));
        }
        if state.moments.len() != n * 2 {
            return Err(GpuError::ShapeMismatch(format!(
                "state holds {} elements, expected 2*{n} = {}",
                state.moments.len(),
                n * 2
            )));
        }

        state.step += 1;

        let next_moments = self.alloc(1, n * 2);
        self.dispatch_resident(
            &self.adam_moments,
            &state.moments.buffer,
            &grad.buffer,
            &next_moments.buffer,
            Dims {
                m: 1,
                k: state.step,
                n: n as u32,
                _pad: lr.to_bits(),
            },
            (n.div_ceil(256).max(1) as u32, 1, 1),
        );

        let out = self.alloc(param.rows, param.cols);
        self.dispatch_resident(
            &self.adam_update,
            &param.buffer,
            &next_moments.buffer,
            &out.buffer,
            Dims {
                m: 1,
                k: state.step,
                n: n as u32,
                _pad: lr.to_bits(),
            },
            (n.div_ceil(256).max(1) as u32, 1, 1),
        );

        state.moments = next_moments;
        Ok(out)
    }

    fn dims_of(&self, t: &GpuTensor) -> Dims {
        Dims {
            m: t.rows as u32,
            k: t.cols as u32,
            n: t.cols as u32,
            _pad: 0,
        }
    }

    /// Encode and submit one kernel over already-resident buffers. No upload,
    /// no readback.
    fn dispatch_resident(
        &self,
        pipeline: &wgpu::ComputePipeline,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        c: &wgpu::Buffer,
        dims: Dims,
        groups: (u32, u32, u32),
    ) {
        let dims_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let mut pending = self.pending.borrow_mut();

        if pending.encoder.is_none() {
            pending.encoder = Some(self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("aether-gpu batch"),
                },
            ));
        }

        {
            let encoder = pending.encoder.as_mut().expect("encoder just created");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }

        // Parked, not dropped: the recorded commands reference these, and the
        // queue reads them at submission. Released in `flush`.
        pending.bind_groups.push(bind_group);
        pending.dims_buffers.push(dims_buf);
        pending.recorded += 1;
    }

    fn map_and_read(&self, staging: &wgpu::Buffer) -> Result<Vec<f32>, GpuError> {
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });

        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Readback(format!("{e:?}")))?;

        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(GpuError::Readback(format!("{e:?}"))),
            Err(e) => return Err(GpuError::Readback(e.to_string())),
        }

        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    /// The adapter these kernels actually run on.
    pub fn adapter_info(&self) -> &AdapterInfo {
        &self.info
    }

    /// `C[m,n] = A[m,k] * B[k,n]`, row-major, f32.
    pub fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, GpuError> {
        if a.len() != m * k {
            return Err(GpuError::ShapeMismatch(format!(
                "a has {} elements, expected m*k = {}",
                a.len(),
                m * k
            )));
        }
        if b.len() != k * n {
            return Err(GpuError::ShapeMismatch(format!(
                "b has {} elements, expected k*n = {}",
                b.len(),
                k * n
            )));
        }

        // 16x16 workgroup over an (m, n) grid, rounded up so the tail tile is
        // covered. The bounds check in the shader discards the overhang.
        let groups_x = m.div_ceil(16) as u32;
        let groups_y = n.div_ceil(16) as u32;

        self.dispatch(
            &self.matmul,
            a,
            b,
            m * n,
            Dims {
                m: m as u32,
                k: k as u32,
                n: n as u32,
                _pad: 0,
            },
            (groups_x.max(1), groups_y.max(1), 1),
        )
    }

    /// `C[m,n] = A[m,n] + bias[n]`, broadcasting the bias across rows.
    pub fn add_bias(
        &self,
        a: &[f32],
        bias: &[f32],
        m: usize,
        n: usize,
    ) -> Result<Vec<f32>, GpuError> {
        if a.len() != m * n {
            return Err(GpuError::ShapeMismatch(format!(
                "a has {} elements, expected m*n = {}",
                a.len(),
                m * n
            )));
        }
        if bias.len() != n {
            return Err(GpuError::ShapeMismatch(format!(
                "bias has {} elements, expected n = {n}",
                bias.len()
            )));
        }

        let total = m * n;
        self.dispatch(
            &self.add_broadcast_row,
            a,
            bias,
            total,
            Dims {
                m: m as u32,
                k: 0,
                n: n as u32,
                _pad: 0,
            },
            (total.div_ceil(256).max(1) as u32, 1, 1),
        )
    }

    /// Elementwise ReLU.
    pub fn relu(&self, a: &[f32]) -> Result<Vec<f32>, GpuError> {
        let total = a.len();
        self.dispatch(
            &self.relu,
            a,
            &[0.0f32],
            total,
            Dims {
                m: 1,
                k: 0,
                n: total as u32,
                _pad: 0,
            },
            (total.div_ceil(256).max(1) as u32, 1, 1),
        )
    }

    /// `grad * (pre > 0)`, elementwise. Zero at exactly zero.
    pub fn relu_backward(&self, pre: &[f32], grad: &[f32]) -> Result<Vec<f32>, GpuError> {
        if pre.len() != grad.len() {
            return Err(GpuError::ShapeMismatch(format!(
                "pre has {} elements, grad has {}",
                pre.len(),
                grad.len()
            )));
        }

        let total = pre.len();
        self.dispatch(
            &self.relu_backward,
            pre,
            grad,
            total,
            Dims {
                m: 1,
                k: 0,
                n: total as u32,
                _pad: 0,
            },
            (total.div_ceil(256).max(1) as u32, 1, 1),
        )
    }

    /// Upload, dispatch, read back. One submission, one map, one poll.
    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        a: &[f32],
        b: &[f32],
        out_len: usize,
        dims: Dims,
        groups: (u32, u32, u32),
    ) -> Result<Vec<f32>, GpuError> {
        // This path submits on its own. Anything recorded by the resident API
        // has to land first, or the two get reordered relative to each other.
        self.flush();

        let out_bytes = (out_len * std::mem::size_of::<f32>()) as u64;

        let a_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("a"),
                contents: bytemuck::cast_slice(a),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let b_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("b"),
                contents: bytemuck::cast_slice(b),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let c_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("c"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let dims_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aether-gpu bind group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aether-gpu encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aether-gpu pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }

        encoder.copy_buffer_to_buffer(&c_buf, 0, &staging, 0, out_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });

        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Readback(format!("{e:?}")))?;

        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(GpuError::Readback(format!("{e:?}"))),
            Err(e) => return Err(GpuError::Readback(e.to_string())),
        }

        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(out)
    }
}

impl Drop for GpuContext {
    /// Submit anything still recorded, then block until the device is idle.
    ///
    /// Two reasons. Recorded-but-unflushed work would otherwise be dropped
    /// silently, so a caller that updates parameters and never reads them back
    /// would lose the update with no error. And tearing down a device while the
    /// queue still has work in flight is a documented way to fault a driver.
    ///
    /// A single `STATUS_ACCESS_VIOLATION` at process exit was observed once
    /// here, on the first run after a rebuild, and did not reproduce in twelve
    /// subsequent runs of the same binary. The cause was never established, so
    /// this is hardening against a plausible mechanism rather than a fix for a
    /// diagnosed one, and it is recorded that way rather than claimed as a fix.
    fn drop(&mut self) {
        if let Some(encoder) = self.pending.borrow_mut().encoder.take() {
            self.queue.submit(Some(encoder.finish()));
        }
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Row-major `f32` matmul on the CPU, used as the parity reference.
///
/// Deliberately the naive triple loop rather than anything clever: this is the
/// thing the GPU result is checked against, and a reference with its own
/// optimisations is a reference that can be wrong in the same direction as the
/// kernel it validates.
/// Squared Euclidean distance matrix on the CPU, the parity reference for
/// [`GpuContext::pairwise_sqdist_resident`].
pub fn cpu_pairwise_sqdist(points: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f32;
            for c in 0..d {
                let delta = points[i * d + c] - points[j * d + c];
                sum += delta * delta;
            }
            out[i * n + j] = sum;
        }
    }
    out
}

/// Row-major `f32` matmul on the CPU, used as the parity reference.
pub fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}
