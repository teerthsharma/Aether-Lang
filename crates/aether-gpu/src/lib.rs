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

//! Every public item carries a doc comment, enforced rather than intended.
//!
//! Turned on after a survey found a doc asserting the opposite of the benchmark
//! that refuted it: `pairwise_sqdist_resident` called itself the operation where
//! a GPU matters most, having been measured as not worth offloading at any size.
//! `missing_docs` cannot catch a wrong doc, but it does catch the absent one,
//! and an item nobody described is where a wrong description starts.
#![deny(missing_docs)]

pub mod datasets;

use std::borrow::Cow;
use std::cell::RefCell;

use aether_core::scheduled::{AttentionGradients, BlockSchedule};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Mirrors `MAX_HEAD_DIM` in `shaders.wgsl`.
///
/// WGSL cannot size a private array from a uniform, so the kernel's scratch is
/// fixed at compile time and the host is the only place that can reject an
/// oversized launch. A test asserts the two constants agree, because a silent
/// divergence here produces wrong numbers rather than an error.
const MAX_HEAD_DIM: usize = 128;

/// Mirrors `MAX_BLOCK` in `shaders.wgsl`. See [`MAX_HEAD_DIM`].
const MAX_BLOCK: usize = 128;

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
    /// A launch this backend cannot serve, though nothing about it is malformed.
    ///
    /// Distinct from [`GpuError::ShapeMismatch`] because the two ask different
    /// things of a caller. Mismatched shapes are a bug at the call site and the
    /// fix is to pass consistent ones. A `head_dim` past the kernel's private
    /// scratch is a limit of this backend: the arguments are coherent, the CPU
    /// path in `aether-core` computes them, and the right response is to use it.
    ///
    /// Collapsing both into one variant leaves a caller that wants to fall back
    /// no way to tell "you called this wrong" from "this size needs the other
    /// implementation", so it either falls back on real bugs or on neither.
    Unsupported(String),
    /// The readback buffer could not be mapped.
    Readback(String),
}

impl core::fmt::Display for GpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GpuError::NoAdapter => write!(f, "no wgpu adapter available"),
            GpuError::NoDevice(e) => write!(f, "could not create wgpu device: {e}"),
            GpuError::ShapeMismatch(e) => write!(f, "shape mismatch: {e}"),
            GpuError::Unsupported(e) => write!(f, "unsupported on this backend: {e}"),
            GpuError::Readback(e) => write!(f, "buffer readback failed: {e}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// Which implementation produced a result from
/// [`GpuContext::scheduled_attention_or_cpu`] or
/// [`GpuContext::scheduled_attention_backward_or_cpu`].
///
/// Returned rather than inferred because the two differ in precision — f32
/// widened to f64 on the GPU, f64 throughout on the CPU — so which one ran is
/// part of what the number means and not an implementation detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionPath {
    /// The WGSL kernels. f32 internally, so agreement with the CPU path is at an
    /// f32 tolerance and never at the 1e-12 used elsewhere in this workspace.
    Gpu,
    /// `aether_core::scheduled::scheduled_attention`, in f64. Taken when the
    /// launch exceeds a kernel ceiling, which is a limit of the backend rather
    /// than anything wrong with the call.
    Cpu,
}

/// Which physical device the kernels are running on.
///
/// Recorded and reported rather than assumed: a benchmark that does not name
/// its adapter is a benchmark that cannot be reproduced, and `wgpu` will
/// silently hand back a software rasterizer if that is all it can find.
#[derive(Clone, Debug)]
pub struct AdapterInfo {
    /// The adapter's reported name, e.g. `NVIDIA GeForce RTX 4060 Laptop GPU`.
    pub name: String,
    /// Which graphics API the device was reached through: `Vulkan`, `Dx12`,
    /// `Metal` or `Gl`. Recorded because kernels can behave differently across
    /// them, and this crate pins one at construction after a teardown crash that
    /// only appeared when several were instantiated together.
    pub backend: String,
    /// `DiscreteGpu`, `IntegratedGpu`, `Cpu`, `VirtualGpu` or `Other`. The `Cpu`
    /// case is a software rasterizer, which is why [`AdapterInfo::is_hardware`]
    /// exists rather than callers trusting that an adapter was found.
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
    scheduled_attention: wgpu::ComputePipeline,
    attention_row_stats: wgpu::ComputePipeline,
    attention_dq: wgpu::ComputePipeline,
    attention_dk: wgpu::ComputePipeline,
    attention_dv: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    pending: RefCell<Pending>,
}

/// Adam's per-parameter state: both moment estimates in one tensor.
///
/// Packed rather than held as two tensors because the bind group carries three
/// storage buffers, and Adam touches parameters, gradients and both moments.
/// First moment occupies `[0, n)`, second `[n, 2n)`.
///
/// # This state cannot leave the process
///
/// `moments` is private and nothing reads it out, so a caller can run Adam and
/// cannot save what Adam learned. Training that stops and resumes rebuilds the
/// state with [`GpuContext::adam_state`], which zeroes both moments and resets
/// the step counter — so the resumed run spends its first steps with bias
/// correction dividing by `1 - beta^t` at `t = 1` again, adapting from nothing
/// while the parameters carry on from where they were.
///
/// That is a silent difference rather than an error: the loss keeps falling and
/// the run looks continuous. It is recorded here because the alternative is
/// finding it in a training curve.
///
/// The size of it is measured rather than asserted.
/// `rebuilding_adam_state_mid_run_changes_the_parameters` runs six steps twice on
/// identical parameters and gradients, rebuilding the state after the third in
/// one of them: every parameter differs, the worst by 6.8e-02. At a learning rate
/// of 0.02 an Adam step moves a parameter by about that much, so six steps move
/// it by around 0.12 — the discontinuity is over half the run's total movement,
/// from a change that produces no error and no warning.
///
/// Nothing in this workspace checkpoints, so nothing is broken by it today. The
/// fix is an accessor returning the packed tensor and a constructor taking one
/// back, which is a public API decision rather than an oversight, and is not made
/// here.
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
    /// Rows in the logical shape. Some buffers are uploaded flat, where this
    /// carries no meaning beyond `rows * cols == len()`.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Columns in the logical shape. See [`GpuTensor::rows`].
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Element count, which is what the kernels index by — every shader here
    /// computes its own offsets from the uniform rather than from this shape.
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    /// Whether the tensor holds no elements. Present because clippy asks for it
    /// alongside [`GpuTensor::len`], and a zero-length dispatch is rejected
    /// before it reaches a kernel.
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
        // Exactly one backend is instantiated, tried in preference order.
        //
        // The obvious construction is `Backends::default()`, which enables every
        // backend at once and lets wgpu pick. That reproduces an intermittent
        // STATUS_ACCESS_VIOLATION at process exit: measured 8 crashes in 60 runs
        // with all backends enabled, against 0 in 180 with any single backend
        // pinned. Pinning is a complete workaround, so the default does it
        // rather than leaving every caller to discover the fault. FEATURES.md
        // carries the bisect.
        //
        // `WGPU_BACKEND` still overrides, and is honoured explicitly because
        // `InstanceDescriptor::default()` does not read it -- setting the
        // variable and assuming it took effect measures whatever backend would
        // have been chosen anyway.
        let candidates: Vec<wgpu::Backends> = match wgpu::Backends::from_env() {
            Some(requested) => vec![requested],
            None => vec![
                wgpu::Backends::VULKAN,
                wgpu::Backends::DX12,
                wgpu::Backends::METAL,
                wgpu::Backends::GL,
            ],
        };

        let mut found = None;
        for backends in candidates {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends,
                ..Default::default()
            });

            if let Ok(adapter) = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
            {
                // The instance is dropped here; the adapter keeps what it needs
                // alive, and holding it would defeat the point of instantiating
                // one backend at a time.
                found = Some(adapter);
                break;
            }
        }

        let adapter = found.ok_or(GpuError::NoAdapter)?;

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
            scheduled_attention: build("scheduled_attention"),
            attention_row_stats: build("attention_row_stats"),
            attention_dq: build("attention_dq"),
            attention_dk: build("attention_dk"),
            attention_dv: build("attention_dv"),
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
    /// This is the O(n²d) term underneath every Vietoris–Rips filtration in
    /// `aether-core`, which is what made it look like the operation where a GPU
    /// would matter to this project specifically.
    ///
    /// # It was measured, and it is not worth offloading at any size
    ///
    /// 90–100% of the call is transfer. The arithmetic is O(n²d) against O(n²)
    /// bytes returned, so the work per byte is `d` — a constant — and no `n`
    /// makes the compute outgrow the copy. `matmul` escapes this because its
    /// O(n³) against O(n²) gives a ratio that grows with `n`, which is why that
    /// one has a crossover at n=128 and this one has none.
    ///
    /// The filtration cannot absorb the difference either: the persistence
    /// reduction that consumes the matrix is CPU-side, so the result has to come
    /// back regardless of where it was computed.
    ///
    /// Kept because the kernel is correct, tested, and is the thing that
    /// established the rule above by failing to pay. The doc comment previously
    /// asserted the opposite — written before the measurement and left standing
    /// after it, which is the failure this crate spends most of `FEATURES.md`
    /// documenting elsewhere.
    ///
    /// Reproduce with `cargo run -p aether-gpu --example gpu_bench --release`.
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
    /// Topology-scheduled attention: the GPU port of
    /// [`aether_core::scheduled::scheduled_attention`].
    ///
    /// Takes a [`BlockSchedule`] rather than raw CSR arrays so that every
    /// schedule invariant — causality, sortedness, no empty row, indices in
    /// range — is already enforced by the constructor that built it. Re-checking
    /// them here would duplicate `aether-core`'s validation and, worse, allow a
    /// caller to bypass it by assembling the arrays directly.
    ///
    /// The two ceilings below are the kernel's, not the algorithm's: WGSL cannot
    /// size a private array dynamically, so the scratch space for scores and the
    /// accumulator is fixed at compile time. Exceeding either would index out of
    /// bounds inside the shader, which WGSL clamps rather than traps — the result
    /// would be silently wrong numbers, so this is checked before dispatch.
    ///
    /// Computes in f32 against `aether-core`'s f64. Parity is asserted in
    /// `tests/attention_parity.rs` against the CPU kernel and against the
    /// quadratic reference, at an f32 tolerance.
    pub fn scheduled_attention_resident(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq: usize,
        head_dim: usize,
        schedule: &BlockSchedule,
        block_size: usize,
    ) -> Result<GpuTensor, GpuError> {
        if head_dim == 0 || block_size == 0 || seq == 0 {
            return Err(GpuError::ShapeMismatch(format!(
                "seq, head_dim and block_size must all be non-zero, got \
                 seq={seq}, head_dim={head_dim}, block_size={block_size}"
            )));
        }
        // `Unsupported` rather than `ShapeMismatch`: nothing about these launches
        // is malformed, and `aether_core::scheduled` computes them. A caller that
        // wants to fall back needs to tell "this size needs the CPU path" from
        // "you passed inconsistent shapes", and one variant for both leaves it
        // falling back on real bugs or on neither.
        if head_dim > MAX_HEAD_DIM {
            return Err(GpuError::Unsupported(format!(
                "head_dim {head_dim} exceeds the kernel's private-array ceiling \
                 of {MAX_HEAD_DIM}; the aether-core CPU path has no such limit"
            )));
        }
        if block_size > MAX_BLOCK {
            return Err(GpuError::Unsupported(format!(
                "block_size {block_size} exceeds the kernel's private-array \
                 ceiling of {MAX_BLOCK}; the aether-core CPU path has no such limit"
            )));
        }
        if !seq.is_multiple_of(block_size) {
            return Err(GpuError::ShapeMismatch(format!(
                "seq {seq} is not a multiple of block_size {block_size}"
            )));
        }

        let expected = seq * head_dim;
        for (name, operand) in [("q", q), ("k", k), ("v", v)] {
            if operand.len() != expected {
                return Err(GpuError::ShapeMismatch(format!(
                    "{name} has {} elements, expected seq*head_dim = {expected}",
                    operand.len()
                )));
            }
        }

        let num_blocks = seq / block_size;
        if schedule.num_blocks() != num_blocks {
            return Err(GpuError::ShapeMismatch(format!(
                "schedule covers {} query blocks, but seq/block_size = {num_blocks}",
                schedule.num_blocks()
            )));
        }

        // One layout serves every kernel in the crate, and attention needs six
        // arrays against its four bindings. Concatenating is what makes that fit
        // without a second layout existing for a single caller.
        let mut operands = Vec::with_capacity(expected * 3);
        operands.extend_from_slice(q);
        operands.extend_from_slice(k);
        operands.extend_from_slice(v);

        // Block indices are bounded by num_blocks and every integer below 2^24
        // is exact in f32, so this is lossless well past any schedule that fits
        // in memory.
        let csr: Vec<f32> = schedule
            .offsets
            .iter()
            .chain(schedule.indices.iter())
            .map(|&i| i as f32)
            .collect();
        let csr_len = csr.len();

        // Uploaded rather than passed through `dispatch`, so the result can stay
        // on the device. `dispatch` flushes and reads back, which is the wrong
        // shape for an operation whose output usually feeds another kernel.
        let operands = self.upload(&operands, 3 * seq, head_dim)?;
        let csr = self.upload(&csr, 1, csr_len)?;
        let out = self.alloc(seq, head_dim);

        self.dispatch_resident(
            &self.scheduled_attention,
            &operands.buffer,
            &csr.buffer,
            &out.buffer,
            Dims {
                m: seq as u32,
                k: head_dim as u32,
                n: block_size as u32,
                _pad: num_blocks as u32,
            },
            (seq.div_ceil(64) as u32, 1, 1),
        );

        Ok(out)
    }

    /// Scheduled attention on the GPU where possible, on the CPU where not.
    ///
    /// This is what [`GpuError::Unsupported`] exists for. Splitting that variant
    /// out of `ShapeMismatch` was pointless while nothing distinguished them at a
    /// call site, and this is the call site: a launch past the kernel's ceilings
    /// is coherent and `aether-core` computes it, so the only correct response is
    /// to compute it there. A malformed launch is still an error and is still
    /// returned, because falling back on a caller's bug would hide it.
    ///
    /// # Precision is not uniform, and that is why the path is returned
    ///
    /// The two routes do not agree to the same tolerance: WGSL has no f64, so the
    /// GPU answer is f32 widened on the way out while the CPU answer is f64
    /// throughout. A helper that hid which one ran would silently change the
    /// precision of a result with the size of its input — the sort of thing that
    /// shows up much later as an unreproducible number.
    ///
    /// Returning [`AttentionPath`] alongside the values makes that switch part of
    /// the signature. A caller that does not care can ignore it; one comparing
    /// runs cannot ignore it by accident.
    pub fn scheduled_attention_or_cpu(
        &self,
        q: &[f64],
        k: &[f64],
        v: &[f64],
        seq: usize,
        head_dim: usize,
        schedule: &BlockSchedule,
        block_size: usize,
    ) -> Result<(Vec<f64>, AttentionPath), GpuError> {
        let narrow = |xs: &[f64]| xs.iter().map(|&x| x as f32).collect::<Vec<f32>>();

        match self.scheduled_attention(
            &narrow(q),
            &narrow(k),
            &narrow(v),
            seq,
            head_dim,
            schedule,
            block_size,
        ) {
            Ok(out) => Ok((
                out.into_iter().map(|x| x as f64).collect(),
                AttentionPath::Gpu,
            )),
            Err(GpuError::Unsupported(_)) => {
                // `aether-core` validates the same launch again and its errors
                // are a different type, so a genuinely malformed call that got
                // this far surfaces as a shape mismatch rather than a panic.
                let out = aether_core::scheduled::scheduled_attention(
                    q, k, v, seq, head_dim, schedule, block_size,
                )
                .map_err(|e| GpuError::ShapeMismatch(format!("{e:?}")))?;
                Ok((out, AttentionPath::Cpu))
            }
            Err(other) => Err(other),
        }
    }

    /// Reverse mode on the GPU where possible, on the CPU where not.
    ///
    /// The counterpart to [`GpuContext::scheduled_attention_or_cpu`], and it
    /// exists because that one did. A caller routing its forward pass through a
    /// helper that handles the ceilings, then meeting a bare `Unsupported` from
    /// the backward at the same size, has to implement the fallback anyway —
    /// having one of the pair was an asymmetry rather than a decision.
    ///
    /// Returns host-side f64 gradients rather than the resident tensors of
    /// [`GpuContext::scheduled_attention_backward_resident`], because the CPU
    /// route cannot produce device buffers and a helper whose return type
    /// depended on which path it took would push the branch straight back to the
    /// caller.
    ///
    /// The same precision caveat applies and for the same reason: the GPU route
    /// is f32 widened on the way out, the CPU route f64 throughout, and
    /// [`AttentionPath`] says which ran.
    #[allow(clippy::too_many_arguments)]
    pub fn scheduled_attention_backward_or_cpu(
        &self,
        q: &[f64],
        k: &[f64],
        v: &[f64],
        seq: usize,
        head_dim: usize,
        schedule: &BlockSchedule,
        block_size: usize,
        d_out: &[f64],
    ) -> Result<(AttentionGradients, AttentionPath), GpuError> {
        let narrow = |xs: &[f64]| xs.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let widen = |xs: Vec<f32>| xs.into_iter().map(|x| x as f64).collect::<Vec<f64>>();

        match self.scheduled_attention_backward_resident(
            &narrow(q),
            &narrow(k),
            &narrow(v),
            seq,
            head_dim,
            schedule,
            block_size,
            &narrow(d_out),
        ) {
            Ok((dq, dk, dv)) => Ok((
                AttentionGradients {
                    dq: widen(self.read(&dq)?),
                    dk: widen(self.read(&dk)?),
                    dv: widen(self.read(&dv)?),
                },
                AttentionPath::Gpu,
            )),
            Err(GpuError::Unsupported(_)) => {
                let grads = aether_core::scheduled::scheduled_attention_backward(
                    q, k, v, seq, head_dim, schedule, block_size, d_out,
                )
                .map_err(|e| GpuError::ShapeMismatch(format!("{e:?}")))?;
                Ok((grads, AttentionPath::Cpu))
            }
            Err(other) => Err(other),
        }
    }

    /// Reverse mode through [`GpuContext::scheduled_attention`].
    ///
    /// The GPU port of [`aether_core::scheduled::scheduled_attention_backward`],
    /// which is the f64 reference these are checked against. The forward kernel
    /// was built the same way round, and the ordering matters: a backward pass
    /// verified only against itself is verified against nothing, because the
    /// failure mode is a gradient that is smooth, finite, and wrong.
    ///
    /// Four dispatches. `attention_row_stats` computes the per-row maximum,
    /// log-sum-exp and delta term once; the three gradient kernels then read them
    /// instead of rebuilding a row's softmax to touch one of its columns, which
    /// is what keeps the cost O(head_dim) per (row, column) pair rather than
    /// quadratic in the sequence.
    ///
    /// The schedule is held constant, as in the reference. A block is selected or
    /// it is not, so there is no derivative to take through the selection, and
    /// the gradient teaches a model to use the blocks it was given rather than to
    /// choose different ones.
    ///
    /// Returns `(dq, dk, dv)`, each `[seq, head_dim]` and resident.
    #[allow(clippy::too_many_arguments)]
    pub fn scheduled_attention_backward_resident(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq: usize,
        head_dim: usize,
        schedule: &BlockSchedule,
        block_size: usize,
        d_out: &[f32],
    ) -> Result<(GpuTensor, GpuTensor, GpuTensor), GpuError> {
        if d_out.len() != seq * head_dim {
            return Err(GpuError::ShapeMismatch(format!(
                "d_out has {} elements, expected seq*head_dim = {}",
                d_out.len(),
                seq * head_dim
            )));
        }
        // The forward call validates every other precondition and rejects launches
        // beyond the kernel ceilings. Running it here rather than repeating those
        // checks means the two paths cannot disagree about what is legal.
        let _ = self.scheduled_attention_resident(q, k, v, seq, head_dim, schedule, block_size)?;

        let num_blocks = seq / block_size;
        let span = seq * head_dim;

        let csr: Vec<f32> = schedule
            .offsets
            .iter()
            .chain(schedule.indices.iter())
            .map(|&i| i as f32)
            .collect();
        let csr_len = csr.len();
        let csr = self.upload(&csr, 1, csr_len)?;

        let dims = Dims {
            m: seq as u32,
            k: head_dim as u32,
            n: block_size as u32,
            _pad: num_blocks as u32,
        };
        let groups = (seq.div_ceil(64) as u32, 1, 1);

        // One buffer holds q|k|v|dOut plus a reserved tail of three floats per
        // row for the statistics. Uploaded flat: the layout is not a rectangle in
        // `head_dim`, and every kernel indexes it from base offsets it computes
        // itself, so the tensor's rows and columns mean nothing beyond length.
        let mut packed = Vec::with_capacity(span * 4 + seq * 3);
        packed.extend_from_slice(q);
        packed.extend_from_slice(k);
        packed.extend_from_slice(v);
        packed.extend_from_slice(d_out);
        packed.resize(span * 4 + seq * 3, 0.0);
        let operands = self.upload(&packed, 1, packed.len())?;

        // The statistics kernel takes that buffer as its *output* and reads the
        // operands back out of it, because a `read_write` binding is readable. It
        // fills the reserved tail in place, and the three gradient kernels below
        // bind the same buffer as `a` with everything already there.
        //
        // This replaces a full round trip. The previous version wrote the
        // statistics to their own tensor, downloaded them, and re-uploaded them
        // concatenated onto the operands, once per backward call, on the reasoning
        // that the shared four-binding layout had no free binding to leave them
        // in. That was true and the conclusion did not follow: the output binding
        // was always readable, so no second layout was ever required.
        //
        // `a` and `b` both receive the CSR buffer. This kernel reads the schedule
        // through `b` and never touches `a`, and two read-only bindings onto one
        // buffer cannot race.
        self.dispatch_resident(
            &self.attention_row_stats,
            &csr.buffer,
            &csr.buffer,
            &operands.buffer,
            dims,
            groups,
        );

        let dq = self.alloc(seq, head_dim);
        let dk = self.alloc(seq, head_dim);
        let dv = self.alloc(seq, head_dim);

        for (pipeline, out) in [
            (&self.attention_dq, &dq),
            (&self.attention_dk, &dk),
            (&self.attention_dv, &dv),
        ] {
            self.dispatch_resident(
                pipeline,
                &operands.buffer,
                &csr.buffer,
                &out.buffer,
                dims,
                groups,
            );
        }

        Ok((dq, dk, dv))
    }

    /// [`GpuContext::scheduled_attention_resident`] with the result read back.
    ///
    /// Kept because most callers want the values, and because every parity test
    /// in the crate compares against a host-side reference. It is the resident
    /// path plus a download, not a separate implementation, so the two cannot
    /// drift — a test asserts they agree bitwise, which is a weaker claim than it
    /// looks precisely because of that.
    pub fn scheduled_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq: usize,
        head_dim: usize,
        schedule: &BlockSchedule,
        block_size: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let out =
            self.scheduled_attention_resident(q, k, v, seq, head_dim, schedule, block_size)?;
        self.read(&out)
    }

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
/// `aether_core::ml::Tensor` matmul, computed on the GPU.
///
/// # Why this is a free function and not a change to `Tensor`
///
/// The measured case for a GPU path is strong — crossover at n=128, 38× at
/// n=512 with the f64↔f32 conversion counted — and FEATURES.md recorded it as
/// "blocked on a semantic decision: may `ml::Tensor` drop to f32?".
///
/// That framing was wrong. `aether-gpu` already depends on `aether-core`, so
/// nothing about `Tensor` has to change: a caller who wants the GPU calls this,
/// and a caller who does not keeps `Tensor::matmul` exactly as it is. The
/// precision question does not need answering globally, because it is now asked
/// per call site by whoever knows what the result feeds.
///
/// Making `Tensor::matmul` dispatch internally is the version that would have
/// needed the decision, and would have imposed f32 on the persistence engine's
/// consumers along with everyone else's.
///
/// # Precision
///
/// The product is computed in f32 and returned as f64. Relative error is about
/// 5e-7 at n=256, growing as the square root of the reduction depth. Fine for
/// training, clustering and anything that thresholds; not fine for a caller
/// asserting at 1e-9, which includes parts of this workspace.
///
/// # When it pays
///
/// Above roughly n=128. Below that the transfers dominate and `Tensor::matmul`
/// is faster. This function does not check the size and fall back, because a
/// silent fallback makes a benchmark unreproducible: the caller asked for the
/// GPU and should get it, or an error.
pub fn tensor_matmul(
    ctx: &GpuContext,
    a: &aether_core::ml::tensor::Tensor,
    b: &aether_core::ml::tensor::Tensor,
) -> Result<aether_core::ml::tensor::Tensor, GpuError> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(GpuError::ShapeMismatch(format!(
            "matmul needs two 2-D tensors, got {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    if a.shape[1] != b.shape[0] {
        return Err(GpuError::ShapeMismatch(format!(
            "inner dimensions disagree: {:?} by {:?}",
            a.shape, b.shape
        )));
    }

    let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);

    // Gather through the strides rather than reading the buffer flat.
    //
    // `Tensor::matmul` indexes `data[i * strides[0] + l * strides[1]]`, so it
    // honours a non-contiguous layout. Reading the backing vector in order
    // instead would make this bridge strictly weaker than the function it
    // mirrors, and wrong rather than slow: a caller who swapped one for the
    // other would get a silently different answer.
    //
    // No operation in `aether-core` currently produces a non-contiguous tensor
    // — `transpose` copies into a fresh one — so this costs nothing today. It
    // exists because the failure it prevents is silent, and the first view or
    // slice operation added to `Tensor` would introduce it without touching
    // this file.
    let gather = |t: &aether_core::ml::tensor::Tensor, rows: usize, cols: usize| -> Vec<f32> {
        let d = t.data.borrow();
        let mut out = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                out.push(d[i * t.strides[0] + j * t.strides[1]] as f32);
            }
        }
        out
    };

    let a32 = gather(a, m, k);
    let b32 = gather(b, k, n);

    let out = ctx.matmul(&a32, &b32, m, k, n)?;
    let out64: Vec<f64> = out.iter().map(|v| *v as f64).collect();

    Ok(aether_core::ml::tensor::Tensor::new(&out64, &[m, n]))
}

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
