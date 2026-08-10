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

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

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
    add_broadcast_row: wgpu::ComputePipeline,
    relu: wgpu::ComputePipeline,
    relu_backward: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
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
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

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
            add_broadcast_row: build("add_broadcast_row"),
            relu: build("relu"),
            relu_backward: build("relu_backward"),
            device,
            queue,
            info,
            layout,
        })
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
