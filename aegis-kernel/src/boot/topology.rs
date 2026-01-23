use alloc::vec::Vec;

/// Represents the I/O capabilities of the organism (hardware).
#[derive(Debug, Clone, Copy, Default)]
pub struct IoCaps {
    pub has_serial: bool,
    pub has_framebuffer: bool,
    pub has_keyboard: bool,
    pub has_network: bool,
}

/// The graph of the hardware organism.
/// Nodes are Cores/Memory, edges are latency/bandwidth.
#[derive(Debug, Clone)]
pub struct HardwareTopology {
    /// Number of neural clusters (CPU Cores)
    pub cpu_cores: usize,
    /// Number of memory localities (NUMA Nodes)
    pub numa_nodes: usize,
    /// Total synaptic space (System Memory) in bytes
    pub total_memory: u64,
    /// Sensory/Motor capabilities (I/O)
    pub io_capabilities: IoCaps,
    /// L3 Cache sharing groups (for manifold parallelism)
    pub cache_groups: Vec<usize>,
}

impl HardwareTopology {
    pub fn new() -> Self {
        Self {
            cpu_cores: 1,
            numa_nodes: 1,
            total_memory: 0,
            io_capabilities: IoCaps::default(),
            cache_groups: Vec::new(),
        }
    }

    /// Perform a "Bio-Scan" to populate the topology from boot info.
    /// This is a placeholder for the actual discovery logic.
    pub fn bio_scan() -> Self {
        // TODO: Ingest actual hardware state.
        // For now, return a safe default (Single-Celled Organism).
        Self::new()
    }
}
