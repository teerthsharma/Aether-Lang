//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS ASCII Renderer
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Terminal-friendly ASCII art visualization for 3D data.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

use heapless::String;

/// ASCII density characters (from sparse to dense)
pub const DENSITY_CHARS: &[char] = &[' ', '.', ':', '+', '*', '#', '@'];

/// ANSI color codes for terminal output
pub mod colors {
    pub const RESET: &str = "\x1b[0m";
    pub const RED: &str = "\x1b[91m";
    pub const GREEN: &str = "\x1b[92m";
    pub const YELLOW: &str = "\x1b[93m";
    pub const BLUE: &str = "\x1b[94m";
    pub const MAGENTA: &str = "\x1b[95m";
    pub const CYAN: &str = "\x1b[96m";
}

/// ASCII Renderer for 3D data visualization
pub struct AsciiRenderer {
    /// Canvas width in characters
    pub width: usize,
    /// Canvas height in characters  
    pub height: usize,
    /// Enable ANSI colors
    pub use_color: bool,
}

impl AsciiRenderer {
    /// Create a new renderer with default 60x20 terminal size
    pub fn new() -> Self {
        Self {
            width: 60,
            height: 20,
            use_color: true,
        }
    }

    /// Render points as ASCII scatter plot
    /// Points are [x, y, z] arrays
    pub fn render_scatter(&self, points: &[[f64; 3]]) -> String<4096> {
        let mut canvas = [[' '; 60]; 20];
        
        if points.is_empty() {
            return self.canvas_to_string(&canvas);
        }
        
        // Find bounds
        let (min_x, max_x, min_y, max_y) = self.find_bounds(points);
        let range_x = (max_x - min_x).max(0.001);
        let range_y = (max_y - min_y).max(0.001);
        
        // Project and render points
        for point in points {
            let x = (((point[0] - min_x) / range_x) * (self.width - 1) as f64) as usize;
            let y = (((point[1] - min_y) / range_y) * (self.height - 1) as f64) as usize;
            
            if x < self.width && y < self.height {
                canvas[self.height - 1 - y][x] = '*';
            }
        }
        
        self.canvas_to_string(&canvas)
    }

    /// Render density heatmap
    pub fn render_density(&self, points: &[[f64; 3]]) -> String<4096> {
        let mut density = [[0u8; 60]; 20];
        
        if points.is_empty() {
            return self.canvas_to_string(&[[' '; 60]; 20]);
        }
        
        let (min_x, max_x, min_y, max_y) = self.find_bounds(points);
        let range_x = (max_x - min_x).max(0.001);
        let range_y = (max_y - min_y).max(0.001);
        
        // Accumulate density
        for point in points {
            let x = (((point[0] - min_x) / range_x) * (self.width - 1) as f64) as usize;
            let y = (((point[1] - min_y) / range_y) * (self.height - 1) as f64) as usize;
            
            if x < self.width && y < self.height {
                density[self.height - 1 - y][x] = density[self.height - 1 - y][x].saturating_add(1);
            }
        }
        
        // Find max density
        let max_d = density.iter().flatten().copied().max().unwrap_or(1);
        
        // Render to characters
        let mut canvas = [[' '; 60]; 20];
        for (y, row) in density.iter().enumerate() {
            for (x, &d) in row.iter().enumerate() {
                if d > 0 {
                    let idx = ((d as usize * (DENSITY_CHARS.len() - 1)) / max_d as usize)
                        .min(DENSITY_CHARS.len() - 1);
                    canvas[y][x] = DENSITY_CHARS[idx];
                }
            }
        }
        
        self.canvas_to_string(&canvas)
    }

    fn find_bounds(&self, points: &[[f64; 3]]) -> (f64, f64, f64, f64) {
        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;
        
        for p in points {
            if p[0] < min_x { min_x = p[0]; }
            if p[0] > max_x { max_x = p[0]; }
            if p[1] < min_y { min_y = p[1]; }
            if p[1] > max_y { max_y = p[1]; }
        }
        
        (min_x, max_x, min_y, max_y)
    }

    fn canvas_to_string(&self, canvas: &[[char; 60]; 20]) -> String<4096> {
        let mut output = String::new();
        
        // Top border
        let _ = output.push_str("+");
        for _ in 0..self.width { let _ = output.push('-'); }
        let _ = output.push_str("+\n");
        
        // Content
        for y in 0..self.height {
            let _ = output.push('|');
            for x in 0..self.width {
                let _ = output.push(canvas[y][x]);
            }
            let _ = output.push_str("|\n");
        }
        
        // Bottom border  
        let _ = output.push_str("+");
        for _ in 0..self.width { let _ = output.push('-'); }
        let _ = output.push_str("+\n");
        
        output
    }
}

impl Default for AsciiRenderer {
    fn default() -> Self {
        Self::new()
    }
}
