//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS WebGL Export
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Exports 3D data to interactive WebGL visualizations.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

use heapless::String;

/// WebGL exporter for 3D visualization
pub struct WebGLExporter {
    /// Point size
    pub point_size: f32,
    /// Enable auto-rotation
    pub auto_rotate: bool,
}

impl WebGLExporter {
    pub fn new() -> Self {
        Self {
            point_size: 0.05,
            auto_rotate: true,
        }
    }

    /// Generate HTML with Three.js visualization
    pub fn export_html(&self, points: &[[f64; 3]], title: &str) -> String<8192> {
        let mut html = String::new();
        
        let _ = html.push_str("<!DOCTYPE html>\n<html><head><title>AEGIS - ");
        let _ = html.push_str(title);
        let _ = html.push_str("</title>\n");
        let _ = html.push_str("<script src=\"https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js\"></script>\n");
        let _ = html.push_str("</head><body style=\"margin:0;background:#1a1a2e\">\n");
        let _ = html.push_str("<script>\nconst points = [");
        
        // Serialize points
        for (i, p) in points.iter().enumerate() {
            if i > 0 { let _ = html.push(','); }
            let _ = html.push('[');
            // Simple float formatting
            let x = p[0] as i32;
            let y = p[1] as i32;
            let z = p[2] as i32;
            let _ = html.push_str(&format_i32(x));
            let _ = html.push(',');
            let _ = html.push_str(&format_i32(y));
            let _ = html.push(',');
            let _ = html.push_str(&format_i32(z));
            let _ = html.push(']');
        }
        
        let _ = html.push_str("];\n");
        let _ = html.push_str("const scene = new THREE.Scene();\n");
        let _ = html.push_str("const camera = new THREE.PerspectiveCamera(75, innerWidth/innerHeight, 0.1, 1000);\n");
        let _ = html.push_str("const renderer = new THREE.WebGLRenderer({antialias:true});\n");
        let _ = html.push_str("renderer.setSize(innerWidth, innerHeight);\n");
        let _ = html.push_str("document.body.appendChild(renderer.domElement);\n");
        let _ = html.push_str("const geo = new THREE.BufferGeometry();\n");
        let _ = html.push_str("const pos = new Float32Array(points.flat());\n");
        let _ = html.push_str("geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));\n");
        let _ = html.push_str("const mat = new THREE.PointsMaterial({color:0x00ff88,size:0.1});\n");
        let _ = html.push_str("const pc = new THREE.Points(geo, mat);\n");
        let _ = html.push_str("scene.add(pc);\n");
        let _ = html.push_str("camera.position.z = 5;\n");
        let _ = html.push_str("function animate(){requestAnimationFrame(animate);pc.rotation.y+=0.005;renderer.render(scene,camera);}\n");
        let _ = html.push_str("animate();\n");
        let _ = html.push_str("</script></body></html>");
        
        html
    }
}

impl Default for WebGLExporter {
    fn default() -> Self {
        Self::new()
    }
}

fn format_i32(n: i32) -> String<16> {
    let mut s = String::new();
    if n < 0 {
        let _ = s.push('-');
    }
    let abs = if n < 0 { -n } else { n } as u32;
    if abs == 0 {
        let _ = s.push('0');
    } else {
        let mut digits = [0u8; 10];
        let mut i = 0;
        let mut val = abs;
        while val > 0 {
            digits[i] = (val % 10) as u8;
            val /= 10;
            i += 1;
        }
        while i > 0 {
            i -= 1;
            let _ = s.push((b'0' + digits[i]) as char);
        }
    }
    s
}
