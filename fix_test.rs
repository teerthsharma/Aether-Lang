const CLUSTER_THRESHOLD: i16 = 15;
const DENSITY_MIN: f64 = 0.1;
const DENSITY_MAX: f64 = 0.6;
const MAX_BETTI_1: u32 = 10;

fn compute_raw_betti_0(data: &[u8]) -> u32 {
    let mut components = 0u32;
    let mut in_component = false;
    for window in data.windows(2) {
        let dist = (window[0] as i16 - window[1] as i16).abs();
        if dist > CLUSTER_THRESHOLD {
            if !in_component {
                components += 1;
                in_component = true;
            }
        } else {
            in_component = false;
        }
    }
    components
}

fn check(raw_b0: u32, b1: u32, len: usize) -> bool {
    let b0 = if raw_b0 == 0 && len > 0 { 1 } else { raw_b0 };
    let density = if len > 0 { b0 as f64 / len as f64 } else { 0.0 };
    if density < DENSITY_MIN || density > DENSITY_MAX { return false; }
    if b1 > MAX_BETTI_1 { return false; }
    true
}

fn main() {
    let nop_sled = vec![0x90; 64];
    let raw_b0 = compute_raw_betti_0(&nop_sled);
    let ok = check(raw_b0, 0, 64);
    println!("raw: {}, density: {}, ok: {}", raw_b0, 1.0 / 64.0, ok);
    println!("density check: {} < 0.1 ? {}", 1.0/64.0, 1.0/64.0 < 0.1);
}
