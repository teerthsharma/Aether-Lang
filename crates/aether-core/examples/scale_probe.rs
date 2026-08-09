//! Measures the real ceiling of the persistence engine. Not a test: a probe that
//! prints the numbers the docs are allowed to quote.
use aether_core::manifold::ManifoldPoint;
use aether_core::persistence::{persistent_homology, ComplexKind, PersistenceConfig};
use std::time::Instant;

fn circle(n: usize) -> Vec<ManifoldPoint<2>> {
    (0..n)
        .map(|i| {
            let t = std::f64::consts::TAU * i as f64 / n as f64;
            ManifoldPoint::new([t.cos(), t.sin()])
        })
        .collect()
}

fn main() {
    println!("{:>6} {:>5} {:>12} {:>10}", "n", "dim", "pairs", "seconds");
    for (n, dim) in [
        (200usize, 0usize),
        (1000, 0),
        (4000, 0),
        (60, 1),
        (120, 1),
        (200, 1),
        (300, 1),
        (30, 2),
        (50, 2),
        (70, 2),
    ] {
        let points = circle(n);
        let config = PersistenceConfig {
            max_homology_dim: dim,
            max_points: 8192,
            max_simplices: 20_000_000,
            max_radius: f64::INFINITY,
            complex_kind: ComplexKind::VietorisRips,
        };
        let start = Instant::now();
        match persistent_homology(&points, config) {
            Ok(d) => println!(
                "{n:>6} {dim:>5} {:>12} {:>10.3}",
                d.pairs.len(),
                start.elapsed().as_secs_f64()
            ),
            Err(e) => println!("{n:>6} {dim:>5} {:>12} {:>10}", format!("{e:?}"), "-"),
        }
    }
}
