//! Does the routed selector become sparse when the keys actually have structure?
//!
//! On uniform random keys, H0 single-linkage chains: 64 keys give components of
//! size [61, 1, 1, 1]. That is not a clustering failure — it is H0 correctly
//! reporting that a uniform cloud has no density gaps. Routing to the top cluster
//! then means routing to everything, so cost equals dense.
//!
//! The real question is whether the method works when the structure exists.
use aether_core::attention::{attention_mass_recovered, single_linkage_clusters, Selector};

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn signed(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }
}

/// `groups` tight direction-clusters on the sphere, with per-key norm spread.
fn clustered_keys(
    seq: usize,
    head_dim: usize,
    groups: usize,
    tightness: f64,
    spread: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    let centers: Vec<Vec<f64>> = (0..groups)
        .map(|_| (0..head_dim).map(|_| rng.signed()).collect())
        .collect();
    let mut k = vec![0.0; seq * head_dim];
    for t in 0..seq {
        let c = &centers[t % groups];
        let gain = 1.0 + spread * rng.unit();
        for d in 0..head_dim {
            k[t * head_dim + d] = (c[d] + tightness * rng.signed()) * gain;
        }
    }
    k
}

fn main() {
    let (seq, head_dim, budget) = (64usize, 8usize, 8usize);
    println!(
        "{:>7} {:>10} {:>8} {:>12} {:>10} {:>11} {:>10}",
        "groups", "tightness", "spread", "sizes(top4)", "dots/row", "vs dense", "placement"
    );

    for (groups, tightness) in [(4usize, 0.05f64), (4, 0.2), (8, 0.05), (8, 0.2), (16, 0.05)] {
        for spread in [0.0f64, 8.0] {
            let clusters = groups;
            let (mut dots, mut place) = (0.0, 0.0);
            let mut shown = String::new();
            let trials = 6;
            for trial in 0..trials {
                let mut rng = Rng((trial * 2 + 67) | 1);
                let n = seq * head_dim;
                let q: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
                let v: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
                let k = clustered_keys(seq, head_dim, groups, tightness, spread, &mut rng);

                let (assignment, _) = single_linkage_clusters(&k, seq, head_dim, clusters, true);
                let cluster_count = assignment.iter().copied().max().map_or(0, |m| m + 1);
                let mut sizes = vec![0usize; cluster_count];
                for &l in &assignment {
                    sizes[l] += 1;
                }
                if trial == 0 {
                    let mut s = sizes.clone();
                    s.sort_by_key(|&x| core::cmp::Reverse(x));
                    shown = format!("{:?}", &s[..s.len().min(4)]);
                }

                let mut row_dots = 0usize;
                for i in 0..seq {
                    let mut candidates = 0usize;
                    let mut labels: Vec<usize> = (0..cluster_count).collect();
                    labels.sort_by_key(|&l| core::cmp::Reverse(sizes[l]));
                    for &l in &labels {
                        if candidates >= budget {
                            break;
                        }
                        candidates += (0..=i).filter(|&j| assignment[j] == l).count();
                    }
                    row_dots += cluster_count + candidates;
                }
                dots += row_dots as f64 / seq as f64;

                let mass =
                    |s: Selector| attention_mass_recovered(s, &q, &k, &v, seq, head_dim, true);
                let random = mass(Selector::Random { budget, seed: 1 });
                let oracle = mass(Selector::OracleTopK { budget });
                let routed = mass(Selector::TopologicalRouted { budget, clusters });
                place += (routed - random) / (oracle - random);
            }
            let t = trials as f64;
            let dense = (0..seq).map(|i| i + 1).sum::<usize>() as f64 / seq as f64;
            println!("{groups:>7} {tightness:>10.2} {spread:>8.1} {shown:>12} {:>10.1} {:>11.3} {:>10.3}",
                dots/t, (dots/t)/dense, place/t);
        }
    }
}
