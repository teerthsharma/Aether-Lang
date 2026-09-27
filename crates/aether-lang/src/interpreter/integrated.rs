//! Aether's bindings to the certified mathematics modules of `aether-core`.
//!
//! `import linking`, `import certify`, `import arrangement`, `import resolvent`,
//! `import orbit`, `import monodromy`, `import track`, `import coupling`,
//! `import kvwitness` and `import planner` each
//! bind a few native functions (`from linking import writhe` binds one). Results
//! are numbers, lists, strings or records read with `r.field`. Verdicts are
//! strings a program branches on (`r.verdict == "linked"`). Every typed refusal
//! of the core becomes a runtime error of the form `"<module> refused: <Variant
//! { data }>"`, never a number standing in for the answer.
//!
//! Points are lists of numbers, zero-padded to the dimension a routine works in
//! (a planar curve is a curve in `z = 0`). Matrices are lists of rows.

use super::{Interpreter, NativeFunction, Value};
use crate::ast::{CallArg, FnDecl};
use aether_core::arrangement::{self, ArrangementConfig, Segment};
use aether_core::certify::{self, Trit};
use aether_core::coupling::{self, CouplingOperator, IslandMetric};
use aether_core::kvwitness;
use aether_core::linking::{self, LinkVerdict};
use aether_core::manifold::ManifoldPoint;
use aether_core::monodromy::{self, CollisionVerdict};
use aether_core::orbit::{self, Partition};
use aether_core::planner::{self, TensorLifetime};
use aether_core::resolvent::{self, Switches};
use aether_core::track::{self, FlowConfig};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::{format, vec};
use core::cell::{Cell, RefCell};
use core::fmt::Debug;

/// `by_dim!(fn_name, d, f(args))` calls `f::<d>(args)` for `d` in `1..=4`.
macro_rules! by_dim {
    ($name:expr, $d:expr, $f:ident ( $($arg:expr),* )) => {
        match $d {
            1 => $f::<1>($($arg),*),
            2 => $f::<2>($($arg),*),
            3 => $f::<3>($($arg),*),
            4 => $f::<4>($($arg),*),
            d => Err(format!("{}: dimension {} is outside 1..=4", $name, d)),
        }
    };
}

/// Module name and the functions `import <module>` binds.
const MODULES: &[(&str, &[&str])] = &[
    ("linking", &["linking_number", "writhe", "knot_determinant"]),
    (
        "certify",
        &["certified_argmin", "certified_topk", "certified_threshold"],
    ),
    ("arrangement", &["euler"]),
    ("resolvent", &["resolvent_attend"]),
    ("orbit", &["orbit_partition"]),
    (
        "monodromy",
        &[
            "collision_certificate",
            "symmetry_group",
            "ph_dimension",
            "lyapunov",
        ],
    ),
    ("track", &["track"]),
    (
        "coupling",
        &["coupling_rollout", "coupling_fixed_point", "rips_islands"],
    ),
    (
        "kvwitness",
        &["witness_topk", "segment_coverage", "attention_mass_recall"],
    ),
    (
        "planner",
        &["plan_memory", "transitive_reduction", "islands"],
    ),
];

pub(super) fn exports(module: &str) -> Option<&'static [&'static str]> {
    MODULES.iter().find(|(m, _)| *m == module).map(|(_, f)| *f)
}

/// The static name of `function` when `module` exports it.
pub(super) fn lookup(module: &str, function: &str) -> Option<&'static str> {
    exports(module)?.iter().copied().find(|f| *f == function)
}

impl Interpreter {
    pub(super) fn import_integrated(
        &mut self,
        module: &str,
        symbol: Option<&str>,
        names: &'static [&'static str],
    ) -> Result<Value, String> {
        let bind = |this: &mut Self, name: &'static str| {
            this.variables.insert(
                String::from(name),
                Value::NativeFn(NativeFunction::Integrated(name)),
            );
        };
        match symbol {
            Some(symbol) => match names.iter().find(|n| **n == symbol) {
                Some(name) => bind(self, name),
                None => return Err(format!("Symbol '{}' not found in {}", symbol, module)),
            },
            None => {
                // Module first, so `import track` leaves `track` the function.
                self.variables
                    .insert(String::from(module), Value::Module(String::from(module)));
                for name in names {
                    bind(self, name);
                }
            }
        }
        Ok(Value::Unit)
    }

    pub(super) fn call_integrated(
        &mut self,
        name: &'static str,
        args: &[CallArg],
    ) -> Result<Value, String> {
        let mut a = Args {
            name,
            pos: Vec::new(),
            named: BTreeMap::new(),
        };
        for arg in args {
            match arg {
                CallArg::Positional(expr) => a.pos.push(self.evaluate_expr(expr)?),
                CallArg::Named { name, value } => {
                    let value = self.evaluate_expr(value)?;
                    a.named.insert(name.clone(), value);
                }
            }
        }
        match name {
            "linking_number" => linking_number(&a),
            "writhe" => linking::writhe(&points::<3>(name, a.at(0)?)?)
                .map(Value::Num)
                .map_err(|e| refused("linking", e)),
            "knot_determinant" => linking::knot_determinant(&points::<3>(name, a.at(0)?)?)
                .map(|d| Value::Num(d as f64))
                .map_err(|e| refused("linking", e)),
            "certified_argmin" => certify::certified_argmin(&nums(name, a.at(0)?)?, &radii(&a)?)
                .map(|i| Value::Num(i as f64))
                .map_err(|e| refused("certify", e)),
            "certified_topk" => certified_topk(&a),
            "certified_threshold" => certified_threshold(&a),
            "euler" => euler(&a),
            "resolvent_attend" => resolvent_attend(&a),
            "orbit_partition" => orbit_partition(&a),
            "collision_certificate" => self.collision_certificate(&a),
            "symmetry_group" => symmetry_group(&a),
            "ph_dimension" => self.ph_dimension(&a),
            "lyapunov" => {
                let js = list(name, a.at(0)?)?;
                let dt = a.named_num("dt", 1.0)?;
                by_dim!(
                    name,
                    js.first().map_or(Ok(1), |j| dim(name, j))?,
                    lyapunov(name, js, dt)
                )
            }
            "track" => track_frames(&a),
            "coupling_rollout" => {
                let steps = index(name, a.at(3)?)?;
                by_dim!(name, dim(name, a.at(0)?)?, rollout(&a, steps))
            }
            "coupling_fixed_point" => by_dim!(name, dim(name, a.at(0)?)?, fixed_point(&a)),
            "rips_islands" => rips_islands(&a),
            "witness_topk" => witness_topk(&a),
            "segment_coverage" => Ok(Value::Num(kvwitness::segment_coverage(
                &tokens(name, a.at(0)?)?,
                index(name, a.at(1)?)?,
                index(name, a.at(2)?)?,
            ))),
            "attention_mass_recall" => Ok(Value::Num(kvwitness::attention_mass_recall(
                &tokens(name, a.at(0)?)?,
                &nums(name, a.at(1)?)?,
            ))),
            "plan_memory" => plan_memory(&a),
            "transitive_reduction" => reduce_edges(&a),
            "islands" => planner_islands(&a),
            _ => Err(format!("{}: no binding", name)),
        }
    }

    fn collision_certificate(&mut self, a: &Args) -> Result<Value, String> {
        let (map, sampler) = (user_fn(a.name, a.at(0)?)?, user_fn(a.name, a.at(1)?)?);
        let sizes = a.sizes(&[100, 200, 400, 800])?;
        let (repeats, seed) = (a.named_index("repeats", 5)?, a.named_index("seed", 0)?);
        let dim = Cell::new(1);
        let host = Host::new(self);
        let result = monodromy::collision_certificate::<3, _, _>(
            |x| host.map(&map, x, dim.get()),
            |n, s| host.sample(&sampler, n, s, &dim),
            &sizes,
            repeats,
            seed as u64,
        );
        host.finish()?;
        let c = result.map_err(|e| refused("monodromy", e))?;
        let verdict = match c.verdict {
            CollisionVerdict::CollisionExhibited => "collision_exhibited",
            CollisionVerdict::NoCollisionAtThisSampling => "no_collision_at_this_sampling",
            CollisionVerdict::Undecided => "undecided",
        };
        let d = dim.get();
        Ok(record(vec![
            ("verdict", Value::Str(String::from(verdict))),
            ("collision", Value::Bool(c.collision_found())),
            ("lambda", Value::Num(c.witness.lambda)),
            ("free_ratio", Value::Num(c.free_ratio)),
            ("slope", Value::Num(c.scaling.slope)),
            ("image_gap", Value::Num(c.image_gap)),
            ("domain_gap", Value::Num(c.domain_gap)),
            ("p", unpad(&c.witness.p.coords, d)),
            ("q", unpad(&c.witness.q.coords, d)),
        ]))
    }

    fn ph_dimension(&mut self, a: &Args) -> Result<Value, String> {
        let sampler = user_fn(a.name, a.at(0)?)?;
        let alpha = a.named_num("alpha", 1.0)?;
        let sizes = a.sizes(&[100, 200, 400, 800])?;
        let (repeats, seed) = (a.named_index("repeats", 3)?, a.named_index("seed", 0)?);
        let dim = Cell::new(1);
        let host = Host::new(self);
        let result = monodromy::ph_dimension::<3, _>(
            |n, s| host.sample(&sampler, n, s, &dim),
            alpha,
            &sizes,
            repeats,
            seed as u64,
        );
        host.finish()?;
        let fit = result.map_err(|e| refused("monodromy", e))?;
        Ok(record(vec![
            ("dimension", Value::Num(fit.dimension)),
            ("ci_low", Value::Num(fit.ci95.0)),
            ("ci_high", Value::Num(fit.ci95.1)),
            ("slope", Value::Num(fit.slope)),
            ("n_obs", Value::Num(fit.n_obs as f64)),
        ]))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// One binding per module function
// ═══════════════════════════════════════════════════════════════════════════════

fn linking_number(a: &Args) -> Result<Value, String> {
    let (x, y) = (
        points::<3>(a.name, a.at(0)?)?,
        points::<3>(a.name, a.at(1)?)?,
    );
    let g = linking::linking_number(&x, &y).map_err(|e| refused("linking", e))?;
    let mut fields = vec![
        ("value", Value::Num(g.value)),
        ("error_bound", Value::Num(g.error_bound)),
    ];
    // An undetermined verdict has no `lk` field, so reading it is an error
    // rather than a zero.
    let verdict = match g.certify() {
        LinkVerdict::Linked { lk } => {
            fields.push(("lk", Value::Num(lk as f64)));
            "linked"
        }
        LinkVerdict::ZeroLinking => {
            fields.push(("lk", Value::Num(0.0)));
            "zero_linking"
        }
        LinkVerdict::Undetermined { .. } => "undetermined",
    };
    fields.push(("verdict", Value::Str(String::from(verdict))));
    Ok(record(fields))
}

/// One radius for every score, or one per score.
fn radii(a: &Args) -> Result<Vec<f64>, String> {
    match a.at(1)? {
        Value::Num(r) => Ok(vec![*r]),
        v => nums(a.name, v),
    }
}

fn certified_topk(a: &Args) -> Result<Value, String> {
    let k = index(a.name, a.at(2)?)?;
    let (largest, ordered) = (a.named_bool("largest")?, a.named_bool("ordered")?);
    certify::certified_topk(&nums(a.name, a.at(0)?)?, &radii(a)?, k, largest, ordered)
        .map(|set| nums_value(set.into_iter().map(|i| i as f64)))
        .map_err(|e| refused("certify", e))
}

fn certified_threshold(a: &Args) -> Result<Value, String> {
    let t = num(a.name, a.at(2)?)?;
    let trits = certify::certified_threshold(&nums(a.name, a.at(0)?)?, &radii(a)?, t)
        .map_err(|e| refused("certify", e))?;
    Ok(Value::List(
        trits
            .into_iter()
            .map(|t| {
                Value::Str(String::from(match t {
                    Trit::Below => "below",
                    Trit::Undetermined => "undetermined",
                    Trit::Above => "above",
                }))
            })
            .collect(),
    ))
}

fn euler(a: &Args) -> Result<Value, String> {
    let segments = list(a.name, a.at(0)?)?
        .iter()
        .map(|s| match list(a.name, s)? {
            [p, q] => Ok([point::<2>(a.name, p)?, point::<2>(a.name, q)?]),
            _ => Err(format!("{}: a segment is [[x1, y1], [x2, y2]]", a.name)),
        })
        .collect::<Result<Vec<Segment>, String>>()?;
    let grid = a.named.get("grid").map(|g| num(a.name, g)).transpose()?;
    let config = ArrangementConfig {
        grid,
        ..ArrangementConfig::default()
    };
    let c = arrangement::arrange(&segments, &config).map_err(|e| refused("arrangement", e))?;
    Ok(record(vec![
        ("pieces", Value::Num(c.pieces as f64)),
        ("faces", Value::Num(c.faces as f64)),
        ("chi", Value::Num(c.chi as f64)),
        ("vertices", Value::Num(c.vertices as f64)),
        ("edges", Value::Num(c.edges as f64)),
        ("radius", Value::Num(c.radius)),
    ]))
}

fn resolvent_attend(a: &Args) -> Result<Value, String> {
    let (q, seq, d, _) = matrix(a.name, a.at(0)?)?;
    let (k, seq_k, d_k, _) = matrix(a.name, a.at(1)?)?;
    let (v, seq_v, value_dim, flat) = matrix(a.name, a.at(2)?)?;
    let gates = nums(a.name, a.at(3)?)?;
    let switches = match text(a.name, a.at(4)?)? {
        "softmax" => Switches::SOFTMAX,
        "kernel" => Switches::UNNORMALIZED_KERNEL,
        "path" => Switches::PATH_PRODUCT,
        other => {
            return Err(format!(
                "{}: unknown switch '{}', expected \"softmax\", \"kernel\" or \"path\"",
                a.name, other
            ))
        }
    };
    // The core asserts shapes; a program gets an error instead of a panic.
    if seq_k != seq || seq_v != seq || gates.len() != seq || d_k != d || (seq > 0 && d == 0) {
        return Err(format!(
            "{}: q, k, v and gates need one row per position and q, k one width; got q {}x{}, k {}x{}, v {} rows, {} gates",
            a.name, seq, d, seq_k, d_k, seq_v, gates.len()
        ));
    }
    let theta = vec![0.0; seq];
    let w = resolvent::operator(&q, &k, &gates, &theta, seq, d, switches)
        .map_err(|e| refused("resolvent", e))?;
    // Gates enter with phase 0, so every imaginary lane is exactly 0.
    let out: Vec<f64> = resolvent::readout(&w, &v, seq, value_dim)
        .iter()
        .map(|c| c.re)
        .collect();
    Ok(if flat {
        nums_value(out)
    } else {
        Value::List(
            out.chunks(value_dim.max(1))
                .map(|r| nums_value(r.iter().copied()))
                .collect(),
        )
    })
}

fn orbit_partition(a: &Args) -> Result<Value, String> {
    let values = keys(a.name, a.at(0)?)?;
    let p = Partition::from_map(&values);
    let (n, m, flagged) = (p.n(), p.m(), p.flagged());
    let floor = orbit::orbit_error_bound(n, m).ok_or_else(|| {
        format!(
            "orbit refused: orbit_error_bound undefined at n = {}, m = {} (an empty map has no orbits)",
            n, m
        )
    })?;
    let mut fields = vec![
        ("n", Value::Num(n as f64)),
        ("m", Value::Num(m as f64)),
        ("largest", Value::Num(p.largest() as f64)),
        ("flagged", Value::Num(flagged as f64)),
        ("collapsed", Value::Num(p.collapsed_blocks() as f64)),
        ("error_floor", Value::Num(floor as f64)),
        ("error_rate", Value::Num(floor as f64 / n as f64)),
    ];
    if let Some(r) = orbit::pooling_recovery_bound(p.largest()) {
        fields.push(("recovery_ceiling", Value::Num(r)));
    }
    // Undefined on a map with no collision (precision of an empty set): the
    // field is absent rather than 0 or 1.
    if let Some(prec) = orbit::certified_precision_bound(n, m, flagged) {
        fields.push(("precision_floor", Value::Num(prec)));
    }
    if let Some(gold) = a.named.get("gold") {
        let gold = keys(a.name, gold)?;
        let m_star = orbit::admissible_distinct(&values, &gold).ok_or_else(|| {
            format!(
                "orbit refused: admissible_distinct undefined, gold holds fewer than n = {} distinct values",
                n
            )
        })?;
        fields.push(("m_star", Value::Num(m_star as f64)));
        if let Some(e) = orbit::admissible_error_bound(n, m_star) {
            fields.push(("admissible_error_floor", Value::Num(e as f64)));
        }
        if let Some(r) = orbit::recall_floor(n, m_star) {
            fields.push(("recall_floor", Value::Num(r)));
        }
    }
    Ok(record(fields))
}

/// Orbit values are compared exactly, so each becomes an exact, ordered key.
fn keys(f: &str, v: &Value) -> Result<Vec<String>, String> {
    list(f, v)?
        .iter()
        .map(|x| match x {
            Value::Num(n) if n.is_nan() => Err(format!("{}: NaN has no orbit", f)),
            Value::Num(n) if *n == 0.0 => Ok(String::from("0")),
            Value::Num(_) | Value::Str(_) => Ok(format!("{:?}", x)),
            other => Err(format!(
                "{}: values must be numbers or strings, got {}",
                f,
                kind(other)
            )),
        })
        .collect()
}

fn symmetry_group(a: &Args) -> Result<Value, String> {
    let cloud: Vec<ManifoldPoint<2>> = points::<2>(a.name, a.at(0)?)?
        .into_iter()
        .map(ManifoldPoint::new)
        .collect();
    let step = a.named_num("step", monodromy::DEFAULT_GRID_STEP_DEG)?;
    let g = monodromy::recover_dihedral(&cloud, step).map_err(|e| refused("monodromy", e))?;
    Ok(record(vec![
        ("order", Value::Num(g.order as f64)),
        ("dihedral", Value::Bool(g.dihedral)),
        ("power_fraction", Value::Num(g.power_fraction)),
    ]))
}

fn lyapunov<const D: usize>(f: &str, js: &[Value], dt: f64) -> Result<Value, String> {
    let js = js
        .iter()
        .map(|j| square::<D>(f, j))
        .collect::<Result<Vec<_>, _>>()?;
    monodromy::lyapunov_spectrum::<D, _>(js, dt)
        .map(|l| nums_value(l.iter().copied()))
        .map_err(|e| refused("monodromy", e))
}

fn track_frames(a: &Args) -> Result<Value, String> {
    let frames = list(a.name, a.at(0)?)?
        .iter()
        .map(|frame| points::<3>(a.name, frame))
        .collect::<Result<Vec<_>, _>>()?;
    let gate = num(a.name, a.at(1)?)?;
    // Isotropic unit spacing: distances are in the frames' own units.
    let scale = [1.0; 3];
    let divide = a
        .named
        .get("divide")
        .map_or(Ok(true), |v| boolean(a.name, v))?;
    let mut fields = Vec::new();
    let graph = if divide {
        let config = FlowConfig {
            max_um: gate,
            ..FlowConfig::default()
        };
        let r = track::flow_track(&frames, &config, scale).map_err(|e| refused("track", e))?;
        fields.push(("certified", Value::Bool(r.certified())));
        fields.push(("cost", Value::Num(r.cost)));
        r.graph
    } else {
        track::link_sequence(&frames, gate, scale).map_err(|e| refused("track", e))?
    };
    let node = |i: usize| (graph.nodes[i].frame as f64, graph.nodes[i].index as f64);
    let edges = graph.edges.iter().map(|&(p, c)| {
        let ((fp, ip), (fc, ic)) = (node(p), node(c));
        nums_value([fp, ip, fc, ic])
    });
    let dividing = graph.divisions.iter().map(|&i| {
        let (f, x) = node(i);
        nums_value([f, x])
    });
    fields.extend([
        ("nodes", Value::Num(graph.nodes.len() as f64)),
        ("edges", Value::List(edges.collect())),
        ("divisions", Value::Num(graph.divisions.len() as f64)),
        ("dividing", Value::List(dividing.collect())),
    ]);
    Ok(record(fields))
}

fn coupling_operator<const D: usize>(a: &Args) -> Result<CouplingOperator<D, 0>, String> {
    Ok(CouplingOperator::autonomous(
        square::<D>(a.name, a.at(0)?)?,
        vector::<D>(a.name, a.at(1)?)?,
    ))
}

fn rollout<const D: usize>(a: &Args, steps: usize) -> Result<Value, String> {
    let op = coupling_operator::<D>(a)?;
    let z0 = vector::<D>(a.name, a.at(2)?)?;
    let mut out = vec![[0.0; D]; steps];
    op.rollout(&z0, &[], &mut out)
        .map_err(|e| refused("coupling", e))?;
    Ok(Value::List(
        out.iter().map(|z| nums_value(z.iter().copied())).collect(),
    ))
}

fn fixed_point<const D: usize>(a: &Args) -> Result<Value, String> {
    let op = coupling_operator::<D>(a)?;
    let fp = op.fixed_point().ok_or_else(|| {
        String::from(
            "coupling refused: NoFixedPoint (I - T is singular or (I - T)^-1 c is not finite)",
        )
    })?;
    let rho = op.spectral_norm();
    Ok(record(vec![
        ("point", nums_value(fp.point.iter().copied())),
        ("converged", Value::Bool(fp.converged)),
        ("rho", Value::Num(rho)),
        ("contractive", Value::Bool(rho < 1.0)),
    ]))
}

fn rips_islands(a: &Args) -> Result<Value, String> {
    let cloud: Vec<ManifoldPoint<3>> = points::<3>(a.name, a.at(0)?)?
        .into_iter()
        .map(ManifoldPoint::new)
        .collect();
    let mut labels = vec![0; cloud.len()];
    let radius = num(a.name, a.at(1)?)?;
    coupling::island_labels(&cloud, radius, IslandMetric::Euclidean, &mut labels)
        .map(|n| Value::Num(n as f64))
        .map_err(|e| refused("coupling", e))
}

/// Merged top-k rows: `learned` is row-major `[rows, topk]`, one context length
/// per row (a number for one row), and the last `segments` slots of each row
/// give way to witnesses of the segments the row misses. `segments = 0` is off.
fn witness_topk(a: &Args) -> Result<Value, String> {
    let mut rows = tokens(a.name, a.at(0)?)?;
    let topk = index(a.name, a.at(1)?)?;
    let lens = match a.at(2)? {
        Value::Num(n) => vec![token(a.name, *n)?],
        v => tokens(a.name, v)?,
    };
    let segments = index(a.name, a.at(3)?)?;
    kvwitness::scatter_topology_witnesses(&mut rows, topk, &lens, segments)
        .map_err(|e| refused("kvwitness", e))?;
    Ok(nums_value(rows.into_iter().map(f64::from)))
}

/// `tensors` is a list of `[size, first_use, last_use]`.
fn plan_memory(a: &Args) -> Result<Value, String> {
    let tensors = list(a.name, a.at(0)?)?
        .iter()
        .enumerate()
        .map(|(i, t)| match list(a.name, t)? {
            [size, first, last] => {
                let (size, first_use, last_use) =
                    (index(a.name, size)?, index(a.name, first)?, index(a.name, last)?);
                if first_use > last_use {
                    return Err(format!(
                        "planner refused: ReversedLifetime {{ tensor: {}, first_use: {}, last_use: {} }}",
                        i, first_use, last_use
                    ));
                }
                Ok(TensorLifetime { size, first_use, last_use })
            }
            _ => Err(format!("{}: a tensor is [size, first_use, last_use]", a.name)),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let plan = planner::plan_offsets(&tensors);
    Ok(record(vec![
        (
            "offsets",
            nums_value(plan.offsets.iter().map(|&o| o as f64)),
        ),
        ("arena", Value::Num(plan.arena_size as f64)),
        (
            "peak",
            Value::Num(planner::peak_live_bytes(&tensors) as f64),
        ),
    ]))
}

fn reduce_edges(a: &Args) -> Result<Value, String> {
    let n = index(a.name, a.at(0)?)?;
    let edges = pairs(a.name, a.at(1)?)?;
    // The core panics on these; a program gets the refusal instead.
    if let Some((i, (u, v))) = edges
        .iter()
        .enumerate()
        .find(|(_, (u, v))| u >= v || *v >= n)
    {
        return Err(format!(
            "planner refused: NotTopological {{ edge: {}, u: {}, v: {}, node_count: {} }}",
            i, u, v, n
        ));
    }
    Ok(Value::List(
        planner::transitive_reduction(n, &edges)
            .into_iter()
            .map(|(u, v)| nums_value([u as f64, v as f64]))
            .collect(),
    ))
}

/// Island of every node, `-1` for a node no incidence touches (MuJoCo's mark).
fn planner_islands(a: &Args) -> Result<Value, String> {
    let n = index(a.name, a.at(0)?)?;
    let incidences = pairs(a.name, a.at(1)?)?;
    if let Some((i, (x, y))) = incidences
        .iter()
        .enumerate()
        .find(|(_, (x, y))| *x >= n || *y >= n)
    {
        return Err(format!(
            "planner refused: NodeOutOfRange {{ incidence: {}, nodes: [{}, {}], node_count: {} }}",
            i, x, y, n
        ));
    }
    let (labels, count) = planner::islands(n, &incidences);
    Ok(record(vec![
        ("count", Value::Num(count as f64)),
        (
            "labels",
            nums_value(labels.iter().map(|l| l.map_or(-1.0, |k| k as f64))),
        ),
    ]))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Calling a program's functions from inside a core routine
// ═══════════════════════════════════════════════════════════════════════════════

/// Lets core closures call back into the interpreter. The core takes plain
/// closures and has no error type of ours, so the first error a user function
/// raises is parked here, NaN or an empty draw stands in for its value, and
/// [`Host::finish`] reports it ahead of whatever the core made of the stand-in.
struct Host<'a> {
    interp: RefCell<&'a mut Interpreter>,
    error: RefCell<Option<String>>,
}

impl<'a> Host<'a> {
    fn new(interp: &'a mut Interpreter) -> Self {
        Self {
            interp: RefCell::new(interp),
            error: RefCell::new(None),
        }
    }

    fn call(&self, func: &FnDecl, args: Vec<Value>) -> Option<Value> {
        if self.error.borrow().is_some() {
            return None;
        }
        let result = self.interp.borrow_mut().call_user_fn(func, args);
        result.map_err(|e| self.fail(e)).ok()
    }

    fn fail(&self, e: String) {
        self.error.borrow_mut().get_or_insert(e);
    }

    /// `func` on a `dim`-dimensional point, padded back to three coordinates.
    fn map(&self, func: &FnDecl, x: &[f64; 3], dim: usize) -> [f64; 3] {
        let result = self.call(func, vec![unpad(x, dim)]);
        match result.map(|v| point::<3>(&func.name, &v)) {
            Some(Ok(p)) => p,
            Some(Err(e)) => {
                self.fail(e);
                [f64::NAN; 3]
            }
            None => [f64::NAN; 3],
        }
    }

    /// `func(n, seed)`, a list of points; records their dimension in `dim`.
    fn sample(
        &self,
        func: &FnDecl,
        n: usize,
        seed: u64,
        dim: &Cell<usize>,
    ) -> Vec<ManifoldPoint<3>> {
        let Some(draw) = self.call(func, vec![Value::Num(n as f64), Value::Num(seed as f64)])
        else {
            return Vec::new();
        };
        if let Some(Ok(c)) = list(&func.name, &draw)
            .ok()
            .and_then(|p| p.first())
            .map(coords)
        {
            dim.set(c.len());
        }
        match points::<3>(&func.name, &draw) {
            Ok(p) => p.into_iter().map(ManifoldPoint::new).collect(),
            Err(e) => {
                self.fail(e);
                Vec::new()
            }
        }
    }

    fn finish(self) -> Result<(), String> {
        self.error.into_inner().map_or(Ok(()), Err)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Values in and out
// ═══════════════════════════════════════════════════════════════════════════════

struct Args {
    name: &'static str,
    pos: Vec<Value>,
    named: BTreeMap<String, Value>,
}

impl Args {
    fn at(&self, i: usize) -> Result<&Value, String> {
        self.pos
            .get(i)
            .ok_or_else(|| format!("{}: missing argument {}", self.name, i + 1))
    }

    fn named_num(&self, key: &str, default: f64) -> Result<f64, String> {
        self.named
            .get(key)
            .map_or(Ok(default), |v| num(self.name, v))
    }

    fn named_index(&self, key: &str, default: usize) -> Result<usize, String> {
        self.named
            .get(key)
            .map_or(Ok(default), |v| index(self.name, v))
    }

    fn named_bool(&self, key: &str) -> Result<bool, String> {
        self.named
            .get(key)
            .map_or(Ok(false), |v| boolean(self.name, v))
    }

    fn sizes(&self, default: &[usize]) -> Result<Vec<usize>, String> {
        match self.named.get("sizes") {
            Some(v) => list(self.name, v)?
                .iter()
                .map(|s| index(self.name, s))
                .collect(),
            None => Ok(default.to_vec()),
        }
    }
}

fn refused(module: &str, err: impl Debug) -> String {
    format!("{} refused: {:?}", module, err)
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Num(_) => "number",
        Value::Bool(_) => "bool",
        Value::Str(_) => "string",
        Value::List(_) => "list",
        Value::Record(_) => "record",
        Value::Unit => "unit",
        Value::Function(_) | Value::NativeFn(_) => "function",
        _ => "value",
    }
}

fn num(f: &str, v: &Value) -> Result<f64, String> {
    match v {
        Value::Num(n) => Ok(*n),
        other => Err(format!("{}: expected a number, got {}", f, kind(other))),
    }
}

fn index(f: &str, v: &Value) -> Result<usize, String> {
    let n = num(f, v)?;
    if n >= 0.0 && libm::trunc(n) == n && n <= usize::MAX as f64 {
        Ok(n as usize)
    } else {
        Err(format!("{}: expected a non-negative integer, got {}", f, n))
    }
}

fn boolean(f: &str, v: &Value) -> Result<bool, String> {
    match v {
        Value::Bool(b) => Ok(*b),
        other => Err(format!("{}: expected a bool, got {}", f, kind(other))),
    }
}

fn text<'v>(f: &str, v: &'v Value) -> Result<&'v str, String> {
    match v {
        Value::Str(s) => Ok(s),
        other => Err(format!("{}: expected a string, got {}", f, kind(other))),
    }
}

fn list<'v>(f: &str, v: &'v Value) -> Result<&'v [Value], String> {
    match v {
        Value::List(xs) => Ok(xs),
        other => Err(format!("{}: expected a list, got {}", f, kind(other))),
    }
}

fn user_fn(f: &str, v: &Value) -> Result<FnDecl, String> {
    match v {
        Value::Function(decl) => Ok(decl.clone()),
        other => Err(format!(
            "{}: expected a fn defined in the program, got {}",
            f,
            kind(other)
        )),
    }
}

/// A token offset: an integer in `i32`, `-1` being padding.
fn token(f: &str, x: f64) -> Result<i32, String> {
    if libm::trunc(x) == x && (i32::MIN as f64..=i32::MAX as f64).contains(&x) {
        Ok(x as i32)
    } else {
        Err(format!("{}: expected an i32 token offset, got {}", f, x))
    }
}

fn tokens(f: &str, v: &Value) -> Result<Vec<i32>, String> {
    list(f, v)?.iter().map(|x| token(f, num(f, x)?)).collect()
}

/// A list of `[u, v]` index pairs.
fn pairs(f: &str, v: &Value) -> Result<Vec<(usize, usize)>, String> {
    list(f, v)?
        .iter()
        .map(|p| match list(f, p)? {
            [u, v] => Ok((index(f, u)?, index(f, v)?)),
            _ => Err(format!("{}: expected a pair [u, v]", f)),
        })
        .collect()
}

fn nums(f: &str, v: &Value) -> Result<Vec<f64>, String> {
    list(f, v)?.iter().map(|x| num(f, x)).collect()
}

fn coords(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Num(n) => Ok(vec![*n]),
        Value::Point(p) => Ok(p.to_vec()),
        other => nums("point", other),
    }
}

/// A point of at most `N` coordinates, zero-padded to `N`.
fn point<const N: usize>(f: &str, v: &Value) -> Result<[f64; N], String> {
    let c = coords(v).map_err(|e| format!("{}: {}", f, e))?;
    if c.is_empty() || c.len() > N {
        return Err(format!(
            "{}: expected a point of 1 to {} coordinates, got {}",
            f,
            N,
            c.len()
        ));
    }
    let mut p = [0.0; N];
    p[..c.len()].copy_from_slice(&c);
    Ok(p)
}

fn points<const N: usize>(f: &str, v: &Value) -> Result<Vec<[f64; N]>, String> {
    list(f, v)?.iter().map(|p| point::<N>(f, p)).collect()
}

fn unpad(x: &[f64], dim: usize) -> Value {
    if dim == 1 {
        Value::Num(x[0])
    } else {
        nums_value(x[..dim].iter().copied())
    }
}

/// Exactly `D` numbers (a number for `D = 1`).
fn vector<const D: usize>(f: &str, v: &Value) -> Result<[f64; D], String> {
    let c = coords(v).map_err(|e| format!("{}: {}", f, e))?;
    c.as_slice()
        .try_into()
        .map_err(|_| format!("{}: expected {} numbers, got {}", f, D, c.len()))
}

/// A `D x D` matrix as a list of rows (a number for `D = 1`).
fn square<const D: usize>(f: &str, v: &Value) -> Result<[[f64; D]; D], String> {
    let mut m = [[0.0; D]; D];
    if let (Value::Num(x), 1) = (v, D) {
        m[0][0] = *x;
        return Ok(m);
    }
    let rows = list(f, v)?;
    if rows.len() != D {
        return Err(format!(
            "{}: expected a {}x{} matrix, got {} rows",
            f,
            D,
            D,
            rows.len()
        ));
    }
    for (out, row) in m.iter_mut().zip(rows) {
        *out = vector::<D>(f, row)?;
    }
    Ok(m)
}

/// The matrix dimension a value encodes: 1 for a number, else its row count.
fn dim(f: &str, v: &Value) -> Result<usize, String> {
    match v {
        Value::Num(_) => Ok(1),
        other => Ok(list(f, other)?.len()),
    }
}

/// A list of rows, or a flat list read as one column (`flat`), row-major.
fn matrix(f: &str, v: &Value) -> Result<(Vec<f64>, usize, usize, bool), String> {
    let rows = list(f, v)?;
    if rows.iter().all(|r| matches!(r, Value::Num(_))) {
        return Ok((nums(f, v)?, rows.len(), 1, true));
    }
    let width = rows
        .first()
        .map_or(Ok(0), |r| list(f, r).map(<[Value]>::len))?;
    let mut data = Vec::with_capacity(rows.len() * width);
    for r in rows {
        let r = nums(f, r)?;
        if r.len() != width {
            return Err(format!(
                "{}: ragged matrix, rows of {} and {}",
                f,
                width,
                r.len()
            ));
        }
        data.extend(r);
    }
    Ok((data, rows.len(), width, false))
}

fn nums_value(xs: impl IntoIterator<Item = f64>) -> Value {
    Value::List(xs.into_iter().map(Value::Num).collect())
}

fn record(fields: Vec<(&str, Value)>) -> Value {
    Value::Record(
        fields
            .into_iter()
            .map(|(k, v)| (String::from(k), v))
            .collect(),
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Language support: fields, elements, and the equality `stable()` seals on
// ═══════════════════════════════════════════════════════════════════════════════

pub(super) fn field(v: &Value, name: &str) -> Result<Value, String> {
    match v {
        Value::Record(fields) => fields
            .get(name)
            .cloned()
            .ok_or_else(|| format!("record has no field '{}'", name)),
        other => Err(format!("cannot read field '{}' of a {}", name, kind(other))),
    }
}

pub(super) fn element(list: &Value, i: &Value) -> Result<Value, String> {
    let Value::List(xs) = list else {
        return Err(format!("cannot index a {}", kind(list)));
    };
    let i = index("index", i)?;
    xs.get(i)
        .cloned()
        .ok_or_else(|| format!("index {} out of range for a list of {}", i, xs.len()))
}

/// Structural equality over the values an invariant can take.
pub(super) fn same(a: &Value, b: &Value) -> Result<bool, String> {
    Ok(match (a, b) {
        (Value::Num(x), Value::Num(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Unit, Value::Unit) => true,
        (Value::List(x), Value::List(y)) => x.len() == y.len() && all_same(x.iter().zip(y))?,
        (Value::Record(x), Value::Record(y)) => {
            x.len() == y.len() && x.keys().eq(y.keys()) && all_same(x.values().zip(y.values()))?
        }
        (x, y) if kind(x) != kind(y) => false,
        (x, _) => return Err(format!("stable() cannot compare a {}", kind(x))),
    })
}

fn all_same<'v>(mut pairs: impl Iterator<Item = (&'v Value, &'v Value)>) -> Result<bool, String> {
    pairs.try_fold(true, |acc, (x, y)| Ok(acc && same(x, y)?))
}
