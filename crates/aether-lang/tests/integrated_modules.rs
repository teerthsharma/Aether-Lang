//! The certified mathematics modules as language features: each `import`
//! binds functions whose results match closed-form truths, and each typed
//! refusal of the core reaches the program as an error naming the variant.

use aether_lang::interpreter::Value;
use aether_lang::{Interpreter, Parser};

fn run(src: &str) -> Result<Interpreter, String> {
    let program = Parser::new(src).parse().map_err(|e| e.to_string())?;
    let mut interp = Interpreter::new();
    interp.execute(&program)?;
    Ok(interp)
}

fn var(src: &str, name: &str) -> Value {
    let interp = run(src).unwrap_or_else(|e| panic!("program failed: {e}"));
    interp.variables[name].clone()
}

fn refusal(src: &str) -> String {
    run(src).err().expect("program should be refused")
}

fn num(v: &Value) -> f64 {
    match v {
        Value::Num(n) => *n,
        other => panic!("expected a number, got {other:?}"),
    }
}

fn nums(v: &Value) -> Vec<f64> {
    match v {
        Value::List(xs) => xs.iter().map(num).collect(),
        other => panic!("expected a list, got {other:?}"),
    }
}

fn text(v: &Value) -> &str {
    match v {
        Value::Str(s) => s,
        other => panic!("expected a string, got {other:?}"),
    }
}

fn field<'a>(v: &'a Value, name: &str) -> &'a Value {
    match v {
        Value::Record(fields) => &fields[name],
        other => panic!("expected a record, got {other:?}"),
    }
}

const SQUARE: &str = "let square = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]~";
const SQUARE_2D: &str =
    "let square = [[[0, 0], [2, 0]], [[2, 0], [2, 2]], [[2, 2], [0, 2]], [[0, 2], [0, 0]]]~
     let spokes = [[[0, 0], [1, 1]], [[2, 0], [1, 1]], [[2, 2], [1, 1]], [[0, 2], [1, 1]]]~";

#[test]
fn linking_certifies_the_hopf_link_and_the_classical_invariants() {
    let src = format!(
        "import linking~ {SQUARE}
         let hopf = linking_number(square, [[0.5, 0.5, -1], [0.5, 0.5, 1], [0.5, -0.5, 1], [0.5, -0.5, -1]])~
         let split = linking_number(square, [[0.5, -0.5, -1], [0.5, -0.5, 1], [0.5, -1.5, 1], [0.5, -1.5, -1]])~
         let w = writhe(square)~
         let det = knot_determinant(square)~"
    );
    let interp = run(&src).unwrap();
    let hopf = &interp.variables["hopf"];
    assert_eq!(text(field(hopf, "verdict")), "linked");
    assert_eq!(num(field(hopf, "lk")), 1.0);
    assert!(num(field(hopf, "error_bound")) < 0.5);
    let split = &interp.variables["split"];
    assert_eq!(text(field(split, "verdict")), "zero_linking");
    assert_eq!(num(field(split, "lk")), 0.0);
    assert_eq!(num(&interp.variables["w"]), 0.0);
    assert_eq!(num(&interp.variables["det"]), 1.0);
}

#[test]
fn linking_refusal_names_the_variant() {
    let err = refusal("import linking~ writhe([[0, 0, 0], [1, 0, 0]])~");
    assert_eq!(err, "linking refused: TooFewVertices { curve: 0, len: 2 }");
}

#[test]
fn certify_decides_argmin_topk_and_threshold() {
    let src = "import certify~
               let s = [3.0, 1.0, 2.0, 5.0]~
               let arg = certified_argmin(s, 0.1)~
               let top = certified_topk(s, 0.1, 2)~
               let big = certified_topk(s, [0.1, 0.1, 0.1, 0.1], 1, largest=true)~
               let side = certified_threshold(s, 0.1, 2.05)~";
    let interp = run(src).unwrap();
    assert_eq!(num(&interp.variables["arg"]), 1.0);
    assert_eq!(nums(&interp.variables["top"]), [1.0, 2.0]);
    assert_eq!(nums(&interp.variables["big"]), [3.0]);
    let Value::List(side) = &interp.variables["side"] else {
        panic!("expected trits")
    };
    let side: Vec<&str> = side.iter().map(text).collect();
    assert_eq!(side, ["above", "below", "undetermined", "above"]);
}

#[test]
fn certify_refusal_names_the_frontier() {
    let err = refusal("import certify~ certified_argmin([1.0, 2.0], 0.6)~");
    assert!(
        err.starts_with("certify refused: BoundaryUndetermined"),
        "{err}"
    );
    assert!(err.contains("straddling: [0, 1]"), "{err}");
}

#[test]
fn arrangement_counts_the_square_with_half_diagonals() {
    let src = format!(
        "import arrangement~ {SQUARE_2D}
         let alone = euler(square)~
         for i in 0..4 {{ square.push(spokes[i])~ }}
         let full = euler(square)~"
    );
    let interp = run(&src).unwrap();
    let alone = &interp.variables["alone"];
    assert_eq!(
        [
            field(alone, "pieces"),
            field(alone, "faces"),
            field(alone, "chi")
        ]
        .map(num),
        [1.0, 1.0, 0.0]
    );
    let full = &interp.variables["full"];
    assert_eq!(
        [
            field(full, "pieces"),
            field(full, "faces"),
            field(full, "chi")
        ]
        .map(num),
        [1.0, 4.0, -3.0]
    );
    assert_eq!(num(field(full, "vertices")), 5.0);
    assert_eq!(num(field(full, "edges")), 8.0);
}

#[test]
fn arrangement_refuses_unnoded_crossing_diagonals() {
    let err = refusal("import arrangement~ euler([[[0, 0], [2, 2]], [[2, 0], [0, 2]]])~");
    assert_eq!(
        err,
        "arrangement refused: EdgesCross { first: 0, second: 1, count: 1 }"
    );
}

#[test]
fn resolvent_switches_reach_the_three_corners() {
    let src = "import resolvent~
               let q = [[0], [0], [0]]~
               let g = [1, 0.5, 0.5]~
               let soft = resolvent_attend(q, q, [1, 1, 1], g, \"softmax\")~
               let kern = resolvent_attend(q, q, [1, 1, 1], g, \"kernel\")~
               let path = resolvent_attend(q, q, [0, 1, 1], g, \"path\")~
               let cut = resolvent_attend(q, q, [1, 1, 1], [1, 0, 1], \"path\")~";
    let interp = run(src).unwrap();
    for s in nums(&interp.variables["soft"]) {
        assert!((s - 1.0).abs() < 1e-15, "softmax row sum {s}");
    }
    assert_eq!(nums(&interp.variables["kern"]), [1.0, 2.0, 3.0]);
    assert_eq!(nums(&interp.variables["path"]), [0.0, 1.0, 1.5]);
    assert_eq!(nums(&interp.variables["cut"]), [1.0, 1.0, 2.0]);
}

#[test]
fn resolvent_refuses_a_nan_query() {
    let err = refusal(
        "import resolvent~
         resolvent_attend([[0 / 0], [0]], [[0], [0]], [1, 1], [1, 1], \"softmax\")~",
    );
    assert_eq!(err, "resolvent refused: NonFinite");
}

#[test]
fn orbit_partition_reports_the_proved_bounds() {
    let o = var(
        "import orbit~ let o = orbit_partition([1, 1, 2, 3, 3, 3], gold=[1, 2, 3, 4, 5, 6])~",
        "o",
    );
    let get = |k| num(field(&o, k));
    assert_eq!((get("n"), get("m"), get("error_floor")), (6.0, 3.0, 3.0));
    assert_eq!(get("precision_floor"), 0.6);
    assert_eq!(get("recovery_ceiling"), 1.0 / 3.0);
    assert_eq!((get("m_star"), get("recall_floor")), (3.0, 0.5));
}

#[test]
fn orbit_refuses_an_empty_map_and_omits_undefined_precision() {
    let err = refusal("import orbit~ orbit_partition([])~");
    assert!(
        err.starts_with("orbit refused: orbit_error_bound undefined at n = 0"),
        "{err}"
    );
    let err = refusal("import orbit~ let o = orbit_partition([1, 2])~ o.precision_floor~");
    assert_eq!(err, "record has no field 'precision_floor'");
}

const SAMPLER: &str = "fn line(n, seed) {
        let xs = []~
        let i = 0~
        while i < n {
            let u = i + 1 + seed * 7919~
            let frac = u * 0.618034 % 1~
            xs.push(2 * frac - 1)~
            i = i + 1~
        }
        return xs~
    }";

#[test]
fn monodromy_decides_collisions_symmetry_dimension_and_exponents() {
    let src = format!(
        "import monodromy~ import math~ {SAMPLER}
         fn fold(x) {{ return x * x~ }}
         fn affine(x) {{ return 2 * x + 1~ }}
         let fold_c = collision_certificate(fold, line, sizes=[25, 50, 100, 200], repeats=3)~
         let affine_c = collision_certificate(affine, line, sizes=[25, 50, 100, 200], repeats=3)~
         let outline = []~
         for e in 0..4 {{
             let e1 = e + 1~
             let x0 = cos(pi * e / 2)~
             let y0 = sin(pi * e / 2)~
             let dx = cos(pi * e1 / 2) - x0~
             let dy = sin(pi * e1 / 2) - y0~
             for s in 0..8 {{ outline.push([x0 + dx * s / 8, y0 + dy * s / 8])~ }}
         }}
         let g = symmetry_group(outline)~
         let phd = ph_dimension(line, sizes=[50, 100, 200, 400], repeats=1)~
         let lam = lyapunov([[[2, 0], [0, 0.5]], [[2, 0], [0, 0.5]], [[2, 0], [0, 0.5]]])~"
    );
    let interp = run(&src).unwrap();
    let fold = &interp.variables["fold_c"];
    assert_eq!(text(field(fold, "verdict")), "collision_exhibited");
    // The witness pair straddles the fibre x ~ -x.
    let (p, q) = (num(field(fold, "p")), num(field(fold, "q")));
    assert!(p * q < 0.0 && (p + q).abs() < 0.05, "witness {p}, {q}");
    let affine = &interp.variables["affine_c"];
    assert_eq!(
        text(field(affine, "verdict")),
        "no_collision_at_this_sampling"
    );
    assert_eq!(num(field(affine, "lambda")), 2.0);

    let g = &interp.variables["g"];
    assert_eq!(num(field(g, "order")), 4.0);
    assert!(matches!(field(g, "dihedral"), Value::Bool(true)));

    let d = num(field(&interp.variables["phd"], "dimension"));
    assert!((d - 1.0).abs() < 0.1, "segment PH dimension {d}");

    let lam = nums(&interp.variables["lam"]);
    let ln2 = std::f64::consts::LN_2;
    assert!(
        (lam[0] - ln2).abs() < 1e-15 && (lam[1] + ln2).abs() < 1e-15,
        "{lam:?}"
    );
}

#[test]
fn monodromy_refusal_names_the_variant_and_user_errors_surface() {
    let err = refusal("import monodromy~ symmetry_group([[0, 0]])~");
    assert_eq!(err, "monodromy refused: TooFewPoints { actual: 1, min: 2 }");
    // An error raised inside a program function the core calls back into is
    // reported as that error, not as whatever the core made of a stand-in.
    let err = refusal(&format!(
        "import monodromy~ {SAMPLER} fn bad(x) {{ return \"oops\"~ }}
         collision_certificate(bad, line, sizes=[4, 8])~"
    ));
    assert!(err.contains("expected a list, got string"), "{err}");
}

#[test]
fn track_reports_one_certified_division() {
    let t = var(
        "import track~
         let t = track([[[0, 0, 0]], [[2, 0, 0], [-2, 0, 0]], [[2, 1, 0], [-2, 1, 0]]], 12)~",
        "t",
    );
    assert_eq!(num(field(&t, "nodes")), 5.0);
    assert_eq!(num(field(&t, "divisions")), 1.0);
    assert!(matches!(field(&t, "certified"), Value::Bool(true)));
    let Value::List(dividing) = field(&t, "dividing") else {
        panic!("expected dividing nodes")
    };
    assert_eq!(nums(&dividing[0]), [0.0, 0.0]);
}

#[test]
fn track_refuses_a_zero_gate() {
    let err = refusal("import track~ track([[[0, 0, 0]], [[1, 0, 0]]], 0)~");
    assert_eq!(err, "track refused: InvalidGate");
}

#[test]
fn coupling_rolls_out_to_the_banach_fixed_point() {
    let src = "import coupling~
               let T = [[0.5, 0], [0, 0.5]]~
               let roll = coupling_rollout(T, [1, 1], [0, 0], 3)~
               let fp = coupling_fixed_point(T, [1, 1])~
               let isl = rips_islands([[0, 0], [1, 0], [5, 0]], 1.5)~";
    let interp = run(src).unwrap();
    let Value::List(roll) = &interp.variables["roll"] else {
        panic!("expected states")
    };
    let roll: Vec<Vec<f64>> = roll.iter().map(nums).collect();
    assert_eq!(roll, [[1.0, 1.0], [1.5, 1.5], [1.75, 1.75]]);
    let fp = &interp.variables["fp"];
    assert_eq!(nums(field(fp, "point")), [2.0, 2.0]);
    assert_eq!(num(field(fp, "rho")), 0.5);
    assert!(matches!(field(fp, "contractive"), Value::Bool(true)));
    assert_eq!(num(&interp.variables["isl"]), 2.0);
}

#[test]
fn coupling_refuses_a_singular_fixed_point() {
    let err = refusal("import coupling~ coupling_fixed_point([[1, 0], [0, 1]], [1, 1])~");
    assert!(err.starts_with("coupling refused: NoFixedPoint"), "{err}");
    let err = refusal("import coupling~ coupling_rollout(0.5, 1, 0 / 0, 2)~");
    assert_eq!(err, "coupling refused: NonFinite");
}

#[test]
fn kvwitness_covers_every_segment() {
    let src = "import kvwitness~
               let learned = [13, 14, 15, 12, -1, -1]~
               let merged = witness_topk(learned, 6, 16, 4)~
               let before = segment_coverage(learned, 16, 4)~
               let after = segment_coverage(merged, 16, 4)~
               let recall = attention_mass_recall(merged, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])~";
    let interp = run(src).unwrap();
    assert_eq!(
        nums(&interp.variables["merged"]),
        [13.0, 14.0, 0.0, 4.0, 8.0, -1.0]
    );
    assert_eq!(num(&interp.variables["before"]), 0.25);
    assert_eq!(num(&interp.variables["after"]), 1.0);
    assert_eq!(num(&interp.variables["recall"]), 5.0 / 16.0);
}

#[test]
fn kvwitness_refuses_more_segments_than_the_mask_holds() {
    let err = refusal("import kvwitness~ witness_topk([1, 2], 2, 16, 65)~");
    assert_eq!(
        err,
        "kvwitness refused: SegmentsOutOfRange { num_segments: 65 }"
    );
}

#[test]
fn planner_reuses_the_leading_gap_and_reduces_the_shortcut() {
    let src = "import planner~
               let plan = plan_memory([[100, 0, 1], [80, 1, 2], [60, 2, 3]])~
               let red = transitive_reduction(4, [[0, 1], [1, 2], [2, 3], [0, 3]])~
               let isl = islands(6, [[0, 1], [1, 2], [4, 5]])~";
    let interp = run(src).unwrap();
    let plan = &interp.variables["plan"];
    assert_eq!(nums(field(plan, "offsets")), [0.0, 100.0, 0.0]);
    assert_eq!(num(field(plan, "arena")), 180.0);
    assert_eq!(num(field(plan, "peak")), 180.0);
    let Value::List(red) = &interp.variables["red"] else {
        panic!("expected edges")
    };
    let red: Vec<Vec<f64>> = red.iter().map(nums).collect();
    assert_eq!(red, [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]);
    let isl = &interp.variables["isl"];
    assert_eq!(num(field(isl, "count")), 2.0);
    assert_eq!(nums(field(isl, "labels")), [0.0, 0.0, 0.0, -1.0, 1.0, 1.0]);
}

#[test]
fn planner_refuses_instead_of_panicking() {
    let err = refusal("import planner~ transitive_reduction(4, [[2, 1]])~");
    assert_eq!(
        err,
        "planner refused: NotTopological { edge: 0, u: 2, v: 1, node_count: 4 }"
    );
    let err = refusal("import planner~ plan_memory([[10, 3, 1]])~");
    assert_eq!(
        err,
        "planner refused: ReversedLifetime { tensor: 0, first_use: 3, last_use: 1 }"
    );
}

#[test]
fn seal_loop_terminates_when_a_certified_invariant_stabilises() {
    // Faces go 1, 2, 3, 4 as spokes 1..3 join the square and one spoke; the
    // pass after the last spoke leaves 4 unchanged and the loop seals.
    let src = format!(
        "import arrangement~ {SQUARE_2D}
         square.push(spokes[0])~
         let k = 1~
         seal until stable(euler(square).faces) {{
             if k < 4 {{ square.push(spokes[k])~ k = k + 1~ }}
         }}
         let faces = euler(square).faces~"
    );
    let interp = run(&src).unwrap();
    assert_eq!(num(&interp.variables["k"]), 4.0);
    assert_eq!(num(&interp.variables["faces"]), 4.0);
}

#[test]
fn imports_bind_single_symbols_and_module_methods() {
    let src = format!(
        "from linking import writhe~ import coupling~ {SQUARE}
         let w = writhe(square)~
         let fp = coupling.coupling_fixed_point(0.5, 1)~
         let z = fp.point[0]~"
    );
    let interp = run(&src).unwrap();
    assert_eq!(num(&interp.variables["w"]), 0.0);
    assert_eq!(num(&interp.variables["z"]), 2.0);
    assert!(!interp.variables.contains_key("linking_number"));
    let err = refusal("from linking import nope~");
    assert_eq!(err, "Symbol 'nope' not found in linking");
}

#[test]
fn manifold_slices_still_parse_beside_element_indexing() {
    let src = "let data = [1.0, 2.0, 3.0, 4.0, 5.0]~
               manifold M = embed(data, tau=1)~
               block B = M[0:2]~
               let xs = [10, 20, 30]~
               let second = xs[1]~";
    let interp = run(src).unwrap();
    assert!(matches!(interp.variables["B"], Value::Block(_)));
    assert_eq!(num(&interp.variables["second"]), 20.0);
    let err = refusal("let xs = [1]~ xs[3]~");
    assert_eq!(err, "index 3 out of range for a list of 1");
}
