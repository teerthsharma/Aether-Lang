"""Wall-clock timing of `aether run` across binaries, engines and programs.

    python bench/engines/time_engines.py BASELINE_EXE BRANCH_EXE [RUNS]

BASELINE_EXE is master before the reshape (interpreter cloning its variable map
per call, the old TitanVM); BRANCH_EXE is the reshape (interpreter with one frame
per call over read-only globals, the rewritten TitanVM). Both release builds.

Each (binary, mode, program) is run once untimed, then RUNS times (default 9)
round-robin, so drift over the session spreads across every combination. The
time is the whole process, startup included; empty.aegis measures startup
alone. Every run's stdout must hold the program's known result, or the harness
stops. Run from the repository root.
"""
import statistics
import subprocess
import sys
import time

# program -> result it must print; None for empty.aegis
EXPECTED = {"empty": None, "fib": 75025, "loop": 18, "fib_scoped": 75025, "loop_scoped": 18}


def combos(baseline, noclone):
    for p in ("empty", "fib", "loop", "fib_scoped", "loop_scoped"):
        yield ("baseline", baseline, "bio", p)
    for p in ("empty", "fib", "loop"):
        yield ("baseline", baseline, "titan", p)
    for p in ("empty", "fib", "loop", "fib_scoped", "loop_scoped"):
        yield ("noclone", noclone, "bio", p)
    for p in ("empty", "fib", "loop"):
        yield ("noclone", noclone, "titan", p)


def run(exe, mode, program):
    cmd = [exe, "run", f"bench/engines/{program}.aegis", "--mode", mode]
    t0 = time.perf_counter()
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    dt = time.perf_counter() - t0
    want = EXPECTED[program]
    if out.returncode != 0 or (
        want is not None and f"\n{want}\n" not in out.stdout and f"Num({want}.0)" not in out.stdout
    ):
        sys.exit(f"{cmd}: exit {out.returncode}, expected {want}\n{out.stdout}{out.stderr}")
    return dt * 1000


def main():
    baseline, noclone = sys.argv[1], sys.argv[2]
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    cs = list(combos(baseline, noclone))
    for c in cs:
        run(*c[1:])  # warm-up, untimed
    times = {c: [] for c in cs}
    for _ in range(runs):
        for c in cs:
            times[c].append(run(*c[1:]))
    print(f"{'binary':9} {'mode':6} {'program':12} {'median ms':>10} {'min ms':>9} {'max ms':>9}")
    for (name, _, mode, program), ts in times.items():
        print(f"{name:9} {mode:6} {program:12} {statistics.median(ts):10.1f} {min(ts):9.1f} {max(ts):9.1f}")
    med = {(n, m, p): statistics.median(ts) for (n, _, m, p), ts in times.items()}
    for p in ("fib", "loop"):
        r = med[("noclone", "bio", p)] / med[("noclone", "titan", p)]
        print(f"gate {p}: branch interpreter / branch titan = {r:.2f}x (Titan survives at >= 3x on both)")


if __name__ == "__main__":
    main()
