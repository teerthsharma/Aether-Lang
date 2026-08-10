#!/usr/bin/env bash
#
# Mutation testing for the aether-gpu WGSL kernels.
#
#   ./crates/aether-gpu/mutants.sh
#
# Injects one plausible defect per kernel, one at a time, and reports which
# suite catches it. A defect no suite catches is a coverage hole: the tests
# would report success on a kernel that computes the wrong thing.
#
# Two mechanics are load-bearing and easy to get wrong.
#
# `touch` after patching: shaders.wgsl reaches the binary through include_str!,
# and restoring it from a backup file carries that backup's older modification
# time, so Cargo sees no reason to rebuild and the test binary keeps running the
# previous mutant. That produced an incoherent result once -- a mutant reported
# as surviving, then a test failing after it was reverted -- and `git diff`
# showed a clean tree throughout, because the source really was clean and only
# the artifact was stale.
#
# Each suite runs separately: gpu_parity and gradcheck catch disjoint defects,
# and a combined pass/fail hides which one did the work.
#
# Exit status is the number of defects that escaped every suite, so this is
# usable as a gate.

set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
shader="$repo/crates/aether-gpu/src/shaders.wgsl"
cd "$repo" || exit 1

if [ ! -f "$shader" ]; then
    echo "shader not found: $shader" >&2
    exit 1
fi

restore() {
    git checkout -- "$shader" 2>/dev/null
    touch "$shader"
}

# Restoring means `git checkout`, which discards uncommitted work in the shader.
#
# This is not hypothetical. Running the harness while a new kernel was written
# but not yet committed deleted it: every mutation pattern then failed to match
# because the code it targeted was gone, and the final clean-tree check failed
# because the Rust side still referenced pipelines the shader no longer defined.
# Nothing in the output said "your work was deleted"; it said the patterns did
# not match, which reads as a harness bug rather than as data loss.
if ! git diff --quiet -- "$shader" || ! git diff --cached --quiet -- "$shader"; then
    echo "$shader has uncommitted changes." >&2
    echo "This harness restores by 'git checkout', which would discard them." >&2
    echo "Commit or stash the shader first." >&2
    exit 101
fi

trap restore EXIT

# name | perl expression applied to the whole file
mutants=(
"transpose: output index swapped|s/c\[col \* dims\.m \+ row\] = a\[row \* dims\.n \+ col\];/c[row * dims.m + col] = a[row * dims.n + col];/"
"column_sums: skips the first row|s/for \(var r: u32 = 0u; r < dims\.m/for (var r: u32 = 1u; r < dims.m/"
"matmul: reads A transposed|s/a\[row \* dims\.k \+ i\] \* b\[i \* dims\.n \+ col\]/a[i * dims.k + row] * b[i * dims.n + col]/"
"matmul_tiled: second barrier removed|s/(sum = sum \+ tile_a\[lid\.x\]\[i\] \* tile_b\[i\]\[lid\.y\];\s*\n\s*\}\s*\n)\s*workgroupBarrier\(\);/\$1/"
"pairwise_sqdist: distance not squared|s/sum = sum \+ delta \* delta;/sum = sum + delta;/"
"sgd_update: ascends instead of descending|s/c\[idx\] = a\[idx\] - bitcast<f32>/c[idx] = a[idx] + bitcast<f32>/"
"softmax_rows: max subtraction removed|s/let e = exp\(a\[base \+ j\] - mx\);/let e = exp(a[base + j]);/"
"relu_backward: gates on the gradient not the pre-activation|s/if \(a\[idx\] > 0\.0\) \{/if (b[idx] > 0.0) {/"
"adam_update: bias correction dropped|s/let mhat = b\[idx\] \/ \(1\.0 - pow\(ADAM_B1, t\)\);/let mhat = b[idx];/"
"adam_update: epsilon inside the square root|s/sqrt\(vhat\) \+ ADAM_EPS/sqrt(vhat + ADAM_EPS)/"
)

# Without a GPU every test in both suites skips and returns success, so every
# mutant would be reported as surviving and the run would announce a total
# coverage failure that is really an absent adapter. Refuse instead.
#
# This is the same class of mistake the suites themselves guard against: a
# result that looks like a measurement but describes the environment.
probe="$(cargo test -p aether-gpu --test gpu_parity -- --nocapture \
    the_selected_adapter_is_real_hardware_not_a_software_rasterizer 2>&1)"

if printf '%s' "$probe" | grep -q "SKIP: no usable GPU adapter"; then
    echo "no GPU adapter available." >&2
    echo "Every kernel test would skip, so every mutant would appear to survive." >&2
    echo "Refusing to report a coverage result that would describe the machine." >&2
    exit 100
fi

if printf '%s' "$probe" | grep -q "device_type=Cpu"; then
    echo "adapter is a software rasterizer." >&2
    echo "Mutation results would not describe GPU execution. Refusing." >&2
    exit 100
fi

printf '%-46s %-10s %-10s\n' "MUTANT" "gpu_parity" "gradcheck"
printf '%-46s %-10s %-10s\n' "----------------------------------------------" "----------" "----------"

escaped=0

for entry in "${mutants[@]}"; do
    name="${entry%%|*}"
    expr="${entry#*|}"

    restore
    before="$(md5sum < "$shader")"
    perl -0777 -pi -e "$expr" "$shader"
    after="$(md5sum < "$shader")"

    if [ "$before" = "$after" ]; then
        printf '%-46s %-10s %-10s  <- pattern did not match\n' "$name" "SKIPPED" "SKIPPED"
        escaped=$((escaped + 1))
        continue
    fi

    touch "$shader"

    if cargo test -p aether-gpu --test gpu_parity >/dev/null 2>&1; then
        parity="survives"
    else
        parity="CAUGHT"
    fi

    if cargo test -p aether-gpu --test gradcheck >/dev/null 2>&1; then
        grad="survives"
    else
        grad="CAUGHT"
    fi

    if [ "$parity" = "survives" ] && [ "$grad" = "survives" ]; then
        escaped=$((escaped + 1))
        printf '%-46s %-10s %-10s  <- ESCAPED EVERY SUITE\n' "$name" "$parity" "$grad"
    else
        printf '%-46s %-10s %-10s\n' "$name" "$parity" "$grad"
    fi
done

restore

echo
echo "verifying the restored tree still passes"
if cargo test -p aether-gpu >/dev/null 2>&1; then
    echo "  clean tree: pass"
else
    echo "  clean tree: FAIL -- the shader was not restored correctly" >&2
    exit 99
fi

echo
echo "mutants escaping every suite: $escaped / ${#mutants[@]}"
exit "$escaped"
