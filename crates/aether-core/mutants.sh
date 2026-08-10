#!/usr/bin/env bash
#
# Mutation coverage for the schedule-selection code in `src/scheduled.rs`.
#
# Why this exists separately from `crates/aether-gpu/mutants.sh`: that harness
# mutates WGSL and refuses to run without a real adapter, because without one
# every kernel test is ignored and every mutant would appear to survive. Nothing
# here touches the GPU. Copying the adapter guard across would be importing a
# precondition that does not apply, and a guard that can never fire is worse than
# no guard — it reads as protection while providing none.
#
# What it protects. `block_mass_recovered`, `random_block_schedule` and
# `oracle_block_schedule` produce the numbers in `aether-gpu/FEATURES.md` that
# say the topological selector recovers less attention mass than random
# selection, and that the schedules are indistinguishable at equal budget. Those
# are the strongest claims in this repository and they are computed, not
# measured against an external reference. If the eight tests in
# `tests/ablation_baselines.rs` lack power, the claims rest on nothing.
#
# The tests passed on first execution, so no red phase established that power.
# This is the substitute, and it is the same substitute the GPU side used when
# it found a mutant that survived every test in a file that had looked complete.

set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
target="$repo/crates/aether-core/src/scheduled.rs"
cd "$repo" || exit 1

if [ ! -f "$target" ]; then
    echo "source not found: $target" >&2
    exit 1
fi

restore() {
    git checkout -- "$target" 2>/dev/null
    touch "$target"
}

# Restoring means `git checkout`, which discards uncommitted work. The GPU
# harness learned this by deleting a kernel that had been written but not
# committed, and reporting it as patterns failing to match.
if ! git diff --quiet -- "$target" || ! git diff --cached --quiet -- "$target"; then
    echo "$target has uncommitted changes." >&2
    echo "This harness restores by 'git checkout', which would discard them." >&2
    echo "Commit or stash it first." >&2
    exit 101
fi

trap restore EXIT

# A baseline run must pass, or every mutant is "caught" by a failure that was
# already there and the whole table means nothing.
if ! cargo test -p aether-core --test ablation_baselines >/dev/null 2>&1; then
    echo "the clean tree already fails ablation_baselines." >&2
    echo "Every mutant would be reported as caught by a pre-existing failure." >&2
    exit 100
fi

# name | perl expression applied to the whole file
mutants=(
"block_mass_table: rows-per-block normalisation dropped|s/\*entry \/= block_size as f64;/*entry \/= 1.0;/"
"block_mass_table: max subtraction dropped in the denominator|s/exp\(l - max_logit\)/exp(l)/"
"block_mass_recovered: averages over rows not blocks|s/Ok\(total \/ num_blocks as f64\)/Ok(total \/ seq as f64)/"
"oracle_block_schedule: ranks ascending|s/mb\.partial_cmp\(&ma\)/ma.partial_cmp(\&mb)/"
"random_block_schedule: shuffle is a no-op|s/let j = i \+ \(next\(\) as usize\) % \(candidates - i\);/let j = i;/"
"inverted selector: admits the zero-salience block|s/\.filter\(\|&b\| salience\[b\] != 0\.0\)//"
"schedule_budget: reports one block too many|s/\.map\(\|q_block\| schedule\.row\(q_block\)\.len\(\)\)/.map(|q_block| schedule.row(q_block).len() + 1)/"
)

printf '%-58s %-10s\n' "MUTANT" "ablation"
printf '%-58s %-10s\n' "----------------------------------------------------------" "----------"

escaped=0

for entry in "${mutants[@]}"; do
    name="${entry%%|*}"
    expr="${entry#*|}"

    restore
    before="$(md5sum < "$target")"
    perl -0777 -pi -e "$expr" "$target"
    after="$(md5sum < "$target")"

    # A pattern that matches nothing tests nothing. Counting it as an escape is
    # deliberate: the alternative reports a stale mutant as coverage.
    if [ "$before" = "$after" ]; then
        printf '%-58s %-10s  <- pattern did not match\n' "$name" "SKIPPED"
        escaped=$((escaped + 1))
        continue
    fi

    touch "$target"

    # A mutant that fails to compile is caught: the change is not a silent
    # behaviour difference, which is the class this harness is looking for.
    if cargo test -p aether-core --test ablation_baselines >/dev/null 2>&1; then
        escaped=$((escaped + 1))
        printf '%-58s %-10s  <- ESCAPED\n' "$name" "survives"
    else
        printf '%-58s %-10s\n' "$name" "CAUGHT"
    fi
done

restore

echo
echo "verifying the restored tree still passes"
if clean="$(cargo test -p aether-core --test ablation_baselines 2>&1)"; then
    echo "  clean tree: pass"
else
    echo "  clean tree: FAIL" >&2
    if ! git diff --quiet -- "$target"; then
        echo "  the source differs from HEAD, so restore did not complete" >&2
    else
        echo "  the source matches HEAD, so the failure is elsewhere" >&2
    fi
    printf '%s\n' "$clean" >&2
    exit 99
fi

echo
echo "mutants escaping: $escaped / ${#mutants[@]}"
[ "$escaped" -eq 0 ]
