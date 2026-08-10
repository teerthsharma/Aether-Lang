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
if ! cargo test -p aether-core --test ablation_baselines --test scheduled_attention --test attention_backward >/dev/null 2>&1; then
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
# ── block_salience and topology_block_schedule ──────────────────────────────
#
# The mechanism itself, and the reason it is covered here rather than trusted.
# This repository concluded that the salience ranking is anti-correlated with
# attention mass because it scores blocks by isolation. That explanation assumes
# `block_salience` computes the H0 death times it claims to; if it does not, the
# anti-correlation is an artefact and the explanation is a story told about a
# bug. The tests asserting it were never measured either.
"block_salience: elder rule inverted|s/if members\[root_left\]\.len\(\) > members\[root_right\]\.len\(\) \{/if members[root_left].len() < members[root_right].len() {/"
"block_salience: death recorded for the surviving component|s/for &block in &members\[root_left\] \{/for \&block in \&members[root_right] {/"
"block_salience: merges in decreasing distance|s/a\.0\.total_cmp\(&b\.0\)/b.0.total_cmp(\&a.0)/"
"block_salience: centroids summed not averaged|s/centroids\[block \* dim \+ d\] \/= block_size as f64;/centroids[block * dim + d] \/= 1.0;/"
"block_salience: squared distance left unrooted|s/edges\.push\(\(sqrt\(sum\), i, j\)\);/edges.push((sum, i, j));/"
"topology_block_schedule: local window narrowed to the diagonal|s/for block in q_block\.saturating_sub\(config\.local_radius_blocks\)\.\.=q_block \{/for block in q_block..=q_block {/"
"topology_block_schedule: sink blocks dropped|s/for block in 0\.\.config\.sink_blocks\.min\(num_blocks\)\.min\(q_block \+ 1\) \{/for block in 0..0usize {/"
# ── scheduled_attention_backward ────────────────────────────────────────────
#
# A wrong backward pass is the least visible defect here: the forward stays
# correct, the loss still falls, and the model converges to a plausible worse
# optimum with no crash and no NaN to notice. These target the terms that fail
# that way rather than loudly.
"backward: softmax rank-one correction dropped|s/let ds = p \* \(dp - delta\) \* scale;/let ds = p * dp * scale;/"
"backward: delta unweighted by the attention weights|s/delta \+= weights\[idx\] \* dp;/delta += dp;/"
"backward: dq accumulates q instead of k|s/dq\[row \* head_dim \+ d\] \+= ds \* k\[col \* head_dim \+ d\];/dq[row * head_dim + d] += ds * q[col * head_dim + d];/"
"backward: dv accumulates the cotangent unweighted|s/dv\[col \* head_dim \+ d\] \+= p \* d_out\[row \* head_dim \+ d\];/dv[col * head_dim + d] += d_out[row * head_dim + d];/"
"backward: scale factor omitted|s/let ds = p \* \(dp - delta\) \* scale;/let ds = p * (dp - delta);/"
)

# Suites run against each mutant. A mutant escapes only if it survives every one.
suites=(ablation_baselines scheduled_attention attention_backward)

printf '%-58s' "MUTANT"
for suite in "${suites[@]}"; do printf ' %-12s' "$suite"; done
printf '\n'
printf '%-58s' "----------------------------------------------------------"
for _ in "${suites[@]}"; do printf ' %-12s' "------------"; done
printf '\n'

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
        printf '%-58s' "$name"
        for _ in "${suites[@]}"; do printf ' %-12s' "SKIPPED"; done
        printf '  <- pattern did not match\n'
        escaped=$((escaped + 1))
        continue
    fi

    touch "$target"

    # A mutant that fails to compile is caught: the change is not a silent
    # behaviour difference, which is the class this harness is looking for.
    any_caught=0
    printf '%-58s' "$name"
    for suite in "${suites[@]}"; do
        if cargo test -p aether-core --test "$suite" >/dev/null 2>&1; then
            printf ' %-12s' "survives"
        else
            printf ' %-12s' "CAUGHT"
            any_caught=1
        fi
    done

    if [ "$any_caught" -eq 0 ]; then
        escaped=$((escaped + 1))
        printf '  <- ESCAPED EVERY SUITE\n'
    else
        printf '\n'
    fi
done

restore

echo
echo "verifying the restored tree still passes"
if clean="$(cargo test -p aether-core --test ablation_baselines --test scheduled_attention --test attention_backward 2>&1)"; then
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

