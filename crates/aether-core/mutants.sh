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
sources="$repo/crates/aether-core/src"
cd "$repo" || exit 1

if [ ! -d "$sources" ]; then
    echo "source tree not found: $sources" >&2
    exit 1
fi

# Every mutant names its own file, so restore covers the whole crate source
# rather than one path. Mutants used to target `scheduled.rs` alone; the
# persistence engine and the diagram metrics live elsewhere and had no coverage
# measurement at all, while the README reported a count for them that no command
# in the tree could reproduce.
restore() {
    git checkout -- "$sources" 2>/dev/null
    find "$sources" -name '*.rs' -exec touch {} +
}

# Restoring means `git checkout`, which discards uncommitted work. The GPU
# harness learned this by deleting a kernel that had been written but not
# committed, and reporting it as patterns failing to match.
if ! git diff --quiet -- "$sources" || ! git diff --cached --quiet -- "$sources"; then
    echo "$sources has uncommitted changes." >&2
    echo "This harness restores by 'git checkout', which would discard them." >&2
    echo "Commit or stash them first." >&2
    exit 101
fi

trap restore EXIT

# A baseline run must pass, or every mutant is "caught" by a failure that was
# already there and the whole table means nothing.
if ! cargo test -p aether-core --test ablation_baselines --test scheduled_attention --test attention_backward --test persistence_invariants --test diagram_distance >/dev/null 2>&1; then
    echo "the clean tree already fails ablation_baselines." >&2
    echo "Every mutant would be reported as caught by a pre-existing failure." >&2
    exit 100
fi

# name | file relative to the repo root | perl expression applied to that file
mutants=(
"block_mass_table: rows-per-block normalisation dropped|crates/aether-core/src/scheduled.rs|s/\*entry \/= block_size as f64;/*entry \/= 1.0;/"
"block_mass_table: max subtraction dropped in the denominator|crates/aether-core/src/scheduled.rs|s/exp\(l - max_logit\)/exp(l)/"
"block_mass_recovered: averages over rows not blocks|crates/aether-core/src/scheduled.rs|s/Ok\(total \/ num_blocks as f64\)/Ok(total \/ seq as f64)/"
"oracle_block_schedule: ranks ascending|crates/aether-core/src/scheduled.rs|s/mb\.partial_cmp\(&ma\)/ma.partial_cmp(\&mb)/"
"random_block_schedule: shuffle is a no-op|crates/aether-core/src/scheduled.rs|s/let j = i \+ \(next\(\) as usize\) % \(candidates - i\);/let j = i;/"
"inverted selector: admits the zero-salience block|crates/aether-core/src/scheduled.rs|s/\.filter\(\|&b\| salience\[b\] != 0\.0\)//"
"schedule_budget: reports one block too many|crates/aether-core/src/scheduled.rs|s/\.map\(\|q_block\| schedule\.row\(q_block\)\.len\(\)\)/.map(|q_block| schedule.row(q_block).len() + 1)/"
# ── block_salience and topology_block_schedule ──────────────────────────────
#
# The mechanism itself, and the reason it is covered here rather than trusted.
# This repository concluded that the salience ranking is anti-correlated with
# attention mass because it scores blocks by isolation. That explanation assumes
# `block_salience` computes the H0 death times it claims to; if it does not, the
# anti-correlation is an artefact and the explanation is a story told about a
# bug. The tests asserting it were never measured either.
"block_salience: elder rule inverted|crates/aether-core/src/scheduled.rs|s/if members\[root_left\]\.len\(\) > members\[root_right\]\.len\(\) \{/if members[root_left].len() < members[root_right].len() {/"
"block_salience: death recorded for the surviving component|crates/aether-core/src/scheduled.rs|s/for &block in &members\[root_left\] \{/for \&block in \&members[root_right] {/"
"block_salience: merges in decreasing distance|crates/aether-core/src/scheduled.rs|s/a\.0\.total_cmp\(&b\.0\)/b.0.total_cmp(\&a.0)/"
"block_salience: centroids summed not averaged|crates/aether-core/src/scheduled.rs|s/centroids\[block \* dim \+ d\] \/= block_size as f64;/centroids[block * dim + d] \/= 1.0;/"
"block_salience: squared distance left unrooted|crates/aether-core/src/scheduled.rs|s/edges\.push\(\(sqrt\(sum\), i, j\)\);/edges.push((sum, i, j));/"
"topology_block_schedule: local window narrowed to the diagonal|crates/aether-core/src/scheduled.rs|s/for block in q_block\.saturating_sub\(config\.local_radius_blocks\)\.\.=q_block \{/for block in q_block..=q_block {/"
"topology_block_schedule: sink blocks dropped|crates/aether-core/src/scheduled.rs|s/for block in 0\.\.config\.sink_blocks\.min\(num_blocks\)\.min\(q_block \+ 1\) \{/for block in 0..0usize {/"
# ── scheduled_attention_backward ────────────────────────────────────────────
#
# A wrong backward pass is the least visible defect here: the forward stays
# correct, the loss still falls, and the model converges to a plausible worse
# optimum with no crash and no NaN to notice. These target the terms that fail
# that way rather than loudly.
"backward: softmax rank-one correction dropped|crates/aether-core/src/scheduled.rs|s/let ds = p \* \(dp - delta\) \* scale;/let ds = p * dp * scale;/"
"backward: delta unweighted by the attention weights|crates/aether-core/src/scheduled.rs|s/delta \+= weights\[idx\] \* dp;/delta += dp;/"
"backward: dq accumulates q instead of k|crates/aether-core/src/scheduled.rs|s/dq\[row \* head_dim \+ d\] \+= ds \* k\[col \* head_dim \+ d\];/dq[row * head_dim + d] += ds * q[col * head_dim + d];/"
"backward: dv accumulates the cotangent unweighted|crates/aether-core/src/scheduled.rs|s/dv\[col \* head_dim \+ d\] \+= p \* d_out\[row \* head_dim \+ d\];/dv[col * head_dim + d] += d_out[row * head_dim + d];/"
"backward: scale factor omitted|crates/aether-core/src/scheduled.rs|s/let ds = p \* \(dp - delta\) \* scale;/let ds = p * (dp - delta);/"
# ── the persistence engine ──────────────────────────────────────────────────
#
# These three are the defects the README reports under "Persistence invariants",
# with per-mutant catch counts of 4/11, 4/11 and 7/11. That table was a record of
# a run and not a command: nothing in the tree reproduced it, so a reader could
# not check it and a later change could not invalidate it. The document promises
# that every number carries a command, and this is the command.
"persistence: triangle filtration drops one of three edges|crates/aether-core/src/persistence.rs|s/distances\[j \* n \+ k\],/distances[i * n + j],/"
"persistence: absolute epsilon added to the filtration radius|crates/aether-core/src/persistence.rs|s/if r <= config\.max_radius \{/if r <= config.max_radius + 0.001 {/"
"persistence: column reduction stops after one operation|crates/aether-core/src/persistence.rs|s/            column = xor_sorted\(&column, owner_column\);/            column = xor_sorted(\&column, owner_column);\n            break;/"
# ── the diagram metrics ─────────────────────────────────────────────────────
#
# The four runnable defects the README reports under "Diagram metrics", with
# claimed counts of 2/17, 1/17, 1/17 and 1/17. Three of those are a single test
# away from zero coverage, which is the state the filtration epsilon was already
# found to have reached: caught by four, then by none, without any assertion
# being touched.
#
# The fifth defect in that table, bottleneck forbidding diagonal projection, is
# recorded there as never having run because infinite costs diverge the matching
# search. It is left out rather than quietly dropped -- adding a mutant that
# hangs the harness would trade one unverified claim for an unusable one.
"landscape: per-sample descending sort skipped|crates/aether-core/src/diagram.rs|s/    tents\.sort_by\(\|a, b\| b\.total_cmp\(a\)\);//"
"image: linear persistence weight dropped|crates/aether-core/src/diagram.rs|s/let weight = persistence;/let weight = 1.0;/"
"image: gaussian width hardcoded, sigma ignored|crates/aether-core/src/diagram.rs|s/let two_sigma_sq = 2\.0 \* config\.sigma \* config\.sigma;/let two_sigma_sq = 2.0 * 0.1 * 0.1;/"
"wasserstein: returns the max instead of the sum|crates/aether-core/src/diagram.rs|s/\(1\.\.=n\)\.map\(\|j\| cost\[assignment\[j\] - 1\]\[j - 1\]\)\.sum\(\)/(1..=n).map(|j| cost[assignment[j] - 1][j - 1]).fold(0.0, f64::max)/"
)

# Suites run against each mutant. A mutant escapes only if it survives every one.
suites=(ablation_baselines scheduled_attention attention_backward persistence_invariants diagram_distance)

printf '%-58s' "MUTANT"
for suite in "${suites[@]}"; do printf ' %-12s' "$suite"; done
printf '\n'
printf '%-58s' "----------------------------------------------------------"
for _ in "${suites[@]}"; do printf ' %-12s' "------------"; done
printf '\n'

escaped=0

for entry in "${mutants[@]}"; do
    name="${entry%%|*}"
    rest="${entry#*|}"
    target="$repo/${rest%%|*}"
    expr="${rest#*|}"

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
    #
    # Reported as failing/total rather than as a verdict. "CAUGHT" answers
    # whether a defect is detected; the fraction answers how much of the suite
    # notices, which is what the README's per-mutant counts claim and what a bare
    # pass/fail cannot check. A defect caught by one test of twelve is one
    # rewrite away from being caught by none, and reads identically to a defect
    # caught by all twelve until the fraction is printed.
    any_caught=0
    printf '%-58s' "$name"
    for suite in "${suites[@]}"; do
        out="$(cargo test -p aether-core --test "$suite" 2>&1)"
        if printf '%s' "$out" | grep -q "test result: ok"; then
            printf ' %-12s' "survives"
        else
            # `N passed; M failed` is absent when the mutant does not compile,
            # which is a real distinction: a build failure is caught by the type
            # system rather than by any assertion.
            counts="$(printf '%s' "$out" | grep -oE '[0-9]+ passed; [0-9]+ failed' | head -1)"
            if [ -z "$counts" ]; then
                printf ' %-12s' "build"
            else
                passed="${counts%% passed;*}"
                failed="${counts##*; }"
                failed="${failed%% failed}"
                printf ' %-12s' "$failed/$((passed + failed))"
            fi
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
if clean="$(cargo test -p aether-core --test ablation_baselines --test scheduled_attention --test attention_backward --test persistence_invariants --test diagram_distance 2>&1)"; then
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


