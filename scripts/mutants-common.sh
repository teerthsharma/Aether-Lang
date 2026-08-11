#!/usr/bin/env bash
#
# Shared outcome classification for the mutation harnesses.
#
# This exists because the same defect was fixed twice, badly. Both harnesses
# treated any non-zero cargo exit as "mutant caught", which counts an invocation
# that never ran as evidence about the code: a broken command marks every mutant
# caught and prints "0 escaping" having measured nothing, in the same words a
# genuine clean sweep uses.
#
# The fix landed in `crates/aether-core/mutants.sh` and the GPU harness kept the
# flaw for another change, because the logic was written out twice and only one
# copy was in front of anyone. Writing it a third time would set that up again.
#
# Sourced by both harnesses. Callers set `MUTANT_CATCHERS` to a file path before
# the first call.

# Run one suite against the current tree and classify what came back.
#
# Prints a fixed-width cell and returns 0 if the mutant survived, 1 if it was
# caught. Exits 97 without returning when cargo produced neither a test result
# nor a compile error, which is a fault in the calling script rather than
# evidence about the mutant.
#
# Arguments: package, suite, mutant name, cell width, extra cargo flags.
mutant_run_suite() {
    local pkg="$1" suite="$2" mutant="$3" width="$4" extra="${5:-}"
    local out counts passed failed attempt

    # A crashed binary is retried once before being scored.
    #
    # The crash this handles is a documented intermittent in wgpu's instance
    # teardown, and what identifies it as environmental is that consecutive runs
    # put it on different mutants in different suites. A fault that moves every
    # run is one a single repeat almost always clears, so scoring the first
    # crash costs a cell that the retry would have measured — and when that cell
    # is the only suite catching a mutant, the whole run reports an escape that
    # is not there.
    #
    # Only crashes retry. A test that fails is evidence and re-running it to see
    # whether it fails again would be sampling until the answer is convenient.
    for attempt in 1 2; do
        # shellcheck disable=SC2086 -- extra is a flag list and must word-split.
        out="$(cargo test -p "$pkg" $extra --test "$suite" 2>&1)"
        if [ "$attempt" -eq 1 ] && printf '%s' "$out" \
            | grep -qE 'STATUS_ACCESS_VIOLATION|test exited abnormally|didn.t exit successfully'; then
            continue
        fi
        break
    done

    if printf '%s' "$out" | grep -q "test result: ok"; then
        printf " %-${width}s" "survives"
        return 0
    fi

    # A count of failing tests is more than a verdict: a defect caught by one
    # test of seventeen is one rewrite from being caught by none, and reads
    # identically to one caught by all seventeen until the fraction is printed.
    counts="$(printf '%s' "$out" | grep -oE '[0-9]+ passed; [0-9]+ failed' | head -1)"
    if [ -n "$counts" ]; then
        passed="${counts%% passed;*}"
        failed="${counts##*; }"
        failed="${failed%% failed}"
        printf " %-${width}s" "$failed/$((passed + failed))"
        MUTANT_FAILURES=$((${MUTANT_FAILURES:-0} + failed))

        # Which tests, not only how many. Three defects each at 1/17 could be
        # three independent tests or one carrying all three, and only the second
        # means a single fixture is load-bearing for three defects at once.
        if [ -n "${MUTANT_CATCHERS:-}" ]; then
            printf '%s' "$out" \
                | grep -oE '^test [A-Za-z0-9_:]+ \.\.\. FAILED' \
                | sed -E 's/^test (.*) \.\.\. FAILED/'"$mutant"'\t\1/' \
                >> "$MUTANT_CATCHERS"
        fi
        return 1
    fi

    # Specifically a rustc diagnostic, not any line beginning with "error". A
    # mistyped flag makes cargo print `error: unexpected argument`, which is a
    # fault in the harness and not a mutant the type system rejected; matching it
    # would reinstate the conflation this function exists to remove. That was not
    # anticipated — the first version matched `^error` and classified a bad flag
    # as a build failure, silently.
    if printf '%s' "$out" | grep -qE 'error\[E[0-9]+\]|error: could not compile'; then
        printf " %-${width}s" "build"
        return 1
    fi

    # A test binary that died twice rather than reporting — the retry above has
    # already been spent. This is a third thing again, and it is deliberately
    # *not* counted as catching the mutant.
    #
    # A crash has two possible causes and the output cannot separate them: the
    # mutant broke something badly enough to take the process down, or something
    # unrelated did. This crate has a documented intermittent of the second kind
    # — a STATUS_ACCESS_VIOLATION in wgpu's instance teardown, worked around by
    # pinning a single backend and never fully eliminated. Scoring a crash as a
    # catch would credit coverage to whichever of those it was.
    #
    # Returning "survives" errs toward reporting less coverage than exists, which
    # is the safe direction for a number whose whole purpose is to say how much
    # the suite would notice.
    if printf '%s' "$out" | grep -qE 'STATUS_ACCESS_VIOLATION|test exited abnormally|didn.t exit successfully'; then
        printf " %-${width}s" "crash"
        return 0
    fi

    printf '\n'
    echo "cargo produced neither a test result nor a compile error for" >&2
    echo "  mutant: $mutant" >&2
    echo "  suite:  $suite" >&2
    echo "Counting this as caught would mark every mutant caught for the" >&2
    echo "same reason and report a clean sweep of nothing." >&2
    printf '%s\n' "$out" >&2
    exit 97
}

# Report which tests caught what, ordered by how many defects each carries.
#
# Takes the total number of failures the fractions accounted for. The two are
# parsed from the same cargo output by different means — the counts from its
# "N passed; M failed" line, the names from its per-test FAILED lines — so
# requiring them to agree makes each a check on the other. Without it the name
# parser could stop matching and the run would print an empty summary beneath a
# table of entirely correct counts.
mutant_report_catchers() {
    local expected="$1" found
    [ -n "${MUTANT_CATCHERS:-}" ] || return 0

    found="$(wc -l < "$MUTANT_CATCHERS" | tr -d ' ')"
    if [ "$found" -ne "$expected" ]; then
        echo >&2
        echo "the failing-test parser recorded $found names for $expected failures." >&2
        echo "Both are parsed from the same cargo output, so disagreement means" >&2
        echo "one of them has stopped matching. The table above is still correct," >&2
        echo "which is what would have made this quiet." >&2
        exit 98
    fi

    echo
    echo "tests that caught each mutant, by how many defects they carry"
    if [ -s "$MUTANT_CATCHERS" ]; then
        cut -f2 "$MUTANT_CATCHERS" | sort | uniq -c | sort -rn | while read -r n test; do
            printf '  %2d  %s\n' "$n" "$test"
        done
    fi
}
