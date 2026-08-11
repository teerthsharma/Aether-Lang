#!/usr/bin/env bash
#
# Mechanically generated mutation testing for the aether-gpu WGSL kernels.
#
#   ./crates/aether-gpu/mutants-mechanical.sh
#
# `mutants.sh` injects defects someone chose. That bounds the suite from below
# and says nothing about defects nobody thought of: a curated mutant set and the
# tests it scores were written by the same person with the same idea of what
# breaks, so a blind spot shared between them is invisible to both.
#
# This enumerates every comparison operator in the shader and flips each one in
# turn. Nobody chose the sites, so a survivor here is a defect class the curated
# set does not reach.
#
# # Boundary flips specifically
#
# Comparisons are chosen over arithmetic because the flip is meaning-preserving
# enough to compile and small enough to be plausible: `<` to `<=` is an
# off-by-one at a loop bound or a guard, which is the defect that survives review
# and produces a wrong answer rather than a crash.
#
# # Reading the result
#
# A survivor is NOT automatically a coverage hole. Some flips are *equivalent
# mutants* -- they change the text without changing what the program computes,
# and no test can catch them because there is nothing to catch. A guard that is
# unreachable, a bound already excluded by an earlier clamp, or a comparison on
# values that are never equal all produce survivors that mean nothing.
#
# So the exit status is the survivor count and not a verdict, and every survivor
# has to be read and classified by hand before it is called a hole. Reporting the
# raw count as missing coverage would be the same error as reporting a skipped
# test as a passing one.

set -uo pipefail

# shellcheck source=../../scripts/mutants-common.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/mutants-common.sh"

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

# Same guard as the curated harness, for the same reason: restoring means
# `git checkout`, which discards uncommitted work in the shader without saying so.
if ! git diff --quiet -- "$shader" || ! git diff --cached --quiet -- "$shader"; then
    echo "$shader has uncommitted changes." >&2
    echo "This harness restores by 'git checkout', which would discard them." >&2
    echo "Commit or stash the shader first." >&2
    exit 101
fi

MUTANT_CATCHERS="$(mktemp)"
MUTANT_RETRY_LOG="$(mktemp)"
MUTANT_FAILURES=0
trap 'restore; rm -f "$MUTANT_CATCHERS" "$MUTANT_RETRY_LOG"' EXIT

# Whitespace on both sides is required, and that is the whole reason this
# enumeration is trustworthy.
#
# WGSL spells generics with angle brackets -- `vec3<u32>`, `array<f32>`,
# `textureStore<...>` -- so a pattern matching bare `<` and `>` finds 174 sites in
# this shader of which only 92 are comparisons. The other 82 are type syntax, and
# flipping one produces a file that does not parse. `mutant_run_suite` scores a
# compile error as *caught*, which is right for a real defect the type system
# rejects and badly wrong here: it would report 82 phantom catches and a coverage
# figure that mostly measures the WGSL grammar.
#
# Comparisons in this shader are always written spaced (`i < n`), and generics
# never are (`vec3<u32>`), so requiring spaces separates them exactly. Verified:
# the spaced pattern matches zero type parameters.
readonly OP_RE='(?<=\s)(<=|>=|==|<|>)(?=\s)'

flip() {
    case "$1" in
        '<') echo '<=' ;;
        '<=') echo '<' ;;
        '>') echo '>=' ;;
        '>=') echo '>' ;;
        '==') echo '!=' ;;
        *) echo "" ;;
    esac
}

mapfile -t ops < <(perl -0777 -ne "while (/$OP_RE/g) { print \"\$1\\n\" }" "$shader")
total="${#ops[@]}"

# Sites inside `//` comments, which are skipped.
#
# Found by running this: two survivors were flips of comparisons written in
# prose -- `// that for x < -88 ...` and `// c = grad * (pre >= 0)`. Neither can
# change what the program computes, so both are guaranteed survivors, and a
# survivor count that includes them overstates the uncaught set by exactly as
# many comparison operators as the comments happen to contain. They were 2 of 54.
#
# Site numbering is left alone rather than renumbered around the skips, so an
# index in this table means the same thing as an index in a run made before the
# filter existed.
mapfile -t comment_sites < <(perl -0777 -ne '
    my $s = $_;
    my $n = 0;
    while ($s =~ /(?<=\s)(<=|>=|==|<|>)(?=\s)/g) {
        $n++;
        my $pos = pos($s);
        my $ls = rindex($s, "\n", $pos) + 1;
        print "$n\n" if index(substr($s, $ls, $pos - $ls), "//") >= 0;
    }
' "$shader")

is_comment_site() {
    local want="$1" s
    for s in "${comment_sites[@]}"; do
        [ "$s" = "$want" ] && return 0
    done
    return 1
}

if [ "$total" -eq 0 ]; then
    echo "no comparison operators matched." >&2
    echo "The pattern or the shader's spacing convention has changed." >&2
    exit 1
fi

# Same feature requirement as the curated harness: without `--features gpu` every
# hardware test is ignored, each suite exits zero having run nothing, and every
# mutant is reported as surviving. That would print a table of 92 coverage holes
# describing only how the script was invoked.
features="--features gpu"

if ! probe="$(cargo test -p aether-gpu $features --test gpu_parity -- --nocapture \
    the_selected_adapter_is_real_hardware_not_a_software_rasterizer 2>&1)"; then
    echo "the adapter probe failed:" >&2
    printf '%s\n' "$probe" >&2
    exit 100
fi

if ! printf '%s' "$probe" | grep -q "1 passed"; then
    echo "the adapter probe ran no test." >&2
    printf '%s\n' "$probe" >&2
    exit 100
fi

if printf '%s' "$probe" | grep -q "device_type=Cpu"; then
    echo "adapter is a software rasterizer. Refusing." >&2
    exit 100
fi

suites=(gpu_parity gradcheck attention_parity)

echo "$total comparison sites in $(basename "$shader")"
echo
printf '%-5s %-58s' "SITE" "CONTEXT"
for suite in "${suites[@]}"; do printf ' %-10s' "$suite"; done
printf '\n'

survived=0
survivors=()

skipped_comments=0

for ((i = 1; i <= total; i++)); do
    op="${ops[i - 1]}"
    rep="$(flip "$op")"

    if is_comment_site "$i"; then
        skipped_comments=$((skipped_comments + 1))
        continue
    fi

    restore

    # Flip the i-th match and nothing else. The counter runs over the same
    # pattern used for the enumeration, so site numbering is stable between the
    # listing and the patch.
    before="$(md5sum < "$shader")"
    perl -0777 -pi -e "
        BEGIN { \$idx = $i; \$rep = quotemeta('$rep'); \$rep =~ s/\\\\//g }
        my \$n = 0;
        s/$OP_RE/++\$n == \$idx ? \$rep : \$1/ge
    " "$shader"
    after="$(md5sum < "$shader")"

    if [ "$before" = "$after" ]; then
        printf '%-5s %-58s  <- site did not patch\n' "$i" "($op unchanged)"
        continue
    fi

    # The line the flip landed on, for reading the table without a diff.
    context="$(git diff -U0 -- "$shader" | grep '^+[^+]' | head -1 | cut -c2- | sed 's/^[[:space:]]*//' | cut -c1-56)"

    touch "$shader"

    any_caught=0
    printf '%-5s %-58s' "$i" "$context"
    for suite in "${suites[@]}"; do
        if ! mutant_run_suite aether-gpu "$suite" "site $i: $op -> $rep" 10 "$features"; then
            any_caught=1
        fi
    done

    # A survival is confirmed before it is believed; a catch is not re-run.
    #
    # Some mutants are caught intermittently. Flipping `pairwise_sqdist`'s column
    # guard lets thread `j = m` write `c[i*m + m]`, which is the cell thread
    # `(i+1, 0)` writes legitimately, so two threads race for it and which value
    # survives depends on GPU scheduling. Run alone that mutant passed 8 of 8;
    # run inside the full suite, where the harness executes tests in parallel and
    # the device is under load, it failed 2 of 3. A single measurement called it
    # a survivor, and it was reported as a coverage hole it is not.
    #
    # Only survivors are re-run, which costs a rerun on the minority of cells and
    # nothing on the rest. The asymmetry is deliberate: a catch is positive
    # evidence and repeating it cannot overturn anything, while a survival is the
    # absence of evidence and is exactly what an unlucky schedule fabricates.
    #
    # Two observations do not make this sound, only less wrong. A mutant caught
    # one run in ten still reads as a survivor most of the time.
    if [ "$any_caught" -eq 0 ]; then
        for suite in "${suites[@]}"; do
            if ! mutant_run_suite aether-gpu "$suite" "site $i: $op -> $rep (confirm)" 0 "$features"; then
                any_caught=1
            fi
        done
    fi

    if [ "$any_caught" -eq 0 ]; then
        survived=$((survived + 1))
        survivors+=("$i|$op -> $rep|$context")
        printf '  <- SURVIVED (confirmed twice)\n'
    else
        printf '\n'
    fi
done

restore

echo
echo "survivors: $survived / $((total - skipped_comments))"
echo "($skipped_comments of $total sites lie inside comments and were skipped)"

if [ "$survived" -gt 0 ]; then
    echo
    echo "Each of these needs reading before it is called a coverage hole. A flip"
    echo "that does not change what the program computes cannot be caught by any"
    echo "test, and is not a gap in the suite:"
    for s in "${survivors[@]}"; do
        printf '  site %-4s %-24s %s\n' "${s%%|*}" "$(echo "$s" | cut -d'|' -f2)" "$(echo "$s" | cut -d'|' -f3)"
    done
fi

mutant_report_catchers "$MUTANT_FAILURES"

exit "$survived"
