#!/usr/bin/env bash
#
# Classify the survivors of `mutants-mechanical.sh` as equivalent or as holes.
#
#   ./crates/aether-gpu/mutants-mechanical.sh > mech.txt
#   ./crates/aether-gpu/classify-survivors.sh mech.txt
#
# A survivor of the mechanical sweep is a comparison flip that no suite caught.
# That is not the same as a coverage hole: a flip which does not change what the
# program computes is an *equivalent mutant*, and no test can catch it because
# there is nothing to catch.
#
# The suites cannot tell the two apart. They report pass or fail against a
# tolerance, so "survives" means the outputs agreed within that tolerance rather
# than that they were identical. `examples/equivalence_probe.rs` dispatches every
# kernel in the shader and checksums the raw bits of each result, which answers
# the question exactly:
#
#   identical checksum  -> the mutation changed nothing. Equivalent. Not a hole.
#   different checksum  -> the mutation changed a result and every suite still
#                          passed. A tolerance absorbed a real difference, and
#                          that is a hole with a measurement behind it.
#
# The second outcome is the one worth having. It cannot be reached by reading the
# code, because the question is not whether the flip *could* matter but whether
# the suites would notice that it did.

set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
shader="$repo/crates/aether-gpu/src/shaders.wgsl"
cd "$repo" || exit 1

sweep="${1:-}"
if [ -z "$sweep" ] || [ ! -f "$sweep" ]; then
    echo "usage: $0 <output of mutants-mechanical.sh>" >&2
    exit 1
fi

if ! git diff --quiet -- "$shader" || ! git diff --cached --quiet -- "$shader"; then
    echo "$shader has uncommitted changes." >&2
    echo "This restores by 'git checkout', which would discard them." >&2
    exit 101
fi

restore() {
    git checkout -- "$shader" 2>/dev/null
    touch "$shader"
}
trap restore EXIT

readonly OP_RE='(?<=\s)(<=|>=|==|<|>)(?=\s)'

probe() {
    cargo run -q -p aether-gpu --example equivalence_probe --release --features gpu 2>&1 \
        | grep COMBINED | awk '{print $2}'
}

restore
baseline="$(probe)"

if [ -z "$baseline" ]; then
    echo "the probe printed no checksum on a clean tree." >&2
    echo "Refusing to classify anything against an empty baseline." >&2
    exit 100
fi

echo "baseline $baseline"
echo

# The survivor block of the sweep output: "  site N  op -> rep   context".
mapfile -t survivors < <(grep -oE '^  site [0-9]+ +[<>=!]+ -> [<>=!]+' "$sweep" \
    | sed -E 's/^  site ([0-9]+) +([<>=!]+) -> ([<>=!]+)/\1 \3/')

if [ "${#survivors[@]}" -eq 0 ]; then
    echo "no survivors parsed from $sweep." >&2
    echo "Either the sweep found none or its output format changed; the second" >&2
    echo "would report a clean classification having examined nothing." >&2
    exit 100
fi

printf '%-6s %-8s %-20s %s\n' "SITE" "FLIP" "CHECKSUM" "VERDICT"

equivalent=0
holes=0
hole_sites=()

for entry in "${survivors[@]}"; do
    i="${entry%% *}"
    rep="${entry##* }"

    restore
    perl -0777 -pi -e "
        BEGIN { \$idx = $i; \$rep = '$rep' }
        my \$n = 0;
        s/$OP_RE/++\$n == \$idx ? \$rep : \$1/ge
    " "$shader"
    touch "$shader"

    sum="$(probe)"

    if [ -z "$sum" ]; then
        # A mutation that stops the probe running is not evidence either way, and
        # scoring it as equivalent would credit the mutant with changing nothing
        # when the truth is that nothing was measured.
        printf '%-6s %-8s %-20s %s\n' "$i" "-> $rep" "(no output)" "UNMEASURED - probe failed"
        continue
    fi

    if [ "$sum" = "$baseline" ]; then
        equivalent=$((equivalent + 1))
        printf '%-6s %-8s %-20s %s\n' "$i" "-> $rep" "$sum" "equivalent"
    else
        holes=$((holes + 1))
        hole_sites+=("$i")
        printf '%-6s %-8s %-20s %s\n' "$i" "-> $rep" "$sum" "*** CHANGED OUTPUT - suites missed it"
    fi
done

restore

echo
echo "of ${#survivors[@]} survivors: $equivalent equivalent, $holes changed output"

if [ "$holes" -gt 0 ]; then
    echo
    echo "These flips changed a kernel result and every suite still passed."
    echo "Each is a defect the tests would not report, at sites: ${hole_sites[*]}"
fi

exit "$holes"
