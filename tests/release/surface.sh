#!/usr/bin/env bash
# Real test of `wisent-gradio-release surface` through the built binary.
#
# It reads this repository's own package and checks that each kind the
# version check compares is there: an `__all__` export, a literal gr.Tab label,
# a CommandGroup label and a command declared through the `_ci` shorthand,
# which the reader finds from the code because `_ci` returns a CommandInfo.
# Then it builds small package trees under this run's directory and checks
# each refusal: a root without the package, a setup.py that declares console
# scripts, a wheel whose dist-info declares them, a module that does not parse
# (and the tolerant read that skips it and names it), a package missing a
# whole kind, a computed `__all__`, and an invocation the command does not
# take. Every command, whether it was accepted, and its answer go to the run's
# report.txt; a failed check stops the run (set -e) after naming itself there.
#
# Usage: WISENT_GRADIO_RELEASE=release/target/debug/wisent-gradio-release tests/release/surface.sh
set -eu
cd "$(dirname "$0")/../.."
BIN=${WISENT_GRADIO_RELEASE:?set WISENT_GRADIO_RELEASE to the binary under test, e.g. release/target/debug/wisent-gradio-release}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/release/target/real-tests/surface-$RUN"
REPORT="$ROOT/report.txt"
mkdir -p "$ROOT"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

fail() {
  echo "FAIL: $1" | tee -a "$REPORT" >/dev/stderr
  false
}
# run accepted|refused|usage ARGS...: the command must end the way named:
# accepted exits 0, refused is a refusal (exit 1), usage an invocation the
# command does not take (exit 2). An accepted run's answer is its stdout; any
# other run's answer is everything it wrote.
run() {
  expected=$1
  shift
  if [ "$expected" = accepted ]; then
    if "$BIN" "$@" >"$ROOT/output" </dev/null; then status=$?; else status=$?; fi
  else
    if "$BIN" "$@" &>"$ROOT/output" </dev/null; then status=$?; else status=$?; fi
  fi
  case "$status" in
    "0") outcome=accepted ;;
    "1") outcome=refused ;;
    "2") outcome=usage ;;
    *) outcome="exit $status" ;;
  esac
  out=$(cat "$ROOT/output")
  printf '$ wisent-gradio-release %s\noutcome: %s\noutput: %s\n\n' "$*" "$outcome" "$out" >>"$REPORT"
  [ "$outcome" = "$expected" ] || fail "$* was $outcome, expected $expected: $out"
}
check() {
  [ "$2" = "$3" ] || fail "$1: got '$2', expected '$3'"
  echo "ok: $1 = $2" >>"$REPORT"
}
has() {
  check "the surface holds $1" "$(jq --arg name "$1" '.surface | index($name) != null' "$ROOT/output")" "true"
}
refused() {
  case "$out" in
    *"$1"*) echo "ok: refused with: $1" >>"$REPORT" ;;
    *) fail "expected a refusal containing '$1', got: $out" ;;
  esac
}
# tree NAME: a package tree that declares every kind once, the shape the
# repository's own package has.
tree() {
  mkdir -p "$ROOT/$1/wisent/app"
  printf '__all__ = ["launch"]\n' >"$ROOT/$1/wisent/app/__init__.py"
  printf 'class CommandInfo:\n    name: str\n\n\ndef _ci(name, help_text):\n    return CommandInfo(name, help_text)\n\n\nCOMMANDS = [_ci("generate-pairs", "Generate pairs")]\n' >"$ROOT/$1/wisent/app/groups.py"
  printf 'import gradio as gr\n\nwith gr.Tab(label="Wizard"):\n    pass\n' >"$ROOT/$1/wisent/app/interface.py"
  echo "$ROOT/$1"
}

run accepted surface "$PWD"
has "api:wisent.app:launch"
has "tab:Wizard"
has "tab:Inspect"
has "tab:Generation"
has "command:generate-pairs"
check "a source tree that parses skips nothing" "$(jq 'has("unparseable")' "$ROOT/output")" "false"

GOOD=$(tree good)
run accepted surface "$GOOD"
check "a fixture tree's surface" "$(jq -c .surface "$ROOT/output")" '["api:wisent.app:launch","command:generate-pairs","tab:Wizard"]'

run refused surface "$ROOT/missing"
refused "is not a directory"

SCRIPTS=$(tree scripts)
printf 'from setuptools import setup\n\nsetup(entry_points={"console_scripts": ["wisent-ui = wisent.app:launch"]})\n' >"$SCRIPTS/setup.py"
run refused surface "$SCRIPTS"
refused "declares entry_points"

NONE=$(tree no-scripts)
printf 'from setuptools import setup\n\nsetup(entry_points=None)\n' >"$NONE/setup.py"
run accepted surface "$NONE"

WHEEL=$(tree wheel)
mkdir -p "$WHEEL/wisent_gradio.dist-info"
printf '[console_scripts]\nwisent-ui = wisent.app:launch\n' >"$WHEEL/wisent_gradio.dist-info/entry_points.txt"
run refused surface "$WHEEL"
refused "declares [console_scripts]"

BROKEN=$(tree broken)
printf 'def broken(:\n' >"$BROKEN/wisent/app/broken.py"
run refused surface "$BROKEN"
refused "does not parse"
run accepted surface "$BROKEN" --tolerant
check "a tolerant read names what it skipped" "$(jq -c .unparseable "$ROOT/output")" '["wisent/app/broken.py"]'
check "a tolerant read keeps the rest" "$(jq -c .surface "$ROOT/output")" '["api:wisent.app:launch","command:generate-pairs","tab:Wizard"]'

KINDLESS=$(tree kindless)
rm "$KINDLESS/wisent/app/groups.py"
run refused surface "$KINDLESS"
refused "no command: names found"

COMPUTED=$(tree computed)
printf '__all__ = [name for name in ("launch",)]\n' >"$COMPUTED/wisent/app/__init__.py"
run refused surface "$COMPUTED"
refused "__all__ is not a literal list or tuple"

run usage surface
refused "usage: wisent-gradio-release surface ROOT"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
