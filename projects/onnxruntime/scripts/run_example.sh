#!/usr/bin/env bash
# Run one example from a CI working copy of the target tree.
# Overlay CLI args come from OVERLAY_ARGS (JSON array). Never
# git add/commit/push.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
EXEC_REL="${EXEC:?EXEC is required}"
PROFILE="${PROFILE:-}"

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"

if [[ ! -e "$EXAMPLE_PATH" ]]; then
  echo "example not found: $EXAMPLE_PATH" >&2
  exit 1
fi

EXEC_PATH="$TARGET_ROOT/$EXEC_REL"
if [[ ! -f "$EXEC_PATH" ]]; then
  echo "exec not found: $EXEC_PATH" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

expand_overlay() {
  "$PYTHON" - <<'PY'
import json
import os
import shlex

raw = os.environ.get('OVERLAY_ARGS', '').strip()
if not raw or raw in ('null', '""'):
    raise SystemExit(0)
try:
    items = json.loads(raw)
except json.JSONDecodeError as exc:
    raise SystemExit(f'OVERLAY_ARGS is not valid JSON: {exc}') from exc
if items in (None, ''):
    raise SystemExit(0)
if not isinstance(items, list):
    raise SystemExit(
        f'OVERLAY_ARGS must be a JSON array, got {type(items).__name__}')
tokens = []
for item in items:
    if not isinstance(item, str):
        raise SystemExit(
            f'OVERLAY_ARGS items must be strings, got {type(item).__name__}')
    tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
print(' '.join(shlex.quote(token) for token in tokens))
PY
}

source_cann() {
  export PATH="/usr/local/sbin:/usr/local/bin:$PATH"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
}

eval "EXTRA_ARGS=( $(expand_overlay) )"

echo "running $EXEC_PATH for $EXAMPLE_REL with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

source_cann
cd "$TARGET_ROOT"
export CI_OUTPUT_DIR ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

RUN_LOG="$CI_OUTPUT_DIR/$(basename "$EXEC_REL").log"

assert_cann-gtest() {
  if grep -qF '0 tests from 0 test suites' "$RUN_LOG"; then
    echo "gtest ran 0 tests; --gtest_filter matched nothing" >&2
    exit 1
  fi
  if ! grep -qF 'CannExecutionProviderTest.FunctionTest' "$RUN_LOG"; then
    echo "gtest log missing CannExecutionProviderTest.FunctionTest; empty --gtest_filter is a false green" >&2
    exit 1
  fi
  if grep -Eq '\[  PASSED  \] 1 test' "$RUN_LOG"; then
    :
  elif grep -Eq '\[  PASSED  \] [1-9][0-9]* tests?' "$RUN_LOG"; then
    :
  else
    echo "gtest did not report a non-zero PASSED count" >&2
    exit 1
  fi
  if grep -Eq 'CANN failure|CANNGRAPH failure' "$RUN_LOG"; then
    echo "gtest log contains a CANN failure marker" >&2
    exit 1
  fi
}

assert_cmake-consumer() {
  if ! grep -qF 'Result: PASS' "$RUN_LOG"; then
    echo "sample did not print Result: PASS" >&2
    exit 1
  fi
  if ! grep -qF 'ONNX Runtime version:' "$RUN_LOG"; then
    echo "sample did not print ONNX Runtime version:" >&2
    exit 1
  fi
}

"$EXEC_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_LOG"

guard_fn=""
if [[ -n "$PROFILE" ]] && declare -F "assert_${PROFILE}" >/dev/null 2>&1; then
  guard_fn="assert_${PROFILE}"
elif [[ "$EXAMPLE_REL" == "onnxruntime/test/providers/cann" ]]; then
  guard_fn="assert_cann-gtest"
elif [[ "$EXAMPLE_REL" == "samples/cxx" ]]; then
  guard_fn="assert_cmake-consumer"
else
  echo "no stdout guard for profile=${PROFILE:-} path=${EXAMPLE_REL}" >&2
  exit 1
fi
"$guard_fn"
