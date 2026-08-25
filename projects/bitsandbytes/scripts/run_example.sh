#!/usr/bin/env bash
# Run one bitsandbytes pytest file from a CI working copy of the target tree.
# Overlay CLI args come from OVERLAY_ARGS (JSON array).
set -euo pipefail

export PYTHONNOUSERSITE=1

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <test-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"

case "$EXAMPLE_REL" in
  tests/test_ops.py|tests/test_linear4bit.py)
    ;;
  *)
    echo "unsupported bitsandbytes guard path: $EXAMPLE_REL" >&2
    exit 1
    ;;
esac

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"
if [[ ! -f "$EXAMPLE_PATH" ]]; then
  echo "upstream test not found: $EXAMPLE_PATH" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR"
RUN_LOG="$CI_OUTPUT_DIR/run.log"

expand_overlay() {
  python - <<'PY'
import json
import os
import shlex

raw = os.environ.get('OVERLAY_ARGS', '').strip()
if not raw or raw in ('null', '""'):
    raise SystemExit(0)
items = json.loads(raw)
if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
    raise SystemExit('OVERLAY_ARGS must be a JSON array of strings')
tokens = []
for item in items:
    tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
print(' '.join(shlex.quote(token) for token in tokens))
PY
}

eval "EXTRA_ARGS=( $(expand_overlay) )"

export PATH="/usr/local/sbin:$PATH"
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export BNB_TEST_DEVICE=npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PROJECT_ROOT}/scripts${PYTHONPATH:+:$PYTHONPATH}"

echo "running upstream bitsandbytes test: $EXAMPLE_REL"
cd "$TARGET_ROOT"
python -m pytest -p bnb_npu_bootstrap "$EXAMPLE_REL" -v "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$RUN_LOG"

if ! grep -qE '\[npu|-npu\]' "$RUN_LOG"; then
  echo "pytest did not parametrize on npu; BNB_TEST_DEVICE did not take effect, exit 0 is fake green" >&2
  exit 1
fi
if grep -qE '\[cpu|-cpu\]|\[cuda|-cuda\]' "$RUN_LOG"; then
  echo "pytest log contains cpu or cuda node id; refusing silent fallback" >&2
  exit 1
fi
if ! grep -q 'passed' "$RUN_LOG"; then
  echo "fake-green: pytest log has no passed marker" >&2
  exit 1
fi
