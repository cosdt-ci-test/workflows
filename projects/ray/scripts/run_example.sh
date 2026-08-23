#!/usr/bin/env bash
# Run one existing upstream Ray NPU test from the target checkout.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <test-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"

case "$EXAMPLE_REL" in
  python/ray/tests/accelerators/test_npu.py|python/ray/train/tests/test_torch_device_manager.py)
    ;;
  *)
    echo "unsupported Ray guard path: $EXAMPLE_REL" >&2
    exit 1
    ;;
esac

if [[ ! -f "$EXAMPLE_PATH" ]]; then
  echo "upstream test not found: $EXAMPLE_PATH" >&2
  exit 1
fi

expand_overlay() {
  python - <<'PY'
import json
import os
import shlex

raw = os.environ.get("OVERLAY_ARGS", "").strip()
items = [] if not raw or raw in {"null", '""'} else json.loads(raw)
if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
    raise SystemExit("OVERLAY_ARGS must be a JSON array of strings")
tokens = []
for item in items:
    tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
print(" ".join(shlex.quote(token) for token in tokens))
PY
}

eval "EXTRA_ARGS=( $(expand_overlay) )"

export PATH="/usr/local/sbin:$PATH"
# The selected CANN container provides this file.
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export RAY_USAGE_STATS_ENABLED=0
export RAY_DEDUP_LOGS=0

echo "running upstream Ray test: $EXAMPLE_REL"
cd "$TARGET_ROOT"
python -m pytest "$EXAMPLE_PATH" -sv "${EXTRA_ARGS[@]}"
