#!/usr/bin/env bash
# Run one example from a CI working copy of the target tree.
# If the example does not already pass "$@" through, append it after
# `--model_name swift-robot` in this working copy only. Never git add/commit/push.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <example-relpath> <overlay-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
OVERLAY_REL="$2"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
FIXTURE_DIR="${FIXTURE_DIR:?FIXTURE_DIR is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"
OVERLAY_PATH="$PROJECT_ROOT/$OVERLAY_REL"

if [[ ! -f "$EXAMPLE_PATH" ]]; then
  echo "example not found: $EXAMPLE_PATH" >&2
  exit 1
fi
if [[ ! -f "$OVERLAY_PATH" ]]; then
  echo "overlay not found: $OVERLAY_PATH" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

ensure_passthrough() {
  local script="$1"
  if grep -qE '"\$@"' "$script"; then
    echo "example already has \"\$@\"; skipping sed"
    return
  fi
  "$PYTHON" - "$script" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding='utf-8')
needle = '--model_name swift-robot'
passthrough = needle + ' "$@"'
if passthrough in text:
    raise SystemExit(0)
if needle not in text:
    raise SystemExit(f'{path}: cannot find {needle!r} to attach "$@"')
idx = text.rfind(needle)
path.write_text(text[:idx] + passthrough + text[idx + len(needle):], encoding='utf-8')
print(f'patched {path} to pass "$@"')
PY
}

expand_overlay() {
  "$PYTHON" - "$OVERLAY_PATH" <<'PY'
import os
import shlex
import sys
from pathlib import Path

text = os.path.expandvars(Path(sys.argv[1]).read_text(encoding='utf-8'))
tokens = []
for raw in text.splitlines():
    line = raw.strip()
    if not line or line.startswith('#'):
        continue
    tokens.extend(shlex.split(line, posix=True))
print(' '.join(shlex.quote(token) for token in tokens))
PY
}

ensure_passthrough "$EXAMPLE_PATH"

eval "EXTRA_ARGS=( $(expand_overlay) )"

echo "running $EXAMPLE_PATH with ${#EXTRA_ARGS[@]} overlay args"
printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"

cd "$TARGET_ROOT"
# Export so the example script and megatron see the same cache/output contract.
export FIXTURE_DIR CI_OUTPUT_DIR ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
bash "$EXAMPLE_PATH" "${EXTRA_ARGS[@]}"
