#!/usr/bin/env bash
# Run one example from a CI working copy of the target tree.
# Overlay CLI args come from OVERLAY_ARGS (JSON array). If the example does
# not already pass "$@", attach it in this working copy only: after
# `--model_name swift-robot` when that flag exists, otherwise on the last
# command line. Also rewrite <your_local_megatron_lm_path> to MEGATRON_LM_PATH
# so megatron examples can start. Never git add/commit/push.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
FIXTURE_DIR="${FIXTURE_DIR:?FIXTURE_DIR is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"

if [[ ! -f "$EXAMPLE_PATH" ]]; then
  echo "example not found: $EXAMPLE_PATH" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

rewrite_megatron_placeholders() {
  local script="$1"
  if ! grep -qF '<your_local_megatron_lm_path>' "$script"; then
    return
  fi
  if [[ -z "${MEGATRON_LM_PATH:-}" ]]; then
    echo "example has <your_local_megatron_lm_path> but MEGATRON_LM_PATH is unset" >&2
    return
  fi
  "$PYTHON" - "$script" "$MEGATRON_LM_PATH" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
replacement = sys.argv[2]
text = path.read_text(encoding='utf-8')
updated = text.replace('<your_local_megatron_lm_path>', replacement)
if updated != text:
    path.write_text(updated, encoding='utf-8')
    print(f'rewrote megatron path placeholder in {path}')
PY
}

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
if needle in text:
    passthrough = needle + ' "$@"'
    if passthrough not in text:
        idx = text.rfind(needle)
        text = text[:idx] + passthrough + text[idx + len(needle):]
        path.write_text(text, encoding='utf-8')
        print(f'patched {path} to pass "$@" after {needle}')
    raise SystemExit(0)

lines = text.splitlines(keepends=True)
for i in range(len(lines) - 1, -1, -1):
    stripped = lines[i].strip()
    if stripped and not stripped.startswith('#'):
        raw = lines[i]
        newline = ''
        if raw.endswith('\r\n'):
            newline = '\r\n'
            raw = raw[:-2]
        elif raw.endswith('\n'):
            newline = '\n'
            raw = raw[:-1]
        lines[i] = raw.rstrip() + ' "$@"' + newline
        path.write_text(''.join(lines), encoding='utf-8')
        print(f'patched {path} to pass "$@" on last command line')
        raise SystemExit(0)
raise SystemExit(f'{path}: cannot find a command line to attach "$@"')
PY
}

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

eval "EXTRA_ARGS=( $(expand_overlay) )"

rewrite_megatron_placeholders "$EXAMPLE_PATH"

echo "running $EXAMPLE_PATH with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
  ensure_passthrough "$EXAMPLE_PATH"
fi

cd "$TARGET_ROOT"
# Export so the example script and megatron see the same cache/output contract.
export FIXTURE_DIR CI_OUTPUT_DIR ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
bash "$EXAMPLE_PATH" "${EXTRA_ARGS[@]}"
