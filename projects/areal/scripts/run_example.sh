#!/usr/bin/env bash
# Run one AReaL example from a CI working copy of the target tree.
# $1 is the example path relative to TARGET_ROOT. Overlay CLI args come from
# OVERLAY_ARGS (JSON array, serialized by the workflow from
# manifest.overlay_args); ${...} items are expanded from the job environment.
#
# AReaL examples are plain `python examples/.../foo.py <overrides>` entrypoints
# (Hydra-style key=value overrides), so no launcher is involved.
set -euo pipefail

EXAMPLE_REL="${1:?example path is required}"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
LAUNCH_PATH="$TARGET_ROOT/$EXAMPLE_REL"
[[ -f "$LAUNCH_PATH" ]] || { echo "no such example: $LAUNCH_PATH" >&2; exit 2; }

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
items = json.loads(raw)
if items in (None, ''):
    raise SystemExit(0)
tokens = []
for item in items:
    tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
print(' '.join(shlex.quote(token) for token in tokens))
PY
}

eval "EXTRA_ARGS=( $(expand_overlay) )"
echo "running $LAUNCH_PATH with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

cd "$TARGET_ROOT"
# Repo root first so dotted module paths used by workflow kwargs resolve
# (e.g. boba_grpo.py's reward_fn "examples.math.boba_grpo.boba_reward_fn");
# the example dir covers sibling modules imported by the script itself.
export PYTHONPATH="$TARGET_ROOT:$(dirname "$LAUNCH_PATH")${PYTHONPATH:+:$PYTHONPATH}"
# examples/scaffolding/*.py use package-relative imports (from ._compat
# import ...), so they must run as modules, not as plain scripts.
#
# Forked runtime logs (data-worker / data-router / data-gateway under
# <fileroot>/logs/) are NOT part of this job's stdout, so a crash there is
# otherwise invisible; dump them when the example fails.
dump_forked_logs() {
  local root="${AREAL_LOG_ROOT:-/tmp/areal/experiments/logs}"
  [[ -d "$root" ]] || return 0
  local f
  while IFS= read -r f; do
    echo "===== $f (tail -n 200) ====="
    tail -n 200 "$f" || true
  done < <(find "$root" -type f -name 'data-*.log' 2>/dev/null)
}

run_rc=0
case "$EXAMPLE_REL" in
  examples/scaffolding/*.py)
    MODULE="${EXAMPLE_REL%.py}"
    MODULE="${MODULE//\//.}"
    "$PYTHON" -m "$MODULE" "${EXTRA_ARGS[@]}" || run_rc=$?
    ;;
  *.sh)
    # Shell examples dispatch with bash (generic guard contract: .sh -> bash,
    # .py -> python).
    bash "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" || run_rc=$?
    ;;
  *)
    "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" || run_rc=$?
    ;;
esac

if ((run_rc != 0)); then
  echo "example exited rc=$run_rc; dumping forked runtime logs"
  dump_forked_logs
fi
exit "$run_rc"
