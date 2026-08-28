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

stop_pid() {
  local p="${1:-}"
  [[ -n "$p" ]] || return 0
  kill -TERM "$p" 2>/dev/null || true
  local _
  for _ in $(seq 1 30); do
    if ! kill -0 "$p" 2>/dev/null; then
      wait "$p" 2>/dev/null || true
      return 0
    fi
    sleep 1
  done
  kill -KILL "$p" 2>/dev/null || true
  wait "$p" 2>/dev/null || true
}

parse_listen_endpoint() {
  local log="$1"
  "$PYTHON" - "$log" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding='utf-8', errors='replace').read()
matches = re.findall(r'listening on (\S+:\d+)', text)
if not matches:
    raise SystemExit(1)
print(matches[-1])
PY
}

eval "EXTRA_ARGS=( $(expand_overlay) )"

echo "running $EXEC_PATH for $EXAMPLE_REL with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

if [[ "$PROFILE" != "ascend-direct" ]]; then
  echo "unknown profile for run: ${PROFILE:-empty} (supported: ascend-direct)" >&2
  exit 2
fi

source_cann
export GLOG_logtostderr=1
export GLOG_alsologtostderr=1
cd "$TARGET_ROOT"

TARGET_LOG="$CI_OUTPUT_DIR/target.log"
INIT_LOG="$CI_OUTPUT_DIR/initiator.log"
COMBINED_LOG="$CI_OUTPUT_DIR/transfer_engine_ascend_direct_perf.log"

launch=("$EXEC_PATH")
if command -v stdbuf >/dev/null 2>&1; then
  launch=(stdbuf -oL -eL "$EXEC_PATH")
fi

target_pid=""
set +e
"${launch[@]}" \
  --mode=target \
  --device_logicid=0 \
  --local_server_name=127.0.0.1:12345 \
  --metadata_server=P2PHANDSHAKE \
  "${EXTRA_ARGS[@]}" >"$TARGET_LOG" 2>&1 &
target_pid=$!
set -e
# shellcheck disable=SC2064
trap "stop_pid $target_pid" EXIT

endpoint=""
for _ in $(seq 1 60); do
  if ! kill -0 "$target_pid" 2>/dev/null; then
    echo "target exited before listen" >&2
    cat "$TARGET_LOG" >&2
    exit 1
  fi
  if endpoint=$(parse_listen_endpoint "$TARGET_LOG"); then
    break
  fi
  sleep 1
done
if [[ -z "$endpoint" ]]; then
  echo "target did not print a listening endpoint" >&2
  cat "$TARGET_LOG" >&2
  exit 1
fi
echo "target listening on $endpoint"

set +e
"${launch[@]}" \
  --mode=initiator \
  --device_logicid=1 \
  --local_server_name=127.0.0.1:12346 \
  --metadata_server=P2PHANDSHAKE \
  --segment_id="$endpoint" \
  --operation=write \
  "${EXTRA_ARGS[@]}" >"$INIT_LOG" 2>&1
init_ec=$?
set -e
cat "$INIT_LOG"

{
  echo '===== target ====='
  cat "$TARGET_LOG"
  echo '===== initiator ====='
  cat "$INIT_LOG"
} >"$COMBINED_LOG"

if [[ "$init_ec" != 0 ]]; then
  echo "initiator exited $init_ec" >&2
  exit "$init_ec"
fi
if grep -q 'Failed to install Ascend transport' "$COMBINED_LOG"; then
  echo "Ascend Direct transport failed to install" >&2
  exit 1
fi
if grep -qE 'getTransferStatus FAILED|Sync data transfer timeout' "$INIT_LOG"; then
  echo "initiator reported a failed or timed-out transfer" >&2
  exit 1
fi
if ! grep -q 'Success to initialize adxl engine' "$COMBINED_LOG"; then
  echo "ADXL engine did not initialize; check /etc/hccn.conf and card mounts" >&2
  exit 1
fi
if ! grep -q 'Test completed:' "$INIT_LOG"; then
  echo "initiator log missing Test completed" >&2
  exit 1
fi
# Example registerLocalMemory location is "npu:<logicid>". Keep mem type:device
# as an alternate library phrasing seen on the same path.
if ! grep -Eq 'npu:[0-9]+|mem type:device' "$COMBINED_LOG"; then
  echo "run used no NPU device buffer; check card mounts and USE_ASCEND_DIRECT" >&2
  exit 1
fi

stop_pid "$target_pid"
trap - EXIT
