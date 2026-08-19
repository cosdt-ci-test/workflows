#!/usr/bin/env bash
# Prepare the CI environment for one supported example.
# $1 is the manifest profile. Unknown profiles fail before any install.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

# 910B DataType table lists FP16 / Q8_0 / Q4_0 / BF16 (Q4_K_M etc. not in table); upstream docs/backend/CANN.md.
MODEL_URL=https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/master/qwen2.5-0.5b-instruct-q4_0.gguf
MODEL_FILE=qwen2.5-0.5b-instruct-q4_0.gguf

build_llama_cann() {
  # cmake derives SOC_TYPE by running npu-smi, mounted at /usr/local/sbin.
  export PATH="/usr/local/sbin:$PATH"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  cmake -S "$TARGET_ROOT" -B "$TARGET_ROOT/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CANN=on
  # llama-simple links llama only, so this skips common/, tools/ and server.
  cmake --build "$TARGET_ROOT/build" --target llama-simple -j "$(nproc)"
  if ! compgen -G "$TARGET_ROOT/build/bin/libggml-cann.*" >/dev/null; then
    echo "build produced no CANN backend library under build/bin" >&2
    exit 1
  fi
}

fetch_model() {
  local dest="$GITHUB_WORKSPACE/models/$MODEL_FILE"
  mkdir -p "$(dirname "$dest")"
  curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 "$MODEL_URL" -o "$dest"
  if [[ "$(head -c 4 "$dest")" != "GGUF" ]]; then
    echo "downloaded file is not a GGUF model: $dest" >&2
    exit 1
  fi
  echo "LLAMA_CI_MODEL=$dest" >> "$GITHUB_ENV"
  echo "model ready: $dest"
}

setup_cann() {
  build_llama_cann
  fetch_model
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

"setup_${PROFILE}"
