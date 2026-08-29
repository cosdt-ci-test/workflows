#!/usr/bin/env bash
# Prepare the CI environment for xllm examples on NPU.
# $1 is the manifest profile. Unknown profiles fail before any install.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
ASCEND_MIRROR_PIP_INDEX=https://mirrors.huaweicloud.com/ascend/repos/pypi
ASCEND_VARIANT_PIP_INDEX=https://mirrors.huaweicloud.com/ascend/repos/pypi/variant
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"
FALLBACK_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

pip_ascend() {
  python -m pip install --extra-index-url "$ASCEND_PIP_INDEX" "$@"
}

pip_ascend_variant() {
  python -m pip install \
    --extra-index-url "$ASCEND_VARIANT_PIP_INDEX" \
    --extra-index-url "$ASCEND_MIRROR_PIP_INDEX" \
    "$@"
}

select_pip_index() {
  if python -c "
import os
import urllib.error
import urllib.request
try:
    urllib.request.urlopen(os.environ['CLUSTER_PIP_INDEX'], timeout=3)
except urllib.error.HTTPError:
    pass
" 2>/dev/null; then
    export PIP_INDEX_URL="$CLUSTER_PIP_INDEX"
    export PIP_TRUSTED_HOST="$CLUSTER_PIP_HOST"
  else
    export PIP_INDEX_URL="$FALLBACK_PIP_INDEX"
    unset PIP_TRUSTED_HOST
  fi
  echo "pip index: $PIP_INDEX_URL"
}

ensure_torch_stack() {
  # The CANN base image usually ships torch/torch_npu; verify the versions
  # match what xllm expects (2.9.0 / 2.9.0.post2) and reinstall if not.
  if python -c "
import torch, torch_npu
print('found torch', torch.__version__, 'torch_npu', torch_npu.__version__)
raise SystemExit(0 if torch.__version__.startswith('2.9.0') and torch_npu.__version__.startswith('2.9.0') else 1)
"; then
    echo "reusing image torch stack"
    return
  fi
  echo "installing torch==2.9.0 torch_npu==2.9.0.post2"
  pip_ascend torch==2.9.0 torch_npu==2.9.0.post2
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

# Ensure the example model is present; reuse the mounted CI cache when
# available, otherwise pull it from ModelScope (idempotent).
ensure_model() {
  local model_dir="/root/.cache/modelscope/Qwen2-7B-Instruct"
  if [ -d "$model_dir" ] && [ -n "$(ls -A "$model_dir" 2>/dev/null)" ]; then
    echo "model already cached at $model_dir; skipping download"
    return
  fi
  echo "downloading Qwen2-7B-Instruct via modelscope to $model_dir"
  python -m pip install -q modelscope 2>/dev/null || true
  python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2-7B-Instruct', local_dir='$model_dir')"
}

# Install the toolchain needed to build xllm from source on the CANN base
# image (which ships CANN + torch but not cmake/rust/vcpkg).
ensure_build_deps() {
  echo "installing build dependencies (cmake/ninja/rust)..."
  apt-get update -qq 2>/dev/null || true
  apt-get install -y --no-install-recommends \
    python3-dev libssl-dev pkg-config git curl ca-certificates 2>/dev/null || true
  python -m pip install -q "cmake>=3.27" ninja
  CMAKE_BIN_DIR="$(python -c 'import sys,os;print(os.path.dirname(sys.executable))')"
  export PATH="$CMAKE_BIN_DIR:$PATH"
  echo "cmake: $(cmake --version | head -n1)"
  if ! command -v cargo >/dev/null 2>&1; then
    export RUSTUP_DIST_SERVER=https://rsproxy.cn
    export RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
    curl -fsSL https://rsproxy.cn/rustup-init.sh -o /tmp/rustup-init.sh
    sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain stable
    export PATH="$HOME/.cargo/bin:$PATH"
  fi
  # Redirect vcpkg's github fetch to the gitcode mirror (custom vcpkg, cxx11_abi off).
  git config --global url."https://gitcode.com/xLLM-AI/vcpkg.git".insteadOf \
    "https://github.com/microsoft/vcpkg.git" || true
}

# Build the xllm wheel from a checked-out xllm tree and install it.
build_xllm() {
  local root="$1"
  local build_log=/tmp/xllm-build.log
  echo "building xllm wheel from $root (device=npu arch=arm); full log -> $build_log"
  export SKIP_TEST=1
  if ( cd "$root" && python setup.py bdist_wheel --device npu --arch arm ) >"$build_log" 2>&1; then
    echo "build succeeded; tail of $build_log:"
    tail -n 300 "$build_log"
  else
    echo "!! xllm build failed; tail of $build_log:"
    tail -n 300 "$build_log"
    exit 1
  fi
  local wheel
  wheel=$(ls "$root"/dist/*.whl 2>/dev/null | head -n1)
  [ -n "$wheel" ] || { echo "xllm wheel not found after build"; exit 1; }
  python -m pip install --force-reinstall --no-deps "$wheel"
  python -c "import xllm; print('xllm:', xllm.__version__)"
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

# Later steps are a new shell; do not rely on a previous workflow step.
source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip
ensure_torch_stack

# Verify NPU is available
python -c "import torch, torch_npu; print('torch:', torch.__version__, 'torch_npu:', torch_npu.__version__, 'npu_count:', torch.npu.device_count())"
npu-smi info

"setup_${PROFILE}"

# Default profile: build xllm from source on the CANN base image (no xllm-
# provided image). The checked-out target tree at TARGET_ROOT is built and
# installed; `examples/` is importable because run_example.sh does
# `cd "$TARGET_ROOT"`.
setup_default() {
  echo "profile=default: building xllm from source on the CANN base image"
  ensure_build_deps
  build_xllm "$TARGET_ROOT"
  ensure_model
}