#!/usr/bin/env bash
# Prepare the CI environment for one supported example.
# $1 is the manifest profile. Unknown profiles fail before any install.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

MEGATRON_LM_REF=core_v0.16.0
MINDSPEED_REF=core_r0.16.0
MCORE_BRIDGE_REF=v1.6.1

setup_megatron() {
  mkdir -p "$GITHUB_WORKSPACE/deps"
  git clone --depth 1 --branch "$MEGATRON_LM_REF" \
    https://github.com/NVIDIA/Megatron-LM.git "$GITHUB_WORKSPACE/deps/Megatron-LM"
  git clone --depth 1 --branch "$MINDSPEED_REF" \
    https://gitcode.com/Ascend/MindSpeed.git "$GITHUB_WORKSPACE/deps/MindSpeed"
  git clone --depth 1 --branch "$MCORE_BRIDGE_REF" \
    https://github.com/modelscope/mcore-bridge.git "$GITHUB_WORKSPACE/deps/mcore-bridge"
  python -m pip install -e "$GITHUB_WORKSPACE/deps/MindSpeed"
  python -m pip install -e "$GITHUB_WORKSPACE/deps/mcore-bridge"
  python -m pip install triton-ascend==3.2.1 \
    --find-links https://repo.huaweicloud.com/ascend/repos/pypi/triton-ascend/
  echo "MEGATRON_LM_PATH=$GITHUB_WORKSPACE/deps/Megatron-LM" >> "$GITHUB_ENV"
  echo "PYTHONPATH=$GITHUB_WORKSPACE/deps/Megatron-LM" >> "$GITHUB_ENV"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python -c "import mindspeed.megatron_adaptor; from swift.megatron.init import init_megatron_env; init_megatron_env()"
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: megatron)" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

python -m pip install -U pip
python -m pip install -e "$TARGET_ROOT"
# requirements/npu.txt pins torchvision for torch 2.7.1; this image ships torch 2.9.0.
python -m pip install torchvision==0.24.0 decorator

"setup_${PROFILE}"
