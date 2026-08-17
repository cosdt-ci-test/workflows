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

# Huawei/Ascend wheels for the current CANN line. Keep both indexes:
# repo.huaweicloud.com is what this runner already resolves; mirrors +
# variant are what current vllm-ascend install docs use.
ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
ASCEND_PIP_MIRROR=https://mirrors.huaweicloud.com/ascend/repos/pypi
ASCEND_PIP_VARIANT=https://mirrors.huaweicloud.com/ascend/repos/pypi/variant

# CANN base image has no torch. Pin the same 2.9 stack as
# ms-swift-quick-start.yml, then install ms-swift + torchvision
# before setup_${PROFILE}. Each function below only adds extras.

pip_ascend() {
  python -m pip install \
    --extra-index-url "$ASCEND_PIP_INDEX" \
    --extra-index-url "$ASCEND_PIP_MIRROR" \
    --extra-index-url "$ASCEND_PIP_VARIANT" \
    "$@"
}

install_qwen_extras() {
  # ms-swift Qwen3.5 / Omni processor check requires these extras.
  python -m pip install "qwen_vl_utils>=0.0.14" decord
}

setup_swift() {
  :
}

setup_deepspeed() {
  python -m pip install deepspeed
}

setup_vllm() {
  # ms-swift NPU-support.md pin for torch 2.9 / A2. Do not jump to
  # vllm-ascend 0.23: that line wants torch 2.10.
  python -m pip install vllm==0.18.0
  pip_ascend vllm-ascend==0.18.0
  # vllm 0.18 requires transformers<5 and downgrades it. Qwen3.5 needs
  # transformers>=5.2. vllm-ascend 0.18 only requires >=4.57.4, so put
  # 5.2 back without letting pip uninstall vllm to satisfy the <5 cap.
  python -m pip install --no-deps "transformers>=5.2.0"
  install_qwen_extras
}

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
  pip_ascend triton-ascend==3.2.1 \
    --find-links https://repo.huaweicloud.com/ascend/repos/pypi/triton-ascend/
  echo "MEGATRON_LM_PATH=$GITHUB_WORKSPACE/deps/Megatron-LM" >> "$GITHUB_ENV"
  echo "PYTHONPATH=$GITHUB_WORKSPACE/deps/Megatron-LM" >> "$GITHUB_ENV"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python -c "import mindspeed.megatron_adaptor; from swift.megatron.init import init_megatron_env; init_megatron_env()"
}

setup_megatron_vllm() {
  setup_megatron
  setup_vllm
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

python -m pip install -U pip
python -m pip install torch==2.9.0
pip_ascend torch_npu==2.9.0.post2
python -m pip install -e "$TARGET_ROOT"
# requirements/npu.txt pins torchvision for torch 2.7.1; we use 0.24.0 for torch 2.9.0.
python -m pip install torchvision==0.24.0 decorator
install_qwen_extras

"setup_${PROFILE}"
