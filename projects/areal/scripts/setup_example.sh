#!/usr/bin/env bash
# Prepare the CI environment for one supported AReaL example.
# $1 is the manifest profile. Unknown profiles fail before any install.
#
# AReaL's NPU support lives on the `ascend-v1.0.5` branch (not main) and needs
# a specific stack (CANN 9.0.1 / torch 2.10 / torch_npu 2.10.0.post2 /
# vLLM 0.23.0 + vLLM-Ascend / transformers 5.5.4 / Megatron-Core 0.16.1 /
# MindSpeed / Megatron-Bridge). The upstream ghcr image
# (ghcr.io/hwvanici/areal_npu:v1.0.5-a2) is not reachable from the runners, so
# this script installs that stack into the CANN base image at run time. It is a
# translation of the repo's Dockerfile.a2.
#
# Contract: docs/guarding-examples.md "项目运行脚本契约".
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

SUPPORTED_PROFILES="areal-vlm-grpo areal-vlm-mt-grpo"
case "$PROFILE" in
  areal-vlm-grpo) MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct" ;;
  areal-vlm-mt-grpo) MODEL_ID="Qwen/Qwen3-VL-2B-Instruct" ;;
  *)
    echo "unknown profile: ${PROFILE} (supported: ${SUPPORTED_PROFILES})" >&2
    exit 1
    ;;
esac

if [[ -z "${TARGET_ROOT:-}" || -z "${GITHUB_WORKSPACE:-}" || -z "${GITHUB_ENV:-}" ]]; then
  echo "TARGET_ROOT / GITHUB_WORKSPACE / GITHUB_ENV must be set" >&2
  exit 2
fi

# Pinned to the versions shipped by the v1.0.5 A2 image (see README/Dockerfile.a2).
MEGATRON_TAG=core_v0.16.1
MINDSPEED_BRANCH=core_r0.16.0
MINDSPEED_COMMIT=79626c1380b78f5cea8a971f265a49f96b04d416
MEGATRON_BRIDGE_COMMIT=de93536e
VLLM_TAG=v0.23.0
VLLM_ASCEND_BRANCH=releases/v0.23.0
VLLM_ASCEND_COMMIT=eaefc536d4475808e18b0c12b42c7710bc4bccb9
WORKSPACE_DIR="$GITHUB_WORKSPACE/areal-workspace"
mkdir -p "$WORKSPACE_DIR"

# System build deps (the clean CANN base ships none) + pip/uv bootstrap.
apt-get update -y
apt-get install -y --no-install-recommends \
  gcc g++ cmake libnuma-dev wget git curl jq build-essential gawk clang-15
update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 20
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 20
python3 -m pip install -U pip "setuptools==80.10.2" uv

# vLLM + vLLM-Ascend (torch-npu comes from vllm-ascend requirements.txt).
export PIP_EXTRA_INDEX_URL=https://triton-ascend.osinfra.cn/pypi/simple/
export PIP_TRUSTED_HOST=triton-ascend.osinfra.cn
export SOC_VERSION=ascend910b1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

git clone --depth 1 --branch "$VLLM_TAG" https://github.com/vllm-project/vllm.git "$WORKSPACE_DIR/vllm"
git clone --branch "$VLLM_ASCEND_BRANCH" https://github.com/vllm-project/vllm-ascend.git "$WORKSPACE_DIR/vllm-ascend"
git -C "$WORKSPACE_DIR/vllm-ascend" checkout "$VLLM_ASCEND_COMMIT"
(
  cd "$WORKSPACE_DIR/vllm"
  VLLM_TARGET_DEVICE=empty python3 -m pip install -v -e .
  git apply -v "$TARGET_ROOT/patches/vllm.$VLLM_TAG.patch"
  python3 -m pip uninstall -y triton
)
(
  cd "$WORKSPACE_DIR/vllm-ascend"
  python3 -m pip install -r requirements.txt
  COMPILE_CUSTOM_KERNELS=1 python3 -m pip install --no-build-isolation --no-deps -v -e .
  git apply -v "$TARGET_ROOT/patches/vllm-ascend.$VLLM_TAG.patch"
)

# Megatron & MindSpeed (megatron-core lives inside the MindSpeed tree).
git clone --depth 1 --branch "$MEGATRON_TAG" https://github.com/NVIDIA/Megatron-LM.git "$WORKSPACE_DIR/Megatron-LM"
git clone --branch "$MINDSPEED_BRANCH" https://gitcode.com/Ascend/MindSpeed.git "$WORKSPACE_DIR/MindSpeed"
git -C "$WORKSPACE_DIR/MindSpeed" checkout "$MINDSPEED_COMMIT"
cp -r "$WORKSPACE_DIR/Megatron-LM/megatron" "$WORKSPACE_DIR/MindSpeed/megatron"
python3 -m pip install -e "$WORKSPACE_DIR/MindSpeed" --no-deps

# AReaL deps (pyproject.npu.toml). No --upgrade: keep vllm-ascend's torch stack.
uv pip install -r "$TARGET_ROOT/pyproject.npu.toml" --system --group dev

# Megatron-Bridge
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git "$WORKSPACE_DIR/Megatron-Bridge"
git -C "$WORKSPACE_DIR/Megatron-Bridge" checkout "$MEGATRON_BRIDGE_COMMIT"

# AReaL itself (release checkout under test).
uv pip install --no-deps -e "$TARGET_ROOT" --system

# Runtime env (mirrors the image's ENV).
echo "PYTHONPATH=$WORKSPACE_DIR/MindSpeed:$WORKSPACE_DIR/Megatron-Bridge/src:${PYTHONPATH:-}" >> "$GITHUB_ENV"
echo "HCCL_IF_BASE_PORT=63000" >> "$GITHUB_ENV"
echo "HCCL_NPU_SOCKET_PORT_RANGE=62100-62350" >> "$GITHUB_ENV"
echo "TASK_QUEUE_ENABLE=1" >> "$GITHUB_ENV"
echo "OMP_NUM_THREADS=1" >> "$GITHUB_ENV"
# Keep wandb offline; mitigate NPU allocator fragmentation; disable optimized models.
echo "WANDB_MODE=disabled" >> "$GITHUB_ENV"
echo "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True" >> "$GITHUB_ENV"
echo "USE_OPTIMIZED_MODEL=0" >> "$GITHUB_ENV"

# The FSDP/Megatron engine loads safetensors from a LOCAL directory (it does
# not accept a HF repo id), so pre-download the model and export the path the
# manifest references as ${AREAL_MODEL_PATH}.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
python3 -m pip install -q "huggingface_hub<1.0"
export AREAL_MODEL_ID="$MODEL_ID"
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ["AREAL_MODEL_ID"]
dest = os.path.join(os.environ["GITHUB_WORKSPACE"], "areal_models", model_id.split("/")[-1])
snapshot_download(model_id, local_dir=dest)
with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as fh:
    fh.write(f"AREAL_MODEL_PATH={dest}\n")
print(f"AREAL_MODEL_PATH={dest}", flush=True)
PY
