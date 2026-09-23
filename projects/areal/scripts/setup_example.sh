#!/usr/bin/env bash
# Prepare the CI environment for one supported AReaL example.
# $1 is the manifest profile. Unknown profiles fail before any install.
#
# AReaL's NPU support lives on the `ascend-v1.0.5` branch (not main). The CI base
# image is AReaL's official NPU image (ghcr.io/hwvanici/areal_npu:v1.0.5-a2),
# which already ships the full stack: CANN 9.0.1 / py3.11 / torch 2.10 /
# torch_npu 2.10.0.post2 / vLLM 0.23.0 + vLLM-Ascend releases/v0.23.0 /
# transformers 5.5.4 / Megatron-Core 0.16.1 / MindSpeed / Megatron-Bridge under
# /areal-workspace, plus all AReaL Python deps from pyproject.npu.toml. So this
# script only installs the AReaL source under test and pre-downloads the model.
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

# The image ships the whole stack (Megatron/MindSpeed/Bridge sources + Python
# deps) under /areal-workspace.
WORKSPACE_DIR=/areal-workspace

# Install the AReaL source under test (the image has a possibly-stale /AReaL).
command -v uv >/dev/null 2>&1 || python3 -m pip install -q uv
uv pip install --no-deps -e "$TARGET_ROOT" --system

# Runtime env (mirrors the image's ENV; written to GITHUB_ENV so the run step
# inherits it).
echo "PYTHONPATH=$WORKSPACE_DIR/MindSpeed:$WORKSPACE_DIR/Megatron-Bridge/src:${PYTHONPATH:-}" >> "$GITHUB_ENV"
echo "HCCL_IF_BASE_PORT=63000" >> "$GITHUB_ENV"
echo "HCCL_NPU_SOCKET_PORT_RANGE=62100-62350" >> "$GITHUB_ENV"
echo "TASK_QUEUE_ENABLE=1" >> "$GITHUB_ENV"
echo "OMP_NUM_THREADS=1" >> "$GITHUB_ENV"
# Keep wandb offline; mitigate NPU allocator fragmentation; disable optimized models.
echo "WANDB_MODE=disabled" >> "$GITHUB_ENV"
echo "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True" >> "$GITHUB_ENV"
echo "USE_OPTIMIZED_MODEL=0" >> "$GITHUB_ENV"

# The FSDP/Megatron engine loads safetensors from a LOCAL directory (it does not
# accept a HF repo id), so pre-download the model and export the path the
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
