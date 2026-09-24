#!/usr/bin/env bash
# Prepare the CI environment for one supported AReaL example.
# $1 is the manifest profile. Unknown profiles fail before any install.
#
# CI base image is ghcr.io/hwvanici/areal_npu, which already has the full stack:
# CANN, torch, vLLM-Ascend, Megatron, MindSpeed, huggingface_hub etc.
# We should NOT reinstall these; just install the target AReaL source and assets.
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

# -------------------------------------------------------
# 1. Install the AReaL source under test.
# The official image ships its own copy of AReaL, but we need to use
# the checked-out TARGET_ROOT version because it may contain fixes.
# -------------------------------------------------------
command -v uv >/dev/null 2>&1 || pip install -q uv
uv pip install --no-deps -e "$TARGET_ROOT" --system

# -------------------------------------------------------
# 2. Runtime Environment setup.
# -------------------------------------------------------
echo "PYTHONPATH=/areal-workspace/MindSpeed:/areal-workspace/Megatron-Bridge/src:${PYTHONPATH:-}" >> "$GITHUB_ENV"
echo "HCCL_IF_BASE_PORT=63000" >> "$GITHUB_ENV"
echo "HCCL_NPU_SOCKET_PORT_RANGE=62100-62350" >> "$GITHUB_ENV"
echo "TASK_QUEUE_ENABLE=1" >> "$GITHUB_ENV"
echo "OMP_NUM_THREADS=1" >> "$GITHUB_ENV"
echo "WANDB_MODE=disabled" >> "$GITHUB_ENV"
echo "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True" >> "$GITHUB_ENV"
echo "USE_OPTIMIZED_MODEL=0" >> "$GITHUB_ENV"

# -------------------------------------------------------
# 3. Pre-download Model & Dataset (using image's native tools)
# -------------------------------------------------------
# Model and dataset are both fetched online (no local fixtures). Export the
# Hub mirror through GITHUB_ENV so the run-example step inherits it too: AReaL
# downloads the dataset at train time via `load_dataset` inside the data
# service, not here.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "HF_ENDPOINT=${HF_ENDPOINT}" >> "$GITHUB_ENV"
export AREAL_MODEL_ID="$MODEL_ID"

python3 <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ["AREAL_MODEL_ID"]
dest = os.path.join(os.environ["GITHUB_WORKSPACE"], "areal_models", model_id.split("/")[-1])
snapshot_download(model_id, local_dir=dest)
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"AREAL_MODEL_PATH={dest}\n")
print(f"Downloaded model to {dest}", flush=True)
PY
