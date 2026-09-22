#!/usr/bin/env bash
# Prepare the CI environment for one supported AReaL example.
# $1 is the manifest profile. Unknown profiles fail before any install.
#
# AReaL's NPU support lives on the `ascend-v1.0.5` branch (not main), and the
# example is expected to run inside the dedicated NPU image
# ghcr.io/hwvanici/areal_npu:v1.0.5-a2 (CANN 9.0.1 / torch 2.10 / torch_npu
# 2.10 / vLLM-Ascend / Megatron-Core / MindSpeed / Megatron-Bridge). This
# script only installs AReaL itself and pre-downloads the model to a local dir.
#
# Contract: docs/guarding-examples.md "项目运行脚本契约".
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

SUPPORTED_PROFILES="areal-vlm-grpo"
case "$PROFILE" in
  areal-vlm-grpo) ;;
  *)
    echo "unknown profile: ${PROFILE} (supported: ${SUPPORTED_PROFILES})" >&2
    exit 1
    ;;
esac

if [[ -z "${TARGET_ROOT:-}" || -z "${GITHUB_WORKSPACE:-}" || -z "${GITHUB_ENV:-}" ]]; then
  echo "TARGET_ROOT / GITHUB_WORKSPACE / GITHUB_ENV must be set" >&2
  exit 2
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Install AReaL from the release checkout under test. The NPU image already
# carries every dependency (pyproject.npu.toml); install the package only.
python -m pip install --no-deps -e "$TARGET_ROOT"

# The Megatron/FSDP engine loads safetensors from a LOCAL directory (it does
# not accept a HF repo id), so pre-download the model and export the path the
# manifest references as ${AREAL_MODEL_PATH}.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MODEL_DIR="$GITHUB_WORKSPACE/areal_models/Qwen2.5-VL-3B-Instruct"
python -m pip install -q "huggingface_hub<1.0"
python - <<'PY'
import os
from huggingface_hub import snapshot_download

dest = os.path.join(os.environ["GITHUB_WORKSPACE"], "areal_models", "Qwen2.5-VL-3B-Instruct")
snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", local_dir=dest)
with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as fh:
    fh.write(f"AREAL_MODEL_PATH={dest}\n")
print(f"AREAL_MODEL_PATH={dest}", flush=True)
PY

# Keep wandb offline (AReaL defaults to wandb reporting).
echo "WANDB_MODE=disabled" >> "$GITHUB_ENV"
# expandable_segments mitigates NPU allocator fragmentation on 8 cards.
echo "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True" >> "$GITHUB_ENV"
# Some vLLM-Ascend optimized models are unsuitable for RLHF training.
echo "USE_OPTIMIZED_MODEL=0" >> "$GITHUB_ENV"
