#!/bin/bash
# setup_example.sh - Prepare environment for cache-dit example based on profile
# Called from workflow YAML's setup_example.sh job
# Positional argument: $1 is the manifest profile (e.g., flux, wan2.2)

set -euo pipefail

PROFILE="$1"
TARGET_ROOT="${TARGET_ROOT:-/workspace/cache-dit}"

# Print supported profiles and exit if profile not recognized
supported_profiles="flux wan2.2"

if echo "$supported_profiles" | grep -qw "$PROFILE"; then
    echo "Profile '$PROFILE' is supported"
else
    echo "Unsupported profile: $PROFILE"
    echo "Supported profiles: $supported_profiles"
    exit 1
fi

# Install cache-dit and dependencies
echo "Installing cache-dit and dependencies..."

# CANN 9.1.0 base image ships without torch/torch_npu; install the
# official pairing (same torch line as the diffusers project).
ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi

pip_ascend() {
  python3 -m pip install --extra-index-url "$ASCEND_PIP_INDEX" "$@"
}

ensure_torch_stack() {
  if python3 -c "
import torch, torch_npu
raise SystemExit(
    0 if torch.__version__.startswith('2.9.0')
    and torch_npu.__version__.startswith('2.9.0') else 1)
"; then
    echo "reusing image torch stack ($(python3 -c 'import torch; print(torch.__version__)'))"
    return
  fi
  echo "installing torch==2.9.0 torch_npu==2.9.0.post2"
  pip_ascend torch==2.9.0 torch_npu==2.9.0.post2
}

ensure_torch_stack
pip3 install --no-deps torchvision==0.24.0  # torch 2.9.0 pairing

pip3 install -U cache-dit
pip3 install einops sentencepiece accelerate

# Install diffusers for parallel support
pip3 install -U diffusers  # 要求 >= 0.36.0（PyPI latest，避免走 github 代理）

# modelscope pinned to 1.37.0 (hub split started at 1.38; same rationale
# as projects/diffusers/scripts/setup_example.sh)
python3 -m pip install -q "modelscope==1.37.0"

# Set NPU environment variables
export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES:-0}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Source CANN environment if available
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# ensure_flux_model: download FLUX.1-dev (diffusers layout) via ModelScope
# mirror so the generate.py example never touches HF gated endpoints.
ensure_flux_model() {
  local model_dir="/root/.cache/modelscope/FLUX.1-dev"
  if [ -d "$model_dir" ] && [ -n "$(ls -A "$model_dir" 2>/dev/null)" ]; then
    echo "model already cached at $model_dir; skipping download"
    return
  fi
  echo "downloading FLUX.1-dev via modelscope to $model_dir"
  python3 -c "from modelscope import snapshot_download; snapshot_download('AI-ModelScope/FLUX.1-dev', local_dir='$model_dir')"
}

setup_flux() {
  echo "profile=flux: ensuring FLUX.1-dev model"
  ensure_flux_model
}

setup_flux

echo "Environment setup complete for profile: $PROFILE"