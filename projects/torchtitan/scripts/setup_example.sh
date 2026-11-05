#!/usr/bin/env bash
# Prepare the environment for the torchtitan examples.
#
# Usage: setup_example.sh <profile>
#
# Env (provided by the examples engine):
#   TARGET_ROOT      checkout of pytorch/torchtitan at the tested ref
#   GITHUB_WORKSPACE workspace root (writable)
#   GITHUB_ENV       file where exported env vars are persisted
#
# Profile:
#   torchtitan       llama3 debugmodel smoke (1-card / 2-card)
set -euo pipefail

[[ $# -lt 1 ]] && { echo "usage: $0 <profile>" >&2; exit 2; }
PROFILE="$1"

# The examples engine ships a pip cache; fall back to the public mirrors.
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
if curl -sm 5 -o /dev/null "http://${CLUSTER_PIP_HOST}/pypi/simple/"; then
    PIP_INDEX_URL="http://${CLUSTER_PIP_HOST}/pypi/simple"
    PIP_EXTRA_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
else
    PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
    PIP_EXTRA_INDEX_URL="https://repo.huaweicloud.com/ascend/repos/pypi"
fi

export PIP_INDEX_URL PIP_EXTRA_INDEX_URL
export PIP_TRUSTED_HOST="cache-service.nginx-pypi-cache.svc.cluster.local,mirrors.aliyun.com,repo.huaweicloud.com,pypi.org"

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"

# Keep the community CUDA wheels and the community `triton` package out of
# the environment: the community triton has CUDA-only backends and would
# shadow the Ascend fork's `triton/` tree.
export PIP_CONSTRAINT="$PROJECT_ROOT/constraints-npu.txt"

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

# Versions pinned to the stack this project is verified against:
#   CANN 9.2.0 (container image) / torch 2.14.0 / torch_npu 2.14.0
#   triton-ascend 3.6.0 (Ascend fork of the triton 3.6 baseline)
TORCH_VERSION="2.14.0"
TORCH_NPU_VERSION="2.14.0"
TRITON_ASCEND_VERSION="3.6.0"
MODELSCOPE_VERSION="1.40.2"

# aliyun hosts the aarch64 torch CPU wheels under pytorch-wheels/cpu.
TORCH_WHEEL_INDEX="https://mirrors.aliyun.com/pytorch-wheels/cpu"
# The Ascend Triton fork: 3.6.0 stable lives in the `test` channel.
TRITON_ASCEND_INDEX="https://repo.huaweicloud.com/ascend/repos/pypi/test"

supported_profiles() {
    declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/,""); print }' | paste -sd' ' -
}

setup_torchtitan() {
    echo "== setup: torchtitan profile =="
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    python -m pip install -U pip setuptools wheel

    # torch + torch_npu (aarch64 wheels).
    python -m pip install -f "$TORCH_WHEEL_INDEX" "torch==$TORCH_VERSION"
    python -m pip install --extra-index-url "$PIP_EXTRA_INDEX_URL" \
        "torch_npu==$TORCH_NPU_VERSION"

    # Ascend Triton fork. --no-deps keeps the community `triton` out of the
    # environment; pybind11 is the only skipped dependency the Ascend
    # backend imports at runtime (it compiles npu_utils.cpp on first use).
    python -m pip install --no-deps \
        --extra-index-url "$TRITON_ASCEND_INDEX" \
        "triton-ascend==$TRITON_ASCEND_VERSION"
    python -m pip install pybind11

    # modelscope: the tokenizer download demo in the quick start.
    python -m pip install "modelscope==$MODELSCOPE_VERSION"

    # torchtitan itself, editable, from the checkout under test.
    python -m pip install -e "$TARGET_ROOT"

    # Tokenizer assets for the size-tier recipes (qwen3_0_6b): those configs
    # point hf_assets_path at ./assets/hf/<Model> instead of the local
    # tests/assets/tokenizer the debugmodel recipes use.  The cluster cannot
    # reach huggingface.co, so the files come from modelscope.  They land in
    # the CI working copy (never committed) and the modelscope cache on the
    # runner keeps them warm across jobs.
    python - <<'PY'
import os
from modelscope import snapshot_download

ASSETS = os.path.join(os.environ["TARGET_ROOT"], "assets", "hf")
TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "config.json", "generation_config.json",
]
for repo, dest in [
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B"),
    ("Qwen/Qwen3-1.7B", "Qwen3-1.7B"),
    ("Qwen/Qwen3-8B", "Qwen3-8B"),
    ("Qwen/Qwen3-14B", "Qwen3-14B"),
    ("deepseek-ai/deepseek-moe-16b-base", "deepseek-moe-16b-base"),
]:
    target = os.path.join(ASSETS, dest)
    if os.path.isdir(target) and os.listdir(target):
        print(f"assets: {dest} already present")
        continue
    os.makedirs(target, exist_ok=True)
    snapshot_download(
        repo, local_dir=target, allow_patterns=TOKENIZER_FILES
    )
    print(f"assets: {dest} -> {sorted(os.listdir(target))}")
PY

    # Sanity: NPU visible, stack importable, override module registered.
    python - <<'PY'
import torch, torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available(), "devices:", torch.npu.device_count())
assert torch.npu.is_available(), "no NPU device visible"
PY

    PYTHONPATH="$PROJECT_ROOT/example${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
import npu_compat
from torchtitan.config.override import _REGISTRY
registered = sorted(_REGISTRY)
print("registered overrides:", registered)
assert registered == ["npu_compat.npu_attention"], registered
PY

    python -c "import torchtitan; print('torchtitan', torchtitan.__version__)"
}

# Unknown profile must fail before anything is installed.
if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
    echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
    exit 1
fi

"setup_${PROFILE}"
