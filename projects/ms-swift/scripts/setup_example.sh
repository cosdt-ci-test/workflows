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
ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"
FALLBACK_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

pip_ascend() {
  python -m pip install --extra-index-url "$ASCEND_PIP_INDEX" "$@"
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

# Install the wheel, then its deps except torch / CUDA / transformers.
# vllm 0.18 declares torch==2.10 and nvidia-* wheels.
install_dist_filtered() {
  local spec="$1"
  python -m pip install --no-deps --extra-index-url "$ASCEND_PIP_INDEX" "$spec"
  python - "$spec" <<'PY'
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, requires

spec = sys.argv[1]
dist_name = re.split(r"[<>=!~]", spec, maxsplit=1)[0].strip()
banned_exact = {
    "torch",
    "torchvision",
    "torchaudio",
    "torch-npu",
    "transformers",
}
banned_prefixes = ("nvidia-", "cuda-", "flashinfer")

def dist_requires(name):
    for candidate in (name, name.replace("-", "_"), name.replace("_", "-")):
        try:
            return requires(candidate) or []
        except PackageNotFoundError:
            continue
    raise SystemExit(f"installed dist not found: {name}")

def req_name(spec_line):
    return re.split(r"[<>=!~\[\s]", spec_line.strip(), maxsplit=1)[0].strip().lower().replace("_", "-")

kept = []
for req in dist_requires(dist_name):
    body, _, marker = req.partition(";")
    if "extra" in marker and "==" in marker:
        continue
    name = req_name(body)
    if name in banned_exact or any(name.startswith(prefix) for prefix in banned_prefixes):
        print(f"skip dep {req}", flush=True)
        continue
    kept.append(req)

if kept:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *kept])
PY
}

ensure_torch_stack() {
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

# framework.txt includes gradio / fastapi / uvicorn for web-ui and deploy.
install_ms_swift() {
  python -m pip install -e "$TARGET_ROOT" --no-deps
  python - "$TARGET_ROOT" <<'PY'
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
skip = {"gradio", "fastapi", "uvicorn"}
deps = []
for raw in (root / "requirements" / "framework.txt").read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or line.startswith("-"):
        continue
    name = re.split(r"[<>=!~;\[]", line, maxsplit=1)[0].strip().lower()
    if name in skip:
        print(f"skip {line}", flush=True)
        continue
    deps.append(line)
# requirements/npu.txt pins torchvision for torch 2.7.1; we use 0.24.0 for torch 2.9.0.
# decord has no aarch64 wheel or sdist.
deps.extend(["torchvision==0.24.0", "decorator", "qwen_vl_utils>=0.0.14"])
subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])
PY
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
  install_dist_filtered vllm==0.18.0
  install_dist_filtered vllm-ascend==0.18.0
  # vllm 0.18 requires transformers<5 and downgrades it. Qwen3.5 needs
  # transformers>=5.2. vllm-ascend 0.18 only requires >=4.57.4, so put
  # 5.2 back without letting pip uninstall vllm to satisfy the <5 cap.
  python -m pip install --no-deps "transformers>=5.2.0"
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

HERE=$(cd "$(dirname "$0")" && pwd)
export PIP_CONSTRAINT="$(cd "$HERE/.." && pwd)/constraints-npu.txt"

select_pip_index
python -m pip install -U pip
ensure_torch_stack
install_ms_swift

"setup_${PROFILE}"
