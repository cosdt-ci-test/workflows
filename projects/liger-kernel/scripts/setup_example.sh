#!/usr/bin/env bash
# Prepare the CI environment for one supported Liger-Kernel example.
# $1 is the manifest profile. Unknown profiles fail before any install.
#
# Liger itself is installed from TARGET_ROOT - the release checkout the
# shared engine resolved - so the guarded kernel source and the examples
# are always the same release.
#
# Stack: the CANN 9.1.0 image already ships torch 2.9.0 + torch_npu 2.9.0;
# liger's own setup.py and Huawei's Ascend-CI recipe pin exactly that pair
# plus triton-ascend 3.2.2 (the Ascend Triton fork is what the _ascend
# backend compiles against). Never let pip resolve a PyPI torch over it.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
TRITON_ASCEND_INDEX=https://triton-ascend.osinfra.cn/pypi/simple
FALLBACK_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"

select_pip_index() {
  # Runners live in mainland China: prefer the cluster pip cache, fall
  # back to the Tsinghua mirror.
  if python -c "
import urllib.error
import urllib.request
try:
    urllib.request.urlopen('${CLUSTER_PIP_INDEX}', timeout=3)
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

verify_torch_stack() {
  # The image's torch/torch_npu pair is the only one torch_npu kernels are
  # built against; a reinstall would break the compiled op library.
  python - <<'PY'
import torch
import torch_npu

print("torch", torch.__version__, "torch_npu", torch_npu.__version__)
if not torch.__version__.startswith("2.9.0"):
    raise SystemExit(f"expected the CANN 9.1 image torch 2.9.0, got {torch.__version__}")
if not torch_npu.__version__.startswith("2.9.0"):
    raise SystemExit(f"expected torch_npu 2.9.0, got {torch_npu.__version__}")
if not torch.npu.is_available() or torch.npu.device_count() < 1:
    raise SystemExit("NPU unavailable; refusing to run examples on CPU")
print("NPU devices:", torch.npu.device_count())
PY
}

ensure_triton_ascend() {
  # The _ascend backend lowers kernels through the Ascend Triton fork; the
  # CUDA triton wheel must not shadow it. Not on default PyPI.
  if python -c "import triton; print('triton', triton.__version__)" 2>/dev/null \
      && python -c "import triton_ascend" 2>/dev/null; then
    echo "triton-ascend already present"
    return
  fi
  python -m pip uninstall -y triton 2>/dev/null || true
  python -m pip install "triton-ascend==3.2.2" \
    --extra-index-url "$TRITON_ASCEND_INDEX" \
    --trusted-host triton-ascend.osinfra.cn --no-cache-dir
}

install_liger_from_checkout() {
  # --no-deps: the container already carries torch/torch_npu, and liger's
  # runtime deps are installed explicitly per profile below.
  python -m pip install --no-deps -e "$TARGET_ROOT"
  python - <<'PY'
import os
from pathlib import Path

import liger_kernel

loaded = Path(liger_kernel.__file__).resolve()
target = Path(os.environ["TARGET_ROOT"]).resolve()
print("liger_kernel", liger_kernel.__version__, "source", loaded)
if not loaded.is_relative_to(target):
    raise SystemExit(f"liger_kernel is not loaded from the tested checkout: {loaded}")
PY
}

ms_download_models() {
  # ModelScope direct pull; the runner's persistent cache makes repeat runs
  # free. Env names are exported through GITHUB_ENV to the run step.
  python -m pip install -q "modelscope==1.37.0"
  TQDM_MININTERVAL="${TQDM_MININTERVAL:-15}" python - "$@" <<'PY'
import os
import sys

from modelscope import snapshot_download

cache = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
for pair in sys.argv[1:]:
    env_name, model_id = pair.split("=", 1)
    local = snapshot_download(model_id, cache_dir=cache)
    with open(os.environ["GITHUB_ENV"], "a") as handle:
        handle.write(f"{env_name}={local}\n")
    print(f"{env_name}={local}", flush=True)
PY
}

stage_fixtures() {
  # CI fixtures live in the workflows checkout; copy them under TARGET_ROOT
  # so the examples' local-path arguments resolve inside one tree.
  local src="${FIXTURE_DIR:?FIXTURE_DIR is required}"
  local dst="$TARGET_ROOT/fixtures"
  mkdir -p "$dst"
  cp "$src"/*.json "$src"/*.jsonl "$dst/"
  ls -la "$dst/"
}

# ----- profile: liger_hf_trainer (examples/huggingface/training.py) -----
# AutoLigerKernelForCausalLM monkey-patches Qwen2 with Liger kernels and
# trains through trl's SFTTrainer. The example itself is unchanged; only
# the model/dataset/step flags are overridden.
setup_liger_hf_trainer() {
  # Pin the transformers/trl pair the example's API surface needs:
  #   dtype= (not torch_dtype=) requires transformers>=4.56.0;
  #   SFTTrainer(max_seq_length=) was removed from trl 0.13.0 (moved into SFTConfig);
  #   DataCollatorForCompletionOnlyLM was removed in trl 0.20.0.
  # verifier: examples/huggingface/training.py byte-identical run on trl 0.12.1 + 4.57.1.
  python -m pip install "transformers==4.57.1" "trl==0.12.1" \
    "datasets>=3.0.0" "accelerate>=0.34" "sentencepiece" "pillow"
  python -c "import transformers, trl; print('transformers', transformers.__version__, 'trl', trl.__version__)"
  stage_fixtures
  # load_dataset(path) needs a DIRECTORY containing a train split file; a
  # bare .jsonl path raises FileNotFoundError. The staging dir gives that.
  mkdir -p "$TARGET_ROOT/fixtures/ci_alpaca_8"
  cp "$TARGET_ROOT/fixtures/ci_alpaca_8.jsonl" "$TARGET_ROOT/fixtures/ci_alpaca_8/train.jsonl"
  ms_download_models "LIGER_MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct"
  printf 'LIGER_DATASET_PATH=%s\n' "$TARGET_ROOT/fixtures/ci_alpaca_8" >> "$GITHUB_ENV"
}

# ----- profile: liger_medusa (examples/medusa/train.py) -----
# Medusa multi-head retraining on a frozen backbone; trains the heads with
# Liger's fused_linear_cross_entropy. Needs scikit-learn (train_test_split)
# and safetensors; the wrapper's nvidia-smi launcher is not used.
setup_liger_medusa() {
  # Same transformers/trl window as the HF profile: medusa's train.py calls
  # Trainer(tokenizer=...), which exists only before transformers 5.0.
  python -m pip install "transformers==4.57.1" "trl==0.12.1" \
    "datasets>=3.0.0" "accelerate>=0.34" "scikit-learn" "safetensors" "sentencepiece" "pillow"
  python -c "import transformers, sklearn, safetensors; print('transformers', transformers.__version__)"
  stage_fixtures
  ms_download_models "LIGER_MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct"
  printf 'LIGER_FIXTURE_JSON=%s\n' "$TARGET_ROOT/fixtures/ci_sharegpt_8.json" >> "$GITHUB_ENV"
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
FIXTURE_DIR="${FIXTURE_DIR:-$GITHUB_WORKSPACE/workflows/projects/liger-kernel/fixtures}"
export FIXTURE_DIR

source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip setuptools wheel
verify_torch_stack
ensure_triton_ascend
install_liger_from_checkout

"setup_${PROFILE}"
