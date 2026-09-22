#!/usr/bin/env bash
# Prepare the CI environment for one supported llm-compressor example.
# $1 is the manifest profile. Unknown profiles fail before any install.
set -euo pipefail

export PYTHONNOUSERSITE=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=cpu_example_spec.sh
source "$SCRIPT_DIR/cpu_example_spec.sh"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
ASCEND_PIP_VARIANT=https://mirrors.huaweicloud.com/ascend/repos/pypi/variant
FALLBACK_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"

pip_ascend() {
  python -m pip install \
    --extra-index-url "$ASCEND_PIP_VARIANT" \
    --extra-index-url "$ASCEND_PIP_INDEX" \
    "$@"
}

select_pip_index() {
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

ensure_torch_npu_stack() {
  if python -c "
import torch, torch_npu
print('found torch', torch.__version__, 'torch_npu', torch_npu.__version__)
raise SystemExit(
    0 if torch.__version__.startswith('2.10.0')
    and torch_npu.__version__.startswith('2.10.0')
    else 1)
"; then
    echo "reusing torch 2.10 / torch_npu 2.10 stack"
    return
  fi
  echo "installing torch==2.10.0 torch_npu==2.10.0.post4 numpy pyyaml"
  pip_ascend torch==2.10.0 torch_npu==2.10.0.post4 numpy pyyaml
}

ensure_cpu_torch() {
  python -m pip uninstall -y torch_npu >/dev/null 2>&1 || true
  echo "installing CPU torch==2.10.0 numpy pyyaml"
  python -m pip install torch==2.10.0 numpy pyyaml
  python -c "
import torch
assert torch.__version__.startswith('2.10.0'), torch.__version__
try:
    import torch_npu
except ImportError:
    print('cpu torch', torch.__version__, 'torch_npu absent')
else:
    raise SystemExit('cpu profile must not import torch_npu')
"
}

install_llmcompressor() {
  # modelscope pinned to the last pre-hub-split line: >=1.38 depends on
  # the separate modelscope-hub package, and 1.40.1's loose ">=0.4.2"
  # floor breaks import when the pip mirror lags on hub 0.4.3. The
  # plant_hf_from_modelscope import path (modelscope.hub.snapshot_download)
  # exists in 1.37.0.
  python -m pip install "modelscope==1.37.0" huggingface_hub
  pip_ascend -e "$TARGET_ROOT" torch==2.10.0
}

prefetch_hf_model() {
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export HF_HUB_DISABLE_XET=1
  python - <<'PY'
import os

from huggingface_hub import snapshot_download

model_id = os.environ['PREFETCH_HF_ID']
try:
    snapshot_download(model_id)
except Exception as exc:
    print(f'mirror download failed ({exc}); retrying huggingface.co')
    os.environ.pop('HF_ENDPOINT', None)
    snapshot_download(model_id)
print(f'prefetched {model_id}')
PY
}

plant_hf_from_modelscope() {
  export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/modelscope}"
  python - <<'PY'
import hashlib
import os
from pathlib import Path

from huggingface_hub import try_to_load_from_cache
from modelscope.hub.snapshot_download import snapshot_download

ms_id = os.environ['PREFETCH_MS_ID']
hf_id = os.environ['PREFETCH_HF_ID']
repo_type = os.environ.get('PREFETCH_REPO_TYPE', 'model')
if repo_type not in ('model', 'dataset'):
    raise SystemExit(f'bad PREFETCH_REPO_TYPE {repo_type}')
src = snapshot_download(
    ms_id, repo_type=repo_type, ignore_file_pattern=['original/*']
)
hf_home = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface'))
prefix = 'models' if repo_type == 'model' else 'datasets'
repo_dir = hf_home / 'hub' / f"{prefix}--{hf_id.replace('/', '--')}"
# huggingface_hub only resolves refs that look like git commit SHAs.
snap_id = hashlib.sha1(f'{hf_id}|modelscope'.encode()).hexdigest()
snap_dir = repo_dir / 'snapshots' / snap_id
refs_dir = repo_dir / 'refs'
refs_dir.mkdir(parents=True, exist_ok=True)
# huggingface_hub compares this string to the snapshot folder name
# without stripping; a trailing newline makes the cache miss.
(refs_dir / 'main').write_text(snap_id)


def cache_marker(directory: Path):
    if repo_type == 'model':
        path = directory / 'config.json'
        return path if path.is_file() else None
    skipped = {'.gitattributes'}
    for item in sorted(directory.rglob('*')):
        if item.is_file() and item.name not in skipped and item.suffix != '.md':
            return item
    return None


if cache_marker(snap_dir) is not None:
    print(f'reusing planted {hf_id} at {snap_dir}')
else:
    snap_dir.mkdir(parents=True, exist_ok=True)
    src_path = Path(src)
    for item in src_path.rglob('*'):
        if not item.is_file():
            continue
        rel = item.relative_to(src_path)
        if rel.parts and rel.parts[0] == 'original':
            continue
        dest = snap_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            continue
        dest.symlink_to(item.resolve())
    print(f'planted {ms_id} -> {hf_id} at {snap_dir}')
marker = cache_marker(snap_dir)
if marker is None:
    raise SystemExit(f'planted cache has no files: {snap_dir}')
marker_name = marker.relative_to(snap_dir).as_posix()
cached = try_to_load_from_cache(
    hf_id,
    marker_name,
    cache_dir=str(hf_home / 'hub'),
    repo_type=repo_type,
)
if not cached:
    raise SystemExit(
        f'huggingface_hub cannot see planted {hf_id} '
        f'(refs/main={snap_id})'
    )
print(f'hub cache hit {hf_id} {marker_name} -> {cached}')
PY
}

prefetch_dataset() {
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export HF_HUB_DISABLE_XET=1
  local planted=0
  if [[ -n "${PREFETCH_MS_DATASET_ID:-}" ]]; then
    if PREFETCH_REPO_TYPE=dataset \
        PREFETCH_MS_ID="$PREFETCH_MS_DATASET_ID" \
        PREFETCH_HF_ID="$PREFETCH_DATASET_ID" \
        plant_hf_from_modelscope; then
      planted=1
    else
      echo "modelscope dataset plant failed for ${PREFETCH_MS_DATASET_ID}" >&2
    fi
    unset PREFETCH_REPO_TYPE
  fi
  PREFETCH_PLANTED="$planted" python - <<'PY'
import os

from datasets import load_dataset

dataset_id = os.environ['PREFETCH_DATASET_ID']
split = os.environ.get('PREFETCH_DATASET_SPLIT') or None
config = os.environ.get('PREFETCH_DATASET_CONFIG') or None
args = (dataset_id, config) if config else (dataset_id,)
kwargs = {'split': split} if split else {}


def load():
    load_dataset(*args, **kwargs)


loaded = False
if os.environ.get('PREFETCH_PLANTED') == '1':
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    try:
        load()
    except Exception as exc:
        print(f'planted dataset unreadable offline ({exc}); trying hub')
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('HF_DATASETS_OFFLINE', None)
    else:
        print(f'prefetched planted dataset {dataset_id} {split or "<all splits>"}')
        loaded = True
if not loaded:
    try:
        load()
    except Exception as exc:
        print(f'mirror dataset failed ({exc}); retrying huggingface.co')
        os.environ.pop('HF_ENDPOINT', None)
        load()
    print(f'prefetched dataset {dataset_id} {split or "<all splits>"}')
PY
}

prefetch_cpu_assets() {
  local path="${EXAMPLE_PATH:-examples/quantization_w8a8_fp8/llama3_example.py}"
  local spec ms_model hf_model ms_data hf_data split config extras
  spec=$(cpu_example_spec "$path") || {
    echo "cpu profile has no prefetch for ${path}" >&2
    exit 1
  }
  {
    read -r ms_model
    read -r hf_model
    read -r ms_data
    read -r hf_data
    read -r split
    read -r config
    read -r extras
  } <<< "$spec"
  if [[ "$extras" != "-" ]]; then
    # shellcheck disable=SC2086
    python -m pip install $extras torch==2.10.0
  fi
  if [[ "$ms_model" != "-" ]]; then
    PREFETCH_REPO_TYPE=model \
      PREFETCH_MS_ID="$ms_model" \
      PREFETCH_HF_ID="$hf_model" \
      plant_hf_from_modelscope
    unset PREFETCH_REPO_TYPE
  fi
  if [[ "$hf_data" != "-" ]]; then
    export PREFETCH_DATASET_ID="$hf_data"
    unset PREFETCH_DATASET_SPLIT PREFETCH_MS_DATASET_ID PREFETCH_DATASET_CONFIG
    if [[ "$split" != "-" ]]; then
      export PREFETCH_DATASET_SPLIT="$split"
    fi
    if [[ "$ms_data" != "-" ]]; then
      export PREFETCH_MS_DATASET_ID="$ms_data"
    fi
    if [[ "$config" != "-" ]]; then
      export PREFETCH_DATASET_CONFIG="$config"
    fi
    prefetch_dataset
    unset PREFETCH_DATASET_ID PREFETCH_DATASET_SPLIT PREFETCH_MS_DATASET_ID PREFETCH_DATASET_CONFIG
  fi
}

# from_pretrained(gated_or_planted_id) still hits the Hub for the real
# commit SHA unless offline. The planted snapshot name is not that SHA,
# so leave this until after every prefetch that needs the network.
emit_offline_hub() {
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    {
      echo 'HF_HUB_OFFLINE=1'
      echo 'TRANSFORMERS_OFFLINE=1'
      echo 'HF_DATASETS_OFFLINE=1'
    } >> "$GITHUB_ENV"
  fi
}

setup_npu_inference() {
  export PATH="/usr/local/sbin:$PATH"
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  select_pip_index
  python -m pip install -U pip setuptools wheel
  ensure_torch_npu_stack
  python -c "
import torch, torch_npu
assert torch.npu.is_available()
print(torch.__version__, torch_npu.__version__, torch.npu.device_count())
"
  install_llmcompressor
  python -c "
import llmcompressor
import torch
import torch_npu
assert torch.__version__.startswith('2.10.0'), torch.__version__
assert torch_npu.__version__.startswith('2.10.0'), torch_npu.__version__
print('llmcompressor', llmcompressor.__version__)
"
  PREFETCH_HF_ID=nm-testing/tinyllama-fp8-dynamic-compressed prefetch_hf_model
}

setup_cpu() {
  select_pip_index
  python -m pip install -U pip setuptools wheel
  ensure_cpu_torch
  python -m pip install "modelscope==1.37.0" huggingface_hub
  python -m pip install -e "$TARGET_ROOT" torch==2.10.0
  python -m pip uninstall -y torch_npu >/dev/null 2>&1 || true
  python -c "
import llmcompressor
import torch
assert torch.__version__.startswith('2.10.0'), torch.__version__
try:
    import torch_npu
except ImportError:
    pass
else:
    raise SystemExit('cpu profile must not import torch_npu after install')
print('llmcompressor', llmcompressor.__version__, 'cpu torch', torch.__version__)
"
  prefetch_cpu_assets
  emit_offline_hub
}

setup_npu_int8() {
  export PATH="/usr/local/sbin:$PATH"
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  select_pip_index
  python -m pip install -U pip setuptools wheel
  ensure_torch_npu_stack
  python -c "
import torch, torch_npu
assert torch.npu.is_available()
print(torch.__version__, torch_npu.__version__, torch.npu.device_count())
"
  install_llmcompressor
  python -c "
import llmcompressor
import torch
import torch_npu
assert torch.npu.is_available()
print('llmcompressor', llmcompressor.__version__)
"
  PREFETCH_MS_ID=LLM-Research/gemma-2-2b-it \
    PREFETCH_HF_ID=google/gemma-2-2b-it \
    plant_hf_from_modelscope
  PREFETCH_DATASET_ID=mlabonne/open-perfectblend \
    PREFETCH_DATASET_SPLIT=train[:512] \
    prefetch_dataset
  emit_offline_hub
}

setup_npu_autoround() {
  export PATH="/usr/local/sbin:$PATH"
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  select_pip_index
  python -m pip install -U pip setuptools wheel
  ensure_torch_npu_stack
  python -c "
import torch, torch_npu
assert torch.npu.is_available()
print(torch.__version__, torch_npu.__version__, torch.npu.device_count())
"
  install_llmcompressor
  python -c "
import auto_round
import llmcompressor
import torch
import torch_npu
assert torch.npu.is_available()
print('llmcompressor', llmcompressor.__version__, 'auto_round ok')
"
  PREFETCH_MS_ID=Qwen/Qwen3-8B \
    PREFETCH_HF_ID=Qwen/Qwen3-8B \
    plant_hf_from_modelscope
  PREFETCH_DATASET_ID=HuggingFaceH4/ultrachat_200k \
    PREFETCH_DATASET_SPLIT=train_sft \
    prefetch_dataset
  emit_offline_hub
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"

"setup_${PROFILE}"
