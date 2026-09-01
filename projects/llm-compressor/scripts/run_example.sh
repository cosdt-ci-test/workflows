#!/usr/bin/env bash
# Run one llm-compressor example from a CI working copy of the target tree.
# Path selects the guard: NPU inference, CPU oneshot, or NPU generate-after-oneshot.
set -euo pipefail

export PYTHONNOUSERSITE=1

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"

case "$EXAMPLE_REL" in
  examples/compressed_inference/fp8_compressed_inference.py)
    KIND=npu_inference
    ;;
  examples/quantization_w8a8_fp8/llama3_example.py)
    KIND=cpu_oneshot
    ;;
  examples/quantization_w8a8_int8/gemma2_example.py|examples/autoround/quantization_wNa16/qwen3_example_custom_dataset.py)
    KIND=npu_oneshot
    ;;
  *)
    echo "unsupported llm-compressor guard path: $EXAMPLE_REL" >&2
    exit 1
    ;;
esac

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"
if [[ ! -f "$EXAMPLE_PATH" ]]; then
  echo "upstream example not found: $EXAMPLE_PATH" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR"
RUN_LOG="$CI_OUTPUT_DIR/run.log"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

run_cpu_oneshot() {
  unset ASCEND_RT_VISIBLE_DEVICES || true
  export CUDA_VISIBLE_DEVICES=""
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  # transformers 5.x compiles on first generate; 8B CPU compile
  # eats the 90-minute job before oneshot is even the bottleneck.
  export TORCHDYNAMO_DISABLE=1
  export TORCH_COMPILE_DISABLE=1
  python - "$EXAMPLE_PATH" <<'PY' 2>&1 | tee "$RUN_LOG"
import runpy
import sys
from pathlib import Path

import torch

accel = getattr(torch, 'accelerator', None)
if accel is not None:
    try:
        accel.get_memory_info()
    except Exception:
        accel.get_memory_info = lambda device=None: (32 * 1024**3, 32 * 1024**3)
    try:
        accel.max_memory_allocated()
    except Exception:
        accel.max_memory_allocated = lambda device=None: 0

path = sys.argv[1]
ns = runpy.run_path(path)
model = ns.get('model')
if model is None:
    raise SystemExit('oneshot example did not expose model')
param_devices = {str(param.device) for param in model.parameters()}
if any(device.startswith('npu') for device in param_devices):
    raise SystemExit(
        'cpu oneshot landed on NPU: '
        f'{sorted(param_devices)}'
    )
config = getattr(model, 'config', None)
qc = getattr(config, 'quantization_config', None) if config is not None else None
save_hits = list(Path.cwd().glob('*-FP8-Dynamic'))
if qc is None and not save_hits:
    raise SystemExit('oneshot finished without quantization_config or save dir')
print('LLM_COMPRESSOR_ONESHOT_DEVICE=cpu')
print('oneshot_param_devices', sorted(param_devices))
PY
  if ! grep -q 'LLM_COMPRESSOR_ONESHOT_DEVICE=cpu' "$RUN_LOG"; then
    echo "missing CPU oneshot anchor" >&2
    exit 1
  fi
}

run_npu_oneshot() {
  export PATH="/usr/local/sbin:$PATH"
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TORCHDYNAMO_DISABLE=1
  export TORCH_COMPILE_DISABLE=1
  python - "$EXAMPLE_PATH" <<'PY' 2>&1 | tee "$RUN_LOG"
import runpy
import sys

import llmcompressor
import torch
import torch_npu

accel = getattr(torch, 'accelerator', None)
if accel is not None:
    try:
        accel.get_memory_info()
    except Exception:
        accel.get_memory_info = lambda device=None: (32 * 1024**3, 32 * 1024**3)
    try:
        accel.max_memory_allocated()
    except Exception:
        accel.max_memory_allocated = lambda device=None: 0

# String dataset aliases without splits load the whole DatasetDict
# (perfectblend is 1.4M rows) and tokenize it in memory. That hits the
# 32GiB cgroup on this workspace before GPTQ starts. The example already
# asks for num_calibration_samples; pass the matching HF slice.
_oneshot = llmcompressor.oneshot


def oneshot(*args, **kwargs):
    dataset = kwargs.get('dataset')
    if isinstance(dataset, str) and not kwargs.get('splits'):
        count = kwargs.get('num_calibration_samples') or 512
        kwargs = dict(kwargs)
        kwargs['splits'] = f'train[:{count}]'
        print(f'LLM_COMPRESSOR_ONESHOT_SPLITS={kwargs["splits"]}')
    return _oneshot(*args, **kwargs)


llmcompressor.oneshot = oneshot

path = sys.argv[1]
ns = runpy.run_path(path)
model = ns.get('model')
if model is None:
    raise SystemExit('oneshot example did not expose model')
param_devices = {str(param.device) for param in model.parameters()}
if 'npu:0' not in param_devices:
    raise SystemExit(
        'workload not on npu:0 after oneshot/dispatch: '
        f'{sorted(param_devices)}'
    )
print('LLM_COMPRESSOR_WORKLOAD_DEVICE=npu:0')
print('oneshot_param_devices', sorted(param_devices))
PY
  if ! grep -q 'LLM_COMPRESSOR_WORKLOAD_DEVICE=npu:0' "$RUN_LOG"; then
    echo "missing NPU workload device anchor" >&2
    exit 1
  fi
}

run_npu_inference() {
  export PATH="/usr/local/sbin:$PATH"
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
  python - "$EXAMPLE_PATH" <<'PY' 2>&1 | tee "$RUN_LOG"
import runpy
import sys

import torch
import torch_npu

path = sys.argv[1]
ns = runpy.run_path(path)
model = ns['compressed_model']
inputs = ns['inputs']
param_devices = {str(param.device) for param in model.parameters()}
input_devices = set()
values = inputs.values() if hasattr(inputs, 'values') else [inputs]
for value in values:
    if hasattr(value, 'device'):
        input_devices.add(str(value.device))
if param_devices != {'npu:0'} or input_devices != {'npu:0'}:
    raise SystemExit(
        'workload not on npu:0: '
        f'model={sorted(param_devices)} inputs={sorted(input_devices)}'
    )
print('LLM_COMPRESSOR_WORKLOAD_DEVICE=npu:0')
PY
  if ! grep -q 'LLM_COMPRESSOR_WORKLOAD_DEVICE=npu:0' "$RUN_LOG"; then
    echo "missing NPU workload device anchor" >&2
    exit 1
  fi
}

case "$KIND" in
  cpu_oneshot)
    run_cpu_oneshot
    ;;
  npu_oneshot)
    run_npu_oneshot
    ;;
  npu_inference)
    run_npu_inference
    ;;
  *)
    echo "unknown kind: $KIND" >&2
    exit 1
    ;;
esac
