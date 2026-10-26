#!/usr/bin/env bash
# Single-card llama3 debugmodel smoke on Ascend NPU.
#
# Runs torchtitan's own trainer entry point:
#   python -m torchtitan.train --module llama3 --config <config> ...
# The NPU-incompatible model components (complex64 RoPE cache, flex
# attention) are swapped through torchtitan's official override
# mechanism (--override.imports), so no upstream source file is touched.
#
# Env:
#   TARGET_ROOT     checkout of pytorch/torchtitan (required)
#   CI_OUTPUT_DIR   where --dump-folder points (optional)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"

# The override module lives next to this script (project side, outside the
# upstream checkout) and is imported by name via --override.imports.
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

# Serial inductor compilation: parallel compile workers spike host memory
# past the runner's cgroup limit and get OOM-killed mid-run.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

# torchtitan resolves the tokenizer from ./tests/assets/tokenizer, i.e.
# relative to the checkout root.
cd "$TARGET_ROOT"

exec torchrun \
    --nproc_per_node=1 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    -m torchtitan.train \
    "$@"
