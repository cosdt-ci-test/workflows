#!/usr/bin/env bash
# Run the original upstream tutorial with NPU as the default tensor device.
set -euo pipefail

: "${TARGET_ROOT:?TARGET_ROOT is required}"
: "${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
entry="${1:-}"
case "$entry" in
  tutorials/sphinx_tuto/tensordict_keys.py|\
  tutorials/sphinx_tuto/tensordict_shapes.py|\
  tutorials/sphinx_tuto/tensordict_preallocation.py) ;;
  *) echo "unsupported TensorDict tutorial entry: ${entry:-<missing>}" >&2; exit 2 ;;
esac

if [[ "${OVERLAY_ARGS:-[]}" != '[]' ]]; then
  echo "TensorDict tutorials have no CLI overlay arguments" >&2
  exit 2
fi

examples_root="${EXAMPLES_ROOT:-$TARGET_ROOT}"
tutorial="$examples_root/$entry"
if [[ ! -f "$tutorial" ]]; then
  echo "tutorial not found: $tutorial" >&2
  exit 1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
mkdir -p "$CI_OUTPUT_DIR"

python - "$tutorial" <<'PY'
import runpy
import sys

import torch
import torch_npu
from tensordict import TensorDictBase

if not torch.npu.is_available() or torch.npu.device_count() < 1:
    raise SystemExit("NPU is unavailable; refusing CPU fallback")
torch.npu.set_device(0)
torch.set_default_device("npu:0")

# Run the unmodified upstream file, including its own assertions.
namespace = runpy.run_path(sys.argv[1], run_name="__main__")
result = namespace.get("tensordict")
if not isinstance(result, TensorDictBase):
    raise SystemExit("tutorial did not leave a TensorDict result for device verification")
leaves = list(result.values(include_nested=True, leaves_only=True))
if not leaves or not all(isinstance(leaf, torch.Tensor) for leaf in leaves):
    raise SystemExit("tutorial result has no tensor leaves")
bad = [str(leaf.device) for leaf in leaves if leaf.device.type != "npu"]
if bad:
    raise SystemExit(f"tutorial fell back from NPU: leaf devices={bad}")
torch.npu.synchronize()
print(f"NPU tutorial passed: {sys.argv[1]} ({len(leaves)} NPU tensor leaves)")
PY
