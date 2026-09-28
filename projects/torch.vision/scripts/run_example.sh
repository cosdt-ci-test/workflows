#!/usr/bin/env bash
# Run the original upstream script and require an observable NPU result.
set -euo pipefail

: "${TARGET_ROOT:?TARGET_ROOT is required}"
: "${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
entry="${1:-}"
case "$entry" in
  gallery/transforms/plot_custom_transforms.py|\
  gallery/transforms/plot_custom_tv_tensors.py|\
  references/classification/train.py) ;;
  *) echo "unsupported TorchVision example entry: ${entry:-<missing>}" >&2; exit 2 ;;
esac

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
mkdir -p "$CI_OUTPUT_DIR"

python - "$entry" <<'PY'
import json
import os
from pathlib import Path
import runpy
import sys

import torch
import torch_npu
import torchvision

entry = sys.argv[1]
root = Path(os.environ.get("EXAMPLES_ROOT") or os.environ["TARGET_ROOT"]).resolve()
script = (root / entry).resolve()
if not script.is_relative_to(root) or not script.is_file():
    raise SystemExit(f"upstream example missing or outside checkout: {entry}")

overlay = json.loads(os.environ.get("OVERLAY_ARGS") or "[]")
if not isinstance(overlay, list) or not all(isinstance(arg, str) for arg in overlay):
    raise SystemExit("OVERLAY_ARGS must be a JSON string array")
if not torch.npu.is_available() or torch.npu.device_count() < 1:
    raise SystemExit("NPU unavailable; refusing CPU fallback")
torch.npu.set_device(0)

if entry.startswith("gallery/"):
    if overlay:
        raise SystemExit("gallery entry does not take CLI overlay arguments")
    torch.set_default_device("npu:0")
    sys.argv = [str(script)]
else:
    data_root = os.environ.get("VISION_DATA_ROOT")
    if not data_root or not Path(data_root, "train").is_dir() or not Path(data_root, "val").is_dir():
        raise SystemExit("five-class ImageFolder fixture missing")
    if "--device" not in overlay or "npu:0" not in overlay:
        raise SystemExit("classification entry requires --device npu:0")
    output = Path(os.environ["CI_OUTPUT_DIR"]) / "classification"
    output.mkdir(parents=True, exist_ok=True)
    sys.argv = [str(script), *overlay, "--data-path", data_root, "--output-dir", str(output)]

os.chdir(script.parent)
sys.path.insert(0, str(script.parent))
print(f"running unchanged upstream {entry} with torchvision {torchvision.__version__}")
namespace = runpy.run_path(str(script), run_name="__main__")

def require_npu(name):
    value = namespace.get(name)
    if not isinstance(value, torch.Tensor) or value.device.type != "npu":
        raise SystemExit(f"{entry}: {name} did not finish on NPU: {getattr(value, 'device', None)}")

if entry.endswith("plot_custom_transforms.py"):
    require_npu("out_img")
    require_npu("out_bboxes")
elif entry.endswith("plot_custom_tv_tensors.py"):
    # The v0.29 tutorial leaves the custom tensor and hflip result in
    # namespace variables; do not rely on an earlier release's examples.
    require_npu("my_dp")
    require_npu("wrapped")
    require_npu("wrapped_dog")
else:
    if not (output / "model_0.pth").is_file():
        raise SystemExit("classification entry did not finish an epoch and save a checkpoint")
    if torch.npu.max_memory_allocated() <= 0:
        raise SystemExit("classification entry did not allocate NPU memory")

torch.npu.synchronize()
print(f"NPU example passed: {entry}")
PY
