#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"
# A profile selects the smallest dependency set needed by one example family;
# it is not a transformers feature flag. Keep unknown profiles failing before
# any package is installed so a manifest typo cannot run a partial setup.
case "$PROFILE" in
  generation)
    DEPS=(accelerate)
    ;;
  glue)
    DEPS=(accelerate datasets evaluate scikit-learn)
    ;;
  *)
    echo "unknown profile: $PROFILE (supported: generation glue)" >&2
    exit 1
    ;;
esac

: "${TARGET_ROOT:?TARGET_ROOT is required}"
: "${GITHUB_ENV:?GITHUB_ENV is required}"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_TRUSTED_HOST="repo.huaweicloud.com"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python -m pip install -U pip
# The Ascend image normally contains a compatible torch/torch_npu pair. Reuse
# it when possible because these wheels are large; install only on a missing
# or unusable image stack.
if python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.device_count())"; then
  echo "reusing image torch stack"
else
  python -m pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
    torch==2.9.0 torch_npu==2.9.0.post2
fi
# Install the current target checkout, while keeping dependency resolution
# explicit in this script so pip cannot replace the NPU torch stack.
python -m pip install -e "$TARGET_ROOT" --no-deps
python -m pip install "${DEPS[@]}"
