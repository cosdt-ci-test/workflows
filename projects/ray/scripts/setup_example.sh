#!/usr/bin/env bash
# Install the Ray wheel built from the exact upstream commit under test.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"

source_cann() {
  export PATH="/usr/local/sbin:$PATH"
  # The selected CANN container provides this file.
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
}

ray_version() {
  python - "$TARGET_ROOT/python/ray/_version.py" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r"""^version = ['\"]([^'\"]+)['\"]""", text, re.MULTILINE)
if not match:
    raise SystemExit(f"could not read Ray version from {sys.argv[1]}")
print(match.group(1))
PY
}

install_target_ray() {
  local extras="$1"
  local sha version py_tag arch wheel_url requirement
  sha=$(git -C "$TARGET_ROOT" rev-parse HEAD)
  version=$(ray_version)
  py_tag=$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
  arch=$(uname -m)
  if [[ "$arch" != "aarch64" && "$arch" != "arm64" ]]; then
    echo "Ray Ascend guard requires aarch64, got: $arch" >&2
    exit 1
  fi

  wheel_url="https://s3-us-west-2.amazonaws.com/ray-wheels/master/${sha}/ray-${version}-${py_tag}-${py_tag}-manylinux2014_aarch64.whl"
  if [[ -n "$extras" ]]; then
    requirement="ray[${extras}] @ ${wheel_url}"
  else
    requirement="ray @ ${wheel_url}"
  fi

  echo "installing Ray ${version} from target commit ${sha}"
  if ! python -m pip install "$requirement"; then
    if [[ "$version" == *dev* ]]; then
      echo "exact master wheel is unavailable: $wheel_url" >&2
      exit 1
    fi
    echo "falling back to the released aarch64 wheel for Ray ${version}"
    if [[ -n "$extras" ]]; then
      python -m pip install "ray[${extras}]==${version}"
    else
      python -m pip install "ray==${version}"
    fi
  fi
  python -m pip install pytest mock
  python -c 'import ray; print("ray", ray.__version__, ray.__file__)'
}

setup_core() {
  install_target_ray ""
}

setup_train() {
  source_cann
  python -c 'import torch, torch_npu; assert torch.npu.is_available(); print(torch.__version__, torch_npu.__version__)'
  install_target_ray train
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

source_cann
python -m pip install -U pip
"setup_${PROFILE}"
