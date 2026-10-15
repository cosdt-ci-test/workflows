#!/usr/bin/env bash
# Upstream Decord 0.6.0 does not publish Linux aarch64 wheels.
set -euo pipefail
if python -c 'import decord; assert decord.__version__ == "0.6.0"' 2>/dev/null; then
    echo "Decord 0.6.0: reusing existing build"
    exit 0
fi
apt-get update -qq
apt-get install -y -qq build-essential cmake pkg-config ffmpeg \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavfilter-dev
build_root="$(mktemp -d /tmp/vbench-decord.XXXXXX)"
git clone --recursive --depth 1 --branch v0.6.0 https://github.com/dmlc/decord.git "$build_root/src"
cmake -S "$build_root/src" -B "$build_root/src/build" \
    -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_root/src/build" --parallel 4
# Decord's wheel builder finds ../build/libdecord.so and copies it into
# the wheel itself. Pre-copying it would trigger shutil.SameFileError.
python -m pip install --no-build-isolation --no-deps "$build_root/src/python"
python -c 'import decord; print("Decord", decord.__version__)'
