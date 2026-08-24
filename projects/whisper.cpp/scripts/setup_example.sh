#!/usr/bin/env bash
# Prepare the CI environment for one supported example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# EXEC comes from the workflow as an environment variable.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

TINY_EN_URL=https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
TINY_EN_DEST=/root/.cache/cosdt-ci-test/whisper.cpp/ggml-tiny.en.bin
TINY_EN_SHA256=921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f

PARAKEET_F16_URL=https://hf-mirror.com/ggml-org/parakeet-GGUF/resolve/main/ggml-parakeet-tdt-0.6b-v3-f16.bin
PARAKEET_F16_DEST=/root/.cache/cosdt-ci-test/whisper.cpp/ggml-parakeet-tdt-0.6b-v3-f16.bin
PARAKEET_F16_SHA256=833bffc9513b2cae867ee9e51633cfd11e4d51aaa5597c8ac02159385a2b426f

VAD_URL=https://hf-mirror.com/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin
VAD_DEST=/root/.cache/cosdt-ci-test/whisper.cpp/ggml-silero-v6.2.0.bin
VAD_SHA256=2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987

NODE_VERSION=v20.18.2
NODE_ARCH=linux-arm64
NODE_NAME="node-${NODE_VERSION}-${NODE_ARCH}"
NODE_DEST="/root/.cache/cosdt-ci-test/whisper.cpp/${NODE_NAME}"
NODE_TARBALL="${NODE_DEST}.tar.xz"
NODE_URL="https://mirrors.huaweicloud.com/nodejs/${NODE_VERSION}/${NODE_NAME}.tar.xz"
NODE_TARBALL_SHA256=5c1437aa16e7e6a2e0687a42c4d3f0a8f8a2039cda8880cb3be8cd983aeefb44
NPM_REGISTRY=https://repo.huaweicloud.com/repository/npm/

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    python3 -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"
  fi
}

file_complete() {
  local path="$1"
  local expected_sha="$2"
  [[ -f "$path" ]] || return 1
  [[ "$(file_sha256 "$path")" == "$expected_sha" ]]
}

source_cann() {
  export PATH="/usr/local/sbin:$PATH"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
}

assert_cann_lib() {
  local root="${1:-$TARGET_ROOT/build}"
  if [[ -z "$(find "$root" -name 'libggml-cann.*' -print -quit)" ]]; then
    echo "build produced no CANN backend library under $root" >&2
    exit 1
  fi
}

download_to_part() {
  local url="$1"
  local part="$2"
  shift 2
  curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 "$@" \
    "$url" -o "$part"
}

# Resume-then-scratch GET into dest. Caller holds the lock if needed.
fetch_pinned() {
  local url="$1"
  local dest="$2"
  local expected_sha="$3"
  local part="$dest.part"
  if file_complete "$dest" "$expected_sha"; then
    echo "reusing $dest"
    return
  fi
  if [[ -f "$dest" ]]; then
    echo "cached $dest is corrupt or the wrong object; re-downloading" >&2
    rm -f "$dest"
  fi
  if ! download_to_part "$url" "$part" -C - \
    || ! file_complete "$part" "$expected_sha"; then
    echo "resume produced no valid file at $part; downloading from scratch" >&2
    rm -f "$part"
    if ! download_to_part "$url" "$part"; then
      echo "failed to download from $url" >&2
      rm -f "$part"
      exit 1
    fi
  fi
  if ! file_complete "$part" "$expected_sha"; then
    echo "downloaded file failed checksum: $part" >&2
    rm -f "$part"
    exit 1
  fi
  mv -f "$part" "$dest"
}

fetch_ggml() {
  local url="$1"
  local dest="$2"
  local expected_sha="$3"
  local env_name="$4"
  mkdir -p "$(dirname "$dest")"
  local lock="$dest.lock"
  (
    flock 9
    fetch_pinned "$url" "$dest" "$expected_sha"
    echo "${env_name}=${dest}" >> "$GITHUB_ENV"
    echo "model ready: $dest"
  ) 9>"$lock"
}

fetch_tiny_en() {
  fetch_ggml "$TINY_EN_URL" "$TINY_EN_DEST" "$TINY_EN_SHA256" \
    WHISPER_CI_MODEL
}

fetch_parakeet_f16() {
  fetch_ggml "$PARAKEET_F16_URL" "$PARAKEET_F16_DEST" \
    "$PARAKEET_F16_SHA256" PARAKEET_CI_MODEL
}

fetch_vad() {
  fetch_ggml "$VAD_URL" "$VAD_DEST" "$VAD_SHA256" WHISPER_CI_VAD_MODEL
}

configure_cann_build() {
  source_cann
  cmake -S "$TARGET_ROOT" -B "$TARGET_ROOT/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CANN=on
}

build_exec_target() {
  cmake --build "$TARGET_ROOT/build" --target "$(basename "$EXEC")" -j "$(nproc)"
}

node_complete() {
  [[ -x "$NODE_DEST/bin/node" ]] || return 1
  [[ "$("$NODE_DEST/bin/node" --version)" == "$NODE_VERSION" ]]
}

fetch_node_tarball() {
  fetch_pinned "$NODE_URL" "$NODE_TARBALL" "$NODE_TARBALL_SHA256"
}

extract_node() {
  local unpack
  unpack="$(mktemp -d "${TMPDIR:-/tmp}/whisper-node.XXXXXX")"
  if ! tar -xJf "$NODE_TARBALL" -C "$unpack"; then
    echo "failed to extract $NODE_TARBALL" >&2
    rm -rf "$unpack" "$NODE_TARBALL"
    exit 1
  fi
  if [[ ! -x "$unpack/$NODE_NAME/bin/node" ]]; then
    echo "Node binary missing after extract: $unpack/$NODE_NAME/bin/node" >&2
    rm -rf "$unpack" "$NODE_TARBALL"
    exit 1
  fi
  if [[ "$("$unpack/$NODE_NAME/bin/node" --version)" != "$NODE_VERSION" ]]; then
    echo "extracted Node version is not $NODE_VERSION" >&2
    rm -rf "$unpack" "$NODE_TARBALL"
    exit 1
  fi
  rm -rf "$NODE_DEST"
  mv "$unpack/$NODE_NAME" "$NODE_DEST"
  rmdir "$unpack"
}

ensure_node() {
  if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    echo "using existing node $(command -v node)"
    return
  fi
  mkdir -p "$(dirname "$NODE_DEST")"
  (
    flock 9
    if node_complete; then
      echo "reusing $NODE_DEST/bin/node"
    else
      fetch_node_tarball
      extract_node
    fi
  ) 9>"${NODE_DEST}.lock"
  if ! node_complete; then
    echo "Node binary missing after extract: $NODE_DEST/bin/node" >&2
    exit 1
  fi
  export PATH="$NODE_DEST/bin:$PATH"
  echo "WHISPER_CI_NODE_BIN=$NODE_DEST/bin" >> "$GITHUB_ENV"
  if [[ -n "${GITHUB_PATH:-}" ]]; then
    echo "$NODE_DEST/bin" >> "$GITHUB_PATH"
  fi
  echo "using cached node $NODE_DEST/bin/node"
}

patch_addon_prints() {
  local index_js="$TARGET_ROOT/examples/addon.node/index.js"
  if [[ ! -f "$index_js" ]]; then
    echo "addon.node index.js missing: $index_js" >&2
    exit 1
  fi
  python3 - "$index_js" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = "  no_prints: true,"
new = "  no_prints: false,"
if old not in text:
    raise SystemExit(
        "working-copy patch failed: index.js has no 'no_prints: true,'")
path.write_text(text.replace(old, new, 1))
print(f"patched {path}: no_prints false so CANN logs stay visible")
PY
}

setup_cann() {
  configure_cann_build
  build_exec_target
  assert_cann_lib
  fetch_tiny_en
}

setup_host() {
  configure_cann_build
  build_exec_target
  assert_cann_lib
  case "${EXAMPLE_PATH:-}" in
    examples/quantize)
      fetch_tiny_en
      ;;
    examples/parakeet-quantize)
      fetch_parakeet_f16
      ;;
    *)
      echo "setup_host: unexpected EXAMPLE_PATH=${EXAMPLE_PATH:-}" >&2
      exit 1
      ;;
  esac
}

setup_vad() {
  configure_cann_build
  build_exec_target
  assert_cann_lib
  fetch_vad
}

setup_parakeet() {
  configure_cann_build
  build_exec_target
  assert_cann_lib
  fetch_parakeet_f16
}

setup_cmake-pkg() {
  source_cann
  local pkg_dir="$TARGET_ROOT/examples/test-cmake"
  local build_dir="$pkg_dir/whisper-build-install"
  local prefix="$pkg_dir/install"
  cmake -S "$TARGET_ROOT" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_EXAMPLES=OFF \
    -DWHISPER_BUILD_SERVER=OFF \
    -DGGML_CANN=on \
    -DCMAKE_INSTALL_PREFIX="$prefix"
  cmake --build "$build_dir" -j "$(nproc)"
  cmake --install "$build_dir"
  assert_cann_lib "$prefix"
  cmake -S "$pkg_dir" -B "$pkg_dir/build" \
    -DCMAKE_PREFIX_PATH="$prefix"
  cmake --build "$pkg_dir/build" -j "$(nproc)"
  local bin="$pkg_dir/build/test-cmake"
  if [[ ! -f "$bin" ]]; then
    echo "cmake-pkg binary not found: $bin" >&2
    exit 1
  fi
  local libdir
  libdir=$(find "$prefix" -name 'libwhisper.so*' -printf '%h\n' -quit)
  if [[ -z "$libdir" ]]; then
    echo "installed libwhisper.so missing under $prefix" >&2
    exit 1
  fi
  echo "LD_LIBRARY_PATH=${libdir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" >> "$GITHUB_ENV"
}

setup_node() {
  source_cann
  ensure_node
  fetch_tiny_en
  patch_addon_prints
  npm install --prefix "$TARGET_ROOT/examples/addon.node" \
    --registry "$NPM_REGISTRY"
  (
    cd "$TARGET_ROOT"
    npx --prefix "$TARGET_ROOT/examples/addon.node" cmake-js compile \
      -d "$TARGET_ROOT" \
      -T addon.node \
      -B Release \
      --dist-url https://mirrors.huaweicloud.com/nodejs \
      --CDGGML_CANN=on
  )
  if [[ ! -f "$TARGET_ROOT/build/Release/addon.node" ]]; then
    echo "cmake-js did not write $TARGET_ROOT/build/Release/addon.node" >&2
    exit 1
  fi
  assert_cann_lib "$TARGET_ROOT/build"
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"
EXEC="${EXEC:?EXEC is required}"

"setup_${PROFILE}"
