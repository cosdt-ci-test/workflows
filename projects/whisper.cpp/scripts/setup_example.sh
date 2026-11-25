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

# Cached models live on the shared persistent runner cache. The pool
# mounts one volume at
# ~/.cache/huggingface (docs/examples-guard-engine.md). The legacy
# default /root/.cache/cosdt-ci-test/whisper.cpp existed only via a
# container bind-mount the engine does not provide.
CACHE_ROOT="${WHISPER_CI_CACHE_ROOT:-${HF_HOME:-${HOME}/.cache/huggingface}/whisper.cpp}"

TINY_EN_DEST=${CACHE_ROOT}/ggml-tiny.en.bin
TINY_EN_SHA256=921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f
TINY_EN_URLS=(
  https://www.modelscope.cn/models/cjc1887415157/whisper.cpp/resolve/master/ggml-tiny.en.bin
  https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
)

PARAKEET_F16_DEST=${CACHE_ROOT}/ggml-parakeet-tdt-0.6b-v3-f16.bin
PARAKEET_F16_SHA256=833bffc9513b2cae867ee9e51633cfd11e4d51aaa5597c8ac02159385a2b426f
PARAKEET_F16_URLS=(
  https://www.modelscope.cn/models/ggml-org/parakeet-GGUF/resolve/master/ggml-parakeet-tdt-0.6b-v3-f16.bin
  https://hf-mirror.com/ggml-org/parakeet-GGUF/resolve/main/ggml-parakeet-tdt-0.6b-v3-f16.bin
)

VAD_DEST=${CACHE_ROOT}/ggml-silero-v6.2.0.bin
VAD_SHA256=2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987
VAD_URLS=(
  https://www.modelscope.cn/models/ggml-org/whisper-vad/resolve/master/ggml-silero-v6.2.0.bin
  https://hf-mirror.com/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin
)

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

# Resume-then-scratch GET into dest, trying each URL in order.
# Usage: fetch_pinned <dest> <sha256> <url> [url...]
# Caller holds the lock if needed.
fetch_pinned() {
  local dest="$1"
  local expected_sha="$2"
  shift 2
  local part="$dest.part"
  if file_complete "$dest" "$expected_sha"; then
    echo "reusing $dest"
    return
  fi
  if [[ -f "$dest" ]]; then
    echo "cached $dest is corrupt or the wrong object; re-downloading" >&2
    rm -f "$dest"
  fi
  local url
  for url in "$@"; do
    echo "fetching $dest from $url"
    if download_to_part "$url" "$part" -C - \
      && file_complete "$part" "$expected_sha"; then
      mv -f "$part" "$dest"
      return
    fi
    echo "resume from $url produced no valid file; downloading from scratch" >&2
    rm -f "$part"
    if download_to_part "$url" "$part" \
      && file_complete "$part" "$expected_sha"; then
      mv -f "$part" "$dest"
      return
    fi
    echo "failed to fetch a valid file from $url" >&2
    rm -f "$part"
  done
  echo "all download sources failed for $dest" >&2
  exit 1
}

fetch_ggml() {
  local dest="$1"
  local expected_sha="$2"
  local env_name="$3"
  shift 3
  mkdir -p "$(dirname "$dest")"
  local lock="$dest.lock"
  (
    flock 9
    fetch_pinned "$dest" "$expected_sha" "$@"
    echo "${env_name}=${dest}" >> "$GITHUB_ENV"
    echo "model ready: $dest"
  ) 9>"$lock"
}

fetch_tiny_en() {
  fetch_ggml "$TINY_EN_DEST" "$TINY_EN_SHA256" WHISPER_CI_MODEL \
    "${TINY_EN_URLS[@]}"
}

fetch_parakeet_f16() {
  fetch_ggml "$PARAKEET_F16_DEST" "$PARAKEET_F16_SHA256" PARAKEET_CI_MODEL \
    "${PARAKEET_F16_URLS[@]}"
}

fetch_vad() {
  fetch_ggml "$VAD_DEST" "$VAD_SHA256" WHISPER_CI_VAD_MODEL \
    "${VAD_URLS[@]}"
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
