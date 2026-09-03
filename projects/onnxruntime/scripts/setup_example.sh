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

ORT_CACHE_ROOT=/root/.cache/cosdt-ci-test/onnxruntime
CMAKE_MIRROR_DIR="$ORT_CACHE_ROOT/cmake-mirror"

source_cann() {
  export PATH="/usr/local/sbin:/usr/local/bin:$PATH"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
}

version_ge() {
  [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n1)" == "$1" ]]
}

# Ubuntu 22.04 defaults (cmake 3.22 / gcc 11) cannot compile aarch64 ORT
# (-march=armv8.2-a+bf16) or samples/cxx (cmake_minimum_required 3.28).
assert_toolchain() {
  local cmake_ver gcc_ver
  if command -v cmake >/dev/null 2>&1; then
    cmake_ver=$(cmake --version | awk 'NR==1 { print $3 }')
  else
    cmake_ver=0
  fi
  if ! version_ge "$cmake_ver" 3.28; then
    # Runners in mainland China cannot rely on pypi.org.
    # cmake 4 dropped compatibility that ORT's FetchContent still needs.
    python3 -m pip install --index-url https://repo.huaweicloud.com/repository/pypi/simple 'cmake>=3.28,<4'
    # ~/.local/bin is pip --user. VIRTUAL_ENV/bin is a coder uv venv.
    # Putting /usr/local/bin first would hide both behind Ubuntu 3.22.
    export PATH="${HOME}/.local/bin${VIRTUAL_ENV:+:${VIRTUAL_ENV}/bin}:$PATH"
    hash -r
    cmake_ver=$(cmake --version | awk 'NR==1 { print $3 }')
    if ! version_ge "$cmake_ver" 3.28; then
      echo "cmake >= 3.28 required, have ${cmake_ver:-missing}" >&2
      exit 1
    fi
  fi

  if command -v gcc >/dev/null 2>&1; then
    gcc_ver=$(gcc -dumpfullversion 2>/dev/null || gcc -dumpversion)
  else
    gcc_ver=0
  fi
  if ! version_ge "$gcc_ver" 12; then
    if ! command -v gcc-12 >/dev/null 2>&1 || ! command -v g++-12 >/dev/null 2>&1; then
      export DEBIAN_FRONTEND=noninteractive
      apt-get update
      apt-get install -y gcc-12 g++-12
    fi
    export CC=gcc-12 CXX=g++-12
    echo "CC=gcc-12" >> "$GITHUB_ENV"
    echo "CXX=g++-12" >> "$GITHUB_ENV"
  fi
}

assert_cann_provider_so() {
  local root="${1:-$TARGET_ROOT}"
  if [[ -z "$(find "$root" -name 'libonnxruntime_providers_cann.so' -print -quit)" ]]; then
    echo "libonnxruntime_providers_cann.so missing under $root (cmake can succeed without --use_cann)" >&2
    exit 1
  fi
}

ort_cache_complete() {
  local dir="$1"
  [[ -x "$dir/bin/onnxruntime_provider_test" ]] || return 1
  [[ -e "$dir/bin/libonnxruntime_providers_cann.so" ]] || return 1
  [[ -e "$dir/bin/libonnxruntime_providers_shared.so" ]] || return 1
  compgen -G "$dir/bin/libonnxruntime.so*" >/dev/null || return 1
  [[ -d "$dir/install/include" && -d "$dir/install/lib" ]] || return 1
  [[ -e "$dir/install/lib/libonnxruntime_providers_cann.so" ]] || return 1
}

copy_so_if_absent() {
  local name="$1"
  local dest="$2"
  shift 2
  mkdir -p "$dest"
  if [[ -e "$dest/$name" ]]; then
    return
  fi
  local found
  found=$(find "$@" -name "$name" -print -quit)
  if [[ -z "$found" ]]; then
    echo "$name missing after --use_cann build" >&2
    exit 1
  fi
  cp -a "$found" "$dest/"
}

copy_libonnxruntime_so() {
  local dest="$1"
  shift
  mkdir -p "$dest"
  if compgen -G "$dest/libonnxruntime.so*" >/dev/null; then
    return
  fi
  local so
  so=$(find "$@" -name 'libonnxruntime.so*' -print -quit)
  if [[ -z "$so" ]]; then
    echo "libonnxruntime.so missing after build" >&2
    exit 1
  fi
  cp -a "$(dirname "$so")"/libonnxruntime.so* "$dest/"
}

stage_ort_products() {
  local src_build="$1"
  local part="$2"
  local prefix="$part/install"
  mkdir -p "$part/bin" "$prefix"
  cmake --install "$src_build" --prefix "$prefix"
  if [[ ! -x "$src_build/onnxruntime_provider_test" ]]; then
    echo "onnxruntime_provider_test missing after build: $src_build/onnxruntime_provider_test" >&2
    exit 1
  fi
  cp -a "$src_build/onnxruntime_provider_test" "$part/bin/"
  copy_so_if_absent libonnxruntime_providers_cann.so "$part/bin" "$src_build" "$prefix"
  copy_so_if_absent libonnxruntime_providers_shared.so "$part/bin" "$src_build" "$prefix"
  copy_so_if_absent libonnxruntime_providers_cann.so "$prefix/lib" "$src_build" "$prefix"
  copy_so_if_absent libonnxruntime_providers_shared.so "$prefix/lib" "$src_build" "$prefix"
  copy_libonnxruntime_so "$part/bin" "$src_build" "$prefix"
  copy_libonnxruntime_so "$prefix/lib" "$src_build" "$prefix"
  assert_cann_provider_so "$part"
}

# --skip_tests skips running tests, not compiling the gtest binaries.
compile_ort() {
  local part="$1"
  local prefix="$part/install"
  mkdir -p "$prefix"
  local -a cmd=(
    ./build.sh
    --config Release
    --build_shared_lib
    --use_cann
    --parallel 8
    --skip_tests
    --skip_submodule_sync
    --compile_no_warning_as_error
    --allow_running_as_root
    --cmake_extra_defines "CMAKE_INSTALL_PREFIX=${prefix}"
  )
  if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    cmd+=(--cann_home /usr/local/Ascend/ascend-toolkit/latest)
  fi
  if [[ -d "$CMAKE_MIRROR_DIR" ]]; then
    cmd+=(--cmake_deps_mirror_dir "$CMAKE_MIRROR_DIR")
  fi
  (
    cd "$TARGET_ROOT"
    "${cmd[@]}"
  )
  stage_ort_products "$TARGET_ROOT/build/Linux/Release" "$part"
}

file_sha1() {
  sha1sum "$1" | awk '{print $1}'
}

# github.com / codeload.github.com: org proxy first, then direct.
# Other https hosts: direct only. Keep going until SHA1 matches.
download_cmake_dep() {
  local url="$1"
  local part="$2"
  local expected_sha="$3"
  local -a sources=()
  case "$url" in
    https://github.com/*|https://codeload.github.com/*)
      sources+=("https://gh-proxy.test.osinfra.cn/${url}" "$url")
      ;;
    *)
      sources+=("$url")
      ;;
  esac
  local src
  for src in "${sources[@]}"; do
    rm -f "$part"
    echo "cmake-mirror fetching $src"
    if curl -fL --http1.1 --retry 5 --retry-delay 3 --connect-timeout 30 \
      -C - -o "$part" "$src" \
      && [[ "$(file_sha1 "$part")" == "$expected_sha" ]]; then
      return 0
    fi
    echo "cmake-mirror download or checksum failed: $src" >&2
  done
  return 1
}

# Layout matches ORT --cmake_deps_mirror_dir: <mirror>/<url with https:// stripped>.
# A missing file is a warning; cmake falls back to the network.
populate_cmake_mirror() {
  local deps="$TARGET_ROOT/cmake/deps.txt"
  if [[ ! -f "$deps" ]]; then
    echo "cmake/deps.txt missing at $deps; cmake will fetch online" >&2
    return 0
  fi
  mkdir -p "$CMAKE_MIRROR_DIR"
  local line name rest url sha1 dest part
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    name="${line%%;*}"
    rest="${line#*;}"
    url="${rest%%;*}"
    sha1="${rest#*;}"
    sha1="${sha1%%;*}"
    sha1="${sha1//[$'\t\r\n ']/}"
    [[ "$url" == https://* ]] || continue
    case "$url" in
      https://www.nuget.org/*)
        echo "cmake-mirror skip nuget dep ${name}"
        continue
        ;;
    esac
    dest="$CMAKE_MIRROR_DIR/${url#https://}"
    if [[ -f "$dest" ]] && [[ "$(file_sha1 "$dest")" == "$sha1" ]]; then
      echo "cmake-mirror hit $dest"
      continue
    fi
    if [[ -f "$dest" ]]; then
      echo "cmake-mirror corrupt $dest; re-downloading" >&2
      rm -f "$dest"
    fi
    mkdir -p "$(dirname "$dest")"
    part="$dest.part"
    if download_cmake_dep "$url" "$part" "$sha1"; then
      mv -f "$part" "$dest"
      echo "cmake-mirror stored $name -> $dest"
    else
      echo "warning: failed to populate cmake-mirror for ${name} ($url); cmake will fetch online" >&2
      rm -f "$part"
    fi
  done < "$deps"
}

restore_ort_products() {
  local cache_dir="$1"
  local dest="$TARGET_ROOT/build/Linux/Release"
  mkdir -p "$dest"
  local f
  for f in "$cache_dir/bin"/*; do
    ln -sfn "$f" "$dest/$(basename "$f")"
  done
  ln -sfn "$cache_dir/install" "$TARGET_ROOT/.ort-install"
}

build_or_restore_ort() {
  source_cann
  local sha
  sha=$(git -C "$TARGET_ROOT" rev-parse HEAD)
  local cache_dir="$ORT_CACHE_ROOT/$sha"
  local part_dir="$ORT_CACHE_ROOT/${sha}.part"
  local lock="$ORT_CACHE_ROOT/${sha}.lock"
  mkdir -p "$ORT_CACHE_ROOT"
  (
    flock 9
    if ort_cache_complete "$cache_dir"; then
      echo "reusing ORT cache $cache_dir"
    else
      rm -rf "$part_dir"
      mkdir -p "$part_dir"
      populate_cmake_mirror
      compile_ort "$part_dir"
      rm -rf "$cache_dir"
      mv "$part_dir" "$cache_dir"
    fi
    restore_ort_products "$cache_dir"
  ) 9>"$lock"
}

require_exec() {
  if [[ ! -e "$TARGET_ROOT/$EXEC" ]]; then
    echo "exec not found: $TARGET_ROOT/$EXEC" >&2
    exit 1
  fi
}

setup_cann-gtest() {
  assert_toolchain
  build_or_restore_ort
  require_exec
  assert_cann_provider_so "$TARGET_ROOT"
}

setup_cmake-consumer() {
  assert_toolchain
  build_or_restore_ort
  local prefix="$TARGET_ROOT/.ort-install"
  local header_dir="$prefix/include/onnxruntime"
  if [[ ! -f "$header_dir/onnxruntime_cxx_api.h" ]]; then
    echo "flattened ORT headers missing at $header_dir (not $prefix/include)" >&2
    exit 1
  fi
  cmake -S "$TARGET_ROOT/samples/cxx" -B "$TARGET_ROOT/samples/cxx/build" \
    -DORT_HEADER_DIR="$header_dir" \
    -DORT_LIBRARY_DIR="$prefix/lib"
  cmake --build "$TARGET_ROOT/samples/cxx/build" --target onnxruntime_sample_program
  require_exec
  # run_example.sh cds to TARGET_ROOT. The sample default model path is
  # cwd-relative add_model.onnx, which is committed under samples/cxx/.
  if [[ ! -f "$TARGET_ROOT/add_model.onnx" ]]; then
    if [[ ! -f "$TARGET_ROOT/samples/cxx/add_model.onnx" ]]; then
      echo "upstream sample model missing: $TARGET_ROOT/samples/cxx/add_model.onnx" >&2
      exit 1
    fi
    cp -a "$TARGET_ROOT/samples/cxx/add_model.onnx" "$TARGET_ROOT/add_model.onnx"
  fi
  echo "LD_LIBRARY_PATH=${prefix}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" >> "$GITHUB_ENV"
  export LD_LIBRARY_PATH="${prefix}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
