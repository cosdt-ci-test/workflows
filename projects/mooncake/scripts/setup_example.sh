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
SUPPORTED_PROFILES="ascend-direct"

if [[ "$PROFILE" != "ascend-direct" ]]; then
  echo "unknown profile: $PROFILE (supported: $SUPPORTED_PROFILES)" >&2
  exit 2
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
EXEC_REL="${EXEC:?EXEC is required}"

if [[ ! -f /etc/hccn.conf ]]; then
  echo "setup: /etc/hccn.conf is missing" >&2
  echo "Ascend Direct (HIXL) reads device NIC IPs from this file." >&2
  echo "The NPU driver writes it on the host. In a container, bind-mount the host file." >&2
  exit 1
fi

DEPS=(
  build-essential
  cmake
  git
  pkg-config
  libgoogle-glog-dev
  libgflags-dev
  libibverbs-dev
  libjsoncpp-dev
  libnuma-dev
  libyaml-cpp-dev
  libssl-dev
  libcurl4-openssl-dev
)

source_cann() {
  export PATH="/usr/local/sbin:/usr/local/bin:$PATH"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
}

missing_debs() {
  local pkg
  for pkg in "${DEPS[@]}"; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
      printf '%s\n' "$pkg"
    fi
  done
}

install_debs() {
  local missing
  missing=$(missing_debs)
  if [[ -z "$missing" ]]; then
    echo "setup: build packages already installed"
    return
  fi
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  # shellcheck disable=SC2086
  apt-get install -y --no-install-recommends $missing
}

init_pybind11() {
  if [[ -f "$TARGET_ROOT/extern/pybind11/CMakeLists.txt" ]]; then
    echo "setup: pybind11 submodule already present"
    return
  fi
  if git -C "$TARGET_ROOT" submodule update --init --depth 1 extern/pybind11; then
    echo "setup: initialized extern/pybind11"
    return
  fi
  echo "setup: submodule update failed; cloning pybind11 via ghfast.top" >&2
  local expect
  expect=$(git -C "$TARGET_ROOT" ls-tree HEAD extern/pybind11 | awk '{print $3}')
  if [[ ! "$expect" =~ ^[0-9a-f]{40}$ ]]; then
    echo "setup: cannot read pybind11 gitlink SHA from target tree" >&2
    exit 1
  fi
  rm -rf "$TARGET_ROOT/extern/pybind11"
  mkdir -p "$TARGET_ROOT/extern/pybind11"
  git -C "$TARGET_ROOT/extern/pybind11" init
  git -C "$TARGET_ROOT/extern/pybind11" remote add origin \
    https://ghfast.top/https://github.com/pybind/pybind11.git
  git -C "$TARGET_ROOT/extern/pybind11" fetch --depth 1 origin "$expect"
  git -C "$TARGET_ROOT/extern/pybind11" checkout --detach FETCH_HEAD
  local got
  got=$(git -C "$TARGET_ROOT/extern/pybind11" rev-parse HEAD)
  if [[ "$got" != "$expect" ]]; then
    echo "setup: pybind11 SHA mismatch: got $got want $expect" >&2
    exit 1
  fi
  echo "setup: cloned extern/pybind11 @$got via ghfast.top"
}

configure_and_build() {
  source_cann
  cmake -S "$TARGET_ROOT" -B "$TARGET_ROOT/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_ASCEND_DIRECT=ON \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_UNIT_TESTS=OFF \
    -DWITH_STORE=OFF \
    -DWITH_STORE_RUST=OFF \
    -DWITH_EP=OFF \
    -DWITH_P2P_STORE=OFF \
    -DUSE_ETCD=OFF \
    -DUSE_REDIS=OFF
  cmake --build "$TARGET_ROOT/build" \
    --target "$(basename "$EXEC_REL")" \
    -j "$(nproc)"
}

assert_exec() {
  local path="$TARGET_ROOT/$EXEC_REL"
  if [[ ! -x "$path" ]]; then
    echo "expected executable missing: $path" >&2
    find "$TARGET_ROOT/build" -name "$(basename "$EXEC_REL")" -print >&2 || true
    exit 1
  fi
  echo "setup: built $path"
}

install_debs
init_pybind11
configure_and_build
assert_exec
