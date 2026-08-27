#!/usr/bin/env bash
# If 127.0.0.1:6060 is free, run the remaining command as-is.
# If it is taken (coder agent pprof), run inside a user+mount namespace
# where /etc/hosts maps localhost to 127.0.0.2. aibrix v0.7.0 gateway-plugins
# hard-codes ListenAndServe("localhost:6060"). Envoy still talks to 127.0.0.1.
# This wrapper is CI/coder only. It is not part of the user-facing doc.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

if python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 6060))' >/dev/null 2>&1; then
  exec "$@"
fi

hosts="$(mktemp)"
printf '127.0.0.2\tlocalhost\n127.0.0.1\t%s\n' "$(hostname)" > "$hosts"
exec unshare --user --map-root-user --mount --fork -- \
  bash -c 'mount --bind "$1" /etc/hosts; shift; exec "$@"' \
  bash "$hosts" "$@"
