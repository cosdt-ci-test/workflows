#!/usr/bin/env bash
# Run one ktransformers example from a CI working copy of the target tree.
# The live manifest has no supported path. Any invocation is a contract
# violation and must not exit 0.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"

echo "ktransformers example guard has no supported path; refusing ${EXAMPLE_REL}" >&2
exit 1
