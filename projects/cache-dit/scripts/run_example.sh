#!/bin/bash
# run_example.sh - Run a cache-dit example on NPU
# Called from workflow YAML's run-example job
# Positional argument: $1 is the example path relative to target repo root (e.g., examples/api/run_cache_refresh_flux.py)

set -euo pipefail

EXAMPLE_PATH="$1"
PROJECT_ROOT="$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"
TARGET_ROOT="${TARGET_ROOT:-/workspace/cache-dit}"
FIXTURE_DIR="${FIXTURE_DIR:-${PROJECT_ROOT}/fixtures}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:-${TARGET_ROOT}/output}"

# Environment variables from workflow
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Source CANN environment if available
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# Create output directory
mkdir -p "${CI_OUTPUT_DIR}"

# Determine how to run the example
RUN_CMD=""

# Check if EXEC is set (override script to run)
if [ -n "${EXEC:-}" ]; then
    RUN_CMD="${EXEC}"
else
    # Default: run the example Python script
    RUN_CMD="python3 "${EXAMPLE_PATH}""
fi

# Add overlay args if provided
OVERLAY_ARGS_STR=""
if [ -n "${OVERLAY_ARGS:-}" ]; then
    IFS=',' read -ra ARRAY <<< "${OVERLAY_ARGS}"
    for arg in "${ARRAY[@]}"; do
        OVERLAY_ARGS_STR="${OVERLAY_ARGS_STR} ${arg}"
    done
fi

# Execute the example
echo "Running example: ${EXAMPLE_PATH}"
echo "With overlay args: ${OVERLAY_ARGS_STR}"
echo "Using NPU devices: ${ASCEND_RT_VISIBLE_DEVICES}"

eval "${RUN_CMD} ${OVERLAY_ARGS_STR}"

# Exit with the result code from the example
exit $?