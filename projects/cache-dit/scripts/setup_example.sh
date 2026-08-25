#!/bin/bash
# setup_example.sh - Prepare environment for cache-dit example based on profile
# Called from workflow YAML's setup_example.sh job
# Positional argument: $1 is the manifest profile (e.g., flux, wan2.2)

set -euo pipefail

PROFILE="$1"
TARGET_ROOT="${TARGET_ROOT:-/workspace/cache-dit}"

# Print supported profiles and exit if profile not recognized
supported_profiles="flux wan2.2"

if echo "$supported_profiles" | grep -qw "$PROFILE"; then
    echo "Profile '$PROFILE' is supported"
else
    echo "Unsupported profile: $PROFILE"
    echo "Supported profiles: $supported_profiles"
    exit 1
fi

# Install cache-dit and dependencies
echo "Installing cache-dit and dependencies..."
pip3 install -U cache-dit
pip3 install --no-deps torchvision==0.16.0
pip3 install einops sentencepiece accelerate

# Install diffusers for parallel support
pip3 install git+https://github.com/huggingface/diffusers.git # or >= 0.36.0

# Set NPU environment variables
export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES:-0}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Source CANN environment if available
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

echo "Environment setup complete for profile: $PROFILE"