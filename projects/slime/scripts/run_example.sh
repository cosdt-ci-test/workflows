#!/usr/bin/env bash
# Run one supported slime example.
#
# $1 is the manifest entry path. The upstream launchers hardcode paths,
# models and GPU layouts without "$@" passthrough, so the CI train
# recipe lives in the manifest overlay_args (mirroring the fork's NPU
# CI tests) and this script only maps the per-entry engine-call metadata
# that the engine cannot pass through (megatron model type / train
# script), then invokes the fork's own execute_train() helper. Never
# git add/commit/push here.
set -euo pipefail

validate_agent_rollouts() {
  "$PYTHON" - "$@" <<'PY'
"""Validate native Slime rollout/gradient dumps without intercepting generation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import math
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_sample(sample):
    require(isinstance(sample, dict), 'native samples must be dictionaries')
    tokens, length = sample.get('tokens'), sample.get('response_length')
    require(isinstance(tokens, list) and tokens and all(isinstance(x, int) for x in tokens), 'missing token trajectory')
    require(isinstance(length, int) and 0 < length < len(tokens), 'invalid response/prompt token lengths')
    require(isinstance(sample.get('response'), str) and sample['response'].strip(), 'empty response')
    require(sample.get('status') in ('completed', 'truncated'), 'failed/aborted agent trajectory')
    require(finite_number(sample.get('reward')), 'missing/nonfinite reward')
    probs = sample.get('rollout_log_probs')
    if probs is not None:
        require(len(probs) == length and all(finite_number(x) for x in probs), 'invalid rollout log probabilities')


def validate_multi_agent(samples):
    groups = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        validate_sample(sample)
        require(sample.get('group_id') is not None, 'multi-agent samples need native group_id')
        prompt = sample.get('prompt', '')
        require(isinstance(prompt, str), 'multi-agent prompt must be a string')
        if '### Task: Solution Rewriting Based on Previous Solutions ###' in prompt:
            stage = 'rewriter'
        elif 'End your evaluation with exactly:' in prompt and 'Judgment: IDX' in prompt:
            stage = 'selector'
        else:
            stage = 'solver'
        groups[sample['group_id']][stage].append(sample)
    require(bool(groups), 'empty multi-agent rollout')
    for group, stages in groups.items():
        require({key: len(value) for key, value in stages.items()} == {'solver': 5, 'rewriter': 5, 'selector': 1},
                f'group {group} did not complete five solver/five rewriter/one selector trajectories')
        for stage in ('solver', 'rewriter', 'selector'):
            for sample in stages[stage]:
                require(isinstance(sample.get('response_content'), str) and sample['response_content'].strip(),
                        f'group {group} {stage} did not pass the original </think> parser')
        selected = re.findall(r'Judgment:\s*(\d+)', stages['selector'][0]['response_content'])
        require(bool(selected) and 1 <= int(selected[0]) <= 5, f'group {group} selector did not select a valid rewrite')
    return {'groups': len(groups), 'samples': len(samples), 'stages_per_group': {'solver': 5, 'rewriter': 5, 'selector': 1}}


def validate_strands(samples, log):
    require(bool(samples), 'empty Strands rollout')
    called = 0
    for sample in samples:
        validate_sample(sample)
        length = sample['response_length']
        mask, probs = sample.get('loss_mask'), sample.get('rollout_log_probs')
        require(isinstance(mask, list) and len(mask) == length and all(x in (0, 1) for x in mask) and 1 in mask,
                'Strands needs aligned nonempty generated-token TITO loss masks')
        require(isinstance(probs, list) and len(probs) == length and all(finite_number(x) for x in probs),
                'Strands needs aligned finite TITO log probabilities')
        calls, iters = sample.get('tool_calls'), sample.get('tool_iters')
        require(isinstance(calls, int) and calls >= 0 and isinstance(iters, int) and iters >= 0,
                'missing original ToolLimiter counters')
        if calls > 0:
            require(iters > 0 and 0 in mask, 'tool trajectory must include masked non-model tokens')
            called += 1
    require(called > 0, 'Strands did not call a tool')
    require(re.search(r'Executing Python code: ```python\s*\S.*?``` and get execution result: ```python.*?```', log, re.S),
            'missing original execute_python_code execution/result log')
    return {'samples': len(samples), 'tool_trajectories': called, 'tito_verified': True}


def nonempty_tensor(value):
    # The real native dump contains tensors. Tests use tensor-shaped objects;
    # no torch import is needed to validate the other recipe contracts.
    return hasattr(value, 'numel') and callable(value.numel) and value.numel() > 0


def validate_geo3k(samples):
    require(bool(samples), 'empty geo3k rollout')
    multi_turn = 0
    for sample in samples:
        validate_sample(sample)
        vision = sample.get('multimodal_train_inputs')
        require(isinstance(vision, dict) and nonempty_tensor(vision.get('pixel_values'))
                and nonempty_tensor(vision.get('image_grid_thw')), 'geo3k needs original nonempty visual tensors/grid')
        length = sample['response_length']
        mask, probs = sample.get('loss_mask'), sample.get('rollout_log_probs')
        require(isinstance(mask, list) and len(mask) == length and all(value in (0, 1) for value in mask),
                'geo3k needs aligned model/feedback loss mask')
        require(isinstance(probs, list) and len(probs) == length and all(finite_number(value) for value in probs),
                'geo3k needs aligned finite rollout log probabilities')
        # Collapse adjacent equal mask values; 1->0->1 means model generation,
        # original environment observation, then another model generation.
        segments = [value for index, value in enumerate(mask) if index == 0 or value != mask[index - 1]]
        response = sample['response']
        calls = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.S)
        invoked = False
        for payload in calls:
            try:
                call = json.loads(payload)
            except json.JSONDecodeError:
                continue
            name = call.get('name') or call.get('function', {}).get('name')
            invoked |= name in ('calc_score', 'calc_geo3k_reward')
        if segments[:3] == [1, 0, 1] and invoked and re.search(r'calc_score result:\s*[01](?:\.0)?\b', response):
            multi_turn += 1
    require(multi_turn > 0, 'geo3k did not execute a scoring tool/feedback and a second model turn')
    return {'samples': len(samples), 'visual_multiturn_trajectories': multi_turn, 'vision_verified': True}


def validate_dumps(mode, dumps, grad_norms, log='', expected_rollouts=2):
    require(len(dumps) == expected_rollouts, f'expected {expected_rollouts} native rollout dumps')
    ids = [dump.get('rollout_id') for dump in dumps]
    require(sorted(ids) == list(range(expected_rollouts)), 'missing/duplicate native rollout IDs')
    require(len(grad_norms) >= expected_rollouts and all(finite_number(value) and value >= 0 for value in grad_norms),
            'missing/nonfinite native post-optimizer gradient norms')
    losses = re.findall(r"['\"]train/[^'\"]*loss[^'\"]*['\"]\s*:\s*([^,}\s]+)", log)
    require(bool(losses), 'missing native train loss metrics')
    require(all(finite_number(float(value)) for value in losses), 'nonfinite native train loss metrics')
    reports = []
    for dump in dumps:
        samples = dump.get('samples')
        require(isinstance(samples, list), 'native debug dump must contain samples')
        if mode == 'multi_agent':
            reports.append(validate_multi_agent(samples))
        elif mode == 'strands':
            reports.append(validate_strands(samples, log))
        elif mode == 'geo3k':
            reports.append(validate_geo3k(samples))
        else:
            raise ValueError(f'unknown agent mode: {mode}')
    return {'mode': mode, 'rollouts': reports, 'finite_optimizer_updates': len(grad_norms)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('multi_agent', 'strands', 'geo3k'))
    parser.add_argument('--rollout-glob', required=True)
    parser.add_argument('--grad-glob', required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-rollouts', type=int, default=2)
    args = parser.parse_args()
    import torch
    files = sorted(glob.glob(args.rollout_glob))
    # VLM samples carry locally prepared PIL images in multimodal_inputs. Only
    # these CI-generated geo3k dumps need the native object deserializer;
    # callers must pass their own freshly generated CI output paths.
    dumps = [torch.load(path, map_location='cpu', weights_only=args.mode != 'geo3k') for path in files]
    norms = []
    for path in sorted(glob.glob(args.grad_glob)):
        value = torch.load(path, map_location='cpu', weights_only=True)
        norms.append(value.item() if isinstance(value, torch.Tensor) and value.numel() == 1 else value)
    report = validate_dumps(args.mode, dumps, norms, Path(args.log).read_text(encoding='utf-8'), args.expected_rollouts)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'validated native {args.mode} core agent trajectories and finite optimizer updates')


if __name__ == '__main__':
    main()
PY
}


if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
EXAMPLES_ROOT="${EXAMPLES_ROOT:-$TARGET_ROOT}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
DEPS_ROOT="$GITHUB_WORKSPACE/deps"
SLIME_PROJECT_ENV="$DEPS_ROOT/slime-example.env"
if [[ -f "$SLIME_PROJECT_ENV" ]]; then
  # GITHUB_ENV normally carries values between Actions steps. Keep a
  # workspace-local copy as a fallback for container runners where that
  # file-command handoff was not reflected in the next step's environment.
  # shellcheck disable=SC1090
  source "$SLIME_PROJECT_ENV"
fi
SLIME_FORK_ROOT="${SLIME_FORK_ROOT:?SLIME_FORK_ROOT was not exported by setup}"
export PYTHONPATH="$DEPS_ROOT/sglang/python:$SLIME_FORK_ROOT:$DEPS_ROOT/Megatron-LM:$DEPS_ROOT/Megatron-Bridge/src:${PYTHONPATH:-}"

EXAMPLE_PATH="$EXAMPLES_ROOT/$EXAMPLE_REL"
if [[ ! -f "$EXAMPLE_PATH" ]]; then
  echo "example not found: $EXAMPLE_PATH" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR" "$CI_OUTPUT_DIR/agent-rollouts" "$CI_OUTPUT_DIR/agent-gradients"

# Vendor env scripts assume a login shell and die under `set -u`; relax
# strict mode only while sourcing them (same trap as setup_example.sh).
source_vendor_env() {
  local vendor_file="$1"
  [[ -f "$vendor_file" ]] || return 0
  set +eu
  # shellcheck disable=SC1090
  source "$vendor_file"
  set -eu
}

source_vendor_env /usr/local/Ascend/ascend-toolkit/set_env.sh
source_vendor_env /usr/local/Ascend/nnal/atb/set_env.sh

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

# OPD's teacher listens on a separate local port. The manifest expands
# this value into --rm-url before the fork launcher starts Ray.
export SLIME_TEACHER_PORT=$((13141 + ${GITHUB_RUN_ID:-0} % 1000))
export SLIME_TEACHER_URL="http://127.0.0.1:${SLIME_TEACHER_PORT}/generate"

expand_overlay() {
  "$PYTHON" - <<'PY'
import json
import os
import shlex

raw = os.environ.get('OVERLAY_ARGS', '').strip()
if not raw or raw in ('null', '""'):
    raise SystemExit(0)
try:
    items = json.loads(raw)
except json.JSONDecodeError as exc:
    raise SystemExit(f'OVERLAY_ARGS is not valid JSON: {exc}') from exc
if items in (None, ''):
    raise SystemExit(0)
if not isinstance(items, list):
    raise SystemExit(
        f'OVERLAY_ARGS must be a JSON array, got {type(items).__name__}')
tokens = []
for item in items:
    if not isinstance(item, str):
        raise SystemExit(
            f'OVERLAY_ARGS items must be strings, got {type(item).__name__}')
    # Parse the recipe first: a local model directory containing spaces
    # remains one CLI token after environment substitution.
    tokens.extend(os.path.expandvars(token) for token in shlex.split(item, posix=True))
print(' '.join(shlex.quote(token) for token in tokens))
PY
}

eval "EXTRA_ARGS=( $(expand_overlay) )"

echo "running $EXAMPLE_REL with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

entry_key="$EXAMPLE_REL"

require_visible_devices() {
  local required="$1"
  local default_devices="$2"
  local -a devices

  if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES="$default_devices"
  fi
  IFS=',' read -r -a devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
  if ((${#devices[@]} < required)); then
    echo "insufficient NPU devices for $entry_key: required=$required visible=$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi
  echo "NPU devices for $entry_key: required=$required visible=$ASCEND_RT_VISIBLE_DEVICES"
}

# Environment contract copied from the fork's ascend launchers and NPU CI
# tests (scripts/ascend_script/run-qwen3-8B-npu.sh, tests/tests_npu/):
# Ray must not rewrite the visible-device mask, HCCL needs its port range
# and the NPU allocator keeps expandable_segments (no vLLM CaMemAllocator
# in this stack, unlike projects/roll).
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
# Offline wandb: get_default_wandb_args() stays quiet without
# WANDB_API_KEY, and WANDB_MODE=offline guarantees no network call even
# if one is set.
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

# run-id derived ray dashboard port avoids collisions between parallel
# matrix legs that share a runner.
export RAY_DASHBOARD_PORT=$((8265 + ${GITHUB_RUN_ID:-0} % 100))

# execute_train() takes the full train-arg list as one shell-quoted
# string (fork API contract); rebuild it from the expanded overlay.
case "$entry_key" in
  examples/fully_async/run-qwen2.5-0.5B-fully_async.sh)
    # Recipe source: tests/tests_npu/nightly_CI/
    # test_qwen2.5_0.5B_fully_async_short_npu.py (fork-verified on NPU).
    require_visible_devices 4 '0,1,2,3'
    NUM_GPUS=4
    MODEL_TYPE=qwen2.5-0.5B
    TRAIN_SCRIPT=train_async.py
    ;;
  examples/on_policy_distillation/run-qwen3-8B-opd.sh)
    # Fork NPU ST: 4 actor + 3 rollout devices in Ray, 1 teacher server.
    require_visible_devices 8 '0,1,2,3,4,5,6,7'
    IFS=',' read -r -a opd_devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
    export SLIME_OPD_TRAIN_DEVICES="$(IFS=,; echo "${opd_devices[*]:0:7}")"
    export SLIME_OPD_TEACHER_DEVICE="${opd_devices[7]}"
    NUM_GPUS=7
    MODEL_TYPE=qwen2.5-0.5B
    TRAIN_SCRIPT=train.py
    ;;
  examples/retool/retool_qwen3_4b_rl.sh)
    # The upstream CUDA script colocates train and rollout on four cards.
    # On NPU run #16 the TP2 servers became unhealthy while the train
    # actor was initializing. Reserve four cards for each side instead.
    require_visible_devices 8 '0,1,2,3,4,5,6,7'
    export SLIME_RETOOL_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
    NUM_GPUS=8
    MODEL_TYPE=qwen3-4B-Instruct-2507
    TRAIN_SCRIPT=train.py
    export PYTHONPATH="$SLIME_FORK_ROOT/examples/retool:$PYTHONPATH"
    ;;
  examples/retool/retool_qwen3_4b_sft.sh)
    require_visible_devices 2 '0,1'
    IFS=',' read -r -a sft_devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
    export SLIME_RETOOL_DEVICES="$(IFS=,; echo "${sft_devices[*]:0:2}")"
    : "${SLIME_SFT_FIXTURE_JSONL:?SFT fixture was not exported by setup}"
    NUM_GPUS=2
    MODEL_TYPE=qwen3-4B-Instruct-2507
    TRAIN_SCRIPT=train_async.py
    ;;
  examples/on_policy_distillation/run-qwen3-8B-opd-megatron.sh)
    require_visible_devices 8 '0,1,2,3,4,5,6,7'
    export SLIME_RETOOL_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
    NUM_GPUS=8
    MODEL_TYPE=qwen2.5-0.5B
    TRAIN_SCRIPT=train.py
    ;;
  examples/train_infer_mismatch_helper/run-qwen3-4b-mis.sh)
    require_visible_devices 8 '0,1,2,3,4,5,6,7'
    export SLIME_RETOOL_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
    NUM_GPUS=8
    MODEL_TYPE=qwen3-4B-Instruct-2507
    TRAIN_SCRIPT=train.py
    ;;
  examples/multi_agent/run-qwen3-30B-A3B-multi-agent.sh)
    require_visible_devices 8 '0,1,2,3,4,5,6,7'
    export SLIME_RETOOL_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
    NUM_GPUS=8
    MODEL_TYPE=qwen3-4B
    TRAIN_SCRIPT=train.py
    ;;
  examples/eval_multi_task/multi_task.sh)
    require_visible_devices 8 '0,1,2,3,4,5,6,7'
    export SLIME_RETOOL_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
    : "${SLIME_EVAL_CONFIG:?Multi-task evaluation config was not exported by setup}"
    NUM_GPUS=8
    MODEL_TYPE=qwen2.5-0.5B
    TRAIN_SCRIPT=train.py
    ;;
  examples/strands_sglang/strands_qwen3_8b.sh)
    require_visible_devices 8 '0,1,2,3,4,5,6,7'
    export SLIME_RETOOL_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
    NUM_GPUS=8
    MODEL_TYPE=qwen3-4B-Instruct-2507
    TRAIN_SCRIPT=train.py
    export PYTHONPATH="$SLIME_FORK_ROOT/examples/strands_sglang:$PYTHONPATH"
    ;;
  examples/search-r1/run_qwen2.5_3B.sh)
    require_visible_devices 8 '0,1,2,3,4,5,6,7'
    export SLIME_RETOOL_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
    : "${SLIME_SEARCH_INDEX:?Search-R1 BM25 index was not exported by setup}"
    NUM_GPUS=8
    MODEL_TYPE=qwen2.5-0.5B
    TRAIN_SCRIPT=train.py
    export PYTHONPATH="$SLIME_FORK_ROOT/examples/search-r1:$PYTHONPATH"
    ;;
  examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py)
    require_visible_devices 4 '0,1,2,3'
    IFS=',' read -r -a geo_devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
    export SLIME_RETOOL_DEVICES="$(IFS=,; echo "${geo_devices[*]:0:4}")"
    export MODEL_ARGS_ROTARY_BASE=5000000
    NUM_GPUS=4
    MODEL_TYPE=qwen3-1.7B
    TRAIN_SCRIPT=train.py
    ;;
  *)
    echo "no engine-call metadata mapping for $entry_key" >&2
    exit 1
    ;;
esac

: "${SLIME_MODEL_PATH:?SLIME_MODEL_PATH was not exported by setup}"
if [[ "$entry_key" != examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py ]]; then
  : "${SLIME_TORCH_DIST_PATH:?SLIME_TORCH_DIST_PATH was not exported by setup}"
fi
: "${SLIME_FIXTURE_JSONL:?SLIME_FIXTURE_JSONL was not exported by setup}"

cd "$SLIME_FORK_ROOT"

"$PYTHON" - "$SLIME_FORK_ROOT" "$NUM_GPUS" "$MODEL_TYPE" "$TRAIN_SCRIPT" "${EXTRA_ARGS[@]}" <<'PY' 2>&1 | tee "$CI_OUTPUT_DIR/agent-train.log"
import importlib.util
import json
import os
import signal
import shlex
import re
import subprocess
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

fork_root, num_gpus, model_type, train_script, *train_args = sys.argv[1:]

# The fork execute_train() owns ray start/submit, NPU resource
# injection and the runtime env; load it straight from the fork tree.
if str(fork_root) not in sys.path:
    sys.path.insert(0, str(fork_root))
spec = importlib.util.spec_from_file_location(
    "_fork_command_utils",
    str(Path(fork_root) / "slime" / "utils" / "external_utils" / "command_utils.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

print(f"execute_train: model_type={model_type} train_script={train_script} num_gpus={num_gpus}")
print("train args:", " ".join(train_args))

teacher_device = os.environ.get("SLIME_OPD_TEACHER_DEVICE")
teacher_process = None
search_process = None
search_log = Path(os.environ["CI_OUTPUT_DIR"]) / "search-retriever.log"
teacher_log = Path(os.environ["CI_OUTPUT_DIR"]) / "opd-teacher.log"


def teacher_log_tail():
    if teacher_log.exists():
        print("teacher log (last 40 lines):", flush=True)
        with teacher_log.open(errors="replace") as output:
            print("".join(deque(output, maxlen=40)), flush=True)


def start_opd_teacher():
    global teacher_process
    teacher_env = os.environ.copy()
    teacher_env.update({
        "CUDA_VISIBLE_DEVICES": teacher_device,
        "ASCEND_RT_VISIBLE_DEVICES": teacher_device,
        "GLOO_SOCKET_IFNAME": "lo",
        "NCCL_SOCKET_IFNAME": "lo",
        "TP_SOCKET_IFNAME": "lo",
        "no_proxy": "127.0.0.1",
    })
    for proxy in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        teacher_env.pop(proxy, None)
    command = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", os.environ["SLIME_MODEL_PATH"],
        "--host", "127.0.0.1",
        "--port", os.environ["SLIME_TEACHER_PORT"],
        "--tp", "1",
        "--mem-fraction-static", "0.6",
    ]
    with teacher_log.open("w") as output:
        teacher_process = subprocess.Popen(
            command, env=teacher_env, stdout=output,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    print(f"OPD teacher: device={teacher_device} pid={teacher_process.pid} log={teacher_log}", flush=True)
    health_url = os.environ["SLIME_TEACHER_URL"].removesuffix("/generate") + "/health_generate"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if teacher_process.poll() is not None:
            teacher_log_tail()
            raise RuntimeError(f"OPD teacher exited with code {teacher_process.returncode}")
        try:
            with opener.open(health_url, timeout=3) as response:
                if response.status == 200:
                    print("OPD teacher ready", flush=True)
                    return
        except Exception:
            pass
        time.sleep(5)
    teacher_log_tail()
    raise TimeoutError("OPD teacher did not become healthy within 600 seconds")


launch_options = {}
retool_devices = os.environ.get("SLIME_RETOOL_DEVICES")
if teacher_device:
    train_devices = os.environ["SLIME_OPD_TRAIN_DEVICES"]
    # The fork builds Ray's runtime env from its own defaults, so override
    # the hard-coded 0..7 mask to keep the teacher's physical NPU isolated.
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = train_devices
    os.environ["CUDA_VISIBLE_DEVICES"] = train_devices
    launch_options = {
        "before_ray_job_submit": start_opd_teacher,
        "extra_env_vars": {
            "ASCEND_RT_VISIBLE_DEVICES": train_devices,
            "CUDA_VISIBLE_DEVICES": train_devices,
            "ASCEND_TOOLKIT_HOME": "/usr/local/Ascend/ascend-toolkit/latest/",
            "ASCEND_HOME_PATH": "/usr/local/Ascend/ascend-toolkit/latest/",
            "HCCL_IF_IP": "127.0.0.1",
            "TP_SOCKET_IFNAME": "lo",
            "GLOO_SOCKET_IFNAME": "lo",
        },
    }
elif retool_devices:
    # execute_train() otherwise hardcodes 0..7 in its Ray runtime env,
    # even though this colocated job reserves only four physical NPUs.
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = retool_devices
    os.environ["CUDA_VISIBLE_DEVICES"] = retool_devices
    launch_options = {
        "extra_env_vars": {
            "ASCEND_RT_VISIBLE_DEVICES": retool_devices,
            "CUDA_VISIBLE_DEVICES": retool_devices,
        },
    }
    print(f"ReTool Ray-visible NPUs: {retool_devices}", flush=True)


def start_search_server():
    global search_process
    command = [sys.executable, str(Path(fork_root) / "examples/search-r1/local_dense_retriever/retrieval_server.py"),
        "--retriever_name", "bm25", "--index_path", os.environ["SLIME_SEARCH_INDEX"], "--topk", "3"]
    with search_log.open("w") as output:
        search_process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 120
    request = urllib.request.Request("http://127.0.0.1:8000/retrieve",
        data=json.dumps({"queries": ["France capital"], "topk": 3, "return_scores": False}).encode(),
        headers={"Content-Type": "application/json"})
    while time.monotonic() < deadline:
        if search_process.poll() is not None:
            raise RuntimeError(f"BM25 retriever exited; see {search_log}")
        try:
            with opener.open(request, timeout=3) as response:
                result = json.load(response)
            if "Paris" not in json.dumps(result):
                raise RuntimeError("Native retriever returned no matching fixture fact")
            print("Search-R1 native BM25 /retrieve ready", flush=True)
            return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise TimeoutError(f"Native BM25 retriever did not become ready; see {search_log}")


if os.environ.get("SLIME_SEARCH_INDEX"):
    launch_options["before_ray_job_submit"] = start_search_server


def print_retool_worker_errors():
    # Ray's submitted-job traceback may omit the actor's real assertion
    # site. Keep a bounded excerpt from its local worker logs on failure.
    logs = Path("/tmp/ray/session_latest/logs")
    if not logs.exists():
        return
    try:
        candidates = list(logs.glob("worker-*.err"))
        candidates += list(logs.glob("python-core-worker-*.log"))
        for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[:4]:
            with path.open(errors="replace") as output:
                tail = "".join(deque(output, maxlen=35))
            if "Traceback" in tail or "AssertionError" in tail or "ERROR" in tail:
                print(f"Ray worker log tail ({path.name}):\n{tail}", flush=True)
    except OSError as log_error:
        print(f"could not read Ray worker diagnostics: {log_error}", flush=True)


try:
    module.execute_train(
        train_args=shlex.join(train_args),
        num_gpus_per_node=int(num_gpus),
        megatron_model_type=model_type,
        train_script=train_script,
        **launch_options,
    )
    if search_process:
        if search_process.poll() is not None:
            raise RuntimeError(f"BM25 retriever exited during training; see {search_log}")
        # One request is setup readiness. Require a further real rollout
        # request, rather than accepting a no-search training run as success.
        queries = len(re.findall(r'"POST /retrieve HTTP/[^"\n]+" 200(?: |$)', search_log.read_text(errors="replace")))
        if queries < 2:
            raise RuntimeError(f"Search-R1 rollout did not call native retrieval: requests={queries}; see {search_log}")
except Exception:
    if retool_devices:
        print_retool_worker_errors()
    raise
finally:
    if search_process and search_process.poll() is None:
        os.killpg(search_process.pid, signal.SIGTERM)
        try:
            search_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(search_process.pid, signal.SIGKILL)
            search_process.wait()
    if teacher_process and teacher_process.poll() is None:
        os.killpg(teacher_process.pid, signal.SIGTERM)
        try:
            teacher_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(teacher_process.pid, signal.SIGKILL)
            teacher_process.wait()
PY

case "$entry_key" in
  examples/multi_agent/run-qwen3-30B-A3B-multi-agent.sh) validation_mode=multi_agent ;;
  examples/strands_sglang/strands_qwen3_8b.sh) validation_mode=strands ;;
  examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py) validation_mode=geo3k ;;
  *) validation_mode= ;;
esac
if [[ -n "$validation_mode" ]]; then
  validate_agent_rollouts \
    "$validation_mode" --rollout-glob "$CI_OUTPUT_DIR/agent-rollouts/[0-9]*.pt" \
    --grad-glob "$CI_OUTPUT_DIR/agent-gradients/actor_*.pt" \
    --log "$CI_OUTPUT_DIR/agent-train.log" --output "$CI_OUTPUT_DIR/agent-validation.json"
fi
