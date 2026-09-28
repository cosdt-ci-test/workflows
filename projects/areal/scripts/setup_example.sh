#!/usr/bin/env bash
# Prepare the CI environment for one supported AReaL example.
# $1 is the manifest profile. Unknown profiles fail before any install.
#
# CI base image is ghcr.io/hwvanici/areal_npu, which already has the full stack:
# CANN, torch, vLLM-Ascend, Megatron, MindSpeed, huggingface_hub etc.
# We should NOT reinstall these; just install the target AReaL source and assets.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

SUPPORTED_PROFILES="areal-vlm-grpo areal-vlm-mt-grpo areal-vlm-sft areal-tir-grpo areal-scaffold-grpo areal-agents-grpo areal-math-grpo areal-math-sft areal-math-aime areal-math-boba areal-countdown-grpo areal-align"
AIME_PREP=0
BOBA_PREP=0
COUNTDOWN_PREP=0
HHRLHF_PREP=0
case "$PROFILE" in
  areal-vlm-grpo) MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct" ;;
  areal-vlm-mt-grpo) MODEL_ID="Qwen/Qwen3-VL-2B-Instruct" ;;
  areal-vlm-sft) MODEL_ID="Qwen/Qwen3-VL-2B-Instruct" ;;
  # tir/train_tir.py: the torl_data loader self-downloads its small parquets
  # from GitHub (GAIR-NLP/ToRL) at load time, so no dataset prep here.
  areal-tir-grpo) MODEL_ID="Qwen/Qwen2.5-Math-1.5B" ;;
  # gsm8k_rlvr_scaffolding.py: pure RLVR, dataset is online gsm8k, no prep.
  areal-scaffold-grpo) MODEL_ID="Qwen/Qwen2.5-3B-Instruct" ;;
  areal-agents-grpo) MODEL_ID="Qwen/Qwen2-1.5B-Instruct" ;;
  # gsm8k_rl.py and gsm8k_eval.py share the same model.
  areal-math-grpo) MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct" ;;
  areal-math-sft) MODEL_ID="Qwen/Qwen3-1.7B" ;;
  areal-math-aime) MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"; AIME_PREP=1 ;;
  areal-math-boba) MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"; BOBA_PREP=1 ;;
  areal-countdown-grpo) MODEL_ID="Qwen/Qwen2.5-3B-Instruct"; COUNTDOWN_PREP=1 ;;
  # hhrlhf_dpo.py and hhrlhf_rw.py share model + dataset.
  areal-align) MODEL_ID="Qwen/Qwen2.5-7B"; HHRLHF_PREP=1 ;;
  *)
    echo "unknown profile: ${PROFILE} (supported: ${SUPPORTED_PROFILES})" >&2
    exit 1
    ;;
esac

if [[ -z "${TARGET_ROOT:-}" || -z "${GITHUB_WORKSPACE:-}" || -z "${GITHUB_ENV:-}" ]]; then
  echo "TARGET_ROOT / GITHUB_WORKSPACE / GITHUB_ENV must be set" >&2
  exit 2
fi

# -------------------------------------------------------
# 1. Install the AReaL source under test.
# The official image ships its own copy of AReaL, but we need to use
# the checked-out TARGET_ROOT version because it may contain fixes.
# -------------------------------------------------------
command -v uv >/dev/null 2>&1 || pip install -q uv
uv pip install --no-deps -e "$TARGET_ROOT" --system

# -------------------------------------------------------
# 2. Runtime Environment setup.
# -------------------------------------------------------
echo "PYTHONPATH=/areal-workspace/MindSpeed:/areal-workspace/Megatron-Bridge/src:${PYTHONPATH:-}" >> "$GITHUB_ENV"
echo "HCCL_IF_BASE_PORT=63000" >> "$GITHUB_ENV"
echo "HCCL_NPU_SOCKET_PORT_RANGE=62100-62350" >> "$GITHUB_ENV"
echo "TASK_QUEUE_ENABLE=1" >> "$GITHUB_ENV"
echo "OMP_NUM_THREADS=1" >> "$GITHUB_ENV"
echo "WANDB_MODE=disabled" >> "$GITHUB_ENV"
echo "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True" >> "$GITHUB_ENV"
echo "USE_OPTIMIZED_MODEL=0" >> "$GITHUB_ENV"
echo "AREAL_ALLOW_DEFAULT_ADMIN_KEY=1" >> "$GITHUB_ENV"

# -------------------------------------------------------
# 3. Pre-download Model & Dataset (using image's native tools)
# -------------------------------------------------------
# Model and dataset are both fetched online (no local fixtures). Export the
# Hub mirror through GITHUB_ENV so the run-example step inherits it too: AReaL
# downloads the dataset at train time via `load_dataset` inside the data
# service, not here.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "HF_ENDPOINT=${HF_ENDPOINT}" >> "$GITHUB_ENV"
export AREAL_MODEL_ID="$MODEL_ID"

python3 <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ["AREAL_MODEL_ID"]
dest = os.path.join(os.environ["GITHUB_WORKSPACE"], "areal_models", model_id.split("/")[-1])
snapshot_download(model_id, local_dir=dest)
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"AREAL_MODEL_PATH={dest}\n")
print(f"Downloaded model to {dest}", flush=True)
PY

# -------------------------------------------------------
# 4. AIME dataset + derived config (areal-math-aime).
# -------------------------------------------------------
# areal/dataset/aime.py loads <path>/aime_train.parquet and
# <path>/aime_test.parquet from a LOCAL directory (upstream ships no aime
# config; the loader takes no HF repo id). Convert the public MathArena
# parquets (same flow as examples/distillation/mopd/README.md), then derive
# the GRPO config from gsm8k_grpo_npu.yaml: point the datasets at the local
# dir and set rollout.agent to null (aime_rl.py runs the standard
# RLVRWorkflow, not an agent workflow). The generated config is exposed to
# overlay args via AREAL_AIME_CONFIG.
if [[ "$AIME_PREP" == 1 ]]; then
python3 <<'PY'
import os

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download

workspace = os.environ["GITHUB_WORKSPACE"]
target_root = os.environ["TARGET_ROOT"]
aime_dir = os.path.join(workspace, "areal_data", "aime")
os.makedirs(aime_dir, exist_ok=True)

# hf_hub_download honors HF_ENDPOINT (mirror) set in section 3.
for repo, dst in [
    ("MathArena/aime_2025", "aime_train.parquet"),
    ("MathArena/aime_2026", "aime_test.parquet"),
]:
    src = hf_hub_download(
        repo_id=repo, repo_type="dataset",
        filename="data/train-00000-of-00001.parquet",
    )
    df = pd.read_parquet(src).rename(columns={"problem": "question"})
    df = df[["question", "answer"]]
    df.to_parquet(os.path.join(aime_dir, dst))
    print(f"prepared {dst}: {len(df)} rows", flush=True)

base = os.path.join(target_root, "examples", "math", "gsm8k_grpo_npu.yaml")
with open(base, encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
cfg["experiment_name"] = "aime-grpo"
cfg["train_dataset"]["path"] = aime_dir
cfg["valid_dataset"]["path"] = aime_dir
# cfg["rollout"]["agent"] = None
config_path = os.path.join(aime_dir, "aime_grpo_npu.yaml")
with open(config_path, "w", encoding="utf-8") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)

with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"AREAL_AIME_CONFIG={config_path}\n")
print(f"generated {config_path}", flush=True)
PY
fi

# -------------------------------------------------------
# 5. BoBa dataset (areal-math-boba).
# -------------------------------------------------------
# boba_grpo.py loads train_dataset.path as a LOCAL jsonl file
# (load_dataset("json", data_files=path)); the upstream yaml ships the HF
# repo id, which would fail with FileNotFoundError. Pre-download the public
# AReaL-boba-106k.jsonl and expose its local path via AREAL_BOBA_DATA.
if [[ "$BOBA_PREP" == 1 ]]; then
python3 <<'PY'
import os
import shutil

from huggingface_hub import hf_hub_download

src = hf_hub_download(
    repo_id="inclusionAI/AReaL-boba-Data", repo_type="dataset",
    filename="AReaL-boba-106k.jsonl",
)
dst = os.path.join(
    os.environ["GITHUB_WORKSPACE"], "areal_data", "boba", "AReaL-boba-106k.jsonl"
)
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copyfile(src, dst)
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"AREAL_BOBA_DATA={dst}\n")
print(f"downloaded boba dataset to {dst}", flush=True)
PY
fi

# -------------------------------------------------------
# 6. Countdown dataset (areal-countdown-grpo).
# -------------------------------------------------------
# countdown.py writes ./data/countdown/qwen/*.jsonl relative to CWD; run it
# from TARGET_ROOT so the generated files land where train_config.yaml's
# default train_dataset.path=data/countdown/qwen/train_e.jsonl expects them
# (run_example.sh also cds to TARGET_ROOT). The default generation size is
# 500k samples; 32/8 is plenty for the 1-step smoke. The tokenizer comes
# from the model predownloaded in section 3, so generation is offline.
if [[ "$COUNTDOWN_PREP" == 1 ]]; then
  MODEL_DIR="$GITHUB_WORKSPACE/areal_models/${MODEL_ID##*/}"
  mkdir -p "$TARGET_ROOT/data/countdown/qwen"
  (cd "$TARGET_ROOT" && python3 examples/countdown/countdown.py \
    --num_samples 32 --eval_size 8 --tokenizer_path "$MODEL_DIR")
fi

# -------------------------------------------------------
# 7. HH-RLHF dataset (areal-align: hhrlhf_dpo.py / hhrlhf_rw.py).
# -------------------------------------------------------
# Anthropic/hh-rlhf is public (NOT gated), but its jsonl.gz files live in
# per-subset subdirectories (harmless-base/, helpful-base/, ...), so the
# upstream loader's load_dataset(path, split=...) without a config name
# fails with a multi-config error. Work around it boba-style: download the
# harmless-base subset online (via HF_ENDPOINT mirror) into a flat local
# dir; load_dataset(<dir>, split="train"/"test") then infers splits from
# the file names. Exposed to overlay args via AREAL_HHRLHF_DATA.
# NOTE: the dir is deliberately named "hh-rlhf" — areal's dataset dispatch
# matches the literal substring "hh-rlhf" in the path; a dir named
# "hhrlhf" falls through to the load_from_disk fallback and errors out.
if [[ "$HHRLHF_PREP" == 1 ]]; then
python3 <<'PY'
import os
import shutil

from huggingface_hub import hf_hub_download

dst_dir = os.path.join(os.environ["GITHUB_WORKSPACE"], "areal_data", "hh-rlhf")
os.makedirs(dst_dir, exist_ok=True)
for filename in ("train.jsonl.gz", "test.jsonl.gz"):
    src = hf_hub_download(
        repo_id="Anthropic/hh-rlhf", repo_type="dataset",
        filename=f"harmless-base/{filename}",
    )
    shutil.copyfile(src, os.path.join(dst_dir, filename))
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"AREAL_HHRLHF_DATA={dst_dir}\n")
print(f"prepared hh-rlhf harmless-base at {dst_dir}", flush=True)
PY
fi
