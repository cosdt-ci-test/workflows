#!/usr/bin/env bash
# Prepare the CI environment for one supported torchtune example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# torchtune itself is installed from TARGET_ROOT (the release checkout
# under test), so the guarded tag is exactly the code that runs.
#
# Profile split (2026-09-22, after CI run 35721666282 7-leg fail):
# The original `setup_torchtune` profile pinned torch_npu==2.11.0 in a
# shared `ensure_torch_stack` function. That single pin couldn't satisfy
# both:
#   - single-device recipes (generate / quantize / eleuther_eval / ppo /
#     lora_finetune_single / full_finetune_single / lora_dpo_single):
#     torch_npu 2.11.0 works; pre-download was the problem (ModelScope
#     ._____temp staging dir missing on cold container → FileDownloadError).
#   - distributed recipes (lora_finetune_distributed / full_finetune_distributed /
#     lora_dpo_distributed / full_dpo_distributed / knowledge_distillation_distributed):
#     torch_npu 2.11.0's c10d hccl backend raises
#     `RuntimeError: Distributed package doesn't have NCCL built in`
#     at recipe `init_process_group(self.distributed_backend)`, AND
#     its `dist.broadcast` from `_distributed.py:92` on a CPU tensor
#     raises `No backend type associated with device type cpu`
#     (per torch.distributed.pipelining × torch_npu c10d ABI gap memory).
#
# So each profile now does its own torch stack install. The shared path
# keeps ONLY what's truly universal (pip index selection, pip upgrade,
# fixture copy, HF_ENDPOINT mirror); `ensure_torch_stack` is now a
# per-profile helper that takes the torch_npu version as $1 (default
# 2.11.0 for single; TBD for distributed, set after npu-3 verification).
#
# Asset sourcing (also 2026-09-22): pre-download of
# Qwen/Qwen2.5-{0.5,1.5}B-Instruct no longer lives in setup; it's
# declared in cache-seed/torchtune/ms_seeds.yaml and planted into the
# shared HF hub cache by the cache-seed workflow. setup only calls
# `resolve_seed_envs` to read `refs/main` and inject TT_MODEL_PATH /
# TT_TEACHER_PATH for the overlay args. (The 1.5B safetensors 2.88GB
# was what made the cold-container ModelScope download fail; offloading
# to a one-time seed dispatch removes the per-job network cost too.)
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"
ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
ALIYUN_PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/

pip_ascend() {
  python -m pip install --extra-index-url "$ASCEND_PIP_INDEX" "$@"
}

select_pip_index() {
  # Runners live in mainland China: prefer the cluster pip cache, fall
  # back to the Aliyun mirror. The ascend index stays available via
  # PIP_EXTRA_INDEX_URL (set by the engine) for torch_npu wheels.
  if python -c "
import os
import urllib.error
import urllib.request
try:
    urllib.request.urlopen(os.environ['CLUSTER_PIP_INDEX'], timeout=3)
except urllib.error.HTTPError:
    pass
" 2>/dev/null; then
    export PIP_INDEX_URL="$CLUSTER_PIP_INDEX"
    export PIP_TRUSTED_HOST="$CLUSTER_PIP_HOST"
  else
    export PIP_INDEX_URL="$ALIYUN_PIP_INDEX"
    unset PIP_TRUSTED_HOST
  fi
  echo "pip index: $PIP_INDEX_URL"
}

# Per-profile torch stack install (torch/torchvision versions are
# uniform across profiles; torch_npu version varies).
#
# $1 = torch_npu version (mandatory; caller picks the version that
#      matches the recipes it serves).
#
# Why torch + torchvision stay the same across profiles:
#   - torch 2.11.0+cpu: floor for `from torch.nn.functional import
#     ScalingType` that torchao 0.18 needs (main HEAD path). The +cpu
#     wheel comes from aliyun's mirror of pytorch.org/whl/cpu; the
#     cluster pip cache ships only CUDA torch wheels (Requires-Dist:
#     cuda-toolkit) which collide with constraints-npu.txt's
#     cuda-toolkit<0 — see torchtitan-cuda-torch-wheel-trap memory.
#     `--find-links` to the +cpu-only directory sidesteps it.
#   - torchvision 0.26.0+cpu: main HEAD unconditionally imports
#     torchvision at torchtune/data/_utils.py:12, triggered by
#     `from torchtune import datasets` (datasets/__init__.py:7 →
#     multimodal → _llava → data._messages → data → data._utils →
#     import torchvision). Per upstream's published compat table,
#     torch 2.11 ↔ torchvision 0.26; pinning ≤0.28 also dodges the
#     v0.29.0 stable ABI requirement (calls stable::permute, torch
#     2.14 only; torch_npu has no 2.14 release) — see
#     torchvision-v29-stable-abi memory. The +cpu wheel is the only
#     one in the aliyun pytorch-wheels/cpu find-links repo.
ensure_torch_stack() {
  local torch_npu_version="${1:?ensure_torch_stack requires torch_npu version as \$1}"
  echo "installing torch==2.11.0+cpu (aliyun pytorch-wheels/cpu find-links, deps from PIP_INDEX_URL)"
  # The 148MB torch+cpu aarch64 wheel is the long pole of setup — CI run
  # 35483424375 (2026-09-20) saw every leg print the "Downloading torch-
  # 2.11.0+cpu... (148.1 MB)" banner then hang for 28+ min before the
  # 30-min job timeout cancelled the job. Split download from install so
  # a slow link can be retried with curl's byte-range resume (-C -) and
  # bounded by --max-time, and so pip installs a local wheel instead of
  # re-resolving through the aliyun find-links (pure-python deps still
  # come from PIP_INDEX_URL). Same hardening as projects/xtuner
  # (CI run 35483448757).
  TORCH_WHEEL_DIR=/tmp/torchtune-wheels
  mkdir -p "$TORCH_WHEEL_DIR"
  TORCH_WHEEL="$TORCH_WHEEL_DIR/torch-2.11.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl"
  if [[ ! -f "$TORCH_WHEEL" ]]; then
    curl -fsSL --retry 5 --retry-delay 5 --retry-all-errors --max-time 1500 \
      -C - -o "$TORCH_WHEEL" \
      "https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.11.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl" \
      || python -m pip install --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.11.0
  fi
  if [[ -f "$TORCH_WHEEL" ]]; then
    python -m pip install --find-links "$TORCH_WHEEL_DIR" torch==2.11.0 || \
      python -m pip install --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.11.0
  fi
  echo "installing torchvision==0.26.0+cpu (matches torch 2.11 per upstream compat table)"
  python -m pip install \
    --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu \
    torchvision==0.26.0
  echo "installing torch_npu==${torch_npu_version} (Huawei ascend index)"
  pip_ascend "torch_npu==${torch_npu_version}"
}

# Copy CI fixture data files into the target root so that example
# recipes can load them via a local path under $TARGET_ROOT/fixtures/
# (engine contract: overlay_args uses ${TARGET_ROOT}/fixtures/...;
# same decoupling from the workflows-checkout subtree as peft).
prepare_fixtures() {
  local src="${FIXTURE_DIR:?FIXTURE_DIR is required}"
  local dst="$TARGET_ROOT/fixtures"
  echo "preparing fixtures from $src to $dst"
  # Fail-soft: we don't want to hard-fail if the project only ships a
  # custom-task YAML and no JSON/JSONL — the eleuther_eval recipe, for
  # example, ships fixtures/eleuther_tasks/*.yaml but no top-level JSON.
  # The actual presence/validity of the file is checked downstream when
  # the recipe opens it; a missing file at setup is a config bug that
  # surfaces as a real stack trace, not a setup-script one.
  if ! compgen -G "$src/*" >/dev/null; then
    echo "FATAL: fixture dir $src is empty" >&2
    exit 1
  fi
  mkdir -p "$dst"
  # Mirror the entire tree: flat *.json / *.jsonl for dataset recipes,
  # plus *.yaml subdirs (e.g. fixtures/eleuther_tasks/) for lm_eval
  # custom-task definitions consumed by recipes/eleuther_eval.py via
  # cfg.include_path = ${TARGET_ROOT}/fixtures/eleuther_tasks.
  cp -r "$src"/. "$dst"/
  echo "copied $(find "$dst" -type f | wc -l) fixture file(s) to $dst"
}

resolve_seed_envs() {
  # $@ = alternating (hf_id, env var) pairs. Resolve each seeded asset's
  # snapshot path from the shared HF hub cache (refs/main -> sha) and
  # append VAR=<snapshot> to GITHUB_ENV for manifest overlay_args.
  # The shared cache root is populated by the cache-seed workflow
  # (ms plant: cache-seed/torchtune/ms_seeds.yaml;
  #  curl plant: cache-seed/torchtune/curl_seeds.yaml;
  #  → SHARED_CACHE_ROOT, default ~/.cache/huggingface).
  # Nothing downloads in example jobs anymore.
  python - "$@" <<'PY'
import os
import sys
from pathlib import Path

HUB_ROOT = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"
pairs = sys.argv[1:]
if len(pairs) % 2:
    raise SystemExit("resolve_seed_envs: expected alternating hf_id var pairs")
for hf_id, var in zip(pairs[::2], pairs[1::2]):
    repo_dir = HUB_ROOT / f"models--{hf_id.replace('/', '--')}"
    refs = repo_dir / "refs" / "main"
    if not refs.is_file():
        raise SystemExit(
            f"{hf_id} missing from shared cache root — dispatch the "
            f"cache-seed workflow (spec: cache-seed/torchtune/ms_seeds.yaml)")
    sha = refs.read_text().strip()
    snap = repo_dir / "snapshots" / sha
    if not snap.is_dir() or not any(snap.iterdir()):
        raise SystemExit(f"{hf_id}: refs/main -> {sha[:8]} has no snapshot files")
    with open(os.environ["GITHUB_ENV"], "a") as fh:
        fh.write(f"{var}={snap}\n")
    print(f"{var}={snap}", flush=True)
PY
}

# Install torchtune + its declared deps (transformers / omegaconf /
# tokenizers / safetensors / tqdm / pyyaml). Shared by every profile —
# the version selection / dynamic torchao probe are torchtune-checkout-
# dependent and have nothing to do with the torch_npu version split
# above. Probe runs against the installed source (non-editable, see
# Quick-start-Ascend.md:164-167), so it reflects whatever ref the
# engine checked out (release tag OR main HEAD).
install_torchtune_pkg() {
  echo "installing torchtune from $TARGET_ROOT (non-editable, see Quick-start-Ascend.md:164-167)"
  patch_main_head_bugs
  python -m pip install "$TARGET_ROOT"
  python -m pip install "transformers==4.57.1" "omegaconf>=2.3,<3" \
    tokenizers safetensors tqdm pyyaml

  # Probe which NF4Tensor import path the torchtune checkout uses, then
  # install the matching torchao exact pin.
  local import_path torchao_pin
  import_path="$(python -c "
import importlib.util, pathlib
p = pathlib.Path('$TARGET_ROOT') / 'torchtune' / 'modules' / 'common_utils.py'
src = p.read_text() if p.exists() else ''
for line in src.splitlines():
    s = line.strip()
    if s.startswith('from torchao') and 'NF4Tensor' in s:
        print(s)
        break
else:
    print('NF4Tensor_NOT_IMPORTED')
")"
  echo "torchtune common_utils.py NF4Tensor import: $import_path"
  case "$import_path" in
    "from torchao.dtypes.nf4tensor import NF4Tensor")
      torchao_pin="torchao==0.13.0"
      ;;
    "from torchao.quantization import NF4Tensor")
      # 0.18.0 re-exposed NF4Tensor under torchao.quantization; main
      # HEAD depends on the exact 0.18 series (later 0.18.x keep the
      # symbol but pin to whatever the upstream test grid currently
      # passes — 0.18.0 is the first stable release with the re-add).
      torchao_pin="torchao==0.18.0"
      ;;
    "NF4Tensor_NOT_IMPORTED")
      # Newer torchtune may drop NF4Tensor entirely. Default to a
      # neutral recent torchao and let setup proceed; recipe failures
      # downstream will surface real reasons rather than setup noise.
      echo "WARN: common_utils.py does not import NF4Tensor; skipping torchao pin"
      torchao_pin=""
      ;;
    *)
      echo "FATAL: unexpected NF4Tensor import line: $import_path" >&2
      exit 1
      ;;
  esac
  if [ -n "$torchao_pin" ]; then
    echo "pinning $torchao_pin (matched import path)"
    python -m pip install "$torchao_pin"
  fi

  # importlib.metadata.version returns the real install tag for both
  # editable installs (where __version__ is empty string) and wheel
  # installs. torchtune.__version__ is "" by default in the source
  # tree; reading it directly would print a confusing blank.
  python -c "
import importlib.metadata as md
import torchao, omegaconf, transformers
import torchtune  # noqa: just to confirm the import chain
print('torchtune', md.version('torchtune'), '/ torchao', torchao.__version__, '/ transformers', transformers.__version__)
"
}

# Profile: single-device recipes. torch_npu 2.11.0 is verified to
# work for these on coder npu-3; the previous blocker was the per-job
# ModelScope download failing on cold /root/.cache/modelscope, now
# moved to cache-seed/torchtune/ms_seeds.yaml plant.
setup_torchtune_single() {
  ensure_torch_stack 2.11.0
  install_torchtune_pkg

  # lm_eval is only needed for the eleuther_eval recipe (not declared as
  # an upstream dep of torchtune). The recipe's __init__ checks
  # `version("lm-eval") < "0.4.5"` (recipes/eleuther_eval.py:446) using
  # STRING comparison (importlib.metadata.version returns a string, no
  # version coercion), not packaging.version. So "0.4.13" < "0.4.5" is
  # True ("0.4.1" prefix beats "0.4.5" lexicographically) and 0.4.10-0.4.49
  # all get rejected — only 0.4.5-0.4.9 and 0.4.50+ pass. Pin exact 0.4.5
  # (lowest acceptable) to keep the floor obvious; the import chain only
  # uses evaluator/models/tasks/utils which is stable across 0.4.x.
  python -m pip install "lm-eval==0.4.5"

  # Resolve model snapshot paths from the cache-seed workflow. The
  # shared HF hub cache is populated by ms_seed.py per
  # cache-seed/torchtune/ms_seeds.yaml (Qwen/Qwen2.5-0.5B-Instruct for
  # student / SFT base; Qwen/Qwen2.5-1.5B-Instruct for KD teacher).
  # No network access in this leg — TT_MODEL_PATH / TT_TEACHER_PATH
  # point into ~/.cache/huggingface/hub/.
  resolve_seed_envs Qwen/Qwen2.5-0.5B-Instruct TT_MODEL_PATH
  resolve_seed_envs Qwen/Qwen2.5-1.5B-Instruct TT_TEACHER_PATH
}

# Profile: distributed recipes. torch_npu stays at 2.11.0 — verified
# on coder npu-3 (2026-09-22) that hccl init succeeds AND that the
# two upstream bugs patched by patch_main_head_bugs() below are
# sufficient for end-to-end exit 0 (Loss 2.92→0.47 for full_finetune,
# 49 steps in 12s). See patch_main_head_bugs comments for the exact
# lines.
setup_torchtune_distributed() {
  ensure_torch_stack 2.11.0
  install_torchtune_pkg

  resolve_seed_envs Qwen/Qwen2.5-0.5B-Instruct TT_MODEL_PATH
  resolve_seed_envs Qwen/Qwen2.5-1.5B-Instruct TT_TEACHER_PATH
}

# Profile: PPO full finetune single_device. Needs the RM
# smohammadi/tinyllama_rm_sentiment_1b in addition to the base
# single-device stack. RM is delivered by the curl plant at
# cache-seed/torchtune/curl_seeds.yaml (xet-backed file, can't go
# through huggingface_hub 0.36.2; ms_seed.py doesn't apply because
# the repo isn't on ModelScope).
setup_torchtune_ppo() {
  setup_torchtune_single
  if [[ -z "${TT_RM_PATH:-}" ]]; then
    resolve_seed_envs smohammadi/tinyllama_rm_sentiment_1b TT_RM_PATH
  else
    echo "TT_RM_PATH (caller-provided): ${TT_RM_PATH}"
  fi
}

# Patch upstream bugs in main HEAD torchtune (and v0.6.1 distributed
# recipes that hardcode a CUDA-only backend string + a v0.6.1
# _broadcast_tensor that doesn't handle hccl). Each block has a
# grep/string guard so the patch becomes a no-op on refs where the
# anchor is absent — no script change needed if torchtune upstream
# ships a fix. Operates on the SOURCE in $TARGET_ROOT before
# `pip install` copies it to site-packages, so the installed files
# inherit the fixes.
patch_main_head_bugs() {
  local dpo="$TARGET_ROOT/torchtune/rlhf/loss/dpo.py"
  local quant="$TARGET_ROOT/torchtune/training/quantization.py"
  local dist_py="$TARGET_ROOT/torchtune/training/_distributed.py"

  # Bug 1: torchtune/rlhf/loss/dpo.py:14 does
  #     T = TypeVar("T", bound=dataclass)
  # but the header only imports torch + torchtune internals — no
  # `from typing import TypeVar`, no `from typing import Optional/Tuple`,
  # no `from dataclasses import dataclass`. The class body itself
  # also references Optional[T] / Tuple[...] at type annotations on
  # PreferenceLoss.forward, so all four names must be in scope at
  # class-definition time (not just call time). `import torchtune.rlhf.loss`
  # → `from .dpo import DPOLoss` → NameError cascading from TypeVar →
  # Optional. Triggered by `lora_dpo_single_device.py` via overlay_args
  # `loss._component_=torchtune.rlhf.loss.DPOLoss`.
  #
  # Guard: `from torchtune.utils._logging import deprecated` exists in
  # main HEAD only (v0.6.1's dpo.py has no such import line at all —
  # it's a different file shape). The previous guard
  # `! grep -q "^from typing import.*TypeVar"` was wrong: v0.6.1 also
  # has no TypeVar import, so the guard was TRUE there and the assert
  # inside the python heredoc blew up with
  # `AssertionError: anchor missing in dpo.py` (CI run 35329364605,
  # tested ref=v0.6.1). Using the deprecated-import as the discriminator
  # means the patch only fires on ref where the bug is actually present.
  if [[ -f "$dpo" ]] && grep -qF "from torchtune.utils._logging import deprecated" "$dpo"; then
    echo "patching $dpo: adding typing/dataclass imports (main HEAD bug)"
    _PATCH_DPO="$dpo" python - <<'PY'
import pathlib, os, sys
p = pathlib.Path(os.environ["_PATCH_DPO"])
src = p.read_text()
needle = "from torchtune.utils._logging import deprecated\n"
assert needle in src, "anchor missing in dpo.py: %s" % p
addition = "from typing import Optional, Tuple, TypeVar\nfrom dataclasses import dataclass\n"
if "from typing import" not in src.split(needle)[0]:
    src = src.replace(needle, needle + addition, 1)
    p.write_text(src)
    print("  patched: %s" % p)
else:
    print("  already patched (race), skipping: %s" % p)
PY
  fi

  # Bug 2: torchtune/training/quantization.py imports `from torch import nn`
  # only (no bare `import torch`), but the Int8DynActInt4WeightQuantizer
  # uses `weight_dtype=torch.int4` at the call site. `quantize.py`
  # recipe triggers this and aborts with NameError. Add bare `import torch`
  # so the `torch.int4` lookup resolves.
  #
  # Guard: `weight_dtype=torch.int4` exists in main HEAD only. v0.6.1's
  # Int8DynActInt4WeightQuantizer uses the older
  # `int8_dynamic_activation_int4_weight(groupsize)` callable from torchao
  # 0.13, no `torch.int4` lookup, so no patch needed there. Same mistake
  # as the dpo guard above: the previous `! grep -q "^import torch$"`
  # check was TRUE for v0.6.1 too (v0.6.1 has no bare `import torch`
  # either, only `from torch import nn`), so the patch would fire and
  # silently add a redundant `import torch` — harmless but misleading.
  if [[ -f "$quant" ]] && grep -qF "weight_dtype=torch.int4" "$quant"; then
    echo "patching $quant: adding bare 'import torch' (main HEAD bug)"
    _PATCH_QUANT="$quant" python - <<'PY'
import pathlib, os
p = pathlib.Path(os.environ["_PATCH_QUANT"])
src = p.read_text()
needle = "from typing import Callable, Optional\n"
assert needle in src, "anchor missing in quantization.py: %s" % p
addition = "import torch\n"
if "import torch\n" not in src.split("from typing import Callable, Optional\n")[0]:
    src = src.replace(needle, needle + addition, 1)
    p.write_text(src)
    print("  patched: %s" % p)
else:
    print("  already patched (race), skipping: %s" % p)
PY
  fi

  # Bug 3: torchtune/training/_distributed.py:_broadcast_tensor handles
  # nccl by moving CPU tensors to CUDA before broadcast:
  #     if dist.get_backend() == "nccl":
  #         tensor = tensor.to(get_device("cuda"))
  #     dist.broadcast(tensor, src=src, group=None)
  # but does NOT handle hccl. On NPU, training.set_seed builds
  # `rand_seed` on CPU (`torch.empty(1, dtype=torch.int64).random_()`)
  # and `_broadcast_tensor` then hits hccl with a CPU tensor →
  # RuntimeError: No backend type associated with device type cpu
  # (CI run 35721666282 full_finetune_distributed leg). Verified fix on
  # coder npu-3 (2026-09-22): mirror the nccl branch for hccl, moving
  # CPU → NPU before broadcast and back after.
  #
  # Guard: `if dist.get_backend() == "nccl":` exists in both v0.6.1 and
  # main HEAD; we extend this conditional rather than replace it, so
  # the anchor remains valid regardless of upstream edits.
  if [[ -f "$dist_py" ]] && grep -qF 'if dist.get_backend() == "nccl":' "$dist_py"; then
    echo "patching $dist_py: extend _broadcast_tensor to handle hccl (CPU → NPU)"
    _PATCH_DIST="$dist_py" python - <<'PY'
import pathlib, os
p = pathlib.Path(os.environ["_PATCH_DIST"])
src = p.read_text()
needle = '''        if dist.get_backend() == "nccl":
            tensor = tensor.to(get_device("cuda"))
        dist.broadcast(tensor, src=src, group=None)'''
addition_template = '''        backend = dist.get_backend()
        if backend == "nccl":
            tensor = tensor.to(get_device("cuda"))
        elif backend == "hccl":
            # hccl backend only supports NPU tensors; torchtune creates
            # rand_seed on CPU in training.set_seed, so move it to NPU
            # before broadcast and back after. Mirrors the nccl branch
            # above (per torch.distributed.pipelining × torch_npu c10d
            # ABI gap memory). NPU-only patch — no effect on CUDA runs.
            tensor = tensor.to(get_device("npu"))
        dist.broadcast(tensor, src=src, group=None)'''
if needle not in src:
    raise SystemExit("anchor missing in _distributed.py: %s" % p)
if 'elif backend == "hccl":' in src:
    print("  already patched (race), skipping: %s" % p)
else:
    src = src.replace(needle, addition_template, 1)
    p.write_text(src)
    print("  patched: %s" % p)
PY
  fi

  # Bug 4: 4 distributed recipes hardcode
  #     init_process_group("cuda:nccl,cpu:gloo")
  # at recipe_main() before constructing the recipe. The multi-backend
  # string is parsed by torch.distributed as "use nccl for cuda tensors,
  # gloo for cpu tensors" — and torch.distributed on the CI image
  # (torch==2.11.0+cpu) does NOT ship the nccl backend → init raises
  # "Distributed package doesn't have NCCL built in" (CI run 35721666282,
  # lora_dpo_distributed / lora_finetune_distributed / full_dpo_distributed
  # / knowledge_distillation_distributed legs). On NPU we never use nccl
  # or gloo anyway (hccl handles NPU tensors, world_size=1 means no
  # actual collectives), so the simplest fix is to call init_process_group
  # with just "hccl". Verified fix on coder npu-3 (2026-09-22): all 4
  # recipes exit 0 after the patch.
  #
  # Guard: anchor is the literal `init_process_group("cuda:nccl,cpu:gloo")`
  # call site (one per recipe); the string is unique per file so a plain
  # `grep -lF` is enough.
  for recipe in lora_dpo_distributed.py lora_finetune_distributed.py \
                full_dpo_distributed.py knowledge_distillation_distributed.py; do
    local rp="$TARGET_ROOT/recipes/$recipe"
    if [[ -f "$rp" ]] && grep -qF 'init_process_group("cuda:nccl,cpu:gloo")' "$rp"; then
      echo "patching $rp: replace cuda:nccl,cpu:gloo → hccl (NPU-only)"
      _PATCH_RECIPE="$rp" python - <<'PY'
import pathlib, os
p = pathlib.Path(os.environ["_PATCH_RECIPE"])
src = p.read_text()
needle = 'init_process_group("cuda:nccl,cpu:gloo")'
replacement = 'init_process_group("hccl")'
if needle not in src:
    raise SystemExit("anchor missing in %s" % p)
if replacement in src:
    print("  already patched (race), skipping: %s" % p)
else:
    src = src.replace(needle, replacement, 1)
    p.write_text(src)
    print("  patched: %s" % p)
PY
    fi
  done

  # Bug 5: lora_dpo_distributed.py:691 and full_dpo_distributed.py:886
  # both do:
  #     num_tokens = 0
  #     ...
  #     num_tokens += torch.tensor(batch[0].numel())
  # and later call
  #     torch.distributed.all_reduce(num_tokens)
  # `torch.tensor(...)` defaults to CPU, so `num_tokens` becomes a CPU
  # scalar tensor — and the hccl backend only accepts NPU tensors for
  # collectives (init succeeded thanks to Bug 4 patch, but the first
  # all_reduce on a CPU tensor raises
  # `RuntimeError: No backend type associated with device type cpu`,
  # CI run 35729818620 lora_dpo_distributed + full_dpo_distributed
  # legs). lora_finetune_distributed / full_finetune_distributed /
  # knowledge_distillation_distributed use a different num_tokens
  # expression (device tensor from `(batch["labels"] != ignore).sum()`
  # after batch_to_device) so they're already on-device and don't need
  # this patch.
  #
  # Fix: route the new tensor through self._device, mirroring how the
  # other recipes keep it device-resident. Single line per file.
  #
  # Guard: `num_tokens += torch.tensor(batch[0].numel())` is the only
  # literal match in the file (unique per recipe, line numbers shifted
  # by upstream edits but the anchor string survives).
  for recipe in lora_dpo_distributed.py full_dpo_distributed.py; do
    local rp="$TARGET_ROOT/recipes/$recipe"
    if [[ -f "$rp" ]] && grep -qF 'num_tokens += torch.tensor(batch[0].numel())' "$rp"; then
      echo "patching $rp: route num_tokens += through self._device (NPU-only)"
      _PATCH_RECIPE="$rp" python - <<'PY'
import pathlib, os
p = pathlib.Path(os.environ["_PATCH_RECIPE"])
src = p.read_text()
needle = 'num_tokens += torch.tensor(batch[0].numel())'
replacement = 'num_tokens += torch.tensor(batch[0].numel(), device=self._device)'
if needle not in src:
    raise SystemExit("anchor missing in %s" % p)
if replacement in src:
    print("  already patched (race), skipping: %s" % p)
else:
    src = src.replace(needle, replacement, 1)
    p.write_text(src)
    print("  patched: %s" % p)
PY
    fi
  done
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

HERE=$(cd "$(dirname "$0")" && pwd)
export PIP_CONSTRAINT="$(cd "$HERE/.." && pwd)/constraints-npu.txt"

# recipes/eleuther_eval.py:311 hard-codes `super().__init__(pretrained="gpt2", ...)`
# which triggers transformers.AutoConfig.from_pretrained("gpt2"). On coder pods
# huggingface.co is unreachable (curl hang); on CI runner it's fine. Setting
# HF_ENDPOINT=https://hf-mirror.com routes both cases through the China mirror
# for the gpt2 config only (the torchtune model itself comes from cache-seed
# into the HF hub cache, so this doesn't affect Qwen2.5-0.5B-Instruct — that
# asset's resolve_seed_envs lookup is local-cache only).
# CI cluster has direct HF egress; on direct egress the env is harmless (just
# changes the endpoint). Idempotent: if the runner already exports HF_ENDPOINT
# the keep-existing behavior lets operators override the mirror.
if [[ -z "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT=https://hf-mirror.com
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip setuptools wheel
prepare_fixtures

"setup_${PROFILE}"