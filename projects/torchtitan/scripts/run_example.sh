#!/usr/bin/env bash
# Run one torchtitan example from a CI working copy of the target tree.
# $1 is the manifest entry path. Overlay CLI args come from OVERLAY_ARGS
# (JSON array, possibly []). Shell launchers already forward "$@"; this
# script never patches them. Never git add/commit/push.
#
# The manifest stores two fields per supported entry:
#   path: the upstream file we want to verify exists (used by
#         manifest-check, never executed)
#   exec: the .sh launcher setup writes under $TARGET_ROOT/scripts/
#         (execs `torchrun -m torchtitan.train "$@"` so LOCAL_RANK
#         is set by torchrun — running `python torchtitan/train.py`
#         directly crashes on `LOCAL_RANK must be set`)
# When EXEC is set we run the launcher; otherwise we fall back to
# the path field (ad-hoc Python entry points).
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
# The engine exports EXEC = manifest.entry.exec when set; run_example
# scripts for projects that have no exec field (ad-hoc .py entries)
# silently keep the legacy behaviour.
EXEC_REL="${EXEC:-}"

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"
[[ -e "$EXAMPLE_PATH" ]] || { echo "example not found: $EXAMPLE_PATH" >&2; exit 1; }

if [[ -n "$EXEC_REL" ]]; then
  LAUNCH_PATH="$TARGET_ROOT/$EXEC_REL"
else
  LAUNCH_PATH="$EXAMPLE_PATH"
fi
if [[ ! -f "$LAUNCH_PATH" ]]; then
  echo "launchable file not found: $LAUNCH_PATH" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch, torch_npu; print('NPU available:', torch.npu.is_available(), 'devices:', torch.npu.device_count())"

expand_overlay() {
  # The workflow serializes manifest.overlay_args as JSON. Expand each
  # item with shell quoting intact, then allow CI paths such as
  # ${CI_OUTPUT_DIR} to resolve only in this job's environment.
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
    if not isinstance(item, str) or not item.strip():
        raise SystemExit('OVERLAY_ARGS items must be non-empty strings')
    tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
print(' '.join(shlex.quote(token) for token in tokens))
PY
}

eval "EXTRA_ARGS=( $(expand_overlay) )"

echo "running $LAUNCH_PATH with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

# sitecustomize.py injects the CUDA->NPU transfer at interpreter
# startup. torchtitan imports torch but does not pin device="cuda"
# directly; however, downstream deps (datasets, accelerate's plugin
# loader, fbgemm-like helpers) sometimes do, and the c10d backend map
# test in run_example.sh checks npu.is_available(). The transfer
# itself is a no-op if no torch.cuda call has been made yet.
#
# transfer_to_npu patches torch.Tensor.is_cuda = torch.Tensor.is_npu
# and wraps torch.cuda.get_device_capability -> torch.npu.get_device_capability.
# torch 2.12's c10d broadcast() then evaluates
# `tensor.is_cuda and torch.cuda.get_device_capability(tensor.device)[0] >= 9`
# (sm90 check), and torch.npu.get_device_capability returns None unless
# TORCH_NPU_DEVICE_CAPABILITY is set -> `None[0]` TypeError in
# set_determinism / DTensor OffsetBasedRNGTracker (run 35224441230).
# Setting the capability env *before* the transfer import makes the
# shim return (9, 0) so the sm90 check resolves instead of crashing.
# torch 2.12 is required (torch < 2.12 lacks torch.distributed._local_tensor,
# which spmd_types==0.2.3 imports) so this env var is the fix, not a downgrade.
prepare_shims() {
  local shim_dir="$GITHUB_WORKSPACE/ci_patch"
  mkdir -p "$shim_dir"
  cat > "$shim_dir/sitecustomize.py" <<'PY'
import os
os.environ.setdefault("TORCH_NPU_DEVICE_CAPABILITY", "9.0")

# --- torch.distributed.set_timeout fallback (torch 2.12 missing) ---
# set_pg_timeouts in torchtitan/distributed/utils.py:595 calls
# torch.distributed.set_timeout(timeout, group). torch 2.12 ships only the
# ProcessGroup.set_timeout() -- no module-level alias. Without this shim,
# the trainer crashes inside __init__ before the first step.
import torch.distributed as _dist
if not hasattr(_dist, "set_timeout"):
    _get_default_group = _dist.distributed_c10d._get_default_group
    def _set_timeout(timeout, group=None):
        g = group if group is not None else _get_default_group()
        g.set_timeout(timeout)
    _dist.set_timeout = _set_timeout

# --- transfer_to_npu (cuda->npu) ---
from torch_npu.contrib import transfer_to_npu  # noqa: F401

# --- create_block_mask kwarg shim (separate_full_blocks, torch>=2.13 only) ---
# torchtitan v0.3.0 passes separate_full_blocks=... to create_block_mask
# in torchtitan/models/common/decoder.py:274. torch 2.12's signature does
# not accept it. Strip the kwarg before delegating.
import functools as _ft
import torch as _t
_orig_create_block_mask = _t.nn.attention.flex_attention.create_block_mask
@_ft.wraps(_orig_create_block_mask)
def _shim_create_block_mask(*args, **kwargs):
    kwargs.pop("separate_full_blocks", None)
    return _orig_create_block_mask(*args, **kwargs)
_t.nn.attention.flex_attention.create_block_mask = _shim_create_block_mask

# --- force ALL llama3 attn_backends to SDPA (avoids FlexAttention CANN
# compile wall + ComplexRoPE aclnnIndex DT_COMPLEX64 wall) ---
# v0.3.0 config_registry hardcodes attn_backend="flex" for llama3_debugmodel*.
# get_attention_config rejects "sdpa". We monkey-patch the lookup to map every
# backend to ScaledDotProductAttention.Config(), bypassing both walls in one
# shot. The ce_loss path was empirically verified (2026-09-22 on hdc-stable-npu-2
# with this shim, training completed step 1-2 with loss 8.16->7.88, exit 0).
import torchtitan.models.llama3 as _llama3_mod
import torchtitan.models.common.attention as _attn_mod
from torchtitan.models.common.attention import ScaledDotProductAttention
def _force_sdpa(_backend):
    return ScaledDotProductAttention.Config()
_llama3_mod.get_attention_config = _force_sdpa
_attn_mod.get_attention_config = _force_sdpa

# --- ComplexRoPE -> CosSinRoPE(scaling="none") shim ---
# Even with SDPA, llama3 model_registry still constructs
# ComplexRoPE.Config(scaling="llama", ...). complex64 rope caches trigger
# aclnnIndex DT_COMPLEX64 not implemented on NPU. Map any ComplexRoPE.Config
# construction to a real-valued CosSinRoPE.Config(scaling="none", ...).
from torchtitan.models.common.rope import ComplexRoPE, CosSinRoPE
def _shim_complex_config(**kwargs):
    kwargs.pop("scaling", None)
    return CosSinRoPE.Config(scaling="none", **kwargs)
ComplexRoPE.Config = classmethod(lambda cls, **kw: _shim_complex_config(**kw))

# --- ChunkedLossWrapper bypass shim (NPU autograd.Function meta-leak) ---
# torchtitan v0.3.0 default llama3_debugmodel uses ChunkedLossWrapper, which
# saves accumulated_grad via ctx.save_for_backward inside
# _DecoderOutputGradientBackProp.forward and returns it from
# _DecoderOutputGradientBackProp.backward as the grad for hidden_states.
# On NPU (torch_npu 2.12.0 + CANN 9.1.0), the C++ autograd engine raises
# "RuntimeError: The tensor has a non-zero number of elements, but its data
# is not allocated yet." the moment this custom-Function-returned grad enters
# the decoder backward graph, regardless of save_for_backward data_ptr
# validity (verified empirically on hdc-stable-npu-2 2026-09-22 with three
# patches that all failed identically: clone accumulated_grad before save,
# contiguous + npu.synchronize around the call site, replace the Function
# with a manual hidden_states.backward). The saved tensor at backward entry
# is a real npu:0 bf16 tensor with a valid data_ptr; the leak is in how
# torch_npu's autograd engine wires the Function-returned grad into the
# downstream graph traversal.
#
# Workaround: replace ChunkedLossWrapper.__call__ with a passthrough that
# runs lm_head once (to keep FSDP's all_gather_state consistent) and then
# delegates to vanilla CrossEntropyLoss on the logits. ce_loss variant
# already takes this path and is verified exit 0 with the same shim set.
# This drops ChunkedLossWrapper's num_chunks memory-saving benefit on the
# smoke path; we're keeping the no-source-patch policy by staying in
# sitecustomize. Upstream fix is on torch_npu autograd; until then this
# shim is the only way to keep the default / dist_gemm / sft 1-card smoke
# entries running.
import torchtitan.components.loss as _clw_loss_mod
from torchtitan.components.loss import CrossEntropyLoss as _CE
_orig_clw_init = _clw_loss_mod.ChunkedLossWrapper.__init__
def _patched_clw_init(self, config, *, compile_config=None):
    _orig_clw_init(self, config, compile_config=compile_config)
    self._ce_loss = _CE(_CE.Config())
def _patched_clw_call(self, pred, labels, global_valid_tokens=None, **loss_inputs):
    if self.lm_head is not None:
        logits = self.lm_head(pred)
    else:
        logits = pred
    return self._ce_loss(logits, labels, global_valid_tokens, **loss_inputs)
_clw_loss_mod.ChunkedLossWrapper.__init__ = _patched_clw_init
_clw_loss_mod.ChunkedLossWrapper.__call__ = _patched_clw_call
_clw_loss_mod.ChunkedLossWrapper.set_lm_head = lambda self, lm_head: setattr(self, "lm_head", lm_head) or None
PY
  export PYTHONPATH="$shim_dir:${PYTHONPATH:-}"
}

prepare_shims

# .sh launchers in scripts/ (written by setup_example.sh) cd to
# $TARGET_ROOT internally and exec torchrun -m torchtitan.train;
# we still cd to the launcher's dir for logging consistency.
case "$LAUNCH_PATH" in
  *.sh)
    cd "$(dirname "$LAUNCH_PATH")"
    bash "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *)
    cd "$TARGET_ROOT"
    python "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    ;;
esac
