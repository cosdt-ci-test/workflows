"""Compatibility shim: torchtitan >= 0.4.0 needs torch >= 2.15 APIs.

torchtitan v0.4.0 is developed against torch 2.15 (its own CI pins
``torch==2.15.0``, and ``torchtitan/observability/README.md`` asks for "a
PyTorch nightly from 2026-09-21 or later").  Ascend publishes no ``torch_npu``
newer than 2.14.0 (and ``torch_npu`` pins ``torch==2.14.0.*``), so torch
2.14.0 is the newest torch the NPU stack can run.  Two torch-2.15 APIs that
torchtitan v0.4.0 uses therefore have to be bridged down to 2.14.

**Gap 1 -- ``torch.distributed.pipelining`` symbols.**

torchtitan v0.4.0 pulls its **optional** Dist-MoE support in eagerly, and
that package's runtime module starts with::

    from torch.distributed.pipelining import (
        analyze_pipeline_activation_liveness,
        PipelineStageInfo,
    )

Both names were introduced in **torch 2.15.0**.  On torch 2.14 the import
chain of *every* torchtitan model flavor dies before a model is built::

    torchtitan.train -> torchtitan.trainer -> torchtitan.components.data
      -> torchtitan.models.deepseek_v3 -> torchtitan.config.transform
      -> torchtitan.models.common.dist_moe -> runtime
      -> ImportError: cannot import name 'analyze_pipeline_activation_liveness'

This module installs the two missing names into
``torch.distributed.pipelining``:

* ``PipelineStageInfo`` -- copied from torch 2.15: a frozen, keyword-only
  dataclass with three fields.  torchtitan only uses it as a type and as a
  value object inside the Dist-MoE runtime.
* ``analyze_pipeline_activation_liveness`` -- a placeholder that raises
  ``RuntimeError``.  The real function computes pipeline activation slot
  plans for the Dist-MoE runtime; it needs torch >= 2.15 *and* the optional
  ``dist_moe`` package (``git+https://github.com/meta-pytorch/dist_moe.git@main``),
  which is not installed here either.  The quick-start never runs Dist-MoE,
  so the placeholder is never reached -- it exists only so that the import
  succeeds, and it fails loudly instead of silently if it is ever called.

**Gap 2 -- ``MixedPrecisionPolicy(param_dtype_override_fn=...)``.**

``torchtitan/distributed/fsdp.py`` builds the FSDP2 mixed-precision policy
with::

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
        param_dtype_override_fn=param_dtype_override_fn,
    )

``param_dtype_override_fn`` is a torch 2.15 addition (a callback that lets
FSDP pick a per-parameter dtype); torch 2.14's
``torch.distributed.fsdp.MixedPrecisionPolicy`` is a frozen dataclass with
only ``param_dtype`` / ``reduce_dtype`` / ``output_dtype`` /
``cast_forward_inputs``, so the call raises::

    TypeError: MixedPrecisionPolicy.__init__() got an unexpected keyword
    argument 'param_dtype_override_fn'

upstream torchtitan only ever passes ``None`` for that callback (the
parameter defaults to ``None`` and no production caller overrides it -- only
``tests/unit_tests/cpu/distributed/test_fsdp.py`` passes a real one), so the
shim accepts the keyword and drops it when it is ``None``, which is exactly
the behaviour of torch 2.15 with a ``None`` callback.  A non-``None``
callback is refused loudly: torch 2.14's FSDP2 has no per-parameter dtype
override at all, so silently ignoring it would change the numerics.

Nothing in the upstream torchtitan checkout is modified: the shim lives next
to the NPU override modules and is activated either explicitly
(``import npu_torch_compat; npu_torch_compat.install()``) or automatically
through ``sitecustomize.py`` -- CPython imports ``sitecustomize`` at startup
whenever the directory holding it is on ``PYTHONPATH``, which is early enough
(torchtitan's own import chain reaches ``torch.distributed.pipelining``
before any user module is loaded, and the FSDP policy is only built while the
model is parallelized).
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PipelineStageInfo:
    """torch 2.15's ``torch.distributed.pipelining.PipelineStageInfo``.

    Attributes:
        stage_index: Global logical pipeline-stage index.
        microbatch_index: Microbatch index within the current pipeline step.
        is_metadata_inference: Whether this invocation is the representative
            dynamic metadata-inference probe rather than real execution.
    """

    stage_index: int
    microbatch_index: int
    is_metadata_inference: bool = False


def _analyze_pipeline_activation_liveness(*args: Any, **kwargs: Any) -> Any:
    """Placeholder for torch 2.15's pipeline activation liveness analysis."""
    raise RuntimeError(
        "analyze_pipeline_activation_liveness requires torch >= 2.15; "
        "Dist-MoE is not available on the torch 2.14 / torch_npu 2.14 stack."
    )


def _patch_pipelining_symbols() -> bool:
    """Add the two torch 2.15 names to ``torch.distributed.pipelining``."""
    try:
        import torch.distributed.pipelining as pipelining
    except Exception:
        # No torch (yet) -- nothing to patch; the caller imports torchtitan
        # later, and by then torch is importable and the patch happens.
        return False

    patched = False
    if not hasattr(pipelining, "PipelineStageInfo"):
        pipelining.PipelineStageInfo = PipelineStageInfo  # type: ignore[attr-defined]
        patched = True
    if not hasattr(pipelining, "analyze_pipeline_activation_liveness"):
        pipelining.analyze_pipeline_activation_liveness = (  # type: ignore[attr-defined]
            _analyze_pipeline_activation_liveness
        )
        patched = True
    return patched


def _patch_mixed_precision_policy() -> bool:
    """Accept (and drop) torch 2.15's ``param_dtype_override_fn`` keyword.

    The class object itself is patched, not the name binding in
    ``torchtitan.distributed.fsdp`` -- upstream imports the class by name at
    import time, so patching the class reaches every holder of it.
    """
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy
    except Exception:
        return False

    try:
        params = inspect.signature(MixedPrecisionPolicy.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False
    if "param_dtype_override_fn" in params:
        # torch >= 2.15 has the keyword natively -- nothing to bridge.
        return False

    original_init = MixedPrecisionPolicy.__init__

    def __init__(
        self,
        *args: Any,
        param_dtype_override_fn: Any = None,
        **kwargs: Any,
    ) -> None:
        if param_dtype_override_fn is not None:
            raise RuntimeError(
                "MixedPrecisionPolicy(param_dtype_override_fn=...) requires "
                "torch >= 2.15; per-parameter dtype overrides are not "
                "available on the torch 2.14 / torch_npu 2.14 stack."
            )
        original_init(self, *args, **kwargs)

    MixedPrecisionPolicy.__init__ = __init__  # type: ignore[method-assign]
    return True


def install() -> bool:
    """Install the torch 2.15 compatibility shims torchtitan >= 0.4.0 needs.

    Returns:
        True when something was actually patched (torch < 2.15), False when
        the APIs already exist (torch >= 2.15) or when torch is not
        importable at all.
    """
    patched = False
    patched |= _patch_pipelining_symbols()
    patched |= _patch_mixed_precision_policy()
    return patched


__all__ = [
    "PipelineStageInfo",
    "install",
]
