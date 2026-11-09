"""Compatibility shim: torchtitan >= 0.4.0 imports two torch >= 2.15 symbols.

torchtitan v0.4.0 pulls its **optional** Dist-MoE support in eagerly, and
that package's runtime module starts with::

    from torch.distributed.pipelining import (
        analyze_pipeline_activation_liveness,
        PipelineStageInfo,
    )

Both names were introduced in **torch 2.15.0**.  Ascend publishes no
``torch_npu`` newer than 2.14.0 (and ``torch_npu`` pins ``torch==2.14.0.*``),
so torch 2.14.0 is the newest torch the NPU stack can run -- and on it the
import chain of *every* torchtitan model flavor dies before a model is built::

    torchtitan.train -> torchtitan.trainer -> torchtitan.components.data
      -> torchtitan.models.deepseek_v3 -> torchtitan.config.transform
      -> torchtitan.models.common.dist_moe -> runtime
      -> ImportError: cannot import name 'analyze_pipeline_activation_liveness'

(the same two names are missing from torch 2.13.0, so the ceiling is the
torch version, not a fluke of one release).

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

Nothing in the upstream torchtitan checkout is modified: the shim lives next
to the NPU override modules and is activated either explicitly
(``import npu_torch_compat; npu_torch_compat.install()``) or automatically
through ``sitecustomize.py`` -- CPython imports ``sitecustomize`` at startup
whenever the directory holding it is on ``PYTHONPATH``, which is early enough
(torchtitan's own import chain reaches ``torch.distributed.pipelining``
before any user module is loaded).
"""

from __future__ import annotations

import dataclasses
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


def install() -> bool:
    """Add the torch 2.15 symbols torchtitan >= 0.4.0 imports.

    Returns:
        True when something was actually patched (torch < 2.15), False when
        the symbols already exist (torch >= 2.15) or when torch is not
        importable at all.
    """
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


__all__ = [
    "PipelineStageInfo",
    "install",
]
