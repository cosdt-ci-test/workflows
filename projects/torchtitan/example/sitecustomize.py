"""Startup hook: CPython imports ``sitecustomize`` when this directory is on ``PYTHONPATH``.

It installs the torch 2.15 compatibility shim (``npu_torch_compat``) before
anything imports torchtitan.  torchtitan's own import chain reaches
``torch.distributed.pipelining`` very early -- through the optional Dist-MoE
package that every model flavor pulls in -- so the shim cannot live inside
the recipe / override modules: those are imported too late.  See
``npu_torch_compat.py`` for the details.

Every statement is guarded: a broken shim must never keep the interpreter
from starting.
"""

try:
    import npu_torch_compat
except Exception:  # pragma: no cover - the shim is best-effort
    pass
else:
    try:
        npu_torch_compat.install()
    except Exception:  # pragma: no cover - the shim is best-effort
        pass
