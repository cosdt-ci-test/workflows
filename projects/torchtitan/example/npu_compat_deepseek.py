"""NPU override for the deepseek_v3 family.

deepseek_v3's attention is its own class (``models/deepseek_v3/attention.py``:
``class DeepSeekMLAAttention(Attention)``), not ``GQAttention`` -- so the
override in ``npu_compat.py`` does not reach it and the backend it asks for
cannot compile on NPU.  This module adds the same two fixes for that class:

* the complex64 RoPE cache is gathered in real space
  (``RealGatherComplexRoPE``, shared with ``npu_compat``);
* the innermost attention becomes an SDPA with an explicit dense
  block-causal mask (``NpuBlockCausalSDPA``, shared with ``npu_compat``).

deepseek_v3 uses multi-head latent attention, so its kernel sits one level
deeper than llama3's: ``DeepSeekMLAAttention.Config.inner_attention`` is an
``MLAAttention.Config``, whose own ``inner_attention`` is the kernel that
actually runs the attention.  The override therefore swaps the *nested*
kernel.

Kept in a separate module from ``npu_compat`` on purpose: importing
``torchtitan.models.deepseek_v3`` pulls in that family's own dependencies,
and the llama3 / qwen3 entries must not depend on them.

Activate with::

    --override npu_compat_deepseek.deepseek_attention
"""

from dataclasses import dataclass

from torchtitan.config.override import derive, override
from torchtitan.models.common.attention import MLAAttention
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.deepseek_v3.attention import DeepSeekMLAAttention

from npu_compat import NpuBlockCausalSDPA, RealGatherComplexRoPE


class NpuDeepSeekMLAAttention(DeepSeekMLAAttention):
    """upstream ``deepseek_v3`` attention, wired for NPU.

    The forward is upstream's: it already forwards the per-document metadata
    to the inner attention, and the NPU inner attention rebuilds the
    block-causal mask from it.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekMLAAttention.Config):
        pass


@override(
    target=DeepSeekMLAAttention.Config,
    description=(
        "deepseek_v3 attention: real-space RoPE gather + SDPA with an "
        "explicit block-causal mask"
    ),
)
def deepseek_attention(
    cfg: DeepSeekMLAAttention.Config,
) -> NpuDeepSeekMLAAttention.Config:
    new = derive(cfg, NpuDeepSeekMLAAttention.Config)
    if isinstance(cfg.rope, ComplexRoPE.Config):
        new.rope = derive(cfg.rope, RealGatherComplexRoPE.Config)
    # The MLA backend owns the kernel one level down.
    mla = derive(cfg.inner_attention, MLAAttention.Config)
    # derive() keeps the shared fields (sharding_config among them) that a
    # fresh Config() would silently drop.
    mla.inner_attention = derive(
        cfg.inner_attention.inner_attention, NpuBlockCausalSDPA.Config
    )
    new.inner_attention = mla
    return new


__all__ = ["NpuDeepSeekMLAAttention", "deepseek_attention"]
