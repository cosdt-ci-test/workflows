"""NPU override for the deepseek_v3 family.

deepseek_v3's attention is its own class (``models/deepseek_v3/model.py``:
``class Attention(BaseAttention)``), not a ``GQAttention`` -- so the
override in ``npu_compat.py`` does not reach it and the flex backend it
asks for cannot compile on NPU.  This module adds the same two fixes for
that class:

* the complex64 RoPE cache is gathered in real space
  (``RealGatherComplexRoPE``, shared with ``npu_compat``);
* the inner attention becomes an SDPA with an explicit dense block-causal
  mask (``BlockCausalSDPA``, shared with ``npu_compat``).

The attention body is upstream v0.3.0's ``Attention.forward`` with exactly
one addition: ``positions=positions`` in the inner-attention call, because
upstream drops the positions before the inner attention and a non-Flex
backend needs them to rebuild the per-document mask.

Kept in a separate module from ``npu_compat`` on purpose: importing
``torchtitan.models.deepseek_v3`` pulls in that family's own dependencies,
and the llama3 / qwen3 / muse_glimmer entries must not depend on them.

Activate with::

    --override.imports npu_compat_deepseek.deepseek_attention
"""

from dataclasses import dataclass

import torch
from spmd_types import runtime as spmd

from npu_compat import BlockCausalSDPA, RealGatherComplexRoPE

from torchtitan.config.override import derive, override
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.deepseek_v3.model import Attention as DeepSeekAttention


class NpuDeepSeekAttention(DeepSeekAttention):
    """upstream ``deepseek_v3`` attention + positions passthrough."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekAttention.Config):
        pass

    def forward(
        self,
        x: torch.Tensor,
        attention_masks,
        positions: torch.Tensor | None = None,
    ):
        # upstream v0.3.0 Attention.forward, verbatim except the
        # `positions=positions` added to the inner-attention call.
        bsz, seqlen, _ = x.size()

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_a(x)
            q = self.wq_b(self.q_norm(q))

        with spmd.local():
            q = q.view(bsz, seqlen, -1, self.qk_head_dim)
            if spmd.is_type_checking():
                spmd.assert_type(
                    q,
                    {"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.S(2)},
                )

        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Key-value projection
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        q_pe, k_pe = self.rope(q_pe, k_pe.unsqueeze(2), positions)
        q = torch.cat([q_nope, q_pe], dim=-1)

        kv = self.wkv_b(self.kv_norm(kv))

        with spmd.local():
            kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            k = torch.cat([k_nope, k_pe.expand(-1, -1, k_nope.size(2), -1)], dim=-1)
            if spmd.is_type_checking() and not torch.compiler.is_compiling():
                for t in [k, v]:
                    spmd.assert_type(
                        t,
                        {"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.S(2)},
                    )

        output = self.inner_attention(
            q,
            k,
            v,
            attention_masks=attention_masks,
            positions=positions,
            scale=self.softmax_scale,
        ).contiguous()
        output = output.view(bsz, seqlen, -1)
        return self.wo(output)


@override(
    target=DeepSeekAttention.Config,
    description=(
        "deepseek_v3 attention: real-space RoPE gather + SDPA with an "
        "explicit block-causal mask"
    ),
)
def deepseek_attention(cfg: DeepSeekAttention.Config) -> NpuDeepSeekAttention.Config:
    new = derive(cfg, NpuDeepSeekAttention.Config)
    if isinstance(cfg.rope, ComplexRoPE.Config):
        new.rope = derive(cfg.rope, RealGatherComplexRoPE.Config)
    new.inner_attention = BlockCausalSDPA.Config()
    return new
