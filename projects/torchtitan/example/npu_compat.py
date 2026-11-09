"""NPU compatibility overrides for torchtitan v0.4.0 on Ascend.

Activated through torchtitan's official override mechanism
(``--override npu_compat.npu_attention``); no upstream source file is
modified -- the torchtitan checkout stays git clean.

One override replaces one config node (``GQAttention.Config``) and swaps two
of its children:

1. **RoPE** -- upstream ``ComplexRoPE`` keeps a complex64 cache and indexes
   it with the position ids; ``aclnnIndex`` has no ``DT_COMPLEX64``
   implementation.  ``RealGatherComplexRoPE`` performs the same gather in
   real space (the real / imaginary halves as float32) and hands the result
   back to upstream's own rotation code, so the values -- including the
   llama frequency scaling and the adjacent-dimension pairing convention --
   are upstream's.  Upstream's ``CosSinRoPE`` is *not* a substitute: it
   uses the rotate-half pairing convention and rejects
   ``scaling="llama"`` outright.

2. **Attention** -- upstream ships two backends for language models:
   ``FlexInnerAttention`` (``torch.nn.attention.flex_attention``) and
   ``VarlenInnerAttention`` (``torch.nn.attention.varlen``).  Neither
   compiles on NPU.  Upstream's SDPA does run on NPU, but upstream forbids
   it as a language-model backend (``get_attention_config("sdpa")`` raises)
   because it cannot consume per-document positions.
   ``NpuBlockCausalSDPA`` is an ``InnerAttention`` that rebuilds the dense
   block-causal mask from the per-document positions and hands it to SDPA,
   which keeps upstream's masking semantics: a token attends to the tokens
   of its own document that precede it.

Both replacements are numerically equivalent to the upstream components
they stand in for.  ``check_npu_overrides.py`` (same directory) asserts
that on CPU; the quick-start doc runs it before the training steps.

Upstream private helpers used below (``_maybe_check_max_pos``,
``_reshape_for_broadcast``): if upstream renames them this module fails
loudly at import time.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F

from torchtitan.config.override import derive, override
from torchtitan.models.common.attention import GQAttention, InnerAttention
from torchtitan.models.common.rope import (
    ComplexRoPE,
    _maybe_check_max_pos,
    _reshape_for_broadcast,
)


class RealGatherComplexRoPE(ComplexRoPE):
    """``ComplexRoPE`` whose position gather happens in real space.

    Upstream's ``ComplexRoPE._reshape_cache`` indexes the complex64 cache
    directly (``rope_cache[positions]``), which NPU cannot do.  Here the
    cache is viewed as float32 (real / imaginary interleaved), gathered as
    a plain float32 tensor, and reinterpreted as complex64 before upstream's
    ``apply_rotary_emb`` runs.  The gathered values are bit-identical to
    upstream's; only the kernel that produces them differs.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ComplexRoPE.Config):
        pass

    def _reshape_cache(self, query, positions=None):
        if positions is not None:
            _maybe_check_max_pos(positions, max_valid_pos=self.cache.shape[0] - 1)
        if torch.compiler.is_compiling():
            # Upstream's compile path already gathers the real view; keep it.
            return _reshape_for_broadcast(
                torch.view_as_real(self.cache).flatten(-2), query.shape, positions
            )
        # complex64 (S, dim / 2) -> float32 (S, dim) with real / imaginary
        # interleaved -- exactly the byte layout a complex64 tensor stores.
        real_cache = torch.view_as_real(self.cache).flatten(-2)
        # Same broadcast rules as upstream: the pair shape is
        # (num_tokens, n_heads, head_dim / 2).
        pair_shape = (*query.shape[:-1], query.shape[-1] // 2)
        # The gather happens on float32 -- aclnnIndex has no DT_COMPLEX64.
        gathered = _reshape_for_broadcast(real_cache, pair_shape, positions)
        # Back to the complex64 layout upstream's rotation expects.
        return gathered.view(torch.complex64)


def _block_causal_mask(positions: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Dense ``(1, 1, T, T)`` bool mask: same document, not in the future.

    Mirrors upstream's per-document (block causal) masking: positions reset
    to 0 at every document boundary, so tokens of one document must not
    attend to another document's tokens.  Upstream derives the document ids
    the same way (``get_document_mask_mod``).
    """
    if positions.dim() != 1:
        raise ValueError(
            f"block-causal masking expects 1-D positions, got {positions.shape}"
        )
    doc = torch.cumsum((positions == 0).to(torch.int64), dim=0) - 1  # (T,)
    same_doc = doc[:, None] == doc[None, :]  # (T, T)
    causal = torch.ones(
        num_tokens, num_tokens, dtype=torch.bool, device=positions.device
    ).tril()
    return (same_doc & causal).unsqueeze(0).unsqueeze(0)


class NpuBlockCausalMetadata(NamedTuple):
    """Per-microbatch metadata handed to ``NpuBlockCausalSDPA``.

    Upstream's inner attentions receive a backend specific metadata object
    (a ``BlockMask`` for flex, cumulative sequence lengths for varlen).  The
    NPU backend needs the per-document positions themselves, because it
    rebuilds the dense block-causal mask.
    """

    positions: torch.Tensor


class NpuBlockCausalSDPA(InnerAttention):
    """``scaled_dot_product_attention`` with an explicit block-causal mask.

    Upstream's ``ScaledDotProductAttention`` refuses an ``attention_masks``
    argument, and the trainer only builds a mask for the Flex / Varlen
    backends.  This backend keeps the per-document causal masking by
    handing SDPA a dense mask (aclnn SDPA accepts ``attn_mask``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        @property
        def is_full_attention(self) -> bool:
            """This backend attends to the whole (causal) document."""
            return True

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> NpuBlockCausalMetadata:
            del padding_mask, max_num_documents, max_context_length
            return NpuBlockCausalMetadata(positions=positions)

    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: NpuBlockCausalMetadata | None = None,
        scale: float | None = None,
        enable_gqa: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if attention_metadata is None:
            raise ValueError("NpuBlockCausalSDPA requires per-document positions")
        positions = attention_metadata.positions
        mask = _block_causal_mask(positions, q_THK.shape[0])
        out_1HTV = F.scaled_dot_product_attention(
            q_THK.transpose(0, 1).unsqueeze(0),
            k_THK.transpose(0, 1).unsqueeze(0),
            v_THV.transpose(0, 1).unsqueeze(0),
            attn_mask=mask,
            scale=scale,
            is_causal=False,
            enable_gqa=enable_gqa,
        )
        return out_1HTV.squeeze(0).transpose(0, 1)


class NpuGQAttention(GQAttention):
    """``GQAttention`` wired for NPU.

    Upstream's ``GQAttention.forward`` already hands the per-document
    metadata to the inner attention, and the NPU inner attention rebuilds
    the block-causal mask from it -- so the forward is upstream's, unchanged.
    This subclass exists so the override has a distinct config class (and a
    distinct name in the override log).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        pass


@override(
    target=GQAttention.Config,
    description=(
        "NPU attention stack: real-space RoPE gather (aclnnIndex has no "
        "DT_COMPLEX64) + SDPA with an explicit block-causal mask"
    ),
)
def npu_attention(cfg: GQAttention.Config) -> NpuGQAttention.Config:
    new = derive(cfg, NpuGQAttention.Config)
    if isinstance(cfg.rope, ComplexRoPE.Config):
        # complex64 RoPE cache -> gather it in real space.  Only for
        # ComplexRoPE-based models: a CosSinRoPE cache is real already and
        # needs no change, and swapping it for the ComplexRoPE convention
        # would silently change the dim pairing (CosSinRoPE pairs
        # (d0,d8),(d1,d9)..., ComplexRoPE pairs (d0,d1),(d2,d3)...).
        new.rope = derive(cfg.rope, RealGatherComplexRoPE.Config)
    # flex / varlen -> SDPA with a dense block-causal mask.  The config class
    # must not be a Flex / Varlen config, otherwise the trainer would try to
    # build a BlockMask (that path does not compile on NPU).
    #
    # derive() keeps the fields the two configs share -- notably
    # ``sharding_config`` (llama3 puts the local-SPMD sharding of q/k/v on
    # the inner attention config), which a fresh Config() would silently drop.
    new.inner_attention = derive(cfg.inner_attention, NpuBlockCausalSDPA.Config)
    return new


__all__ = [
    "NpuBlockCausalMetadata",
    "NpuBlockCausalSDPA",
    "NpuGQAttention",
    "RealGatherComplexRoPE",
    "npu_attention",
]
