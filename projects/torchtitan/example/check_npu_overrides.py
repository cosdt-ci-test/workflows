"""Equivalence checks for the NPU override module (CPU only, no NPU needed).

Run: ``python check_npu_overrides.py``

Asserts that the two components the NPU overrides substitute in are
numerically equivalent to the upstream components they stand in for:

* ``RealGatherComplexRoPE`` vs upstream ``ComplexRoPE`` -- same cache math,
  same rotation, only the gather kernel differs.
* ``NpuBlockCausalSDPA`` vs a naive per-document causal reference -- the
  masking semantics upstream gets from its BlockMask, both without and with
  a sliding window (muse_glimmer's per-layer window).
* ``NpuMuseAttention`` vs upstream's muse_glimmer attention -- the family's
  own fields (query scaling, output gate, sliding window) must survive the
  override; the generic replacement would drop them.

and that the override only touches the ComplexRoPE-based models: qwen3 uses
upstream's ``CosSinRoPE`` (a real cache, no DT_COMPLEX64 problem,
rotate-half dim pairing) and must be left alone.

Prints one stable line on stdout (the quick-start doc matches it); the
measured numbers go to stderr so CI logs keep them.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# torchtitan >= 0.4.0 imports two torch >= 2.15 symbols through its optional
# Dist-MoE package; install the shim before the first torchtitan import.
# (``sitecustomize.py`` in this directory does the same at interpreter
# startup, which also covers ``python -m torchtitan.train``.)
import npu_torch_compat  # noqa: E402

npu_torch_compat.install()

from npu_compat import (  # noqa: E402
    NpuBlockCausalMetadata,
    NpuBlockCausalSDPA,
    RealGatherComplexRoPE,
)

from torchtitan.models.common.rope import (  # noqa: E402
    ComplexRoPE,
    CosSinRoPE,
)

TOL = 1e-5


def _naive_block_causal(q, k, v, mask, scale):
    """Reference attention: masked softmax over the (q, kv) grid.

    ``q`` / ``k`` / ``v`` are ``(T, H, D)`` -- packed tokens, like upstream's
    inner attentions.  ``mask`` is ``(1, 1, T, T)``.
    """
    q2 = q.transpose(0, 1).unsqueeze(0)
    k2 = k.transpose(0, 1).unsqueeze(0)
    v2 = v.transpose(0, 1).unsqueeze(0)
    att = (q2.float() @ k2.float().transpose(-1, -2)) * scale
    att = att.masked_fill(~mask, float("-inf")).softmax(-1)
    return (att @ v2.float()).squeeze(0).transpose(0, 1)


def check_rope() -> float:
    """RealGatherComplexRoPE must match upstream ComplexRoPE exactly."""
    torch.manual_seed(0)
    dim, max_context_length = 16, 128
    shared = dict(
        dim=dim,
        max_context_length=max_context_length,
        theta=500000.0,
        scaling="llama",
    )
    upstream = ComplexRoPE.Config(**shared).build()
    ours = RealGatherComplexRoPE.Config(**shared).build()

    # (T, N, H): packed tokens, heads, head dim -- upstream's layout.
    q = torch.randn(8, 4, dim)
    k = torch.randn(8, 4, dim)
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    out_upstream, _ = upstream(q, k, positions)
    out_ours, _ = ours(q, k, positions)
    diff = (out_upstream - out_ours).abs().max().item()
    assert diff < TOL, f"rope mismatch: max_abs_diff={diff}"
    return diff


def check_mask() -> float:
    """NpuBlockCausalSDPA must match a naive per-document causal reference."""
    torch.manual_seed(0)
    num_tokens, heads, head_dim = 16, 4, 8
    scale = head_dim**-0.5
    q = torch.randn(num_tokens, heads, head_dim)
    k = torch.randn(num_tokens, heads, head_dim)
    v = torch.randn(num_tokens, heads, head_dim)
    # two documents in this microbatch
    positions = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    attention = NpuBlockCausalSDPA.Config().build()
    out = attention(
        q,
        k,
        v,
        attention_metadata=NpuBlockCausalMetadata(positions=positions),
        scale=scale,
        enable_gqa=False,
    )

    doc = torch.cumsum((positions == 0).to(torch.int64), dim=0) - 1
    mask = (
        (doc[:, None] == doc[None, :])
        & torch.ones(num_tokens, num_tokens, dtype=torch.bool).tril()
    ).unsqueeze(0).unsqueeze(0)
    out_ref = _naive_block_causal(q, k, v, mask, scale)
    diff = (out - out_ref).abs().max().item()
    assert diff < TOL, f"mask mismatch: max_abs_diff={diff}"

    # the mask must actually block the cross-document pairs
    blocked = (~mask).sum().item()
    total = mask.numel()
    assert blocked < total, "mask blocks nothing"
    return diff


def check_window_mask() -> float:
    """NpuBlockCausalSDPA with a sliding window must match the naive reference.

    muse_glimmer's layers ask for a per-layer sliding window; upstream
    expresses it as ``window_size=(window-1, 0)`` on its varlen backend, the
    NPU backend masks it inside the dense block-causal mask.
    """
    torch.manual_seed(0)
    num_tokens, heads, head_dim = 16, 4, 8
    window = 4
    scale = head_dim**-0.5
    q = torch.randn(num_tokens, heads, head_dim)
    k = torch.randn(num_tokens, heads, head_dim)
    v = torch.randn(num_tokens, heads, head_dim)
    # two documents in this microbatch
    positions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7])

    attention = NpuBlockCausalSDPA.Config(sliding_window_size=window).build()
    out = attention(
        q,
        k,
        v,
        attention_metadata=NpuBlockCausalMetadata(positions=positions),
        scale=scale,
        enable_gqa=False,
    )

    doc = torch.cumsum((positions == 0).to(torch.int64), dim=0) - 1
    idx = torch.arange(num_tokens)
    mask = (
        (doc[:, None] == doc[None, :])
        & torch.ones(num_tokens, num_tokens, dtype=torch.bool).tril()
        & (idx[None, :] >= idx[:, None] - (window - 1))
    ).unsqueeze(0).unsqueeze(0)
    out_ref = _naive_block_causal(q, k, v, mask, scale)
    diff = (out - out_ref).abs().max().item()
    assert diff < TOL, f"window mask mismatch: max_abs_diff={diff}"
    return diff


def check_muse_override() -> str:
    """The muse override must keep the family's own attention fields.

    muse_glimmer's attention adds ``scale_query_by``, ``o_gate`` and a
    per-layer ``window_size`` on top of ``GQAttention.Config``.  The generic
    override's replacement has none of them, and ``derive()`` drops what the
    target does not have -- so muse needs its own override, and this check
    asserts the fields survive it (together with the two NPU fixes).
    """
    from torchtitan.config.override import OverrideConfig, apply_overrides
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.models.muse_glimmer import build_model_config as muse_build

    import npu_compat_muse

    spec = muse_build(
        "debugmodel",
        seq_len=256,
        attn_backend="varlen",
        parallelism=ParallelismConfig(),
    )
    before = spec.layers[0].attention
    apply_overrides(
        OverrideConfig(imports=["npu_compat_muse.muse_attention"]), spec
    )
    after = spec.layers[0].attention

    assert type(after) is npu_compat_muse.NpuMuseAttention.Config, type(after)
    assert after.scale_query_by == before.scale_query_by, "query scaling lost"
    assert after.o_gate == before.o_gate, "output gate lost"
    assert after.window_size == before.window_size, "sliding window lost"
    assert isinstance(after.rope, RealGatherComplexRoPE.Config), type(after.rope)
    assert isinstance(
        after.inner_attention, NpuBlockCausalSDPA.Config
    ), type(after.inner_attention)
    assert after.inner_attention.sliding_window_size == before.window_size
    return f"muse={type(after).__name__}, window={before.window_size}"


def check_rope_selection() -> str:
    """The override must only touch the ComplexRoPE-based models.

    qwen3 / qwen3_5 / gpt_oss use upstream's ``CosSinRoPE`` (a real cache,
    no DT_COMPLEX64 problem, rotate-half dim pairing); llama3 / deepseek_v3
    use ``ComplexRoPE`` (complex64 cache, adjacent-dim pairing).  Swapping
    the CosSinRoPE ones would silently change the pairing convention, so
    they must be left alone.
    """
    from torchtitan.config.override import OverrideConfig, apply_overrides
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.models.llama3 import build_model_config as llama3_build
    from torchtitan.models.qwen3 import build_model_config as qwen3_build

    import npu_compat  # noqa: F401  (registers the override)

    parallelism = ParallelismConfig()
    llama = llama3_build(
        "debugmodel",
        seq_len=256,
        attn_backend="varlen",
        parallelism=parallelism,
    )
    qwen = qwen3_build(
        "debugmodel",
        seq_len=256,
        attn_backend="varlen",
        parallelism=parallelism,
    )
    for spec in (llama, qwen):
        apply_overrides(
            OverrideConfig(imports=["npu_compat.npu_attention"]), spec
        )

    llama_rope = llama.layers[0].attention.rope
    qwen_rope = qwen.layers[0].attention.rope
    assert isinstance(llama_rope, RealGatherComplexRoPE.Config), type(llama_rope)
    assert type(qwen_rope) is CosSinRoPE.Config, type(qwen_rope)
    return f"llama3={type(llama_rope).__name__}, qwen3={type(qwen_rope).__name__}"


def check_deepseek_rope_selection() -> str:
    """The deepseek override must swap its ComplexRoPE (it uses ComplexRoPE).

    deepseek_v3 has its own attention class, so the per-family override
    (npu_compat_deepseek.py) carries the same real-space gather.
    """
    from torchtitan.config.override import OverrideConfig, apply_overrides
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.models.deepseek_v3 import build_model_config as deepseek_build

    import npu_compat_deepseek  # noqa: F401  (registers the override)

    spec = deepseek_build(
        "debugmodel",
        seq_len=256,
        attn_backend="varlen",
        parallelism=ParallelismConfig(),
    )
    apply_overrides(
        OverrideConfig(imports=["npu_compat_deepseek.deepseek_attention"]), spec
    )
    rope = spec.layers[0].attention.rope
    assert isinstance(rope, RealGatherComplexRoPE.Config), type(rope)
    return f"deepseek={type(rope).__name__}"


def main() -> int:
    rope_diff = check_rope()
    mask_diff = check_mask()
    window_diff = check_window_mask()
    selection = (
        check_rope_selection()
        + ", "
        + check_deepseek_rope_selection()
        + ", "
        + check_muse_override()
    )
    print(
        "npu override equivalence: rope OK, mask OK, rope selection OK",
        file=sys.stdout,
    )
    print(
        f"rope max_abs_diff={rope_diff:.3e} (tol {TOL:.0e}), "
        f"mask max_abs_diff={mask_diff:.3e} (tol {TOL:.0e}), "
        f"window mask max_abs_diff={window_diff:.3e} (tol {TOL:.0e}), "
        f"rope selection: {selection}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
