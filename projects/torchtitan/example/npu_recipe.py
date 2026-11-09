"""Quick-start recipe for torchtitan (latest release) on Ascend NPU.

torchtitan's config system is recipe-based: a recipe is a plain Python
function that returns a ``Trainer.Config``, and the CLI selects it with
``--module <module> --config <function>``.  The CLI deliberately exposes a
small surface -- there is no ``--training.steps`` / ``--comm.mode`` /
``--dump-folder`` knob any more -- so a quick-start needs a recipe of its
own instead of patching upstream's config tree.

This module provides the tiny recipe the quick-start trains with:

* llama3 ``debugmodel`` (dim 256 / 6 layers / 16 heads / vocab 2048, ~6 M
  parameters);
* 2 training steps;
* one 2048-token microbatch per step;
* the bundled ``tests/assets/c4_test`` dataset -- no download, no network.

The loss is upstream's plain ``CrossEntropyLoss`` rather than
``ChunkedLossWrapper``: the chunked wrapper computes the lm_head + loss in
a custom autograd Function built on FSDP2 internals, which is the part of
torchtitan that has repeatedly broken on ``torch_npu`` (meta-tensor leaks
in the backward).  With a plain loss the model keeps its lm_head inside the
forward, which is the simplest path and the one the NPU quick-start wants.

The recipe lives outside the torchtitan checkout (next to the NPU override
modules) and is activated with ``PYTHONPATH``::

    PYTHONPATH="$HOME/tt-npu-overrides" torchrun ... -m torchtitan.train \\
        --module npu_recipe --config llama3_npu_debugmodel \\
        --override npu_compat.npu_attention --output-dir /tmp/torchtitan-quickstart
"""

from __future__ import annotations

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optim import AdamW, Optim, OptimizersContainer
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.llama3 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

SEQ_LEN = 256
"""Training context length (also the RoPE cache length)."""

NUM_TOKENS_PER_MICROBATCH = 8 * SEQ_LEN
"""Token slots per data-parallel rank per microbatch."""

STEPS = 2
"""How many train steps to run."""

MAX_NUM_DOCUMENTS = max(16, (NUM_TOKENS_PER_MICROBATCH + 511) // 512)
"""Upper bound on document segments packed into one microbatch.

Same estimate upstream's own recipes use (``c4_document_capacity``): ~16
document slots per 512 tokens, at least 16 for short batches.  It has to be
an upper bound because the packed rows are padded to it.
"""


def _debugmodel_config(*, bf16: bool = False) -> Trainer.Config:
    """Build the quick-start config.

    ``bf16`` switches the FSDP parameter dtype to bfloat16 (mixed
    precision); the fp32 variant keeps the parameters in float32.
    """
    parallelism = ParallelismConfig()
    model_config = build_model_config(
        "debugmodel",
        seq_len=SEQ_LEN,
        attn_backend="varlen",
        parallelism=parallelism,
    )
    # torchtitan compiles the loss / activation / rope regions when they are
    # listed in ``local_compile_regions``.  The NPU quick-start keeps
    # everything eager -- the compiled regions are where torch_npu's
    # lowering gaps bite.
    model_config.local_compile_regions = []
    return Trainer.Config(
        loss=CrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(model_config),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model=model_config,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
            )
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=NUM_TOKENS_PER_MICROBATCH,
            max_context_length=model_config.max_context_length,
            steps=STEPS,
            mixed_precision_param="bfloat16" if bf16 else "float32",
        ),
        dataloader=GrainDataLoader.Config(
            max_num_documents=MAX_NUM_DOCUMENTS,
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            shuffle=False,
        ),
        metrics=MetricsProcessor.Config(log_freq=1, disable_color_printing=True),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
        validator=None,
    )


def llama3_npu_debugmodel() -> Trainer.Config:
    """Single-card quick-start: 2 steps of the llama3 debugmodel, fp32."""
    return _debugmodel_config()


def llama3_npu_debugmodel_bf16() -> Trainer.Config:
    """Dual-card quick-start: same recipe with bfloat16 parameters."""
    return _debugmodel_config(bf16=True)
