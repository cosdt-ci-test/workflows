# Unsloth Ascend single-card trial

This is an **experimental run**, not a previously validated Ascend example.
The maintainer recommends trying 16-bit LoRA on Ascend in
[issue #1794](https://github.com/unslothai/unsloth/issues/1794);
full backend work is tracked in [#10772](https://github.com/unslothai/unsloth/issues/10772).

The upstream example inspected is
[Qwen3 (4B) Instruct](https://github.com/unslothai/notebooks/blob/3f6b3049c7a17da13d2f09b2a881cb93d7db6010/python_scripts/Qwen3_%284B%29-Instruct.py).
Our independently written smoke script follows its FastLanguageModel load →
get_peft_model → SFTTrainer.train → save → generate API sequence, with:

- Qwen/Qwen3-0.6B from ModelScope rather than the 4B Instruct model;
- BF16 non-quantized LoRA, rank 8, ordinary AdamW instead of bitsandbytes;
- 12 local chat samples, 256 tokens, 3 steps, batch size 1;
- checkpointing disabled for this small workload;
- NPU placement and parameter-update assertions, adapter reload before generation.

The workflow pins Unsloth and Zoo commits, uses the repository's CANN 9.1.0
A2 image and torch 2.9.0 / torch_npu 2.9.0.post2 / triton-ascend 3.2.2 stack.
Transformers 4.57.6 and TRL 0.22.2 follow the upstream example's install cell.
NumPy is pinned to 1.26.4 as required by Triton-Ascend 3.2.2, and PyArrow
to 21.0.0 to satisfy datasets 4.3.0 without requiring NumPy 2.x.
Installation checks the three data-library imports and creates a small Dataset
before proceeding to Unsloth.
This combination has not yet been validated with Unsloth on our hardware.
Declared non-CUDA dependencies are installed from the pinned source metadata;
CUDA Triton, torchao, cut-cross-entropy and bitsandbytes are omitted for this
non-quantized path. No source/backend patches or CPU substitutes are applied.
Other dependency versions and the resolved ModelScope cache path appear in logs;
the model and all transitive dependencies are not fully locked.

`output/result.json` records the last attempted stage and completed stages.
The workflow prints installation logs, package versions and the result to Actions,
and puts the result in the job summary, including on failure. Success requires
three optimizer steps, finite training loss, actual LoRA parameter changes,
adapter save/reload and NPU generation. It does not measure model quality.

The initial branch push triggers this workflow without adding it to main.
No schedule, PR trigger or automatic rerun is configured.
