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
- checkpointing selected by the workflow's `scenario` input;
- NPU placement and parameter-update assertions, adapter reload before generation.

The workflow pins Unsloth and Zoo commits, uses the repository's CANN 9.1.0
A2 image and torch 2.9.0 / torch_npu 2.9.0.post2 / triton-ascend 3.2.2 stack.
Transformers 4.57.6 and TRL 0.22.2 follow the upstream example's install cell.
NumPy is pinned to 1.26.4 as required by Triton-Ascend 3.2.2, and PyArrow
to 21.0.0 to satisfy datasets 4.3.0 without requiring NumPy 2.x.
Installation checks the three data-library imports and creates a small Dataset
before proceeding to Unsloth.
The checkpointing-disabled baseline passed on Ascend910B1 in
[run #4](https://github.com/cosdt-ci-test/workflows/actions/runs/41382865054).
A Trainer callback checks that the actual checkpointing state matches the selected
scenario at training start and records enabled modules in result.json. Standard
checkpointing still requires a real NPU run; no memory or speed claim is made.
Declared non-CUDA dependencies are installed from the pinned source metadata;
CUDA Triton, torchao and cut-cross-entropy are omitted for this non-quantized
path. bitsandbytes 0.50.2 is installed with `--no-deps` because the upstream
loader unconditionally imports its Linear4bit type, even for BF16 models.
Installation verifies both bitsandbytes and PEFT Linear4bit imports. This does
not enable bitsandbytes quantization or its optimizers on the NPU.
No source/backend patches or CPU substitutes are applied.
Other dependency versions and the resolved ModelScope cache path appear in logs;
the model and all transitive dependencies are not fully locked.

`output/result.json` records the last attempted stage and completed stages.
The workflow prints installation logs, package versions and the result to Actions,
and puts the result in the job summary, including on failure. Success requires
three optimizer steps, finite training loss, actual LoRA parameter changes,
adapter save/reload and NPU generation. It does not measure model quality.

The workflow runs only through manual workflow_dispatch after a branch push.
No push trigger, schedule, PR trigger or automatic rerun is configured.

Select a scenario when dispatching the same workflow:

| scenario | Flow | Status |
| --- | --- | --- |
| `baseline` (default) | Checkpointing disabled, matching the run #4 training configuration | Passed in run #4; added scenario/state checks still need a run |
| `standard-gc` | Standard checkpointing (`True`) in loading, LoRA setup and SFTConfig | Passed functionally in run #5 |
| `unsloth-gc` | Unsloth checkpointing (`"unsloth"`) with 512-token input and Trainer's standard flag left off | Passed in run #6 |
| `checkpoint-resume` | Train one step, save full Trainer state, exit, then resume to step 3 in a new process | Passed in run #8 |
| `merge-16bit` | Train and merge in one process, then reload with Transformers in a clean second process | Passed in run #10 |
| `dpo` | Train LoRA with local preference pairs, then save, reload and generate | Passed in run #11 |
| `full-finetune` | Train every model parameter in BF16, then reload the saved model in a clean process | Passed in run #13 |
| `continued-pretrain` | Train raw text across the full sequence with LoRA on model and embedding outputs | Awaiting NPU validation |
| `inference-extended` | Run mixed-length padded batches and a custom stopping criterion | Awaiting NPU validation |
| `ddp-2` | Run two ranks with explicit HCCL, synchronized LoRA updates and rank-0 adapter save | Awaiting NPU validation |

All scenarios run the full three-step LoRA training, adapter save/reload and
inference checks. The `unsloth-gc` dataset reaches the 512-token cap, and its
training-start check requires the effective `UnslothCheckpointFunction` rather
than a silent fallback to standard checkpointing. Its `SFTConfig` standard
checkpointing flag remains off, following the official Qwen3 example, so the
Trainer does not replace the custom checkpoint function. Unsupported scenario values
fail before importing the NPU stack.

The `checkpoint-resume` scenario runs two separate Python processes. The first
saves `checkpoint-1` with trainer, optimizer, scheduler, RNG and PEFT adapter
state. The second loads that checkpoint, asserts training restarts at step 2,
continues to step 3, checks restored optimizer and scheduler state, confirms
checkpoint LoRA tensors changed, then saves and reloads the final adapter. This
scenario keeps gradient checkpointing disabled so the resume check has one
variable. The continuation check compares PEFT's serialized adapter state, which
omits the default adapter name, with the same format produced after resume.

The `merge-16bit` scenario trains the same three-step LoRA, calls
`save_pretrained_merged(..., save_method="merged_16bit", dtype=torch.bfloat16)`,
checks the serialized floating tensors are BF16, then starts a separate process
that never imports Unsloth. That process reloads the merged directory through
standard `AutoModelForCausalLM` and generates without the original adapter. The
export is local and does not push to Hugging Face.

The `dpo` scenario uses six local prompt/chosen/rejected pairs repeated twice,
rank-8 BF16 LoRA, `DPOTrainer` with its reference path provided by disabling the
PEFT adapter, and three optimizer steps. It checks finite loss and changed LoRA
weights, saves and reloads the DPO adapter, and generates an answer. It records
the chosen/rejected reward metrics when TRL provides them but does not require
quality improvement from this smoke run.

The `full-finetune` scenario disables LoRA and quantization, asserts every model
parameter is trainable, and performs three BF16 optimizer steps. It compares
sampled parameters across the model, saves a complete BF16 model, then reloads
it in a separate process that does not import Unsloth. This is the first half of
item 07; continued pretraining will use a separate data and objective check.
BF16 full fine-tuning in Unsloth intentionally keeps normalization weights in
FP32 for numerical stability. The export check therefore requires all non-norm
floating weights to be BF16 and permits FP32 only for tensors whose names contain
`norm`; the exact FP32 tensor names are recorded in result.json.

The `continued-pretrain` scenario follows the official CPT pattern: raw local
documents with EOS appended, rank-8 LoRA on the transformer projections, and
`embed_tokens`/`lm_head` in `modules_to_save`, with loss over every token. Qwen3
ties the output head to the input embedding, so Unsloth drops the duplicate tied
output module while training the shared embedding weight. It does not apply the
chat template or assistant-only response masking. Three optimizer steps must
change LoRA weights before the CPT adapter is saved, reloaded and used for a
generation smoke check.

The `inference-extended` scenario loads the base BF16 Qwen3 model through
Unsloth, runs one three-prompt batch with left padding and different prompt
lengths, then runs a single prompt with a custom three-token stopping criterion.
It records token lengths, outputs and EOS observations. This verifies inference
plumbing only, not answer quality.

The `ddp-2` scenario selects the `linux-aarch64-a2-2` runner and mounts both
DAVINCI devices. It starts two processes with `torch.distributed.run`, creates
an explicit HCCL process group, trains rank-specific text batches for three DDP
steps, and gathers parameter checksums from both ranks after DDP synchronization.
Only rank 0 saves the adapter, and the other slices continue to FSDP2 later.
The selected scenario appears in the Actions run title, result.json and job summary,
including when environment setup fails. The `"unsloth"` checkpointing mode uses
upstream's long-context threshold and is not treated as a performance claim.

```shell
gh workflow run unsloth-npu-example.yml --repo cosdt-ci-test/workflows --ref feat/unsloth-adapt -f scenario=baseline
```

```shell
gh workflow run unsloth-npu-example.yml --repo cosdt-ci-test/workflows --ref feat/unsloth-adapt -f scenario=standard-gc
```

```shell
gh workflow run unsloth-npu-example.yml --repo cosdt-ci-test/workflows --ref feat/unsloth-adapt -f scenario=unsloth-gc
```

```shell
gh workflow run unsloth-npu-example.yml --repo cosdt-ci-test/workflows --ref feat/unsloth-adapt -f scenario=checkpoint-resume
```

```shell
gh workflow run unsloth-npu-example.yml --repo cosdt-ci-test/workflows --ref feat/unsloth-adapt -f scenario=merge-16bit
```

```shell
gh workflow run unsloth-npu-example.yml --repo cosdt-ci-test/workflows --ref feat/unsloth-adapt -f scenario=dpo
```

```shell
gh workflow run unsloth-npu-example.yml --repo cosdt-ci-test/workflows --ref feat/unsloth-adapt -f scenario=full-finetune
```

```shell
gh workflow run unsloth-npu-example.yml --repo cosdt-ci-test/workflows --ref feat/unsloth-adapt -f scenario=continued-pretrain
```

```shell
gh workflow run unsloth-npu-example.yml --repo cosdt-ci-test/workflows --ref feat/unsloth-adapt -f scenario=inference-extended
```

```shell
gh workflow run unsloth-npu-example.yml --repo cosdt-ci-test/workflows --ref feat/unsloth-adapt -f scenario=ddp-2
```
