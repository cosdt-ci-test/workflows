# Unsloth Ascend trial

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

The GRPO scenario follows the API shape in the official
[Qwen3 (4B) GRPO script](https://github.com/unslothai/notebooks/blob/d85d113eacfc8a87a7003e73ffb2a86403a94344/python_scripts/Qwen3_%284B%29-GRPO.py):
chat-format prompts, multiple sampled completions per prompt, Python reward
functions, `GRPOConfig`, `GRPOTrainer`, and adapter save/reload. The local smoke
uses Qwen3-0.6B, four generations per prompt and three optimizer steps. It uses
TRL's transformers generation path and keeps vLLM for the separate rollout item.

The workflow pins Unsloth and Zoo commits, uses the repository's CANN 9.1.0
A2 image and torch 2.9.0 / torch_npu 2.9.0.post2 / triton-ascend 3.2.2 stack.
It downloads both upstream sources from GitHub's source-archive endpoint at
the full pinned commit hashes, with bounded retries, rather than fetching
Git objects. Archive layout is checked before extraction; commit IDs and
archive SHA-256 values are printed. This addresses the interrupted Git
transfers in [run #29](https://github.com/cosdt-ci-test/workflows/actions/runs/42001225991),
which never reached dependency installation or packing.
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
| `continued-pretrain` | Train raw text across the full sequence with LoRA on model and embedding outputs | Passed in run #15 |
| `inference-extended` | Run mixed-length padded batches and a custom stopping criterion | Passed in run #16 |
| `grpo` | Generate groups, score rewards, update the LoRA policy, then save, reload and infer | Passed without vLLM in run #22 |
| `vllm-ascend` | Compare native vLLM-Ascend generation with Unsloth fast generation and a LoRA request | Passed in run #26 |
| `grpo-vllm-ascend` | Run GRPO with the colocated vLLM engine and verify fresh LoRA tensors reach each rollout | Passed in run #27 |
| `packing` | Pack short tokenized samples into full sequences, then train, save, reload and generate | Awaiting NPU validation |
| `ddp-2` | Run two ranks with explicit HCCL, synchronized LoRA updates and rank-0 adapter save | Passed in run #18 |
| `fsdp2-2` | Run two-rank Accelerate FSDP2, save step 1, then resume in new processes | Passed in run #20 |

Training scenarios run three optimizer steps and end with adapter or model save,
reload and inference checks. The `unsloth-gc` dataset reaches the 512-token
cap, and its
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

The `grpo` scenario uses a small local question/answer dataset, rank-8 BF16 LoRA,
four sampled completions per prompt and `GRPOTrainer` on a single Ascend NPU. It
starts with `fast_inference=False` and `use_vllm=False`, so TRL uses the training
model's `generate` path for rollouts. The policy must log finite rewards on all
three steps, have non-zero reward variation in at least one completion group,
produce finite non-zero advantages and LoRA gradients, change LoRA weights
without producing non-finite values, and finish three optimizer steps. The
adapter is then saved, reloaded in a fresh model instance and used for
deterministic generation. Loss is recorded but not asserted because the short
sampled run can legitimately produce a zero or near-zero GRPO loss.
This test does not use a reference model, vLLM rollout or an external reward
model. It starts with `UNSLOTH_COMPILE_DISABLE=partial` and disables GRPO
sequence packing/prefix grouping so compilation and packing remain separate
adaptation questions. It does not establish answer-quality improvement.

[Run #21](https://github.com/cosdt-ci-test/workflows/actions/runs/41825910070)
stopped before the first optimizer update with non-zero advantages but zero
LoRA gradients. Unsloth had enabled DAPO truncation masking. To isolate that
setting, this scenario now explicitly keeps capped completions in the loss.
Before backward, it records each completion's EOS status, advantage, and
active loss-token count, plus how many tokens the old masking rule would keep
for the same rollout. The non-zero gradient and weight-update checks still apply.

[Run #22](https://github.com/cosdt-ci-test/workflows/actions/runs/41834043814)
passed on one Ascend910B4. The old mask would have excluded all 128 completion
tokens in the first batch. With masking off, all three steps produced non-zero
gradients, 392 LoRA tensors changed, and adapter save, reload and generation passed.

The `vllm-ascend` scenario uses the repository's dedicated vLLM-Ascend 0.23.0
image and keeps its matched Torch, torch_npu, Triton and NumPy packages.
This scenario uses Transformers 5.5.4 as required by vLLM-Ascend 0.23.0,
with PEFT 0.19.1 and TRL 0.29.1 for Unsloth imports. The training scenarios
keep their original pins. A single manual dispatch runs both inference checks.
The first process runs native BF16 Qwen3-0.6B generation on the unchanged
image. Afterward, pip installs Unsloth's dependencies without requesting the
preinstalled vLLM packages again: resolving their video dependencies would
pit OpenCV's NumPy 2 requirement against Triton-Ascend's NumPy 1.26.4 pin
([run #25](https://github.com/cosdt-ci-test/workflows/actions/runs/41949054299)).
The installer verifies that the image packages retain their original versions.
This check covers text only. A separate process loads the same model via
Unsloth's `fast_inference=True`,
generates without an adapter, then saves and reloads a probe LoRA through
Unsloth and calls `fast_generate` with the LoRA request. It checks the returned
adapter ID and requires a change in generated tokens or first-token log
probability. The probe weights are synthetic and this run does not train or
test GRPO rollouts, standby mode, or throughput.

[Run #26](https://github.com/cosdt-ci-test/workflows/actions/runs/41954617493)
passed on one Ascend910B4: both prompts matched native vLLM before applying
LoRA and produced different token sequences after applying it.

The `grpo-vllm-ascend` scenario uses the same image and runs three BF16 LoRA
optimizer steps. It checks that TRL reuses Unsloth's colocated engine, and
that each rollout request contains the current adapter tensors, has a fresh
adapter ID, and returns four completions with that ID. The adapter hashes
must change between rollouts. This checks rollout and adapter refresh with
importance sampling correction and standby disabled. TRL 0.29.1 warns that
vLLM 0.23.0 is outside its tested range; runtime compatibility is part of
the smoke test.

[Run #27](https://github.com/cosdt-ci-test/workflows/actions/runs/41962465270)
passed on one Ascend910B1. Each of the three rollouts used a new adapter ID
(1, 2, 3), carried tensors matching the live training model and returned four
completions. The adapter hash changed after each preceding update, 392 LoRA
tensors changed over training, and the final adapter was saved.

The `packing` scenario returns to the single-card training image and uses
TRL 0.22.2's `wrapped` strategy with Unsloth's `padding_free=True`. It checks that
the trainer combines multiple short examples into full 256-token blocks
without changing or dropping any input token, then completes three training
steps, adapter save/reload and generation. Each training batch takes up to
two packed blocks. Checks cover flattened tokens, position-ID resets,
`packed_seq_lengths`, masked boundary labels and the absence of a padded
attention mask, including the metadata passed to actual NPU training forwards.
Wrapped packing can split original examples and permits attention within
each block across original example boundaries. This does not validate BFD,
isolation of original documents, a particular fused kernel or a speed improvement.
[Run #28](https://github.com/cosdt-ci-test/workflows/actions/runs/41991041132)
passed the token-preservation checks but stopped before training because
the old script incorrectly required padding-free to remain off; Unsloth
enables it whenever sample packing is active.
The separate 4-bit QLoRA item remains a backend gap under the maintainer's
[Ascend support statement](https://github.com/unslothai/unsloth/issues/1794).

The `ddp-2` scenario selects the `linux-aarch64-a2-2` runner and mounts both
DAVINCI devices. It starts two processes with `torch.distributed.run`, creates
an explicit HCCL process group, trains rank-specific text batches for three DDP
steps, and gathers parameter checksums from both ranks after DDP synchronization.
Rank-local losses need not match because each rank consumes different text;
the test requires the post-update parameter checksums to match instead. Only
rank 0 saves the adapter, and the other slices continue to FSDP2 later.

The `fsdp2-2` scenario runs the same `a2-2` runner with
`UNSLOTH_FSDP2=1`. It asserts Accelerate selected FSDP version 2, the HCCL
backend is active, and `FSDPModule` wrappers exist. One process saves a
one-step checkpoint; a fresh two-process launch restores it at step 1,
continues to step 3, and writes the final adapter through the FSDP-aware save
path. DDP and FSDP2 are separate acceptance cases.
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
gh workflow run unsloth-npu-example.yml --repo cosdt-ci-test/workflows --ref feat/unsloth-adapt -f scenario=grpo
```

```shell
gh workflow run unsloth-npu-example.yml --repo cosdt-ci-test/workflows --ref feat/unsloth-adapt -f scenario=vllm-ascend
```

```shell
gh workflow run unsloth-npu-example.yml --repo cosdt-ci-test/workflows --ref feat/unsloth-adapt -f scenario=grpo-vllm-ascend
```

```shell
gh workflow run unsloth-npu-example.yml --repo cosdt-ci-test/workflows --ref feat/unsloth-adapt -f scenario=packing
```

```shell
gh workflow run unsloth-npu-example.yml --repo cosdt-ci-test/workflows --ref feat/unsloth-adapt -f scenario=ddp-2
```

```shell
gh workflow run unsloth-npu-example.yml --repo cosdt-ci-test/workflows --ref feat/unsloth-adapt -f scenario=fsdp2-2
```
