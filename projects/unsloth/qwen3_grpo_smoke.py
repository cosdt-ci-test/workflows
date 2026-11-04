"""Minimal BF16 GRPO trial for Unsloth on one Ascend NPU.

This follows the official Qwen3 GRPO flow but deliberately uses TRL's
transformers generation path. vLLM rollout is tested separately, so a GRPO
pass here means generation, reward scoring, policy loss/backward, optimizer
updates, adapter save/reload and inference worked without vLLM.
"""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback

OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)
RESULT_PATH = OUTPUT / "result.json"
ADAPTER = OUTPUT / "grpo_adapter"
MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_SEQ_LENGTH = 256
MAX_PROMPT_LENGTH = 128
MAX_COMPLETION_LENGTH = 32
NUM_GENERATIONS = 4

# A full rolling batch needs one prompt and four completions per optimizer step.
PER_DEVICE_BATCH_SIZE = NUM_GENERATIONS

SYSTEM_PROMPT = "Answer the user's question with only the short final answer."
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

QUESTIONS = [
    ("What is 2 + 2? Reply with only the number.", "4"),
    ("What is the capital of France? Reply with only the city name.", "Paris"),
    ("What is 7 + 5? Reply with only the number.", "12"),
    ("Name the largest planet in our solar system. Reply with only the name.", "Jupiter"),
    ("What is the chemical formula for water? Reply with only the formula.", "H2O"),
    ("What is 9 - 4? Reply with only the number.", "5"),
    ("What is the capital of Japan? Reply with only the city name.", "Tokyo"),
    ("What is 3 + 6? Reply with only the number.", "9"),
]

COMPLETION_GROUPS = []
ADVANTAGE_CHECKS = []
GRADIENT_CHECKS = []
result = {
    "status": "running",
    "stage": "grpo_setup",
    "completed": [],
    "scenario": "grpo",
    "model": MODEL_NAME,
    "rollout_backend": "transformers_generate",
    "vllm_enabled": False,
    "loss_type": "dapo",
    "num_generations": NUM_GENERATIONS,
    "max_prompt_length": MAX_PROMPT_LENGTH,
    "max_completion_length": MAX_COMPLETION_LENGTH,
}


def save(stage, completed=False):
    result["stage"] = stage
    if completed and stage not in result["completed"]:
        result["completed"].append(stage)
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"stage": stage, "completed": completed}), flush=True)


def completion_texts(completions):
    """Accept both TRL string completions and conversational completions."""
    texts = []
    for completion in completions:
        if isinstance(completion, str):
            texts.append(completion)
        elif isinstance(completion, list) and completion:
            texts.append(str(completion[-1].get("content", "")))
        else:
            texts.append("")
    return texts


def contains_answer(response, target):
    response = response.strip().lower()
    target = target.strip().lower()
    if re.fullmatch(r"-?\d+(?:\.\d+)?", target):
        return any(float(number) == float(target)
                   for number in re.findall(r"-?\d+(?:\.\d+)?", response))
    return re.search(
        rf"(?<![a-z0-9]){re.escape(target)}(?![a-z0-9])", response
    ) is not None


def reward_answer(completions, answer, prompts=None, **kwargs):
    texts = completion_texts(completions)
    prompt_texts = []
    if prompts is not None:
        for prompt in prompts:
            if isinstance(prompt, list) and prompt:
                prompt_texts.append(str(prompt[-1].get("content", "")))
            else:
                prompt_texts.append(str(prompt))
    COMPLETION_GROUPS.append(
        {
            "prompts": prompt_texts,
            "completions": texts,
            "answers": list(answer),
        }
    )
    return [1.0 if contains_answer(text, target) else 0.0
            for text, target in zip(texts, answer)]


def reward_concise(completions, **kwargs):
    """A continuous, non-empty reward gives a useful group signal at smoke scale."""
    return [
        0.0 if not text.strip() else 1.0 / (1.0 + len(text.strip()) / 32.0)
        for text in completion_texts(completions)
    ]


def run_reward_self_test():
    answers = ["4", "4", "4", "4"]
    completions = [
        "The answer is 4.",
        "24",
        [{"role": "assistant", "content": "I think 4"}],
        "",
    ]
    correct = reward_answer(
        completions,
        answers,
        prompts=[[{"role": "user", "content": "What is 2 + 2?"}]] * 4,
    )
    concise = reward_concise(completions)
    assert correct == [1.0, 0.0, 1.0, 0.0], correct
    assert len(concise) == len(completions)
    assert all(math.isfinite(value) for value in concise), concise
    assert max(concise) > min(concise), concise
    print(json.dumps({"reward_self_test": "passed", "correct": correct,
                      "concise": concise}))


def snapshot_lora(model):
    import torch

    tensors = {}
    digest = hashlib.sha256()
    finite = True
    for name, parameter in sorted(model.named_parameters()):
        if "lora_" not in name:
            continue
        tensor = parameter.detach().float().cpu().contiguous()
        finite = finite and bool(torch.isfinite(tensor).all())
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
        tensors[name] = tensor.clone()
    if not tensors:
        raise AssertionError("No LoRA parameters found")
    return {
        "tensors": tensors,
        "sha256": digest.hexdigest(),
        "finite": finite,
    }


def changed_lora_parameters(before, after):
    import torch

    names = before.keys()
    assert names == after.keys()
    changed = [name for name in names
               if not torch.equal(before[name], after[name])]
    return changed


def json_scalar(value):
    if hasattr(value, "item") and not isinstance(value, str):
        value = value.item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def numeric_history(history):
    selected = []
    for entry in history:
        if "step" not in entry:
            continue
        selected.append({
            key: json_scalar(value) for key, value in entry.items()
            if key in {
                "step", "loss", "grad_norm", "learning_rate", "reward",
                "reward_std", "frac_reward_zero_std",
                "completions/mean_length", "completions/max_length",
                "completions/min_length",
            } or key.startswith("rewards/")
        })
    return selected


if "--self-test" in sys.argv:
    run_reward_self_test()
    raise SystemExit(0)

# Entry 14 handles packed/fused operator paths separately. Keep this GRPO smoke
# on the conservative TRL generation and loss path so a padding issue cannot be
# mistaken for a core GRPO failure.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "partial")
os.environ.setdefault("UNSLOTH_GRPO_SEQ_PACKING", "0")
os.environ.setdefault("UNSLOTH_GRPO_PREFIX_GROUPER", "0")

try:
    save("scenario_setup", True)
    import torch
    import torch_npu  # noqa: F401 - registers the NPU backend
    from unsloth import FastLanguageModel
    from unsloth.device_type import DEVICE_TYPE
    from datasets import Dataset
    from modelscope import snapshot_download
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    assert DEVICE_TYPE == "npu", DEVICE_TYPE
    assert torch.npu.is_available()
    torch.npu.set_device(0)
    assert torch.npu.is_bf16_supported(), "BF16 is required for this trial"
    result["device"] = torch.npu.get_device_name(0)
    save("import", True)

    model_path = snapshot_download(MODEL_NAME)
    result["model_path"] = model_path
    save("model_download", True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        fast_inference=False,
        device_map={"": "npu:0"},
        use_gradient_checkpointing=False,
    )
    assert all(parameter.device.type == "npu" for parameter in model.parameters())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Avoid Qwen3's default thinking prefix: short completions must contain
    # the answer and measurable length variation for this core GRPO smoke.
    tokenizer.chat_template = CHAT_TEMPLATE
    result["chat_template"] = "simple_chatml_no_thinking_prefix"
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=3407,
    )
    adapter_before = snapshot_lora(model)
    result["adapter_before"] = {
        "tensor_count": len(adapter_before["tensors"]),
        "sha256": adapter_before["sha256"],
        "finite": adapter_before["finite"],
    }
    save("model_and_lora_setup", True)

    dataset = Dataset.from_list([
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "answer": answer,
        }
        for question, answer in QUESTIONS
    ])
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_answer, reward_concise],
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=str(OUTPUT / "grpo_trainer"),
            use_vllm=False,
            generation_batch_size=PER_DEVICE_BATCH_SIZE,
            per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=1,
            num_generations=NUM_GENERATIONS,
            max_prompt_length=MAX_PROMPT_LENGTH,
            max_completion_length=MAX_COMPLETION_LENGTH,
            temperature=0.8,
            top_p=0.95,
            top_k=20,
            beta=0.0,
            num_iterations=1,
            loss_type="dapo",
            # Unsloth enables this for explicit DAPO when left unset. Keep
            # capped rollouts in the loss to isolate the zero-gradient failure.
            mask_truncated_completions=False,
            scale_rewards="group",
            max_steps=3,
            learning_rate=5e-6,
            weight_decay=0.0,
            warmup_steps=0,
            lr_scheduler_type="constant",
            logging_steps=1,
            optim="adamw_torch",
            bf16=True,
            fp16=False,
            gradient_checkpointing=False,
            disable_dropout=True,
            use_transformers_paged=False,
            use_liger_loss=False,
            remove_unused_columns=False,
            shuffle_dataset=False,
            save_strategy="no",
            report_to="none",
            seed=3407,
            dataloader_num_workers=0,
        ),
    )

    result["mask_truncated_completions"] = trainer.mask_truncated_completions
    result["trainer_eos_token_id"] = trainer.eos_token_id
    save("grpo_trainer")
    assert trainer.mask_truncated_completions is False, "Unexpected truncation masking"
    original_prepare_inputs = trainer._prepare_inputs

    def observe_training_batch(generation_batch, _trainer=trainer):
        prepared = original_prepare_inputs(generation_batch)
        advantages = prepared.get("advantages") if isinstance(prepared, dict) else None
        if advantages is not None:
            values = advantages.detach().float().cpu()
            completion_ids = prepared["completion_ids"].detach().cpu()
            completion_mask = prepared["completion_mask"].detach().cpu()
            has_eos = completion_ids.eq(_trainer.eos_token_id).any(dim=1)
            loss_tokens = completion_mask.sum(dim=1)
            # TRL would zero the entire row for each completion without EOS.
            tokens_with_truncation_mask = loss_tokens * has_eos
            ADVANTAGE_CHECKS.append({
                "step_before_update": _trainer.state.global_step,
                "shape": list(values.shape),
                "finite": bool(torch.isfinite(values).all()),
                "nonzero": bool((values != 0).any()),
                "absolute_sum": float(values.abs().sum()),
                "advantage_per_completion": values.tolist(),
                "has_eos_per_completion": has_eos.tolist(),
                "loss_tokens_per_completion": loss_tokens.tolist(),
                "loss_tokens_if_truncation_masked": tokens_with_truncation_mask.tolist(),
            })
            result["advantages"] = ADVANTAGE_CHECKS
            RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps({"grpo_training_batch": ADVANTAGE_CHECKS[-1]}), flush=True)
            assert int(loss_tokens.sum()) > 0, "GRPO completion mask has no active tokens"
        return prepared

    trainer._prepare_inputs = observe_training_batch

    class CheckLoRAGradients(TrainerCallback):
        def on_pre_optimizer_step(
            self, args, state, control, model=None, _trainer=trainer, **kwargs
        ):
            active_model = model if model is not None else _trainer.model
            gradient_count = 0
            finite = True
            squared_norm = 0.0
            for name, parameter in active_model.named_parameters():
                if "lora_" not in name or parameter.grad is None:
                    continue
                gradient = parameter.grad.detach().float()
                gradient_count += 1
                finite = finite and bool(torch.isfinite(gradient).all())
                squared_norm += float((gradient * gradient).sum().cpu())
            GRADIENT_CHECKS.append({
                "step": int(state.global_step),
                "tensor_count": gradient_count,
                "finite": finite,
                "gradient_norm": math.sqrt(squared_norm),
            })
            result["lora_gradients"] = GRADIENT_CHECKS
            RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
            assert gradient_count > 0, "No LoRA gradients reached the optimizer"
            assert finite, "Non-finite LoRA gradient"
            assert squared_norm > 0.0, "LoRA gradients were all zero"

    trainer.add_callback(CheckLoRAGradients())
    train_metrics = trainer.train().metrics
    torch.npu.synchronize()
    assert trainer.state.global_step == 3, trainer.state.global_step

    history = numeric_history(trainer.state.log_history)
    step_rows = [entry for entry in history if "loss" in entry]
    assert len(step_rows) == 3, step_rows
    losses = [float(entry["loss"]) for entry in step_rows]
    rewards = [float(entry["reward"]) for entry in step_rows]
    reward_stds = [float(entry["reward_std"]) for entry in step_rows]
    assert all(math.isfinite(value) for value in losses), losses
    assert all(math.isfinite(value) for value in rewards), rewards
    assert all(math.isfinite(value) for value in reward_stds), reward_stds
    assert any(value > 0 for value in reward_stds), reward_stds
    assert len(COMPLETION_GROUPS) == 3, len(COMPLETION_GROUPS)
    assert all(len(group["completions"]) == NUM_GENERATIONS
               for group in COMPLETION_GROUPS)
    assert any(text.strip() for group in COMPLETION_GROUPS
               for text in group["completions"])
    assert len(ADVANTAGE_CHECKS) == 3, ADVANTAGE_CHECKS
    assert all(check["finite"] for check in ADVANTAGE_CHECKS), ADVANTAGE_CHECKS
    assert any(check["nonzero"] for check in ADVANTAGE_CHECKS), ADVANTAGE_CHECKS
    assert len(GRADIENT_CHECKS) == 3, GRADIENT_CHECKS
    assert all(check["tensor_count"] > 0 and check["finite"]
               and check["gradient_norm"] > 0 for check in GRADIENT_CHECKS), GRADIENT_CHECKS

    adapter_after = snapshot_lora(model)
    changed = changed_lora_parameters(
        adapter_before["tensors"], adapter_after["tensors"]
    )
    assert changed, "No LoRA parameters changed after GRPO"
    assert adapter_before["sha256"] != adapter_after["sha256"]
    assert adapter_after["finite"], "Non-finite LoRA weights after GRPO"
    result.update(
        steps=3,
        changed_lora_parameters=len(changed),
        adapter_after={
            "tensor_count": len(adapter_after["tensors"]),
            "sha256": adapter_after["sha256"],
            "finite": adapter_after["finite"],
        },
        train_metrics={
            key: json_scalar(value) for key, value in train_metrics.items()
            if isinstance(value, (int, float, bool)) or hasattr(value, "item")
        },
        log_history=history,
        completion_groups=COMPLETION_GROUPS,
        advantages=ADVANTAGE_CHECKS,
        lora_gradients=GRADIENT_CHECKS,
        compile_mode=os.environ.get("UNSLOTH_COMPILE_DISABLE"),
        sequence_packing=os.environ.get("UNSLOTH_GRPO_SEQ_PACKING"),
    )
    save("grpo_training", True)

    ADAPTER.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER))
    tokenizer.save_pretrained(str(ADAPTER))
    assert (ADAPTER / "adapter_config.json").is_file()
    assert (ADAPTER / "adapter_model.safetensors").is_file()
    save("grpo_adapter_saved", True)

    del trainer, model, dataset, adapter_before, adapter_after
    gc.collect()
    torch.npu.empty_cache()
    save("grpo_adapter_reload")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        fast_inference=False,
        device_map={"": "npu:0"},
    )
    assert all(parameter.device.type == "npu" for parameter in model.parameters())
    loaded_lora = [name for name, _ in model.named_parameters() if "lora_" in name]
    assert loaded_lora, "Reloaded model has no LoRA parameters"
    FastLanguageModel.for_inference(model)
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is 2 + 2? Reply with only the number."},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    batch = tokenizer(prompt, return_tensors="pt").to("npu")
    with torch.inference_mode():
        tokens = model.generate(**batch, max_new_tokens=16, do_sample=False)
    torch.npu.synchronize()
    generated = tokens[0, batch.input_ids.shape[1]:]
    assert generated.numel() > 0
    generation = tokenizer.decode(generated, skip_special_tokens=True).strip()
    assert generation, "Reloaded adapter generated an empty response"
    result.update(
        reloaded_lora_tensors=len(loaded_lora),
        post_reload_generation=generation,
    )
    print("Generated after GRPO adapter reload:", generation)
    save("grpo_adapter_reload", True)
    result["status"] = "passed"
    save("complete")
except Exception as exc:
    result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    save(result["stage"])
    traceback.print_exc()
    raise
