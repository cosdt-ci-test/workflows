"""Small NPU trial derived from the official Qwen3 Instruct notebook.

Source and intentional changes are documented in README.md. No Unsloth
forward/kernel patches or Transformers-only substitutes are installed here.
"""
import json
import math
import os
from pathlib import Path
import traceback

OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)
scenario = os.environ.get("TEST_SCENARIO", "baseline")
result = {"status": "running", "stage": "import", "completed": [],
          "unsloth_ref": os.environ.get("UNSLOTH_REF"),
          "zoo_ref": os.environ.get("ZOO_REF"), "model": "Qwen/Qwen3-0.6B",
          "scenario": scenario}


def record(stage, completed=False):
    result["stage"] = stage
    if completed:
        result["completed"].append(stage)
    (OUTPUT / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"stage": stage, "completed": completed}), flush=True)


def checkpointing_for_scenario(value):
    mode = {
        "baseline": False,
        "standard-gc": True,
        "unsloth-gc": "unsloth",
        "packing": False,
    }.get(value)
    if mode is None and value != "baseline":
        raise ValueError(f"Unknown test scenario: {value!r}")
    return mode


try:
    record("scenario_setup")
    checkpointing = checkpointing_for_scenario(scenario)
    gradient_checkpointing = bool(checkpointing)
    trainer_checkpointing = checkpointing is True
    # Unsloth falls back to standard checkpointing below 512 tokens.
    max_seq_length = 512 if checkpointing == "unsloth" else 256
    result["gradient_checkpointing_mode"] = (
        "disabled" if checkpointing is False else
        "standard" if checkpointing is True else checkpointing
    )
    result["max_seq_length"] = max_seq_length
    print("Test scenario:", scenario)
    record("import")
    # torch_npu registers the NPU backend; Unsloth precedes HF/TRL imports.
    import torch
    import torch_npu  # noqa: F401 - registers the NPU backend
    from unsloth import FastLanguageModel
    from unsloth.device_type import DEVICE_TYPE
    from datasets import Dataset
    from modelscope import snapshot_download
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    class CheckGradientCheckpointing(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            # Trainer enables checkpointing inside train(), before this event.
            enabled_modules = [
                name for name, module in model.named_modules()
                if getattr(module, "gradient_checkpointing", False)
            ]
            enabled = bool(getattr(model, "is_gradient_checkpointing", False))
            effective_mode = getattr(model, "_unsloth_gradient_checkpointing", None)
            checkpoint_function = torch.utils.checkpoint.CheckpointFunction.__name__
            result["gradient_checkpointing_enabled"] = enabled
            result["trainer_gradient_checkpointing"] = bool(args.gradient_checkpointing)
            result["gradient_checkpointing_modules"] = enabled_modules
            result["effective_checkpointing_mode"] = effective_mode
            result["checkpoint_function"] = checkpoint_function
            record("training")
            assert bool(args.gradient_checkpointing) == trainer_checkpointing, "Trainer checkpointing disagrees with scenario"
            assert effective_mode == checkpointing, (
                f"Effective checkpointing mode {effective_mode!r} != {checkpointing!r}"
            )
            if gradient_checkpointing:
                assert enabled and enabled_modules, "Model checkpointing is not enabled"
            else:
                assert not enabled and not enabled_modules, "Baseline unexpectedly enabled checkpointing"
            if checkpointing == "unsloth":
                assert checkpoint_function == "UnslothCheckpointFunction", checkpoint_function
            print("Checkpointing state:", scenario, enabled, enabled_modules)

    assert DEVICE_TYPE == "npu", DEVICE_TYPE
    assert torch.npu.is_available()
    torch.npu.set_device(0)
    assert torch.npu.is_bf16_supported(), "BF16 is required for this trial"
    result["device"] = torch.npu.get_device_name(0)
    record("import", True)

    record("model_download")
    model_path = snapshot_download(result["model"])
    result["model_path"] = model_path
    record("model_download", True)

    record("model_load")
    # Keep the baseline loader call equivalent to successful run #4.
    load_kwargs = {"use_gradient_checkpointing": checkpointing} if gradient_checkpointing else {}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path, max_seq_length=max_seq_length, dtype=torch.bfloat16,
        load_in_4bit=False, load_in_8bit=False, load_in_16bit=True,
        full_finetuning=False, fast_inference=False, device_map={"": "npu:0"},
        **load_kwargs,
    )
    assert all(p.device.type == "npu" for p in model.parameters())
    record("model_load", True)

    record("lora_setup")
    model = FastLanguageModel.get_peft_model(
        model, r=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing=checkpointing, random_state=3407,
    )
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    before = {n: p.detach().cpu().clone() for n, p in trainable.items()}
    record("lora_setup", True)

    # Local conversations keep dataset downloads out of the training trial.
    base_conversations = [
        [{"role": "user", "content": "What is the capital of France?"},
         {"role": "assistant", "content": "The capital of France is Paris."}],
        [{"role": "user", "content": "What is two plus two?"},
         {"role": "assistant", "content": "Two plus two equals four."}],
        [{"role": "user", "content": "Name a primary color."},
         {"role": "assistant", "content": "Red is a primary color."}],
    ]
    if checkpointing == "unsloth":
        long_context = (
            "This sentence is repeated to exercise long-context activation "
            "recomputation under Unsloth checkpointing. " * 50
        )
        conversations = [
            [{"role": "user", "content": long_context + c[0]["content"]}, c[1]]
            for c in base_conversations
        ] * 4
    else:
        conversations = base_conversations * (8 if scenario == "packing" else 4)
    texts = [tokenizer.apply_chat_template(
        c, tokenize=False, add_generation_prompt=False, enable_thinking=False
    ) for c in conversations]
    token_lengths = [len(tokenizer(text, truncation=True, max_length=max_seq_length)["input_ids"])
                     for text in texts]
    result["max_training_tokens"] = max(token_lengths)
    if checkpointing == "unsloth":
        assert result["max_training_tokens"] == max_seq_length, result["max_training_tokens"]
    training_dataset = Dataset.from_dict({"text": texts})
    packing_options = {}
    if scenario == "packing":
        # Pre-tokenize without truncation so the packed dataset can be
        # compared with the exact original token stream.
        source_ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
        assert all(len(ids) < max_seq_length for ids in source_ids)
        training_dataset = Dataset.from_dict({"input_ids": source_ids})
        packing_options = dict(packing_strategy="wrapped", padding_free=False)
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=training_dataset,
        callbacks=[CheckGradientCheckpointing()],
        args=SFTConfig(
            output_dir=str(OUTPUT / "trainer"), dataset_text_field="text",
            max_length=max_seq_length, packing=scenario == "packing", per_device_train_batch_size=1,
            gradient_accumulation_steps=1, max_steps=3, learning_rate=2e-4,
            logging_steps=1, optim="adamw_torch", bf16=True, fp16=False,
            gradient_checkpointing=trainer_checkpointing, save_strategy="no", report_to="none",
            seed=3407, dataloader_num_workers=0,
            **packing_options,
        ),
    )
    if scenario == "packing":
        record("packing_validation")
        packed_ids = trainer.train_dataset["input_ids"]
        expected = [token for sample in source_ids for token in sample]
        actual = [token for sample in packed_ids for token in sample]
        assert actual == expected, "Packing changed, lost or reordered source tokens"
        assert 3 <= len(packed_ids) < len(source_ids)
        assert all(len(ids) == max_seq_length for ids in packed_ids[:-1])
        assert 0 < len(packed_ids[-1]) <= max_seq_length
        assert trainer.args.packing and trainer.args.packing_strategy == "wrapped"
        assert not trainer.padding_free, "Unexpected padding-free attention path"
        # The first full block contains several original sample boundaries.
        boundary = 0
        first_block_boundaries = 0
        for ids in source_ids:
            boundary += len(ids)
            first_block_boundaries += boundary < max_seq_length
        assert first_block_boundaries >= 2, first_block_boundaries
        batch = trainer.data_collator([trainer.train_dataset[0]])
        assert batch["input_ids"].shape == (1, max_seq_length)
        assert int(batch["attention_mask"].sum()) == max_seq_length
        assert bool((batch["labels"] != -100).any())
        result["packing"] = {
            "strategy": "wrapped", "padding_free": trainer.padding_free,
            "source_samples": len(source_ids), "packed_samples": len(packed_ids),
            "source_tokens": len(expected), "packed_tokens": len(actual),
            "first_block_sample_boundaries": first_block_boundaries,
            "packed_lengths": [len(ids) for ids in packed_ids],
            "attention_implementation": model.config._attn_implementation,
        }
        record("packing_validation", True)
    record("training")
    metrics = trainer.train().metrics
    torch.npu.synchronize()
    assert trainer.state.global_step == 3, trainer.state.global_step
    assert math.isfinite(metrics["train_loss"]), metrics
    changed = [n for n, p in trainable.items()
               if not torch.equal(before[n], p.detach().cpu())]
    assert changed, "No LoRA parameters changed"
    result.update(metrics=metrics, changed_lora_parameters=len(changed), steps=3)
    record("training", True)

    record("save")
    adapter = OUTPUT / "adapter"
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    assert (adapter / "adapter_config.json").is_file()
    assert (adapter / "adapter_model.safetensors").is_file()
    record("save", True)

    record("reload_and_inference")
    # Reload through Unsloth, so generation checks the saved adapter too.
    import gc
    del trainer, model, trainable, before
    gc.collect()
    torch.npu.empty_cache()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter), max_seq_length=max_seq_length, dtype=torch.bfloat16,
        load_in_4bit=False, load_in_8bit=False, load_in_16bit=True,
        fast_inference=False, device_map={"": "npu:0"},
    )
    assert all(p.device.type == "npu" for p in model.parameters())
    FastLanguageModel.for_inference(model)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is two plus two?"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    batch = tokenizer(prompt, return_tensors="pt").to("npu")
    with torch.inference_mode():
        tokens = model.generate(**batch, max_new_tokens=32, do_sample=False)
    torch.npu.synchronize()
    generated = tokens[0, batch.input_ids.shape[1]:]
    assert generated.numel() > 0
    result["generation"] = tokenizer.decode(generated, skip_special_tokens=True)
    print("Generated:", result["generation"])
    record("reload_and_inference", True)
    result["status"] = "passed"
    record("complete")
except Exception as exc:
    result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    record(result["stage"])
    traceback.print_exc()
    raise
