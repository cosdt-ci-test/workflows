"""Minimal local-preference DPO trial for Unsloth on Ascend NPU."""
import gc
import json
import math
from pathlib import Path
import traceback

OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)
RESULT_PATH = OUTPUT / "result.json"
ADAPTER = OUTPUT / "dpo_adapter"
result = {
    "status": "running", "stage": "dpo_setup", "completed": [],
    "scenario": "dpo", "gradient_checkpointing_mode": "disabled",
    "model": "Qwen/Qwen3-0.6B", "reference_strategy": "peft_adapter_disabled",
}


def save(stage, completed=False):
    result["stage"] = stage
    if completed and stage not in result["completed"]:
        result["completed"].append(stage)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps({"stage": stage, "completed": completed}), flush=True)


try:
    save("scenario_setup", True)
    import torch
    import torch_npu
    from unsloth import FastLanguageModel
    from unsloth.device_type import DEVICE_TYPE
    from datasets import Dataset
    from modelscope import snapshot_download
    from trl import DPOConfig, DPOTrainer

    assert DEVICE_TYPE == "npu", DEVICE_TYPE
    assert torch.npu.is_available()
    torch.npu.set_device(0)
    assert torch.npu.is_bf16_supported()
    result["device"] = torch.npu.get_device_name(0)
    save("import", True)

    model_path = snapshot_download(result["model"])
    result["model_path"] = model_path
    save("model_download", True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path, max_seq_length=256, dtype=torch.bfloat16,
        load_in_4bit=False, load_in_8bit=False, load_in_16bit=True,
        full_finetuning=False, fast_inference=False,
        device_map={"": "npu:0"}, use_gradient_checkpointing=False,
    )
    assert all(parameter.device.type == "npu" for parameter in model.parameters())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model, r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing=False, random_state=3407,
    )
    trainable = {name: parameter for name, parameter in model.named_parameters()
                 if parameter.requires_grad}
    assert trainable and all("lora_" in name for name in trainable)
    before = {name: parameter.detach().cpu().clone()
              for name, parameter in trainable.items()}
    save("model_and_lora_setup", True)

    preferences = [
        {"prompt": "What is two plus two?", "chosen": "4", "rejected": "5"},
        {"prompt": "What is the capital of France?", "chosen": "Paris", "rejected": "Lyon"},
        {"prompt": "Name a primary color.", "chosen": "Red", "rejected": "Orange"},
        {"prompt": "What state is water at zero Celsius?", "chosen": "Ice", "rejected": "Steam"},
        {"prompt": "In which direction does the sun rise?", "chosen": "East", "rejected": "West"},
        {"prompt": "Which is the largest planet in our solar system?", "chosen": "Jupiter", "rejected": "Mars"},
    ] * 2
    dataset = Dataset.from_list(preferences)
    trainer = DPOTrainer(
        model=model, ref_model=None, processing_class=tokenizer,
        train_dataset=dataset,
        args=DPOConfig(
            output_dir=str(OUTPUT / "dpo_trainer"),
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            max_steps=3, learning_rate=5e-6, logging_steps=1,
            optim="adamw_torch", bf16=True, fp16=False, beta=0.1,
            max_length=256, max_prompt_length=128, max_completion_length=64,
            gradient_checkpointing=False, save_strategy="no",
            report_to="none", seed=3407, dataloader_num_workers=0,
            dataset_num_proc=1, padding_value=tokenizer.pad_token_id,
            disable_dropout=True, precompute_ref_log_probs=False,
            sync_ref_model=False, loss_type="sigmoid",
        ),
    )
    save("dpo_training")
    metrics = trainer.train().metrics
    torch.npu.synchronize()
    assert trainer.state.global_step == 3, trainer.state.global_step
    loss = metrics.get("train_loss", metrics.get("loss"))
    assert loss is not None and math.isfinite(loss), metrics
    changed = [name for name, parameter in trainable.items()
               if not torch.equal(before[name], parameter.detach().cpu())]
    assert changed, "No LoRA parameters changed after DPO"
    result.update(metrics=metrics, changed_lora_parameters=len(changed), steps=3)
    save("dpo_training", True)

    ADAPTER.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER))
    tokenizer.save_pretrained(str(ADAPTER))
    assert (ADAPTER / "adapter_config.json").is_file()
    assert (ADAPTER / "adapter_model.safetensors").is_file()
    save("dpo_adapter_saved", True)

    del trainer, model, trainable, before
    gc.collect()
    torch.npu.empty_cache()
    save("dpo_adapter_reload")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER), max_seq_length=256, dtype=torch.bfloat16,
        load_in_4bit=False, load_in_8bit=False, load_in_16bit=True,
        fast_inference=False, device_map={"": "npu:0"},
    )
    assert all(parameter.device.type == "npu" for parameter in model.parameters())
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
    save("dpo_adapter_reload", True)
    result["status"] = "passed"
    save("complete")
except Exception as exc:
    result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    save(result["stage"])
    traceback.print_exc()
    raise
