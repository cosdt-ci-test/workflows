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
result = {"status": "running", "stage": "import", "completed": [],
          "unsloth_ref": os.environ.get("UNSLOTH_REF"),
          "zoo_ref": os.environ.get("ZOO_REF"), "model": "Qwen/Qwen3-0.6B"}


def record(stage, completed=False):
    result["stage"] = stage
    if completed:
        result["completed"].append(stage)
    (OUTPUT / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"stage": stage, "completed": completed}), flush=True)


try:
    record("import")
    # torch_npu registers the NPU backend; Unsloth precedes HF/TRL imports.
    import torch
    import torch_npu
    from unsloth import FastLanguageModel
    from unsloth.device_type import DEVICE_TYPE
    from datasets import Dataset
    from modelscope import snapshot_download
    from trl import SFTConfig, SFTTrainer

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
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path, max_seq_length=256, dtype=torch.bfloat16,
        load_in_4bit=False, load_in_8bit=False, load_in_16bit=True,
        full_finetuning=False, fast_inference=False, device_map={"": "npu:0"},
    )
    assert all(p.device.type == "npu" for p in model.parameters())
    record("model_load", True)

    record("lora_setup")
    model = FastLanguageModel.get_peft_model(
        model, r=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing=False, random_state=3407,
    )
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    before = {n: p.detach().cpu().clone() for n, p in trainable.items()}
    record("lora_setup", True)

    # Local conversations keep dataset downloads out of the training trial.
    conversations = [
        [{"role": "user", "content": "What is the capital of France?"},
         {"role": "assistant", "content": "The capital of France is Paris."}],
        [{"role": "user", "content": "What is two plus two?"},
         {"role": "assistant", "content": "Two plus two equals four."}],
        [{"role": "user", "content": "Name a primary color."},
         {"role": "assistant", "content": "Red is a primary color."}],
    ] * 4
    texts = [tokenizer.apply_chat_template(
        c, tokenize=False, add_generation_prompt=False, enable_thinking=False
    ) for c in conversations]
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=Dataset.from_dict({"text": texts}),
        args=SFTConfig(
            output_dir=str(OUTPUT / "trainer"), dataset_text_field="text",
            max_length=256, packing=False, per_device_train_batch_size=1,
            gradient_accumulation_steps=1, max_steps=3, learning_rate=2e-4,
            logging_steps=1, optim="adamw_torch", bf16=True, fp16=False,
            gradient_checkpointing=False, save_strategy="no", report_to="none",
            seed=3407, dataloader_num_workers=0,
        ),
    )
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
        model_name=str(adapter), max_seq_length=256, dtype=torch.bfloat16,
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
