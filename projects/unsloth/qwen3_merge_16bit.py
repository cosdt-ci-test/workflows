"""Train LoRA, merge into BF16 weights, then reload without Unsloth."""
import gc
import json
import math
from pathlib import Path
import traceback

OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)
RESULT_PATH = OUTPUT / "result.json"
MERGED = OUTPUT / "merged_16bit"
result = {
    "status": "running", "stage": "merge_setup", "completed": [],
    "scenario": "merge-16bit", "gradient_checkpointing_mode": "disabled",
    "model": "Qwen/Qwen3-0.6B",
}


def save(stage, completed=False):
    result["stage"] = stage
    if completed and stage not in result["completed"]:
        result["completed"].append(stage)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps({"stage": stage, "completed": completed}), flush=True)


def merged_weight_files():
    files = sorted(MERGED.glob("*.safetensors"))
    index_path = MERGED / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shards = sorted(set(index.get("weight_map", {}).values()))
        missing = [name for name in shards if not (MERGED / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing merged weight shards: {missing}")
    return files


try:
    save("scenario_setup", True)
    import torch
    import torch_npu
    from unsloth import FastLanguageModel
    from unsloth.device_type import DEVICE_TYPE
    from datasets import Dataset
    from modelscope import snapshot_download
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

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

    conversations = [
        [{"role": "user", "content": "What is the capital of France?"},
         {"role": "assistant", "content": "The capital of France is Paris."}],
        [{"role": "user", "content": "What is two plus two?"},
         {"role": "assistant", "content": "Two plus two equals four."}],
        [{"role": "user", "content": "Name a primary color."},
         {"role": "assistant", "content": "Red is a primary color."}],
    ] * 4
    texts = [tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    ) for conversation in conversations]
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=Dataset.from_dict({"text": texts}),
        args=SFTConfig(
            output_dir=str(OUTPUT / "trainer"), dataset_text_field="text",
            max_length=256, packing=False, per_device_train_batch_size=1,
            gradient_accumulation_steps=1, max_steps=3, learning_rate=2e-4,
            logging_steps=1, optim="adamw_torch", bf16=True, fp16=False,
            gradient_checkpointing=False, save_strategy="no",
            report_to="none", seed=3407, dataloader_num_workers=0,
        ),
    )
    save("training")
    metrics = trainer.train().metrics
    torch.npu.synchronize()
    assert trainer.state.global_step == 3
    assert math.isfinite(metrics["train_loss"]), metrics
    changed = [name for name, parameter in trainable.items()
               if not torch.equal(before[name], parameter.detach().cpu())]
    assert changed, "No LoRA parameters changed"
    result.update(metrics=metrics, changed_lora_parameters=len(changed))
    save("training", True)

    save("merge_export")
    MERGED.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        str(MERGED), tokenizer, save_method="merged_16bit",
        dtype=torch.bfloat16,
    )
    weight_files = merged_weight_files()
    assert weight_files, "No merged safetensors files were written"
    float_dtypes = set()
    for weight_file in weight_files:
        with safe_open(str(weight_file), framework="pt", device="cpu") as data:
            for key in data.keys():
                dtype = data.get_slice(key).get_dtype()
                if dtype in ("F16", "F32", "BF16"):
                    float_dtypes.add(dtype)
    config = json.loads((MERGED / "config.json").read_text())
    configured_dtype = config.get("dtype", config.get("torch_dtype"))
    assert float_dtypes == {"BF16"}, float_dtypes
    assert configured_dtype == "bfloat16", configured_dtype
    result.update(
        merged_directory=str(MERGED),
        merged_weight_files=[path.name for path in weight_files],
        merged_float_dtypes=sorted(float_dtypes),
        configured_dtype=configured_dtype,
        merged_size_bytes=sum(path.stat().st_size for path in weight_files),
    )
    save("merge_export", True)

    del trainer, model, trainable, before
    gc.collect()
    torch.npu.empty_cache()

    save("independent_reload")
    reloaded_tokenizer = AutoTokenizer.from_pretrained(str(MERGED))
    reloaded = AutoModelForCausalLM.from_pretrained(
        str(MERGED), dtype=torch.bfloat16,
    ).to("npu").eval()
    assert all(parameter.device.type == "npu" for parameter in reloaded.parameters())
    prompt = reloaded_tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is two plus two?"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    batch = reloaded_tokenizer(prompt, return_tensors="pt").to("npu")
    with torch.inference_mode():
        tokens = reloaded.generate(
            **batch, max_new_tokens=32, do_sample=False,
        )
    torch.npu.synchronize()
    generated = tokens[0, batch.input_ids.shape[1]:]
    assert generated.numel() > 0
    result["generation"] = reloaded_tokenizer.decode(
        generated, skip_special_tokens=True,
    )
    print("Generated:", result["generation"])
    save("independent_reload", True)
    result["status"] = "passed"
    save("complete")
except Exception as exc:
    result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    save(result["stage"])
    traceback.print_exc()
    raise
