"""Minimal BF16 full-parameter fine-tuning trial on Ascend NPU."""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import traceback

OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)
RESULT_PATH = OUTPUT / "result.json"
MODEL_DIR = OUTPUT / "full_finetuned_model"
result = {}


def load_result(phase):
    if RESULT_PATH.exists():
        data = json.loads(RESULT_PATH.read_text())
    else:
        data = {
            "status": "running", "stage": "full_finetune_setup",
            "completed": [], "scenario": "full-finetune",
            "model": "Qwen/Qwen3-0.6B", "gradient_checkpointing_mode": "disabled",
        }
    data["phase"] = phase
    return data


def save(stage, completed=False):
    result["stage"] = stage
    if completed and stage not in result["completed"]:
        result["completed"].append(stage)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps({"phase": result["phase"], "stage": stage,
                      "completed": completed}), flush=True)


def weight_files():
    files = sorted(MODEL_DIR.glob("*.safetensors"))
    index_path = MODEL_DIR / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        missing = [name for name in set(index.get("weight_map", {}).values())
                   if not (MODEL_DIR / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing full-model weight shards: {sorted(missing)}")
    return files


def validate_float_dtypes(dtypes_by_key):
    float_dtypes = set(dtypes_by_key.values())
    fp32_tensors = sorted(
        key for key, dtype in dtypes_by_key.items() if dtype == "F32"
    )
    non_norm_dtypes = {
        dtype for key, dtype in dtypes_by_key.items()
        if "norm" not in key.lower()
    }
    assert "BF16" in float_dtypes, float_dtypes
    assert non_norm_dtypes == {"BF16"}, non_norm_dtypes
    assert all("norm" in key.lower() for key in fp32_tensors), fp32_tensors
    return sorted(float_dtypes), fp32_tensors


def sample_parameters(model):
    wanted = {
        "model.embed_tokens.weight": 128,
        "model.layers.0.self_attn.q_proj.weight": None,
        "model.layers.27.mlp.down_proj.weight": None,
    }
    samples = {}
    for name, parameter in model.named_parameters():
        if name not in wanted:
            continue
        value = parameter.detach().cpu()
        rows = wanted[name]
        if rows is not None:
            value = value[:rows]
        samples[name] = value.clone()
    assert set(samples) == set(wanted), sorted(samples)
    return samples


def train_and_save():
    import torch
    import torch_npu
    from unsloth import FastLanguageModel
    from unsloth.device_type import DEVICE_TYPE
    from datasets import Dataset
    from modelscope import snapshot_download
    from safetensors import safe_open
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
        load_in_4bit=False, load_in_8bit=False, load_in_16bit=False,
        full_finetuning=True, fast_inference=False,
        device_map={"": "npu:0"}, use_gradient_checkpointing=False,
    )
    parameters = list(model.parameters())
    trainable_count = sum(parameter.numel() for parameter in parameters
                          if parameter.requires_grad)
    total_count = sum(parameter.numel() for parameter in parameters)
    assert trainable_count == total_count, (trainable_count, total_count)
    assert all(parameter.device.type == "npu" for parameter in parameters)
    result.update(trainable_parameters=trainable_count, total_parameters=total_count,
                  full_finetuning=True)
    samples = sample_parameters(model)
    save("full_model_setup", True)

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
            output_dir=str(OUTPUT / "full_finetune_trainer"),
            dataset_text_field="text", max_length=256, packing=False,
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            max_steps=3, learning_rate=2e-5, logging_steps=1,
            optim="adamw_torch", bf16=True, fp16=False,
            gradient_checkpointing=False, save_strategy="no",
            report_to="none", seed=3407, dataloader_num_workers=0,
        ),
    )
    save("full_finetune_training")
    metrics = trainer.train().metrics
    torch.npu.synchronize()
    assert trainer.state.global_step == 3
    assert math.isfinite(metrics["train_loss"]), metrics
    changed = 0
    current = dict(model.named_parameters())
    for name, before in samples.items():
        value = current[name].detach().cpu()
        if value.shape != before.shape:
            value = value[:before.shape[0]]
        changed += int(not torch.equal(before, value))
    assert changed == len(samples), changed
    result.update(metrics=metrics, changed_sample_parameters=changed, steps=3)
    save("full_finetune_training", True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    files = weight_files()
    assert files, "No full-model safetensors files were written"
    config = json.loads((MODEL_DIR / "config.json").read_text())
    configured_dtype = config.get("dtype", config.get("torch_dtype"))
    float_dtypes_by_key = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as data:
            for key in data.keys():
                dtype = data.get_slice(key).get_dtype()
                if dtype in ("F16", "F32", "BF16"):
                    float_dtypes_by_key[key] = dtype
    float_dtypes, fp32_tensors = validate_float_dtypes(float_dtypes_by_key)
    assert configured_dtype == "bfloat16", configured_dtype
    result.update(model_directory=str(MODEL_DIR),
                  model_weight_files=[path.name for path in files],
                  model_float_dtypes=float_dtypes,
                  fp32_norm_tensors=fp32_tensors,
                  configured_dtype=configured_dtype,
                  model_size_bytes=sum(path.stat().st_size for path in files),
                  status="full_model_ready")
    save("full_model_saved", True)

    del trainer, model, parameters, current, samples
    gc.collect()
    torch.npu.empty_cache()


def reload_and_inference():
    import torch
    import torch_npu
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert "unsloth" not in sys.modules, "Unsloth was imported in clean reload process"
    assert "unsloth_zoo" not in sys.modules, "Unsloth Zoo was imported in clean reload process"
    assert result.get("status") == "full_model_ready", result.get("status")
    assert torch.npu.is_available()
    torch.npu.set_device(0)
    result["device"] = torch.npu.get_device_name(0)
    result["unsloth_imported"] = False
    save("clean_reload_environment", True)

    files = weight_files()
    assert files and (MODEL_DIR / "config.json").is_file()
    save("full_model_files_verified", True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16,
    ).to("npu").eval()
    module = type(model).__module__
    assert module.startswith("transformers."), module
    assert all(parameter.device.type == "npu" for parameter in model.parameters())
    result.update(reloaded_class=type(model).__name__, reloaded_module=module)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is two plus two?"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    batch = tokenizer(prompt, return_tensors="pt").to("npu")
    position_ids = torch.arange(
        batch.input_ids.shape[1], dtype=torch.long, device="npu",
    ).unsqueeze(0)
    with torch.inference_mode():
        tokens = model.generate(
            **batch, position_ids=position_ids,
            max_new_tokens=32, do_sample=False,
        )
    torch.npu.synchronize()
    generated = tokens[0, batch.input_ids.shape[1]:]
    assert generated.numel() > 0
    result["generation"] = tokenizer.decode(generated, skip_special_tokens=True)
    print("Generated:", result["generation"])
    save("full_model_reload", True)
    result["status"] = "passed"
    save("complete")


def main():
    global result
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["train", "reload"], required=True)
    args = parser.parse_args()
    result = load_result(args.phase)
    try:
        save("scenario_setup", True)
        if args.phase == "train":
            train_and_save()
        else:
            reload_and_inference()
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(result["stage"])
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
