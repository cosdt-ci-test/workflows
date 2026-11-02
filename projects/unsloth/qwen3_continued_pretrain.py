"""Minimal raw-text continued pretraining trial for Unsloth on Ascend NPU."""
import gc
import json
import math
from pathlib import Path
import traceback

OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)
RESULT_PATH = OUTPUT / "result.json"
ADAPTER = OUTPUT / "continued_pretrain_adapter"
result = {
    "status": "running", "stage": "continued_pretrain_setup",
    "completed": [], "scenario": "continued-pretrain",
    "model": "Qwen/Qwen3-0.6B", "gradient_checkpointing_mode": "disabled",
    "objective": "causal_language_modeling_full_text",
    "chat_template_applied_to_training": False,
    "response_masking": False,
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
    tied_word_embeddings = bool(model.config.tie_word_embeddings)
    tied_parameter_identity = (
        model.get_input_embeddings().weight is model.get_output_embeddings().weight
    )
    model = FastLanguageModel.get_peft_model(
        model, r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing=False, random_state=3407,
        use_rslora=True,
    )
    trainable = {name: parameter for name, parameter in model.named_parameters()
                 if parameter.requires_grad}
    assert trainable, "No trainable CPT parameters"
    assert any("embed_tokens" in name for name in trainable), sorted(trainable)[:10]
    assert any("modules_to_save" in name for name in trainable), sorted(trainable)[-10:]
    before = {name: parameter.detach().cpu().clone()
              for name, parameter in trainable.items()}
    result["trainable_parameters"] = sum(
        parameter.numel() for parameter in trainable.values()
    )
    result["tied_word_embeddings"] = tied_word_embeddings
    result["tied_parameter_identity"] = tied_parameter_identity
    result["separate_lm_head_trainable"] = any(
        "lm_head" in name for name in trainable
    )
    save("continued_pretrain_lora_setup", True)

    # Raw documents train every token. No chat template or assistant masking
    # is applied, which distinguishes CPT from instruction fine-tuning.
    documents = [
        "Qwen models process text by converting it into tokens and predicting the next token. "
        "Language modeling learns statistical structure from complete documents. ",
        "Continued pretraining adapts a model with unlabeled text so that its next-token "
        "predictions reflect a target domain while retaining general language behavior. ",
        "A transformer tokenizer maps text into integer identifiers, and the language model "
        "learns to predict each following token from the preceding context. ",
    ] * 4
    texts = [document.strip() + tokenizer.eos_token for document in documents]
    token_lengths = [len(tokenizer(text, truncation=True, max_length=256)["input_ids"])
                     for text in texts]
    result["max_training_tokens"] = max(token_lengths)
    result["training_documents"] = len(texts)
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=Dataset.from_dict({"text": texts}),
        args=SFTConfig(
            output_dir=str(OUTPUT / "continued_pretrain_trainer"),
            dataset_text_field="text", max_length=256, packing=False,
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            max_steps=3, learning_rate=2e-5, logging_steps=1,
            optim="adamw_torch", bf16=True, fp16=False,
            gradient_checkpointing=False, save_strategy="no",
            report_to="none", seed=3407, dataloader_num_workers=0,
        ),
    )
    save("continued_pretrain_training")
    metrics = trainer.train().metrics
    torch.npu.synchronize()
    assert trainer.state.global_step == 3
    assert math.isfinite(metrics["train_loss"]), metrics
    changed = [name for name, parameter in trainable.items()
               if not torch.equal(before[name], parameter.detach().cpu())]
    assert changed, "No LoRA parameters changed during continued pretraining"
    result.update(metrics=metrics, changed_lora_parameters=len(changed), steps=3)
    save("continued_pretrain_training", True)

    ADAPTER.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER))
    tokenizer.save_pretrained(str(ADAPTER))
    assert (ADAPTER / "adapter_config.json").is_file()
    assert (ADAPTER / "adapter_model.safetensors").is_file()
    save("continued_pretrain_adapter_saved", True)

    del trainer, model, trainable, before
    gc.collect()
    torch.npu.empty_cache()
    save("continued_pretrain_reload")
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
    save("continued_pretrain_reload", True)
    result["status"] = "passed"
    save("complete")
except Exception as exc:
    result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    save(result["stage"])
    traceback.print_exc()
    raise
