"""Verify full Trainer checkpoint and cross-process resume on Ascend NPU."""
import argparse
import gc
import json
import math
from pathlib import Path
import traceback

OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)
RESULT_PATH = OUTPUT / "result.json"
CHECKPOINT = OUTPUT / "trainer" / "checkpoint-1"
ADAPTER = OUTPUT / "adapter"


def load_result(phase):
    if RESULT_PATH.exists():
        result = json.loads(RESULT_PATH.read_text())
    else:
        result = {
            "status": "running", "stage": "checkpoint_resume_setup",
            "completed": [], "scenario": "checkpoint-resume",
            "gradient_checkpointing_mode": "disabled", "phase": phase,
        }
    result["phase"] = phase
    return result


def save(result, stage, completed=False):
    result["stage"] = stage
    if completed and stage not in result["completed"]:
        result["completed"].append(stage)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps({"phase": result["phase"], "stage": stage,
                      "completed": completed}), flush=True)


def trainer_checkpoint_files(path):
    required = {
        "trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
        "adapter_config.json",
    }
    present = {item.name for item in path.iterdir()}
    missing = required - present
    adapter_weights = [item for item in path.glob("adapter_model.*")]
    if not adapter_weights:
        missing.add("adapter_model.*")
    if missing:
        raise FileNotFoundError(f"Incomplete Trainer checkpoint {path}: {sorted(missing)}")
    return sorted(present), [item.name for item in adapter_weights]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["train", "resume"], required=True)
    args = parser.parse_args()
    result = load_result(args.phase)

    try:
        save(result, "scenario_setup", True)
        import torch
        import torch_npu
        from unsloth import FastLanguageModel
        from unsloth.device_type import DEVICE_TYPE
        from datasets import Dataset
        from modelscope import snapshot_download
        from safetensors.torch import load_file
        from transformers import TrainerCallback
        from trl import SFTConfig, SFTTrainer

        assert DEVICE_TYPE == "npu", DEVICE_TYPE
        assert torch.npu.is_available()
        torch.npu.set_device(0)
        assert torch.npu.is_bf16_supported()
        result["device"] = torch.npu.get_device_name(0)
        result["model"] = "Qwen/Qwen3-0.6B"
        save(result, "import", True)

        model_path = snapshot_download(result["model"])
        result["model_path"] = model_path
        save(result, "model_download", True)

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
        save(result, "model_and_lora_setup", True)

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

        class ResumeChecks(TrainerCallback):
            def on_train_begin(self, args, state, control, optimizer=None, **kwargs):
                result["resume_start_step"] = state.global_step
                result["optimizer_state_at_train_begin"] = bool(
                    optimizer is not None and optimizer.state
                )
                save(result, "resumed_training")
                assert state.global_step == 1, state.global_step

            def on_step_begin(self, args, state, control, **kwargs):
                if "resumed_first_step" not in result:
                    result["resumed_first_step"] = state.global_step + 1
                    save(result, "resumed_training")

        if args.phase == "train":
            trainer = SFTTrainer(
                model=model, processing_class=tokenizer,
                train_dataset=Dataset.from_dict({"text": texts}),
                args=SFTConfig(
                    output_dir=str(OUTPUT / "trainer"),
                    dataset_text_field="text", max_length=256, packing=False,
                    per_device_train_batch_size=1, gradient_accumulation_steps=1,
                    max_steps=1, save_strategy="steps", save_steps=1,
                    save_total_limit=1, learning_rate=2e-4, logging_steps=1,
                    optim="adamw_torch", bf16=True, fp16=False,
                    gradient_checkpointing=False, report_to="none", seed=3407,
                    dataloader_num_workers=0,
                ),
            )
            metrics = trainer.train().metrics
            torch.npu.synchronize()
            assert trainer.state.global_step == 1
            assert math.isfinite(metrics["train_loss"]), metrics
            files, adapters = trainer_checkpoint_files(CHECKPOINT)
            state = json.loads((CHECKPOINT / "trainer_state.json").read_text())
            assert state["global_step"] == 1, state["global_step"]
            result.update(initial_metrics=metrics, checkpoint=str(CHECKPOINT),
                          checkpoint_files=files, checkpoint_adapters=adapters,
                          checkpoint_global_step=state["global_step"],
                          status="checkpoint_ready")
            save(result, "trainer_checkpoint_saved", True)
            return

        files, adapters = trainer_checkpoint_files(CHECKPOINT)
        result.update(checkpoint_files=files, checkpoint_adapters=adapters)
        checkpoint_weights = load_file(str(CHECKPOINT / "adapter_model.safetensors"))
        trainer = SFTTrainer(
            model=model, processing_class=tokenizer,
            train_dataset=Dataset.from_dict({"text": texts}),
            callbacks=[ResumeChecks()],
            args=SFTConfig(
                output_dir=str(OUTPUT / "trainer-resumed"),
                dataset_text_field="text", max_length=256, packing=False,
                per_device_train_batch_size=1, gradient_accumulation_steps=1,
                max_steps=3, learning_rate=2e-4, logging_steps=1,
                optim="adamw_torch", bf16=True, fp16=False,
                gradient_checkpointing=False, save_strategy="no",
                report_to="none", seed=3407, dataloader_num_workers=0,
            ),
        )
        metrics = trainer.train(resume_from_checkpoint=str(CHECKPOINT)).metrics
        torch.npu.synchronize()
        assert trainer.state.global_step == 3, trainer.state.global_step
        assert result.get("resumed_first_step") == 2, result.get("resumed_first_step")
        optimizer_state = trainer.optimizer.state_dict().get("state", {})
        scheduler_state = trainer.lr_scheduler.state_dict()
        assert optimizer_state, "Optimizer state was not restored"
        assert scheduler_state, "Scheduler state was not restored"

        current_weights = model.state_dict()
        common_keys = checkpoint_weights.keys() & current_weights.keys()
        changed = sum(
            not torch.equal(checkpoint_weights[key].cpu(), current_weights[key].detach().cpu())
            for key in common_keys
        )
        assert changed, "No LoRA weights changed after resumed training"
        result.update(
            resumed_metrics=metrics, optimizer_state_entries=len(optimizer_state),
            scheduler_state=scheduler_state, changed_lora_tensors=changed,
            steps=3,
        )
        save(result, "resume_verified", True)

        ADAPTER.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ADAPTER))
        tokenizer.save_pretrained(str(ADAPTER))
        assert (ADAPTER / "adapter_config.json").is_file()
        assert (ADAPTER / "adapter_model.safetensors").is_file()
        save(result, "final_adapter_saved", True)

        del trainer, model
        gc.collect()
        torch.npu.empty_cache()
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
            tokens = model.generate(
                **batch, max_new_tokens=32, do_sample=False,
            )
        torch.npu.synchronize()
        generated = tokens[0, batch.input_ids.shape[1]:]
        assert generated.numel() > 0
        result["generation"] = tokenizer.decode(generated, skip_special_tokens=True)
        print("Generated:", result["generation"])
        save(result, "reload_and_inference", True)
        result["status"] = "passed"
        save(result, "complete")
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(result, result["stage"])
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
