"""Two-rank FSDP2 checkpoint/resume trial for Unsloth on Ascend NPU."""
import argparse
import json
import math
import os
from pathlib import Path
import traceback

# Unsloth must see this before it imports transformers and accelerate.
os.environ.setdefault("UNSLOTH_FSDP2", "1")

OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)
RESULT_PATH = OUTPUT / "result.json"
TRAINER_DIR = OUTPUT / "fsdp2_trainer"
CHECKPOINT = TRAINER_DIR / "checkpoint-1"
FINAL_ADAPTER = OUTPUT / "fsdp2_adapter"
rank = int(os.environ.get("RANK", "-1"))
local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
world_size = int(os.environ.get("WORLD_SIZE", "-1"))
result = {
    "status": "running", "stage": "fsdp2_setup", "completed": [],
    "scenario": "fsdp2-2", "model": "Qwen/Qwen3-0.6B",
    "world_size": world_size, "backend": "hccl",
}


def load_result(phase):
    if RESULT_PATH.exists():
        data = json.loads(RESULT_PATH.read_text())
    else:
        data = dict(result)
    data["phase"] = phase
    return data


def save(stage, completed=False):
    result["stage"] = stage
    if completed and stage not in result["completed"]:
        result["completed"].append(stage)
    if rank == 0:
        RESULT_PATH.write_text(json.dumps(result, indent=2))
        print(json.dumps({"rank": rank, "stage": stage,
                          "completed": completed}), flush=True)


def checkpoint_files():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    return sorted(
        str(path.relative_to(CHECKPOINT)) for path in CHECKPOINT.rglob("*")
        if path.is_file()
    )


def build_trainer(torch, tokenizer, model, max_steps, save_strategy,
                  callback, dataset):
    from trl import SFTConfig, SFTTrainer
    return SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=dataset,
        callbacks=[callback],
        args=SFTConfig(
            output_dir=str(TRAINER_DIR), dataset_text_field="text",
            max_length=128, packing=False, per_device_train_batch_size=1,
            gradient_accumulation_steps=1, max_steps=max_steps,
            save_strategy=save_strategy, save_steps=1, save_total_limit=1,
            learning_rate=2e-4, logging_steps=1, optim="adamw_torch",
            bf16=True, fp16=False, gradient_checkpointing=False,
            report_to="none", seed=3407, dataloader_num_workers=0,
        ),
    )


def main():
    global result
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["train", "resume"], required=True)
    args = parser.parse_args()
    phase = args.phase
    result = load_result(phase)
    dist = None
    try:
        save("scenario_setup", True)
        import torch
        import torch_npu
        import torch.distributed as dist
        from unsloth import FastLanguageModel
        from unsloth.device_type import DEVICE_TYPE
        from datasets import Dataset
        from modelscope import snapshot_download
        try:
            from torch.distributed.fsdp import FSDPModule
        except ImportError:
            from torch.distributed._composable.fsdp import FSDPModule
        from transformers import TrainerCallback

        assert rank in (0, 1) and local_rank in (0, 1) and world_size == 2
        torch.npu.set_device(local_rank)
        assert DEVICE_TYPE == "npu", DEVICE_TYPE
        assert torch.npu.is_available()
        assert torch.npu.is_bf16_supported()
        result.update(rank=rank, local_rank=local_rank)
        model_path = snapshot_download(result["model"])

        expected_start = 0 if phase == "train" else 1
        holder = {}

        class VerifyFSDP2(TrainerCallback):
            def on_train_begin(self, args, state, control, model=None, **kwargs):
                trainer = holder["trainer"]
                assert trainer.is_fsdp_enabled
                distributed_type = trainer.accelerator.state.distributed_type
                distributed_name = getattr(
                    distributed_type, "name", str(distributed_type)
                )
                plugin = trainer.accelerator.state.fsdp_plugin
                assert plugin is not None, "Accelerate FSDP plugin missing"
                version = int(getattr(plugin, "fsdp_version", 0))
                modules = [
                    name for name, module in trainer.model.named_modules()
                    if isinstance(module, FSDPModule)
                ]
                assert distributed_name == "FSDP", distributed_name
                assert version == 2, version
                assert modules, "No FSDP2-sharded modules found"
                assert torch.distributed.is_initialized()
                assert torch.distributed.get_backend() == "hccl"
                assert os.environ.get("ACCELERATE_USE_FSDP", "").lower() == "true"
                assert os.environ.get("FSDP_VERSION") == "2"
                assert state.global_step == expected_start, state.global_step
                result.update(
                    distributed_type=distributed_name,
                    fsdp_version=version,
                    fsdp_module_count=len(modules),
                    rank_backend=torch.distributed.get_backend(),
                    fsdp_start_step=state.global_step,
                    fsdp_state_dict_type=os.environ.get("FSDP_STATE_DICT_TYPE"),
                    device=torch.npu.get_device_name(local_rank),
                )
                if rank == 0:
                    save("fsdp2_verified", True)

            def on_step_begin(self, args, state, control, **kwargs):
                if phase == "resume" and "fsdp2_first_step" not in result:
                    result["fsdp2_first_step"] = state.global_step + 1
                    if rank == 0:
                        save("fsdp2_resumed")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path, max_seq_length=128, dtype=torch.bfloat16,
            load_in_4bit=False, load_in_8bit=False, load_in_16bit=True,
            full_finetuning=False, fast_inference=False,
            device_map={"": f"npu:{local_rank}"},
            use_gradient_checkpointing=False,
        )
        assert all(parameter.device.type == "npu" for parameter in model.parameters())
        model = FastLanguageModel.get_peft_model(
            model, r=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16, lora_dropout=0, bias="none",
            use_gradient_checkpointing=False, random_state=3407,
        )
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
        dataset = Dataset.from_dict({"text": texts})
        callback = VerifyFSDP2()
        save_strategy = "steps" if phase == "train" else "no"
        max_steps = 1 if phase == "train" else 3
        trainer = build_trainer(
            torch, tokenizer, model, max_steps, save_strategy, callback, dataset,
        )
        holder["trainer"] = trainer

        if phase == "train":
            metrics = trainer.train().metrics
            torch.npu.synchronize()
            assert trainer.state.global_step == 1
            assert math.isfinite(metrics["train_loss"]), metrics
            dist.barrier()
            if rank == 0:
                files = checkpoint_files()
                state = json.loads((CHECKPOINT / "trainer_state.json").read_text())
                assert state["global_step"] == 1, state["global_step"]
                result.update(
                    checkpoint=str(CHECKPOINT), checkpoint_files=files,
                    checkpoint_global_step=state["global_step"],
                    initial_metrics=metrics, status="checkpoint_ready",
                )
                save("fsdp2_checkpoint_saved", True)
            return

        checkpoint_files()
        metrics = trainer.train(resume_from_checkpoint=str(CHECKPOINT)).metrics
        torch.npu.synchronize()
        assert trainer.state.global_step == 3
        assert result.get("fsdp2_first_step") == 2
        assert math.isfinite(metrics["train_loss"]), metrics
        result.update(resumed_metrics=metrics, steps=3)
        if rank == 0:
            save("fsdp2_resume_training", True)

        trainer.save_model(str(FINAL_ADAPTER))
        torch.distributed.barrier()
        if rank == 0:
            tokenizer.save_pretrained(str(FINAL_ADAPTER))
            assert (FINAL_ADAPTER / "adapter_config.json").is_file()
            adapter_weights = list(FINAL_ADAPTER.glob("adapter_model.*"))
            assert adapter_weights, "No final adapter weights were saved"
            result.update(
                final_adapter=str(FINAL_ADAPTER),
                final_adapter_weights=[path.name for path in adapter_weights],
                status="passed",
            )
            save("complete")
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(result["stage"])
        traceback.print_exc()
        raise
    finally:
        if dist is not None and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
