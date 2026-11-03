"""Two-rank HCCL/DDP training check for Unsloth on Ascend NPU."""
import json
import math
import os
from pathlib import Path
import traceback

OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)
RESULT_PATH = OUTPUT / "result.json"
ADAPTER = OUTPUT / "ddp_adapter"
rank = int(os.environ.get("RANK", "-1"))
local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
world_size = int(os.environ.get("WORLD_SIZE", "-1"))
result = {
    "status": "running", "stage": "ddp_setup", "completed": [],
    "scenario": "ddp-2", "model": "Qwen/Qwen3-0.6B",
    "world_size": world_size, "backend": "hccl",
}


def save(stage, completed=False):
    result["stage"] = stage
    if completed and stage not in result["completed"]:
        result["completed"].append(stage)
    if rank == 0:
        RESULT_PATH.write_text(json.dumps(result, indent=2))
        print(json.dumps({"rank": rank, "stage": stage,
                          "completed": completed}), flush=True)


def gathered_scalars(torch, device, values):
    import torch.distributed as dist
    local = torch.tensor(values, dtype=torch.float32, device=device)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return gathered


def parameter_checksum(torch, device, parameters):
    total = torch.zeros((), dtype=torch.float32, device=device)
    squares = torch.zeros((), dtype=torch.float32, device=device)
    for parameter in parameters:
        value = parameter.detach().float()
        total.add_(value.sum())
        squares.add_(value.square().sum())
    return total.item(), squares.item()


def main():
    global result
    torch = None
    dist = None
    try:
        save("scenario_setup", True)
        import torch
        import torch_npu
        import torch.distributed as dist
        from unsloth import FastLanguageModel
        from unsloth.device_type import DEVICE_TYPE
        from modelscope import snapshot_download
        from torch.nn.parallel import DistributedDataParallel

        assert rank in (0, 1) and local_rank in (0, 1) and world_size == 2
        torch.npu.set_device(local_rank)
        assert DEVICE_TYPE == "npu", DEVICE_TYPE
        assert torch.npu.is_available()
        assert torch.npu.is_bf16_supported()
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
        device = torch.device(f"npu:{local_rank}")
        result.update(
            rank=rank, local_rank=local_rank,
            rank_backend=dist.get_backend(),
            device=torch.npu.get_device_name(local_rank),
        )
        if rank == 0:
            save("hccl_initialized", True)

        model_path = snapshot_download(result["model"])
        result["model_path"] = model_path
        save("model_download", True)
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
        trainable = [parameter for parameter in model.parameters()
                     if parameter.requires_grad]
        assert trainable
        ddp_model = DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False,
        )
        ddp_model.train()
        optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.0)

        conversations = [
            "What is two plus two? Two plus two equals four.",
            "What is the capital of France? The capital of France is Paris.",
            "Name a primary color. Red is a primary color.",
            "What state is water at zero Celsius? Water freezes into ice.",
            "Which direction does the sun rise? The sun rises in the east.",
            "Which planet is largest? Jupiter is the largest planet.",
        ]
        rank_texts = conversations[rank::world_size]
        assert rank_texts
        initial = parameter_checksum(torch, device, trainable)
        initial_rows = gathered_scalars(torch, device, list(initial))
        assert torch.allclose(
            initial_rows[0], initial_rows[1], rtol=1e-5, atol=1e-6
        ), [row.tolist() for row in initial_rows]
        result["initial_parameter_checksum"] = initial_rows[0].tolist()
        if rank == 0:
            save("ddp_models_synchronized", True)

        step_metrics = []
        for step in range(3):
            text = rank_texts[step % len(rank_texts)]
            batch = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128,
            )
            batch = {name: value.to(device) for name, value in batch.items()}
            batch["labels"] = batch["input_ids"].clone()
            optimizer.zero_grad(set_to_none=True)
            loss = ddp_model(**batch).loss
            loss.backward()
            torch.npu.synchronize()
            optimizer.step()
            torch.npu.synchronize()

            checksum = parameter_checksum(torch, device, trainable)
            rows = gathered_scalars(
                torch, device, [loss.item(), *checksum],
            )
            assert torch.allclose(
                rows[0][1:], rows[1][1:], rtol=1e-5, atol=1e-6
            ), [row.tolist() for row in rows]
            step_metrics.append({
                "step": step + 1,
                "rank_local_losses": [row[0].item() for row in rows],
                "parameters_sum": rows[0][1].item(),
                "parameters_square_sum": rows[0][2].item(),
            })
            if rank == 0:
                print("DDP step", step + 1, step_metrics[-1], flush=True)

        assert all(math.isfinite(row["parameters_sum"]) for row in step_metrics)
        result.update(rank_step_metrics=step_metrics, steps=3)
        if rank == 0:
            save("ddp_training", True)

        dist.barrier()
        if rank == 0:
            ADAPTER.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ADAPTER))
            tokenizer.save_pretrained(str(ADAPTER))
            assert (ADAPTER / "adapter_config.json").is_file()
            assert (ADAPTER / "adapter_model.safetensors").is_file()
            result["adapter_directory"] = str(ADAPTER)
            save("rank0_adapter_saved", True)
        dist.barrier()
        if rank == 0:
            result["status"] = "passed"
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
