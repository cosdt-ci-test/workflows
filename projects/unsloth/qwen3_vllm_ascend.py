"""Compare native vLLM-Ascend with Unsloth fast generation on one NPU."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path

OUTPUT = Path("output")
RESULT = OUTPUT / "result.json"
MODEL = "Qwen/Qwen3-0.6B"
PROMPTS = ["The capital of France is", "Two plus two equals"]


def run(phase, result):
    def record(stage, completed=False):
        result["stage"] = stage
        if completed:
            result["completed"].append(stage)
        RESULT.write_text(json.dumps(result, indent=2))
        print(json.dumps({"stage": stage, "completed": completed}), flush=True)

    try:
        record(f"{phase}_import")
        import torch
        import torch_npu  # noqa: F401 - registers the NPU backend
        if phase == "unsloth":
            from unsloth import FastLanguageModel
            from unsloth.device_type import DEVICE_TYPE
            assert DEVICE_TYPE == "npu", DEVICE_TYPE

        from vllm import LLM, SamplingParams
        from vllm.platforms import current_platform

        assert current_platform.device_type == "npu", current_platform
        assert torch.npu.is_available() and torch.npu.device_count() == 1
        torch.npu.set_device(0)
        result["device"] = torch.npu.get_device_name(0)
        result["versions"] = {name: version(name) for name in (
            "torch", "torch-npu", "transformers", "vllm", "vllm-ascend",
        )}
        record(f"{phase}_import", True)
        if phase == "native":
            record("model_download")
            from modelscope import snapshot_download
            result["model_path"] = snapshot_download(MODEL)
            record("model_download", True)

        model_path = result["model_path"]
        sampling = SamplingParams(temperature=0, max_tokens=8, logprobs=1)

        def generate(stage, generator, **kwargs):
            record(stage)
            outputs = generator(PROMPTS, sampling_params=sampling, use_tqdm=False, **kwargs)
            assert len(outputs) == len(PROMPTS), len(outputs)
            evidence = []
            for output in outputs:
                assert output.finished and len(output.outputs) == 1
                completion = output.outputs[0]
                assert completion.token_ids and completion.text.strip()
                request = kwargs.get("lora_request")
                if request is not None:
                    assert output.lora_request is not None
                    assert output.lora_request.lora_int_id == request.lora_int_id
                else:
                    assert output.lora_request is None
                first_id = completion.token_ids[0]
                first_logprob = completion.logprobs[0][first_id].logprob
                assert math.isfinite(first_logprob)
                evidence.append({
                    "text": completion.text,
                    "token_ids": list(completion.token_ids),
                    "first_token_logprob": first_logprob,
                })
            result[stage] = evidence
            record(stage, True)
            return evidence

        if phase == "native":
            record("native_load")
            engine = LLM(
                model=model_path, dtype="bfloat16", tensor_parallel_size=1,
                enforce_eager=True, max_model_len=256, max_num_seqs=2,
                gpu_memory_utilization=0.5,
            )
            record("native_load", True)
            generate("native_generate", engine.generate)
            record("unsloth_environment_setup")
            return

        record("unsloth_engine_load")
        model, _ = FastLanguageModel.from_pretrained(
            model_name=model_path, max_seq_length=256, dtype=torch.bfloat16,
            load_in_4bit=False, load_in_8bit=False, load_in_16bit=True,
            fast_inference=True, gpu_memory_utilization=0.5, max_lora_rank=8,
            unsloth_vllm_standby=False, use_gradient_checkpointing=False,
            enforce_eager=True, max_num_seqs=2, max_logprobs=1,
            enable_prefix_caching=False,
        )
        assert isinstance(getattr(model, "vllm_engine", None), LLM), "No vLLM engine"
        record("unsloth_engine_load", True)
        base = generate("unsloth_base_generate", model.fast_generate)

        record("unsloth_lora_setup")
        model = FastLanguageModel.get_peft_model(
            model, r=8, target_modules=["q_proj", "v_proj"], lora_alpha=16,
            lora_dropout=0, bias="none", use_gradient_checkpointing=False,
            random_state=3407,
        )
        # A fresh adapter is normally a no-op. Give it deterministic, nonzero
        # weights so we can detect whether the engine actually uses it.
        rng = torch.Generator(device="cpu").manual_seed(3407)
        with torch.no_grad():
            lora_b = [p for name, p in model.named_parameters() if "lora_B" in name]
            assert lora_b
            for parameter in lora_b:
                parameter.copy_(0.05 * torch.randn(parameter.shape, generator=rng, device="cpu"))
        adapter = OUTPUT / "vllm_probe_adapter"
        model.save_pretrained(str(adapter))
        assert (adapter / "adapter_model.safetensors").is_file()
        request = model.load_lora(str(adapter))
        assert request is not None
        result["lora_request_id"] = request.lora_int_id
        record("unsloth_lora_setup", True)

        adapted = generate("unsloth_lora_generate", model.fast_generate, lora_request=request)
        record("unsloth_lora_effect")
        result["lora_changes_output"] = any(
            a["token_ids"] != b["token_ids"]
            or abs(a["first_token_logprob"] - b["first_token_logprob"]) > 1e-3
            for a, b in zip(base, adapted)
        )
        assert result["lora_changes_output"], "No measurable effect from the LoRA adapter"
        record("unsloth_lora_effect", True)
        result["status"] = "passed"
        record("complete")
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        record(result["stage"])
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("native", "unsloth"), required=True)
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    if args.phase == "native":
        report = {"status": "running", "completed": [], "model": MODEL}
    else:
        report = json.loads(RESULT.read_text())
        assert "native_generate" in report["completed"]
    run(args.phase, report)
