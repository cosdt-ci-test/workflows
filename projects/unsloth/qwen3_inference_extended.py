"""Exercise mixed-length batch generation and stopping on Ascend NPU."""
import json
from pathlib import Path
import traceback

OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)
RESULT_PATH = OUTPUT / "result.json"
result = {
    "status": "running", "stage": "inference_setup", "completed": [],
    "scenario": "inference-extended", "model": "Qwen/Qwen3-0.6B",
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
    from modelscope import snapshot_download
    from transformers import StoppingCriteria

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
        model_name=model_path, max_seq_length=512, dtype=torch.bfloat16,
        load_in_4bit=False, load_in_8bit=False, load_in_16bit=True,
        full_finetuning=False, fast_inference=False,
        device_map={"": "npu:0"}, use_gradient_checkpointing=False,
    )
    assert all(parameter.device.type == "npu" for parameter in model.parameters())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    FastLanguageModel.for_inference(model)
    save("model_and_tokenizer_setup", True)

    short_prompt = "What is two plus two?"
    medium_prompt = "Explain in one sentence why the sky appears blue."
    repeated = (
        "A language model predicts the next token from the preceding context. " * 35
    )
    long_prompt = repeated + "Continue with one short sentence about that process."
    prompts = [short_prompt, medium_prompt, long_prompt]
    texts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    ) for prompt in prompts]

    save("mixed_length_batch")
    batch = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=512,
    ).to("npu")
    prompt_lengths = batch.attention_mask.sum(dim=1).tolist()
    assert batch.input_ids.shape[0] == 3
    assert len(set(prompt_lengths)) == 3, prompt_lengths
    assert max(prompt_lengths) > min(prompt_lengths) + 20, prompt_lengths
    assert batch.input_ids.shape[1] <= 512
    with torch.inference_mode():
        outputs = model.generate(
            **batch, max_new_tokens=24, do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    torch.npu.synchronize()
    generated = outputs[:, batch.input_ids.shape[1]:]
    decoded = [tokenizer.decode(row, skip_special_tokens=True) for row in generated]
    assert all(text.strip() for text in decoded), decoded
    eos_generated = [bool((row == tokenizer.eos_token_id).any()) for row in generated]
    result.update(
        batch_size=3, prompt_token_lengths=prompt_lengths,
        batch_generation=decoded, batch_eos_generated=eos_generated,
    )
    print("Batch generations:", decoded)
    save("mixed_length_batch", True)

    class StopAfterNewTokens(StoppingCriteria):
        def __init__(self, prompt_length, limit):
            self.prompt_length = prompt_length
            self.limit = limit

        def __call__(self, input_ids, scores, **kwargs):
            reached = input_ids.shape[-1] >= self.prompt_length + self.limit
            return torch.full(
                (input_ids.shape[0],), reached, dtype=torch.bool,
                device=input_ids.device,
            )

    save("custom_stopping")
    single = tokenizer(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": short_prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        ), return_tensors="pt",
    ).to("npu")
    prompt_length = single.input_ids.shape[1]
    with torch.inference_mode():
        stopped = model.generate(
            **single, max_new_tokens=16, do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=[StopAfterNewTokens(prompt_length, 3)],
        )
    torch.npu.synchronize()
    stopped_tokens = stopped[0, prompt_length:]
    assert 0 < stopped_tokens.numel() <= 3, stopped_tokens.numel()
    result.update(
        custom_stopping_limit=3,
        custom_stopping_generated_tokens=int(stopped_tokens.numel()),
        custom_stopping_generation=tokenizer.decode(
            stopped_tokens, skip_special_tokens=True,
        ),
    )
    print("Stopped generation:", result["custom_stopping_generation"])
    save("custom_stopping", True)
    result["status"] = "passed"
    save("complete")
except Exception as exc:
    result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    save(result["stage"])
    traceback.print_exc()
    raise
