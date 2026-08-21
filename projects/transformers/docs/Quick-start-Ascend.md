# Transformers Ascend Quick Start

This smoke test runs a public text-generation pipeline on an Ascend NPU. It does not require a Hugging Face token.

## Environment setup

The guard uses the Ascend CANN 9.1.0 image below on the `linux-aarch64-a2-2` runner:

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12
```

The image provides the CANN runtime and normally includes a compatible `torch` / `torch_npu` stack. Before running the example, the workflow:

1. Sources `/usr/local/Ascend/ascend-toolkit/set_env.sh` and checks `npu-smi`.
2. Reuses the image's `torch` and `torch_npu` when import and NPU device detection succeed; otherwise it installs `torch==2.9.0` and `torch_npu==2.9.0.post2` from the Ascend package index.
3. Installs the monitored transformers checkout with `python -m pip install -e target --no-deps`.
4. Installs `accelerate`, which supplies the device selected by `Accelerator().device`.

The runner exposes NPU device `0` as `ASCEND_RT_VISIBLE_DEVICES=0`. Package downloads use `https://repo.huaweicloud.com/ascend/repos/pypi`; the model is public, so no `HF_HUB_READ_TOKEN` or other secret is required. The first run may download `Qwen/Qwen2.5-1.5B` into the runner cache.

```pycon
>>> from accelerate import Accelerator
>>> from transformers import pipeline
>>>
>>> device = Accelerator().device
>>> pipe = pipeline(
...     "text-generation",
...     model="Qwen/Qwen2.5-1.5B",
...     device=device,
... )
>>> result = pipe("The secret to baking a good cake is ", max_new_tokens=16)
>>> print(result[0]["generated_text"])
...
```

The guard workflow runs this snippet after installing the target `huggingface/transformers` checkout. The runner exposes the NPU selected by the workflow and the test only requires the process to exit successfully and emit generated text.
