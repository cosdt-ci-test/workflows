# 快速开始：在昇腾 NPU 上跑通 FlagScale 的一次双卡离线推理

> **阅读本文前**，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。本文介绍如何在 AscendHub 的 CANN 镜像里装上与 CANN 9.1 匹配的 PyTorch NPU、vLLM 0.20.2、vllm-ascend、FlagGems 和 FlagScale 的 FL 插件，再用离线 `flagscale inference` 在两张卡上做一次短生成。

[FlagScale](https://github.com/flagos-ai/FlagScale) 是一套训练 / 推理 / 服务的编排工具。昇腾路径依赖 **vLLM + vllm-plugin-FL**。vLLM 选中插件 `fl` 之后，设备类型是 `npu`，通信后端是 HCCL。

---

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为**两张卡**（tensor parallel = 2）。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装并可 `source set_env.sh` |
| ATB | Ascend Transformer Boost（`nnal/atb`）。vLLM EngineCore 子进程要加载 `libatb.so` |
| Python | 3.12 |
| 编译依赖 | `cmake`、`ninja`；缺 `numaif.h` 时安装 `libnuma-dev` |
| PyTorch | `torch==2.10.0` 与 `torch-npu==2.10.0.post4`，见下文 |
| vLLM | 源码 `v0.20.2`（`VLLM_TARGET_DEVICE=empty`）+ 源码 `vllm-ascend` `v0.20.2rc1` |
| FlagScale | 上游 Release tag（撰写时 `v2.0.0`）+ FlagGems `v5.3.0` + `vllm-plugin-FL` `v0.2.1`（配对 vLLM 0.20.2） |
| 模型 | [Qwen/Qwen2.5-0.5B](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B)，首次运行会从 ModelScope 下载 |

**配套机器**：Atlas 900 A2 PODc（Ascend 910B4）。**配套镜像**：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

---

## 1. 加载 CANN 环境

在终端里 source 以下两个脚本，加载 CANN toolkit 和 ATB 算子库。后续命令都假设这个终端已经加载好环境。

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

---

## 2. 检查环境是否就绪

### 2.1 确认 NPU 在线

```shell
npu-smi info
```

**预期**：命令退出码为 0，并打印设备列表。设备列表里应能看到至少两张卡。表格中的功耗、HBM 占用每次不同，**不必**与任何样例逐字一致。

若 `npu-smi` 找不到，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载（如 `/dev/davinci0`、`/dev/davinci1`）。

### 2.2 确认工具可用

```shell #test-setup
test -n "$ASCEND_HOME_PATH"
command -v npu-smi
```

**预期**：`ASCEND_HOME_PATH` 非空；`npu-smi` 能找到。

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy="xxx"
Python 3.12.xxx
```

---

## 3. 安装 PyTorch NPU 栈

下面这份约束文件让 pip 跳过所有 NVIDIA / CUDA 轮子（`nvidia-*<0`、`cuda-*<0` 表示把这些包当作不存在）。装 vLLM 时也要复用这份文件。

```shell #test id="install-torch"
cat > constraints-npu-vllm.txt <<'EOF'
torch==2.10.0
torchvision==0.25.0
torchaudio==2.10.0
torch-npu==2.10.0.post4
triton-ascend==3.2.2
cuda-toolkit<0
cuda-python<0
cuda-bindings<0
cuda-core<0
cuda-pathfinder<0
flashinfer-python<0
nvidia-cublas<0
nvidia-cuda-runtime<0
nvidia-cuda-nvrtc<0
nvidia-cuda-cupti<0
nvidia-cudnn<0
nvidia-cudnn-frontend<0
nvidia-cufft<0
nvidia-curand<0
nvidia-cusolver<0
nvidia-cusparse<0
nvidia-cutlass-dsl<0
nvidia-cutlass-dsl-libs-base<0
nvidia-cutlass-dsl-libs-core<0
nvidia-cutlass-dsl-libs-cu12<0
nvidia-ml-py<0
nvidia-nccl<0
nvidia-nvjitlink<0
nvidia-nvtx<0
nvidia-cublas-cu12<0
nvidia-cuda-nvdisasm<0
nvidia-cuda-runtime-cu12<0
nvidia-cuda-nvrtc-cu12<0
nvidia-cuda-cupti-cu12<0
nvidia-cudnn-cu12<0
nvidia-cufft-cu12<0
nvidia-curand-cu12<0
nvidia-cusolver-cu12<0
nvidia-cusparse-cu12<0
nvidia-cusparselt-cu12<0
nvidia-nccl-cu12<0
nvidia-nvjitlink-cu12<0
nvidia-nvtx-cu12<0
cupy-cuda12x<0
cupy-cuda11x<0
cufile-python<0
nvtx<0
nixl<0
EOF
export PIP_CONSTRAINT="$PWD/constraints-npu-vllm.txt"
python -m pip install --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  torch==2.10.0 torch-npu==2.10.0.post4 torchvision==0.25.0 torchaudio==2.10.0 \
  numpy pyyaml packaging
python -m pip install --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  triton-ascend==3.2.2
python -c "import numpy, yaml, torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
```

输出结果如下：

```shell #test-result id="install-torch"
...
torch 2.10.0...
torch_npu 2.10.0.post4
npu_available True
```

`npu_available` 必须是 `True`，否则不要继续，先查 CANN、驱动和可见设备。`torch` 版本串可能带 `+cpu` 后缀，这是本地标签、不代表跑在 CPU 上，以 `npu_available True` 为准。

---

## 4. 安装 vLLM 0.20.2

按 `empty` 目标从源码安装 vLLM，昇腾算子交给下一节的 vllm-ascend。每个装包块都要重新 `export PIP_CONSTRAINT`，复用第 3 节的约束文件。

```shell #test id="install-vllm"
export PIP_CONSTRAINT="$PWD/constraints-npu-vllm.txt"
git config --global http.version HTTP/1.1
python -m pip install \
  "cmake>=3.26" pyyaml nanobind ninja setuptools-rust wheel \
  "setuptools-scm>=8" "setuptools>=77,<81" pybind11
git clone --depth 1 --branch v0.20.2 \
  https://github.com/vllm-project/vllm.git vllm-src
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation --no-deps -e ./vllm-src
python -m pip install \
  regex pydantic requests tqdm tokenizers sentencepiece tiktoken \
  protobuf msgspec cloudpickle blake3 einops fastapi transformers modelscope \
  aiohttp pyzmq prometheus_client py-cpuinfo diskcache==5.6.3 lark==1.2.2 \
  outlines_core==0.2.14 xgrammar 'llguidance>=1.3.0,<1.4.0' \
  'compressed-tensors==0.15.0.1' gguf python-json-logger setproctitle \
  watchfiles pybase64 'partial-json-parser' pillow psutil six ninja cbor2 ijson \
  'openai>=2.0.0' 'lm-format-enforcer==0.11.3' \
  'prometheus-fastapi-instrumentator>=7.0.0' depyf==0.20.0 \
  scipy pandas msgpack numba openai-harmony cachetools uvloop
python -c "import vllm; print('vllm', vllm.__version__)"
```

输出结果如下：

```shell #test-result id="install-vllm"
...
vllm 0.20.2...
```

版本串可能带 `+empty`，这是 empty 目标的本地标签，属预期现象。

---

## 5. 按 910B4 编译 vllm-ascend

从源码编译 vllm-ascend，`SOC_VERSION` 要和你的卡对应（本文是 `Ascend910B4`，换型号就改这一项）。`CMAKE_BUILD_PARALLEL_LEVEL` / `MAX_JOBS` 限制并行度，避免内存被打满。

```shell #test id="install-vllm-ascend"
export PIP_CONSTRAINT="$PWD/constraints-npu-vllm.txt"
export SOC_VERSION=Ascend910B4
export COMPILE_CUSTOM_KERNELS=1
export CMAKE_BUILD_PARALLEL_LEVEL=4
export MAX_JOBS=4
git clone --depth 1 --branch v0.20.2rc1 \
  https://github.com/vllm-project/vllm-ascend.git vllm-ascend-src
rm -rf vllm-ascend-src/build
SOC_VERSION=Ascend910B4 COMPILE_CUSTOM_KERNELS=1 \
  python -m pip install --no-build-isolation --no-deps -e ./vllm-ascend-src
python -c "from importlib.metadata import version; print('vllm_ascend', version('vllm-ascend'))"
```

输出结果如下：

```shell #test-result id="install-vllm-ascend"
...
vllm_ascend 0.20.2rc1
```

---

## 6. 安装 FlagGems 与 vllm-plugin-FL

[FlagGems](https://github.com/flagos-ai/FlagGems) 提供给 FL 插件调度的算子，从 `v5.3.0` 源码装。`--no-deps` 避免它把 `packaging` 钉死成别的版本，`--no-build-isolation` 复用当前环境里已装好的 cmake。

[vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL) 是 vLLM 的平台插件，注册名 `fl`。`main` 分支配对的是 vLLM 0.24，本文用 vLLM 0.20.2，所以从配套 tag `v0.2.1` 安装，不要跟 `main`。

```shell #test id="install-flag-stack"
export PIP_CONSTRAINT="$PWD/constraints-npu-vllm.txt"
python -m pip install sqlalchemy==2.0.48 scikit-build-core
git clone --depth 1 --branch v5.3.0 \
  https://github.com/flagos-ai/FlagGems.git FlagGems
python -m pip install --no-build-isolation --no-deps ./FlagGems
git clone --depth 1 --branch v0.2.1 \
  https://github.com/flagos-ai/vllm-plugin-FL.git vllm-plugin-FL
python -m pip install --no-build-isolation --no-deps ./vllm-plugin-FL
python -c "import flag_gems; from importlib.metadata import version; print('flag_gems', version('flag_gems')); print('vllm_fl', version('vllm-plugin-fl'))"
```

输出结果如下：

```shell #test-result id="install-flag-stack"
...
flag_gems 5.3.0...
vllm_fl 0.2.1...
```

导入 FlagGems 时可能看到 `get_device_capability returned None` 的警告。这是 `torch_npu` 的兼容性提示，不是安装失败。

---

## 7. 安装 FlagScale

将 `<UPSTREAM_REF>` 换成目标 **tag**（撰写时最新 Release 是 `v2.0.0`）。`--no-deps` 避免 FlagScale 的元数据去拉一份不匹配的 torch。CLI 还需要 `hydra-core`、`omegaconf` 和 `typer`，单独装上。`v2.0.0` 是 GitHub Release 的 tag 名；`pyproject.toml` 里的包装版本仍是 `1.0.0`，所以下面打印出来的是 `flagscale 1.0.0`。

克隆目录用 `FlagScale`，不要用当前目录下的 `flagscale/`，以免挡住已安装的包。
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="install-flagscale" load="upstream_ref>>UPSTREAM_REF"
export PIP_CONSTRAINT="$PWD/constraints-npu-vllm.txt"
git clone --depth 1 -b <UPSTREAM_REF> \
  https://github.com/flagos-ai/FlagScale.git FlagScale
python -m pip install --no-build-isolation --no-deps -e ./FlagScale
python -m pip install hydra-core omegaconf typer pyyaml packaging
python -c "import flagscale; from importlib.metadata import version; print('flagscale', version('flagscale'))"
```

输出结果如下：

```shell #test-result id="install-flagscale"
...
flagscale 1.0.0...
```

pip 可能会提示 FlagScale 声明的依赖还没装全。只要上面打出 `flagscale` 版本行，就可以继续。

---

## 8. 确认 FL 插件选中了 NPU

环境里同时有 `vllm-ascend`（插件名 `ascend`）和 `vllm-plugin-FL`（插件名 `fl`），vLLM 一次只允许激活一个平台插件。FlagScale 的昇腾推理走 **fl**，所以推理前要设：

```text
export VLLM_PLUGINS=fl
export VLLM_FL_PLATFORM=ascend
```

不设 `VLLM_PLUGINS` 时两个插件一起加载会直接报错，属预期行为。

```shell #test id="probe"
export VLLM_PLUGINS=fl
export VLLM_FL_PLATFORM=ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1
python - <<'PY'
import vllm
import vllm_fl
from vllm.platforms import current_platform as p
print("vllm", vllm.__version__)
print("platform_name", type(p).__name__)
print("device_type", p.device_type)
print("dist_backend", p.dist_backend)
PY
```

输出结果如下：

```shell #test-result id="probe"
...Platform plugin fl is activated...
platform_name PlatformFL
device_type npu
dist_backend hccl
...
```

`device_type` 必须是 `npu`，`dist_backend` 必须是 `hccl`，`platform_name` 必须是 `PlatformFL`。若这里是 `cuda` / `cpu`，或插件名不是 `fl`，先回到第 4–7 节，不要开始推理。

---

## 9. 用离线 inference 做一次双卡生成

本节用 Qwen2.5-0.5B 在两张卡上跑一次离线生成。下面的 yaml 已把规模收到几分钟内可结束（`gpu_memory_utilization` 0.4、`max_model_len` 512、`max_tokens` 8、单条 prompt），仅供首次验证，不是生产配置。

先下载模型，把标准输出里的绝对路径填进下面 yaml 的 `<MODEL_PATH>`。

<!--
```shell #test-setup
python - <<'PY'
from workflows.modelscope_cache import (
    ensure_safetensors,
    purge_corrupt_models,
    resolve_modelscope_cache,
)
ensure_safetensors()
purge_corrupt_models(resolve_modelscope_cache())
PY
```
-->

```shell #test-setup store="model_path"
set -o pipefail
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-0.5B'))" | grep '^/' | tail -n 1
```

下面这条命令在当前目录写好两份 yaml 后 `cd FlagScale` 再执行。`--test` 让推理在前台跑完才返回（不加会后台启动、立刻看不到结果）；末尾的 `2>&1` 把打在 stderr 的 `NPU compatibility enabled`、`backend=hccl` 并进标准输出。

```shell #test id="inference" load="model_path>>MODEL_PATH"
export PYTHONHASHSEED=0
export VLLM_PLUGINS=fl
export VLLM_FL_PLATFORM=ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1
export ASCEND_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p FlagScale/qs_conf/inference
cat > FlagScale/qs_conf/qwen25_05b_tp2_ascend.yaml <<'YAML'
defaults:
  - _self_
  - inference: qwen25_05b_tp2_ascend

experiment:
  exp_name: qwen25_05b
  exp_dir: qs_out
  task:
    type: inference
    backend: vllm
    entrypoint: flagscale/inference/inference_llm.py
  runner:
    hostfile: null
  envs:
    HYDRA_FULL_ERROR: 1
    ASCEND_VISIBLE_DEVICES: "0,1"
    ASCEND_RT_VISIBLE_DEVICES: "0,1"
    HCCL_WHITELIST_DISABLE: 1
    HCCL_CONNECT_TIMEOUT: 600
    PYTHONHASHSEED: 0
    VLLM_TARGET_DEVICE: "npu"
    VLLM_PLUGINS: "fl"
    VLLM_FL_PLATFORM: "ascend"
    VLLM_WORKER_MULTIPROC_METHOD: "spawn"

action: run

hydra:
  run:
    dir: ${experiment.exp_dir}/hydra
YAML
cat > FlagScale/qs_conf/inference/qwen25_05b_tp2_ascend.yaml <<'YAML'
llm:
  model: <MODEL_PATH>
  tokenizer: <MODEL_PATH>
  trust_remote_code: true
  tensor_parallel_size: 2
  pipeline_parallel_size: 1
  gpu_memory_utilization: 0.4
  seed: 1234
  enforce_eager: true
  max_model_len: 512
  max_num_batched_tokens: 512
  max_num_seqs: 1
  attention_backend: "TORCH_SDPA"
  disable_custom_all_reduce: true

generate:
  prompts: [
    "The first President of the United States",
  ]
  sampling:
    top_p: 0.1
    top_k: 1
    temperature: 0.0
    seed: 1234
    max_tokens: 8
YAML
cd FlagScale
flagscale inference qwen25_05b --config "$PWD/qs_conf/qwen25_05b_tp2_ascend.yaml" --test 2>&1
```

输出结果如下：

```shell #test-result id="inference"
...Platform plugin fl is activated...NPU compatibility enabled: torch.Event -> torch.npu.Event...backend=hccl...
output.outputs[0].text=...
```

生成的文字每次可能不同，不必和样例一致。第一次推理会编译/预热算子，可能要一两分钟。`Platform plugin fl is activated`、`NPU compatibility enabled`、`backend=hccl` 是这次走上昇腾 + 两卡 HCCL 的证据；退出码 0 不算数（运行脚本末尾的 `sync` 可能盖住中间失败），必须看到 `output.outputs[0].text=`。

---

## 故障排查

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| pip 去拉 `nvidia-*` 或 `cuda-*` | 没导出 `PIP_CONSTRAINT` | 确认约束文件还在，每个装包块都重新 `export` |
| pip 把 `torch` 升到 `2.11.0` | 装 vLLM 时没加 `--no-deps` | 卸掉后按第 3–4 节重装 |
| `torch.npu.is_available()` 为 `False` | 未 `source set_env.sh`，或设备未挂进容器 | 重做第 1–2 节 |
| `Only one platform plugin can be activated` | 同时装了 ascend 与 fl，却没设 `VLLM_PLUGINS=fl` | 按第 8 节导出这两个变量 |
| `device_type` 不是 `npu` | 插件没激活，或 `VLLM_FL_PLATFORM` 没设 | 重做第 6、8 节 |
| EngineCore 报找不到 `libatb.so` | 只 source 了 toolkit，没 source ATB | 按第 1 节再加上 ATB 的 `set_env.sh` |
| `Cannot re-initialize NPU in forked subprocess` | 没用 spawn | `export VLLM_WORKER_MULTIPROC_METHOD=spawn` |
| `ld.lld: unknown file type` | 上次编译的 `build/` 还在 | 删掉 `vllm-ascend-src/build` 再编 |
| pip 卡住去拉 `cmake`（pypi.org，CPU 接近 0） | 装 FlagGems 时没加 `--no-build-isolation` | 按第 6 节加上该旗标，复用上一节已经装好的 cmake |
| `Cannot import 'scikit_build_core.build'` | `--no-build-isolation` 时当前环境没有打包后端 | 按第 6 节先装 `scikit-build-core` |
| 编译把内存打满 / 被 cgroup 杀掉 | cmake 默认 `-j` 太大 | 确认设了 `CMAKE_BUILD_PARALLEL_LEVEL=4` |
| `git clone` 报 `curl 16` 或 `Error in the HTTP2 framing layer` | GitHub 的 HTTP/2 偶发失败 | 确认第 4 节写过 `git config --global http.version HTTP/1.1`，再重试一次 |
| `vllm_ascend` 没有 `__version__` | 当前目录下有一份叫 `vllm-ascend/` 的源码树挡住了已安装的包 | 按第 5 节克隆到 `vllm-ascend-src` |
| 去连 huggingface.co | 把 ModelScope 模型 id 直接传给了 `LLM` | 先 `snapshot_download`，把返回的本地目录写进 yaml |
| `flagscale inference` 立刻返回、没有生成 | 没加 `--test`，runner 在后台启动 | 加上 `--test` |
| 退出码 0 但没有 `output.outputs[0].text=` | 运行脚本末尾的 `sync` 把失败盖掉了 | 检查日志是否出现 `output.outputs[0].text=`，不要只看退出码 |
| `Undefined soc_version` 或预编译轮子装不上 | 轮子按另一套 SOC 名字打包 | 不要用预编译轮子，按第 5 节从源码编 |
| 只有一张卡可见 | 容器或环境没挂第二张卡 | 检查 `/dev/davinci1` 和 `ASCEND_RT_VISIBLE_DEVICES=0,1` |
