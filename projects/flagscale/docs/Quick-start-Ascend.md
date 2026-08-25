# 快速开始：在昇腾 NPU 上跑通 FlagScale 的一次双卡离线推理

> **阅读本文前**，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。本文聚焦**第一次跑通**。在 AscendHub 的 CANN 镜像里装上与 CANN 9.1 匹配的 PyTorch NPU、vLLM 0.20.2、vllm-ascend、FlagGems 和 FlagScale 的 FL 插件，再用离线 `flagscale inference` 在两张卡上做一次短生成。

[FlagScale](https://github.com/flagos-ai/FlagScale) 是一套训练 / 推理 / 服务的编排工具。昇腾路径依赖 **vLLM + vllm-plugin-FL**。vLLM 选中插件 `fl` 之后，设备类型是 `npu`，通信后端是 HCCL。上游文档有时会用 FlagScale 自己的容器镜像。那份镜像里的自定义算子是按另一代芯片编的，**910B4 上对不上**。本文不走那条路，而是在已经装好 CANN 的机器上，按下面的命令自己装栈、自己按 910B4 重编算子。

也不要用 `vllm serve` 做第一次验证。服务进程不会自己退出。离线 `flagscale inference` 会调用 `LLM(...)` 和 `llm.generate(...)`，生成结束后进程退出。

vLLM 从 PyPI 装时会去拉 NVIDIA 的 CUDA 轮子，而且 vLLM 0.20.2 的元数据把 `torch` 钉在 `2.11.0`。CANN 9.1 上能配对的是 `torch==2.10.0` 与 `torch-npu==2.10.0.post4`。所以先写一份约束文件挡住 CUDA 轮子，再按 **empty** 目标、带 `--no-deps` 从源码安装 vLLM。`--no-deps` 只跳过依赖解析，不跳过 vLLM 自己的代码。后面会把 `import vllm.LLM` 真正要用的包单独装上。

---

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为**两张卡**（tensor parallel = 2）。编译 vllm-ascend 时把 `SOC_VERSION` 设成 `Ascend910B4`。若你的卡是别的 910B 型号，换成对应的 SOC 再编。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装并可 `source set_env.sh` |
| ATB | Ascend Transformer Boost（`nnal/atb`）。vLLM EngineCore 子进程要加载 `libatb.so` |
| Python | 3.12 |
| 编译依赖 | `cmake`、`ninja`；缺 `numaif.h` 时安装 `libnuma-dev` |
| PyTorch | `torch==2.10.0` 与 `torch-npu==2.10.0.post4`，见下文 |
| vLLM | 源码 `v0.20.2`（`VLLM_TARGET_DEVICE=empty`）+ 源码 `vllm-ascend` `v0.20.2rc1` |
| FlagScale | 上游 Release tag（撰写时 `v2.0.0`）+ FlagGems `v5.3.0` + `vllm-plugin-FL` commit `53adefb26`（配对 vLLM 0.20.2） |
| 模型 | [Qwen/Qwen2.5-0.5B](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B)，首次运行会从 ModelScope 下载 |

**配套机器**：Atlas 900 A2 PODc（Ascend 910B4）。**配套镜像**：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

上游功能测试用的是 Qwen3-4B。第一次跑通用 0.5B 足够证明两张卡、HCCL 和 FL 插件都工作。把 yaml 里的模型路径换成 4B 即可放大，不必改其它步骤。

---

## 1. 加载 CANN 环境

新开终端后 CANN 变量不会自动生效。常见容器里 `npu-smi` 在 `/usr/local/sbin`，需要把该目录加入 `PATH`。vLLM 的 EngineCore 子进程还要加载 ATB（Ascend Transformer Boost）算子库。只 source toolkit 时，主进程能 `import torch_npu`，子进程会报找不到 `libatb.so`。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set +u
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
```

`set +u` 是因为 ATB 的 `set_env.sh` 会读未定义的 `ZSH_VERSION`。如果你的 shell 开了 `nounset`，不写这一行会直接退出。`PYTHONNOUSERSITE=1` 让 Python 忽略用户目录里的包。本机如果曾经 `pip install --user` 过 CANN 相关包，不设这个变量时，pip 解析器可能被带偏。

---

## 2. 检查环境是否就绪

### 2.1 确认 NPU 在线

```shell
npu-smi info
```

**预期**：命令退出码为 0，并打印设备列表。你应该能看到至少两张卡。表格中的功耗、HBM 占用每次不同，**不必**与任何样例逐字一致。

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

## 3. 挡住 CUDA 轮子，安装 PyTorch NPU 栈

下面这份约束只在本机当前目录生成一份文本文件，**不是**再装一套 CUDA。`nvidia-*<0` 和 `cuda-*<0` 的意思是，解析器如果想拉这些包，直接当作不存在。后面装 vLLM 时也要用同一份文件。

`torch_npu` 要从华为的 variant 索引安装，并钉死与 CANN 9.1 匹配的版本。`numpy` 和 `pyyaml` 也要一起装。`torch_npu` 的 wheel **没有声明**这两项依赖，但 `import torch` 会自动加载 `torch_npu`，缺了会在你显式 `import torch_npu` 之前就失败。

`triton-ascend` 给后面的 FlagGems 用。版本用 `3.2.2`。元数据有时会写 `3.2.1`，CANN 9.1 上 `3.2.2` 能装上、也能跑。

```shell #test id="install-torch"
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set +u
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
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

`npu_available` 必须是 `True`。`False` 时不要继续，先查 CANN、驱动和可见设备。

`torch 2.10.0` 后面可能带 `+cpu`。这是该 wheel 的本地版本标签，不是说计算会落到 CPU。以 `npu_available True` 为准。

---

## 4. 安装 vLLM 0.20.2（empty，不编 CUDA 算子）

先装编译工具，再按 **empty** 目标从源码安装 vLLM。`VLLM_TARGET_DEVICE=empty` 让 vLLM 自己不编 CUDA/NPU 算子。昇腾算子由下一节的 vllm-ascend 提供。必须加 `--no-build-isolation`，否则 pip 会在隔离环境里另装一份 torch，把刚才的 NPU 栈换掉。必须加 `--no-deps`，否则 pip 会按元数据去满足 `torch==2.11.0`，把 2.10 的 NPU 栈换掉。

把源码克隆到 `vllm-src`，不要克隆成当前目录下的 `vllm/`。否则 `import vllm` 会走进这份源码树，而不是 pip 刚装好的包。若 `vllm-src/pyproject.toml` 已经在（例如你刚拉过一次），就跳过 clone，免得 GitHub 超时把这一段打断。

`git -c http.version=HTTP/1.1` 是因为国内访问 GitHub 时 HTTP/2 可能在中途断开（报 `Error in the HTTP2 framing layer`）。

`--no-deps` 之后，`from vllm import LLM` 还缺一批**导入期**依赖，不是可选项。下面一次性装上。每个命令块是一次新的 shell，约束文件还在，但环境变量不会保留，所以这里重新 `export PIP_CONSTRAINT`。

```shell #test id="install-vllm"
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set +u
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
export PIP_CONSTRAINT="$PWD/constraints-npu-vllm.txt"
python -m pip install \
  "cmake>=3.26" pyyaml nanobind ninja setuptools-rust wheel \
  "setuptools-scm>=8" "setuptools>=77,<81" pybind11
if [ ! -f vllm-src/pyproject.toml ]; then
  git -c http.version=HTTP/1.1 clone --depth 1 --branch v0.20.2 \
    https://github.com/vllm-project/vllm.git vllm-src
fi
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

版本串可能带 `+empty`。这是 empty 目标的本地标签，是预期现象。

这一步还**不要** `from vllm.platforms import current_platform`。后面会同时装上 `vllm-ascend` 和 `vllm-plugin-FL` 两个平台插件。不设置 `VLLM_PLUGINS=fl` 时，vLLM 会报 `Only one platform plugin can be activated`。平台探测放到第 8 节，那时环境变量已经设好。

---

## 5. 按 910B4 编译 vllm-ascend

vllm-ascend 0.20.2 没有稳定版，用 `v0.20.2rc1`。华为索引上的预编译轮子按 SOC 名字筛选，常见写法是 `ascend910b1` 这类，和本文用的 `Ascend910B4` 对不上。所以从源码编，并显式设置：

- `SOC_VERSION=Ascend910B4`：告诉构建系统这是 910B4
- `COMPILE_CUSTOM_KERNELS=1`：把自定义算子编进来
- `CMAKE_BUILD_PARALLEL_LEVEL=4` 和 `MAX_JOBS=4`：限制并行编译。机器 CPU 很多时，默认 `-j` 会把内存打满

把源码克隆到 `vllm-ascend-src`，不要克隆成当前目录下的 `vllm-ascend/`。否则 `import vllm_ascend` 会走进这份源码树，包没有 `__version__`。若 `vllm-ascend-src/setup.py` 已经在，就跳过 clone。

编译前删掉目录里的 `build/`。CANN 把昇腾核目标链接进 `.o` 时会原地改文件，留着上次的产物再编，链接器会报 `unknown file type`。

```shell #test id="install-vllm-ascend"
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set +u
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
export PIP_CONSTRAINT="$PWD/constraints-npu-vllm.txt"
export SOC_VERSION=Ascend910B4
export COMPILE_CUSTOM_KERNELS=1
export CMAKE_BUILD_PARALLEL_LEVEL=4
export MAX_JOBS=4
if [ ! -f vllm-ascend-src/setup.py ]; then
  git -c http.version=HTTP/1.1 -c advice.detachedHead=false clone --depth 1 --branch v0.20.2rc1 \
    https://github.com/vllm-project/vllm-ascend.git vllm-ascend-src
fi
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

[FlagGems](https://github.com/flagos-ai/FlagGems) 提供一批给 FL 插件调度的算子。从 `v5.3.0` 源码装，加 `--no-deps`，避免它把 `packaging` 钉死成另一个版本。必须加 `--no-build-isolation`。`build-system` 声明了 `cmake`，上一节已经装过；不设这个旗标时，pip 会在隔离环境里从 pypi.org 再下一份。导入 FlagGems 需要 `sqlalchemy`。

[vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL) 是 vLLM 的平台插件，注册名是 `fl`。必须 `--no-build-isolation --no-deps`。`main` 分支配对的是 vLLM 0.24，本文用 vLLM 0.20.2，所以钉死 commit `53adefb269571684d83a51e997d3ba9be5f88235`（当时 `release/0.2` 的 HEAD）。不要跟 `main`，也不要只写浮动分支名。

```shell #test id="install-flag-stack"
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set +u
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
export PIP_CONSTRAINT="$PWD/constraints-npu-vllm.txt"
python -m pip install sqlalchemy==2.0.48
if [ ! -f FlagGems/pyproject.toml ] && [ ! -f FlagGems/setup.py ]; then
  git -c http.version=HTTP/1.1 clone --depth 1 --branch v5.3.0 \
    https://github.com/flagos-ai/FlagGems.git FlagGems
fi
python -m pip install --no-build-isolation --no-deps ./FlagGems
PLUGIN_SHA=53adefb269571684d83a51e997d3ba9be5f88235
if [ "$(git -C vllm-plugin-FL rev-parse HEAD 2>/dev/null)" != "$PLUGIN_SHA" ]; then
  rm -rf vllm-plugin-FL
  git -c http.version=HTTP/1.1 clone --filter=blob:none --no-checkout \
    https://github.com/flagos-ai/vllm-plugin-FL.git vllm-plugin-FL
  git -C vllm-plugin-FL -c http.version=HTTP/1.1 fetch --depth 1 origin "$PLUGIN_SHA"
  git -C vllm-plugin-FL checkout --detach FETCH_HEAD
fi
python -m pip install --no-build-isolation --no-deps -e ./vllm-plugin-FL
python -c "import flag_gems; from importlib.metadata import version; print('flag_gems_ok', True); print('vllm_fl', version('vllm-plugin-fl'))"
echo "plugin_sha $(git -C vllm-plugin-FL rev-parse --short=9 HEAD)"
```

输出结果如下：

```shell #test-result id="install-flag-stack"
...
flag_gems_ok True
vllm_fl 0.0.0+g53adefb26...
plugin_sha 53adefb26
```

导入 FlagGems 时可能看到 `get_device_capability returned None` 的警告。这是 `torch_npu` 的兼容性提示，不是安装失败。

---

## 7. 安装 FlagScale

将 `<UPSTREAM_REF>` 换成目标 **tag**（撰写时最新 Release 是 `v2.0.0`）。`--no-deps` 避免 FlagScale 的元数据去拉一份不匹配的 torch。CLI 还需要 `hydra-core`、`omegaconf` 和 `typer`，单独装上。

克隆目录用 `FlagScale`，不要用当前目录下的 `flagscale/`，以免挡住已安装的包。
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="install-flagscale" load="upstream_ref>>UPSTREAM_REF"
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set +u
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
export PIP_CONSTRAINT="$PWD/constraints-npu-vllm.txt"
if [ ! -f FlagScale/pyproject.toml ]; then
  git -c http.version=HTTP/1.1 clone --depth 1 -b <UPSTREAM_REF> \
    https://github.com/flagos-ai/FlagScale.git FlagScale
fi
python -m pip install --no-build-isolation --no-deps -e ./FlagScale
python -m pip install hydra-core omegaconf typer pyyaml packaging
python -c "import flagscale; print('flagscale_ok', True)"
```

输出结果如下：

```shell #test-result id="install-flagscale"
...
flagscale_ok True
```

pip 可能会提示 FlagScale 声明的依赖还没装全。只要 `flagscale_ok True`，就可以继续。

---

## 8. 确认 FL 插件选中了 NPU

同时装了 `vllm-ascend`（插件名 `ascend`）和 `vllm-plugin-FL`（插件名 `fl`）。vLLM 一次只允许激活一个平台插件。FlagScale 的昇腾推理走的是 **fl**，所以必须：

```text
export VLLM_PLUGINS=fl
export VLLM_FL_PLATFORM=ascend
```

`VLLM_FL_PLATFORM=ascend` 告诉 FL 插件把算子派发到昇腾后端。不设 `VLLM_PLUGINS` 时，两个插件会一起加载，进程直接报错，这是预期行为，不是环境坏了。

```shell #test id="probe"
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set +u
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
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
print("probe_ok", p.device_type == "npu" and p.dist_backend == "hccl")
PY
```

输出结果如下：

```shell #test-result id="probe"
...Platform plugin fl is activated...
platform_name PlatformFL
device_type npu
dist_backend hccl
probe_ok True
...
```

`device_type` 必须是 `npu`，`dist_backend` 必须是 `hccl`，`platform_name` 必须是 `PlatformFL`。若这里是 `cuda` / `cpu`，或插件名不是 `fl`，先回到第 4–7 节，不要开始推理。

vLLM 0.20.2 默认把 `Platform plugin fl is activated` 打到 stdout，所以这一块不必 `2>&1`。

---

## 9. 用离线 inference 做一次双卡生成

上游功能测试的 yaml 把模型写死在 `/home/gitlab-runner/data/Qwen3-4B`，并把 `ASCEND_VISIBLE_DEVICES` 写成 16 张卡。第一次跑通用 Qwen2.5-0.5B、两张卡。下面在 `FlagScale/qs_conf/` 里放一份给这次作业用的配置（文件名和 `exp_name` 都按 0.5B / TP=2，不要沿用上游的 `qwen3` / `4b`）：

- 模型改成 ModelScope 的 Qwen2.5-0.5B 本地目录。`snapshot_download` 可能在路径之外再打一些行，所以用 `grep '^/' | tail -n 1` 只留下最后一条绝对路径，再写进 yaml
- 可见设备改成 `0,1`
- `gpu_memory_utilization` 收到 0.4，`max_model_len` 收到 512，`max_num_batched_tokens` 收到 512，`max_tokens` 收到 8，prompt 只留一条。这是为了在两张 910B4 上几分钟内结束，不是生产配置
- `VLLM_PLUGINS=fl` 与 `VLLM_FL_PLATFORM=ascend` 写进 yaml 的 `envs`，这样 FlagScale 拉起的子进程也能看到
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` 避免父进程已经初始化过 NPU 之后，子进程再 `fork` 报 `Cannot re-initialize NPU`
- `enforce_eager: true`、`attention_backend: TORCH_SDPA`、`disable_custom_all_reduce: true`：第一次跑通走 PyTorch 注意力，不编自定义 all-reduce
- `HCCL_*` 与 `HYDRA_FULL_ERROR`：两卡集合通信超时，以及 Hydra 出错时打完整栈

`--test` 让 FlagScale 在前台跑推理并等到结束。不加这个旗标时，推理 runner 默认后台启动，命令会立刻返回，你看不到生成结果。

必须在 **FlagScale 仓库根目录**执行 `flagscale inference`。编排脚本会 `cd` 到安装位置，并按相对路径调用 `flagscale/inference/inference_llm.py`。

`2>&1` 把 FlagScale / 子进程打到 stderr 的日志并进标准输出。推理块要匹配 `NPU compatibility enabled` 和 `backend=hccl`，这些不一定走 vLLM 那条 stdout handler。

**怎样算成功**

1. 进程退出码为 0；
2. 日志里出现 `Platform plugin fl is activated`（vLLM 选中 FL 插件）；
3. 日志里出现 `NPU compatibility enabled` 和 `backend=hccl`（这一次走了昇腾和 HCCL，不是 CPU）；
4. 打印 `output.outputs[0].text=`。只看到上一节的探测成功、这次却没有真正 `generate`，仍算失败。FlagScale 生成的运行脚本末尾是 `sync`，中间步骤失败时退出码仍可能是 0，所以必须看到这一行。

```shell #test id="inference"
set -e
set -o pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set +u
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export VLLM_PLUGINS=fl
export VLLM_FL_PLATFORM=ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1
export ASCEND_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
MODEL=$(python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-0.5B'))" | grep '^/' | tail -n 1)
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
cat > FlagScale/qs_conf/inference/qwen25_05b_tp2_ascend.yaml <<YAML
llm:
  model: ${MODEL}
  tokenizer: ${MODEL}
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

生成的具体文字每次可能不同，不必和任何样例一致。第一次推理会编译/预热算子，可能要一两分钟。`Platform plugin fl is activated`、`NPU compatibility enabled` 和 `backend=hccl` 才是这一次上了昇腾、并且走了两卡 HCCL 的证据。

---

## 10. 本文没有覆盖的能力

这些路径不在第一次跑通范围内，正文里也没有对应的可复制命令块：

- `vllm serve` 在线服务（进程不退出）
- 上游功能测试原文里的 Qwen3-4B 与 16 卡可见设备列表
- `flagscale train` / `flagscale serve` / 压缩 / RL
- `flagtree` 以及只在厂商索引上存在的带 local-version 后缀的包
- `harbor.baai.ac.cn` 里的 FlagScale 镜像
- `vllm-plugin-FL` 的 `main` 分支（配对 vLLM 0.24）
- 单卡推理（上游最小昇腾用例是 TP=2）

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
| 编译把内存打满 / 被 cgroup 杀掉 | cmake 默认 `-j` 太大 | 确认设了 `CMAKE_BUILD_PARALLEL_LEVEL=4` |
| `git clone` 报 `curl 16` 或 `Error in the HTTP2 framing layer` | GitHub 的 HTTP/2 偶发失败 | 确认用了 `git -c http.version=HTTP/1.1`，再重试一次 |
| `vllm_ascend` 没有 `__version__` | 当前目录下有一份叫 `vllm-ascend/` 的源码树挡住了已安装的包 | 按第 5 节克隆到 `vllm-ascend-src` |
| 去连 huggingface.co | 把 ModelScope 模型 id 直接传给了 `LLM` | 先 `snapshot_download`，把返回的本地目录写进 yaml |
| `flagscale inference` 立刻返回、没有生成 | 没加 `--test`，runner 在后台启动 | 加上 `--test` |
| 退出码 0 但没有 `output.outputs[0].text=` | 运行脚本末尾的 `sync` 把失败盖掉了 | 按第 9 节的成功标准检查日志，不要只看退出码 |
| `Undefined soc_version` 或预编译轮子装不上 | 轮子按另一套 SOC 名字打包 | 不要用预编译轮子，按第 5 节从源码编 |
| 只有一张卡可见 | 容器或环境没挂第二张卡 | 检查 `/dev/davinci1` 和 `ASCEND_RT_VISIBLE_DEVICES=0,1` |
