# 快速开始：在昇腾 NPU 上跑通 LMCache-Ascend 的一次离线 KV 缓存

> **阅读本文前**，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。本文聚焦**第一次跑通**：在 AscendHub 的 CANN 镜像里装上与 CANN 9.1 匹配的 vLLM-Ascend 栈，编译 [LMCache-Ascend](https://github.com/LMCache/LMCache-Ascend)，再用离线 `vllm.LLM` 走一遍 KV 卸载连接器。

主仓 [LMCache/LMCache](https://github.com/LMCache/LMCache) 没有昇腾实现。昇腾侧在同组织的 **LMCache-Ascend**。上游文档常用 `quay.io/ascend/vllm-ascend` 镜像。本文不走那条路：在已经装好 CANN 的机器上，按下面的命令自己装 PyTorch NPU、vLLM 和插件。

vLLM 从 PyPI 装时会去拉 NVIDIA 的 CUDA 轮子。aarch64 上那些轮子要么没有、要么会把环境弄乱。所以先写一份约束文件，把 `nvidia-*` / `cuda-*` 全部挡住，再装包。

---

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为**单卡**。编译 LMCache-Ascend 时把 `SOC_VERSION` 设成 `Ascend910B4`。若你的卡是别的 910B 型号，换成对应的 SOC 再编。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装并可 `source set_env.sh` |
| ATB | Ascend Transformer Boost（`nnal/atb`）。vLLM EngineCore 子进程要加载 `libatb.so` |
| Python | 3.12 |
| 编译依赖 | `cmake`、`ninja`、`libnuma` 头文件（缺 `numaif.h` 时安装 `libnuma-dev`） |
| PyTorch | `torch==2.10.0` 与 `torch-npu==2.10.0.post4`，见下文 |
| vLLM | 源码 `v0.23.0`（`VLLM_TARGET_DEVICE=empty`）+ `vllm-ascend==0.23.0` |
| LMCache | PyPI `lmcache`（版本号 = Release tag 去掉开头的 `v`；`NO_CUDA_EXT=1` 且 `--no-deps`），再编译 LMCache-Ascend |
| 模型 | [Qwen/Qwen2.5-0.5B](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B)，首次运行会从 ModelScope 下载 |

**配套机器**：Atlas 900 A2 PODc（Ascend 910B4）。**配套镜像**：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

---

## 1. 加载 CANN 环境

新开终端后 CANN 变量不会自动生效。常见容器里 `npu-smi` 在 `/usr/local/sbin`，需要把该目录加入 `PATH`。vLLM-Ascend 的 EngineCore 子进程还要加载 ATB（Ascend Transformer Boost）算子库。只 source toolkit 时，主进程能 `import torch_npu`，子进程会报找不到 `libatb.so`。

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

**预期**：命令退出码为 0，并打印设备列表。表格中的功耗、HBM 占用每次不同，**不必**与任何样例逐字一致。

若 `npu-smi` 找不到，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载（如 `/dev/davinci0`）。

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

下面这份约束只在本机当前目录生成一份文本文件，**不是**再装一套 CUDA。`nvidia-*<0` 和 `cuda-*<0` 的意思是：解析器如果想拉这些包，直接当作不存在。后面装 vLLM 时也要用同一份文件。

`torch_npu` 要从华为的 variant 索引安装，并钉死与 CANN 9.1 / vLLM-Ascend 0.23 匹配的版本。

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
transformers==5.5.4
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
  torch==2.10.0 torch-npu==2.10.0.post4 torchvision==0.25.0 torchaudio==2.10.0
python -c "import torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
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

## 4. 安装 vLLM 与 vllm-ascend

先装编译工具，再按 **empty** 目标从源码安装 vLLM。`VLLM_TARGET_DEVICE=empty` 让 vLLM 自己不编 CUDA/NPU 算子；昇腾算子由下一句的 `vllm-ascend==0.23.0` 提供。必须加 `--no-build-isolation`，否则 pip 会在隔离环境里另装一份 torch，把刚才的 NPU 栈换掉。

把源码克隆到 `vllm-src`，不要克隆成当前目录下的 `vllm/`。否则 `import vllm` 会走进这份源码树，而不是 pip 刚装好的包，既没有 `__version__`，也加载不到 `vllm-ascend` 插件。若 `vllm-src/pyproject.toml` 已经在（例如你刚拉过一次），就跳过 clone，免得 GitHub 超时把这一段打断。

每个命令块是一次新的 shell，约束文件还在，但环境变量不会保留，所以这里重新 `export PIP_CONSTRAINT`。

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
  git -c http.version=HTTP/1.1 clone --depth 1 --branch v0.23.0 https://github.com/vllm-project/vllm.git vllm-src
fi
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation -e ./vllm-src
python -m pip install --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  --no-build-isolation vllm-ascend==0.23.0
python -c "import vllm, torch, torch_npu; from vllm.platforms import current_platform; print('vllm', vllm.__version__); print('vllm_device', current_platform.device_type); print('npu_available', torch.npu.is_available())"
```

输出结果如下：

```shell #test-result id="install-vllm"
...
vllm 0.23.0...
vllm_device npu
npu_available True
```

`vllm_device` 必须是 `npu`。若这里是 `cuda` 或 `cpu`，先回到第 3 节，不要继续。

---

## 5. 安装 LMCache 并编译 LMCache-Ascend

PyPI 上的 `lmcache` 在 aarch64 没有预编译轮子，会现场编 sdist。元数据里写了 `cupy-cuda12x`、`nvtx`、`nixl` 这些 CUDA 包。不要让 pip 去解析它们，否则会下一份 CUDA 轮子。加上 `--no-deps` 和 `NO_CUDA_EXT=1`，只跳过这些 CUDA 依赖，不跳过 LMCache 自己的 Python 代码。`build-system` 还声明了 `torch`；上一节已经装好 NPU 版，必须加 `--no-build-isolation`，否则 pip 会在隔离环境里从 PyPI 再下一份。后面再把 `import lmcache` 真正需要的包单独装上。

然后再克隆 **LMCache-Ascend** 源码，按 910B4 编译 C++ 插件。`build-system` 声明了 `torch` 和 `torch-npu`，必须 `--no-build-isolation`。`--depth 1` 只拉当前 tag，第一次跑通不必要完整历史。`git -c http.version=HTTP/1.1` 是因为国内访问 GitHub 时 HTTP/2 可能在中途断开（报 `Error in the HTTP2 framing layer`）。主仓在 GitHub，子模块在 gitcode / atomgit；和主仓分开拉。若 `LMCache-Ascend/csrc/hixl/CMakeLists.txt` 已经在，就跳过克隆。不要 `cd` 进克隆目录，用 `-e ./LMCache-Ascend` 安装，和上一节的 `vllm-src` 一样；后面的离线脚本还要写在当前目录。编译前删掉目录里的 `build/`。CANN 把昇腾核目标链接进 `.o` 时会原地改文件，留着上次的产物再编，链接器会报 `unknown file type`。

CANN 9.1 会把 HIXL 通道编进去。HIXL 的 CMake 只加了 `pkg_inc/runtime` 这一层 include，而 `runtime/config.h` 会再 `#include "runtime/rt_external_device.h"`，头文件实际在 `pkg_inc/runtime/`。克隆后补上 `pkg_inc` 这一层，再编译。

vLLM 0.23 要求 KV connector 的构造函数第三个参数是 `kv_cache_config`。`lmcache` 主仓已经接了，LMCache-Ascend 当前 Release 的子类还是两个参数。克隆后改这一处。这只改你本地的克隆，不会向 GitHub 提交。若源码里已经出现 `kv_cache_config`，或 CMake 里已经有单独的 `pkg_inc` 行，脚本会跳过，不重复改。

将 `<UPSTREAM_REF>` 换成目标 **tag**（撰写时最新 Release 是 `v0.4.4`）。`<LMCACHE_VER>` 是同一个 tag 去掉开头的 `v`，给 `pip install` 用（pip 不接受 `lmcache==v0.4.4`）。
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
```shell #test-setup store="lmcache_ver"
echo "${UPSTREAM_REF#v}"
```
-->

```shell #test id="install-lmcache-ascend" load="upstream_ref>>UPSTREAM_REF" load="lmcache_ver>>LMCACHE_VER"
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set +u
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
export PIP_CONSTRAINT="$PWD/constraints-npu-vllm.txt"
NO_CUDA_EXT=1 python -m pip install --no-build-isolation lmcache==<LMCACHE_VER> --no-deps
python -m pip install \
  aiofile aiofiles blake3 aiohttp msgspec numpy psutil pyyaml pyzmq \
  redis safetensors sortedcontainers transformers modelscope
if [ ! -f LMCache-Ascend/csrc/hixl/CMakeLists.txt ]; then
  git -c http.version=HTTP/1.1 clone --depth 1 -b <UPSTREAM_REF> https://github.com/LMCache/LMCache-Ascend.git
  git -C LMCache-Ascend -c http.version=HTTP/1.1 submodule update --init --recursive
fi
python - <<'PY'
import re
from pathlib import Path

cmake = Path("LMCache-Ascend/csrc/hixl/CMakeLists.txt")
text = cmake.read_text()
runtime = "${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/pkg_inc/runtime"
pkg = "${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/pkg_inc"
if not re.search(r"/pkg_inc\s*$", text, re.M):
    if runtime not in text:
        raise SystemExit("hixl cmake include line not found")
    cmake.write_text(text.replace(runtime, runtime + "\n    " + pkg, 1))

conn = Path(
    "LMCache-Ascend/lmcache_ascend/integration/vllm/"
    "lmcache_ascend_connector_v1.py"
)
text = conn.read_text()
if "kv_cache_config" not in text:
    old_init = (
        "class LMCacheAscendConnectorV1Dynamic(LMCacheConnectorV1Dynamic):\n"
        "    def __init__(self, vllm_config: \"VllmConfig\", role: KVConnectorRole) -> None:\n"
        "        super().__init__(vllm_config=vllm_config, role=role)\n"
    )
    new_init = (
        "class LMCacheAscendConnectorV1Dynamic(LMCacheConnectorV1Dynamic):\n"
        "    def __init__(\n"
        "        self,\n"
        "        vllm_config: \"VllmConfig\",\n"
        "        role: KVConnectorRole,\n"
        "        kv_cache_config=None,\n"
        "    ) -> None:\n"
        "        super().__init__(\n"
        "            vllm_config=vllm_config,\n"
        "            role=role,\n"
        "            kv_cache_config=kv_cache_config,\n"
        "        )\n"
    )
    if old_init not in text:
        raise SystemExit("connector __init__ not found")
    conn.write_text(text.replace(old_init, new_init, 1))
print("patched_ok", True)
PY
rm -rf LMCache-Ascend/build
SOC_VERSION=Ascend910B4 python -m pip install -v --no-build-isolation -e ./LMCache-Ascend
python -c "import lmcache, lmcache_ascend, torch, torch_npu; from lmcache_ascend import _build_info as b; import lmcache_ascend.c_ops; print('lmcache', lmcache.__version__); print('soc', b.__soc_version__); print('c_ops_ok', True); print('npu_available', torch.npu.is_available())"
```

输出结果如下：

```shell #test-result id="install-lmcache-ascend" load="lmcache_ver>>LMCACHE_VER"
...
patched_ok True
...
lmcache <LMCACHE_VER>
soc Ascend910B4
c_ops_ok True
npu_available True
```

`import lmcache_ascend` 会加载编译出来的 `c_ops`，并读到本机 SOC。`soc` 必须是你编译时设的 `Ascend910B4`。缺 `numaif.h` 时先装 `libnuma-dev` 再编。

---

## 6. 用离线 LLM 做一次 KV 卸载

不要在第一次跑通时开 `vllm serve`。服务进程不会自己退出，也不方便确认这一次推理是否上了 NPU。

上游 `examples/offload.py` 写死了 `meta-llama/Llama-3.1-8B-Instruct`，上下文 8000，单卡 32 GB 很紧，而且还要从 Hugging Face 拉 Llama。下面这段用**同一套连接器**，换成 ModelScope 上的 Qwen2.5-0.5B，并把 `max_model_len` 收到 512、`gpu_memory_utilization` 收到 0.4。这是为了在一张 910B4 上几分钟内结束，不是生产配置。

不要用 `python - <<'PY'` 把脚本贴进标准输入。vLLM 的 EngineCore 要用 **spawn** 起子进程；spawn 会按文件路径重新导入主模块，标准输入没有路径，会报找不到 `<stdin>`。把脚本存成文件，并写成 `if __name__ == "__main__"`。也不要用 `LLM(**asdict(EngineArgs(...)))`：vLLM 0.23 会把未填的 compilation 字段当成 `None`，pydantic 会拒。

`VLLM_WORKER_MULTIPROC_METHOD=spawn` 避免父进程已经初始化过 NPU 之后，子进程再 `fork` 报 `Cannot re-initialize NPU`。`PYTHONHASHSEED=0` 让 LMCache 跨进程的 token hash 稳定。脚本里必须先 `import vllm`，再 `import lmcache_ascend`：后者只在检测到 vLLM 已经加载时，才会把设备检测换成 NPU 版。`2>&1` 把 vLLM / LMCache 打到 stderr 的日志并进标准输出，方便你在同一屏看到设备信息。

**怎样算成功**

1. 进程退出码为 0；
2. 日志里出现 `Platform plugin ascend is activated`（vLLM 选中昇腾插件）；
3. 日志里出现 `NPU device is available. Using NPU for LMCache engine.`（连接器把 LMCache 引擎放到 NPU 上）；
4. 打印 `current_device npu:0`。只看到上一节的编译成功、这次 `LLM()` 却没上卡，仍算失败。

```shell #test id="offload"
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set +u
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cat > offload_qs.py <<'PY'
import os

from modelscope import snapshot_download
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import lmcache_ascend
import torch
import torch_npu
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


def main():
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "2"

    model = snapshot_download("Qwen/Qwen2.5-0.5B")
    ktc = KVTransferConfig(
        kv_connector="LMCacheAscendConnectorV1Dynamic",
        kv_role="kv_both",
        kv_connector_module_path=(
            "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
        ),
    )
    llm = LLM(
        model=model,
        enforce_eager=True,
        kv_transfer_config=ktc,
        max_model_len=512,
        gpu_memory_utilization=0.4,
        trust_remote_code=True,
    )
    params = SamplingParams(temperature=0, top_p=0.95, max_tokens=8)
    llm.generate(["Hello, my name is", "Tell me a short story"], params)
    print("current_device", f"npu:{torch.npu.current_device()}")
    print("npu_available", torch.npu.is_available())
    print("workload_ok", True)
    LMCacheEngineBuilder.destroy(ENGINE_NAME)


if __name__ == "__main__":
    main()
PY
python offload_qs.py 2>&1
```

输出结果如下：

```shell #test-result id="offload"
...Platform plugin ascend is activated...Using NPU for LMCache engine...
current_device npu:0
npu_available True
workload_ok True
...
```

生成的具体文字每次可能不同，不必和任何样例一致。`current_device npu:0` 和「Using NPU for LMCache engine」才是这一次上了昇腾、并且 LMCache 引擎也在 NPU 上的证据。结束时 ZMQ 可能打一条 `Assertion failed: pfd.revents`，只要退出码是 0、上面几行都在，可以忽略。

---

## 7. 本文没有覆盖的能力

这些路径不在第一次跑通范围内，正文里也没有对应的可复制命令块：

- `vllm serve` 在线服务（进程不退出）
- 上游 `examples/offload.py` 原文里的 Llama-3.1-8B 与 8000 上下文
- 磁盘后端（`--use-disk`）
- `quay.io/ascend/vllm-ascend` 镜像里的预装栈
- 多卡 tensor parallel
- SGLang 后端
- 设备间 HIXL P2P 传输（本文只做单卡把 KV 卸到本机 CPU）

---

## 故障排查

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| pip 去拉 `nvidia-*` 或 `cuda-*` | 没导出 `PIP_CONSTRAINT` | 确认约束文件还在，每个装包块都重新 `export` |
| `torch.npu.is_available()` 为 `False` | 未 `source set_env.sh`，或设备未挂进容器 | 重做第 1–2 节 |
| 编译报缺 `numaif.h` | 没装 NUMA 头文件 | `apt-get install -y libnuma-dev` 后重编 |
| `ld.lld: unknown file type` | 上次编译的 `build/` 还在，核目标被原地改过 | 删掉 `LMCache-Ascend/build` 再编 |
| 编译报缺 `runtime/rt_external_device.h` | 没给 HIXL 的 CMake 补 `pkg_inc` | 确认第 5 节的 CMake 修补已经执行 |
| pip 去拉 `cupy-cuda12x` | 装 `lmcache` 时没加 `--no-deps` | 卸掉后按第 5 节用 `--no-deps` 重装 |
| pip 卡住去拉 `torch==2.10.0`（pypi.org，CPU 接近 0） | 装 `lmcache` 时没加 `--no-build-isolation`，隔离环境会从 PyPI 再下一份 | 按第 5 节加上该旗标，复用上一节已经装好的 NPU 版 torch |
| `git clone` 报 `curl 16` 或 `Error in the HTTP2 framing layer` | GitHub 的 HTTP/2 偶发失败 | 确认用了第 4、5 节的 `git -c http.version=HTTP/1.1`，再重试一次 |
| `deprecated 2-argument constructor` | 没改连接器的第三个参数 | 确认第 5 节的 connector 修补已经执行 |
| `Unsupported device platform for LMCache engine` | 先导入了 `lmcache_ascend`，NPU 版设备检测补丁没打上 | 先 `import vllm`，再 `import lmcache_ascend` |
| `vllm` 没有 `__version__`，或 `vllm_device` 不是 `npu` | 当前目录下有一份叫 `vllm/` 的源码树挡住了已安装的包 | 按第 4 节克隆到 `vllm-src`，不要克隆成 `./vllm` |
| 去连 huggingface.co 或报 Invalid repository ID | 把 ModelScope 模型 id 直接传给了 `LLM` | 先 `snapshot_download`，把返回的本地目录传给 `LLM(model=...)` |
| `LLM(**asdict(EngineArgs(...)))` 报 `CompilationConfig` 校验失败 | 上游 `examples/offload.py` 的写法会把 `None` 传进 vLLM 0.23 的配置 | 像第 6 节那样直接给 `LLM(...)` 传关键字参数 |
| EngineCore 报找不到 `libatb.so` | 只 source 了 toolkit，没 source ATB | 按第 1 节再加上 `source /usr/local/Ascend/nnal/atb/set_env.sh` |
| `Cannot re-initialize NPU in forked subprocess` | 没用 spawn，或父进程已经初始化过 NPU 又 fork | `export VLLM_WORKER_MULTIPROC_METHOD=spawn` |
| spawn 报找不到 `<stdin>` | 脚本是 `python - <<'PY'` 贴进标准输入的 | 存成 `.py` 文件再跑，并写 `if __name__ == "__main__"` |
| `An attempt has been made to start a new process` | spawn 重导入时顶层又执行了 `LLM()` | 把启动逻辑放进 `main()`，用 `if __name__ == "__main__"` 调用 |
