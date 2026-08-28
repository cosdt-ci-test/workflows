# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上用 vllm-ascend v0.23.0（配套 vLLM v0.23.0）跑 [Speculators](https://github.com/vllm-project/speculators) 的完整端到端链路：把第三方 speculative decoding draft 模型（DFlash）转换成标准 `speculators` 格式、用 vllm-ascend 抽训练数据、torchrun 训 draft 模型、最后 `vllm serve` 把训好的 draft 挂上做推理 smoke。一个示例覆盖上游 README 列出的全部 4 个核心场景（Standardized Format / Offline Data Gen / Draft Training / Seamless vLLM Integration）。

本文档沿用上游 `convert/entrypoints.py` 里 DFlash + Qwen3-8B 这一组合做端到端验证：`speculators convert` 把 `z-lab/Qwen3-8B-DFlash-b16` 转换为标准 `speculators` 格式——这一步也是上游 `examples/train/dflash_*` / `examples/evaluate/` 里所有训练与评估流程的前置。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch），按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）
- 可用的 `uv`（本文档安装步骤全部走 `uv pip install`）。如果机器上没有：`pip install uv` 或参照 [uv 官方安装指南](https://docs.astral.sh/uv/getting-started/installation/)

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 1）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0.post4 |
| transformers | 由 `speculators` 透传拉入（>=4.56.1,<5.15.0） |
| vllm | 0.23.0（[源码 build](#安装-vllm-ascend)：`VLLM_TARGET_DEVICE=empty` 跳过 CUDA kernel 编译，仅注册 `torch.ops.vllm` schema） |
| triton-ascend | 3.2.2（华为 ascend 源 + `--find-links` 拿 270 MB aarch64 wheel，DFlash proposer JIT 编译依赖） |
| triton | 3.5.0（由 triton-ascend==3.2.2 的 METADATA 钉死，源码 build 装的主线 triton 需先卸掉再重装） |
| vllm-ascend | 0.23.0（`--extra-index-url` 拉华为 ascend 源 + `.../variant` 子路径取 NPU variant wheel，详见下方「[安装 vllm-ascend](#安装-vllm-ascend)」小节） |
| modelscope | 1.37.0 |
| speculators | 最新 release 的源码/二进制 |
| draft 模型 | [z-lab/Qwen3-8B-DFlash-b16](https://www.modelscope.cn/models/z-lab/Qwen3-8B-DFlash-b16)（DFlash draft，~1 GB） |
| verifier | [Qwen/Qwen3-8B](https://www.modelscope.cn/models/Qwen/Qwen3-8B)（~16 GB） |

> Speculators 的训练与 vLLM-Ascend 部署链路（`examples/train/`、`vllm-ascend serve --speculative-config`）需要 vllm-ascend ≥ v0.23.0（对应 vLLM v0.23.0），低于此版本 `extract_hidden_states` 模式与 DFlash proposer 不可用；本文档用单卡 Atlas 900 A2 PODc（Ascend 910B4）做 smoke 验证，**不验证**多卡 DFlash 训练并行（vllm-ascend 的 spec_decode E2E 跑在 `four_card/` 路径）。

### 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

输出类似：

```
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 5     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 5                                                            |
+===========================+===============+====================================================+
```

> 如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。

检查 Python 版本：

```shell #test id="check-py"
python --version
```
输出结果如下：
```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

#### 安装 vllm-ascend

PyPI `vllm==0.23.0` 的 aarch64 wheel 是 **CUDA-only build**（`vllm/_C.abi3.so` 链接 `libcuda.so.1` / `libcudart.so.13`，`vllm/cumem_allocator.abi3.so` 是 CUDA memory allocator，`vllm_flash_attn/_vllm_fa3_C.abi3.so` 内含 `__cudaLaunchKernel`），NPU 上无法用，且其 METADATA 钉 `torch==2.11.0+cpu` 与前置的 `torch==2.10.0+cpu` 冲突。所以本节从源码 build vllm：`VLLM_TARGET_DEVICE=empty` 跳过 CUDA kernel 编译、只注册 `torch.ops.vllm` schema 占位，运行时由 vllm-ascend 通过 `vllm.platform_plugins` entry point 把 NPU fused op 注入 `torch.ops.vllm` namespace。

第一步先把 torch 栈装上（CANN bare 镜像不预装）：

```shell #test id="install-torch"
# torch-npu==2.10.0.post4 只在 /variant 子路径（PEP 708 variant index），不在
# 根 /repos/pypi 索引里；uv 默认只在第一个找到的索引上搜版本，需要
# unsafe-best-match 让它跨索引找
export UV_INDEX_STRATEGY=unsafe-best-match

# --system：CANN bare 镜像是 system Python（不是 venv），uv 默认拒绝非 venv；
# 后面所有 uv pip install / uninstall 也都用 --system
uv pip install --system \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  --find-links https://repo.huaweicloud.com/ascend/repos/pypi/triton-ascend/ \
  torch==2.10.0 torch-npu==2.10.0.post4 torchvision==0.25.0 torchaudio==2.10.0

python -c "import torch, torch_npu; print(f'torch={torch.__version__}'); print(f'torch_npu={torch_npu.__version__}'); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="install-torch"
torch=2.10.0+cpu
torch_npu=2.10.0.post4
is_available: True
count: 1
```

然后源码 build vllm + 装 vllm-ascend + triton-ascend：

```shell #test id="vllm-ascend-install"
# 1. vllm 源码 build 依赖（cmake / ninja / pybind11 / setuptools-scm）。
#    --system 同上一节，CANN bare 镜像是 system Python
uv pip install --system "cmake>=3.26" pyyaml nanobind ninja setuptools-rust wheel \
  "setuptools-scm>=8" "setuptools>=77,<81"

# 2. 加载 CANN env（vllm 源码编译时链接 libascendcl / libatb 需要；
#    torch_npu import 也依赖 libhccl.so / libascendcl.so，必须 source）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 3. clone vllm v0.23.0 源码到 /root/deps/vllm
mkdir -p /root/deps
git clone --depth 1 --branch v0.23.0 \
  https://github.com/vllm-project/vllm.git /root/deps/vllm

# 4. 源码 build：VLLM_TARGET_DEVICE=empty 跳过 CUDA kernel 编译
#    --no-deps：vllm 0.23.0 pyproject 钉 torch==2.11.0+cpu，与前置 torch==2.10.0+cpu 冲突
#    --no-build-isolation：复用环境已装的 torch==2.10.0+cpu，不重新拉 build-time deps
VLLM_TARGET_DEVICE=empty uv pip install --system --no-deps --no-build-isolation \
  -e /root/deps/vllm

# 5. 卸 vllm 装的主线 triton（vllm 拉的是 CUDA 优化版，NPU 上 DFlash JIT 跑不了）。
#    这里改用 python3 -m pip uninstall：uv 在 system Python 模式下需要 --system，
#    且 uv 在大 wheel + 多 transitive deps 下偶尔会卡在 unpack 阶段
python3 -m pip uninstall -y triton

# 6. 装 vllm-ascend==0.23.0 NPU variant wheel（/variant 子路径拿 aarch64 NPU build）。
#    改用 python3 -m pip：同上，uv 偶尔会卡在 unpack 阶段；pip install --no-deps
#    配合 --extra-index-url 拉 NPU variant wheel
python3 -m pip install --no-deps \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  vllm-ascend==0.23.0

# 7. 装 triton-ascend==3.2.2：DFlash proposer JIT 编译依赖
#    它的 METADATA 钉 triton==3.5.0（同时拉回主线 triton），装上即可
python3 -m pip install \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  --find-links https://repo.huaweicloud.com/ascend/repos/pypi/triton-ascend/ \
  triton-ascend==3.2.2

# 8. 装齐 vllm runtime deps（57 个包）。
#    VLLM_TARGET_DEVICE=empty 跳过了 install-time deps 解析（避免装 torch==2.11.0
#    与 torch==2.10.0 冲突），但运行时 `from vllm.config import ...` 会触发整条
#    import 链，需要 cbor2 / pyzmq / gguf / compressed-tensors / lm-format-enforcer
#    / xgrammar / outlines_core / opencv-python-headless 等。pip 装比 uv 稳，
#    --index-strategy unsafe-best-match 跨 cluster cache + 华为源找 NPU aarch64 wheel。
#    `numba` 不是 vllm 必需但 vllm-ascend 0.23.0 `eplb/core/policy/policy_flashlb.py`
#    顶层 `from numba import njit`（--no-deps 跳过了 transitive 安装，runtime 缺）。
python3 -m pip install --quiet \
  -r /root/deps/vllm/requirements/common.txt \
  numba

# 9. Monkey-patch `vllm.triton_utils.HAS_TRITON = True`（CI 上必做，本地 NPU 上做不做都行）。
#    Root cause：triton-ascend 3.2.2 自带的 libtriton.so 是 triton 3.2.0 fork，
#    只编了 ascend backend，没编 nvidia/amd symbol（`import triton._C.libtriton`
#    后 dir() 看到 ['ascend', 'buffer_ir', 'interpreter', 'ir', 'llvm', 'passes']，
#    没有 amd / nvidia）。vllm 0.23.0 `vllm/triton_utils/importing.py:17` 的
#    `HAS_TRITON = find_spec("triton") is not None` 后续 try 块里
#    `from triton.backends import backends` 会触发 triton 主线 wheel 的 import 链，
#    最终某个 submodule `from triton._C.libtriton import amd` —— ImportError 后
#    HAS_TRITON 被强制设回 False。
#    后果：`vllm_ascend/ops/__init__.py:19` 的 `if HAS_TRITON:` 跳过
#    `import vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_rope`，
#    `qkv_rmsnorm_rope` 这个 op 不会被注册到 `torch.ops.vllm` namespace。
#    Step 2 `vllm.LLM()` 跑 `QKNormRopeFusionPass`（head_dim=128 的 attention
#    layer 命中）时 fusion pattern 调 `torch.ops.vllm.qkv_rmsnorm_rope()` 就
#    AttributeError（CI run 33140922182）。
#    修法：写一个 sitecustomize.py 到 site-packages，让 Python 启动时（spawn 子
#    进程也会触发）强制把 HAS_TRITON 设回 True，import 链继续往下走、qkv_rmsnorm_rope
#    op 正常注册。CI container 是 throw-away 的，这个 patch 文件只在 container 内
#    有效，不会污染镜像。try/except 包一层是因为 sitecustomize.py 在每个 Python 启
#    动时都跑（包括 test runner 自身在 vllm 安装前），不能让它抛 ImportError 把整
#    个 Python 启动干崩 —— 此时 vllm 还没装，import 自然失败，except 把异常吞掉，
#    等 vllm 装好后下一次 sitecustomize.py 加载就生效。
cat > /usr/local/python3.12.13/lib/python3.12/site-packages/sitecustomize.py << 'PY'
try:
    import vllm.triton_utils
    vllm.triton_utils.HAS_TRITON = True
except Exception:
    pass
PY

# 验证 qkv_rmsnorm_rope op 注册成功
python -c "
import vllm.triton_utils
vllm.triton_utils.HAS_TRITON = True
import vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_rope
import torch
print('HAS_TRITON:', vllm.triton_utils.HAS_TRITON)
print('qkv_rmsnorm_rope op:', torch.ops.vllm.qkv_rmsnorm_rope)
"

python -c "import importlib.metadata; print(f'vllm={importlib.metadata.version(\"vllm\")}')"
python -c "import importlib.metadata; print(f'vllm_ascend={importlib.metadata.version(\"vllm-ascend\")}')"
python -c "import importlib.metadata; print(f'triton_ascend={importlib.metadata.version(\"triton-ascend\")}')"
python -c "import importlib.metadata; print(f'triton={importlib.metadata.version(\"triton\")}')"
```

输出结果如下：

```shell #test-result id="vllm-ascend-install" fuzzy='xxx'
xxx
HAS_TRITON: True
qkv_rmsnorm_rope op: vllm.qkv_rmsnorm_rope
vllm=0.23.0+empty
vllm_ascend=0.23.0
triton_ascend=3.2.2
triton=3.5.0
```

检查 NPU 设备运行时可用：

```shell #test id="check-npu-runtime"
python -c "import torch, torch_npu; print(f'torch={torch.__version__}'); print(f'torch_npu={torch_npu.__version__}'); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-npu-runtime"
torch=2.10.0+cpu
torch_npu=2.10.0.post4
is_available: True
count: 1
```

#### 安装 modelscope

```shell #test-setup
uv pip install 'modelscope==1.37.0'
```

打印安装版本：
```shell #test id="install-deps"
python -c "import modelscope; print(f'modelscope={modelscope.__version__}')"
```

输出结果如下：

```shell #test-result id="install-deps"
modelscope=1.37.0
```

## 安装 Speculators

### 使用 uv 进行安装

```shell #test id="speculators-install-binary"
uv pip install speculators
speculators --version
python -c "from importlib.metadata import version; print('speculators', version('speculators'))"
```

输出结果类似如下：

```shell #test-result id="speculators-install-binary" fuzzy='xxx'
speculators version: xxx
speculators xxx
```
- xxx 表示最新的版本号
<!--
```shell #test-setup
uv pip uninstall speculators
```
-->

### 从源码安装

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到工作流注入的最新 release tag，安装并且验证：

```shell #test id="speculators-install-source" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/vllm-project/speculators.git
cd speculators
uv pip install -e .
speculators --version
python -c "from importlib.metadata import version; print('speculators', version('speculators'))"
```

\<ref> 为安装的最新的 release tag

输出结果类似如下：

```shell #test-result id="speculators-install-source" fuzzy='xxx'
speculators version: xxx
speculators xxx
```
- xxx 表示最新的版本号

## 完整链路：convert → 训练数据生成 → 训练 → 部署（4 个核心场景端到端示例）

上游 README 列出的 4 个核心场景里，本文档用 vllm-ascend v0.23.0 + 单卡 A2 串行跑全：

| # | 核心场景 | 本节验证 | 依赖 |
| --- | --- | --- | --- |
| 1 | Offline Training Data Generation using vLLM-Ascend | ✅ Step 2 | vllm-ascend `extract_hidden_states` method（离线 `LLM()` API + `ExampleHiddenStatesConnector`） |
| 2 | Draft Model Training Support | ✅ Step 3 | 上游 `scripts/train.py` + 单卡 torchrun |
| 3 | **Standardized, Extensible Format**（HF 兼容 schema + 转换工具） | ✅ Step 1 | `convert_model(algorithm="dflash")` Python API |
| 4 | Seamless vLLM-Ascend Integration | ✅ Step 4 | `vllm-ascend serve --speculative-config '{"method":"dflash",...}'` + curl smoke |

4 步以 store/load 串成一个 pipeline，前一步产物是后一步输入。Step 2 走 vllm 离线 `LLM()` API（无需 HTTP server），Step 3 走 torchrun 离线训练（直接读 Step 2 落盘的 hidden states），Step 4 才是 vllm-ascend 在线 serve + 真实推理 smoke——单卡 A2 上把 vllm 与 train 串行化规避「两者不能同卡跑」的硬件约束。

上游 `examples/` 下分三类入口：`convert/`（格式转换，CPU/NPU 都可跑）、`train/`（在线/离线训练，需要 vllm-ascend NPU 后端）、`evaluate/`（基于 vllm-ascend 服务的人评测）。`convert/entrypoints.py:convert_model` 文档块给出的 DFlash 配方对应 Step 1；上游 `examples/train/dflash_qwen3_8b_sharegpt_online_5k.sh` 完整脚本（prepare_data → launch_vllm → torchrun train）对应 Step 1 → 2 → 3 → 4 的完整组合，本文在单卡约束下做了样本量（5k → 10）与并行度（4 卡 vllm 分离 → 1 卡串行）两处缩量。

### 前置：下载 draft 与 verifier

默认使用 **ModelScope** 进行模型下载（draft + verifier 都在 ModelScope 上有完整镜像）。持久缓存中可能残留之前中断下载产生的残缺权重文件，测试框架会在下载前做 safetensors 完整性校验，损坏的模型目录会被整体清除并重新下载。

```shell #test-setup store="draft_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('z-lab/Qwen3-8B-DFlash-b16'))" | tail -n 1
```

输出类似：

```
/root/.cache/modelscope/hub/models/z-lab/Qwen3-8B-DFlash-b16
```

```shell #test-setup store="verifier_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B'))" | tail -n 1
```

输出类似：

```
/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B
```

### Step 1 Standardized, Extensible Format（convert）

`speculators convert` 把本地 draft 目录 + verifier 目录读进来，按 DFlash 算法重映射权重、写入 `speculators_config`，输出到一个新目录。CLI 的 `--algorithm` 选项只接受 `eagle` / `eagle3` / `mtp` 三个值（见上游 `src/speculators/__main__.py:99` 的 `click.Choice(["eagle", "eagle3", "mtp"])`），DFlash 不在 CLI 白名单里——DFlash 只在 Python API `convert_model(algorithm="dflash", ...)` 里支持（见 `convert/entrypoints.py:32` 的 `Literal["eagle3", "mtp", "dflash"]`），所以本节走 Python API：

```shell #test-setup store="dflash_path" load="draft_path>>draft_path" load="verifier_path>>verifier_path"
python << 'PY'
from speculators.convert import convert_model

convert_model(
    model="<draft_path>",
    verifier="<verifier_path>",
    algorithm="dflash",
    output_path="/root/dflash-qwen3-8b-converted",
)
PY
test -f /root/dflash-qwen3-8b-converted/config.json
test -f /root/dflash-qwen3-8b-converted/model.safetensors
echo "/root/dflash-qwen3-8b-converted"
```

> 这里**不**再 `python | grep` 过滤输出 —— 那个设计有坑：loguru `logger.success("Saved to: ...")` 写在 stderr，grep 在管道的 stdout 端能匹配；可一旦 convert 抛异常（CI 33049460053 跑出 19.7s 的"快速成功"，但实际 config.json / model.safetensors 都没生成），traceback 走 stderr 也进管道、grep 找不到 "Saved to:" 退出 1，bash 没 `set -e / pipefail`，`echo` 还是照样执行、`dflash_path` 照样被捕获，framework 完全看不到失败。改成：让 python 的 stderr 自由流到框架端（异常 traceback 会触发 `ERROR_MARKERS` 命中，被 `_dump_command_output` 全文 dump），再用两个 `test -f` 显式断言输出文件存在 —— 任何一项失败 `test` 退出 1，整个 setup 块 rc 变 1，框架立即 raise 并把 stderr 一起 dump 出来。`echo` 之后单写一行纯路径，让 `store="dflash_path"` 拿到干净的字符串（避免 `rstrip` 后还带 loguru 的 success 行污染下游 `<dflash_path>` 替换）。下面的 `<dflash_path>` 是测试框架的占位符（`load="dflash_path>>dflash_path"`）：执行 `#test` 块前框架把 `<dflash_path>` 替换成捕获值，bash 看到的命令是路径字面量；不要写 `$dflash_path`，那样 shell 变量在每次 `#test` 都是空、且框架不会做 `$`-展开。

```shell #test id="pipeline-step1-convert" load="dflash_path>>dflash_path"
ls -1 <dflash_path>/config.json <dflash_path>/model.safetensors
echo <dflash_path>
```

输出结果如下：

```shell #test-result id="pipeline-step1-convert"
/root/dflash-qwen3-8b-converted/config.json
/root/dflash-qwen3-8b-converted/model.safetensors
/root/dflash-qwen3-8b-converted
```

### Step 2 — 场景 1：Offline Training Data Generation using vLLM-Ascend

用 vllm-ascend 的 `extract_hidden_states` 离线 API（`vllm.LLM()` + `kv_transfer_config` 配 `ExampleHiddenStatesConnector`，路径与 vllm-ascend `tests/e2e/pull_request/one_card/spec_decode/test_extract_hidden_states.py` 一致）让 verifier 在指定层的 forward pass 输出 hidden states，落到 `/root/dflash-train-data/`：

```shell #test-setup store="hidden_states_path" load="verifier_path>>verifier_path"
DATA_DIR=/root/dflash-train-data
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"

# 必须先写 .py 文件再 `python /tmp/...py` 跑 —— vllm v1 engine 在 NPU 上走
# multiprocessing spawn，spawn 子进程 `runpy.run_path(__main__)`，父进程的
# `__main__` 必须是真实文件路径。`python << 'PY'` 把 stdin 当 `__main__`，spawn
# 子进程找不到 `<stdin>` 文件直接 FileNotFoundError（CI 33055740901）。`fork`
# 在 NPU 上会丢 HAI driver context、segfault，所以 spawn 是唯一选项。
cat > /tmp/extract_hidden.py << 'PY'
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams

# 必须把 vllm 调用包进 `if __name__ == "__main__":` —— Python multiprocessing
# spawn 路径要求：spawn 子进程会重新 import __main__，父进程在 main 模块顶层
# 还处于 bootstrapping 阶段时就起新进程会被 `_check_not_importing_main()` 拒掉
# (CI 33061941772)。这是 Python multiprocessing 的标准 idiom，跟 vllm 无关；
# 但因为我们 spawn 是真实文件（不像 heredoc 的 __main__=<stdin>），这个 guard
# 才被实际执行。
if __name__ == "__main__":
    # Qwen3-8B 有 36 层，抽 [2, 18, 34] 三层（vllm-ascend test_extract_hidden_states
    # DENSE_AUX_HIDDEN_STATE_LAYER_IDS 同值）。注意 target_layer_ids 是
    # draft_model_config.hf_config.eagle_aux_hIDDEN 的字段，不是 CLI flag。
    llm = LLM(
        model="<verifier_path>",
        tensor_parallel_size=1,
        enable_chunked_prefill=False,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": [2, 18, 34],
                }
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": "/root/dflash-train-data",
            },
        },
    )

    prompts = [f"Briefly describe AI topic #{i}." for i in range(10)]
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=1))

    # 第一个输出的 hidden_states_path 即可代表整批（kv_connector 对每个 prompt 写一个 .safetensors）。
    # 写到固定文件而不是 print 出来 —— vllm.LLM() teardown 时往 stdout 写
    # `[ERROR] ... applicaiton exception`（CANN 驱动的 typo 字面量），`tail -1` 抓
    # 到的就是错误行而不是路径；写文件 + cat 让 capture 完全跟 vllm 输出解耦。
    with open("/tmp/last_hidden_path.txt", "w") as _f:
        _f.write(outputs[0].kv_transfer_params["hidden_states_path"])
PY

python /tmp/extract_hidden.py
cat /tmp/last_hidden_path.txt
```

> `extract_hidden_states` 是 vllm-ascend 的特殊 spec_decode mode：不真做 decoding、每个请求产出 1 token + 把 hidden states 写到 `shared_storage_path`。`outputs[0].kv_transfer_params["hidden_states_path"]` 是 vllm-ascend v0.23.0 引入的 safetensors 单文件格式（更早版本走 `ExampleHiddenStatesConnector.load_hidden_states`，文件路径不可见但 shape 一致）。**不靠 `tail -1` / 管道抓路径** —— vllm.LLM() 在 process 退出前会触发 CANN 驱动 teardown，teardown **同时**往 stdout 和 stderr 写一行 `[ERROR] ... applicaiton exception`（CANN 驱动的 typo 字面量），不管 `2>&1` 拼不拼、`tail -1` 抓到的都是错误行而不是 print 出来的路径（CI 33058568104 验证：去掉 `2>&1` 后 stdout 仍有 100B 的 `[ERROR] ...`）。改成 python 端把路径写到 `/tmp/last_hidden_path.txt`，bash 端 `cat` 那个固定文件 —— capture 完全跟 vllm 的 stdout / stderr 输出解耦，只看一个我们自己能控制内容的文件。

```shell #test id="pipeline-step2-extract" load="hidden_states_path>>hidden_states_path"
echo <hidden_states_path>
ls -1 /root/dflash-train-data/*.safetensors 2>/dev/null | wc -l
```

输出结果如下：

```shell #test-result id="pipeline-step2-extract" fuzzy='xxx'
/root/dflash-train-data/xxx.safetensors
xxx
```

### Step 3 — 场景 2：Draft Model Training Support

用上游 `scripts/train.py` + 单卡 `torchrun --nproc_per_node=1` 训 1 epoch × 10 sample（**smoke 验证管线通，不指望 loss 真下降**）：

```shell #test-setup store="checkpoint_path" load="hidden_states_path>>data_path" load="verifier_path>>verifier_path"
CHECKPOINT_DIR=/root/dflash-trained
rm -rf "$CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR"

# 复用 source install 那步 clone 的 speculators 仓库（cwd 不跨 #test 块，需重新 cd）
cd /root/speculators

# 单卡 A2 上 torchrun --nproc_per_node=1 等价纯 python，多卡并行需要 ≥4 张 davinci
# （vllm-ascend spec_decode E2E 跑在 four_card/）。
# --speculator-type=dflash 由 train.py 从 SpeculatorModel.registry 动态解析
# （v0.7.0.1 注册了 DFlashDraftModel + Eagle3DraftModel + MTPDraftModel +
# PEagleDraftModel + DSparkDraftModel，见 models/__init__.py）。
# --target-layer-ids 2 18 34 必须与 Step 2 一致。
# --on-missing generate --on-generate delete 是 online 训练模式标志（隐藏状态缺时
# 在线补，补完删原文件）；smoke 场景下 hidden_states 已落盘，这个分支不触发。
torchrun --standalone --nproc_per_node=1 scripts/train.py \
  --verifier-name-or-path "<verifier_path>" \
  --data-path "<data_path>" \
  --vllm-endpoint "http://127.0.0.1:8000/v1" \
  --save-path "$CHECKPOINT_DIR" \
  --draft-vocab-size 32000 \
  --epochs 1 \
  --lr 3e-4 \
  --speculator-type dflash \
  --block-size 8 \
  --max-anchors 3072 \
  --num-layers 5 \
  --target-layer-ids 2 18 34 \
  --on-missing generate --on-generate delete >/dev/null 2>&1

echo "$CHECKPOINT_DIR"
```

```shell #test id="pipeline-step3-train" load="checkpoint_path>>checkpoint_path"
echo <checkpoint_path>
ls -1 <checkpoint_path>
```

输出结果如下：

```shell #test-result id="pipeline-step3-train"
/root/dflash-trained
config.json
model.safetensors
```

### Step 4 — 场景 4：Seamless vLLM-Ascend Integration

把 Step 3 训出的 checkpoint 喂给 `vllm-ascend serve --speculative-config`，做一次 chat completion smoke：

```shell #test id="pipeline-step4-serve" load="checkpoint_path>>draft_model" load="verifier_path>>verifier_path"
# 注意：vllm-ascend 的 (num_speculative_tokens + 1) ≤ 15 受
# npu_fused_infer_attention_score 算子限制（vllm-ascend docs
# speculative_decoding.md "Common Configuration" 段）；传 5 留余量。
nohup vllm serve "<verifier_path>" \
  --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"method":"dflash","model":"<draft_model>","num_speculative_tokens":5}' \
  > /tmp/vllm-serve.log 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null" EXIT

# 等 /health 200（最长 6 min）
for i in {1..180}; do
  curl -sf http://127.0.0.1:8000/health > /dev/null && break
  sleep 2
done

# 8-token completion smoke；返回 JSON shape 不可逐字预测，用 fuzzy
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"Hello"}],"max_tokens":8}'

kill "$VLLM_PID" 2>/dev/null || true
```

> vllm-ascend 启动时同时 load verifier（~16 GB）+ draft（训出的几 MB-几 GB）+ 编译 DFlash proposer NPU graph，比裸 vllm 慢；6 min 是 1-card A2 上 vllm-ascend v0.23.0 + Qwen3-8B 的实测上界（实际通常 3-4 min）。

输出结果如下（精确字符串不可预测，用 fuzzy 容忍 UUID / 内容）：

```shell #test-result id="pipeline-step4-serve" fuzzy='xxx'
{"id":"chatcmpl-xxx","object":"chat.completion","created":xxx,"model":"Qwen/Qwen3-8B","choices":[{"index":0,"message":{"role":"assistant","content":"xxx"},"finish_reason":"length"}]}
```

### 编程式入口：`SpeculatorsConfig` / `VerifierConfig` / `TokenProposalConfig`

`speculators` 同时暴露 Python API，对应上游 Quicktour 的「config / model / proposals」分层。本节用最小代码验证 `config.py` 与 `proposals/greedy.py` 可在 NPU 环境中正常 import 并实例化（不依赖任何 GPU 计算）：

```shell #test id="config-import" load="verifier_path>>verifier_path"
python << 'PY'
from speculators import VerifierConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

# VerifierConfig.architectures 是必填字段（pydantic Field 无默认），直接
# ``VerifierConfig(name_or_path=...)`` 会触发 ValidationError。这里直接
# 给出 Qwen3 的 architecture tag，或者调 ``VerifierConfig.from_pretrained(
# "<verifier_path>")`` 让 transformers 自动从 verifier 的 config.json 读。
verifier = VerifierConfig(
    name_or_path="<verifier_path>",
    architectures=["Qwen3ForCausalLM"],
)
# TokenProposalConfig 是 pydantic 的 registry 基类（base.py），唯一已注册
# 的 proposal_type 是 ``greedy``（见 proposals/__init__.py 唯一 import 的
# GreedyTokenProposalConfig + auto_package="speculators.proposals" +
# registry_auto_discovery=True）。DFlash 在 convert_model 路径里也是用
# greedy 做 token 提议，因此这里实例化 GreedyTokenProposalConfig。
proposal = GreedyTokenProposalConfig(
    proposal_type="greedy",
    speculative_tokens=5,
    verifier_accept_k=1,
    accept_tolerance=0.0,
)
print("verifier:", verifier.name_or_path)
print("verifier architectures:", verifier.architectures)
print("proposal type:", proposal.proposal_type)
print("proposal speculative_tokens:", proposal.speculative_tokens)
PY
```

输出结果如下：

```shell #test-result id="config-import"
verifier: /root/.cache/modelscope/hub/models/Qwen/Qwen3-8B
verifier architectures: ['Qwen3ForCausalLM']
proposal type: greedy
proposal speculative_tokens: 5
```

小贴士：

- 4 个核心场景在 vllm-ascend v0.23.0 上端到端串成一条 pipeline；Step 1 → 2 → 3 → 4 产物链：`/root/dflash-qwen3-8b-converted/`（convert 标准格式） → `/root/dflash-train-data/*.safetensors`（训练数据） → `/root/dflash-trained/`（训出的 draft） → chat completion JSON（推理 smoke）。
- Step 1 + Step 4 是 vllm-ascend 真在 GPU 上干活的环节，Step 2 走 vllm 离线 API、Step 3 走 torchrun 离线训练；单卡约束下「vllm serve 与 train 不能同时跑」通过 Step 2 用离线 API + Step 4 才起 vllm serve 来规避。
- `--validate-device <device>` 在 DFlash 分支下只用作「是否跑校验」的布尔开关，**设备字符串本身被丢弃**（`entrypoints.py` 把 `validate_device is not None` 透传给 `DFlashConverter.convert(validate=...)`），DFlash 的 `_validate` 是纯 CPU 的 `DFlashDraftModel.from_pretrained(...)` + NaN 检查，没有 GPU / NPU 计算。本文档跳过校验——CLI `dflash` 不在白名单里、Python API 校验与转换解耦，「能保存到目录」已是充分信号。
- vllm-ascend 的 DFlash proposer 受 `npu_fused_infer_attention_score` 算子 16 token 单次上限约束，`(num_speculative_tokens + 1) ≤ 15`（vllm-ascend docs `feature_guide/speculative_decoding.md` "Common Configuration" 段），Step 4 传 5 是安全值。
- 上游 `examples/train/dflash_qwen3_8b_sharegpt_online_5k.sh` 是 5k sample × 5 epochs × 4 卡 H100 训 25 min 的脚本，本文 smoke 在 1-card A2 上做了 (5k→10 samples) × (5→1 epoch) × (4 卡并行→1 卡串行) 三处缩量；**Step 3 的 smoke 不验证训练效果**（10 sample × 1 epoch 噪声大于信号），只验证「管线通 + 训出的 checkpoint 形态是 vllm-ascend DFlash proposer 能吃的」。
- `VerifierConfig.name_or_path` 接受 HF Hub repo id、本地路径或 ModelScope repo id；离线场景下预先用 `modelscope.snapshot_download` 缓存到本地再传入本地路径即可避免外网拉取。`architectures` 是必填字段（pydantic 无默认），既可手动给（见上方代码示例），也可调 `VerifierConfig.from_pretrained("<verifier_path>")` 让 transformers 从 verifier 的 `config.json` 自动读出。
- `TokenProposalConfig` 是 draft 阶段的 token 提议策略配置基类（pydantic + registry），与具体 speculative decoding 算法（DFlash / EAGLE-3）解耦——v0.7.0.1 唯一已注册的 proposal 是 `GreedyTokenProposalConfig`（`speculators.proposals.greedy`），DFlash 的 convert 路径也是用 greedy 做 token 提议；切换算法只需换 config，对应代码逻辑不必改动。
- 本文档选 DFlash + Qwen3-8B 而非上游 README 给的 EAGLE-3 + Llama-3.1：前者两个 repo 都在 ModelScope 上（HTTP 200）且非 gated；后者 verifier `meta-llama/Meta-Llama-3.1-8B-Instruct` 在 HF 上 gated（需 HF_TOKEN），且 MS 上没有（HTTP 404），走不了 peft / diffusers 同款的 ModelScope 缓存通道。