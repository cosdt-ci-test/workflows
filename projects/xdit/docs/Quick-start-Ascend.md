# Quick Start (Ascend NPU)

在 2 卡昇腾 NPU 上把 [xDiT](https://github.com/xdit-project/xDiT)（发布到 PyPI 的包名为 [`xfuser`](https://pypi.org/project/xfuser/)）从源码装起、跑通 `python examples/sd3_example.py --help` / NPU 设备检测 / 单卡推理 smoke / 多卡（`torchrun --nproc_per_node=2 examples/sd3_example.py` 走 HCCL）推理 smoke 这条全链路。

> xDiT 0.4.5 的 `setup.py` **没有 `entry_points` 段**（也没有 `xfuser/cli.py`），所以 `xdit` console_script 不存在；本文档所有原 `xdit --xxx` 实际入口都是 `python examples/<model>_example.py --xxx`（用 `sd3_example.py` 作 smoke 模型），多卡用 `torchrun --nproc_per_node=N examples/sd3_example.py --xxx` 显式起 rank。

模型权重通过 [ModelScope](https://www.modelscope.cn) `snapshot_download` 落到本地，再让 xfuser 走 HuggingFace Hub 的 `from_pretrained` 读本地：xfuser 本身不直接认 ModelScope，但这条路径在国内 CI 限速友好，是 ms-swift / diffusers / xtuner / specforge 等已经默认采用的方式。

`flash-attn` 在 NPU 上不可用（xfuser 内部 `xfuser/envs.py:check_flash_attn` 命中 `_is_npu()` 直接返回 False 并 log `flash_attn is not ready on torch_npu for now`），本文档不装它；xfuser 直接走 yunchang 的 `ring/ulysses` 注意力路径。NPU 上 PipeFusion 也尚未支持（PR #566 / mainline 在 `xfuser/core/distributed/parallel_state.py` 里加了 `assert pipeline_parallel_degree == 1`），多卡 smoke 只走 USP（`--ulysses_degree N` + `--ring_degree N`），不设 `--pipefusion_parallel_degree`。

## 前置条件

### 硬件

- **Atlas 900 A2 / A3 训练系列产品**或 **Ascend 950 系列产品**，并按需完成物理机或容器内的设备挂载（`/dev/davinci*` 等）。
- **至少 2 张可见 NPU**：本文档用 `linux-aarch64-a2-2`（2 卡 910B4）这条 runner；单卡也跑得起来单卡 smoke（`--ulysses_degree 1`），但多卡 smoke（`torchrun --nproc_per_node=2`）需要 ≥2 卡才能起 HCCL 多 rank。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 2）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| xfuser | upstream 最新 release 的源码（>= 0.4.5，已合并 `xfuser/envs.py` 里的 `_is_npu()` + `get_torch_distributed_backend() == "hccl"` 分支） |
| diffusers | `>=0.33.0`（xfuser `install_requires`，先装再跑 `python examples/sd3_example.py --model <model_path>` 等需要 diffusers 的模型） |
| yunchang | `>=0.6.0`（xfuser 安装依赖，NPU 上无需 `flash-attn`，yunchang 走纯 PyTorch ring 实现） |
| modelscope | `>=1.18`（用于 `snapshot_download` 把模型权重先拉到本地） |
| 推理后端 | torch_npu / `torch.distributed` / `hccl`（xfuser 自动检测；无需手动配置 `master_addr` / `master_port`） |

### 前置安装

确认能看到 ≥ 2 张 NPU 设备：

```shell #test id="npu-smi-info"
npu-smi info > /dev/null && echo "npu_smi_ok: yes"
```

```shell #test-result id="npu-smi-info" disable_fuzzy
npu_smi_ok: yes
```

`npu-smi info` 完整输出类似：

```
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
| 1     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:42:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
```

> 如果 `npu-smi` 不存在或只看到 1 张 910B4 卡，回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动；本文档的多卡 smoke 在 1 卡机器上跑不起来。

检查 Python 版本：

```shell #test id="check-py"
python --version
```
输出结果如下：
```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

安装 `torch` / `torch_npu`：

```shell #test-setup
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.9.0
uv pip install --extra-index-url https://mirrors.aliyun.com/pypi/simple torch_npu==2.9.0.post2
```

检查 torch / torch_npu 是否装好且 NPU 设备可用：

```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-torch"
torch= 2.9.0+cpu
torch_npu= 2.9.0.post2
is_available: True
count: 2
```

> - 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。
> - `count: 2` 是多卡 smoke 的前置；少卡本文档跑不起来——`torchrun --nproc_per_node=2` 在 1 卡机器上 InitProcessGroup 就报 `torch.distributed.DistBackendError`。

装 xfuser `install_requires` 里**没**声明、但 NPU smoke 用得着的两个包（`diffusers` / `transformers` / `accelerate` / `sentencepiece` / `beautifulsoup4` / `einops` 都在 [xfuser 0.4.5 setup.py](https://github.com/xdit-project/xDiT/blob/0.4.5/setup.py) 的 `install_requires` 里，下一步 `uv pip install -e .` 会自动带）：

```shell #test-setup
uv pip install peft modelscope
```

## 安装 xDiT

xDiT 的发布名是 `xfuser`（PyPI / setup.py 的 `name`）。`setup.py` **没有** `entry_points` 段，`xfuser/cli.py` 也不存在——`xdit` console_script 在 xDiT 0.4.5 **没有注册**；统一 CLI 入口实际是 `python examples/<model>_example.py`（用 SD 3.5 medium 跑 smoke 时是 `python examples/sd3_example.py`），底层是 `xfuser.config.FlexibleArgumentParser + xfuser.xFuserArgs`。

### 从源码安装

GitHub Releases 当前最新稳定是 0.4.5；`xfuser/envs.py` 里 `_is_npu()` / `get_torch_distributed_backend()` / `check_npu_flash_attn()` 都是 mainline 已有能力，无需 PR / patch。

```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```

克隆上游仓库并 checkout 到工作流注入的最新 release tag，安装并且验证

```shell #test id="xfuser-install-source" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/xdit-project/xDiT.git
cd xDiT
uv pip install -e .
python -c "from importlib.metadata import version; print('xfuser', version('xfuser'))"
```

\<ref> 为安装的最新的 release 分支（如 `0.4.5`）。

输出结果如下：

```shell #test-result id="xfuser-install-source" fuzzy='xxx'
xfuser xxx
```

- xxx 表示最新的版本号
- `uv pip install -e .` 会按 [xfuser 0.4.5 setup.py 的 `install_requires`](https://github.com/xdit-project/xDiT/blob/0.4.5/setup.py) 自动装 `torch>=2.4.1` / `accelerate>=0.33.0` / `transformers>=4.39.1` / `sentencepiece>=0.1.99` / `beautifulsoup4>=4.12.3` / `distvae` / `yunchang>=0.6.0` / `einops` / `diffusers>=0.33.0`。NPU 镜像已装的 `torch==2.9.0+cpu` / `torch_npu==2.9.0.post2` 满足 `torch>=2.4.1`，pip 不会重装。
- `peft`（LoRA adapter 推理）和 `modelscope`（用 `snapshot_download` 下权重）**不在** xfuser `install_requires` 里，已经在前面的「前置安装」里手动装好了。
- `av`（视频模型 `Wan` / `HunyuanVideo` 的视频编码要用到）**也不在** xfuser `install_requires` / `extras_require` 里，需要单独装：

打印安装版本：

```shell #test id="install-deps"
python -c "import diffusers, transformers, accelerate, einops, sentencepiece, modelscope; print('diffusers', diffusers.__version__); print('transformers', transformers.__version__); print('accelerate', accelerate.__version__); print('einops', einops.__version__); print('sentencepiece', sentencepiece.__version__); print('modelscope', modelscope.__version__)"
```

输出结果如下：

```shell #test-result id="install-deps" fuzzy='xxx'
diffusers xxx
transformers xxx
accelerate xxx
einops xxx
sentencepiece xxx
modelscope xxx
```

```shell #test-setup
uv pip install av
```

打印最终依赖版本：

```shell #test id="xfuser-deps"
python -c "from importlib.metadata import version; import yunchang, distvae; print('xfuser', version('xfuser')); print('yunchang', yunchang.__version__); print('distvae', version('distvae'))"
```

输出结果如下：
```shell #test-result id="xfuser-deps" fuzzy='xxx'
xfuser xxx
yunchang xxx
distvae xxx
```

## 下载基础模型（ModelScope）

xfuser 默认走 HuggingFace Hub 的 `from_pretrained`；本文档用 ModelScope 镜像预拉，因为 CI 集群拉 HFace 限速明显。`snapshot_download` 返回本地路径之后导出 `XFUSER_MODEL_PATH`，让 xfuser 直接读本地 hub layout。

SD 3.5 medium（约 2 B 参数，bf16 ≈ 4 GB）做 smoke：在 2 张 910B4 64 GB HBM 内还留有足够 activation 余量，且不开 `trust_remote_code`、不在 HF 上 gating，ModelScope 镜像同步存在。

```shell #test-setup store="model_path"
set -o pipefail
python -c "from modelscope import snapshot_download; print(snapshot_download('stabilityai/stable-diffusion-3-medium-diffusers'))" | grep '^/' | tail -n 1
```

输出类似：

```
/root/.cache/modelscope/hub/models/stabilityai/stable-diffusion-3-medium-diffusers
```

> ModelScope 的 `hub/models/<org>/<model>` 目录结构与 HF Hub 对齐；xfuser 的 `from_pretrained` 用同一套 repo-id 解析逻辑，能直接读 ModelScope 落地的本地路径，无需 `HF_HUB_OFFLINE=1` 之类的开关。模型 id 在 HF 和 ModelScope 上同名时这条路径最稳；两者 id 不同（比如 `AI-ModelScope/...` 那个 namespace）时改 id 即可。

## 使用样例

### 验证 NPU 路径

xfuser 在 mainline 已经把 NPU 识别 + `hccl` 后端 + `npu_fused_infer_attention_score` flash 路径都接好了（见 `xfuser/envs.py`）。下面这条命令只校验 xfuser 在 NPU 上跑 import 时调用到的分发函数不会因为 `_is_cuda()` / `_is_hip()` / `_is_npu()` 的分支判断点出错：

```shell #test id="xfuser-detect-npu"
python - <<'PY'
from xfuser.envs import get_device, get_device_name, get_torch_distributed_backend, _is_npu
print('device_name', get_device_name())
print('dist_backend', get_torch_distributed_backend())
print('local_rank0_device', get_device(0))
print('is_npu', bool(_is_npu()))
PY
```

输出结果如下：

```shell #test-result id="xfuser-detect-npu"
torch.npu synchronize
device_name npu
dist_backend hccl
local_rank0_device npu:0
is_npu True
```

> 这条命令**不**真的启动 `torch.distributed.init_process_group`，所以卡数 `count` 不影响：通过 `xfuser/envs.py` 里的纯函数（`_is_npu()` / `get_device()` 等）验证分发路径。如果 `platform` 不是 `npu`，回到「前置安装」检查 `torch_npu` 是否装对。

### 验证 CLI

xDiT 0.4.5 没有 `setup.py entry_points` 注入的 `xdit` console_script（[setup.py:23-72](https://github.com/xdit-project/xDiT/blob/0.4.5/setup.py) 没有 `entry_points` 段；`xfuser/cli.py` 也不存在），统一 CLI 入口实际是 `python examples/<model>_example.py`（如 `python examples/sd3_example.py`），底层是 `xfuser.config.FlexibleArgumentParser + xfuser.xFuserArgs`。这条 `--help` 验证只需要 Python + xfuser，不需要 NPU 设备：

```shell #test id="xdit-help"
python -c "
from xfuser.config import FlexibleArgumentParser
from xfuser import xFuserArgs
p = FlexibleArgumentParser()
xFuserArgs.add_cli_args(p)
p.print_help()
" 2>&1
```

输出结果包含（按需 fuzzy 匹配，不锁行号；CAN 里 `torch.npu synchronize` 是 import-time 副作用，会进 stdout；transformers import 还会出两行 torchvision warning）：

```shell #test-result id="xdit-help" fuzzy='...'
...torch.npu synchronize
...--model MODEL         Name or path of the huggingface model to use.
...--ulysses_degree ULYSSES_DEGREE
...--ring_degree RING_DEGREE
...--pipefusion_parallel_degree PIPEFUSION_PARALLEL_DEGREE
...--use_cache           Use cache config for attention compression.
```

### 单卡推理 smoke（`--ulysses_degree 1`，2 卡机器也跑）

完整脚本会下载 Hub 上的模型权重并实际跑一遍 xfuser 自带的 runner。本文档用最小 smoke（`--num_inference_steps 1` + `--height 256 --width 256`），目的是把"install + 启动 xfuser runtime + 走通单 rank 的 torch.distributed init"完整跑一遍；模型用 SD 3.5 medium（约 4 GB），单卡 64 GB HBM 内还留有 activation 余量。

`torchrun --nproc_per_node=1 examples/sd3_example.py --ulysses_degree 1 --ring_degree 1` 单 rank 单进程（rank=0/world=1）：

```shell #test id="xdit-infer-single" load="model_path>>model_path"
export TORCH_NPU_USE_HCCL=1
# CANN 的 torch_npu import 副作用会往 stdout 打 "torch.npu synchronize" 一行，
# 会污染下面的 $(...) 命令替换：dirname 会把整段（含换行）当一个路径处理，
# 报 "No such file or directory"。先把 xfuser 父目录（=xDiT 仓库根）写进文件，再 cd。
# 注意只 cd 到 dirname 一次：xfuser __file__ = .../xDiT/xfuser/__init__.py，
# dirname 一次 = .../xDiT（已经是仓库根了），不能再 /..，否则落到上层 workflows/。
python -c 'import os, xfuser; open("/tmp/_xdit_root", "w").write(os.path.dirname(os.path.dirname(xfuser.__file__)))'
cd "$(cat /tmp/_xdit_root)"
# xfuser config/args.py:create_config 里 not use_ray and not is_initialized() 会无条件调
# init_distributed_environment → torch.distributed.init_process_group(backend=hccl, env://)，
# 需要 RANK / WORLD_SIZE / LOCAL_RANK。直接 `python examples/sd3_example.py` 没 torchrun 注环境
# 必报 "environment variable RANK expected"。所以单卡也要走 torchrun --nproc_per_node=1，
# 让 torchrun 把 RANK=0 / WORLD_SIZE=1 / LOCAL_RANK=0 注入环境。
torchrun --nproc_per_node=1 examples/sd3_example.py --model "<model_path>" \
    --prompt "a tiny test sketch" \
    --height 256 --width 256 \
    --num_inference_steps 1 \
    --seed 42 \
    --ulysses_degree 1 --ring_degree 1
```

输出结果包含（按需 fuzzy 匹配）：

```shell #test-result id="xdit-infer-single" fuzzy='...'
...
...epoch time: ...
...parameter memory: ...
...peak memory: ...
```

### 多卡推理 smoke（`--ulysses_degree 2`，HCCL 多 rank）

这是 PR #566 / mainline 文档主线场景：「单节点内多卡，DP / USP / CFG 并行」其中一种；NPU 路径下 PipeFusion 仍不可用（PR #566 在 `xfuser/core/distributed/parallel_state.py` 加了 `assert pipeline_parallel_degree == 1`），所以本文档**只走 USP**：`ulysses_degree=2, ring_degree=1, data_parallel_degree=1`，度乘积 = 2 → 用 `torchrun --nproc_per_node=2` 显式起两个 rank 跑 HCCL 集合通信。

```shell #test-setup id="xdit-infer-multi-setup" load="model_path>>model_path"
mkdir -p ./results
# TORCH_NPU_USE_HCCL=1 让 torch.distributed.init_process_group(backend="hccl") 在 PR #566 之前的 1.x 版本上走通；2.9.0.post2 是默认 hccl，
# 但保留 export 以防降级路径。这些 env 通过 torchrun 透传到子进程。
export TORCH_NPU_USE_HCCL=1
```

```shell #test id="xdit-infer-multi" load="model_path>>model_path"
python -c 'import os, xfuser; open("/tmp/_xdit_root", "w").write(os.path.dirname(os.path.dirname(xfuser.__file__)))'
cd "$(cat /tmp/_xdit_root)"
# ulysses_degree=2, ring_degree=1：度乘积 = 2，torchrun --nproc_per_node=2 显式起 2 个 rank；
# xfuser 内部通过 xfuser.envs.get_torch_distributed_backend() 选 hccl，不读 env。
torchrun --nproc_per_node=2 examples/sd3_example.py \
    --model "<model_path>" \
    --prompt "a tiny test sketch" \
    --height 256 --width 256 \
    --num_inference_steps 1 \
    --seed 42 \
    --ulysses_degree 2 --ring_degree 1 \
    --data_parallel_degree 1
```

输出结果至少包含一行以 `epoch time:` 开头（`examples/sd3_example.py` 末尾由 rank=world_size-1 那个进程打）：

```shell #test-result id="xdit-infer-multi" fuzzy='...'
...epoch time: ...
...parameter memory: ...
...peak memory: ...
```

落盘至少一张 `.png`（`examples/sd3_example.py` 把图片写到 cwd 的 `./results/`，文件名带 `dp/cfg/ulysses/ring/pp/patch/rank`）：

```shell #test-setup store="multi_output_dir"
echo ./results
```

```shell #test id="xdit-output-list-multi" load="multi_output_dir>>output_dir"
ls -1 "${output_dir}" 2>/dev/null
```

输出结果至少包含一行（`stable_diffusion_3_result_*` 由 `is_dp_last_group()` 的 rank 落盘）：

```shell #test-result id="xdit-output-list-multi" fuzzy='...'
stable_diffusion_3_result_...png
```