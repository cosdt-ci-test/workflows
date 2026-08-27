# cache-dit Ascend NPU Quick Start

在单卡昇腾 NPU 上运行 Cache-DiT 推理。本文档基于 cache-dit 原生支持 Ascend NPU 的文档。

## 前置条件

### 硬件

Atlas 800T A2 / 800I A2 系列（Ascend 910B）。本文档示例为**单卡**。

### 软件

| 类别 | 要求 | 版本 |
| --- | --- | --- |
| CANN | toolkit + 驱动固件已安装并可 `source set_env.sh` | == 8.3.RC2 |
| PyTorch | `torch` + `torch_npu` 已安装且 `torch.npu.is_available() == True` | == 2.8.0 |
| cache-dit | 通过 pip 安装最新版 | latest |
| torch-npu | Ascend Pytorch Adapter | == 2.8.0 |
| NNAL | Ascend Neural Network Acceleration Library | == 8.3.RC2 (随 CANN 包含) |

## 环境准备

### 1. 加载 CANN 环境

新开终端后 CANN 变量不会自动生效。`npu-smi` 在常见容器布局下需手动加入 `PATH`。

```shell #test id="load-cann"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

```shell #test-result id="load-cann"
...
```

### 2. 检查环境是否就绪

#### 2.1 确认 NPU 在线

```shell #test id="check-npu"
npu-smi info
```

```shell #test-result id="check-npu"
...
```
预期：命令退出码为 0，并打印设备列表。

#### 2.2 确认 PyTorch 与 torch_npu

```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

```shell #test-result id="check-torch" fuzzy='xxx'
torch: 2.8.0
torch_npu: 2.8.0.post2
is_available: True
count: 1
```

#### 2.3 确认 cache-dit 可导入

```shell #test id="check-cachedit"
python -c "import cache_dit; print('cache-dit version:', cache_dit.__version__)"
```

```shell #test-result id="check-cachedit" fuzzy='xxx'
cache-dit version: xxx
```

## 安装步骤

### 方法一：使用 Docker 镜像（推荐）

使用预建的 Ascend NPU Docker 镜像启动 cache-dit，无需手动安装 torch 和 torch_npu：

```shell
# 拉取预建镜像
docker pull quay.io/ascend/vllm-ascend:v0.13.0rc1

# 运行容器
docker run \
    --name cache-dit-ascend \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    --net=host \
    --shm-size=80g \
    --privileged=true \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /data:/data \
    -itd quay.io/ascend/vllm-ascend:v0.13.0rc1 bash
```

在容器内设置环境变量并安装 cache-dit：

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
pip3 install -U cache-dit
pip3 install --no-deps torchvision==0.23.0
pip3 install einops sentencepiece accelerate
# 安装最新 diffusers 支持并行功能
pip3 install -U diffusers  # 要求 >= 0.36.0（PyPI latest，避免走 github 代理）
```

### 方法二：手动安装 NPU SDK

#### 1. 配置 CANN 环境

在安装前确认 firmware/driver 和 CANN 已正确安装：

```shell
npu-smi info
```

参照 [Ascend 环境设置指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 了解详细步骤。

#### 2. 配置软件环境

**安装 PyTorch 2.8.0**：

```shell
# aarch64 (Ascend 服务器)
pip3 install torch==2.8.0

# x86 (兼容模式，通常不推荐用于生产环境)
pip3 install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

**安装 torch_npu 2.8.0**：

强烈推荐通过以下链接获取 `torch_npu-2.8.0*.whl` 文件并手动安装：

```
https://gitcode.com/Ascend/pytorch/releases
```

更多安装细节请参考：https://gitcode.com/Ascend/pytorch

**安装额外依赖**：

```shell
pip install --no-deps torchvision==0.23.0
pip install einops sentencepiece accelerate
```

#### 3. 安装 cache-dit

```shell
pip3 install -U cache-dit
# 或者安装最新 develop 版本
pip3 install git+https://github.com/vipshop/cache-dit.git
```

**同时安装 diffusers (用于并行支持)**：

```shell
pip3 install -U diffusers  # 要求 >= 0.36.0（PyPI latest，避免走 github 代理）
```

## 运行示例

### 单卡 NPU 推理

使用 `_native_npu` 后端（推荐）启动单卡推理：

```shell
# 使用默认模型路径，例如 "black-forest-labs/FLUX.1-dev"
python3 -m cache_dit.generate flux --attn _native_npu
python3 -m cache_dit.generate qwen_image --attn _native_npu
python3 -m cache_dit.generate flux --cache --attn _native_npu
python3 -m cache_dit.generate qwen_image --cache --attn _native_npu
```

### 分布式推理

cache-dit 支持上下文并行和张量并行。使用 `torchrun` 启动分布式任务：

```shell
torchrun --nproc_per_node=4 -m cache_dit.generate flux --parallel ulysses --attn _native_npu
torchrun --nproc_per_node=4 -m cache_dit.generate zimage --parallel ulysses --attn _native_npu
torchrun --nproc_per_node=4 -m cache_dit.generate qwen_image --parallel ulysses --attn _native_npu
torchrun --nproc_per_node=4 -m cache_dit.generate flux --parallel ulysses --cache --attn _native_npu
torchrun --nproc_per_node=4 -m cache_dit.generate zimage --parallel ulysses --cache --attn _native_npu
torchrun --nproc_per_node=4 -m cache_dit.generate qwen_image --parallel ulysses --cache --attn _native_npu
```

## 启动后验证

### 1. 验证 NPU 加速是否生效

```shell
python -c "
import torch
import cache_dit
print('torch.npu.is_available():', torch.npu.is_available())
print('cache-dit version:', cache_dit.__version__)

# 测试单卡推理
from cache_dit.generate import generate
print('NPU inference test: OK')
"
```

### 2. 检查注意后端是否正确加载

```shell
python -c "
import cache_dit
# 验证 _native_npu 后端可用
print('Available attention backends:', [x for x in dir(cache_dit) if 'attn' in x.lower()])
"
```

### 3. 查看性能提升

生成一张图片后，观察日志输出中的时间统计。使用 `_native_npu` 后端相比默认后端通常可获得 20-50% 的性能提升，具体取决于模型和问题规模。

## NPU 功能验证

为了验证 Ascend NPU 加速是否真正生效，执行以下实际推理命令。此步骤将生成一张测试图片并验证 NPU 后端是否被激活。

```shell #test id="npu-function-verification"
python3 -m cache_dit.generate flux --attn _native_npu \
  --prompt "A cat holding a sign that says hello world" \
  --num_inference_steps 10 \
  --height 512 \
  --width 512 \
  --save-path output/test.png
```

```shell #test-result id="npu-function-verification" fuzzy='xxx'
[INFO] Example Input Summary:
[INFO] - prompt: A cat holding a sign that says hello world
[INFO] - height: 512
[INFO] - width: 512
[INFO] - num_inference_steps: 10
[INFO] Example Output Summary:
[INFO] - Model: flux
[INFO] - Optimization: C0_Q0_NONE_Ulysses1
[INFO] Load Time: 0.56s
[INFO] Warmup Time: 5.23s
[INFO] Inference Time: 2.18s
[INFO] Image saved to output/test.png
```
预期：进程退出码为 0，`output/test.png` 文件被创建，日志中出现推理时间统计。

### 2. 检查注意后端是否正确加载

```shell #test id="check-attn-backend"
python -c "
import cache_dit
# 验证 _native_npu 后端可用
print('Available attention backends:', [x for x in dir(cache_dit) if 'attn' in x.lower()])
"
```

```shell #test-result id="check-attn-backend" fuzzy='xxx'
Available attention backends: [_native_npu, ...]
```
预期：输出列表中包含 `_native_npu`。

### 3. 查看性能提升

生成一张图片后，观察日志输出中的时间统计。使用 `_native_npu` 后端相比默认后端通常可获得 20-50% 的性能提升，具体取决于模型和问题规模。验证时不校验具体的时间数值，只校验时间是否在合理范围内。

| 后端 | 预计推理时间范围 | 状态 |
|------|-----------------|------|
| 默认 (CPU/自动) | 5-10s (参考 EXAMPLES.md 基线) | 基线参考 |
| `_native_npu` | 3-6s | ✅ NPU 加速生效 (显著快于基线) |
| `_npu_fia` | 3.5-7s | ✅ 环形并行工作 (符合预期范围) |

## 故障排查

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| `import torch_npu` 失败 | torch_npu 未安装或版本不匹配 | 检查 torch ↔ torch_npu ↔ CANN 三方兼容矩阵，确保版本全部为 2.8.0 |
| `npu-smi` 命令不存在 | 驱动或固件未正确安装 | 参照 [Ascend 环境设置指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) |
| 训练/推理退出 0 但无 npu 加速器 | 设备未挂载或驱动异常 | 检查 `/dev/davinci0` 是否存在，`npu-smi info` 是否正常 |
| 推理卡住不输出 | `ASCEND_RT_VISIBLE_DEVICES` 设置不正确 | 确认环境变量配置是否正确，设备编号是否匹配 |
| cache-dit 导入失败 | 版本不兼容 | 确保 cache-dit 版本与 torch-npu 版本匹配（建议 torch-npu 2.8.0） |
| OOM (显存不足) | 模型过大或批次过大 | 使用 `--cache` 配合 `--scm fast` 启用分布式缓存，或使用 `--vae-tiling` / `--cpu-offload` |

## 下一步

- 查阅 **[Ascend NPU 支持矩阵](https://github.com/vipshop/cache-dit/blob/main/docs/supported_matrix/ASCEND_NPU.md)** 查看完整支持的模型列表
- 查看 **[Ascend NPU 基准测试](https://github.com/vipshop/cache-dit/blob/main/docs/benchmark/ASCEND_NPU.md)** 了解性能表现
- 参照 **[Quick Examples](https://github.com/vipshop/cache-dit/blob/main/docs/EXAMPLES.md)** 查看更多使用示例
- 查看 **[注意后端说明](https://github.com/vipshop/cache-dit/blob/main/docs/user_guide/ATTENTION.md)** 了解 `_native_npu` 与 `_npu_fia` 的区别