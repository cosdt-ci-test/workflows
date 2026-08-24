# 快速开始：在昇腾 NPU 上跑通 bitsandbytes 的 4-bit 量化

> **阅读本文前**，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。本文聚焦**第一次跑通**：装上 PyTorch NPU 栈和 bitsandbytes，在单卡 NPU 上完成一次 NF4 `Linear4bit` 前向。

[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) 官方**不支持**昇腾。跟踪状态见 [issue #1847](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1847)。本文用的是上游**默认后端**里那条设备无关的 4-bit 量化数学：权重量化发生在 CPU 内核，张量再被搬到 `npu:0` 上做矩阵乘。这不是专用 NPU kernel，也不是官方 Ascend 后端。

8-bit 训练和 8-bit 优化器在默认后端里没有实现。未合入的 [PR #1695](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1695) 不在本文范围内。

---

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为**单卡**。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装并可 `source set_env.sh` |
| Python | 3.12 |
| PyTorch | `torch==2.9.0` 与 `torch_npu==2.9.0.post2`，见下文安装 |
| bitsandbytes | 从 PyPI 安装发布版，见下文 |

---

## 1. 加载 CANN 环境

新开终端后 CANN 变量不会自动生效。常见容器里 `npu-smi` 在 `/usr/local/sbin`，需要把该目录加入 `PATH`。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
```

`PYTHONNOUSERSITE=1` 让 Python 忽略用户目录里的包。本机如果曾经 `pip install --user` 过 CANN 相关包，不设这个变量时，pip 解析器可能被带偏。

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
python --version
```

**预期**：`ASCEND_HOME_PATH` 非空；`npu-smi` 与 `python` 都能找到。

---

## 3. 安装 PyTorch NPU 栈

昇腾上的 `torch_npu` 要从华为 PyPI 额外索引安装，并钉死与 CANN 匹配的版本。`numpy` 和 `pyyaml` 也要一起装。`torch_npu` 的 wheel **没有声明**这两项依赖，但 `import torch` 会自动加载 `torch_npu`，缺了会在你显式 `import torch_npu` 之前就失败。

```shell #test id="install-torch"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
python -m pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
  torch==2.9.0 torch_npu==2.9.0.post2 numpy pyyaml
python -c "import numpy, yaml, torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
```

输出结果如下：

```shell #test-result id="install-torch"
...
torch 2.9.0...
torch_npu 2.9.0.post2
npu_available True
```

`npu_available` 必须是 `True`。`False` 时不要继续，先查 CANN、驱动和可见设备。

---

## 4. 安装 bitsandbytes

将 `<UPSTREAM_REF>` 换成目标 **PyPI 版本号**（例如当前最新发布 `0.50.1`）。看护流水线会写入本次要测的 release tag。
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="install-bnb" load="upstream_ref>>UPSTREAM_REF"
python -m pip install bitsandbytes==<UPSTREAM_REF>
python -c "import bitsandbytes as bnb; import bitsandbytes.cextension as ce; print('bitsandbytes', bnb.__version__); print('BNB_BACKEND', ce.BNB_BACKEND); print('lib', type(ce.lib).__name__)"
```

输出结果如下：

```shell #test-result id="install-bnb"
...
bitsandbytes ...
BNB_BACKEND CPU
lib BNBNativeLibrary
```

`BNB_BACKEND=CPU` **不是**失败，也**不能**当成 NPU 证据。默认后端在昇腾上走 CPU 原生 `.so` 做 4-bit 量化数学，设备名打印 `CPU` 是预期行为。真正的 NPU 证据是下一节里 `out.device` 为 `npu:0`。

如果 `lib` 打印的是 `ErrorHandlerMockBNBNativeLibrary`，说明原生库没加载成功，后面的 4-bit 前向不可信。

---

## 5. 在 NPU 上做一次 NF4 Linear4bit 前向

下面这段在 CPU 上构造 `Linear4bit(64, 32)`，搬到 `npu:0` 量化，再用 float16 输入做一次前向。

**怎样算成功**

1. 进程退出码为 0；
2. `out.device` 必须是 `npu:0`。若打印 `cpu`，那是静默回退，视为失败。

```shell #test id="nf4-forward"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
python - <<'PY'
import torch
import torch_npu
import bitsandbytes as bnb

layer = bnb.nn.Linear4bit(
    64, 32, bias=False, compute_dtype=torch.float16, quant_type="nf4",
    compress_statistics=False,
)
layer = layer.to("npu:0")
x = torch.randn(4, 64, dtype=torch.float16, device="npu:0")
out = layer(x)
weight = layer.weight
print("weight.device", weight.device)
print("weight.dtype", weight.dtype)
print("weight.bnb_quantized", weight.bnb_quantized)
print("out.device", out.device)
print("out.shape", tuple(out.shape))
print("out.dtype", out.dtype)
print("NF4 forward on Ascend NPU: OK")
PY
```

输出结果如下：

```shell #test-result id="nf4-forward"
weight.device npu:0
weight.dtype torch.uint8
weight.bnb_quantized True
out.device npu:0
out.shape (4, 32)
out.dtype torch.float16
NF4 forward on Ascend NPU: OK
```

---

## 6. 本文没有覆盖的能力

默认后端在昇腾上**没有**这些实现。不要照抄 CUDA 文档去跑它们，并期待同样结果：

- 8-bit 训练（`Linear8bitLt` 一类路径）
- 8-bit 优化器
- 分页优化器 / Intel XPU 示例
- 未合入的 [PR #1695](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1695) 以及 `multi-backend-refactor` / `bitsandbytes-npu-beta` 分支

---

## 故障排查

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| `import torch` 报缺 `numpy` 或 `yaml` | `torch_npu` 未声明这两项依赖 | 与 torch 栈一起安装 `numpy` `pyyaml` |
| `torch.npu.is_available()` 为 `False` | 未 `source set_env.sh`，或设备未挂进容器 | 重做第 1–2 节 |
| `lib` 是 `ErrorHandlerMockBNBNativeLibrary` | CPU 原生 `.so` 没编出来或没加载 | 确认 `cmake` / `g++` 可用后从源码 `pip install -e .` |
| 前向报缺 `bucketize` 或 uint8 算子 | 这条 4-bit 路径用到的 PyTorch 算子在当前 `torch_npu` 上还没有 | 记录算子名；这是默认路径在昇腾上的缺口，不是安装步骤写错 |
| `out.device` 为 `cpu` 但退出码 0 | 静默 CPU 回退 | 按失败处理，检查 `.to("npu:0")` 与可见设备 |
| pip 找不到 `torch_npu==2.9.0.post2` | 没用华为 extra-index，或用了会 first-index-wins 的安装器 | 用 `python -m pip` 加上文的 `--extra-index-url` |
