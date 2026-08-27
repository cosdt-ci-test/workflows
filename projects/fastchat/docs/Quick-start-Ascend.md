# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上用 FastChat 命令行（CLI）对 `lmsys/vicuna-7b-v1.5` 做**非交互式**推理。

> 本文档聚焦**单卡** NPU（`--device npu`），并从 **ModelScope** 下载模型（设置 `FASTCHAT_USE_MODELSCOPE=True`，绕开 HuggingFace Hub 的网络限制）。CLI 以 **非交互（programmatic）** 方式运行：通过管道把预设输入喂给 stdin，以 `__END_OF_A_MESSAGE_47582648__` 作为一条消息的结束标记，第二条结束标记让 CLI 自动退出——全程无需人工键盘输入。

## 前置条件

### 硬件

Atlas 900 A2 单卡（Ascend NPU），并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import`、`torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

按上游 README 的方式设置 CANN 环境变量：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 单卡
- **操作系统**：Ubuntu 22.04

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| transformers | `>=4.31` |
| fschat | 0.2.36 |
| 模型 | `lmsys/vicuna-7b-v1.5`（经 ModelScope 下载） |

## 环境检查

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
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
count: 1
```

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

## 安装 fschat

```shell #test id="install-fschat"
pip install "fschat[model_worker]"
python -c "import fschat; print('fschat', fschat.__version__)"
```

输出结果如下：

```shell #test-result id="install-fschat"
fschat 0.2.36
```

> `transformers>=4.31` 已随 `fschat[model_worker]` 一并解析安装。

## 非交互式命令行推理（单卡 NPU）

用管道把预设输入喂给 CLI，`--style programmatic` 会按 `[!OP:user]: ...` / `[!OP:assistant]: ...` 格式打印每一轮消息；第二条 `__END_OF_A_MESSAGE_47582648__` 让 CLI 自动退出，整个命令**无需任何人工交互**即可跑完：

```shell #test id="cli-chat"
printf '你好\n __END_OF_A_MESSAGE_47582648__\n __END_OF_A_MESSAGE_47582648__\n' | \
    FASTCHAT_USE_MODELSCOPE=True \
    python3 -m fastchat.serve.cli \
        --model-path lmsys/vicuna-7b-v1.5 \
        --device npu \
        --style programmatic \
        --max-new-tokens 128
```

输出结果如下（ProgrammaticChatIO 输出格式，`xxx` 为模型回复内容）：

```shell #test-result id="cli-chat" fuzzy='xxx' fuzzy='...'
...[!OP:user]: 你好
[!OP:assistant]: xxx
...
```

> - `FASTCHAT_USE_MODELSCOPE=True`：从 ModelScope 下载模型权重，而不是 HuggingFace Hub；首次运行会下载模型，请耐心等待。
> - `--device npu`：使用昇腾 NPU 加速（单卡）。
> - `--style programmatic`：非交互式输出格式，配合管道输入使用，适合自动化脚本与 CI 验证。
