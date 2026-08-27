# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上用 FastChat 命令行（CLI）对 `Qwen/Qwen2.5-0.5B-Instruct` 做**非交互式**推理。

> 单卡昇腾 NPU 上非交互式运行 FastChat CLI（`--device npu`），模型经 **ModelScope** 下载（`FASTCHAT_USE_MODELSCOPE=True`）。

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
| transformers | `>=4.31, <5` |
| fschat | 0.2.36 |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct`（经 ModelScope 下载） |

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
python -m pip install "fschat[model_worker]" "transformers<5"
python -c "import fastchat, transformers; print('fastchat', fastchat.__version__); print('transformers', transformers.__version__)"
```

输出结果如下（安装日志较长，此处仅展示最后的版本验证输出）：

```shell #test-result id="install-fschat" fuzzy='xxx' fuzzy='...'
...fastchat 0.2.36
transformers xxx
...
```

> - PyPI 发行名为 `fschat`，安装后的 Python 模块目录为 `fastchat`。
> - `transformers<5`：fschat 0.2.36 与 transformers 5.x 的 API 不兼容，固定在 4.x 最新版。
> - 其余依赖已随 `fschat[model_worker]` 一并解析安装。

## 非交互式命令行推理（单卡 NPU）

用管道把预设输入喂给 CLI，`--style programmatic` 会按 `[!OP:user]: ...` / `[!OP:assistant]: ...` 格式打印每一轮消息；第二条 `__END_OF_A_MESSAGE_47582648__` 让 CLI 自动退出，整个命令**无需任何人工交互**即可跑完：

```shell #test id="cli-chat"
printf '你好\n __END_OF_A_MESSAGE_47582648__\n __END_OF_A_MESSAGE_47582648__\n' | \
    FASTCHAT_USE_MODELSCOPE=True \
    python -m fastchat.serve.cli \
        --model-path Qwen/Qwen2.5-0.5B-Instruct \
        --revision master \
        --device npu \
        --style programmatic \
        --max-new-tokens 128
```

输出结果如下（ProgrammaticChatIO 输出格式；qwen 模板的角色名自带 `<|im_start|>` 标记，`xxx` 为模型回复内容）：

```shell #test-result id="cli-chat" fuzzy='xxx' fuzzy='...'
...[!OP:<|im_start|>user]: 你好
...[!OP:<|im_start|>assistant]: xxx...
```

> - `FASTCHAT_USE_MODELSCOPE=True`：从 ModelScope 下载模型权重，而不是 HuggingFace Hub；首次运行会下载模型，请耐心等待。
> - `--revision master`：ModelScope 仓库的默认分支是 `master`，而 FastChat 的 `--revision` 默认值沿用 HuggingFace 惯例的 `main`，需显式指定才能命中 ModelScope 分支。
> - `--device npu`：使用昇腾 NPU 加速（单卡）。
> - `--style programmatic`：非交互式输出格式，配合管道输入使用，适合自动化脚本与 CI 验证。
