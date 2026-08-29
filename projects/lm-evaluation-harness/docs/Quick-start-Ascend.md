# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上安装 lm-evaluation-harness（lm-eval），对 `Qwen/Qwen2.5-0.5B-Instruct` 跑一次真实的 `arc_easy` 评测并输出准确率。

> 单卡昇腾 NPU 上以 HuggingFace 后端（`--model hf --device npu:0`）运行 lm-eval CLI；模型经 **ModelScope** 下载后以本地路径加载，评测数据集经 HF 镜像下载，`--limit 10` 控制规模，全程无需人工交互。

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
| lm-eval | 0.4.12（经 PyPI 安装） |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct`（经 ModelScope 下载） |
| 评测任务 | `arc_easy`（`--limit 10` 功能测试口径） |

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

## 安装 lm-eval

```shell #test id="install-lmeval"
python -m pip install "lm_eval[hf]" "transformers<5" modelscope
python -c "from importlib.metadata import version; print('lm_eval', version('lm_eval')); print('transformers', version('transformers')); print('modelscope', version('modelscope'))"
```

输出结果如下：

```shell #test-result id="install-lmeval" fuzzy='xxx' fuzzy='...'
...lm_eval 0.4.xxx
transformers 4.xxx
modelscope 1.xxx
```

> - `lm_eval[hf]` 安装 HuggingFace transformers 后端（0.4.x 起 base 包不含 `transformers`/`torch`；本机已装的 torch 栈不会被改动）。
> - 显式钉住 `transformers<5`：transformers 5.x 与 lm-eval 0.4.x 生态的兼容性未经上游验证。
> - `modelscope` 为 ModelScope 下载模型所需，需显式安装。

## 下载模型

```shell #test-setup store="model_dir"
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', revision='master'))" | tail -n 1
```

> `tail -n 1` 过滤下载进度输出，仅保留模型目录路径；Qwen2.5-0.5B-Instruct fp16 约 1GB，首次运行请耐心等待。

## 运行评测（单卡 NPU）

```shell #test id="run-eval" load="model_dir>>model"
export HF_ENDPOINT=https://hf-mirror.com
python -m lm_eval run --model hf \
    --model_args pretrained=<model> \
    --tasks arc_easy \
    --device npu:0 \
    --batch_size 8 \
    --limit 10 \
    --output_path /tmp/lm_eval_out.json
```

输出结果如下（评测日志较长，此处仅校验关键锚点）：

```shell #test-result id="run-eval" fuzzy='...'
...arc_easy...acc...
```

- `--model hf`：HuggingFace transformers 后端；`pretrained=<本地目录>` 直接加载已下载的模型，评测进程不再联网取模型。
- `--device npu:0`：lm-eval 的设备白名单以 `npu:<i>` 形式收录 NPU 设备（单卡即 `npu:0`）；裸 `npu` 不在白名单内，勿省略索引。
- `--tasks arc_easy --limit 10`：AI2 ARC-Easy 小规模功能评测；`--limit 10` 只取前 10 个样本，控制墙钟时间（数值本身不代表模型能力）。
- `--batch_size 8`：0.5B 模型单卡 batch 8。
- `--output_path`：结果 JSON 落盘，供下一步校验。
- `HF_ENDPOINT=https://hf-mirror.com`：评测数据集 `allenai/ai2_arc` 默认从 HuggingFace Hub 下载；机器不可达 HuggingFace 时必须设置镜像。

## 检查评测结果

```shell #test id="check-acc"
python -c "import json; r = json.load(open('/tmp/lm_eval_out.json')); acc = r['results']['arc_easy']['acc,none']; assert 0.0 <= acc <= 1.0, acc; print('acc', round(acc, 4))"
```

输出结果如下：

```shell #test-result id="check-acc" fuzzy='xxx'
acc xxx
```

> 结果 JSON 的 `results.arc_easy` 下 `acc,none`（原始准确率）与 `acc_norm,none`（长度归一化准确率）均为 [0,1] 区间浮点数。
