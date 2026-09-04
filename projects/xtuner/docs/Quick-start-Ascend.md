# 快速开始

在单卡昇腾 NPU 上跑通 [xtuner](https://github.com/InternLM/xtuner) 的最小链路。

本文档以 **Qwen1.5-1.8B-Chat** + Colorist 指令微调数据为例，端到端走通「权重下载 → 配置修改 → 单/多卡训练 → LoRA 合并 → chat 推理」全链路。

## 你会做什么

按顺序走完 6 步，每步都附带一段可执行的验证命令，验证通过再进下一步：

1. **装环境**——CANN + torch + torch_npu + xtuner
2. **下权重**——Qwen1.5-1.8B-Chat
3. **下数据**——Colorist 颜色描述数据集，几 MB
4. **改 cfg**——拷 xtuner 模板 cfg，改 4 处（路径 / 数据 / 量化 / epoch）
5. **跑训练**——5 轮迭代的最小训练（约 30 秒），验证整条训练链路和训练中的采样输出
6. **merge + chat**——LoRA adapter 合并回 base，跟合并后 / adapter 模型对话

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4）。本文档示例模型 1.8B + fp16 base + plain LoRA。
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.11.0+cpu |
| torch_npu | 2.11.0 |
| xtuner | 最新 release tag（从 https://github.com/InternLM/xtuner/releases 查询） |

### 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

`npu-smi info` 完整输出类似：

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

装 `torch` / `torch_npu`：
```shell #test-setup
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.11.0
uv pip install --extra-index-url https://mirrors.aliyun.com/pypi/simple torch_npu==2.11.0
```

检查 torch / torch_npu 是否装好且 NPU 设备可用：
```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：
```shell #test-result id="check-torch" fuzzy='xxx'
torch= 2.11.0+cpu
torch_npu= 2.11.0
is_available: True
count: xxx
```

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

装 `modelscope`：

```shell #test-setup
uv pip install modelscope
```

## 安装 xtuner

xtuner 支持两种安装方式，本文档都验证一遍：**方式一** PyPI 二进制安装最简单；**方式二** 源码安装——后续的训练 / 转换 / chat 命令都基于源码 clone 的目录，想跟进最新 tag 或改源码也用这种。NPU 上两种方式都要 `--no-deps` 绕开 bitsandbytes 硬 pin（aarch64 无可用 wheel），运行依赖手动列出。

### 方式一：PyPI 二进制安装

```shell #test id="xtuner-install-binary"
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple --no-deps xtuner
uv pip install 'mmengine==0.10.6' 'transformers==4.48.0' 'peft>=0.14.0' \
    'datasets>=3.2.0,<4.0.0' einops loguru openpyxl 'scikit-image' scipy \
    SentencePiece tiktoken transformers_stream_generator cyclopts \
    'opencv-python-headless<=4.12.0.88' timm pyarrow pydantic tensorboard \
    xxhash imageio 'py-libnuma' GitPython
python -c "import xtuner; from xtuner.version import __version__; print('xtuner', __version__)"
```

输出结果类似如下（`xxx` 是版本号）：

```shell #test-result id="xtuner-install-binary" fuzzy='xxx'
xtuner xxx
```

### 方式二：源码安装

换成源码版（覆盖方式一的二进制安装）：

<!--
```shell #test-setup
uv pip uninstall xtuner -y
```
-->

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到最新 release tag，装 xtuner 本体 + 运行依赖，最后打印版本号验证：

```shell #test id="xtuner-install-source" load="upstream_ref>>ref"
[ -d xtuner ] || git clone --depth 1 --branch <ref> https://github.com/InternLM/xtuner.git
cd xtuner
uv pip install --no-deps -e .
uv pip install 'mmengine==0.10.6' 'transformers==4.48.0' 'peft>=0.14.0' \
    'datasets>=3.2.0,<4.0.0' einops loguru openpyxl 'scikit-image' scipy \
    SentencePiece tiktoken transformers_stream_generator cyclopts \
    'opencv-python-headless<=4.12.0.88' timm pyarrow pydantic tensorboard \
    xxhash imageio 'py-libnuma' GitPython
python -c "import xtuner; from xtuner.version import __version__; print('xtuner', __version__)"
```

\<ref> 是 xtuner 最新 release tag（在跑前从 https://github.com/InternLM/xtuner/releases 取）。

输出结果类似如下（`xxx` 是版本号）：

```shell #test-result id="xtuner-install-source" fuzzy='xxx'
xtuner xxx
```

## 安装验证

装好后一次性验证：顶层包 + CLI 入口能被解析，且 `xtuner.entry_point.MODES` 覆盖本文档用到的 `train` / `list-cfg` / `chat` 三个子命令：

```shell #test id="xtuner-import-check"
python -c "
import importlib.util as u
specs = {m: u.find_spec(m) for m in ['xtuner', 'xtuner.entry_point']}
for m, s in specs.items():
    print(m, 'ok' if s is not None else 'MISSING')
from xtuner.entry_point import MODES
print('modes_count:', len(MODES))
print('has_train:', 'train' in MODES)
print('has_list_cfg:', 'list-cfg' in MODES)
print('has_chat:', 'chat' in MODES)
"
```

输出结果如下：

```shell #test-result id="xtuner-import-check" fuzzy='xxx'
xtuner ok
xtuner.entry_point ok
modes_count: xxx
has_train: True
has_list_cfg: True
has_chat: True
```

## LLM 大模型微调

本文档的训练按 [xtuner legacy 快速上手模板](https://xtuner.readthedocs.io/zh-cn/latest/legacy/get_started/quickstart.html) 的顺序展开。

### 准备模型权重

本文档示例使用 **Qwen1.5-1.8B-Chat**——1.8B 参数 + TikToken BPE 分词（`tokenizer.json` 7 MB，无 sentencepiece），fp16 权重 ≈ 3.5 GB。

下载 Qwen1.5-1.8B-Chat 权重：

```shell #test-setup store="xtuner_weights_path"
# modelscope 把权重落到自己的 cache 目录结构里。
# 用 snapshot_download 的返回值（绝对路径）供「修改配置文件」「模型转换 + LoRA 合并」「与模型对话」步骤引用。
python -c "
import os
from modelscope import snapshot_download
path = snapshot_download('qwen/Qwen1.5-1.8B-Chat', cache_dir='./qwen')
print(os.path.abspath(path))
"
```

权重落盘校验——modelscope 的 cache 目录结构随版本变化，所以用 `find` 定位
`config.json` 而不是写死路径：

```shell #test id="xtuner-pull-weights"
ws=$(find ./qwen -name config.json -print -quit)
test -n "$ws" && test -f "$ws" && echo "weights_ok"
ls -la "$(dirname "$ws")" | head -1
```

输出结果类似：

```shell #test-result id="xtuner-pull-weights" fuzzy='xxx'
weights_ok
total xxx
```

权重落到 `./qwen` 下的 modelscope cache 目录。

### 准备微调数据集

Colorist 数据集：根据颜色描述给 16 进制颜色编码的指令微调集，几 MB：

```shell #test-setup
python -c "
import os, shutil
from modelscope import snapshot_download
path = snapshot_download('fanqiNO1/colors', repo_type='dataset', cache_dir='/tmp/xtuner_ms_cache')
target = './colors'
if os.path.isdir(target):
    shutil.rmtree(target)
os.makedirs(target, exist_ok=True)
for entry in os.listdir(path):
    src = os.path.join(path, entry)
    dst = os.path.join(target, entry)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
print('dataset at', target)
"
ls -la colors/ | head -1
```

验证数据集完整落盘——必需文件逐个检查，再列出目录实际内容：

```shell #test id="xtuner-pull-dataset"
# 三个必需文件逐个断言存在，缺了任何一个直接退出报错
for f in colors.json README.md train.jsonl; do
    test -f "colors/$f" || { echo "MISSING: colors/$f"; exit 1; }
done
# 列出目录的实际内容
ls colors/
```

输出结果是 `./colors/` 的实际目录内容：

```shell #test-result id="xtuner-pull-dataset" disable_fuzzy
README.md
colors.json
train.jsonl
```

> 数据集只有几 MB，下载几秒完成；重复执行时脚本会先清掉旧的 `./colors/` 再重建。

#### 把 Colorist 数据集转成 Qwen chat 模板要的 OpenAI 格式

xtuner 的 Qwen 自定义 cfg 用 `openai_map_fn`，要求每行 JSON 形如：

```json
{"messages": [
  {"role": "user", "content": "Tell me about the color #000000"},
  {"role": "assistant", "content": "Pure Black: ..."}
]}
```

但 Colorist 数据集原始格式是 `{color, description}`，需要先转换一下：

```shell #test-setup
mkdir -p ./colors_openai
python -c "
import json
src = './colors/train.jsonl'
dst = './colors_openai/train.jsonl'
n = 0
with open(src) as fin, open(dst, 'w') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        msg = {
            'messages': [
                {'role': 'user', 'content': f\"Tell me about the color {row['color']}\"},
                {'role': 'assistant', 'content': row['description']},
            ]
        }
        fout.write(json.dumps(msg, ensure_ascii=False) + '\n')
        n += 1
print(f'converted {n} rows -> {dst}')
"
```

验证转换结果——输出文件存在、行数与原始数据集逐行守恒、抽第一条验消息格式正确：

```shell #test id="xtuner-convert-colors"
test -f ./colors_openai/train.jsonl || { echo "MISSING: ./colors_openai/train.jsonl"; exit 1; }
# 行数守恒（与 pull-dataset 后的 ./colors/train.jsonl 逐行对比）
src_n=$(wc -l < ./colors/train.jsonl)
dst_n=$(wc -l < ./colors_openai/train.jsonl)
test "$src_n" = "$dst_n" || { echo "row count mismatch: src=$src_n dst=$dst_n"; exit 1; }
echo "converted ${dst_n} rows -> ./colors_openai/train.jsonl"
# 抽第一条做字面比对，验格式正确
head -1 ./colors_openai/train.jsonl | python -c "
import sys, json
row = json.loads(sys.stdin.read())
assert 'messages' in row, 'missing messages key'
assert isinstance(row['messages'], list) and len(row['messages']) == 2, 'expected 2 messages'
assert row['messages'][0]['role'] == 'user', 'first message role must be user'
assert row['messages'][1]['role'] == 'assistant', 'second message role must be assistant'
print('format_ok')
"
```

输出结果：

```shell #test-result id="xtuner-convert-colors" fuzzy='xxx'
converted xxx rows -> ./colors_openai/train.jsonl
format_ok
```

### 准备配置文件

XTuner 自带大量开箱即用的 config：

```shell #test id="xtuner-list-cfg"
# 直接调 Python API 而不是 `xtuner list-cfg` console script：CANN 的 set_env.sh 给
# PYTHONPATH 留了尾部空 entry，等于把 cwd 挂进 sys.path；从 clone 的父目录跑 `xtuner`
# 会把 clone 根目录（无 __init__.py）误判成 namespace package 而 ImportError。
python -c "
from xtuner.configs import cfgs_name_path
names = sorted(cfgs_name_path.keys())
print('lines:', len(names))
print('head_first:', names[0] if names else '')
print('qwen_1_8b_chat_count:', sum(1 for n in names if 'qwen1_5_1_8b_chat_qlora_custom_sft_e1' in n))
"
```

输出结果如下（`xxx` 是 cfg 总数 / 首个 cfg 名 / 匹配到的目标 cfg 个数）：

```shell #test-result id="xtuner-list-cfg" fuzzy='xxx'
lines: xxx
head_first: xxx
qwen_1_8b_chat_count: xxx
```

从 list-cfg 拷一份 Qwen1.5-1.8B-Chat qlora + custom sft 配置到本地（xtuner v0.2.0 的这个 cfg 名字固定为 `qwen1_5_1_8b_chat_qlora_custom_sft_e1`）：

```shell #test-setup store="xtuner_llm_cfg_path"
# 同样绕开 console_script wrapper，直接调 Python API 拷 cfg。
# `xtuner copy-cfg` 把 save_dir 当目录，文件实际写到 save_dir/<basename>_copy.py；
# 脚本最后打印拷出的 cfg 文件绝对路径——下一步「修改配置文件」的 patch 脚本要用它。
config_name='qwen1_5_1_8b_chat_qlora_custom_sft_e1'
python -c "
import os
import os.path as osp
import shutil
from xtuner.configs import cfgs_name_path
from xtuner.tools.copy_cfg import add_copy_suffix
config_path = cfgs_name_path['$config_name']
save_dir = '/tmp/xtuner_npu_llm_cfg.py'
save_path = osp.join(save_dir, add_copy_suffix(osp.basename(config_path)))
os.makedirs(save_dir, exist_ok=True)
shutil.copyfile(config_path, save_path)
print(save_path)
"
```

输出路径到下一节「修改配置文件」

### 修改配置文件

拷出来的 config 跟模板原版完全一致，按下面 4 处修改规则调整。

**占位符说明**：
- `<cfg>` = 上一节「准备配置文件」拷 cfg 那一步落到的文件绝对路径，典型值 `/tmp/xtuner_npu_llm_cfg.py/qwen1_5_1_8b_chat_qlora_custom_sft_e1_copy.py`。**本地手动跑**：自己跑 `xtuner copy-cfg qwen1_5_1_8b_chat_qlora_custom_sft_e1 /tmp/xtuner_npu_llm_cfg.py`，然后 `ls /tmp/xtuner_npu_llm_cfg.py/` 找 `_copy.py` 后缀的那个文件路径替换。
- `<weights_dir>` = 「准备模型权重」下权重那一步落到的 Qwen 权重绝对路径，典型值 `./qwen/models/qwen--Qwen1.5-1.8B-Chat/snapshots/master`（modelscope cache 布局，随版本变化）。**本地手动跑**：用前一步 `xtuner-pull-weights` 块里 `find ./qwen -name config.json -print -quit` 的 `dirname` 结果替换。

1. `pretrained_model_name_or_path`：替成本地真实权重路径（pull-weights 阶段落盘路径）
2. `data_files[0]`：替成转换后的 OpenAI 格式 jsonl 绝对路径
3. **strip `quantization_config` + `BitsAndBytesConfig` 导入**：QLoRA 路径需要 bnb，aarch64 NPU 上装不上，所以退化为 plain LoRA 走 fp16 base
4. **strip `train_cfg` 里的 `max_epochs`**：`xtuner train_cfg` 强制 `max_iters` 和 `max_epochs` 二选一，本文档用 `max_iters=5` 限制迭代数

```shell #test-setup load="xtuner_llm_cfg_path>>cfg" load="xtuner_weights_path>>weights_dir" store="xtuner_llm_cfg_path"
# 对 cfg 模板做 4 处 patch（用 Python str/re 比 sed 稳：cfg 全用双引号，sed 引号易踩坑）：
#   1) pretrained_model_name_or_path → 本地权重绝对路径
#   2) data_files[0]               → OpenAI 格式 jsonl 绝对路径
#   3) strip quantization_config   + BitsAndBytesConfig import（plain LoRA 路径）
#   4) strip train_cfg 里 max_epochs（TrainLoop 强制 max_iters / max_epochs 二选一）
python -c "
import re, os
path = '<cfg>'
weights_dir = '<weights_dir>'
with open(path) as f:
    text = f.read()

# patch 1: pretrained_model_name_or_path → 本地权重路径
text, n = re.subn(
    r'pretrained_model_name_or_path = \"Qwen/Qwen1\.5-1\.8B-Chat\"',
    f\"pretrained_model_name_or_path = {weights_dir!r}\",
    text,
)
assert n == 1, f'patch 1 applied {n} times (expected 1)'

# patch 2: data_files[0] → OpenAI jsonl 绝对路径
data_abs = os.path.abspath('./colors_openai/train.jsonl')
old = 'data_files = [\"/path/to/json/file.json\"]'
new = f'data_files = [{data_abs!r}]'
assert old in text, f'patch 2 source not found: {old!r}'
text = text.replace(old, new)

# patch 3: strip quantization_config block + BitsAndBytesConfig import
text = re.sub(
    r',\s*\n\s*quantization_config=dict\(\n(?:\s+[^\n]*,\n)+?\s*\),\n',
    '\n',
    text,
    count=1,
)
text = re.sub(
    r'(from transformers import [^\n]*?), BitsAndBytesConfig',
    r'\1',
    text,
    count=1,
)

# patch 4: train_cfg 去掉 max_epochs（TrainLoop 强制二选一；用 max_iters=5 限迭代数）
text, n = re.subn(
    r'train_cfg = dict\(type=TrainLoop, max_epochs=max_epochs\)',
    'train_cfg = dict(type=TrainLoop)',
    text,
)
assert n == 1, f'patch 4 applied {n} times (expected 1)'

with open(path, 'w') as f:
    f.write(text)
print(path)
"
```

<!-- # py_compile 验 cfg 是合法 Python + 4 处 patch 都生效（grep 关键串）。
# 不用 mmengine.config.Config.fromfile：它会执行 cfg 顶层 import 触发 torchvision::nms，
# NPU base image 的 torchvision 缺 C++ op 直接 RuntimeError。 -->
验证 patch 结果——cfg 能通过编译 + 4 处修改都已生效：

```shell #test id="xtuner-patch-cfg" load="xtuner_llm_cfg_path>>cfg" load="xtuner_weights_path>>weights_dir"
python -c "
import py_compile
py_compile.compile('<cfg>', doraise=True)
print('cfg_compiles_ok')
with open('<cfg>') as f:
    text = f.read()
weights_dir = '<weights_dir>'
import os
data_abs = os.path.abspath('./colors_openai/train.jsonl')
checks = [
    ('weights', f\"pretrained_model_name_or_path = '{weights_dir}'\"),
    ('data', f\"data_files = ['{data_abs}']\"),
]
for name, expected in checks:
    assert expected in text, f'missing patch ({name}): {expected!r}'
assert 'quantization_config' not in text
assert 'BitsAndBytesConfig' not in text
assert 'train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)' not in text
print('cfg_patch_ok')
print(f'weights= {weights_dir}')
print(f'data= {data_abs}')
"
```

输出结果类似：

```shell #test-result id="xtuner-patch-cfg" fuzzy='xxx'
cfg_compiles_ok
cfg_patch_ok
weights= xxx
data= xxx
```

> 这里不验 cfg 训出来的实际效果，那要等下面「启动微调」章节真跑。

### 启动微调

训练日志（loss、学习率）每次跑都不一样，没法写死预期值。本文档只跑 5 轮迭代验证整条训练链路（约 30 秒，~10 GB 峰值显存），不指望训出有意义结果。`EvaluateChatHook` 每轮打印 `Sample output:` 采样段，下面的验证命令检查 `.pth` 落盘 + 采样段格式。

#### 单卡

跑最小训练：

<!--
  plain LoRA 而非 QLoRA：aarch64 NPU 上没有可用的 bitsandbytes 装法（PyPI 无 aarch64
  wheel，source-build 各处报错）；Qwen1.5-1.8B fp16 ~3.5 GB，plain LoRA 在 32 GB NPU 上
  峰 RSS ≈ 10.3 GB，无需量化。

  cv2 / torchvision stub：NPU base image 的 cv2 缺 libxcb.so.1，torchvision 缺 C++ op
  （torch.ops.torchvision.* 直接 RuntimeError），而 mmengine / transformers.bloom 顶层
  import 就会撞上。stub 放 PYTHONPATH 最前，no-op 实现够 5 轮迭代的验证训练用；
  merge / chat 块复用同一组 stub。
-->

```shell #test-setup id="xtuner-train-smoke-setup" load="xtuner_llm_cfg_path>>cfg"
mkdir -p /tmp/cv2_stub/cv2
cat > /tmp/cv2_stub/cv2/__init__.py <<'PYEOF'
__version__ = "4.12.0"

def imread(*args, **kwargs):
    return None

def imwrite(*args, **kwargs):
    return True

def cvtColor(*args, **kwargs):
    return None

def resize(*args, **kwargs):
    return None

def setNumThreads(*args, **kwargs):
    return None
PYEOF

mkdir -p /tmp/torchvision_stub/torchvision/ops /tmp/torchvision_stub/torchvision/transforms
cat > /tmp/torchvision_stub/torchvision/__init__.py <<'PYEOF'
__version__ = "0.24.0"
PYEOF
cat > /tmp/torchvision_stub/torchvision/ops/__init__.py <<'PYEOF'
def nms(*args, **kwargs):
    return None
PYEOF
cat > /tmp/torchvision_stub/torchvision/transforms/__init__.py <<'PYEOF'
from enum import Enum

class InterpolationMode(Enum):
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BOX = "box"
    BILINEAR = "bilinear"
    HAMMING = "hamming"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"

def Compose(*args, **kwargs):
    return None

def ToTensor(*args, **kwargs):
    return None

def Resize(*args, **kwargs):
    return None

def CenterCrop(*args, **kwargs):
    return None

def Normalize(*args, **kwargs):
    return None
PYEOF
cat > /tmp/torchvision_stub/torchvision/transforms/v2.py <<'PYEOF'
from torchvision.transforms import functional
PYEOF
cat > /tmp/torchvision_stub/torchvision/transforms/functional.py <<'PYEOF'
from torchvision.transforms import InterpolationMode

def normalize(*args, **kwargs):
    return None

def pil_to_tensor(*args, **kwargs):
    return None

def to_tensor(*args, **kwargs):
    return None

def to_pil_image(*args, **kwargs):
    return None

def resize(*args, **kwargs):
    return None
PYEOF

# bnb 相关配置已在「修改配置文件」一步删干净，训练不用再处理。
cp <cfg> /tmp/xtuner_npu_smoke_single_cfg.py

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_NPU_USE_HCCL=1
# stub 必须放 PYTHONPATH 最前（前面 stub 优先于 site-packages 的坏 torchvision）。
export PYTHONPATH=/tmp/torchvision_stub:/tmp/cv2_stub${PYTHONPATH:+:$PYTHONPATH}
mkdir -p /tmp/xtuner_sft_llm_out_single
set -o pipefail

# 5 处 --cfg-options override：
#   train_cfg.max_iters=5                    限 5 iter 跑通就够，不训 full epoch
#   default_hooks.checkpoint.interval=1      每 iter 落盘，方便验 .pth
#   custom_hooks.1.every_n_iters=1           EvaluateChatHook 每 iter 打 Sample output
#   custom_hooks.1.evaluation_inputs=...     覆盖 cfg 默认 prompt 改成 color
#   train_dataset.max_length=256             colors 样本短，2048 太浪费
#   optim_wrapper.accumulative_counts=1      5 iter 不需要梯度累积
cd xtuner
python -c "
import sys
sys.argv = ['xtuner.tools.train',
            '/tmp/xtuner_npu_smoke_single_cfg.py',
            '--work-dir', '/tmp/xtuner_sft_llm_out_single',
            '--cfg-options',
            'train_cfg.max_iters=5',
            'default_hooks.checkpoint.interval=1',
            'custom_hooks.1.every_n_iters=1',
            'custom_hooks.1.evaluation_inputs=[Tell me about the color #000000, Tell me about the color #FF5733]',
            'train_dataset.max_length=256',
            'optim_wrapper.accumulative_counts=1']
import xtuner.tools  # noqa: F401  触发 xtuner.__init__.py 完整加载，避免 namespace package 误判
from xtuner.tools import train
train.main()
" 2>&1 | tee /tmp/xtuner_sft_llm_out_single/train.log
```

查 .pth 有没有落盘 + 训练日志里的 Sample output 段：

```shell #test id="xtuner-train-smoke"
ls -t /tmp/xtuner_sft_llm_out_single/*.pth 2>/dev/null | head -1
echo "---SAMPLE_OUTPUT---"
# EvaluateChatHook 每个 prompt 打一个 "Sample output:" 段（两个 evaluation_inputs 相邻成对）。
# awk 从第一个 "Sample output:" 行开始打印（跳过 mmengine 环境信息 dump 等 ~480 行前导日志），
# 到第 3 个段头（即下一轮 eval）截断——正好覆盖第一轮 eval 的两个 prompt 段。
awk '/Sample output:/{c++; if(c>2) exit} c>=1 {print}' /tmp/xtuner_sft_llm_out_single/train.log 2>/dev/null | sed -E 's/^[0-9]{2}\/[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} - mmengine - (INFO|WARNING|ERROR|DEBUG) - //' | head -25
```

输出结果如下（`...` 通配模型生成的具体内容——未训练模型的采样输出每次运行都不同，不能字面比对）：

```shell #test-result id="xtuner-train-smoke" fuzzy='...'
/tmp/xtuner_sft_llm_out_single/iter_5.pth
---SAMPLE_OUTPUT---
Sample output:
<|im_start|>user
Tellmeaboutthecolor#000000<|im_end|>
<|im_start|>assistant
...
Sample output:
<|im_start|>user
Tellmeaboutthecolor#FF5733<|im_end|>
<|im_start|>assistant
...
```

#### 多卡（双卡 DDP）

跑最小训练：

```shell #test-setup id="xtuner-train-smoke-multi-setup" load="xtuner_llm_cfg_path>>cfg"
# stub 复用「启动微调·单卡」准备命令写好的 /tmp/cv2_stub + /tmp/torchvision_stub
# （多卡走 DDP 也是同一 xtuner.tools.train 入口，import 链相同）。
# 单独跑本节前，先执行上面单卡的准备命令（写 stub 的那段）。
test -f /tmp/torchvision_stub/torchvision/ops/__init__.py || {
    echo "stubs missing: 先跑上面单卡的准备命令"; exit 1; }

cp <cfg> /tmp/xtuner_npu_smoke_multi_cfg.py

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_NPU_USE_HCCL=1
export PYTHONPATH=/tmp/torchvision_stub:/tmp/cv2_stub${PYTHONPATH:+:$PYTHONPATH}
mkdir -p /tmp/xtuner_sft_llm_out_multi
set -o pipefail

cd xtuner
NPROC_PER_NODE=2 python -c "
import sys
sys.argv = ['xtuner.tools.train',
            '/tmp/xtuner_npu_smoke_multi_cfg.py',
            '--work-dir', '/tmp/xtuner_sft_llm_out_multi',
            '--cfg-options',
            'train_cfg.max_iters=5',
            'default_hooks.checkpoint.interval=1',
            'custom_hooks.1.every_n_iters=1',
            'custom_hooks.1.evaluation_inputs=[Tell me about the color #000000, Tell me about the color #FF5733]',
            'train_dataset.max_length=256',
            'optim_wrapper.accumulative_counts=1']
import xtuner.tools  # noqa: F401
from xtuner.tools import train
train.main()
" 2>&1 | tee /tmp/xtuner_sft_llm_out_multi/train.log
```

查 .pth + Sample output：

```shell #test id="xtuner-train-smoke-multi"
ls -t /tmp/xtuner_sft_llm_out_multi/*.pth 2>/dev/null | head -1
echo "---SAMPLE_OUTPUT---"
# 跟单卡同样的 awk + sed：从第一个 "Sample output:" 段头开始打印，
# 第 3 个段头截断，覆盖第一轮 eval 的两个 prompt 段。
awk '/Sample output:/{c++; if(c>2) exit} c>=1 {print}' /tmp/xtuner_sft_llm_out_multi/train.log 2>/dev/null | sed -E 's/^[0-9]{2}\/[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} - mmengine - (INFO|WARNING|ERROR|DEBUG) - //' | head -25
```

输出结果如下：
```shell #test-result id="xtuner-train-smoke-multi" fuzzy='xxx' fuzzy='...'
/tmp/xtuner_sft_llm_out_multi/iter_xxx.pth
---SAMPLE_OUTPUT---
Sample output:
<|im_start|>user
Tellmeaboutthecolor#000000<|im_end|>
<|im_start|>assistant
...
Sample output:
<|im_start|>user
Tellmeaboutthecolor#FF5733<|im_end|>
<|im_start|>assistant
...
```

### 模型转换 + LoRA 合并

训练产物是 LoRA adapter 的 `.pth`（只含 adapter 参数），要跟纯 base 模型对话需要两步：`pth_to_hf` 把 `.pth` 转成 HuggingFace 格式（PEFT adapter），`merge` 把 adapter 合并回 base：

```shell #test-setup load="xtuner_llm_cfg_path>>cfg" load="xtuner_weights_path>>weights_dir"
# merge 入口也会触发 transformers.bloom → torchvision.transforms，复用「启动微调」准备命令建好的 stub。
# pth_to_hf 的第一个参数是 cfg 文件（「准备配置文件」一步拷出的 <cfg>），不是 /tmp/xtuner_npu_llm_cfg.py 目录；
# merge 的 LLM 参数用「准备模型权重」一步下载的 <weights_dir>。
export PYTHONPATH=/tmp/torchvision_stub:/tmp/cv2_stub${PYTHONPATH:+:$PYTHONPATH}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
src_pth=$(ls -t /tmp/xtuner_sft_llm_out_single/*.pth 2>/dev/null | head -1)
[ -n "$src_pth" ] || { echo "no .pth: 先跑上面的单卡训练"; exit 1; }
hf_dir="${src_pth%.pth}_hf"
merged_dir=/tmp/xtuner_sft_llm_out_single/merged
rm -rf "$hf_dir" "$merged_dir"
mkdir -p "$hf_dir" "$merged_dir"

# pth → hf（PEFT 格式：adapter_config.json + adapter_model.bin）
python -m xtuner.tools.model_converters.pth_to_hf \
    <cfg> \
    "$src_pth" \
    "$hf_dir"

# merge（PEFT adapter 合并回 base → sharded pytorch_model-*.bin，~3.5 GB）
python -m xtuner.tools.model_converters.merge \
    <weights_dir> \
    "$hf_dir" \
    "$merged_dir" \
    --max-shard-size 2GB
```

验合并产物落盘：

```shell #test id="xtuner-merge-verify"
ls -t /tmp/xtuner_sft_llm_out_single/merged/*.bin 2>/dev/null | head -3
```

输出结果如下（`xxx` 通配 shard 编号与个数；1.8B fp16 ≈ 3.5 GB 按 2GB 分片）：

```shell #test-result id="xtuner-merge-verify" fuzzy='xxx'
/tmp/xtuner_sft_llm_out_single/merged/pytorch_modelxxx.bin
```

### 与模型对话

合并完权重后，用 `xtuner chat` 跟模型对话（本文档用等价的 `python -m xtuner.tools.chat` 入口）：`--prompt-template qwen_chat` 套 Qwen 对话模板，`--system-template colorist` 加上颜色助手 system prompt。

先跟合并后的模型对话（用上一步合并出的 1.8B merged/ 目录）：

```shell #test id="xtuner-chat-merged"
export PYTHONPATH=/tmp/torchvision_stub:/tmp/cv2_stub${PYTHONPATH:+:$PYTHONPATH}
echo -e "Tell me about the color #66ccff\n\nEXIT\n" | \
python -m xtuner.tools.chat /tmp/xtuner_sft_llm_out_single/merged \
    --prompt-template qwen_chat \
    --system-template colorist \
    --no-streamer \
    --max-new-tokens 64 2>&1 | grep -E "^Load LLM from|Log: Exit!"
```

输出结果如下：

```shell #test-result id="xtuner-chat-merged"
Load LLM from /tmp/xtuner_sft_llm_out_single/merged
...Log: Exit!
```

不合并、只跟 LLM + LoRA adapter 直接对话（adapter 版）：

```shell #test id="xtuner-chat-adapter" load="xtuner_weights_path>>weights_dir"
# 不合并、只跟 base + LoRA adapter 直接对话：--adapter 指向上一步 pth_to_hf 转出的 iter_*_hf 目录，
# base 模型用「准备模型权重」一步下载的 <weights_dir>（modelscope cache 目录结构，不能写死字面路径）。
# stub PYTHONPATH 和空行提交输入的原因同上一条 chat 命令。
export PYTHONPATH=/tmp/torchvision_stub:/tmp/cv2_stub${PYTHONPATH:+:$PYTHONPATH}
hf_dir=$(ls -td /tmp/xtuner_sft_llm_out_single/iter_*_hf 2>/dev/null | head -1)
[ -n "$hf_dir" ] || { echo "no iter_*_hf: 先跑上面的模型转换"; exit 1; }
echo -e "Tell me about the color #66ccff\n\nEXIT\n" | \
python -m xtuner.tools.chat <weights_dir> \
    --adapter "$hf_dir" \
    --prompt-template qwen_chat \
    --system-template colorist \
    --no-streamer \
    --max-new-tokens 64 2>&1 | grep -E "^Load LLM from|^Load adapter from|Log: Exit!"
```

输出结果如下：

```shell #test-result id="xtuner-chat-adapter" load="xtuner_weights_path>>weights_dir" fuzzy='xxx' fuzzy='...'
Load LLM from <weights_dir>
Load adapter from /tmp/xtuner_sft_llm_out_single/iter_xxx_hf
...Log: Exit!
```
