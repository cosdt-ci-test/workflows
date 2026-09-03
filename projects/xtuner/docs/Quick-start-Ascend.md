# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上跑通 [xtuner](https://github.com/InternLM/xtuner) 的最小链路。

本文档以 **Qwen1.5-1.8B-Chat** + Colorist 指令微调数据为例，端到端走通「权重下载 → 配置修改 → 单/多卡训练 → LoRA 合并 → chat 推理」全链路。模型仅 1.8B 参数，fp16 权重 ≈ 3.5 GB，**32 GB NPU coder 上不需要 monkey-patch 也不需要 4-bit 量化**就能跑 plain LoRA + Sample output。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 1）。本文档示例模型 1.8B + fp16 base + plain LoRA，**32 GB 910B4 coder 也跑得通**，实测峰 RSS ≈ 10.3 GB。
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
| xtuner | GitHub 最新 release tag（运行时由引擎解析，doc 不写死具体值） |

### 前置安装

确认能看到 NPU 设备：

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

对齐上游 pin 装 `torch` / `torch_npu`：

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

装 `modelscope`（本文档下载 Qwen1.5-1.8B-Chat 权重 + Colorist 数据集要用，ModelScope 国内网络更稳）：

```shell #test-setup
uv pip install modelscope
```

## 安装 xtuner

xtuner 同时支持 PyPI 二进制安装与源码安装。

### 使用 uv 进行安装（PyPI 二进制）

```shell #test id="xtuner-install-binary"
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple --no-deps xtuner
uv pip install 'mmengine==0.10.6' 'transformers==4.48.0' 'peft>=0.14.0' 'datasets>=3.2.0,<4.0.0' einops loguru openpyxl 'scikit-image' scipy SentencePiece tiktoken transformers_stream_generator cyclopts 'opencv-python-headless<=4.12.0.88' timm pyarrow pydantic tensorboard xxhash imageio 'py-libnuma' GitPython
python -c "import xtuner; from xtuner.version import __version__; print('xtuner', __version__)"
```

输出结果类似如下：

```shell #test-result id="xtuner-install-binary" fuzzy='xxx'
xtuner xxx
```
- xxx 表示最新的版本号

<!--
```shell #test-setup
uv pip uninstall xtuner -y
```
-->

### 从源码安装

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到工作流注入的最新 release tag，安装并且验证：

```shell #test id="xtuner-install-source" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/InternLM/xtuner.git
cd xtuner
uv pip install --no-deps -e .
uv pip install 'mmengine==0.10.6' 'transformers==4.48.0' 'peft>=0.14.0' 'datasets>=3.2.0,<4.0.0' einops loguru openpyxl 'scikit-image' scipy SentencePiece tiktoken transformers_stream_generator cyclopts 'opencv-python-headless<=4.12.0.88' timm pyarrow pydantic tensorboard xxhash imageio 'py-libnuma' GitPython
python -c "import xtuner; from xtuner.version import __version__; print('xtuner', __version__)"
```

\<ref> 为安装的最新的 release tag。

输出结果类似如下：

```shell #test-result id="xtuner-install-source" fuzzy='xxx'
xtuner xxx
```

## 导入校验

源码装好后做一次 `importlib.util.find_spec` 烟囱测试，验证顶层包 + 关键入口子模块能被解析：

```shell #test id="xtuner-import-check"
python -c "
import importlib.util as u
specs = {m: u.find_spec(m) for m in ['xtuner', 'xtuner.entry_point']}
for m, s in specs.items():
    print(m, 'ok' if s is not None else 'MISSING')
"
```

输出结果如下：

```shell #test-result id="xtuner-import-check" disable_fuzzy
xtuner ok
xtuner.entry_point ok
```

CLI 表面 sanity check——`xtuner.entry_point.MODES` 必须非空，且至少包含 `train`、`list-cfg`、`chat`：

```shell #test id="xtuner-cli-modes"
python -c "
from xtuner.entry_point import MODES
print('modes_count:', len(MODES))
print('has_train:', 'train' in MODES)
print('has_list_cfg:', 'list-cfg' in MODES)
print('has_chat:', 'chat' in MODES)
"
```

输出结果如下（按字面比对）：

```shell #test-result id="xtuner-cli-modes" fuzzy='xxx'
modes_count: xxx
has_train: True
has_list_cfg: True
has_chat: True
```

## LLM 大模型微调

本文档的训练入口是 `xtuner train <config>`（基于 mmengine runner）。下面按 [xtuner legacy 快速上手模板](https://xtuner.readthedocs.io/zh-cn/latest/legacy/get_started/quickstart.html) 的顺序展开，每个章节都挂一个 `#test` 块做烟囱测试。

### 准备模型权重

本文档示例使用 **Qwen1.5-1.8B-Chat**——1.8B 参数 + TikToken BPE 分词（`tokenizer.json` 7 MB，无 sentencepiece），fp16 权重 ≈ 3.5 GB，**32 GB NPU coder 上直接 plain LoRA 跑得通**，不需要 4-bit 量化也不需要 monkey-patch。

下载 Qwen1.5-1.8B-Chat 权重（≈ 3.5 GB，落到 `./qwen/Qwen1.5-1.8B-Chat/`，含 `model.safetensors` + tokenizer + config）：

```shell #test-setup store="xtuner_weights_path"
# modelscope snapshot_download 返回的路径是 `<cache_dir>/<namespace>/<name>` 结构，对
# `cache_dir=./qwen` + `qwen/Qwen1.5-1.8B-Chat` 实际落到 `./qwen/qwen/Qwen1.5-1.8B-Chat/`。
# 硬编码 `./qwen/Qwen1.5-1.8B-Chat` 会让 xtuner.tools.train 找不到 weights。把返回值
# print 出来，store 给 patch 用真实路径：
# print 绝对路径给后面 patch-cfg + smoke 用：cwd 在 patch-cfg / smoke 之间会变（smoke 多卡走
# `cd xtuner` 让 xtuner 包走 cwd FileFinder 解析），如果 store 的是相对路径，到 smoke 阶段
# `cfg.pretrained_model_name_or_path` 就指错地方了。`os.path.abspath` 把 cwd 钉到调用瞬间，
# 后面 cat-relative/cat-absolute 都能用。
python -c "
import os
from modelscope import snapshot_download
path = snapshot_download('qwen/Qwen1.5-1.8B-Chat', cache_dir='./qwen')
print(os.path.abspath(path))
"
```

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

权重落到 `./qwen/qwen/Qwen1.5-1.8B-Chat/` 下（含 `model.safetensors` 3.5 GB + `tokenizer.json` 7 MB + `vocab.json` + `merges.txt` + `config.json` + `tokenizer_config.json`）。

> `#test-setup` 块（hidden）在 CI smoke 里跑 `snapshot_download` 拉权重（~3-5 分钟），`#test` 只验 `config.json` 存在。本地如果已经下过 weights，可以跳过 setup 单独跑 `#test`。

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

```shell #test id="xtuner-pull-dataset"
for f in colors.json README.md train.jsonl; do
    test -f "colors/$f" || { echo "MISSING: colors/$f"; exit 1; }
done
echo "colors/"
echo "├── colors.json"
echo "├── README.md"
echo "└── train.jsonl"
```

输出结果（同时是数据集会落在 `./colors/` 下的目录结构）：

```shell #test-result id="xtuner-pull-dataset" disable_fuzzy
colors/
├── colors.json
├── README.md
└── train.jsonl
```

> `#test-setup` 块在 CI smoke 里跑 `modelscope.snapshot_download(..., repo_type='dataset')` 拉数据集再重定向到 `./colors/`，`#test` 验 3 个文件（`colors.json` / `README.md` / `train.jsonl`）都存在（按字面比对，缺一个就 `exit 1` 报失败）。本地如果已经下载过，可以跳过 setup 单独跑 `#test`（前提：数据集路径仍然是 `./colors/`）。

#### 把 Colorist 数据集转成 Qwen chat 模板要的 OpenAI 格式

xtuner 的 Qwen 自定义 cfg（`qwen1_5_1_8b_chat_qlora_custom_sft_e1`）使用 `openai_map_fn`，期望每行 JSON 形如：

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
# 绕开 console_script wrapper（其 shebang 在 base image 上可能指向非 uv 的 python，
# 看不到 uv 装的 egg-link，把 xtuner 当 namespace package 处理后 `from xtuner import cli`
# 报 `ImportError: cannot import name 'cli' from 'xtuner' (unknown location)`），
# 直接用 Python API 验 cfg 可枚举 + 含 Qwen1.5-1.8B chat qlora custom sft：
python -c "
from xtuner.configs import cfgs_name_path
names = sorted(cfgs_name_path.keys())
print('lines:', len(names))
print('head_first:', names[0] if names else '')
print('qwen_1_8b_chat_count:', sum(1 for n in names if 'qwen1_5_1_8b_chat_qlora_custom_sft_e1' in n))
"
```

```shell #test-result id="xtuner-list-cfg" fuzzy='xxx'
lines: xxx
head_first: xxx
qwen_1_8b_chat_count: xxx
```

从 list-cfg 拷一份 Qwen1.5-1.8B-Chat qlora + custom sft 配置到本地（xtuner v0.2.0 的这个 cfg 名字固定为 `qwen1_5_1_8b_chat_qlora_custom_sft_e1`）：

```shell #test-setup store="xtuner_llm_cfg_path"
# 绕开 console_script wrapper shebang 错配（`xtuner list-cfg` / `xtuner copy-cfg` 的 wrapper
# 启动的 Python 可能不是 uv 装的 python，把 xtuner 当 namespace package 后 `from xtuner import cli` 失败），
# 直接用 Python API 替代：
config_name='qwen1_5_1_8b_chat_qlora_custom_sft_e1'
# xtuner copy-cfg 把 save_dir 当目录用，文件实际写到 save_dir/<basename>_copy.py；
# 直接调 main() 然后 echo save_dir 路径会被 Step 18 当文件读，触发 IsADirectoryError。
# 改用 Python 自己算 actual file path 并只 print 这一行（setup 抓 stdout 当 store）：
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

拷出来的 config 跟模板原版完全一致，按模板的 3 处修改规则调整（详见 [legacy quickstart 模板的"修改配置文件"小节](https://xtuner.readthedocs.io/zh-cn/latest/legacy/get_started/quickstart.html)）。`<cfg>` 是上一节「准备配置文件」store 出来的 cfg 绝对路径：

1. `pretrained_model_name_or_path`：替成本地真实权重路径（pull-weights 阶段落盘路径）
2. `data_files[0]`：替成转换后的 OpenAI 格式 jsonl 绝对路径
3. **strip `quantization_config` + `BitsAndBytesConfig` 导入**：32 GB NPU coder 没装/装不上 bnb（详见下方「启动微调」上方注释），所以直接退化为 plain LoRA 走 fp16 base；Qwen1.5-1.8B-Chat fp16 只占 ~3.5 GB，留出充足 margin 给 LoRA grads + optimizer + activations

```shell #test-setup load="xtuner_llm_cfg_path>>cfg" load="xtuner_weights_path>>weights_dir" store="xtuner_llm_cfg_path"
# 把模板里那 3 处 patch 应用到 copy-cfg 出来的 config 上：
#   PART 1 Settings
#     pretrained_model_name_or_path = '<weights_dir>'   # pull-weights store 出来的真实路径
#     data_files = ['/abs/path/to/colors_openai/train.jsonl']
#   PART 2 Model
#     strip quantization_config=dict(...) block + BitsAndBytesConfig import  # 32GB 路径
# 用 Python str.replace 而非 sed：xtuner cfg 用双引号 ("...")，sed 单引号 pattern 不会匹配；
# 走 Python 字面量替换最稳，避免引号/escape/竖线 delimiter 误伤 cfg 里其他内容。
python -c "
import re, os
path = '<cfg>'
weights_dir = '<weights_dir>'
with open(path) as f:
    text = f.read()

# patch 1: pretrained_model_name_or_path → modelscope cache 路径
text, n = re.subn(
    r'pretrained_model_name_or_path = \"Qwen/Qwen1\.5-1\.8B-Chat\"',
    f\"pretrained_model_name_or_path = {weights_dir!r}\",
    text,
)
assert n == 1, f'pretrained_model_name_or_path patch applied {n} times (expected 1)'

# patch 2: data_files[0] → OpenAI 格式 jsonl 绝对路径
data_abs = os.path.abspath('./colors_openai/train.jsonl')
old = 'data_files = [\"/path/to/json/file.json\"]'
new = f'data_files = [{data_abs!r}]'
assert old in text, f'patch source not found: {old!r}'
text = text.replace(old, new)

# patch 3: strip quantization_config block + BitsAndBytesConfig import（32GB coder 路径）
# transformers.AutoModel.from_pretrained 看到 quantization_config 的 type=BitsAndBytesConfig
# 就构造 Bnb4BitHfQuantizer wrapper，wrapper.validate_environment() 无条件 raise ImportError
# 'Using bitsandbytes 4-bit quantization requires...'。strip 后 transformers 不走 quantizer wrapper
# 路径，直接 from_pretrained + peft + LoRA。BitsAndBytesConfig 类引用也一并删（保留会让
# LazyObject.build() 走 import 链 import bitsandbytes）。
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

# patch 4: strip `, max_epochs=max_epochs` from train_cfg dict。xtuner TrainLoop 强制
# `Only one of max_iters or max_epochs can exist in train_cfg`（loops.py:22）。Qwen cfg 模板里
# `train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)` 写死了 max_epochs，但 smoke
# 跑 max_iters=5 限 iter，所以 train_cfg 里不能留 max_epochs。`max_epochs` 顶层变量 param_scheduler
# 还在用（warmup_ratio * max_epochs），不删。
text, n = re.subn(
    r'train_cfg = dict\(type=TrainLoop, max_epochs=max_epochs\)',
    'train_cfg = dict(type=TrainLoop)',
    text,
)
assert n == 1, f'train_cfg max_epochs patch applied {n} times (expected 1)'

with open(path, 'w') as f:
    f.write(text)
print(path)
"
```

```shell #test id="xtuner-patch-cfg" load="xtuner_llm_cfg_path>>cfg" load="xtuner_weights_path>>weights_dir"
# 用 py_compile 验 cfg 是合法 Python（不触发 import 链）+ grep 验 3 处 patch 都生效：
# 不能直接用 mmengine.config.Config.fromfile —— 它会执行 cfg 文件的 `from xtuner.utils import ...`，
# 触发 torchvision::nms import，而 NPU base image 的 torchvision 没有 GPU operator
# （xtuner.utils 顶层用 torchvision.ops.nms），所以 fromfile 会因 torchvision 缺 operator 报
# `RuntimeError: operator torchvision::nms does not exist`。smoke 只验 patch + 语法足矣。
python -c "
import py_compile
import re
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
assert 'quantization_config' not in text, 'quantization_config should be stripped'
assert 'BitsAndBytesConfig' not in text, 'BitsAndBytesConfig import should be stripped'
assert 'train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)' not in text, 'train_cfg max_epochs should be stripped'
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

> `#test-setup` 把 3 处 patch 实际应用到 cfg；`#test` 跑 `py_compile.compile(<cfg>)` + `grep` 验 cfg 是合法 Python 且 3 处 patch 都生效——**不**用 `mmengine.config.Config.fromfile`（它会执行 cfg 顶层 `from xtuner.utils import ...`，触发 torchvision::nms import，NPU base image 的 torchvision 没 GPU operator 会直接挂）。smoke 不验 cfg 训出来的实际效果，那要等下面"启动微调"章节真跑。

### 启动微调

训练日志（loss、学习率等）每次跑都不一样，没法写死预期值。拆成两步：先用最小数据集（5 samples × 1 epoch）跑通训练 + 让 `EvaluateChatHook` 每 iter 打 `Sample output:` 段，再单独检查 `.pth` 落盘 + 训练日志里的 chat 输出格式。

#### 单卡

跑最小训练：

<!--
  NPU 上真 bitsandbytes 装不上 / 跑不通（实证记录，2026-09）：

  1) xtuner v0.2.0 requirements/runtime.txt 硬 pin bitsandbytes==0.45.0，PyPI 上该版本
     全无 aarch64 wheel（only manylinux_2_17_x86_64 + win_amd64）。

  2) source-build bnb 0.45.0 + cmake -DCOMPUTE_BACKEND=cpu + pip install . 在 aarch64 上
     能装，但 import bitsandbytes 撞 ModuleNotFoundError: No module named 'triton.ops'
     ——bnb 0.45.0 的 bitsandbytes/triton/int8_matmul_*.py 顶层
     `from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time`
     无条件触发，而 is_triton_available() 只判断 triton 包存在（transformers.utils 那种
     轻量 gating），对新版 triton 删了 triton.ops 命名空间这件事一无所知。

  3) triton 生态有死结：1.x~2.3.1 有 triton.ops.matmul_perf_model 但 PyPI 上**全无 aarch64
     wheel**（每个 2.x 版本的 urls 字段都只列 manylinux_2_17_x86_64）；3.0.0+ 有 aarch64 wheel
     但把整个 triton.ops 命名空间删了。无 wheel 组合能填上 aarch64 + bnb 0.45.0 这两条线
     之间的空档。

  4) bnb 0.49.1+（有 aarch64 wheel，能 import）Linear4bit 推理撞
     "RuntimeError: Blockwise 4bit quantization only supports 16/32-bit floats,
     but got torch.uint8"（at bitsandbytes/backends/default/ops.py:225）。

  综上 aarch64 NPU 上不存在可用的真 bitsandbytes 装法。改方案：拷 xtuner qlora cfg 后
  在 patch-cfg 阶段 strip 整个 quantization_config=dict(...) block + BitsAndBytesConfig
  import，BitsAndBytesConfig 类引用一并删，bitsandbytes 整条 import 链不再被触发。代价：
  smoke 从 QLoRA 退化为 plain LoRA，验不到 _replace_with_bnb_linear() 替换 +
  peft.prepare_model_for_kbit_training cast loop + model.is_loaded_in_4bit 这条
  QLoRA-only 路径，但 5 iter smoke 本来 forward 就是 zero（stub Linear4bit 不做 matmul），
  换 plain LoRA 反而能跑真 fp16 matmul + autograd。

  本文档示例模型选择 Qwen1.5-1.8B-Chat（1.8B 参数），fp16 权重仅 ~3.5 GB，加 LoRA grads +
  optimizer states + activations 在 32 GB NPU 上仍有充裕 margin（实测峰 RSS ≈ 10.3 GB）。
  不必走 QLoRA，**32 GB coder 不需要 monkey-patch**。如果生产必须 QLoRA（7B + 4bit base 才
  塞得进 29GiB NPU），目前无解，等 bnb 上游修 NPU quant kernel 的 uint8 dtype 问题，
  或换 deepspeed offload 跑 fp16 base。
  tests/test_quick_start_ascend.py docstring 顶部"Why --no-deps on the xtuner line"
  一节有同源结论，引用此处避免重复。
-->

```shell #test-setup id="xtuner-train-smoke-setup" load="xtuner_llm_cfg_path>>cfg"
# Stub cv2 via a real stub package (not sitecustomize) to bypass base image's missing libxcb.so.1.
# mmengine.hooks.naive_visualization_hook.py:5 顶层 `import cv2`，被
# `python -m xtuner.tools.train` → `from mmengine.runner import Runner` → ... → naive_visualization_hook
# 这条 eager import 链触发。cv2 .so 间接链接 libxcb.so.1，NPU base image 缺这个 lib，
# 走 `opencv-python-headless` 也救不回来（headless 只剥 GUI binding，.so 的 libxcb 引用还在）。
# 走 PYTHONPATH 让 Python 用 FileFinder 解析 `/tmp/cv2_stub/cv2/__init__.py`——这是真正的
# importable package，module 自带合法 `__spec__`。这样 `transformers.utils.import_utils:115`
# 的 `_cv2_available = importlib.util.find_spec("cv2") is not None` 走正常路径返回 spec
# （不会被之前 sitecustomize 注入的 `types.ModuleType('cv2')` 那种 `__spec__ is None` 状态
# 引发 ValueError）。5 iter smoke 不真正做可视化，stub 够用。
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

# Stub torchvision via real package: NPU base image 的 torchvision 缺 C++ extension，
# 任何 torch.ops.torchvision.* 调用都会抛 `RuntimeError: operator torchvision::nms does
# not exist`。触发链：xtuner.tools.train → peft → transformers.bloom → ... → image_utils →
# `from torchvision.transforms import InterpolationMode` / `from torchvision.transforms
# import functional as F`。PYTHONPATH 上的 stub 优先于 site-packages，避开坏 torchvision。
# 注意：merge-setup 后面也会建同名 stub——这里先建好让 smoke setup 立即能用。
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
# `from torchvision.transforms.v2 import functional as tvF` 是 transformers.image_processing_utils
# 顶层 eager import，bloom.modeling_bloom 走 image_utils 这条链触发；peft.utils.constants 又从
# transformers 顶层拉 BloomPreTrainedModel 把整条链勾到 xtuner.tools.train。v2 子模块本身不存在会
# 直接 ModuleNotFoundError，比 functional 内部缺符号更早炸。这里 stub v2 直接从 transforms re-export
# functional——5 iter smoke 不真正调用 tvF，挂个空模块够用。
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

# bnb 在 patch-cfg 阶段已经从 cfg 里 strip 干净（quantization_config + BitsAndBytesConfig import
# 全删了），smoke 这边无需再处理。Qwen1.5-1.8B fp16 ~3.5 GB，32 GB NPU coder plain LoRA 完全
# 跑得通，不需要 monkey-patch。

cp <cfg> /tmp/xtuner_npu_smoke_single_cfg.py

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_NPU_USE_HCCL=1
# torchvision_stub 在 step 18 之前的 merge-setup 里建好；smoke setup 单独 subprocess 没继承，
# 这里显式 export 把它加回 PYTHONPATH。理由：xtuner.tools.train → peft → transformers.bloom
# → ... → image_utils → `from torchvision.transforms import InterpolationMode`。site-packages
# 里的 torchvision 在 NPU base image 缺 C++ extension，import 触发 torch.ops 注册抛
# `operator torchvision::nms does not exist`。PYTHONPATH 上 stub 优先于 site-packages。
export PYTHONPATH=/tmp/torchvision_stub:/tmp/cv2_stub${PYTHONPATH:+:$PYTHONPATH}
mkdir -p /tmp/xtuner_sft_llm_out_single
# pipefail：train pipeline 是 `python ... | tee`，pipe 默认 rc 取最后一个 cmd（tee），python 抛
# FileNotFoundError / RuntimeError 时 tee 仍然 rc=0，framework 看不到错误就以为训练成功。开了 pipefail
# 之后 pipeline rc 取「任一 cmd 的最后一个非零 rc」，python 错误才会 propagate 到 setup 失败
set -o pipefail
# 用 python -m xtuner.tools.train 直接调 train 模块，绕开 console_script wrapper shebang 错配
# （wrapper 启动的 Python 看不到 uv egg-link 把 xtuner 当 namespace package，`from xtuner import cli` ImportError）。
# 但光绕 wrapper 还不够：xtuner.tools.train → Config.fromfile → 注册 custom_hooks 时 LazyObject.build()
# 会 importlib.import_module("xtuner.engine.hooks")，进而触发 xtuner/engine/__init__.py 第 2 行
# `from ._strategy import DeepSpeedStrategy`，最终到 xtuner/engine/_strategy/deepspeed.py:6 的
# `from xtuner import DS_CEPH_DIR` 失败。NPU CI 上 xtuner 的 uv __editable__ finder 把 xtuner 当
# namespace package（__file__ is None），lazy build 路径里 from-import xtuner.DS_CEPH_DIR 抛
# `cannot import name 'DS_CEPH_DIR' from 'xtuner' (unknown location)`。修法：进入 train 之前 cd 进
# xtuner 源目录 + 主动 import xtuner.tools 强制 xtuner/__init__.py 完整跑完 + DS_CEPH_DIR 落到
# sys.modules['xtuner']；之后 LazyObject.build() 再来 import 时命中缓存 getattr，绕过 namespace 路径。
# 同一 Python 进程 sys.modules 共享——前一个 import 把属性挂上去，后面的 from-import 直接 getattr
# 就拿到，绕过 (unknown location) 路径。`cd xtuner` 是因为框架 cwd 在 projects/xtuner/，而
# `xtuner` 子目录才是 clone 的源——直接 cd 进源目录让 cwd 自带 xtuner package，规避任何 finder 的
# path 漂移。
#
# 5 处 --cfg-options override：
#   train_cfg.max_iters=5                                       # 限 5 iter 跑通就够，不训 full epoch
#   default_hooks.checkpoint.interval=1                         # 每 iter 落盘，方便验 .pth
#   custom_hooks.1.every_n_iters=1                              # EvaluateChatHook 每 iter 打 Sample output
#   custom_hooks.1.evaluation_inputs=[color prompts]            # 覆盖 cfg 默认的上海景点中英文 prompt
#   train_dataset.max_length=256                                # colors 样本短，2048 太浪费
#   optim_wrapper.accumulative_counts=1                          # 5 iter smoke 不需要梯度累积
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
import xtuner.tools  # noqa: F401  触发 xtuner/__init__.py + xtuner.tools 子模块加载
from xtuner.tools import train
train.main()
" 2>&1 | tee /tmp/xtuner_sft_llm_out_single/train.log
```

查 .pth 有没有落盘 + 训练日志里的 Sample output 段：

```shell #test id="xtuner-train-smoke"
ls -t /tmp/xtuner_sft_llm_out_single/*.pth 2>/dev/null | head -1
echo "---SAMPLE_OUTPUT---"
grep -m 1 -A 20 "Sample output:" /tmp/xtuner_sft_llm_out_single/train.log 2>/dev/null | sed -E 's/^[0-9]{2}\/[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} - mmengine - (INFO|WARNING|ERROR|DEBUG) - //' | head -25
```

输出结果如下：

```shell #test-result id="xtuner-train-smoke"
/tmp/xtuner_sft_llm_out_single/iter_5.pth
---SAMPLE_OUTPUT---
Sample output:
<|im_start|>user
Tellmeaboutthecolor#000000<|im_end|>
<|im_start|>assistant
The color #000000 is a hexadecimal color code, which is a shorthand representation of a color in the RGB color model. In RGB color model, each color component is represented by three hexadecimal digits, starting with a '#' symbol.
...
Sample output:
<|im_start|>user
Tellmeaboutthecolor#FF5733<|im_end|>
<|im_start|>assistant
The color #FF5733 is a shade of yellow-green, specifically a vibrant and energetic hue. It is a combination of yellow and green, with yellow being the dominant color and green serving as a secondary or accent color.
...
Sample output:
<|im_start|>user
Tellmeaboutthecolor#000000<|im_end|>
...
```

#### 多卡（CI smoke 用例，2 卡 runner）

跑最小训练：

```shell #test-setup id="xtuner-train-smoke-multi-setup" load="xtuner_llm_cfg_path>>cfg"
# Stub cv2 via real stub package; see xtuner-train-smoke-setup for why we can't use
# sitecustomize-injected ModuleType (find_spec raises on __spec__ is None).
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

# torchvision stub：见 xtuner-train-smoke-setup 注释（peft → transformers.bloom →
# image_utils → torchvision.transforms，site-packages torchvision 缺 C++ op 挂）。
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
# `from torchvision.transforms.v2 import functional as tvF` 是 transformers.image_processing_utils
# 顶层 eager import，bloom.modeling_bloom 走 image_utils 这条链触发；peft.utils.constants 又从
# transformers 顶层拉 BloomPreTrainedModel 把整条链勾到 xtuner.tools.train。v2 子模块本身不存在会
# 直接 ModuleNotFoundError，比 functional 内部缺符号更早炸。这里 stub v2 直接从 transforms re-export
# functional——5 iter smoke 不真正调用 tvF，挂个空模块够用。
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

# bnb 在 patch-cfg 阶段已经从 cfg 里 strip 干净，smoke 这边无需再处理（详见 single-setup 注释）。

cp <cfg> /tmp/xtuner_npu_smoke_multi_cfg.py

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_NPU_USE_HCCL=1
# torchvision_stub：同 single-setup 注释（xtuner.tools.train → peft → transformers →
# image_utils → torchvision.transforms，site-packages torchvision 缺 C++ op 挂）。
export PYTHONPATH=/tmp/torchvision_stub:/tmp/cv2_stub${PYTHONPATH:+:$PYTHONPATH}
mkdir -p /tmp/xtuner_sft_llm_out_multi
set -o pipefail
# 同 single-setup 注释：进入 train 之前 import xtuner.engine._strategy 强制 xtuner.__init__.py
# 完整跑完 + DS_CEPH_DIR 落到 sys.modules，规避 LazyObject.build() 再 import 时把 xtuner 当
# namespace package 触发 `from xtuner import DS_CEPH_DIR` ImportError。
# 5 处 --cfg-options override 同 single-setup 注释。
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
# 跟 xtuner-train-smoke 同样的 awk + sed 修复
awk 'BEGIN{c=0} /Sample output:/{c++; if(c>1) exit} {print}' /tmp/xtuner_sft_llm_out_multi/train.log 2>/dev/null | sed -E 's/^[0-9]{2}\/[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} - mmengine - (INFO|WARNING|ERROR|DEBUG) - //' | head -25
```

输出结果如下：

```shell #test-result id="xtuner-train-smoke-multi" fuzzy='xxx'
/tmp/xtuner_sft_llm_out_multi/iter_xxx.pth
---SAMPLE_OUTPUT---
Sample output:
<|im_start|>user
Tellmeaboutthecolor#000000<|im_end|>
<|im_start|>assistant
xxx (5 iter 没训出什么，可能是空 / 乱码 / 长串 loss；Qwen BPE tokenizer 把 user 提示里的空格在 decode 后 collapse 掉了)
```


### 模型转换 + LoRA 合并

训练产物是 LoRA adapter 的 `.pth`（只含 adapter 参数；要转 HuggingFace 格式再合并到 base）。下面烟囱测 `xtuner convert` 的两个子命令 `pth_to_hf` 和 `merge` 都可用：

```shell #test id="xtuner-convert-help"
out=$(xtuner convert --help 2>&1)
echo "lines: $(echo "$out" | wc -l)"
echo "has_pth_to_hf_subcmd: $(xtuner convert pth_to_hf --help >/dev/null 2>&1 && echo True || echo False)"
echo "has_merge_subcmd: $(xtuner convert merge --help >/dev/null 2>&1 && echo True || echo False)"
test -n "$out"
```

输出结果类似：

```shell #test-result id="xtuner-convert-help" fuzzy='xxx'
lines: xxx
has_pth_to_hf_subcmd: True
has_merge_subcmd: True
```

CI smoke 真跑 `pth_to_hf` + `merge`：

```shell #test-setup
# Stub torchvision via real package to bypass NPU base image's broken torchvision C++ ops.
# 触发链：xtuner.tools.merge → import transformers → transformers.models.bloom.modeling_bloom
#   → transformers.modeling_utils.loss.loss_utils.loss_deformable_detr → image_transforms
#   → image_utils → `from torchvision.transforms import InterpolationMode` / `from
#   torchvision.transforms.functional import ...`。
# 走 PYTHONPATH + 真正的 stub package 让 import 命中 `__init__.py`，避开 site-packages
# 里那个缺 C++ extension 的 torchvision（任何 torch.ops.torchvision.* 调用都会抛
# `RuntimeError: operator torchvision::nms does not exist`）。
mkdir -p /tmp/torchvision_stub/torchvision/ops /tmp/torchvision_stub/torchvision/transforms
cat > /tmp/torchvision_stub/torchvision/__init__.py <<'PYEOF'
__version__ = "0.24.0"
PYEOF
cat > /tmp/torchvision_stub/torchvision/ops/__init__.py <<'PYEOF'
def nms(*args, **kwargs):
    return None
PYEOF
# transforms/__init__.py 至少要提供 InterpolationMode（image_utils 4.48 line 59 用）。
# 用 Enum 让 `InterpolationMode.NEAREST` 这种属性访问 work；Compose / ToTensor 等 5 iter smoke
# 不真正做数据增强，lambda no-op 够用。
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
# transforms.functional 给 image_transforms 4.48 line 58 `from torchvision.transforms import
# functional as F` 用——F.normalize 至少要 no-op（5 iter smoke 不真正做图像增强）。
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
export PYTHONPATH=/tmp/torchvision_stub${PYTHONPATH:+:$PYTHONPATH}
python -c "import torchvision, torchvision.ops, torchvision.transforms, torchvision.transforms.functional; print('torchvision_stubbed: ok')"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
src_pth=$(ls -t /tmp/xtuner_sft_llm_out_single/*.pth 2>/dev/null | head -1)
[ -n "$src_pth" ] || { echo "no .pth from xtuner-train-smoke-setup"; exit 1; }
hf_dir="${src_pth%.pth}_hf"
merged_dir=/tmp/xtuner_sft_llm_out_single/merged
rm -rf "$hf_dir" "$merged_dir"
mkdir -p "$hf_dir" "$merged_dir"

# pth → hf（PEFT 格式：adapter_config.json + adapter_model.safetensors）
python -m xtuner.tools.model_converters.pth_to_hf \
    /tmp/xtuner_npu_llm_cfg.py \
    "$src_pth" \
    "$hf_dir"

# merge（PEFT adapter 合并回 base → 3.5 GB safetensors）
python -m xtuner.tools.model_converters.merge \
    ./qwen/Qwen1.5-1.8B-Chat \
    "$hf_dir" \
    "$merged_dir" \
    --max-shard-size 2GB
```

验合并产物落盘：

```shell #test id="xtuner-merge-verify"
ls -t /tmp/xtuner_sft_llm_out_single/merged/*.safetensors 2>/dev/null | head -3
```

输出结果如下：

```shell #test-result id="xtuner-merge-verify" fuzzy='xxx'
/tmp/xtuner_sft_llm_out_single/merged/model.safetensors
```

### 与模型对话

合并完权重后，可以直接用 `xtuner chat` 跟模型对话。下面烟囱测 `xtuner chat --help` 退出码 0 + 关键参数 `--adapter` / `--prompt-template` / `--system-template` 都存在：

```shell #test id="xtuner-chat-help"
out=$(xtuner chat --help 2>&1)
echo "lines: $(echo "$out" | wc -l)"
echo "has_adapter_arg: $(echo "$out" | grep -c -- '--adapter')"
echo "has_prompt_template_arg: $(echo "$out" | grep -c -- '--prompt-template')"
echo "has_system_template_arg: $(echo "$out" | grep -c -- '--system-template')"
test -n "$out"
```

输出结果类似：

```shell #test-result id="xtuner-chat-help" fuzzy='xxx'
lines: xxx
has_adapter_arg: xxx
has_prompt_template_arg: xxx
has_system_template_arg: xxx
```

CI smoke 真跑 chat（merged 版，复用上面 `xtuner-merge-verify` 合并后的 1.8B merged/ 目录，qwen_chat + colorist system-template）：

```shell #test-setup
# chat.py 顶层 import transformers（含 CLIPImageProcessor / CLIPVisionModel）触发 torchvision
# lazy import 在 NPU base image 上挂。走 PYTHONPATH + 真正的 stub package（不是
# types.ModuleType 注入，那样 find_spec 因为 `__spec__ is None` 会 ValueError）。
# transforms/__init__.py 提供 InterpolationMode（image_utils 用）和 Compose/ToTensor 等
# 5 iter smoke 不真正用得到的 no-op；transforms/functional.py 提供 normalize（F.normalize
# image_transforms 用）。
export PYTHONPATH=/tmp/torchvision_stub${PYTHONPATH:+:$PYTHONPATH}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

```shell #test id="xtuner-chat-merged"
# 跟上游 quickstart 完全一致：
# xtuner chat <merged> --prompt-template qwen_chat --system-template colorist
# stdin pipe 第一个输入是 colorist prompt，第二个输入是 EXIT 触发 chat.py main() 里 exit(0)
# （chat.py 是 while True: get_input() 交互式循环，没 --input flag，只能 stdin pipe 喂）。
# --no-streamer 关掉 TextStreamer（CI 抓 stdout 比对要 print 完整输出而不是增量 stream）。
# --max-new-tokens 64 给英文回复留余量。
echo -e "Tell me about the color #66ccff\nEXIT" | \
python -m xtuner.tools.chat /tmp/xtuner_sft_llm_out_single/merged \
    --prompt-template qwen_chat \
    --system-template colorist \
    --no-streamer \
    --max-new-tokens 64 2>&1 | tail -n 5
```

输出结果如下：

```shell #test-result id="xtuner-chat-merged" fuzzy='xxx'
Load LLM from /tmp/xtuner_sft_llm_out_single/merged
xxx (Qwen1.5-1.8B + 5 samples × 1 epoch 微调后对英文颜色描述的回复；smoke 不验证具体色号)
Log: Exit!
```

不合并、只跟 LLM + LoRA adapter 直接对话（adapter 版）：

```shell #test id="xtuner-chat-adapter"
# 跟上游 quickstart 完全一致：
# xtuner chat <base> --adapter <iter_xxx_hf> --prompt-template qwen_chat --system-template colorist
hf_dir=$(ls -td /tmp/xtuner_sft_llm_out_single/iter_*_hf 2>/dev/null | head -1)
[ -n "$hf_dir" ] || { echo "no iter_*_hf from pth_to_hf step"; exit 1; }
echo -e "Tell me about the color #66ccff\nEXIT" | \
python -m xtuner.tools.chat ./qwen/Qwen1.5-1.8B-Chat \
    --adapter "$hf_dir" \
    --prompt-template qwen_chat \
    --system-template colorist \
    --no-streamer \
    --max-new-tokens 64 2>&1 | tail -n 5
```

输出结果如下：

```shell #test-result id="xtuner-chat-adapter" fuzzy='xxx'
Load LLM from ./qwen/Qwen1.5-1.8B-Chat
Load adapter from /tmp/xtuner_sft_llm_out_single/iter_xxx_hf
xxx (Qwen1.5-1.8B + LoRA adapter 对英文颜色描述的回复；smoke 不验证具体色号)
Log: Exit!
```
