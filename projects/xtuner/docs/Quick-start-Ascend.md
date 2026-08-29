# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上跑通 [xtuner](https://github.com/InternLM/xtuner) 的最小链路。


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

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 1）
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

装 `modelscope`（本文档下载 InternLM2-Chat-7B 权重 + Colorist 数据集要用，ModelScope 国内网络更稳）：

```shell #test-setup
uv pip install modelscope
```

## 安装 xtuner

xtuner 同时支持 PyPI 二进制安装与源码安装。

### 使用 uv 进行安装（PyPI 二进制）

```shell #test id="xtuner-install-binary"
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple --no-deps xtuner
# xtuner 核心依赖（来自 requirements/runtime.txt，扣掉 torch/torchvision/bitsandbytes 这 3 个 NPU 不可用的）
uv pip install 'datasets>=3.2.0,<4.0.0' einops loguru 'mmengine==0.10.6' openpyxl 'peft>=0.14.0' 'scikit-image' scipy SentencePiece tiktoken 'transformers==4.48.0' transformers_stream_generator
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
# 直接从 xtuner 的 runtime.txt 里扣掉 3 个 NPU 不可用的，剩下的全交给 uv
grep -vE '^(torch|torchvision|bitsandbytes)$' requirements/runtime.txt \
    | uv pip install -r /dev/stdin
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

我们要做一个"按颜色描述 → 输出 16 进制色号"的小模型。流程：**下载底座模型 → 准备训练数据 → 复制一份官方训练配置 → 改 4 处指向我们的模型/数据 → 跑训练 → 把训练产物合并回底座**。每一步都挂一个 `#test` 块在 CI 里自动验证。

> 训练入口：`xtuner train <配置文件>`，对应 `python -m xtuner.tools.train`，下面会用到。

### 准备模型权重

底座用 InternLM2-Chat-7B——它"会说话"的能力全在这 14 GB 权重里。从 [ModelScope](https://www.modelscope.cn/) 拉下来：

```shell #test-setup
python -c "from modelscope import snapshot_download; snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b', cache_dir='./Shanghai_AI_Laboratory')"
```

检查是否有对应的文件：
```shell #test id="xtuner-pull-weights"
ws=$(find ./Shanghai_AI_Laboratory -name config.json -print -quit)
ls "$(dirname "$ws")" | grep -E '\.(safetensors|json|model|py)$' | sort
```

输出结果类似：

```shell #test-result id="xtuner-pull-weights"
config.json
configuration.json
configuration_internlm2.py
generation_config.json
model-00001-of-00008.safetensors
model-00002-of-00008.safetensors
model-00003-of-00008.safetensors
model-00004-of-00008.safetensors
model-00005-of-00008.safetensors
model-00006-of-00008.safetensors
model-00007-of-00008.safetensors
model-00008-of-00008.safetensors
tokenization_internlm2.py
tokenizer.model
tokenizer_config.json
```

权重落到 `./Shanghai_AI_Laboratory/internlm2-chat-7b/`（约 14 GB）。目录里有 8 个 `model-*-of-00008.safetensors`（一个完整模型被切成 8 份存），加上 tokenizer（分词器）和 config（模型结构配置）。

### 准备训练数据

我们要让模型"看到颜色描述就回答色号"。去 ModelScope 拉 [Colorist](https://www.modelscope.cn/datasets/fanqiNO1/colors)——一堆"颜色描述 → 16 进制色号"的对话样本，只有几 MB：

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
```

```shell #test id="xtuner-pull-dataset"
for f in colors.json README.md train.jsonl; do
    test -f "colors/$f" || { echo "MISSING: colors/$f"; exit 1; }
done
ls colors/ | sort
```

输出结果（同时是数据集会落在 `./colors/` 下的目录结构）：

```shell #test-result id="xtuner-pull-dataset"
colors.json
README.md
train.jsonl
...
```

下载后落到 `./colors/`：`colors.json`（原始数据）、`README.md`（说明）、`train.jsonl`（每行一条 JSON 对话样本）。

### 准备配置文件

还差一份"训练说明书"告诉 xtuner 怎么训——也就是配置文件。xtuner 内置 600 多种开箱即用配置（底座 × 微调方法 × 数据集的组合），先看都有哪些：

```shell #test id="xtuner-list-cfg"
python -c "
from xtuner.configs import cfgs_name_path
names = sorted(cfgs_name_path.keys())
print('lines:', len(names))
print('head_first:', names[0] if names else '')
print('colorist_count:', sum(1 for n in names if 'colorist' in n))
"
```

```shell #test-result id="xtuner-list-cfg" fuzzy='xxx'
lines: xxx
head_first: xxx
colorist_count: xxx
```

挑一份我们要的："底座 InternLM2（或 Llama）+ 微调方法 QLoRA（LoRA 的省显存版）+ 数据集 Colorist"。先找到这个配置的名字：

```shell #test-setup store="xtuner_colorist_cfg_name"
config_name=$(python -c "
from xtuner.configs import cfgs_name_path
import re
names = sorted(cfgs_name_path.keys())
match = next((n for n in names if re.search(r'(internlm2|llama).*qlora.*colorist', n)), '')
print(match)
")
```

检查config_name是否存在：
```shell #test id="xtuner-find-colorist-cfg" load="xtuner_colorist_cfg_name>>config_name"
test -n "$config_name" || { echo "no matching qlora+colorist config"; exit 1; }
echo "config_name=$config_name"
```

输出结果如下：
```shell #test-result id="xtuner-find-colorist-cfg" fuzzy='xxx'
config_name=xxx
```

拿到配置名后拷一份到本地（不要直接改原版）：
```shell #test-setup store="xtuner_llm_cfg_path" load="xtuner_colorist_cfg_name>>config_name"
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

检查路径是否存在：
```shell #test id="xtuner-copy-cfg" load="xtuner_llm_cfg_path>>cfg"
test -f "$cfg" || { echo "cfg not copied to $cfg"; exit 1; }
echo "cfg_copied: $cfg"
```

输出结果如下：
```shell #test-result id="xtuner-copy-cfg" fuzzy='xxx'
cfg_copied: xxx
```

`config_name` 形如 `internlm2_7b_qlora_colorist_e5`，拷出来的 `cfg_copied` 形如 `/tmp/xtuner_npu_llm_cfg.py/internlm2_7b_qlora_colorist_e5_copy.py`。记下这个路径。

### 修改配置文件

拷出来的配置跟官方模板一模一样——模板里指向的模型路径、数据路径、对话模板都是给"标准环境"写的，对不上我们这台 NPU 上下载好的模型和数据。要改 4 处（详见 [xtuner 快速上手的"修改配置文件"小节](https://xtuner.readthedocs.io/zh-cn/latest/legacy/get_started/quickstart.html)）。`<cfg>` 是上一节最后一行 `cfg_copied:` 后面那个绝对路径：

```shell #test-setup load="xtuner_llm_cfg_path>>cfg" store="xtuner_llm_cfg_path"
# 把模板里那 4 处 patch 应用到 copy-cfg 出来的 config 上：
python -c "
import re
path = '<cfg>'
with open(path) as f:
    text = f.read()
# 4 处 patch：
#   pretrained_model_name_or_path 用 regex 同时覆盖 7b/20b 两种 cfg（xtuner v0.2.0 的 colorist cfg 是
#   llama 版，V1 之后才有 internlm2 版；不同 size 的 HF 模型名不一样，自动取 size 后缀）
text, _ = re.subn(
    r'pretrained_model_name_or_path = \"internlm/internlm2-(\d+b)\"',
    r\"pretrained_model_name_or_path = './Shanghai_AI_Laboratory/internlm2-chat-\1'\",
    text,
)
text = text.replace('data_path = \"burkelibbey/colors\"', \"data_path = './colors/train.jsonl'\")
text = text.replace('prompt_template = PROMPT_TEMPLATE.default', 'prompt_template = PROMPT_TEMPLATE.internlm2_chat')
text = text.replace('dataset=dict(type=load_dataset, path=data_path)', \"dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path))\")
with open(path, 'w') as f:
    f.write(text)
print(path)
"
```

用 py_compile 验 cfg 是合法 Python + grep 验 4 处 patch 都生效：
```shell #test id="xtuner-patch-cfg" load="xtuner_llm_cfg_path>>cfg"
python -c "
import py_compile
import re
py_compile.compile('<cfg>', doraise=True)
print('cfg_compiles_ok')
with open('<cfg>') as f:
    text = f.read()
checks = [
    # model size 后缀随 cfg 选型而变（7b/20b），用 regex 兼容两种：
    ('model_path', re.search(r\"pretrained_model_name_or_path = '(.+/internlm2-chat-\d+b)'\", text).group(1)),
    ('data_path', './colors/train.jsonl'),
    ('prompt_template', 'PROMPT_TEMPLATE.internlm2_chat'),
    ('dataset_format', \"dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path))\"),
]
for name, expected in checks:
    # needle 用 cfg 里的实际字面量（cfg 字段名是 pretrained_model_name_or_path 不是 model_path，
    # 不要把 check 名当字段名拼到 needle 里）：
    assert expected in text, f'missing patch ({name}): {expected!r}'
print('cfg_patch_ok')
print(f'model_name= {checks[0][1]}')
print('data_path= ./colors/train.jsonl')
print('prompt_template= PROMPT_TEMPLATE.internlm2_chat')
"
```

输出结果如下：

```shell #test-result id="xtuner-patch-cfg" fuzzy='xxx'
cfg_compiles_ok
cfg_patch_ok
model_name= ./Shanghai_AI_Laboratory/internlm2-chat-xxx
data_path= ./colors/train.jsonl
prompt_template= PROMPT_TEMPLATE.xxx
```

- `cfg_compiles_ok`：配置文件语法 OK
- `cfg_patch_ok`：4 处修改都生效了
- `model_name`：指向我们下载好的 InternLM2 权重目录
- `data_path`：指向我们下载好的 Colorist 数据文件
- `prompt_template`：用 InternLM2 的"用户提问 / 模型回答"对话格式

### 启动微调

配置改好了，按模板给的单卡 / 多卡命令就能跑。

#### 单卡（CI smoke 用例）

训练日志（loss、学习率等）每次跑都不一样，没法写死预期值。拆成两步：先用最小数据集（5 条样本 × 1 epoch）跑通，再单独检查 `.pth` 有没有落盘：

```shell #test-setup id="xtuner-train-smoke-setup"
cp /tmp/xtuner_npu_llm_cfg.py /tmp/xtuner_npu_smoke_single_cfg.py
cat >> /tmp/xtuner_npu_smoke_single_cfg.py <<'EOF'

train_cfg = dict(max_epochs=1)
train_dataloader = dict(dataset=dict(samples_per_epoch=5))
EOF

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_NPU_USE_HCCL=1
# 用 python -m xtuner.tools.train 直接调 train 模块，绕开 console_script wrapper shebang 错配
# （wrapper 启动的 Python 看不到 uv egg-link 把 xtuner 当 namespace package，`from xtuner import cli` ImportError）。
python -m xtuner.tools.train /tmp/xtuner_npu_smoke_single_cfg.py --work-dir /tmp/xtuner_sft_llm_out_single
```

```shell #test id="xtuner-train-smoke"
ls -t /tmp/xtuner_sft_llm_out_single/*.pth 2>/dev/null | head -1
```

```shell #test-result id="xtuner-train-smoke" fuzzy='xxx'
/tmp/xtuner_sft_llm_out_single/iter_xxx.pth
```

#### 多卡（CI smoke 用例，2 卡 runner）

```shell #test-setup id="xtuner-train-smoke-multi-setup"
cp /tmp/xtuner_npu_llm_cfg.py /tmp/xtuner_npu_smoke_multi_cfg.py
cat >> /tmp/xtuner_npu_smoke_multi_cfg.py <<'EOF'

train_cfg = dict(max_epochs=1)
train_dataloader = dict(dataset=dict(samples_per_epoch=5))
EOF

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_NPU_USE_HCCL=1
NPROC_PER_NODE=2 python -m xtuner.tools.train /tmp/xtuner_npu_smoke_multi_cfg.py --work-dir /tmp/xtuner_sft_llm_out_multi
```

```shell #test id="xtuner-train-smoke-multi"
ls -t /tmp/xtuner_sft_llm_out_multi/*.pth 2>/dev/null | head -1
```

```shell #test-result id="xtuner-train-smoke-multi" fuzzy='xxx'
/tmp/xtuner_sft_llm_out_multi/iter_xxx.pth
```

<!--
完整 5 epoch × 144 step = 720 step 训练命令（720 step 太长不进 CI smoke）。
CI 走 `xtuner-train-smoke`（5 samples × 1 epoch）。真正 720 step 本地按需手动跑 —— 把下面命令的 `--max-epochs 1` 去掉、用 `xtuner train` 替换 `python -m xtuner.tools.train`（去掉 wrapper shebang 错配 workaround，前提是用非 egg-link 装法）：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_NPU_USE_HCCL=1
python -m xtuner.tools.train /tmp/xtuner_npu_llm_cfg.py --work-dir /tmp/xtuner_sft_llm_out

# 多卡（xtuner.tools.train 内置 torchrun 集成，按需调整 NPROC_PER_NODE）
NPROC_PER_NODE=${GPU_NUM} python -m xtuner.tools.train /tmp/xtuner_npu_llm_cfg.py --work-dir /tmp/xtuner_sft_llm_out
```
-->

> **2 步拆分的备注**：`xtuner-train-smoke-setup` 块跑训练（输出不校），`xtuner-train-smoke` 块只跑 `ls -t` 拿最新 `.pth` 路径。`fuzzy='xxx'` 让 `iter_xxx.pth` 里的 `xxx` 当通配符匹配实际的迭代号，所以 CI 不用每次更新预期输出。

### 完整 5 epoch 训练（本地手动）

CI smoke 用 5 samples × 1 epoch 只是"跑通整条链路"——真正想训出能用的模型，要跑完整个 Colorist 数据集（720 step），这一步不进 CI，太慢了：

```shell #test-setup
# CI smoke：复用 xtuner-train-smoke-setup 的 5 samples × 1 epoch，把 max_epochs 显式钉 1
# 跟上面 5 epoch 完整命令对齐（去掉 --max-epochs 1 就是 5 epoch）。work-dir 用 _full 后缀
# 跟注释里的示例路径 /tmp/xtuner_sft_llm_out 区分：
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_NPU_USE_HCCL=1
python -m xtuner.tools.train /tmp/xtuner_npu_llm_cfg.py --work-dir /tmp/xtuner_sft_llm_out_full --max-epochs 1
NPROC_PER_NODE=2 python -m xtuner.tools.train /tmp/xtuner_npu_llm_cfg.py --work-dir /tmp/xtuner_sft_llm_out_full --max-epochs 1
```

### 模型转换 + LoRA 合并

训练完产出的 `.pth` 只包含 LoRA 这部分"增量"参数（几十 MB）——QLoRA / LoRA 的精髓就在这：不动原模型 14 GB 权重，只学一个小的"补丁"。但要拿这个 `.pth` 去做推理，必须先转 HuggingFace 标准格式（`pth_to_hf`），再合并回底座（`merge`），合并后才是一个完整的 14 GB 模型。

先验证 `xtuner convert` 的两个子命令确实注册在 CLI 里（不真跑——会触发 torchvision 缺失）：

```shell #test id="xtuner-convert-help"
python -c "
from xtuner.entry_point import modes
convert_dict = modes.get('convert', {})
print('has_pth_to_hf_subcmd: ' + str('pth_to_hf' in convert_dict))
print('has_merge_subcmd: ' + str('merge' in convert_dict))
print('has_train: ' + str('train' in modes))
print('has_chat: ' + str('chat' in modes))
"
```

输出结果如下：
```shell #test-result id="xtuner-convert-help" disable_fuzzy
has_pth_to_hf_subcmd: True
has_merge_subcmd: True
has_train: True
has_chat: True
```

完整 `pth_to_hf` + `merge` 流程（依赖前面 5 epoch 训练出的 `.pth`，本地按需手动跑）：

<!--
```shell
# 创建存放 hf 格式参数的目录
mkdir -p /tmp/xtuner_sft_llm_out/iter_720_hf

# pth → hf
xtuner convert pth_to_hf /tmp/xtuner_npu_llm_cfg.py \
    /tmp/xtuner_sft_llm_out/iter_720.pth \
    /tmp/xtuner_sft_llm_out/iter_720_hf

# 合并 LoRA adapter 到 base
mkdir -p /tmp/xtuner_sft_llm_out/merged
xtuner convert merge ./Shanghai_AI_Laboratory/internlm2-chat-7b \
    /tmp/xtuner_sft_llm_out/iter_720_hf \
    /tmp/xtuner_sft_llm_out/merged \
    --max-shard-size 2GB
```
-->

```shell #test-setup
# CI smoke 只走"取 smoke 训出的 .pth + 准备目标目录 + print 命令结构"。
# 真跑 xtuner convert pth_to_hf / merge 会触发 torchvision chain 在 NPU image 上挂
# （与 xtuner-convert-help 同因），本地按需手动跑上面 HTML 注释里的命令：
src_pth=$(ls -t /tmp/xtuner_sft_llm_out_full/*.pth 2>/dev/null | head -1)
[ -n "$src_pth" ] || { echo "no .pth from xtuner-train-smoke-setup"; exit 1; }
hf_dir="${src_pth%.pth}_hf"
mkdir -p "$hf_dir" /tmp/xtuner_sft_llm_out_full/merged
echo "xtuner convert pth_to_hf /tmp/xtuner_npu_llm_cfg.py $src_pth $hf_dir"
echo "xtuner convert merge ./Shanghai_AI_Laboratory/internlm2-chat-7b $hf_dir /tmp/xtuner_sft_llm_out_full/merged --max-shard-size 2GB"
```

> `#test` 只烟囱测 `xtuner convert` 的两个子命令 `pth_to_hf` 和 `merge` 在 `xtuner.entry_point.modes` dict 里**注册**（不真跑 `xtuner convert pth_to_hf --help` —— 它会 subprocess 调 `python pth_to_hf.py --help`，触发 peft→transformers→torchvision import chain 在 NPU image 上挂 `torchvision::nms` operator 缺失）。完整转换 + 合并依赖前面训练出的 `.pth`，CI smoke 跑不到，本地按需手动跑。

### 与模型对话

合并完权重后，可以直接用 `xtuner chat` 跟模型对话。下面烟囱测 `xtuner chat --help` 退出码 0 + 关键参数 `--adapter` / `--prompt-template` / `--system-template` 都存在：

```shell #test id="xtuner-chat-help"
# 不能直接 xtuner chat --help —— chat.py 顶层 import peft + transformers，触发 torchvision chain
# 在 NPU image 挂。改成检查 chat.py 文件存在 + 关键参数在源码里有定义：
# 注：xtuner 是 egg-link/namespace 装的，`xtuner.__file__` 是 None；从 xtuner.entry_point.__file__
# （regular module，.py 文件路径）推导：
python -c "
import os.path as osp
import xtuner.entry_point
xtuner_dir = osp.dirname(xtuner.entry_point.__file__)
chat_path = osp.join(xtuner_dir, 'tools', 'chat.py')
with open(chat_path) as f:
    src = f.read()
print('chat_script_exists: ' + str(osp.exists(chat_path)))
print('has_adapter_arg: ' + str('--adapter' in src))
print('has_prompt_template_arg: ' + str('--prompt-template' in src))
print('has_system_template_arg: ' + str('--system-template' in src))
"
```

输出结果类似：

```shell #test-result id="xtuner-chat-help" disable_fuzzy
chat_script_exists: True
has_adapter_arg: True
has_prompt_template_arg: True
has_system_template_arg: True
```

完整 `xtuner chat` 命令（交互式 CLI，依赖前面合并后的权重，本地按需手动跑）：

<!--
```shell
xtuner chat /tmp/xtuner_sft_llm_out/merged \
    --prompt-template internlm2_chat \
    --system-template colorist
```
-->

```shell #test-setup
# CI smoke：交互式 CLI 不能 CI 跑（hang 等输入）；同样不能 xtuner chat --help
# （peft→transformers→torchvision import chain 在 NPU image 上挂）。
# 真实 chat 会话本地按需手动跑 —— 用 `python -m xtuner.tools.chat` 替换 `xtuner chat`
# 绕开 wrapper shebang 错配：
echo "python -m xtuner.tools.chat /tmp/xtuner_sft_llm_out_full/merged --prompt-template internlm2_chat --system-template colorist"
```

也可以不合并、只跟 LLM + LoRA adapter 直接对话：

<!--
```shell
xtuner chat ./Shanghai_AI_Laboratory/internlm2-chat-7b \
    --adapter /tmp/xtuner_sft_llm_out/iter_720_hf \
    --prompt-template internlm2_chat \
    --system-template colorist
```
-->

```shell #test-setup
# 也可以不合并、只跟 LLM + LoRA adapter 直接对话（同上 chat 不能 CI 跑）：
echo "python -m xtuner.tools.chat ./Shanghai_AI_Laboratory/internlm2-chat-7b --adapter /tmp/xtuner_sft_llm_out_full/iter_xxx_hf --prompt-template internlm2_chat --system-template colorist"
```

交互示例（训练前模型 → 训练后模型的输出变化）：

```
double enter to end input (EXIT: exit chat, RESET: reset history) >>> 宁静而又相当明亮的浅天蓝色，介于天蓝色和婴儿蓝之间，因其亮度而带有一丝轻微的荧光感。

#66ccff
```

> `#test` 只烟囱测 `xtuner chat` 脚本存在 + 源码里 `xtuner/tools/chat.py` 定义了 `--adapter` / `--prompt-template` / `--system-template` 三个关键参数——不真跑 `xtuner chat --help`（chat.py 顶层 import peft + transformers，触发 torchvision chain 在 NPU image 挂）。完整交互式对话没法做自动化断言（依赖 stdin），本地按需手动跑。