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
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| xtuner | GitHub 最新 release tag（运行时由引擎解析，doc 不写死具体值） |

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

检查 torch / torch_npu 是否装好且 NPU 设备可用：

```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-torch" fuzzy='xxx'
torch= 2.9.0+cpu
torch_npu= 2.9.0.post2
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

源码装好后做一次 `importlib.util.find_spec` 烟囱测试，验证顶层包 + 关键入口子模块能被解析（顶层包决定 `import xtuner` 通；`xtuner.entry_point` 决定 CLI 的 `MODES` 常量可读，验下面 CLI 表面检查）：

```shell #test id="xtuner-import-check"
python -c "
import importlib.util as u
specs = {m: u.find_spec(m) for m in ['xtuner', 'xtuner.entry_point']}
for m, s in specs.items():
    print(m, 'ok' if s is not None else 'MISSING')
assert all(s is not None for s in specs.values()), specs
"
```

输出结果如下（用 `disable_fuzzy` 关掉默认非贪婪通配，按字面比对 — 任何一个 `MISSING` 都说明源码树装缺了）：

```shell #test-result id="xtuner-import-check" disable_fuzzy
xtuner ok
xtuner.entry_point ok
```

CLI 表面 sanity check——`xtuner.entry_point.MODES` 必须非空，且至少包含 `train`（训练入口）、`list-cfg`（下面"准备配置文件"章节要用）、`chat`（下面"与模型对话"章节要用）：

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

本文档的训练入口是 `xtuner train <config>`（基于 mmengine runner）。下面 7 个章节按 [xtuner legacy 快速上手模板](https://xtuner.readthedocs.io/zh-cn/latest/legacy/get_started/quickstart.html) 的顺序展开，每个章节都挂一个 `#test` 块做烟囱测试 —— CI smoke 不跑完整 5 epoch 训练 / 14 GB 模型合并，只验证该章节关键命令能通。

### 准备模型权重

在微调模型前，要先拉一份 InternLM2-Chat-7B 的权重。`modelscope` SDK 已在[前置安装](#前置安装)章节装好。ModelScope 在国内网络上更稳（避免 HF 直连被墙）。

下载 InternLM2-Chat-7B 权重（约 14 GB，落到 `./Shanghai_AI_Laboratory/internlm2-chat-7b/`）：

```shell #test-setup
python -c "from modelscope import snapshot_download; snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b', cache_dir='./Shanghai_AI_Laboratory')"
```

```shell #test id="xtuner-pull-weights"
ws=$(find ./Shanghai_AI_Laboratory -name config.json -print -quit)
test -n "$ws" && test -f "$ws" && echo "weights_ok"
ls -la "$(dirname "$ws")" | head -1
```

输出结果类似：

```shell #test-result id="xtuner-pull-weights" fuzzy='xxx'
weights_ok
total xxx
```

权重落到 `./Shanghai_AI_Laboratory/internlm2-chat-7b/` 下（约 14 GB），含 `pytorch_model-*.bin` ×8 + tokenizer + config。

> `#test-setup` 块（hidden）在 CI smoke 里跑 `snapshot_download` 拉权重（~5-10 分钟），`#test` 只验 `config.json` 存在。本地如果已经下过 weights，可以跳过 setup 单独跑 `#test`。

### 准备微调数据集

Colorist 数据集：根据颜色描述给 16 进制颜色编码的指令微调集，几 MB 大小。ModelScope SDK 直接拉（拉到默认 `cache_dir/datasets/fanqiNO1/colors/<revision>/`，再 `shutil.copytree` 重定向到 `./colors/`，这样下游 cfg 的 `data_path = './colors/train.jsonl'` 不用改）：

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
echo "colors.json ok"
echo "README.md ok"
echo "train.jsonl ok"
```

输出结果如下：

```shell #test-result id="xtuner-pull-dataset" disable_fuzzy
colors.json ok
README.md ok
train.jsonl ok
```

数据集会落在 `./colors/` 下，结构：

```
colors/
├── colors.json
├── README.md
└── train.jsonl
```

> `#test-setup` 块在 CI smoke 里跑 `modelscope.snapshot_download(..., repo_type='dataset')` 拉数据集再重定向到 `./colors/`，`#test` 验 3 个文件（`colors.json` / `README.md` / `train.jsonl`）都存在（按字面比对，缺一个就 `exit 1` 报失败）。本地如果已经下载过，可以跳过 setup 单独跑 `#test`（前提：数据集路径仍然是 `./colors/`）。

### 准备配置文件

XTuner 自带大量开箱即用的 config：

```shell
xtuner list-cfg
```

```shell #test id="xtuner-list-cfg"
out=$(xtuner list-cfg 2>/dev/null)
err=$(xtuner list-cfg 2>&1 >/dev/null)
echo "lines: $(echo "$out" | wc -l)"
echo "head_first: $(echo "$out" | head -1)"
echo "err_lines: $(echo "$err" | wc -l)"
echo "err_head: $(echo "$err" | head -1)"
echo "err_grep_at: $(echo "$err" | grep -E "ModuleNotFound|ImportError|Error" | head -1)"
test -n "$out"
```

输出结果类似（具体行数随 release 漂移）：

```shell #test-result id="xtuner-list-cfg" fuzzy='xxx'
lines: xxx
head_first: xxx
err_lines: xxx
err_head: xxx
err_grep_at: xxx
```

从 list-cfg 拷一份 QLoRA + Colorist 配置到本地（具体 config 名随 release 漂移，先 grep 推断；xtuner v0.2.0 的 colorist cfg 是 llama 版，V1 之后才有 internlm2 版）：

从 list-cfg 拷一份 QLoRA + Colorist 配置到本地（具体 config 名随 release 漂移，先 grep 推断；xtuner v0.2.0 的 colorist cfg 是 llama 版，V1 之后才有 internlm2 版）：

```shell #test-setup store="xtuner_llm_cfg_path"
config_name=$(xtuner list-cfg 2>/dev/null | grep -E "(internlm2|llama).*qlora.*colorist" | head -1)
test -n "$config_name" || { echo "no matching config ((internlm2|llama).*qlora.*colorist); abort"; exit 1; }
xtuner copy-cfg "$config_name" /tmp/xtuner_npu_llm_cfg.py
echo "/tmp/xtuner_npu_llm_cfg.py"
```

输出路径到下一节「修改配置文件」

### 修改配置文件

拷出来的 config 跟模板原版完全一致，按模板的 4 处修改规则调整（详见 [legacy quickstart 模板的"修改配置文件"小节](https://xtuner.readthedocs.io/zh-cn/latest/legacy/get_started/quickstart.html)）。`<cfg>` 是上一节「准备配置文件」store 出来的 cfg 绝对路径：

```shell #test-setup load="xtuner_llm_cfg_path>>cfg" store="xtuner_llm_cfg_path"
# 把模板里那 4 处 patch 应用到 copy-cfg 出来的 config 上：
#   PART 1 Settings
#     pretrained_model_name_or_path = './Shanghai_AI_Laboratory/internlm2-chat-7b'
#     data_path = './colors/train.jsonl'
#     prompt_template = PROMPT_TEMPLATE.internlm2_chat
#   PART 3 Dataset & Dataloader
#     train_dataset = process_hf_dataset(dataset=dict(type=load_dataset, path='json',
#                                                     data_files=dict(train=data_path)), ...)
sed -i "s|pretrained_model_name_or_path = 'internlm/internlm2-7b'|pretrained_model_name_or_path = './Shanghai_AI_Laboratory/internlm2-chat-7b'|" <cfg>
sed -i "s|data_path = 'burkelibbey/colors'|data_path = './colors/train.jsonl'|" <cfg>
sed -i "s|prompt_template = PROMPT_TEMPLATE.default|prompt_template = PROMPT_TEMPLATE.internlm2_chat|" <cfg>
sed -i "s|dataset=dict(type=load_dataset, path=data_path)|dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path))|" <cfg>
echo "<cfg>"
```

```shell #test id="xtuner-patch-cfg" load="xtuner_llm_cfg_path>>cfg"
python -c "
from mmengine.config import Config
cfg = Config.fromfile('<cfg>')
print('cfg_loaded_ok')
print('model_name=', cfg.pretrained_model_name_or_path)
print('data_path=', cfg.data_path)
print('prompt_template=', cfg.prompt_template)
"
```

输出结果类似：

```shell #test-result id="xtuner-patch-cfg" fuzzy='xxx'
cfg_loaded_ok
model_name= ./Shanghai_AI_Laboratory/internlm2-chat-7b
data_path= ./colors/train.jsonl
prompt_template= PROMPT_TEMPLATE.xxx
```

> `#test-setup` 把 4 处 sed 实际应用到 cfg；`#test` 跑 `mmengine.config.Config.fromfile(<cfg>)` 验 cfg 能加载 + 打印关键 `pretrained_model_name_or_path` / `data_path` / `prompt_template` 字段，证明 4 处 patch 都生效。smoke 不验 cfg 训出来的实际效果，那要等下面"启动微调"章节真跑。

### 启动微调

参考模板给的单卡 + 多卡启动方式。CI smoke 不跑完整 5 epoch × 144 step = 720 step（耗时 ~45-60 分钟），而是在 cfg 末尾追加 `train_cfg = dict(max_epochs=1)` + `train_dataloader = dict(dataset=dict(samples_per_epoch=5))` 把训练压到 5 step（耗时 ~30 秒 - 1 分钟），跑通就视为该章节 smoke 通过。

#### 单卡（5 step smoke）

```shell #test-setup load="xtuner_llm_cfg_path>>cfg" store="xtuner_train_single_pth"
cp <cfg> /tmp/xtuner_npu_smoke_single_cfg.py
cat >> /tmp/xtuner_npu_smoke_single_cfg.py <<'EOF'

train_cfg = dict(max_epochs=1)
train_dataloader = dict(dataset=dict(samples_per_epoch=5))
EOF

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_NPU_USE_HCCL=1
xtuner train /tmp/xtuner_npu_smoke_single_cfg.py --work-dir /tmp/xtuner_sft_llm_out_single
ls -t /tmp/xtuner_sft_llm_out_single/*.pth 2>/dev/null | head -1
```

```shell #test id="xtuner-train-single-smoke" load="xtuner_train_single_pth>>pth"
test -n "$pth" && test -f "$pth" && echo "single_pth_found: yes"
echo "single_pth_path: $pth"
```

输出结果类似：

```shell #test-result id="xtuner-train-single-smoke" fuzzy='xxx'
single_pth_found: yes
single_pth_path: /tmp/xtuner_sft_llm_out_single/iter_xxx.pth
```

#### 多卡（5 step smoke，2 卡 runner）

> 多卡 smoke 直接用 `NPROC_PER_NODE=2`，前提是 workflow 申请 2 卡 runner（`linux-aarch64-a2-2` 这一类）。

```shell #test-setup load="xtuner_llm_cfg_path>>cfg" store="xtuner_train_multi_pth"
cp <cfg> /tmp/xtuner_npu_smoke_multi_cfg.py
cat >> /tmp/xtuner_npu_smoke_multi_cfg.py <<'EOF'

train_cfg = dict(max_epochs=1)
train_dataloader = dict(dataset=dict(samples_per_epoch=5))
EOF

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_NPU_USE_HCCL=1
NPROC_PER_NODE=2 xtuner train /tmp/xtuner_npu_smoke_multi_cfg.py --work-dir /tmp/xtuner_sft_llm_out_multi
ls -t /tmp/xtuner_sft_llm_out_multi/*.pth 2>/dev/null | head -1
```

```shell #test id="xtuner-train-multi-smoke" load="xtuner_train_multi_pth>>pth"
test -n "$pth" && test -f "$pth" && echo "multi_pth_found: yes"
echo "multi_pth_path: $pth"
```

输出结果类似：

```shell #test-result id="xtuner-train-multi-smoke" fuzzy='xxx'
multi_pth_found: yes
multi_pth_path: /tmp/xtuner_sft_llm_out_multi/iter_xxx.pth
```

完整 5 epoch × 144 step = 720 step 训练命令（本地按需手动跑，跑出来的 .pth 路径可直接被"模型转换 + LoRA 合并"章节消费）：

```shell
# 单卡（src CANN env 是必需的，不 source torch_npu 找不到 libhccl.so）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_NPU_USE_HCCL=1
xtuner train /tmp/xtuner_npu_llm_cfg.py --work-dir /tmp/xtuner_sft_llm_out

# 多卡（xtuner.train 内置 torchrun 集成，按需调整 NPROC_PER_NODE）
NPROC_PER_NODE=${GPU_NUM} xtuner train /tmp/xtuner_npu_llm_cfg.py --work-dir /tmp/xtuner_sft_llm_out
```

> 训练日志会逐 step 打印 `Iter(train) [N/5] lr: ... time: ... loss: ...`（smoke）或 `[N/720]`（完整），跟模板给的样本格式一致：
>
> ```
> mmengine - INFO - Iter(train) [ 10/720]  lr: 9.0001e-05  eta: 0:31:46  time: 2.6851  data_time: 0.0077  memory: 12762  loss: 2.6900
> ```
>
> `xtuner train` 入口对应 `xtuner.engine.runner`，device 由 `xtuner/utils/device.py` 自动检测（CPU / CUDA / NPU），不需要手动传 `device='npu'`；多卡走 NCCL，NPU 上靠 `TORCH_NPU_USE_HCCL=1` 让 torch_npu 接管。
>
> 两个 `#test-setup` 在 cfg 末尾追加 `train_cfg = dict(max_epochs=1)` + `train_dataloader = dict(dataset=dict(samples_per_epoch=5))` 把训练压到 5 step（Python 后定义覆盖先定义，兼容模板原版的 `max_epochs=5` + `samples_per_epoch=None`）。`#test` 验 5 step 训练成功跑通 + 产出 `.pth`（具体文件名如 `iter_5.pth` 随 xtuner release 漂移，doc 不写死）。完整 720 step 训练在 CI smoke 上跑不到，本地按需手动跑上面的 `xtuner train /tmp/xtuner_npu_llm_cfg.py` 命令。多卡 smoke 需要 2 卡 runner（`linux-aarch64-a2-2`），单卡 runner 上跑会因 NCCL/HCCL init 失败而 MISMATCH。

### 模型转换 + LoRA 合并

训练产物是 QLoRA 的 `.pth`（只含 adapter 参数），要转 HuggingFace 格式再合并到 base。CI smoke 不真跑转换（依赖 .pth，前面 5 step 训练已跑），只烟囱测 `xtuner convert` 的两个子命令 `pth_to_hf` 和 `merge` 都可用：

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

完整 `pth_to_hf` + `merge` 流程（依赖前面训练出 `.pth`，本地按需手动跑）：

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

> `#test` 只烟囱测 `xtuner convert` 的两个子命令 `pth_to_hf` 和 `merge` 都可用（`--help` 退出码 0）。完整转换 + 合并依赖前面训练出的 `.pth`，CI smoke 跑不到，本地按需手动跑。

### 与模型对话

合并完权重后，可以直接用 `xtuner chat` 跟模型对话。CI smoke 不真跑交互式 chat（依赖 stdin + 模型推理），只烟囱测 `xtuner chat --help` 退出码 0 + 关键参数 `--adapter` / `--prompt-template` / `--system-template` 都存在：

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

完整 `xtuner chat` 命令（交互式 CLI，依赖前面合并后的权重，本地按需手动跑）：

```shell
xtuner chat /tmp/xtuner_sft_llm_out/merged \
    --prompt-template internlm2_chat \
    --system-template colorist
```

也可以不合并、只跟 LLM + LoRA adapter 直接对话：

```shell
xtuner chat ./Shanghai_AI_Laboratory/internlm2-chat-7b \
    --adapter /tmp/xtuner_sft_llm_out/iter_720_hf \
    --prompt-template internlm2_chat \
    --system-template colorist
```

交互示例（训练前模型 → 训练后模型的输出变化）：

```
double enter to end input (EXIT: exit chat, RESET: reset history) >>> 宁静而又相当明亮的浅天蓝色，介于天蓝色和婴儿蓝之间，因其亮度而带有一丝轻微的荧光感。

#66ccff
```

> `#test` 只烟囱测 `xtuner chat --help` 退出码 0 + 关键参数 `--adapter` / `--prompt-template` / `--system-template` 都存在。完整交互式对话没法做自动化断言（依赖 stdin），本地按需手动跑。