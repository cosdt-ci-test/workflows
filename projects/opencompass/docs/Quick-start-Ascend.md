# 快速开始：在昇腾 NPU 上用 OpenCompass 跑一次 GSM8K 评测

> **阅读本文前**，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。

[OpenCompass](https://github.com/open-compass/opencompass) 是开源大模型评测框架。本文用原生 Hugging Face 封装和 `torch_npu`，在单卡 910B 上对 [Qwen2-0.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2-0.5B-Instruct) 跑 `demo_gsm8k_chat_gen`（64 条样本）。

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
| OpenCompass | 从 GitHub 克隆 Release（撰写时 `0.5.4`），见下文 |
| 模型 / 数据 | ModelScope 上的 Qwen2-0.5B-Instruct 与 `opencompass/gsm8k` |

**配套机器**：Atlas 900 A2 PODc（Ascend 910B4）。**配套镜像**：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

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

命令退出码应为 0，并打印设备列表。表格中的功耗、HBM 占用每次不同，不必与任何样例逐字一致。

若 `npu-smi` 找不到，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载（如 `/dev/davinci0`）。

### 2.2 确认工具可用

```shell #test-setup
test -n "$ASCEND_HOME_PATH"
command -v npu-smi
```

`ASCEND_HOME_PATH` 应非空，`npu-smi` 应能找到。

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy="xxx"
Python 3.12.xxx
```

---

## 3. 安装 PyTorch NPU 栈

昇腾上的 `torch_npu` 要从华为 PyPI 额外索引安装，并钉死与 CANN 匹配的版本。`numpy` 和 `pyyaml` 也要一起装：`torch_npu` 的 wheel 没有声明这两项依赖，缺了会在 `import torch_npu` 之前就失败。

```shell #test id="install-torch"
python -m pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
  torch==2.9.0 torch_npu==2.9.0.post2 'numpy>=1.23.4,<2' pyyaml
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

## 4. 安装 OpenCompass

源码装到 `$HOME/opencompass-qs`。`OPENCOMPASS_REF` 默认是当前 Release `0.5.4`；`--no-deps` 之后按 `runtime.txt` 装导入期依赖，但跳过 `torch` 和跟 GitHub `master` 的 `rouge_chinese`。

<!--
```shell #test-setup store="oc_ref"
echo "${OPENCOMPASS_REF:-0.5.4}"
```
-->

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/cosdt-ci-test/opencompass'
ref="${OPENCOMPASS_REF:-0.5.4}"
cached="$ci/src/$ref"
dest="$HOME/opencompass-qs/opencompass"
mkdir -p "$HOME/opencompass-qs"
if [ -d "$cached/.git" ] && [ -f "$cached/opencompass/__init__.py" ]; then
  if git -C "$cached" describe --tags --exact-match 2>/dev/null | grep -qx "$ref"; then
    rm -rf "$dest"
    cp -a "$cached" "$dest"
  else
    rm -rf "$cached"
  fi
fi
```
-->

```shell #test id="install-opencompass" load="oc_ref>>ref"
mkdir -p "$HOME/opencompass-qs"
cd "$HOME/opencompass-qs"
OPENCOMPASS_REF="${OPENCOMPASS_REF:-0.5.4}"
if [ ! -d opencompass/.git ]; then
  GIT_TERMINAL_PROMPT=0 GIT_HTTP_VERSION=HTTP/1.1
  for _ in 1 2 3; do
    git clone --depth 1 --branch "$OPENCOMPASS_REF" \
      https://github.com/open-compass/opencompass.git opencompass && break
    rm -rf opencompass
    sleep 5
  done
fi
cd opencompass
python -m pip install -e . --no-deps
python - <<'PY'
from pathlib import Path
rows = []
for line in Path('requirements/runtime.txt').read_text().splitlines():
    item = line.strip()
    if not item or item.startswith('#') or item.startswith('torch') or 'rouge_chinese' in item:
        continue
    rows.append(item)
Path('../runtime-local.txt').write_text('\n'.join(rows) + '\n')
PY
python -m pip install \
  torch==2.9.0 \
  'transformers>=4.37.0,<5' \
  'datasets>=3.0.0,<4.0.0' \
  'numpy>=1.23.4,<2.0.0' \
  'pandas>=2.0,<3' \
  modelscope \
  rouge-chinese \
  -r "$HOME/opencompass-qs/runtime-local.txt"
python -c "import torch, torch_npu, opencompass; print('torch', torch.__version__); print('opencompass', opencompass.__version__); print('npu_available', torch.npu.is_available())"
```

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/cosdt-ci-test/opencompass'
ref="${OPENCOMPASS_REF:-0.5.4}"
src="$HOME/opencompass-qs/opencompass"
if [ -d "$src/.git" ] && [ -f "$src/opencompass/__init__.py" ] && [ ! -d "$ci/src/$ref/.git" ]; then
  mkdir -p "$ci/src"
  rm -rf "$ci/src/$ref.part"
  cp -a "$src" "$ci/src/$ref.part"
  mv "$ci/src/$ref.part" "$ci/src/$ref"
fi
```
-->

输出结果如下：

```shell #test-result id="install-opencompass" load="oc_ref>>ref"
...
torch 2.9.0...
opencompass <ref>
npu_available True
```

`npu_available` 仍必须是 `True`。若 `torch` 不再以 `2.9.0` 开头，说明后面的包把 NPU 栈换掉了，卸掉后按第 3–4 节重装。

---

## 5. 准备模型和数据

权重从 ModelScope 拉取，避免直连 Hugging Face。第一次大约 1 GB，之后走本地缓存。

```shell #test-setup store="model_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen2-0.5B-Instruct'))"
```

确认权重目录完整：

```shell #test id="check-model" load="model_path>>model_path"
test -f "<model_path>/config.json" && echo has_config True
```

输出结果如下：

```shell #test-result id="check-model"
has_config True
```

GSM8K 在评测时通过 `DATASET_SOURCE=ModelScope` 从 [opencompass/gsm8k](https://www.modelscope.cn/datasets/opencompass/gsm8k) 读取。

---

## 6. 在 NPU 上评测

下面两份文件放在源码根目录。模型子类在生成前打印设备；评测配置把生成长度压到 32。`--debug` 让调度器在当前进程里跑，日志直接出现在终端。

```shell #test id="eval-gsm8k"
MODEL_PATH=$(python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen2-0.5B-Instruct'))")
cd "$HOME/opencompass-qs/opencompass"
cat > npu_chat.py <<'PY'
from opencompass.models.huggingface_above_v4_33 import HuggingFacewithChatTemplate
from opencompass.registry import MODELS


@MODELS.register_module()
class HuggingFaceNPUChat(HuggingFacewithChatTemplate):
    def generate(self, inputs, max_out_len, **kwargs):
        print('\nopencompass_model_device', next(self.model.parameters()).device, flush=True)
        return super().generate(inputs, max_out_len, **kwargs)
PY
cat > eval_qwen2_gsm8k.py <<PY
from mmengine.config import read_base
from npu_chat import HuggingFaceNPUChat

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets

gsm8k_datasets[0]['infer_cfg']['inferencer']['max_out_len'] = 32

datasets = gsm8k_datasets
models = [
    dict(
        type=HuggingFaceNPUChat,
        abbr='qwen2-0.5b-instruct',
        path=r'$MODEL_PATH',
        max_out_len=32,
        batch_size=4,
        model_kwargs=dict(torch_dtype='bfloat16'),
        run_cfg=dict(num_gpus=1),
    )
]
PY
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export DATASET_SOURCE=ModelScope
opencompass eval_qwen2_gsm8k.py --debug -w "$HOME/opencompass-qs/work" 2>&1
echo '--- summary ---'
cat "$HOME/opencompass-qs/work"/*/summary/summary_*.txt
```

输出结果如下：

```shell #test-result id="eval-gsm8k"
...
opencompass_model_device npu:0
...
tabulate format
...
demo_gsm8k ... accuracy ... gen ...
...
```

`opencompass_model_device` 必须是 `npu:0`。若打印 `cpu`，这次生成没有上 NPU，退出码 0 也不能当成功。

---

## 故障排查

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| `torch.npu.is_available()` 为 `False` | 未 `source set_env.sh`，或设备未挂进容器 | 重做第 1–3 节 |
| pip 把 `torch` 换成 CPU / CUDA 轮子 | 安装 OpenCompass 时没加 `--no-deps` | 卸掉后按第 3–4 节重装 |
| `opencompass_model_device` 为 `cpu` | `torch_npu` 没装上，或可见设备为空 | 检查 `npu_available` 和 `ASCEND_RT_VISIBLE_DEVICES` |
| ModelScope 下载超时或落到 HTML | 出口到 modelscope.cn 失败 | 检查网络后重跑；不要改成 Hugging Face 直连 |
| `git clone` 很慢或中断 | 国内直连 GitHub 不稳定 | 重试同一条克隆命令 |
| `from npu_chat import HuggingFaceNPUChat` 失败 | 评测时当前目录不是源码根目录 | 在 `$HOME/opencompass-qs/opencompass` 下按第 6 节导出 `PYTHONPATH` |
