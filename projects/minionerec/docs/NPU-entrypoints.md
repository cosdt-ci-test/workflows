# 入口脚本审计：MiniOneRec NPU 适配的覆盖范围

`Quick-start-Ascend.md` 只跑通 **SFT** 与 **约束解码评估** 两条主路径。本文补齐其余入口脚本的验证，回答"改完能不能都用"：确认 13 个入口脚本都带 `npu_support`，且**导入期**（模块级 `torch.cuda` 使用、`device` 选择、`sys.path` 处理）在 NPU 上不报错。

只做**克隆 + 装依赖 + 导入**，不下载模型与数据，单次约 5 分钟。

---

## 1. 加载 CANN 环境

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

---

## 2. 克隆 MiniOneRec

将 `<UPSTREAM_REF>` 换成目标**分支、tag 或 commit**（本文验证 NPU 适配分支 `npu-support`）。

```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```

```shell #test-setup id="clone" load="upstream_ref>>UPSTREAM_REF"
git clone https://github.com/yzeyu71/MiniOneRec.git
cd MiniOneRec
git checkout <UPSTREAM_REF>
```

---

## 3. 安装依赖

与 `Quick-start-Ascend.md` 用同一份 `requirements-npu.txt`，再补上它注释里标为可选的 RL（`trl`、`scikit-learn`、`wandb`）与 SID 构建（`polars`、`openai`、`aiohttp`）包，使每个入口脚本都能被导入。

```shell #test-setup id="install"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd MiniOneRec
pip install -r requirements-npu.txt
pip install trl==0.24.0 scikit-learn wandb==0.22.2 polars openai requests aiohttp
```

---

## 4. 审计 13 个入口脚本

审计用 `exec_module` 加载脚本：它执行**模块级代码**、跳过 `if __name__ == '__main__'`，等价于 `python <script>` 的导入期而不触发训练。每个脚本都以**自身所在目录**为工作目录执行——`python <script>` 只把脚本目录放进 `sys.path`，不含仓库根目录，这是 `rq/` 下脚本最容易踩到的位置。

`rq/generate_indices.py` 没有 `__main__` 守卫（导入即训练），因此只做静态检查，不做模块导入。

```shell #test id="audit"
cd MiniOneRec
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python - <<'PY'
import pathlib
import subprocess
import sys

ROOT = pathlib.Path.cwd()
ENTRIES = [
    'sft.py',
    'sft_gpr.py',
    'ts_rec_sft.py',
    'evaluate.py',
    'rl.py',
    'rl_gpr.py',
    'sasrec.py',
    'rq/rqvae.py',
    'rq/rqkmeans_plus.py',
    'rq/generate_indices.py',
    'rq/generate_indices_plus.py',
    'rq/text2emb/amazon_text2emb.py',
    'rq/text2emb/amazon_text2emb_gpr.py',
]
NO_IMPORT = {'rq/generate_indices.py'}

# README 里另有一个导入 torch、但不碰设备的脚本：没有 shim 也应在 NPU 环境
# 里能导入，一并检查，避免"改完 torch 就崩"。其余脚本（convert_dataset.py、
# rqkmeans_*.py、merge/split/calc）不 import torch，与 NPU 无关。
EXTRA_IMPORTS = ['data/amazon18_data_process.py']


def run(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def last_line(text):
    return text.strip().splitlines()[-1] if text.strip() else '(no stderr)'


missing = [f for f in ENTRIES
           if 'import npu_support' not in (ROOT / f).read_text(encoding='utf-8')]
print(f'entry scripts importing the shim: {len(ENTRIES) - len(missing)}/{len(ENTRIES)}')

import_fail = []
importable = [f for f in ENTRIES if f not in NO_IMPORT] + EXTRA_IMPORTS
for f in importable:
    script = pathlib.Path(f)
    code = (
        'import importlib.util\n'
        f'spec = importlib.util.spec_from_file_location({script.stem!r}, {script.name!r})\n'
        'module = importlib.util.module_from_spec(spec)\n'
        'spec.loader.exec_module(module)\n'
    )
    r = run([sys.executable, '-c', code], ROOT / script.parent)
    if r.returncode:
        import_fail.append((f, last_line(r.stderr)))
print(f'entry module import: {len(importable) - len(import_fail)}/{len(importable)}')

remap = run([sys.executable, '-c',
             'import npu_support, torch; '
             'print(torch.cuda.is_available(), torch.cuda.device_count())'], ROOT)
print(f'cuda api mapped to npu: {remap.stdout.strip()}')

device = run([sys.executable, '-c',
              'import importlib.util\n'
              "spec = importlib.util.spec_from_file_location('evaluate', 'evaluate.py')\n"
              'module = importlib.util.module_from_spec(spec)\n'
              'spec.loader.exec_module(module)\n'
              'print(module.device)\n'], ROOT)
print(f'device selected by evaluate.py: {device.stdout.strip()}')

for f, err in missing + import_fail:
    print(f'FAIL {f}: {err}', file=sys.stderr)
sys.exit(1 if (missing or import_fail) else 0)
PY
```

输出结果如下：

```shell #test-result id="audit"
entry scripts importing the shim: 13/13
entry module import: 13/13
cuda api mapped to npu: True 1
device selected by evaluate.py: cuda
```

`device selected by evaluate.py: cuda` 是端到端证据：`evaluate.py` 在模块级写的是 `if torch.cuda.is_available(): device = "cuda"`，只有 `transfer_to_npu` 真的把 `torch.cuda` 接口重映射到 NPU，这一行才会取到 `cuda` 而不是 `cpu`。

---

## README Quickstart 步骤覆盖

上游 `README.md` 的 Quickstart 是 5 步，其中三步是**带占位符的模板脚本**（`--base_model your_model_path`、`exp_name="xxx"`、写死 8 卡），任何机器上都不能原样执行：

| README Quickstart | 本次如何处理 |
| --- | --- |
| 1. `conda create -n MiniOneRec python=3.11` | 不适用：容器内为 Python 3.12，不建 conda 环境 |
| 2. `pip install -r requirements.txt` | 换成 `requirements-npu.txt`，由 `Quick-start-Ascend.md` 第 4 节实际安装 |
| 3. `bash sft.sh` | 占位符 + `--nproc_per_node 8`；等价路径为 `python sft.py` 单卡冒烟 |
| 4. `bash rl.sh` | 未跑：`accelerate launch --num_processes 8`，且 GRPO 的 vLLM 路径为 CUDA-only |
| 5. `bash evaluate.sh` | 占位符 + 8 卡分片 + `merge.py`；等价路径为 `python evaluate.py` 单卡 |

可执行的等价序列见 `Quick-start-Ascend.md`（9 节，逐块由 CI 执行）。

## README 脚本覆盖对照

| README 命令 | 本次验证深度 |
| --- | --- |
| `bash sft.sh` | 实跑（`Quick-start-Ascend.md`，落盘 `model.safetensors`） |
| `bash evaluate.sh` | 实跑（199 条预测，`calc.py` 的 CC 为 0） |
| `bash rl.sh` | 导入（`rl.py` 依赖 `trl`/`deepspeed`/`sasrec.py`，实际训练未跑） |
| `bash data/amazon18_data_process.sh` | 导入（不碰设备；实跑需下载 Amazon 原始评论） |
| `bash rq/amazon_text2emb.sh` | 导入（实跑需原始元数据与文本嵌入） |
| `bash rq/rqvae.sh` | 导入（实跑需文本嵌入） |
| `bash rqkmeans_plus.sh` | 导入 |
| `bash rq/generate_indices_plus.sh` | 导入 |
| `python rq/generate_indices.py` | 静态（无 `__main__` 守卫，导入即训练） |
| `bash rqkmeans_constrained.sh`、`python rqkmeans_faiss.py` | 与 NPU 无关（`faiss` / `numpy` / `polars`，不 import torch） |
| `python convert_dataset.py` | 与 NPU 无关（不 import torch） |

结论：README 里**所有 import torch 的脚本**要么实跑、要么经过导入检查；剩下未实跑的只有依赖模型权重、原始数据或 CUDA-only vLLM 的那几条。

## 未覆盖范围

| 未验证项 | 原因 |
| --- | --- |
| RL / GRPO 实际训练 | vLLM 路径为 CUDA-only；`use_vllm=False` 的原生生成路径吞吐过低，不适合作冒烟 |
| 多卡 DDP | 单卡 runner；DeepSpeed 配置未在本审计中执行 |
| `rq/` 预处理的业务输出 | 需要 RQ-VAE checkpoint 与 `.npy` 嵌入，属数据准备而非 NPU 适配 |
| 训练/推理行为本身 | 由 `Quick-start-Ascend.md` 的 SFT + 评估覆盖 |
