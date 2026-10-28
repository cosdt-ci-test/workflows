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

上游 `README.md` 的 `## 🚀 Quickstart` 是 5 步。其中 3 个启动脚本是**带占位符的模板**（`--base_model your_model_path`、`exp_name="xxx"`、写死 8 卡 / 8 进程），任何机器上都不能原样执行，所以验证的是**等价命令**而不是脚本本身：

| # | README Quickstart | 是否执行 | 说明 |
| --- | --- | --- | --- |
| 1 | `conda create -n MiniOneRec python=3.11` | 否 | 镜像自带 Python 3.12，不建 conda 环境 |
| 2 | `pip install -r requirements.txt` | 是，换文件 | `Quick-start-Ascend.md` 第 4 节装 `requirements-npu.txt`；`requirements.txt` 含 CUDA 专属包 |
| 3 | `bash sft.sh` | 部分 | 脚本未执行；第 8 节用 `python sft.py` 单卡实跑 |
| 4 | `bash rl.sh` | 否 | GRPO 的 vLLM 路径 CUDA-only；只做了 `rl.py` 导入检查 |
| 5 | `bash evaluate.sh` | 部分 | 脚本未执行；第 9 节用 `python evaluate.py` + `calc.py` 单进程实跑 |

第 3、5 步"部分"的含义：

| 维度 | `bash sft.sh` | 本次冒烟 |
| --- | --- | --- |
| 并行 | `torchrun --nproc_per_node 8`，进入 `sft.py` 的 `world_size != 1` DDP 分支（`device_map` 按 `LOCAL_RANK` 取卡、梯度累积除以卡数） | 单进程，`world_size == 1` |
| 数据量 | 全量，`cutoff_len` 取默认 512 | `sample 600`/子数据集，`cutoff_len 256` |
| batch | `batch_size 1024` / `micro_batch_size 16`（前者只决定梯度累积步数） | 8 / 2 |
| wandb | 传 `--wandb_project` / `--wandb_run_name` | 不传；且 `TrainingArguments` 里是 `report_to=None`，这两个参数只写进环境变量与 `run_name`，不影响训练路径 |

| 维度 | `bash evaluate.sh` | 本次冒烟 |
| --- | --- | --- |
| 分片 | `split.py` 切 8 份 → 8 个后台进程 → `wait` → `merge.py` 合并 | `head -n 200` 切一份，单进程 |
| 解码参数 | `num_beams 50`、`batch_size 8` | `num_beams 20`、`batch_size 2`（未传的参数取 `evaluate.py` 默认值） |
| 输出 | `results/<exp>/final_result_<category>.json` | `eval_result.json` |
| 选卡 | 逐进程 `CUDA_VISIBLE_DEVICES=$i` | `ASCEND_RT_VISIBLE_DEVICES=0` 单卡 |

因此 `split.py` / `merge.py` 没有在 NPU 上跑过——它们只读写 CSV/JSON、不 import torch，与适配无关。

另外 `evaluate.sh` 传了 `--temperature 1.0 --guidance_scale 1.0`，而 `evaluate.py` 的 `main()` 签名乃至全文都没有这两个参数。实测 `fire==0.7.1`（仓库 pin 的版本）的行为是**先完整执行函数体、再打印 `ERROR: Could not consume arg: --temperature` 并以退出码 2 结束**；`evaluate.sh` 没有 `set -e`、用 `&` + `wait` 收尾，所以链路仍能走到 `merge.py`/`calc.py`，只是每个分片进程返回 2。本次直接调用 `evaluate.py`，绕开了这层包装。

README Quickstart 本身也**不完整**：它没有 `git clone`，也没有预训练 SID 与数据的下载步骤（只说 "we provide"，实际在 HF `kkknight/MiniOneRec`）。这两步由 `Quick-start-Ascend.md` 第 3、6 节补上，其中基座模型取 Qwen2.5-0.5B-**base**、数据只拉 Industrial 子集（约 27 MB，不含 `.npy` 嵌入）。

可执行的等价序列见 `Quick-start-Ascend.md`（9 节，逐块由 CI 执行）。

## README 脚本覆盖对照

| README 命令 | 本次验证深度 |
| --- | --- |
| `bash sft.sh` | 等价命令实跑（`python sft.py`，落盘 `model.safetensors`）；脚本本身含占位符与 8 卡，未执行 |
| `bash evaluate.sh` | 等价命令实跑（199 条预测，`calc.py` 的 CC 为 0）；脚本本身含占位符与 8 卡分片，未执行 |
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

| 未验证项 | 原因 | 能否在 NPU 上补验 |
| --- | --- | --- |
| `bash sft.sh` 的 8 卡 DDP | 当前用单卡 runner | **可以，零改动**：换 `linux-aarch64-a2-8`、`container_options` 补挂 `davinci2..7`，命令保留 `torchrun --nproc_per_node 8`；样本仍要限 `--sample`（全量 36 259 行 × 3 个子数据集 × `sft.sh` 没传的默认 10 epoch） |
| `bash evaluate.sh` 的多卡分片 | `CUDA_VISIBLE_DEVICES` 在 NPU 上不生效——`transfer_to_npu` 全文不处理可见设备环境变量，8 个进程都会落到 `npu:0`（`evaluate.py:54` 自己设的那次也在 torch 初始化之后，同样是空操作） | **可以，但要改脚本**：`CUDA_VISIBLE_DEVICES=$i` → `ASCEND_RT_VISIBLE_DEVICES=$i`；或只留单分片，覆盖 `split.py` → `evaluate.py` → `merge.py` → `calc.py` 链路 |
| RL / GRPO 实际训练 | 四个独立阻塞：`GRPOConfig(optim="paged_adamw_32bit")` 落到 bitsandbytes 的 CUDA kernel 与 paged 分配器；`report_to="wandb"` 写死；`rl.py:85` 里 `sample = -1` 写死，命令行无法缩小数据；`config/zero2_opt.yaml` 是 DEEPSPEED 类型、需要 deepspeed | **不建议**：前三个都要改上游代码（`optim` / `report_to` / `sample`），属于改行为而非 NPU 适配 |
| `rq/` 预处理的业务输出 | 需要 RQ-VAE checkpoint 与 `.npy` 嵌入，属数据准备 | 不补（与 NPU 无关） |
| 训练/推理行为本身 | — | 由 `Quick-start-Ascend.md` 的 SFT + 评估覆盖 |

补充：`minionerec_trainer.py:426` 读的是 `self.args.vllm_device`，而 pin 的 `trl==0.24.0` 的 `GRPOConfig` 已没有该字段（改为 `vllm_mode`）。`rl.py` 从不传 `use_vllm`，默认 `False`，所以这条分支目前走不到——也就是说 RL 留在 `use_vllm=False` 是**版本决定的硬约束**，不是可选优化。
