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

在 `Quick-start-Ascend.md` 的冒烟依赖之外，补上 RL（`trl`、`scikit-learn`）与 RQ 预处理（`polars`、`openai`、`aiohttp`）所需的包，使每个入口脚本都能被导入。

```shell #test-setup id="install"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install torch==2.9.0 torch-npu==2.9.0.post2
pip install modelscope transformers==4.57.1 accelerate==1.10.1 datasets==4.2.0 \
    tokenizers==0.22.1 fire==0.7.1 pandas==2.2.2 tqdm bitsandbytes==0.48.1 \
    trl==0.24.0 scikit-learn polars openai requests aiohttp
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


def run(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def last_line(text):
    return text.strip().splitlines()[-1] if text.strip() else '(no stderr)'


missing = [f for f in ENTRIES
           if 'import npu_support' not in (ROOT / f).read_text(encoding='utf-8')]
print(f'entry scripts importing the shim: {len(ENTRIES) - len(missing)}/{len(ENTRIES)}')

import_fail = []
importable = [f for f in ENTRIES if f not in NO_IMPORT]
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
entry module import: 12/12
cuda api mapped to npu: True 1
device selected by evaluate.py: cuda
```

`device selected by evaluate.py: cuda` 是端到端证据：`evaluate.py` 在模块级写的是 `if torch.cuda.is_available(): device = "cuda"`，只有 `transfer_to_npu` 真的把 `torch.cuda` 接口重映射到 NPU，这一行才会取到 `cuda` 而不是 `cpu`。

---

## 未覆盖范围

| 未验证项 | 原因 |
| --- | --- |
| RL / GRPO 实际训练 | 其 vLLM 路径为 CUDA-only；`use_vllm=False` 的原生生成路径吞吐过低，不适合作冒烟，本次只验证 `rl.py` / `rl_gpr.py` 在 NPU 上可导入 |
| 多卡 DDP | 单卡 runner；DeepSpeed 配置未在本审计中执行 |
| `rq/generate_indices.py` 运行时 | 无 `__main__` 守卫，导入即训练，需要 RQ-VAE checkpoint 与 `.npy` 嵌入 |
| `rq/rqkmeans_faiss.py` / `rq/rqkmeans_constrained.py` | 不依赖 torch 设备（`faiss` / `numpy`+`polars`），与 NPU 适配无关 |
| 各脚本的实际业务逻辑 | 本次只覆盖导入期；训练/推理行为由 `Quick-start-Ascend.md` 的 SFT + 评估覆盖 |
