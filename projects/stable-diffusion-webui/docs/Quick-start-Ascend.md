# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上部署 stable-diffusion-webui（A1111），并通过其 REST API 完成一次**无头**文生图推理。

> 单卡昇腾 NPU 上以 API 模式（`--nowebui`）运行 stable-diffusion-webui，模型经 **ModelScope** 下载，全程无需人工交互；交互式 Web UI 用户去掉 `--nowebui` 后浏览器访问即可。

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
| stable-diffusion-webui | master（原生 NPU 支持） |
| 模型 | `AI-ModelScope/sd-turbo`（经 ModelScope 下载，1 步采样） |

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

## 获取代码

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到工作流注入的最新 release tag：

```shell #test id="clone-repo" load="upstream_ref>>ref"
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui
git checkout <ref>
echo "HEAD $(git log -1 --format=%h)"
```

输出结果如下：

```shell #test-result id="clone-repo" fuzzy='xxx'
HEAD xxx
```

\<ref> 为流水线注入的上游最新 release tag。

## 安装依赖

stable-diffusion-webui 的依赖链（`facexlib` → `opencv-python`）在 import 时动态链接 `libGL.so.1` 与 `libglib-2.0`，基础 CANN 镜像不含这些运行库，需先补装（`libgl1` 提供 libGL，`libglib2.0-0` 提供 libglib/libgthread；GUI 相关的 X11/xcb 库 opencv wheel 自带，无需安装）：

```shell #test-setup
apt-get update -qq && apt-get install -y -qq --no-install-recommends libgl1 libglib2.0-0
```

stable-diffusion-webui 的依赖 pin 为 py3.10 时代版本（部分包没有 cp312 wheel），因此用 `uv` 建一个**独立的 py3.10 venv**（`uv` 会自动托管下载 CPython 3.10），所有安装与运行都走该 venv：

```shell #test-setup
uv venv --python 3.10 --seed /tmp/sd-webui-venv
/tmp/sd-webui-venv/bin/python -m pip install modelscope
/tmp/sd-webui-venv/bin/python -m pip install torch==2.9.0 torchvision
/tmp/sd-webui-venv/bin/python -m pip install torch_npu==2.9.0.post2
cd stable-diffusion-webui
/tmp/sd-webui-venv/bin/python -m pip install -r requirements.txt
/tmp/sd-webui-venv/bin/python -m pip install "setuptools<81" wheel
/tmp/sd-webui-venv/bin/python -m pip install --no-build-isolation "https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip"
```

> - `uv venv` 创建的 venv 默认不含 pip，`--seed` 会预装 pip；`uv` 会自动托管下载 CPython 3.10。
> - 旧版 CLIP 的 `setup.py` 依赖 `pkg_resources`（新版 setuptools 已移除），因此预装 `setuptools<81` 后用 `--no-build-isolation` 从源码安装；launch.py 检测到 `clip` 已装会自动跳过自身的 github zip 安装。
> - `torch==2.9.0` + `torch_npu==2.9.0.post2` 与本机 CANN 9.1 配套，且均有 py3.10 wheel；`torch_npu` 来自华为云昇腾源（流水线已注入 extra index）。
> - `requirements.txt` 按上游原样安装，不做任何 pin 改动。

验证依赖可用：

```shell #test id="install-deps"
/tmp/sd-webui-venv/bin/python -c "import modelscope, gradio, fastapi; print('deps ok')"
```

输出结果如下：

```shell #test-result id="install-deps"
deps ok
```

> 依赖清单以 stable-diffusion-webui 仓库 `requirements.txt` 为准；`modelscope` 为 ModelScope 下载模型所需，需显式安装。

## 下载模型

```shell #test-setup store="model_dir"
/tmp/sd-webui-venv/bin/python -c "
import time, sys
from pathlib import Path
from modelscope import snapshot_download
d = None
for i in range(3):
    try:
        d = snapshot_download('AI-ModelScope/sd-turbo', revision='master')
        if (Path(d) / 'sd_turbo.safetensors').is_file():
            break
        print('attempt %d/3: file not complete, retrying' % (i+1), file=sys.stderr)
        d = None
    except Exception as e:
        print('attempt %d/3 failed: %s' % (i+1, e), file=sys.stderr)
        time.sleep(15)
assert d, 'snapshot_download failed after 3 attempts'
print(d)
" > /tmp/sd-turbo-model-dir.txt && tail -n 1 /tmp/sd-turbo-model-dir.txt
```

> `tail -n 1` 过滤下载进度输出，仅保留模型目录路径；sd-turbo 为 1 步采样的蒸馏模型，下载约 3.4GB，首次运行请耐心等待。

## 无头文生图（单卡 NPU）

以 API 模式启动 stable-diffusion-webui，随后用 curl 验证文生图推理：

```shell #test-setup store="api_pid" load="model_dir>>ckpt"
cd stable-diffusion-webui
mkdir -p db
export GIT_CONFIG_NOSYSTEM=1
export STABLE_DIFFUSION_REPO=https://github.com/w-e-w/stablediffusion.git
python -c "
p = 'modules/devices.py'
t = open(p).read()
t = t.replace(
    'if has_xpu() or has_mps() or cuda_no_autocast():',
    'if has_xpu() or has_mps() or cuda_no_autocast() or npu_specific.has_npu:'
)
open(p, 'w').write(t)
"
nohup /tmp/sd-webui-venv/bin/python launch.py --nowebui --skip-torch-cuda-test --ckpt <ckpt>/sd_turbo.safetensors --port 7861 > /tmp/sdwebui.log 2>&1 &
echo $!
```

- `mkdir -p db`：A1111 把图像历史写入 `db/` 下的 SQLite 数据库，全新 clone 里该目录不存在（被 .gitignore），`--nowebui` 启动不自动创建 → 首次 txt2img 报 `OperationalError: unable to open database file`，这里显式建目录。
- `GIT_CONFIG_NOSYSTEM=1`：launch 会 git clone 数个 assets 仓库，runner 镜像的 `/etc/gitconfig` 把 github.com 重写到需认证的代理，这里让 git 忽略该配置、直连 github。
- `STABLE_DIFFUSION_REPO`：上游默认指向的 `Stability-AI/stablediffusion` 已被删除（2025.12 起，GitHub 返回 404），官方用社区 fork `w-e-w/stablediffusion` 兜底（commit hash 不变）；此处通过环境变量覆盖，后续上游修复后可移除。
- `--nowebui`：API 模式（FastAPI，默认端口 7861），无 Gradio 界面，适合自动化与 CI。
- `--skip-torch-cuda-test`：允许非 CUDA 设备（NPU）。
- `python -c ".../npu_specific.has_npu..."`：A1111 的 `autocast()` 只对 CUDA/MPS/XPU 做 `manual_cast`（自动把输入转 fp16），NPU 上落到 `torch.autocast("cuda")`（no-op）→ float32 输入撞上 fp16 权重 → `RuntimeError: Input type (float) and bias type (c10::Half)`。这行补丁把 NPU 纳入 `manual_cast` 分支（`npu_specific` 已在 `devices.py` 顶部 import），上游未来原生支持后可移除。
- `--ckpt <目录>/sd_turbo.safetensors`：A1111 的 checkpoint 加载器需要**单文件** .safetensors；ModelScope 快照根目录提供了合并后的 `sd_turbo.safetensors`（其余为 diffusers 分片格式），此处指向该单文件。
- 设备选择由 `modules/npu_specific.py` 自动完成（检测到 torch_npu 即用 npu:0）。

等待 API 就绪：

```shell #test id="wait-ready"
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:7861/docs > /dev/null && break
  sleep 5
done
curl -sf http://127.0.0.1:7861/docs > /dev/null && echo "api ready" || { echo "api not ready"; tail -n 100 /tmp/sdwebui.log; exit 1; }
```

输出结果如下：

```shell #test-result id="wait-ready"
api ready
```

发起文生图推理：

```shell #test id="txt2img"
curl -s -X POST http://127.0.0.1:7861/sdapi/v1/txt2img \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a cute cat", "steps": 1, "cfg_scale": 1.0, "width": 512, "height": 512}' \
  > /tmp/sd-turbo-resp.json
python -c "
import json, base64, sys
try:
    r = json.load(open('/tmp/sd-turbo-resp.json'))
    assert 'images' in r, 'txt2img error response: ' + json.dumps(r)[:2000]
except Exception as e:
    print('txt2img failed:', e, file=sys.stderr)
    print('--- /tmp/sdwebui.log tail ---', file=sys.stderr)
    print(open('/tmp/sdwebui.log').read()[-8000:], file=sys.stderr)
    raise SystemExit(1)
imgs = r['images']
print('txt2img images:', len(imgs))
open('/tmp/sd-turbo-out.png', 'wb').write(base64.b64decode(imgs[0]))
"
```

输出结果如下：

```shell #test-result id="txt2img"
txt2img images: 1
```

> - sd-turbo 为蒸馏模型，1 步采样 + `cfg_scale=1.0`（等效无引导）即出图。
> - 返回 JSON 的 `images` 为 base64 PNG，此处解码落盘 `/tmp/sd-turbo-out.png`。

## 清理

<!--
```shell #test-setup load="api_pid>>pid"
kill -9 <pid> 2>/dev/null
sleep 2
echo "api stopped"
```
-->
