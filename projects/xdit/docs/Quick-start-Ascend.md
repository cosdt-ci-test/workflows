# xDiT 快速入门指南（Ascend NPU）

欢迎使用 xDiT！本指南帮助你在单卡昇腾 NPU 上安装 xDiT 并生成第一张图片。

## 🚀 系统要求

- **硬件**：Atlas 900 A2 / A3 训练系列产品（Ascend 910B4 / 910B 等），单卡 32 GB HBM 可跑本文全部内容
- **操作系统**：Linux（Ubuntu 22.04）
- **存储**：至少 40 GB 可用空间

| 组件 | 版本 | 来源 |
| --- | --- | --- |
| CANN | ≥ 8.5.1 | 环境搭建安装（CI 以 9.1.0 验证） |
| Python | ≥ 3.10 | 自备环境（CI 镜像为 3.12） |
| torch / torch_npu | 2.9.0 / 2.9.0.post2 | 下方安装 |
| triton | 3.5.* | 下方安装（xfuser 导入需要） |
| xfuser（xDiT） | PyPI 最新（CI 以 0.6.0 验证） | 下方安装 |

> 也可用带 CANN 的昇腾镜像（如 [ascendhub cann 镜像](https://www.hiascend.com/developer/ascendhub)）跳过 CANN 安装，其余步骤相同。

## 🐧 环境搭建

**安装 CANN**（≥ 8.5.1，与驱动配套，[快速安装脚本](https://ascend.github.io/docs/sources/ascend/quick_install.html)会自动识别卡型），安装完成后：

```shell
source ~/Ascend/ascend-toolkit/set_env.sh
```

**准备 Python 环境**：Python ≥ 3.10。

**安装 torch + torch_npu + triton**：

```shell #test-setup id="xdit-install-torch"
pip install uv
uv pip install "torch==2.9.0" "torch_npu==2.9.0.post2" "triton==3.5.*"
```

## 📦 安装 xDiT

**安装**（PyPI 包名 `xfuser`）：

```shell #test-setup id="xdit-install"
uv pip install xfuser "modelscope==1.37.0"
```

**验证安装**（全部就位时输出 `npu hccl True`）：

```shell #test id="xdit-install-verify"
python -c "
import torch
import torch_npu
from importlib.metadata import version
from xfuser.envs import get_device_name, get_torch_distributed_backend, _is_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('xfuser:', version('xfuser'))
print('npu dispatch:', get_device_name(), get_torch_distributed_backend(), bool(_is_npu()))
"
```

```shell #test-result id="xdit-install-verify" fuzzy='xxx' fuzzy='...'
...torch: 2.9.xxx
torch_npu: 2.9.xxx
xfuser: xxx
npu dispatch: npu hccl True
```

## 🎯 文生图

### 🚀 开始生成

用 [SD3 medium](https://modelscope.cn/models/stabilityai/stable-diffusion-3-medium-diffusers) 生成图片（约 30 GB，首次运行时自动下载到 ModelScope 默认缓存，无需手动下载；单卡、1 步、256×256；`torchrun` 负责初始化运行时环境）：

```shell #test id="xdit-sd3-smoke"
cat > sd3_npu.py <<'PY'
import os
import torch
import torch_npu
from modelscope import snapshot_download
from transformers import T5EncoderModel
from xfuser import xFuserArgs, xFuserStableDiffusion3Pipeline
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import get_runtime_state, get_world_group

parser = FlexibleArgumentParser(description="xFuser SD3 Arguments")
args = xFuserArgs.add_cli_args(parser).parse_args()
engine_args = xFuserArgs.from_cli_args(args)
engine_config, input_config = engine_args.create_config()
local_rank = get_world_group().local_rank

model_path = snapshot_download('stabilityai/stable-diffusion-3-medium-diffusers')
# T5-XXL 单独按 fp16 预载再传入，避免 pipeline 默认路径重复加载
text_encoder_3 = T5EncoderModel.from_pretrained(
    model_path, subfolder="text_encoder_3", torch_dtype=torch.float16
)
pipe = xFuserStableDiffusion3Pipeline.from_pretrained(
    pretrained_model_name_or_path=model_path,
    engine_config=engine_config,
    torch_dtype=torch.float16,
    text_encoder_3=text_encoder_3,
).to(f"npu:{local_rank}")
pipe.prepare_run(input_config)

output = pipe(
    height=input_config.height,
    width=input_config.width,
    prompt=input_config.prompt,
    num_inference_steps=input_config.num_inference_steps,
    output_type=input_config.output_type,
    guidance_scale=input_config.guidance_scale,
    generator=torch.Generator(device="npu").manual_seed(input_config.seed),
)
os.makedirs("results", exist_ok=True)
if pipe.is_dp_last_group():
    output.images[0].save("results/sd3_npu.png")
    print("saved: results/sd3_npu.png")
get_runtime_state().destroy_distributed_env()
PY
torchrun --nproc_per_node=1 sd3_npu.py \
    --prompt "a tiny test sketch" \
    --height 256 --width 256 \
    --num_inference_steps 1 \
    --seed 42
```

```shell #test-result id="xdit-sd3-smoke" fuzzy='...'
...saved: results/sd3_npu.png
```

### 输出校验

校验生成的图片完整有效（PNG 文件头魔数 + 大小 >50 KB 下限，防止空图 / 坏图）：

```shell #test id="xdit-sd3-output"
python - <<'PY'
import os
p = 'results/sd3_npu.png'
size = os.path.getsize(p)
assert size > 50_000, f'output too small: {size} bytes'
with open(p, 'rb') as fh:
    assert fh.read(8) == b'\x89PNG\r\n\x1a\n', 'not a png'
print('size:', size)
PY
```

```shell #test-result id="xdit-sd3-output" fuzzy='xxx'
size: xxx
```

> **注意**：如新开终端执行生成，先 `source ~/Ascend/ascend-toolkit/set_env.sh`。更多用法（多卡并行、更多模型）见 [xDiT examples](https://github.com/xdit-project/xDiT/tree/main/examples)。
