# Quick Start (Ascend NPU)

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载（`/dev/davinci*` 等）。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，32 GB × 2 — CANN / 驱动预留约 2.5 GB，每张卡 PyTorch 实测可用 ~30 GB）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.12.0 |
| torch_npu | 2.12.0 |
| triton-ascend | 3.5.0+dev20260701（Ascend nightly 源） |
| modelscope | 最新release |
| torchtitan | 最新 release |
| 训练配置 | 单卡 Step 12：`torchtitan/models/llama3/config_registry.py::llama3_debugmodel`（debugmodel：dim=256 / 6 层 / 16 head / vocab 2048，~6 M 参数）；多卡 Step 13：`torchtitan/models/llama3/config_registry.py::llama3_8b`（Llama 3 8B：dim=4096 / 32 层 / 32 head / 8 kv head，FSDP shard=2 + cpu_offload + 全量 bf16 装得下） |


### 检查前置是否满足

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

装 `torch` + `torch_npu` ：

```shell #test-setup
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.12.0
uv pip install --extra-index-url https://mirrors.aliyun.com/pypi/simple torch_npu==2.12.0
```

检查 torch / torch_npu 是否装好且 NPU 设备可用：

```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-torch" fuzzy='xxx'
torch= xxx
torch_npu= xxx
is_available: True
count: 2
```

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

### 安装 modelscope

```shell #test-setup
uv pip install modelscope
```

打印安装版本：
```shell #test id="modelscope-install"
python -c "import modelscope; print('modelscope', modelscope.__version__)"
```

输出结果如下：
```shell #test-result id="modelscope-install" fuzzy='xxx'
modelscope xxx
```
### 安装 triton-ascend

NPU 上 `torch.compile`/inductor 走到 `torch_npu._inductor` 时需要 Ascend 的 Triton fork **triton-ascend**（为 `triton` 模块提供 Ascend 后端）；社区版 `triton` 只有 CUDA 后端，装了会在训练第一步报 `RuntimeError: 0 active drivers ([]). There should only be one.`。triton-ascend 的版本号对齐它 fork 的 triton 基线：torch 2.12 配套 triton 3.5，因此用 Ascend nightly 源 3.5.0 线的末位构建。注意华为云源的 triton-ascend（3.2.1 起的稳定版与全部 nightly）都声明 `triton==3.5.0` 依赖、文件设计为覆盖社区版目录：稳定 3.2.x 的 fork 基线（3.2）与所钉社区版（3.5.0）错配，混装后 `import triton` 报 `cannot import name 'Language'`；nightly 3.5.0 线基线匹配且 wheel 为完整 fork 可独立成立，但 `--no-deps` 仍然必要——不让社区版进环境，`triton/` 目录只归属 triton-ascend 一个包。`--no-deps` 跳过的依赖里，被 Ascend 后端运行期 import 的只有 **pybind11**（后端 utils/driver 在首次编译 kernel 时用它把 `npu_utils.cpp` 现场编成扩展，需要容器里有 g++/clang++ 和已 source 的 CANN 环境），单独补装；其余（numpy/pytest/pandas 等）是上游打包夹带的测试依赖，运行期不 import：

```shell #test-setup
uv pip install --no-deps --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi/nightly triton-ascend==3.5.0+dev20260701
uv pip install pybind11
```

打印安装版本：
```shell #test id="triton-install"
python -c "from importlib.metadata import version; print('triton-ascend', version('triton-ascend'))"
```

输出结果如下：
```shell #test-result id="triton-install" fuzzy='xxx'
triton-ascend xxx
```

## 安装 torchtitan

### 通过 uv 安装

torchtitan 上 PyPI 有 `py3-none-any` 的纯 Python wheel（450 KB），不依赖 native build——`uv pip install torchtitan` 直接装最新 stable release，不用管版本号：

```shell #test id="torchtitan-install-binary"
uv pip install --extra-index-url https://mirrors.aliyun.com/pypi/simple/ torchtitan
python -c "import torchtitan; print('torchtitan', torchtitan.__version__)"
```

输出结果类似如下：

```shell #test-result id="torchtitan-install-binary" fuzzy='xxx'
torchtitan xxx
```
- xxx 表示最新的版本号

<!--
```shell #test-setup
uv pip uninstall torchtitan -y
```
-->

### 从源码安装

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="torchtitan-install-source" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/pytorch/torchtitan.git
cd torchtitan
uv pip install -e .
python -c "import torchtitan; print('torchtitan', torchtitan.__version__)"
```
\<ref> 为工作流注入的最新 release tag

输出结果类似如下：

```shell #test-result id="torchtitan-install-source" fuzzy='xxx'
torchtitan xxx
```
- xxx 表示最新的版本号

## 跑训练

### 下载 tokenizer

```shell #test-setup id="modelscope-download-tokenizer" store="ms_tokenizer_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('LLM-Research/Llama-3.2-1B', allow_patterns=['*.json', '*.model', 'tokenizer*']))" | tail -n 1
```

> 输出的路径用于后续「单卡训练」和「多卡训练」章节。

验证 tokenizer 关键文件都落盘：

```shell #test id="modelscope-verify-tokenizer" load="ms_tokenizer_path>>ms_tokenizer_path"
ls <ms_tokenizer_path>
```

```shell #test-result id="modelscope-verify-tokenizer" fuzzy='...' 
config.json
configuration.json
generation_config.json
original
special_tokens_map.json
tokenizer.json
tokenizer_config.json
```

文件名是 Llama 3 tokenizer 必备文件，确认 snapshot_download 命中正确。

### 兼容性补丁

torchtitan v0.3.0 + torch 2.12 + torch_npu 2.12.0 + triton-ascend 3.5.0 是一个双方生态都未验证过的组合，共三个根因、若干处 `sed`，每处都有明确根因和退役条件。

**根因一：torchtitan v0.3.0 按 torch 2.14 nightly 开发（release notes 的 Compatibility 表写明 validated with PyTorch 2.14.0），而 NPU 全家桶最高只配套到 torch 2.12。** v0.3.0 的语言模型路径强制 flex/varlen attention（`sdpa` 被上游显式禁用），flex 必经 inductor，撞上三处断层：

1. `decoder.py::_create_flex_attention_mask` 给 `create_block_mask()` 传 `separate_full_blocks` 关键字参数——torch ≥2.13 才进入稳定版签名，torch 2.12 直接抛 `TypeError`。torch 2.12 内部本就固定 `separate_full_blocks=True`，删掉参数语义不变。
2. flex mask 的 mask_mod 图含 cumsum/scatter，inductor 会融合出**双归约轴 kernel**，而 torch_npu 的 NPU codegen 不支持双归约（其源码自述 "Currently npu don't support multi-reduction ranges trees"），生成残缺代码报 `NameError('r2 is not defined')` / `No valid triton configs`。`create_block_mask` 只是构建 BlockMask 的一次性张量计算，去掉 `attention.py` 里硬编码的 `torch.compile`、改跑 eager，语义不变。
3. torchtitan 默认开 `max_autotune` + `coordinate_descent_tuning`（逐 config 编译+实测搜索），NPU 上每个 config 都要走一遍 bisheng 编译，极易吃满命令超时；对 2 步 smoke 只影响性能不影响语义，关掉（torchtitan 源码注释自己也推荐 "keep max_autotune disabled for faster compilation"）。

**根因二：torch_npu 2.12.0 的 inductor 集成 × triton-ascend 3.5.0 fork——该组合双方都没发布验证过（torch_npu 上游 master 已修复、未回合 2.12.0；Ascend 生态整体还停在 torch 2.10）。** 三处 `sed` 全部镜像 torch_npu master 的官方修法：

1. `codegen/triton.py::find_axis_in_load_store` 把 codegen 行当字符串调 `line.find(...)`，而 torch 2.12 inductor 的行是 `DeferredLine` 对象——四个缓冲区循环统一先解包再使用（master 的 `_iter_codegen_lines()` 修法）。
2. `codegen/common.py::get_system` 与 `patch_triton_hash` 从 `triton.compiler.compiler` import `triton_key`，triton-ascend fork 已把它挪到 `triton.runtime.cache`——改 import 路径，`make_backend` 留在原位置（master 修法）。
3. `npu_triton_heuristics.py::make_launcher` 引用 `CompiledKernel.launch_enter/exit_hook` 类属性，triton 3.5 基线已把 hooks 挪到 `triton.knobs.runtime`——三处引用统一指向新位置（fork 的 `launch_metadata` 内部自带 hook 判空，无条件调用安全）。
4. torch 2.12.0 核心的 `flex_attention.py::_validate_device` 设备白名单 `{"cuda", "cpu", "xpu", "hpu"}` 不含 `npu`，torch_npu 2.12.0 没有 patch 它——往白名单里加 `"npu"`。
5. torch_npu 2.12.0 的 `_inductor/lowering.py` 在启动时把 torch 注册的全部 inductor lowering 过一遍自己的白名单（`lowering_op_list.py::GENERATE_LIST`，约 70 个基础算子 + `invoke_subgraph`/`cond`），**不在白名单的一律替换成 eager fallback**——flex_attention / flex_attention_backward 的 Triton 模板 lowering（表现为 `LoweringException: 'Subgraph' object has no attribute 'dtype'`）和 mask_mod 子图里 `offsets[document_id[q_idx]]` 需要的 `aten.index`（表现为 `SubgraphLoweringException: Buffers cannot be created while lowering a pointwise subgraph`）都是这样被抹掉的。把这三项加进白名单（stock torch 对 `aten.index` 本有正经 lowering，仅布尔索引时才回落 ATen）。
6. torch_npu 2.12.0 的 `NPUTritonScheduling.define_kernel` 签名要求第 4 个参数 `traced_graph_hash`（按 torch 2.12-dev 写的），而 torch 2.12.0 正式版的模板 codegen 路径只传 3 个参数——flex 模板 kernel 构建时报 `missing 1 required positional argument: 'traced_graph_hash'`。给该参数加默认值 `None`（其函数体内本就有 `if traced_graph_hash:` 空值 guard，torch_npu 自己的 4 参调用不受影响）。

安装侧另有两点配合（见「安装 triton-ascend」一节）：`--no-deps` 防止 wheel 声明的社区版 `triton==3.5.0` 依赖混入覆盖 fork 文件；单独补装被跳过依赖里唯一被运行期 import 的 `pybind11`。

**根因三：NPU 算子覆盖缺口。** llama3 注册表默认用 `ComplexRoPE`（complex64 缓存做旋转位置编码），forward 里 `rope_cache[positions]` 的索引落到 CANN 的 `aclnnIndex` 算子，而该算子不支持 `DT_COMPLEX64`（报 `AclNN_Parameter_Error ... not implemented for DT_COMPLEX64`）。换成数学等价的实数实现 `CosSinRoPE`（cos/sin 缓存 + rotate-half，全程实数张量）即可；它唯一的限制是不支持 llama scaling，需一并把 `scaling="llama"` 改为 `"none"`——llama scaling 只影响 >8k 长上下文的频率插值，对本文档 256/8192 seq 的 smoke 训练数值无影响。`sed` 作用于 `torchtitan/models/llama3/__init__.py`（import、6 处 `ComplexRoPE.Config`、6 处 scaling 一并替换）。

torch_npu 侧 `sed`（作用于已安装包，训练前执行一次）：

```shell #test-setup
TN_DIR="$(python -c 'import torch_npu, os; print(os.path.dirname(torch_npu.__file__))')"
sed -i -E "s/for line in self\.(loads|compute|post_loop_store|stores)\._lines:/for line in [l.line if isinstance(l, DeferredLine) else l for l in self.\1._lines]:/" "$TN_DIR/_inductor/codegen/triton.py"
sed -i -E "s/^([[:space:]]*)from triton\.compiler\.compiler import triton_key, make_backend$/\1from triton.runtime.cache import triton_key\n\1from triton.compiler.compiler import make_backend/" "$TN_DIR/_inductor/codegen/triton.py"
sed -i "s/from triton\.compiler\.compiler import triton_key$/from triton.runtime.cache import triton_key/" "$TN_DIR/_inductor/codegen/common.py"
sed -i "s/binary\.__class__\.launch_enter_hook/__import__(\"triton\").knobs.runtime.launch_enter_hook/g; s/binary\.__class__\.launch_exit_hook/__import__(\"triton\").knobs.runtime.launch_exit_hook/g" "$TN_DIR/_inductor/npu_triton_heuristics.py"
sed -i 's/supported_devices = {"cuda", "cpu", "xpu", "hpu"}/supported_devices = {"cuda", "cpu", "xpu", "hpu", "npu"}/' "$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')/nn/attention/flex_attention.py"
sed -i 's/^    torch.ops.higher_order.invoke_subgraph,$/    torch.ops.higher_order.invoke_subgraph,\n    torch.ops.higher_order.flex_attention,\n    torch.ops.higher_order.flex_attention_backward,\n    aten.index,/' "$TN_DIR/_inductor/lowering_op_list.py"
sed -i 's/def define_kernel(self, src_code, node_schedule, kernel, traced_graph_hash: str):/def define_kernel(self, src_code, node_schedule, kernel, traced_graph_hash: str | None = None):/' "$TN_DIR/_inductor/codegen/scheduling.py"
```

torchtitan 侧 `sed`（根因一的三处，在下面两个训练命令里作用于 checkout 出的源码）。

> 退役条件：根因一的 1、2 与根因二全部随 torch_npu 发布配套 torch ≥ 2.13 的版本自然消失（届时 `sed` 无匹配，本身也是无害的空操作）；根因一的 3 是性能取舍，可长期保留；根因三随 CANN 的 `aclnnIndex` 支持 complex64（或 torch_npu 补转换实现）后可移除。

### 单卡训练

用 `torchrun --nproc_per_node=1` 在 1 张 NPU 上跑 `debugmodel` 真跑 2 步，验证配置解析、初始化、加载 tokenizer、build dataloader、forward + backward 整条链路能跑通。`llama3_debugmodel` 是 torchtitan 自带的最小 smoke 配置（dim=256 / 6 层 / 16 head / vocab 2048，~6 M 参数量），单卡 30 GB 完全够装。走真实 HCCL backend（`--comm.mode default`）让 c10d 把 `npu` 路由到 `hccl`，1-rank 下所有集合通信都是 self-barrier，不会真的有跨卡流量；不要用 `--comm.mode fake_backend` —— 它只注册 `fake` PG，v0.2.2 在 step 1 之后调 `set_pg_timeouts` → `torch.distributed.barrier(device_ids=[npu:0])` 时会因 `default_device_backend_map["npu"]="hccl"` 但当前 PG 是 `fake` 抛 `RuntimeError: No backend type associated with device type npu`。8B 模型单卡实测装不下（params + grads 在 bf16 下就要 32 GB > 30 GB 可用），需要双卡 FSDP shard=2 才跑得动，详见下一节「多卡训练」：

```shell #test id="torchtitan-train-debug" load="upstream_ref>>ref"
cd torchtitan && git checkout <ref>
sed -i '/separate_full_blocks=not is_in_batch_invariant_mode()/d' torchtitan/models/common/decoder.py
sed -i 's/_compiled_create_block_mask = torch.compile(create_block_mask)$/_compiled_create_block_mask = create_block_mask/' torchtitan/models/common/attention.py
sed -i 's/"max_autotune": True,/"max_autotune": False,/; s/"coordinate_descent_tuning": True,/"coordinate_descent_tuning": False,/' torchtitan/models/common/attention.py
sed -i 's/^    ComplexRoPE,$/    ComplexRoPE,\n    CosSinRoPE,/; s/ComplexRoPE\.Config(/CosSinRoPE.Config(/; s/scaling="llama",/scaling="none",/' torchtitan/models/llama3/__init__.py
ASCEND_RT_VISIBLE_DEVICES=0 \
torchrun --nproc_per_node=1 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    -m torchtitan.train \
    --module llama3 \
    --config llama3_debugmodel \
    --comm.mode default \
    --training.steps 2 \
    --training.local-batch-size 1 \
    --training.seq-len 256 \
    --metrics.log-freq 1 \
    --metrics.disable-color-printing \
    --dump-folder /tmp/torchtitan-quickstart
```

输出结果类似如下：

```shell #test-result id="torchtitan-train-debug" fuzzy='xxx' fuzzy='...'
[titan] xxx - root - INFO - torchtitan version: xxx
[titan] xxx - root - INFO - Starting job: Llama 3 debug training
...
[titan] xxx - root - INFO - Sleeping 2 seconds for other ranks to complete
[titan] xxx - root - INFO - Training completed
[titan] xxx - root - INFO - Process group destroyed
```

### 多卡训练

用 torchrun 起 2 个 rank 跑 8B 模型真分布式训练，`--training.steps 2` 真跑 2 步。`data_parallel_shard_degree = -1` 在双卡下解析成 2，FSDP 把 params / grads / Adam state 都按 shard 分摊，再加 `--training.enable-cpu-offload` 让 FSDP 把 Adam state 卸到 CPU，每张卡 NPU 实测占用 ~16 GB（params 8 GB + grads 8 GB + 激活张量 <1 GB），单卡 30 GB 装得下。再叠 `--training.dtype bfloat16` 把 params / grads / Adam state 全量 bf16，省掉 fp32 Adam state 那 32 GB 副本：

```shell #test id="torchtitan-train-2card" load="upstream_ref>>ref" load="ms_tokenizer_path>>ms_tokenizer_path"
cd torchtitan && git checkout <ref>
sed -i '/separate_full_blocks=not is_in_batch_invariant_mode()/d' torchtitan/models/common/decoder.py
sed -i 's/_compiled_create_block_mask = torch.compile(create_block_mask)$/_compiled_create_block_mask = create_block_mask/' torchtitan/models/common/attention.py
sed -i 's/"max_autotune": True,/"max_autotune": False,/; s/"coordinate_descent_tuning": True,/"coordinate_descent_tuning": False,/' torchtitan/models/common/attention.py
sed -i 's/^    ComplexRoPE,$/    ComplexRoPE,\n    CosSinRoPE,/; s/ComplexRoPE\.Config(/CosSinRoPE.Config(/; s/scaling="llama",/scaling="none",/' torchtitan/models/llama3/__init__.py
ASCEND_RT_VISIBLE_DEVICES=0,1 \
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=2 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter 0 \
    --tee 3 \
    -m torchtitan.train \
    --module llama3 \
    --config llama3_8b \
    --hf-assets-path <ms_tokenizer_path> \
    --comm.mode default \
    --dataloader.dataset c4_test \
    --training.dtype bfloat16 \
    --training.enable-cpu-offload \
    --training.steps 2 \
    --training.local-batch-size 1 \
    --training.seq-len 256 \
    --metrics.log-freq 1 \
    --metrics.disable-color-printing \
    --dump-folder /tmp/torchtitan-quickstart-2card
```

输出结果类似如下：

```shell #test-result id="torchtitan-train-2card" fuzzy='xxx' fuzzy='...'
[default0]:[titan] xxx - root - INFO - torchtitan version: xxx
[default0]:[titan] xxx - root - INFO - Starting job: Llama 3 8B training
...
[default0]:[titan] xxx - root - INFO - Sleeping 2 seconds for other ranks to complete
[default0]:[titan] xxx - root - INFO - Training completed
[default0]:[titan] xxx - root - INFO - Process group destroyed
```
