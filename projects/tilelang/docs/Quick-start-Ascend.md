# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上从源码构建 tilelang-ascend，并运行官方 GEMM 算例完成一次端到端验证（JIT 编译 → NPU 执行 → 数值校验）。

> tilelang 主仓不包含 NPU 后端，昇腾适配在生态仓 **tilelang-ascend**（Ascend C & PTO 路线）。本文档克隆该仓库源码构建 wheel 并安装，最后运行 `examples/gemm` 示例验证算子可用。

## 前置条件

### 硬件

Atlas 900 A2 单卡（Ascend NPU），并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- Python ≥ 3.10 的环境
- 可用的 CANN ≥ 8.3.RC1（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import`、`torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）
- `git` 与 C/C++ 编译工具链（构建含 C++ 扩展）

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
| tilelang-ascend | \<ref>（源码构建） |

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

克隆 tilelang-ascend 仓库并 checkout 到工作流注入的最新 release tag（`--recursive` 拉取构建所需的 TVM 子模块）：

```shell #test id="clone-repo" load="upstream_ref>>ref"
git clone --recursive https://github.com/tile-ai/tilelang-ascend.git
cd tilelang-ascend
git checkout <ref>
echo "HEAD $(git log -1 --format=%h)"
```

输出结果如下：

```shell #test-result id="clone-repo" fuzzy='xxx' fuzzy='...'
...HEAD xxx
```

\<ref> 为流水线注入的上游最新 release tag。

## 构建安装

按上游推荐方式从源码构建 wheel 并安装。`ASCEND_HOME_PATH` 需指向 CANN toolkit 安装目录（构建脚本硬性检查；若你的安装路径不同请对应替换）：

```shell #test-setup
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
cd tilelang-ascend
python -m pip install -r requirements.txt
# 流水线把 /data/ci-cache/tilelang-wheelhouse 挂载到容器内的
# /tilelang-wheelhouse（本地手动执行无此目录，恒走完整构建）。
CACHE_DIR=/tilelang-wheelhouse
KEY=$(git rev-parse HEAD)
CACHED=$(ls "$CACHE_DIR/$KEY"/tilelang-*.whl 2>/dev/null | head -n 1 || true)
if [ -n "$CACHED" ] && python -m pip install --force-reinstall --no-deps "$CACHED" && python -c "import tilelang"; then
  echo "wheel cache hit ($KEY), skipping source build"
else
python -m pip install -r requirements-build.txt
python - <<'EOF'
from pathlib import Path
p = Path('setup.py')
s = p.read_text()
subs = [
    ('int(multiprocessing.cpu_count() * 0.5)', 'min(12, int(multiprocessing.cpu_count() * 0.5))'),
    ('int(multiprocessing.cpu_count() * 0.75)', 'min(12, int(multiprocessing.cpu_count() * 0.75))'),
]
for old, new in subs:
    assert old in s, f'setup.py patch anchor missing: {old!r} (upstream changed parallelism derivation)'
    s = s.replace(old, new)
p.write_text(s)
print('setup.py build parallelism clamped to <=12')
EOF
./build_wheel_ascend.sh
ls -la dist/
for W in dist/tilelang-*.whl; do
  python -m zipfile -l "$W" | grep -q 'tilelang/__init__.py' || { echo "FATAL: $W missing tilelang package"; exit 1; }
done
if [ -d "$CACHE_DIR" ]; then
  mkdir -p "$CACHE_DIR/$KEY"
  cp dist/tilelang-*.whl "$CACHE_DIR/$KEY"/
  ls -1dt "$CACHE_DIR"/*/ | tail -n +3 | xargs -r rm -rf
fi
python -m pip install --force-reinstall --no-deps dist/tilelang-*.whl
fi
python -c "import tilelang; print('tilelang installed:', tilelang.__version__)"
```

> - 构建脚本会应用 TVM 子模块补丁、执行 C++ 编译并产出 `dist/tilelang-*.whl`，首次构建需要较长时间，请耐心等待。
> - `setup.py` 默认按宿主机核数的一半（`cpu_count() * 0.5`，192 核机器即 `-j96`）推导编译并行度，容器内存限额下必 OOM（OOM killer 杀 cc1plus/ld）。上面的补丁把两条并行度推导钳到最高 12 个任务，小核数机器自动取 `cpu_count()` 原值；补丁锚点若被上游改写会直接报错终止，不会静默失效。
> - 构建脚本内部使用裸 `pip` 安装依赖；构建分支已提前用 `python -m pip` 装好同样的依赖，脚本内对应步骤会显示 already satisfied，保证依赖装进当前 `python` 环境。
> - `requirements-build.txt` / `requirements.txt` 的详细清单以 tilelang-ascend 仓库为准；`requirements-build.txt` 只含构建期工具（cmake、patchelf 等），故只出现在源码构建分支。
> - 流水线在 `/tilelang-wheelhouse` 挂载了宿主机 wheel 缓存，按上游 commit 为 key：同一 commit 重跑（重试、文档修订、手动 dispatch）直接复用已构建的 wheel，跳过约 15 分钟的源码编译；新 release 是新 commit，自动全量构建。本地手动执行没有这个目录，永远走完整构建。缓存命中但安装或导入失败时（wheel 损坏等）自动回落到源码重建并覆盖缓存。
> - wheel 安装用 `--force-reinstall --no-deps`（依赖已由 requirements 步骤装好），避免 pip 以 "Requirement already satisfied" 跳过安装导致 site-packages 里实际没有 tilelang 包。
> - 构建后的 wheel 会做完整性自检（列出 dist/ 内容 + 校验压缩包内有 `tilelang/__init__.py`），块尾的 `import tilelang` 是最终安装断言——任一环节失败都会让整块以非零码退出，日志不会静默丢失。

验证安装：

```shell #test id="install-tilelang"
python -c "import tilelang; print('tilelang', tilelang.__version__)"
```

输出结果如下：

```shell #test-result id="install-tilelang" fuzzy='xxx' fuzzy='...'
...tilelang xxx
```

## 运行 GEMM 算例

运行上游官方 GEMM 示例（默认 1024×1024×1024 fp16，可用 `--m/--n/--k` 调整）。脚本先打印 `init successful!`，JIT 编译后在 NPU 上执行 kernel，并与 PyTorch 参考结果做数值比对，通过后打印 `Kernel Output Match!`：

```shell #test id="run-gemm"
cd tilelang-ascend
python examples/gemm/example_gemm.py > /tmp/gemm.log 2>&1
grep -q "init successful!" /tmp/gemm.log && grep -q "Kernel Output Match!" /tmp/gemm.log && echo "gemm ok" || { tail -n 50 /tmp/gemm.log; exit 1; }
```

输出结果如下：

```shell #test-result id="run-gemm"
...gemm ok
```

> - 输出重定向到 `/tmp/gemm.log` 以便断言两个成功标志；失败时自动打印日志尾部便于定位。
> - 示例默认使用 Expert 模式原语（`T.Scope("C")` / `T.alloc_L1` 等，对应默认环境变量 `TILELANG_ASCEND_MODE=Expert`）；跨硬件可移植的 Developer 模式（`T.alloc_shared` / `T.alloc_fragment` / `T.Pipelined`）见上游 Programming Guide。
> - 更多算子示例（FlashAttention、Softmax、归一化、DeepSeek V4 算子等）见上游 `examples/` 目录。
