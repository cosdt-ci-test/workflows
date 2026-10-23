# 快速开始（零补丁版）

一句话：在单卡昇腾 NPU 上把 **OpenCV 4.11.0** 源码编译成带 CANN 后端的 `cv2`（`WITH_CANN=ON`），**一行源码不打补丁**，然后跑通两件事：

1. **CPU 核心 API**：读图 / 写图 / 颜色空间 / 缩放 / 绘制；
2. **NPU DNN 推理**：`cv2.dnn` + `DNN_BACKEND_CANN` + `DNN_TARGET_NPU`，MobileNetV2 在 910B 上跑出与 CPU 参考一致的 top-1。

## 什么时候用这份文档

主文档 [Quick-start-Ascend.md](Quick-start-Ascend.md) 跟的是「最新 release」（当前 5.0.0），代价是 **一组源码补丁**（`cv::MatShape` 适配 + 窄 op 头 + NULL 流兜底）。本文是另一条路：**版本钉在 4.11.0，补丁 0 组**。

选本文的理由：

- 你要的是「OpenCV 在昇腾上能跑」，不想维护任何本地 patch；
- 你能接受 4.x；
- 你不需要 `cv2.cann` 那套 contrib 经典算子 API（本文不编 `opencv_contrib`）。

不选本文的理由：要用 OpenCV 5.x 的新 DNN 图引擎，或要用 `cv2.cann` 加速经典算子 —— 那些请走主文档。

## 主文档那组补丁，本文怎么绕

| 主文档补丁 | 根因 | 本文怎么绕开 |
| --- | --- | --- |
| `cv::MatShape` 适配 | OpenCV **5.0.0** 把 dnn 层的 shape 换成 `cv::MatShape`，而上游 CANN 后端 `modules/dnn/src/op_cann.hpp` 从 4.11 到 5.0.0 **一个字节都没改**（4.11.0 / 4.12.0 / 4.13.0 / 4.14.0 / 5.0.0 全是 4446 字节，构造函数仍是 `const std::vector<int>&`）→ 5.0 编译不过 | 钉 **4.11.0**：`gemm_layer.cpp` / `matmul_layer.cpp` 在 4.11→4.14 与 CANN 后端同源（仍是 `std::vector<int>`），原样编译通过 |
| 窄 op 头 | `all_ops.h` 一次拉进 ~1500 个 op；Debug(-O0) 下 dnn `.text` 冲到 516MB，超 aarch64 `R_AARCH64_CALL26` 的 ±128MB 分支范围 → 链接必炸 | 改 **Release(-O3)**：实测**不打**窄 op 头的 4.11.0 Release 构建，dnn `.text` 只有 **52.0 MiB**（54,529,328 B），`libopencv_dnn.so` 一次链接通过（实测环境：Atlas 900 A2 / 910B4、Ubuntu 22.04 arm64、gcc 11.4、CANN 9.1.0、OpenCV 4.11.0）。⚠️ 这条**只在 4.11.0 成立**：同样的 Release + `all_ops.h` 在 **5.0.0** 上实测仍然链接失败（`relocation truncated to fit: R_AARCH64_JUMP26`，受害者是 protobuf 的 `AddDescriptorsRunner`）—— Release 不是通用解法，见「已知边界」 |
| NULL 流兜底 | CANN 9.1.0 的 `aclopCompileAndExecute` 拒绝 NULL 流（EH0008） | 本文**不编 `opencv_contrib`**，这段代码不在编译范围 |
| 缓冲夹紧 | 防御性加固（主文档自己测出 `acl_size == host_size == 4000`，并非崩溃根因） | 不加；驱动与 CANN 版本配套时本就不会触发 |

一句话版本：**「钉 4.11.0 + Release 构建 + 不编 contrib + CANN 库指向 `<CANN>/lib64`」= 零补丁**。

## 另一个坑：CANN 9.2 的裁剪版 `libgraph.so`

这一条跟 OpenCV 版本无关，**上游所有版本（4.11.0 / 4.x / 5.0.0）都一样**（`net_cann.cpp` 在 4.11.0 / 4.x / 5.0.0 里那几行一字不差），但它会以「补丁还不够用」的样子出现，所以单开一节。

`modules/dnn/src/net_cann.cpp` 里有一句：

```cpp
std::map<ge::AscendString, ge::AscendString> options = { ... };
```

`std::map` 需要 `ge::AscendString` 的 `operator<`。CANN 9.2.0 的头文件 `graph/ascend_string.h` 声明了它，但**能不能链上，取决于你链的是哪份 `libgraph.so`**：

| CANN 9.2.0 里的 `libgraph.so` | 大小 | `ge::AscendString` 的 4 个**序关系**运算符（`<` `>` `<=` `>=`） |
| --- | --- | --- |
| `<CANN>/lib64/libgraph.so`（= `aarch64-linux/lib64`） | 2.3 MB | **有**（`_ZNK2ge12AscendStringltERKS0_` 等 4 个） |
| `<CANN>/aarch64-linux/devlib/linux/aarch64/libgraph.so` | 198 KB | **没有**（裁剪版只留 `Find` / `Hash` / `GetLength` / `GetString` / `==` / `!=`） |

指错目录的典型症状（**链接期**报错，不是编译期）：

```
undefined reference to `ge::AscendString::operator<(ge::AscendString const&) const'
```

判定方法（一眼看出是完整版还是裁剪版）：

```shell
nm -D --defined-only <CANN>/lib64/libgraph.so | grep -c '_ZNK2ge12AscendStringlt'
```

**1** = 完整版（有 `operator<`）；**0** = 裁剪版。

（顺带一个版本无关的数：完整版 `libgraph.so` 里 `ge::AscendString` 的 const 成员共 12 个，裁剪版 8 个 —— 差的正好是 4 个序关系运算符。）

**怎么绕**：把给 OpenCV 的 CANN 软链一律指向 `<CANN>/lib64`（运行时的 `LD_LIBRARY_PATH` 用的也是它，`set_env.sh` 里写死 `${version_dirpath}/lib64`），**不要**指向 `devlib/...`。见下一节。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或 Ascend 950 系列，已挂载 `/dev/davinci*`。

### 软件（必须已就绪）

- Python 3.12
- CANN toolkit（参见 [Ascend 官方快速安装](https://ascend.github.io/docs/sources/ascend/quick_install.html)），`npu-smi info` 能列出设备
- 可访问 GitHub（clone + 下载模型）

> `opencv-python`（pip wheel）**不带** CANN 后端，必须从源码构建，`cv2.dnn.DNN_BACKEND_CANN` 才会链到真后端。

### 本文示例使用的版本

| 项目 | 值 |
| --- | --- |
| 机器 | Atlas 900 A2 PODc（910B4） |
| OS | Ubuntu 22.04.5 arm64 |
| CANN toolkit | **9.2.0**（`/usr/local/Ascend/cann-9.2.0`）；CI 镜像里是 9.1.0，两者的库目录布局差异见「桥接 CANN 的库目录」。**本文的链接体积实测是在 CANN 9.1.0 上做的**（aarch64 的 ±128 MiB 分支窗口是架构约束，与 CANN 版本无关） |
| Python / CMake / GCC | 3.12.15 / 3.22.1 / 11.4.0 |
| OpenCV | **4.11.0**（源码，`WITH_CANN=ON`，`CMAKE_BUILD_TYPE=Release`）|
| opencv_contrib | **不编**（本文不覆盖 `cv2.cann`）|

### 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

> 如果 `npu-smi` 不存在，按官方快速安装指南补装驱动。

检查 Python / CMake：

```shell #test id="np-check-toolchain"
python --version
cmake --version | head -n 1
```

输出结果如下：
```shell #test-result id="np-check-toolchain" fuzzy='xxx'
Python 3.12.xxx
cmake version xxx
```

加载 CANN 环境变量（每个用到 CANN 变量的块都要自己 `source`）：

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

确认 toolkit 根目录可访问：

```shell #test id="np-check-cann-path"
ls -d /usr/local/Ascend/ascend-toolkit/latest
```

输出结果如下：
```shell #test-result id="np-check-cann-path"
/usr/local/Ascend/ascend-toolkit/latest
```

## 装编译依赖

```shell #test-setup
sed -i 's|http://ports.ubuntu.com/ubuntu-ports/|https://mirrors.aliyun.com/ubuntu-ports/|g' /etc/apt/sources.list
apt-get update -qq && apt-get install -y -qq git build-essential cmake pkg-config zlib1g-dev libjpeg-dev libpng-dev libtiff-dev
```

## clone OpenCV（版本钉死 4.11.0）

本文没有 `<ref>` 占位符 —— 版本是钉死的，钉 4.11.0 是「零补丁」的前提（原因见上面的补丁表）。

```shell #test-setup
git clone --depth 1 --branch 4.11.0 https://github.com/opencv/opencv.git
```

### 确认源码是干净的（零补丁的证据）

打完补丁的树一定会有 diff；没打补丁的树 `git status` 是空的。同时确认 CANN 后端的构造函数签名仍是 4.x 原样（`std::vector<int>`，不是 5.0 的 `cv::MatShape`）。

```shell #test id="np-opencv-source-pristine"
cd opencv
git status --porcelain | wc -l
grep -n "const std::vector<int>& shape, const std::string& name" modules/dnn/src/op_cann.hpp
```

输出结果如下：
```shell #test-result id="np-opencv-source-pristine" fuzzy='xxx'
0
xxx:        CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name);
```

## 桥接 CANN 的库目录

OpenCV 的 `OpenCVFindCANN.cmake` 在 `${CANN_INSTALL_DIR}/{acllib,lib64,compiler/lib64}/` 下找 ACL 库：

| 找什么 | 期望位置 |
| --- | --- |
| `libascendcl.so` | `acllib/lib64/` |
| `libacl_op_compiler.so`、`libacl_dvpp_mpi.so` | `lib64/` |
| `libgraph.so`、`libge_compiler.so`、`libgraph_base.so` | `compiler/lib64/` |

这三个位置在 CANN 里都不叫这三个名字，所以要补软链。**统一指向「完整版 CANN 库目录」**：

- CANN 9.1.x：`<CANN>/lib64` 不存在，完整库目录是 `<CANN>/aarch64-linux/lib64`；
- CANN 9.2.x：`<CANN>/lib64` 已经是指向 `aarch64-linux/lib64` 的软链（同一批库）。

注意 **不要**指向 `<CANN>/aarch64-linux/devlib/linux/aarch64/` —— 那是裁剪过的设备侧运行时（`libgraph.so` 只有 198 KB，少了 `ge::AscendString` 的比较运算符，见上一节）。

先确认本机的 `<CANN>/lib64` 是完整版：

```shell #test id="np-cann-lib-layout"
CANN_DIR=/usr/local/Ascend/ascend-toolkit/latest
ls -l "$CANN_DIR/lib64/libascendcl.so" "$CANN_DIR/lib64/libgraph.so"
nm -D --defined-only "$CANN_DIR/lib64/libgraph.so" | grep -c '_ZNK2ge12AscendStringlt'
```

输出结果如下：
```shell #test-result id="np-cann-lib-layout" fuzzy='xxx'
xxx
xxx
1
```

补 symlink（`acllib` / `compiler` 要先是目录，所以先 `mkdir`）：

```shell #test-setup
CANN_DIR=/usr/local/Ascend/ascend-toolkit/latest
# CANN 9.1 没有 <CANN>/lib64，补一条指向 aarch64-linux/lib64 的软链；
# CANN 9.2 已经有（本身就是 aarch64-linux/lib64 的软链），不动它。
[ -e "$CANN_DIR/lib64" ] || ln -sfn "$CANN_DIR/aarch64-linux/lib64" "$CANN_DIR/lib64"
mkdir -p "$CANN_DIR/acllib" "$CANN_DIR/compiler"
ln -sfn "$CANN_DIR/lib64" "$CANN_DIR/acllib/lib64"
ln -sfn "$CANN_DIR/lib64" "$CANN_DIR/compiler/lib64"
ls -l "$CANN_DIR/acllib/lib64/libascendcl.so" "$CANN_DIR/lib64/libacl_op_compiler.so" "$CANN_DIR/compiler/lib64/libgraph.so"
```

> `ln -sfn` 到已存在的软链是「改指向」而不是「往目录里塞链接」，所以重复执行是安全的（`lib64` 那条还额外用 `[ -e ... ]` 挡掉了「指向自己」的 `are the same file` 报错）。
> `libgraph_base.so` 是 CANN ≥ 6.3.RC1 之后 `OpenCVFindCANN.cmake` 才会去找的库（找不到只 WARN 不报错），但它是 `ge::AscendString` 那组符号的另一个来源，务必确认它在 `compiler/lib64/` 里。
> **不想往 CANN 安装目录里写东西**（只读挂载 / 无 root）时，等价做法是直接把路径喂给 cmake：`-Dfound_lib_ascendcl=... -Dfound_lib_acl_op_compiler=... -Dfound_libacldvppmpi=... -Dfound_lib_graph=... -Dfound_lib_ge_compiler=... -Dfound_libgraph_base=...`（`find_library` 命中已存在的 cache 变量就不再搜目录）。本文的验证构建走的就是这条路线。

确认 cmake 找到了 CANN 库（configure 输出里的 `CANN: lib... is found at ...`）：

```shell #test id="np-cann-cmake-libs"
grep "CANN: libascendcl.so is found" /tmp/opencv-configure.log
grep "CANN: libgraph.so is found" /tmp/opencv-configure.log
```

输出结果如下：
```shell #test-result id="np-cann-cmake-libs" fuzzy='xxx'
-- CANN: libascendcl.so is found at xxx
-- CANN: libgraph.so is found at xxx
```

## 配置并编译

CANN 后端开关 `WITH_CANN=ON`（不是 `BUILD_CANN`）；`-DOPENCV_DOWNLOAD_MIRROR_ID=gitcode` 让 cmake configure 阶段走 gitcode 镜像拉 ADE / IPPICV / TBB。

**三个和主文档不同的地方**：

- `CMAKE_BUILD_TYPE=Release` 而不是 `Debug` —— 这是绕开窄 op 头补丁的关键（Debug 的 `-O0` 会把 `all_ops.h` 里每个 op 的内联函数都落盘，`.text` 撑爆 aarch64 的分支范围）。**注意这条只对 4.11.0 成立**：5.0.0 实测 Release + `all_ops.h` 一样链接失败（`R_AARCH64_JUMP26` 越界）；
- `BUILD_LIST` 里**没有** `cannops` —— 不编 `opencv_contrib`，所以没有 contrib 侧的两处 NULL 流补丁；
- `-DBUILD_opencv_apps=OFF` —— OpenCV 自带的 `apps/model-diagnostics` 是个**可执行文件**，它要直接链 CANN 的 `libge_compiler.so` / `libgraph.so` / `libgraph_base.so`，而这三个库自身还有一堆没被链进来的依赖（`ge::*`、`ascend_private::protobuf::*`、`absl::*`…）。共享库链接允许留未定义符号（所以 `libopencv_dnn.so` 没事），可执行文件链接不允许 → **链接期报几百条 `undefined reference` 并让整个 `all` 目标失败**（CANN 9.2.0 实测）。关掉 apps 就绕开了。

安装前缀用 `/usr/local/opencv-cann-411`（不是主文档的 `/usr/local/opencv-cann`）：两套 OpenCV（5.x 带补丁 / 4.11 零补丁）可能落在同一台自托管 runner 上，装到同一个前缀会互相覆盖。只想跑本文的话，`/usr/local/opencv-cann` 也行，把下面两处（前缀 / `PYTHONPATH`）一起换掉即可。

```shell #test id="np-opencv-cmake-configure"
cd opencv
mkdir -p build && cd build
set -o pipefail
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local/opencv-cann-411 \
      -DWITH_CANN=ON \
      -DCANN_INSTALL_DIR=/usr/local/Ascend/ascend-toolkit/latest \
      -DBUILD_opencv_world=OFF \
      -DBUILD_opencv_apps=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_LIST=core,imgproc,imgcodecs,dnn,python3 \
      -DSOC_VERSION=ascend910b1 \
      -DBUILD_opencv_python3=ON \
      -DBUILD_opencv_python_bindings_generator=ON \
      -DPYTHON_INCLUDE_DIR=/usr/local/python3.12.13/include/python3.12 \
      -DPYTHON_LIBRARY=/usr/local/python3.12.13/lib/libpython3.12.so \
      -DOPENCV_ENABLE_NONFREE=OFF \
      -DOPENCV_DOWNLOAD_MIRROR_ID=gitcode \
      .. | tee /tmp/opencv-configure.log
```

输出结果如下：
```shell #test-result id="np-opencv-cmake-configure" fuzzy='xxx' fuzzy='...'
-- The CXX compiler identification is GNU xxx
...
--   CANN:xxx YES
...
-- Configuring done
-- Generating done
-- Build files have been written to: xxx/build
```

编译 + 安装一体（`--target install`）。**并行度别开高**：`all_ops.h` 那几个 TU 每个要吃掉 3GB 上下内存，容器内存配额（cgroup limit）只有 32GB 时，`-j` 超过 6 会被 OOM killer 直接干掉 `cc1plus`（表现为编译日志里出现 `c++: fatal error: Killed signal terminated program cc1plus`）。所以并行度做成可覆盖的环境变量：默认 `--parallel 2`（保守），CI 机器内存更富裕时可以调高。

```shell #test-setup
cd opencv/build
cmake --build . --target install --parallel "${OPENCV_BUILD_JOBS:-2}"
```

> 只想「一次跑完」的话也可以把 `OPENCV_BUILD_JOBS` 设成 `$(nproc)`；但先看一眼容器的内存配额（`cat /sys/fs/cgroup/memory/memory.limit_in_bytes`，v2 是 `/sys/fs/cgroup/memory.max`），`nproc × 3GB` 超过配额就必然 OOM。

## 校验 build 产物

> `opencv_version` 那个小工具属于 `apps/`（上面 `-DBUILD_opencv_apps=OFF` 关掉了），所以这里直接看装出来的库。

```shell #test id="np-opencv-verify-build"
ls -l /usr/local/opencv-cann-411/lib/libopencv_core.so.4.11.0 /usr/local/opencv-cann-411/lib/libopencv_dnn.so.4.11.0
```

输出结果如下：
```shell #test-result id="np-opencv-verify-build" fuzzy='xxx'
xxx
xxx
```

顺便看一眼 dnn 库的实际体积 —— 这是「Release 不用窄 op 头补丁」那一条的证据（Debug -O0 下同样的库 `.text` 是 516MB）：

> 实测（4.11.0 + CANN 9.1.0 + gcc 11.4 + Release -O3）：`libopencv_dnn.so.4.11.0` 的 `.text` = **54,529,328 B（≈52.0 MiB）**，`Total` = 75,204,691 B，库文件 95,668,672 B。这个数比 aarch64 的 ±128 MiB 分支窗口小一个量级，所以**不需要**窄 op 头补丁。同一份源码在 Debug(-O0) 下 `.text` 会涨到 516MB，直接撑爆窗口。

```shell #test id="np-opencv-dnn-text-size"
size -A /usr/local/opencv-cann-411/lib/libopencv_dnn.so.4.11.0 | grep -E '^\.text|^Total'
```

输出结果如下：
```shell #test-result id="np-opencv-dnn-text-size" fuzzy='xxx' fuzzy='...'
.text xxx
...
Total xxx
```

## OpenCV Python quickstart

把 Python 指向源码版 `cv2`（带 CANN 后端）：

```shell #test-setup
export PYTHONPATH=/usr/local/opencv-cann-411/lib/python3.12/site-packages:${PYTHONPATH:-}
```

> 注意是 **prepend**（保留镜像 `set_env.sh` 注入的 CANN `PYTHONPATH` 条目）。GE 图编译依赖其中的 `tbe` Python 模块；整体覆盖 `PYTHONPATH` 会让 DNN CANN 后端在图编译阶段报 `There is no valid so about OpsKernelInfoStore or GraphOptimizer`（ret=0xFFFFFFFF）。

下面用 clone 下来的 OpenCV 仓库自带测试图 `opencv/samples/data/baboon.jpg`（512×512 狒狒）作样例输入：

![baboon 输入图](images/baboon-input.jpg)

### 读图 / 写图

```shell #test id="np-quickstart-imread-imwrite"
python << 'PY'
import cv2
img = cv2.imread('opencv/samples/data/baboon.jpg')
ok = cv2.imwrite('/tmp/opencv_quickstart.png', img)
print('imwrite ok:', ok)
print('shape:', img.shape, 'dtype:', img.dtype)
PY
ls /tmp/opencv_quickstart.png
```

输出结果如下：
```shell #test-result id="np-quickstart-imread-imwrite"
imwrite ok: True
shape: (512, 512, 3) dtype: uint8
/tmp/opencv_quickstart.png
```

### 颜色空间转换

```shell #test id="np-quickstart-cvtcolor"
python << 'PY'
import cv2
bgr = cv2.imread('opencv/samples/data/baboon.jpg')
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
print('gray shape:', gray.shape, 'mean:', float(gray.mean()))
cv2.imwrite('/tmp/quickstart-gray.png', gray)
PY
```

输出结果如下：
```shell #test-result id="np-quickstart-cvtcolor" fuzzy='xxx'
gray shape: (512, 512) mean: xxx
```

baboon 整体偏中亮（mean ≈ 129.7/255），灰度图长这样：

![灰度转换结果](images/quickstart-gray.png)

### 缩放

```shell #test id="np-quickstart-resize"
python << 'PY'
import cv2
img = cv2.imread('opencv/samples/data/baboon.jpg')
resized = cv2.resize(img, (50, 200), interpolation=cv2.INTER_AREA)
print('resized shape:', resized.shape)
cv2.imwrite('/tmp/quickstart-resize.png', resized)
PY
```

输出结果如下：
```shell #test-result id="np-quickstart-resize"
resized shape: (200, 50, 3)
```

OpenCV shape 是 `(rows, cols, channels)`，`resize(img, (W, H), ...)` 得 `(H, W, C)`；缩到 50×200（宽×高）后：

![缩放结果](images/quickstart-resize.png)

### 绘制

```shell #test id="np-quickstart-draw"
python << 'PY'
import cv2
img = cv2.imread('opencv/samples/data/baboon.jpg')
cv2.rectangle(img, (10, 10), (500, 500), (0, 255, 0), thickness=2)
cv2.putText(img, 'Hello OpenCV', (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
print('rect+text drawn, image mean:', float(img.mean()))
cv2.imwrite('/tmp/quickstart-draw.png', img)
PY
ls /tmp/quickstart-draw.png
```

输出结果如下：
```shell #test-result id="np-quickstart-draw" fuzzy='xxx'
rect+text drawn, image mean: xxx
/tmp/quickstart-draw.png
```

绿框 + 文字绘制结果：

![绘制结果](images/quickstart-draw.png)

> 视频读写（`cv2.VideoWriter`）需要 `videoio` 模块，本文这版 `BUILD_LIST` 没带它（见「已知边界」）；要视频就把 `videoio` 加进 `BUILD_LIST` 再重编一次。

## NPU 上跑一次 DNN 推理

```shell #test id="np-opencv-py-version"
python -c "
import cv2
print('cv2 version:', cv2.__version__)
print('CANN backend available:', cv2.dnn.DNN_BACKEND_CANN)
print('NPU target available:', cv2.dnn.DNN_TARGET_NPU)
"
```

输出结果如下：
```shell #test-result id="np-opencv-py-version" fuzzy='xxx'
cv2 version: xxx
CANN backend available: xxx
NPU target available: xxx
```

跑 MobileNetV2（onnx/models 的 `mobilenetv2-12`，PyTorch 导出，13,964,571 bytes）：

> **模型来源**：CI 与本仓库 checkout 优先用 `fixtures/mobilenetv2-12.onnx`（同字节入仓，避免 GitHub raw 在国内网络的 `RemoteDisconnected` 抖动）；自己机器上跟着跑时没有 fixtures 目录，自动回落到从 onnx/models 下载（重试 5 次）。

> **为什么不用 `OPENCV_FORCE_DNN_ENGINE=1`**：那是主文档（OpenCV 5.0）的坑 —— 5.0 默认 `ENGINE_AUTO` 会选新图引擎（`onnx_importer2`），新引擎尚未支持非 CPU 后端。**4.11 的默认引擎就是 classic 引擎**，`setPreferableBackend` / `setPreferableTarget` 直接生效，所以本文不需要这个环境变量。

```shell #test id="np-opencv-cann-infer"
python << 'PY'
import os
import shutil
import sys
import time
import urllib.request
import numpy as np
import cv2

MODEL_URL = ('https://github.com/onnx/models/raw/main/'
             'validated/vision/classification/mobilenet/'
             'model/mobilenetv2-12.onnx')
MODEL_PATH = '/tmp/mobilenetv2-12.onnx'

if not os.path.exists(MODEL_PATH):
    if os.path.exists('fixtures/mobilenetv2-12.onnx'):
        # 本仓库 checkout（CI）：直接用入仓副本，零网络依赖
        shutil.copy('fixtures/mobilenetv2-12.onnx', MODEL_PATH)
    else:
        # 读者自跑：从 onnx/models 下载；github raw 偶发 RemoteDisconnected
        # （国内网络抖动），重试 5 次，半截文件要丢掉避免下次误判已下载
        for attempt in range(5):
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                break
            except OSError:
                if os.path.exists(MODEL_PATH):
                    os.unlink(MODEL_PATH)
                if attempt == 4:
                    raise
                time.sleep(10)

net = cv2.dnn.readNetFromONNX(MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CANN)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_NPU)

image = cv2.imread('opencv/samples/data/baboon.jpg')
x = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0  # BGR, [0,1]
x = x[..., ::-1]                                              # BGR -> RGB
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)      # ImageNet mean/std
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
x = (x - mean) / std
blob = np.ascontiguousarray(x.transpose(2, 0, 1))[np.newaxis]  # 1xCxHxW float32

net.setInput(blob)
out = net.forward()

# mobilenetv2-12 输出是 logits（不含 softmax），自己归一化
logits = out.reshape(-1)
e = np.exp(logits - logits.max())
prob = e / e.sum()

img = image.copy()
cv2.rectangle(img, (10, 10), (500, 500), (0, 255, 0), thickness=2)
label = 'top-1: #372 patas monkey  %.3f' % float(np.max(prob))
cv2.putText(img, label, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imwrite('/tmp/dnn-result.png', img)

# 结果不经 python 的缓冲 stdout 输出，而是用原始 syscall 写文件：
# GE 图编译会 fork 子进程，子进程 exit(0) 会把继承的 stdout 缓冲冲到
# C++ 日志行的中间（实测 "model bytes" 粘在半截 INFO 行上）。
result = (
    'model bytes: %d\n'
    'output shape: %s\n'
    'output dtype: %s\n'
    'top class index: %d\n'
    'top class score: %s\n'
) % (os.path.getsize(MODEL_PATH), out.shape, out.dtype,
     int(np.argmax(prob)), float(np.max(prob)))
fd = os.open('/tmp/dnn_result.txt', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.write(fd, result.encode())
os.close(fd)
sys.stdout.flush()
os._exit(0)
PY
echo
cat /tmp/dnn_result.txt
```

输出结果如下：
```shell #test-result id="np-opencv-cann-infer" fuzzy='xxx' fuzzy='...'
...
model bytes: xxx
output shape: (1, 1000)
output dtype: float32
top class index: 372
top class score: 0.xxx
...
```

预期：`output shape (1, 1000)`、`float32`；baboon.jpg 的 top-1 是 **ImageNet-1k 第 372 类（patas / hussar monkey，赤猴）**，score ≈ 0.91（4.11.0 + CANN 9.1.0 实测 **0.9131**；与 CPU 参考一致，baboon 本尊 373 排第二 ≈ 0.03）。首尾 `...` 吞掉 OpenCV INFO 日志噪音（导入/逐层转 CANN 算子等，注意这些日志走 stdout）。标注结果如下：

![NPU 推理标注结果](images/dnn-result.png)

> 首次 forward 含 GE 图编译（`converting ge::Graph to OM buffer`），约 0.5~1 分钟出 `Compile success, model size = ...`；进程内二次 forward 约 1ms。

## 已知边界

- **没有 `cv2.cann`**：本文不编 `opencv_contrib`，所以 `cv2.cann.cvtColor` 这类 contrib 经典算子加速 API 不可用（主文档有）。要它就得编 contrib，就得接受 contrib 侧的两处 NULL 流补丁。
- **没有 OpenCV 5.x**：4.11.0 没有 5.0 的新 DNN 图引擎（`onnx_importer2`），也没有 5.0 的其它新特性。
- **没有 `opencv_test_cannops` 单测**：那个 gtest 二进制来自 `opencv_contrib` 的 cannops 模块，本文不编 contrib，所以没有那 53 个用例的证据；NPU 正确性由上面的 MobileNetV2 top-1 对齐来兜。
- **编译并行度受内存配额约束**：`all_ops.h` 那几个 TU 每个吃 3GB 上下，cgroup 32GB 的容器里 `--parallel` 别超过 6，否则 OOM killer 会杀掉 `cc1plus`。
- **「Release 就不用窄 op 头补丁」不是通用解法**：本条只在 **4.11.0** 上实测成立（Release 构建 dnn `.text` 52 MiB → 链接通过）。同样条件下 **5.0.0 实测仍然失败**：Release + `all_ops.h` 链接报 `relocation truncated to fit: R_AARCH64_JUMP26`，受害者是 protobuf 的 `AddDescriptorsRunner::AddDescriptorsRunner`（**报错对象随链接顺序随机变**，5.0.0 的 Debug 构建里同一个坑出现过 `Arena::AllocateAlignedWithCleanup` 版本 —— 所以「链接炸了但报错看着跟 CANN 无关」是常态，别顺着报错对象查根因）。也就是说：**分支范围是 aarch64 的硬约束，缩 `.text` 只是缓解，不是保证**；OpenCV 5.x 要过这一关只有窄 op 头一条路（主文档的 3 组补丁之一）。
- **没有 `videoio`**：本文的 `BUILD_LIST` 只有 `core,imgproc,imgcodecs,dnn,python3`，所以 `cv2.VideoWriter` / `cv2.VideoCapture` 不可用（`cv2` 里没有这两个符号）。要视频读写就把 `videoio` 加进 `BUILD_LIST`（会多编一次，模块列表变了会触发全量重编）。
- **没有 `opencv_version` 等命令行工具**：它们属于 `apps/`，而 `apps` 被 `-DBUILD_opencv_apps=OFF` 关掉了（原因见「配置并编译」一节）。
- **CANN 的「裁剪版运行时」是隐形坑**：`<CANN>/aarch64-linux/devlib/linux/aarch64/` 里有一套同名但被裁剪过的 `.so`（`libgraph.so` 只有 198 KB，少 4 个 `ge::AscendString` 序关系运算符）。symlink 指错目录不会在编译期报错，只会在链接期炸成 `undefined reference to ge::AscendString::operator<`；本文「桥接 CANN 的库目录」一节里的 `nm` 检查就是为了挡住它。
