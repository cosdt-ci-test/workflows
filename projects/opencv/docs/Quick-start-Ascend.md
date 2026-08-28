# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上完成 OpenCV 安装与 CANN (Huawei Ascend) 后端 DNN 推理：先用 pip 装官方 wheel 走完官方 [quickstart 流程](https://opencv-opencv.mintlify.app/quickstart) 的 read/write/draw/color/resize 五类基础 API；再从源码构建带 `WITH_CANN=ON` 的 OpenCV（mainline 5.0.0，opencv_contrib 提供 `cannops` / `cannarithm` 模块；参考 [OpenCV Huawei CANN Backend wiki](https://github.com/opencv/opencv/wiki/Huawei-CANN-Backend) 的模块对照，镜像统一到 CANN 9.1.0 体系），用 [SqueezeNet ONNX](https://opencv-opencv.mintlify.app/installation#python) 跑一次 NPU forward。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载（`/dev/davinci*` 等）。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的驱动 / 固件，且 `npu-smi info` 能正常列出 NPU 设备
- 可访问 GitHub（克隆 opencv + opencv_contrib 源码），以及从 ONNX Model Zoo 下载 SqueezeNet 模型

OpenCV 通过 DNN 模块内置 Huawei CANN 后端：构建阶段把 OpenCV 链接到 `ascend-toolkit` 提供的 ACL/AOE/GE 库，运行时通过 `cv2.dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_CANN)` 把推理图下沉到 NPU。pip 发布的 `opencv-python` wheel 默认**未启用** `WITH_CANN`，必须走源码构建路径才能在 NPU 上跑 DNN。

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
| 驱动 / 固件 | 与 CANN 9.1.0 匹配（镜像内置） |
| CMake | >= 3.18 |
| GCC | 11 (Ubuntu 22.04 自带) |
| opencv | 5.0.0（源码，启用 CANN） |
| opencv_contrib | 与 opencv 同 release tag |

> CANN 9.1.0 体系对应的 ascend-toolkit 默认安装路径为 `/usr/local/Ascend/ascend-toolkit/latest`，其下 `include/acllite`、`lib64/libascendcl.so` 等 ACL 头 / 库为构建 OpenCV CANN 后端的必要依赖。

### 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

> 如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。

检查 Python / CMake 版本：

```shell #test id="check-toolchain"
python --version
cmake --version | head -n 1
```

```shell #test-result id="check-toolchain" fuzzy='xxx'
Python 3.12.xxx
cmake version xxx
```

加载 CANN 环境变量（`set_env.sh` 把 `ASCEND_HOME` / `LD_LIBRARY_PATH` / `PATH` 等灌进当前 shell；每个 `#test` 块都是独立 `subprocess.run`，所以这步只在 load-cann 这一块里 source 一次，下面 cmake configure / make / python 块各自再 source 一次以拿到 CANN 变量——`prepare_environment` 不再做 source）：

```shell #test id="load-cann"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

```shell #test-result id="load-cann"
...
```

预期：source 退出码为 0（`set_env.sh` 自身无输出），`export PATH` 也退出 0。`PATH=/usr/local/sbin:$PATH` 仅在本块生效——`prepare_environment` 不再追加这个目录，每一块自己保证。

检查 CANN 安装路径：

```shell #test id="check-cann-path"
ls -d /usr/local/Ascend/ascend-toolkit/latest
```

```shell #test-result id="check-cann-path" fuzzy='xxx'
/usr/local/Ascend/ascend-toolkit/xxx
```

> 如果路径不存在或 `latest` 软链指向其他版本，回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 校验 CANN 安装。

## 安装 OpenCV（源码，启用 CANN 后端）

pip 发布的 `opencv-python` / `opencv-python-headless` wheel **不带** CANN 后端（enum 常量在但 backend 实现未链接），必须从源码构建：克隆 `opencv` + `opencv_contrib`，用 CMake 把 `WITH_CANN=ON` 打开后编译。安装路径用 `/usr/local/opencv-cann`，与默认 `python -c "import cv2"` 解析路径完全隔离。

#### 安装基础编译依赖

```shell #test-setup
apt-get update -qq && apt-get install -y -qq git build-essential cmake pkg-config libjpeg-dev libpng-dev libtiff-dev
```

#### 从源码克隆 opencv + opencv_contrib

<!-- 工作流注入的 UPSTREAM_REF（最新 release tag）通过这个隐藏的 #test-setup 捕获并注入到下方 clone 命令中；markdown 渲染器会丢掉注释里全部内容，读者看不到这段代码，但 runner 仍然执行它并 store="upstream_ref" -->
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到工作流注入的最新 release tag；opencv 与 opencv_contrib 版本必须一致：

```shell #test-setup load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/opencv/opencv.git
git clone --depth 1 --branch <ref> https://github.com/opencv/opencv_contrib.git
```

> `<ref>` 为工作流注入的最新 release tag。

#### CMake 配置：开启 WITH_CANN

mainline 5.0.0 里 CANN 后端的开关变量是 `WITH_CANN`（不是 `BUILD_CANN`，后者不存在），通过环境变量 `ASCEND_TOOLKIT_HOME` 指向 `ascend-toolkit` 安装根目录（也可用 `-DCANN_INSTALL_DIR=...` 直接覆盖）——这一步与 [OpenCV Huawei CANN Backend wiki](https://github.com/opencv/opencv/wiki/Huawei-CANN-Backend) 的 Step 3 一致：

```shell #test id="opencv-cmake-configure" load="upstream_ref>>ref"
cd opencv
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE \
      -DCMAKE_INSTALL_PREFIX=/usr/local/opencv-cann \
      -DWITH_CANN=ON \
      -DBUILD_opencv_world=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_TESTS=ON \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_opencv_python3=ON \
      -DBUILD_opencv_python_bindings_generator=ON \
      -DOPENCV_ENABLE_NONFREE=OFF \
      -DBUILD_opencv_xfeatures2d=OFF \
      -DBUILD_opencv_face=OFF \
      -DOPENCV_DOWNLOAD_MIRROR_ID=gitcode \
      -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      ..
```

```shell #test-result id="opencv-cmake-configure" fuzzy='xxx'
...
--   CANN:xxx YES
...
```

预期：`CANN: ... YES` 这一行出现在 cmake summary 段——这正是 wiki 上 Step 4（Verification）所要求的"先看 CMake 报告"步骤。`OPENCV_DOWNLOAD_MIRROR_ID=gitcode` 让主仓 cmake configure 阶段拉 ADE / IPPICV / TBB / xfeatures2d 数据 / 字体 / wechat_qrcode 模型时走 gitcode.net 镜像（中国大陆可达）。

#### 编译

```shell #test-setup
cd opencv/build
make -j$(nproc)
```

> 编译时间受 CPU 核数与是否启用 world 影响：单核 `make` 大约 1.5 小时；8 核并行约 20 分钟。CI runner 通常 32+ 核，10 分钟内可以结束。

#### 安装到 /usr/local/opencv-cann

```shell #test-setup
cd opencv/build
make install
```

#### 校验二进制 + CANN 后端可用性

跑 OpenCV 自带的 versioninfo 和 test_dnn 列测试，确认 CANN 后端被识别：

```shell #test id="opencv-verify-build"
/usr/local/opencv-cann/bin/opencv_version
/usr/local/opencv-cann/bin/opencv_test_cannops --gtest_list_tests 2>&1 | head -n 20
```

```shell #test-result id="opencv-verify-build" fuzzy='xxx' fuzzy='...'
xxx
CANNxxx...
```

预期：versioninfo 第一行打印 OpenCV 版本号；`opencv_test_cannops --gtest_list_tests` 输出中包含至少一条 `CANN` 前缀的测试用例（mainline 5.0.0 把 CANN 单元测试放在 contrib `cannops` 模块的 `opencv_test_cannops` 二进制里，主仓 `opencv_test_dnn` 已不再带 `*HUAWEI*` 用例）。

#### 跑一遍 CANN 单元测试

`opencv_test_cannops` 内置了 CANN 后端的小型模型测例，会调用 ACL 把模型图下沉到 NPU：

```shell #test id="opencv-cann-run-tests"
/usr/local/opencv-cann/bin/opencv_test_cannops --gtest_color=no --gtest_brief=1 2>&1 | tail -n 20
```

```shell #test-result id="opencv-cann-run-tests" fuzzy='...' fuzzy='xxx'
[==========]xxx
...
[  PASSED  ] xxx
```


## 使用样例：OpenCV 官方 quickstart

完整跑一遍 [OpenCV Quickstart 文档](https://opencv-opencv.mintlify.app/quickstart) 的五类基础 API：读图 → 保存 → 灰度转换 → 缩放 → 在图上画矩形 + 文字。每步都用一行 Python 写完，输出预期可校验。本节使用上面源码构建的 `cv2`（带 CANN 后端），先把它加入 `PYTHONPATH`，下面所有 Python 块 import 的就是源码版：

```shell #test-setup
export PYTHONPATH=/usr/local/opencv-cann/lib/python3.12/site-packages:${PYTHONPATH:-}
```

### 读图 / 写图（imread / imwrite）

构造一张 100×100 的 BGR 零图像（`np.zeros` 模拟 `cv2.imread` 的输入），`cv2.imwrite` 写到 `/tmp`：

```shell #test id="quickstart-imread-imwrite"
python << 'PY'
import cv2
import numpy as np
img = np.zeros((100, 100, 3), dtype=np.uint8)  # 100x100 black BGR
ok = cv2.imwrite('/tmp/opencv_quickstart.png', img)
print('imwrite ok:', ok)
print('shape:', img.shape, 'dtype:', img.dtype)
PY
ls -la /tmp/opencv_quickstart.png
```

```shell #test-result id="quickstart-imread-imwrite" fuzzy='xxx'
imwrite ok: True
shape: (100, 100, 3) dtype: uint8xxx
```

预期：`imwrite` 返回 `True`、文件被创建，dtype 是 `uint8`（OpenCV 默认 BGR 通道序）。

### 颜色空间转换（cvtColor）

`cv2.cvtColor` 把 BGR 转成 GRAY（单通道）。这一步验证 image-processing 基础管线：

```shell #test id="quickstart-cvtcolor"
python << 'PY'
import cv2
import numpy as np
bgr = np.zeros((100, 100, 3), dtype=np.uint8)
bgr[..., 0] = 255  # set blue channel to 255
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
print('gray shape:', gray.shape, 'mean:', float(gray.mean()))
PY
```

```shell #test-result id="quickstart-cvtcolor" fuzzy='xxx'
gray shape: (100, 100) mean: xxx
```

预期：`gray shape: (100, 100)`（单通道），`mean` 在 0~255 之间（Blue=255 经过 BGR→GRAY 权重后约为 29.85）。

### 缩放（resize）

`cv2.resize` 把 100×100 缩到 50×200，用 `cv2.INTER_AREA` 插值（缩小时抗锯齿效果最好）：

```shell #test id="quickstart-resize"
python << 'PY'
import cv2
import numpy as np
img = np.zeros((100, 100, 3), dtype=np.uint8)
resized = cv2.resize(img, (50, 200), interpolation=cv2.INTER_AREA)
print('resized shape:', resized.shape)
PY
```

```shell #test-result id="quickstart-resize" fuzzy='xxx'
resized shape: (200, 50, 3)xxx
```

预期：输出形状 `(200, 50, 3)`——OpenCV 的 shape 是 `(rows, cols, channels)`，所以 `(50, 200)` 的 `(width, height)` 参数实际对应 `resize` 后 rows=200 / cols=50。

### 绘制（rectangle / putText）

在图上画一个绿色矩形和一行白字。`cv2.rectangle` 接受 `(image, pt1, pt2, color, thickness)`，颜色是 BGR 三元组（这里是 `(0, 255, 0)` = 纯绿）；`cv2.putText` 接受 `(image, text, org, fontFace, fontScale, color)`：

```shell #test id="quickstart-draw"
python << 'PY'
import cv2
import numpy as np
img = np.zeros((200, 400, 3), dtype=np.uint8)
cv2.rectangle(img, (10, 10), (390, 190), (0, 255, 0), thickness=2)
cv2.putText(img, 'Hello OpenCV', (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
print('rect+text drawn, image mean:', float(img.mean()))
cv2.imwrite('/tmp/opencv_draw.png', img)
PY
ls -la /tmp/opencv_draw.png
```

```shell #test-result id="quickstart-draw" fuzzy='xxx'
rect+text drawn, image mean: xxx
```

预期：`image mean` 在 0~30 之间（全黑底图 + 一条绿色矩形 + 一行白字后，平均亮度很低）。`/tmp/opencv_draw.png` 被创建。

### 视频写入（VideoWriter）

`cv2.VideoCapture` 既能从摄像头/文件读，也能接受一个「整数」打开一个合成视频源（`0` 通常是默认摄像头；CI 上跳过）；这里反过来用 `cv2.VideoWriter` 验证 video 模块基础管线可用：

```shell #test id="quickstart-video"
python << 'PY'
import cv2
import numpy as np
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter('/tmp/opencv_video.avi', fourcc, 10.0, (320, 320))
frame = np.zeros((320, 320, 3), dtype=np.uint8)
for _ in range(3):
    writer.write(frame)
writer.release()
print('video frames:', 3)
import os
print('video file size:', os.path.getsize('/tmp/opencv_video.avi'))
PY
```

```shell #test-result id="quickstart-video" fuzzy='xxx'
video frames: 3
video file size: xxx
```

预期：3 帧写入完成，`/tmp/opencv_video.avi` 被创建（MJPG 编码后即使是 3 帧黑帧也有几 KB）。

## 使用样例：用 CANN 后端跑一次 DNN 推理

`PYTHONPATH` 已切到 `/usr/local/opencv-cann/lib/python3.12/site-packages`（quickstart 节那段 `#test-setup`，以及 runner 的 `prepare_environment` 在 `os.environ` 里预设了一份），下面 `python` 块 import 的是上面源码构建的 `cv2`，带 CANN 后端：

校验 cv2 版本与 CANN 后端 enum：

```shell #test id="opencv-py-version"
python -c "
import cv2
print('cv2 version:', cv2.__version__)
print('CANN backend available:', cv2.dnn.DNN_BACKEND_CANN)
print('NPU target available:', cv2.dnn.DNN_TARGET_NPU)
"
```

```shell #test-result id="opencv-py-version" fuzzy='xxx'
cv2 version: xxx
CANN backend available: xxx
NPU target available: xxx
```

预期：`CANN backend available` / `NPU target available` 都打印一个非零整数（cv2 把 backend/target 编码为 enum）。

跑一个最小 DNN 推理：从 ONNX Model Zoo 拉一份 SqueezeNet (~5MB)，构造一张零图像，`cv2.dnn.Net.forward()` 在 NPU 上跑前向：

```shell #test id="opencv-cann-infer"
python << 'PY'
import os
import urllib.request
import numpy as np
import cv2

# SqueezeNet 1.0 ONNX from onnx/models — stable raw URL on github.
# ~5 MB; known to be supported by OpenCV's CANN backend
# (the standard sample in the OpenCV Huawei-CANN-Backend wiki).
MODEL_URL = ('https://github.com/onnx/models/raw/main/'
             'validated/vision/classification/squeezenet/'
             'model/squeezenet1.0-1.onnx')
MODEL_PATH = '/tmp/squeezenet1.0-1.onnx'

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
print('model bytes:', os.path.getsize(MODEL_PATH))

net = cv2.dnn.readNetFromONNX(MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CANN)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_NPU)

# SqueezeNet expects 224x224 BGR, ImageNet mean subtract.
image = np.zeros((224, 224, 3), dtype=np.uint8)
blob = cv2.dnn.blobFromImage(
    image, scalefactor=1.0, size=(224, 224),
    mean=(104.006, 116.669, 122.679), swapRB=False, crop=False,
)
net.setInput(blob)
out = net.forward()

print('output shape:', out.shape)
print('output dtype:', out.dtype)
print('top class index:', int(np.argmax(out[0])))
print('top class score:', float(np.max(out[0])))
PY
```

```shell #test-result id="opencv-cann-infer" fuzzy='xxx'
model bytes: xxx
output shape: (1, 1000)xxx
output dtype: float32
top class index: xxx
top class score: xxx
```

预期：

- `output shape: (1, 1000)`：SqueezeNet 输出是 1×1000 的 ImageNet 1000-class logits。
- `top class score` 在 `0~1` 之间：随机初始化输入对应全 0 图像，输出是数据集 bias，对 **某一** 类的得分会高于其他 999 类；如果输出是 NaN / Inf 或全是同一数值，说明 ACL graph 编译失败。
- 第一次跑会触发 ACL graph 编译（耗时数十秒），日志里有 `acl graph compile ...` 字样；第二次起命中 `$HOME/ascend` 下的 AOE 缓存，forward 耗时降到 10 ms 量级。

> 这里只验证 CANN 后端的"链接 + ACL graph 编译 + forward 输出"链路完整。`opencv-cann-run-tests` 那一步才是对 CANN 后端 **opencv 自带测试用例** 的完整验证，两段合起来覆盖 wiki 上的"模型推理 sample"和"单元测试验证"两路样本。

## 小贴士

- 源码构建把 `cv2` 装到 `/usr/local/opencv-cann/lib/python3.12/site-packages`，独立于系统的 `python -c "import cv2"` 解析路径；想恢复系统版本只需 `unset PYTHONPATH` 或把 `prepare_environment` 加的路径从 `os.environ` 里去掉。runner 已经把 `PYTHONPATH` 切到源码版，本地手动跑时也要 export 同样的值才能 import 到带 CANN 后端的 cv2。
- CANN 9.1.0 与 CANN 8.x 在 `ascend-toolkit` 安装路径上行为一致（都是 `/usr/local/Ascend/ascend-toolkit/latest`），但 ACL 接口在 8.x → 9.x 间做过一次 breaking change（新增 `acltdt` 通道、`aclop` 接口签名变更）。如果 OpenCV 是用 CANN 8.x 的 `acllib` 编译的，运行时挂在 9.1.0 上会报 `ACL_ERROR_INVALID_PARAM`；本文档的 `ASCEND_TOOLKIT_HOME` 直接指向当前镜像的 9.1.0 toolkit，从源头避免这个问题。
- 想验证 NPU 推理真生效：把上面 `opencv-cann-infer` 块的 `np.zeros` 换成真实图片（如 `cv2.imread('/tmp/photo.jpg')`），第一次 forward 会触发 ACL graph 编译（`aoe` 缓存落在 `$HOME/ascend` 下），第二次起命中缓存，性能才会贴近 NPU 真实吞吐。
- 多卡推理：`cv2.dnn` 自身不管理 device 亲和性，多张 NPU 的负载分配需要在进程级别串行或起多进程（每进程 `setPreferableTarget` 到不同 NPU ID）；CANN 9.1.0 上 `ASCEND_RT_VISIBLE_DEVICES=0,1` 限定本进程可见设备。
- 想跑官方 OpenCV samples 里更复杂的 sample（如 `object_detection.py`、`segmentation.py`）：这些 sample 接受 `--backend CANN` 命令行参数，把 OpenCV 编译成带 CANN 即可直接复用。
- 清理：
  ```bash
  rm -rf opencv opencv_contrib /usr/local/opencv-cann \
         /tmp/opencv_quickstart.png /tmp/opencv_draw.png \
         /tmp/opencv_video.avi /tmp/squeezenet1.0-1.onnx
  ```