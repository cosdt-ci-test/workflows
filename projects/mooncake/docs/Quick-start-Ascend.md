# 快速开始：在昇腾 NPU 上跑通 Mooncake Transfer Engine

> **阅读本文前**，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 装好 CANN 与驱动。

[Mooncake](https://github.com/kvcache-ai/Mooncake) 是面向大模型服务的 KV Cache 传输与存储引擎。昇腾上推荐的传输路径是 **Ascend Direct**（CMake `-DUSE_ASCEND_DIRECT=ON`，基于 CANN 的 ADXL / HIXL）。

上游通用 Quick Start 以 `pip install mooncake-transfer-engine-npu` 为主，冒烟示例走 TCP，不证明这次跑在昇腾上。本文走源码编译 + Ascend Direct 例程。

---

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为**两张卡**（一张跑 target，一张跑 initiator）。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | 需要安装 toolkit + 驱动固件，并且可以 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| 设备网卡配置 | `/etc/hccn.conf` 存在（驱动安装时写入；容器里把宿主机这份文件挂进来） |
| 编译工具 | cmake、g++、make、git、pkg-config |
| 依赖库 | glog、gflags、libibverbs、jsoncpp、yaml-cpp、OpenSSL、libcurl（见第 3 节） |

**配套机器**

- **机器类型**：Atlas 900 A2（Ascend 910B，双卡）
- **操作系统**：Ubuntu 22.04

**配套镜像**

`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`

---

## 1. 加载 CANN 环境

```shell
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

`npu-smi` 在常见容器里位于 `/usr/local/sbin` 或 `/usr/local/bin`。

Ascend Direct 还依赖 `/etc/hccn.conf`：NPU 驱动在宿主机上写入每张卡的设备网卡 IP。HIXL 初始化时会读这份文件。容器里请把宿主机的 `/etc/hccn.conf` 挂进来，不要自己编造 IP。没有这份文件时，后面的传输会在 ADXL 初始化阶段失败。

```shell
test -f /etc/hccn.conf
```

## 2. 确认 NPU 在线

```shell
npu-smi info
```

至少两张卡。若提示找不到 `npu-smi`，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载（例如 `/dev/davinci0`、`/dev/davinci1`）。

---

## 3. 安装编译依赖

下列包提供 Transfer Engine 链接所需的头文件与库。

```shell #test id="deps"
apt-get update
apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config \
    libgoogle-glog-dev libgflags-dev libibverbs-dev \
    libjsoncpp-dev libnuma-dev libyaml-cpp-dev \
    libssl-dev libcurl4-openssl-dev
ls /usr/include/glog/logging.h /usr/include/gflags/gflags.h
```

输出结果如下：

```shell #test-result id="deps"
...
/usr/include/gflags/gflags.h
/usr/include/glog/logging.h
```

---

## 4. 获取源码并编译 Ascend Direct

克隆上游仓库，检出要用的 ref，打开 `-DUSE_ASCEND_DIRECT=ON`，只编译 `transfer_engine_ascend_direct_perf`（Ascend Direct 的性能例程）。`extern/pybind11` 是 CMake 配置阶段需要的子模块。

将 `<UPSTREAM_REF>` 换成目标**分支、tag 或 commit**（上游默认分支为 `main`）。
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="compile" load="upstream_ref>>UPSTREAM_REF"
git clone --depth 1 https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
git fetch --depth 1 origin <UPSTREAM_REF>
git checkout FETCH_HEAD
git submodule update --init --depth 1 extern/pybind11
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_ASCEND_DIRECT=ON \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_UNIT_TESTS=OFF \
    -DWITH_STORE=OFF \
    -DWITH_STORE_RUST=OFF \
    -DWITH_EP=OFF \
    -DWITH_P2P_STORE=OFF \
    -DUSE_ETCD=OFF \
    -DUSE_REDIS=OFF
cmake --build build --target transfer_engine_ascend_direct_perf -j$(nproc)
ls build/mooncake-transfer-engine/example/transfer_engine_ascend_direct_perf
```

输出结果如下：

```shell #test-result id="compile"
...
build/mooncake-transfer-engine/example/transfer_engine_ascend_direct_perf
...
```

`<UPSTREAM_REF>` 可以是 tag（例如 `v0.3.13`）或 `main`。`--depth 1` 只拉该 ref 的快照。首次编译可能要几分钟到十几分钟。

`-DWITH_STORE=OFF`、`-DWITH_P2P_STORE=OFF`、`-DUSE_ETCD=OFF`、`-DUSE_REDIS=OFF` 关掉这次用不到的 Store / 元数据后端，钉住上游当前默认值，缩短第一次编译；只验证 Transfer Engine 的 Ascend Direct 路径。完整组件请按上游 [Build Guide](https://kvcache-ai.github.io/Mooncake/getting_started/build.html) 打开对应选项。

---

## 5. 在两张 NPU 之间做一次写传输

例程是双进程：先启动 **target**（在 NPU 0 上注册设备内存并监听），再启动 **initiator**（在 NPU 1 上把一块 device buffer 写到 target）。`P2PHANDSHAKE` 会给 target 选一个实际端口，initiator 的 `--segment_id` 必须填日志里那一行 `listening on <IP>:<port>`。

下面把两个进程放在同一个终端里跑，方便第一次验证。`block_iteration=1`、`batch_size=2`、`block_size=16384` 把传输规模压小；正式测带宽再按上游 [Ascend Direct Transport](https://kvcache-ai.github.io/Mooncake/design/transfer-engine/ascend_direct_transport.html) 加大。glog 默认打到 stderr，所以命令末尾有 `2>&1`。

```shell #test id="transfer"
cd Mooncake
export GLOG_logtostderr=1
build/mooncake-transfer-engine/example/transfer_engine_ascend_direct_perf \
    --mode=target \
    --device_logicid=0 \
    --local_server_name=127.0.0.1:12345 \
    --metadata_server=P2PHANDSHAKE \
    --block_iteration=1 \
    --batch_size=2 \
    --block_size=16384 \
    > /tmp/mooncake-target.log 2>&1 &
target_pid=$!
for _ in $(seq 1 60); do
    if ! kill -0 "$target_pid" 2>/dev/null; then
        echo "target exited before listen" >&2
        cat /tmp/mooncake-target.log >&2
        exit 1
    fi
    endpoint=$(grep -Eo 'listening on [^[:space:]]+:[0-9]+' /tmp/mooncake-target.log | tail -1 | awk '{print $3}')
    if [ -n "$endpoint" ]; then
        break
    fi
    sleep 1
done
if [ -z "$endpoint" ]; then
    echo "target did not print a listening endpoint" >&2
    cat /tmp/mooncake-target.log >&2
    kill "$target_pid" 2>/dev/null || true
    exit 1
fi
build/mooncake-transfer-engine/example/transfer_engine_ascend_direct_perf \
    --mode=initiator \
    --device_logicid=1 \
    --local_server_name=127.0.0.1:12346 \
    --metadata_server=P2PHANDSHAKE \
    --segment_id="$endpoint" \
    --operation=write \
    --block_iteration=1 \
    --batch_size=2 \
    --block_size=16384 \
    2>&1 | tee /tmp/mooncake-initiator.log
xfer_ec=${PIPESTATUS[0]}
kill "$target_pid" 2>/dev/null || true
wait "$target_pid" 2>/dev/null || true
if grep -qE 'getTransferStatus FAILED|Sync data transfer timeout|Failed to install Ascend transport' \
    /tmp/mooncake-initiator.log /tmp/mooncake-target.log; then
    echo "transfer reported FAILED/TIMEOUT or Ascend transport failed to install" >&2
    exit 1
fi
exit "$xfer_ec"
```

输出结果如下：

```shell #test-result id="transfer"
...Success to initialize adxl engine:...
...submit transfer suc.
...Test completed: duration ...
```

initiator 日志里的 `Success to initialize adxl engine` 表示这次走了 Ascend Direct。`Test completed:` 表示这一轮写传输跑完。`submit transfer suc.` 只表示提交成功。上游例程在传输 `FAILED` 或 `TIMEOUT` 时仍可能打印后两句并返回 0，所以上面的命令会再扫 `getTransferStatus FAILED`、`Sync data transfer timeout` 和 `Failed to install Ascend transport`。若出现 `Failed to initialize ACL` 或 `Failed to set device ACL`，回到第 2 节检查设备挂载。

二进制默认 `--local_server_name` 指向实验室地址，必须改成 `127.0.0.1`，否则会去连外网 IP。

---

## 6. 下一步

| 目标 | 参考 |
| --- | --- |
| Ascend Direct 环境变量与 HCCS / RDMA 选择 | 上游 [Ascend Direct Transport](https://kvcache-ai.github.io/Mooncake/design/transfer-engine/ascend_direct_transport.html) |
| 从源码打开更多后端 | 上游 [Build Guide](https://kvcache-ai.github.io/Mooncake/getting_started/build.html) |
| 用 PyPI 的 NPU 轮子接 vLLM / SGLang | 上游 [Quick Start](https://kvcache-ai.github.io/Mooncake/getting_started/quick-start.html) |

---

## 故障排查

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| `cmake` 找不到 ADXL / HIXL 头文件 | 未 `source set_env.sh`，或 CANN 版本过旧 | 重做第 1 节；确认 toolkit 含 `include/adxl` |
| `Failed to get device ip from hccn.conf` | 容器或精简环境没有 `/etc/hccn.conf` | 从宿主机挂入或复制该文件；内容须是本机设备 IP，不要填随意地址 |
| `git submodule` 拉 pybind11 超时 | 访问 GitHub 不稳定 | 换网络后重试，或从镜像克隆 `extern/pybind11` |
| target 起不来 / 一直没有 `listening on` | 端口被占，或 ACL 初始化失败 | 看 `/tmp/mooncake-target.log`；换 `--local_server_name` 的端口 |
| `Failed to set device ACL` | 只挂了一张卡，或 `--device_logicid` 超出可见设备 | 用第 2 节确认两张卡；检查 `ASCEND_RT_VISIBLE_DEVICES` |
| initiator 卡住 | `--segment_id` 不是 target 实际端口 | 必须用日志里 `listening on` 后面的 `IP:port`，不要写死 12345 |
