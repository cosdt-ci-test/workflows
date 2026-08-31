# tensorflow

本目录看护 TensorFlow 2.6.5 在昇腾 NPU 上通过 TF Adapter 9.1.0 运行的
Quick Start。它不是对 TensorFlow 最新 release 的通用 NPU 看护，而是一条
固定、受官方兼容矩阵约束的历史版本基线：

- TensorFlow `v2.6.5`；
- CANN 9.1.0；
- TF Adapter `tfa_v0.0.49_9.1.0` / `npu_device` 2.6.5；
- Python 3.9.25；
- Linux aarch64、单张 Ascend 910B。

用户文档位于 `docs/Quick-start-Ascend.md`。测试代码使用仓库公共的
`MarkdownDocTestBase` 执行文档里带 `#test-setup`、`#test` 和
`#test-result` 标签的 shell 代码块；无标签代码块只用于说明，不参与 CI。

工作流 `.github/workflows/tensorflow-quick-start.yml` 是共享
`quick-start-template.yml` 的薄调用方。可选的 `fixed_ref` 输入使工作流
记录并测试 `v2.6.5`，而不是把 TensorFlow 最新 release 错写进测试结果。
未传 `fixed_ref` 的现有项目仍沿用原来的 latest release/tag/HEAD 回退逻辑。

CI 使用与 ms-swift 相同来源和版本的 CANN 9.1.0 镜像；选择 `devel` 变体是
因为官方 aarch64 安装流程要求 HDF5 1.10.5 和 h5py 3.1.0，而 h5py 需要在
Python 3.9 环境中编译。文档从华为云镜像源码安装 Python 3.9.25，与镜像的
Python 3.12 并存；TensorFlow 2.6.5 使用 Ascend 官方镜像项目公开的
aarch64 wheel，TF Adapter 使用发布的 `npu_device` wheel。Runner 已自动
配置一张 NPU，工作流不重复挂载设备、驱动、`npu-smi` 或缓存目录。
