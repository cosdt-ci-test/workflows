# tensorflow

本目录看护 TensorFlow 1.15 在昇腾 NPU 上通过 TF Adapter 9.1.0 运行的
Quick Start。它不是对 TensorFlow 当前 2.x release 的通用 NPU 看护，而是
一条固定、受官方兼容矩阵约束的历史版本基线：

- TensorFlow `v1.15.0`；
- CANN 9.1.0；
- TF Adapter `tfa_v0.0.49_9.1.0` / `npu_bridge` 1.15.0；
- Python 3.7.10；
- Linux aarch64、单张 Ascend 910B。

用户文档位于 `docs/Quick-start-Ascend.md`。测试代码使用仓库公共的
`MarkdownDocTestBase` 执行文档里带 `#test-setup`、`#test` 和
`#test-result` 标签的 shell 代码块；无标签代码块只用于说明，不参与 CI。

工作流 `.github/workflows/tensorflow-quick-start.yml` 是共享
`quick-start-template.yml` 的薄调用方。共享模板新增的可选 `fixed_ref`
输入是本项目唯一需要的公共修改：它使工作流记录并测试 `v1.15.0`，而不是
把 TensorFlow 最新 2.x release 错写进测试结果。未传 `fixed_ref` 的现有项目
仍沿用原来的 latest release/tag/HEAD 回退逻辑。

CI 使用与 ms-swift 相同来源和版本的 CANN 9.1.0 镜像；选择 `devel` 变体是
因为官方 aarch64 安装流程要求 HDF5 1.10.5 和 h5py 2.8.0，而 h5py 需要在
Python 3.7 环境中编译。文档从华为云镜像源码安装 Python 3.7.10，与镜像的
Python 3.12 并存；uv 由 Python 3.12 启动，通过 `--python` 只向 3.7 安装
TensorFlow 依赖。Runner 已自动配置一张 NPU，工作流不重复挂载设备、驱动、
`npu-smi` 或缓存目录。
