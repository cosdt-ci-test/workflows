# flash-linear-attention

This directory contains the Ascend Quick Start guard for
[fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention).

The Quick Start follows the upstream
[Ascend NPU installation guide](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md#ascend-npu)
and exercises a real `GatedDeltaNet` forward and backward pass. The shared
Quick Start engine selects the latest release; the document checks out that
exact ref and lets its `[npu]` extra select the matching Torch, Torch-NPU, and
Triton-Ascend versions.

The container image is intentionally explicit because it must match the CANN
generation of the selected release. When upstream publishes a release with a
new NPU stack, a failing guard is the signal to update the image and documented
prerequisites together.

The workflow uses Huawei SWR's mirror of the upstream CANN 9.0 image to avoid
pulling the multi-gigabyte image from Docker Hub on a cold NPU runner.
