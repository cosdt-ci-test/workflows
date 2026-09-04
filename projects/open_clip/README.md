# open_clip

This directory contains the Ascend Quick Start guard for
[mlfoundations/open_clip](https://github.com/mlfoundations/open_clip); it does
not contain open_clip source code.

The guard follows the upstream README inference example and runs one real
pretrained `ViT-B-32` image/text similarity calculation on `npu:0`. It clones
the latest upstream release selected by the shared Quick Start engine, installs
that checkout in editable mode, loads the upstream `docs/CLIP.png`, and checks
that `a diagram` is the highest-scoring label.

The first version deliberately does not include training, HCCL, FSDP, audio,
CoCa, or `torch.compile`. Those are separate expansion candidates after the
single-card inference path is stable. Model downloads use a persistent
Hugging Face cache mounted by the workflow.
