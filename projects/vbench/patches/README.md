# Ascend adaptation of the official human_action example

The runner invokes `scripts/apply_patches.py` after fetching the target ref
and before installing VBench. It checks every selected patch before changing
the checkout. No patch is silently ignored.

Validated upstream baselines:

- release v0.1.4: `ec30bdc78ff1ba1ac9debe232267be537be1f859`
- master: `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`

`0001` adds a human_action installation profile, optional torch_npu device
registration, `--device` on evaluation launchers, and single-process execution
without NCCL. UMT uses the original FP32 network, 16-frame preprocessing,
sigmoid/top-5/0.85 scoring. It verifies checkpoint coverage (except the
generated `pos_embed`) and records the actual input device for each video.

The two `0002` patches are alternatives for upstream's different CLI path
quoting code. They pass argv directly and propagate subprocess errors.
`0003` accepts an available torch_npu backend in revisions whose installation
script otherwise requires CUDA. It does not replace the upstream CUDA path.

No torchvision source patch, RAFT/Detectron2 adaptation, other dimensions,
or multi-NPU collectives are included in this scope. Applicability/CPU tests
and NPU acceptance are separate: only the NPU workflow can validate Ascend
kernel execution.
