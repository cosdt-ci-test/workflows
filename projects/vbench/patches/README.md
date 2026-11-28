# Ascend adaptation of the official human_action example

The runner invokes `scripts/apply_patches.py` after fetching the target ref
and before installing VBench. It checks every selected patch before changing
the checkout. No patch is silently ignored.

Validated release:

- v0.1.4: `ec30bdc78ff1ba1ac9debe232267be537be1f859`

The single patch changes six upstream files:

- setup.py, MANIFEST.in and requirements-human-action.txt: a minimal optional
  dependency profile, leaving the full profile unchanged. Concrete version
  pins are maintained in the guard's constraints-npu.txt.
- vbench/cli/evaluate.py: forward --device and propagate child failures.
- vbench/launch/evaluate.py: register torch_npu and select a single NPU without
  NCCL; CUDA retains its original distributed initialization and remains the
  default device.
- vbench/human_action.py: remove an unused CLIP import and log the model device.

The public API, root evaluate.py and distributed.py remain byte-for-byte
unchanged. The UMT network, sampling, scoring and result schema stay intact.
Checkpoint and forward-device checks run through the guard's runtime_checks
helper, not through modified upstream scoring code.

The workflow continues to resolve the latest release dynamically. No master
compatibility branch is maintained; upstream changes require patch review.
Only the NPU acceptance workflow can validate Ascend kernel execution.
