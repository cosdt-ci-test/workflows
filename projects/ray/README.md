# Ray

This directory contains Ascend guard data for
[ray-project/ray](https://github.com/ray-project/ray), not Ray source code.
Ray is registered as a basic-support inference-acceleration project in phase A.

## Examples

The examples workflow runs two files that already exist in the upstream Ray
repository:

- `python/ray/tests/accelerators/test_npu.py` exercises Ray Core's native
  `NPUAcceleratorManager` contract;
- `python/ray/train/tests/test_torch_device_manager.py` exercises Ray Train's
  `NPUTorchDeviceManager`; the CI overlay selects `test_npu_device_manager`
  so unrelated CUDA and TPU tests in the same upstream file are not run.

The project-local `scripts/check_manifest.py` interprets the manifest's
`scan.paths` because these two upstream tests live in unrelated directories.
The shared manifest checker remains unchanged, and the guard does not add or
replace a Ray example.
`setup_example.sh` installs a released target directly from its matching PyPI
version. Development targets such as Ray master use the official Linux aarch64
wheel built from the exact target commit.
Ray wheels intentionally exclude test packages. After the matching runtime is
installed, the setup uses the target checkout's own `python/ray/setup-dev.py`
to link only `ray/tests` into the wheel. A project-local verifier rejects a
runtime version mismatch or a test package that resolves outside the target
checkout.

## Quick Start

`docs/Quick-start-Ascend.md` is based on Ray's upstream Accelerator Support
document. It validates two-device discovery, Task/Actor allocation,
`ASCEND_RT_VISIBLE_DEVICES`, and a real `torch_npu` operation. The workflow uses
the repository's shared Quick Start engine; scheduled polling stays disabled
until the first manual run is green.
