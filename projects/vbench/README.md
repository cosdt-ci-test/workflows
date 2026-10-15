# VBench Ascend Quick Start

Scope: the official README's `human_action` example, exercised through both
the installed `vbench evaluate` command and the `VBench` Python API.

The executable [Quick Start](docs/Quick-start-Ascend.md) checks out the ref
selected by the shared workflow, applies the reviewed patches, installs the
human_action dependency profile and CPU Decord, loads the official UMT
checkpoint, and evaluates five original LaVie videos on one Ascend NPU.
The one-Prompt subset preserves the upstream inputs and scoring algorithm;
it is not a complete benchmark or leaderboard result.

The official LaVie Drive archive may hit quota. A runner can prepopulate
`/data/ci-cache/vbench/lavie.zip` or its five original videos under
`/data/ci-cache/vbench/lavie/human_action/`; the executable document lists
their exact names. Missing data is a test failure, not a skip. HF hosts the
same dataset but requires approved access.

One patch is validated against the latest release (currently v0.1.4).
It retains a small installation profile and adapts only the CLI NPU startup.
The public Python API, root evaluate.py, shared distributed code, scoring
algorithm and result schema retain their upstream behavior. A model-device
log is the only instrumentation added to human_action.

Checkpoint coverage and forward-device/finite-output assertions live in
the guard's runtime_checks helper, using real model forwards. Version pins
live in constraints-npu.txt, rather than the patch. No separate master
compatibility patches are maintained. A new release with incompatible code
fails `git apply --check` and requires patch review; it does not select an
older release.

Local verification (no NPU allocation):

```bash
PYTHONPATH=src python -m unittest discover -s projects/vbench/tests -v
VBENCH_UPSTREAM_ROOT=/path/to/pristine/VBench \
  python -m unittest projects.vbench.tests.test_ascend_patch -v
```

The patch behavior tests require torch and an upstream clone with the tested
release ref (v0.1.4 by default; override with VBENCH_UPSTREAM_REF).
The document acceptance test is gated by `NPU_READY=true` and runs on
`linux-aarch64-a2-1`; schedule remains disabled until a real NPU run passes.
