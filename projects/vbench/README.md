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

Patch application is checked on v0.1.4 and master fd18b3d. CLI quoting differs
between those upstream revisions, so the two small CLI patches are alternatives.
The install-check patch applies only to revisions with the CUDA setup check.
Unknown layouts fail `git apply --check` and must be reviewed explicitly.

Local verification (no NPU allocation):

```bash
PYTHONPATH=src python -m unittest discover -s projects/vbench/tests -v
VBENCH_UPSTREAM_ROOT=/path/to/pristine/VBench \
  python -m unittest projects.vbench.tests.test_ascend_patch -v
```

The patch behavior tests require torch and a clone with v0.1.4 and master refs.
The document acceptance test is gated by `NPU_READY=true` and runs on
`linux-aarch64-a2-1`; schedule remains disabled until a real NPU run passes.
