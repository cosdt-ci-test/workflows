# VBench Ascend Quick Start

Scope: the official README's `human_action` example, exercised through both
the installed `vbench evaluate` command and the `VBench` Python API.

The executable [Quick Start](docs/Quick-start-Ascend.md) checks out the ref
selected by the shared workflow, applies the reviewed patches, installs the
human_action dependency profile and CPU Decord, loads the official UMT
checkpoint, and evaluates five original LaVie videos on one Ascend NPU.
The one-Prompt subset preserves the upstream inputs and scoring algorithm;
it is not a complete benchmark or leaderboard result.

When `cache-seed/vbench/manifest.yaml` is present, asset preparation invokes
the shared `scripts/cache_seed.py` with an explicit VBench cache root. It
verifies the five original videos against the manifest, reuses matching
cached files, and restores corrupt ones from the repository bundle. Invalid
or incomplete bundles fail rather than silently falling back to a download.
No changes to the shared workflow or seeder are needed.

Without a bundle, the official LaVie Drive archive may hit quota. A runner can prepopulate
`/data/ci-cache/vbench/lavie.zip` or its five original videos under
`/data/ci-cache/vbench/lavie/human_action/`; the executable document lists
their exact names. Missing data is a test failure, not a skip. HF hosts the
same dataset but requires approved access.

## Staging the official video cache seed

The repository includes the five original videos and their generated manifest
under [cache-seed/vbench](../../cache-seed/vbench/README.md). Their SHA256 values
were verified against the official dataset's LFS metadata. A cold runner cache
is populated from this bundle, without Google Drive access or an HF token.
The following commands document how to rebuild it; do not add an empty
manifest or substitute generated videos.

On a machine with access to the official data, either extract the five original
files from the upstream LaVie archive, or obtain approved access to the
[official HF dataset](https://huggingface.co/datasets/Vchitect/VBench_sampled_video)
and log in locally. Do not put the token into this repository or the CI logs.
For example, after approval and `hf auth login`:

```bash
for index in 0 1 2 3 4; do
  hf download Vchitect/VBench_sampled_video \
    "lavie/human_action/A person is strumming guitar-${index}.mp4" \
    --repo-type dataset \
    --revision 8c89bb0218a12323d04821d11ec03c780d0dfb9c \
    --local-dir /tmp/vbench-official
done
```

From the guard repository root, stage **only** those five MP4 files. The
shared bundler computes the real SHA256 values; do not hand-write them:

```bash
python scripts/bundle_cache.py --project vbench \
  --src /tmp/vbench-official/lavie/human_action \
  --prefix lavie/human_action
```

Review `cache-seed/vbench/manifest.yaml` and the five files under
`cache-seed/vbench/lavie/human_action/`, including their provenance and
redistribution terms, before committing them. Do not include HF download
metadata or credentials. With the bundle in the checked-out workflow ref,
the existing `prepare-assets` document step seeds it automatically into
`${VBENCH_CACHE_DIR:-/root/.cache/vbench}` and validates video decoding.
An additional `cache-seed.yml` dispatch is unnecessary; that workflow's
default HF root is **not** VBench's mounted cache volume.

## Patch scope and verification

One patch is validated against the latest release (currently v0.1.4).
It retains a small installation profile and adapts only the CLI NPU startup.
The public Python API, root evaluate.py, shared distributed code, scoring
algorithm and result schema retain their upstream behavior. A model-device
log is the only instrumentation added to human_action.

The Quick Start is written for Ascend users: installation, input preparation,
the public CLI/API and output locations. Runner/cache delivery and patch
maintenance details belong in this README, not in the user-facing guide.

Checkpoint coverage and forward-device/finite-output assertions live in
the guard's runtime_checks helper, using real model forwards. The project test
entry wraps the unchanged document API code in `check_api`, which also checks
both CLI/API result files. Users can run that API example without the test
helpers or a preceding CLI run. Version pins
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
