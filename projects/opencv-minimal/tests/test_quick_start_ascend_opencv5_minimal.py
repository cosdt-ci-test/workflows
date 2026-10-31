"""Quick-start-Ascend-OpenCV5-Minimal documentation test: end-to-end case
built on top of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/opencv/docs/Quick-start-Ascend-OpenCV5-Minimal.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every ``shell``
code block carries one of the ``#test`` / ``#test-setup`` / ``#test-result``
labels plus ``id=`` / ``store=`` / ``load='x>>y'`` / ``fuzzy='xxx'``
parameters).

Run: ``python -m unittest tests.test_quick_start_ascend_opencv5_minimal -v 2>&1``

What makes this guard different from the other OpenCV guards:

* it watches the **5.0.0 minimal-patch** doc: the claim under test is
  that OpenCV 5.0.0 needs exactly **two** patch groups (the
  ``cv::MatShape`` overload + the narrow CANN op headers) and that the
  result is byte-identical to the known-good 5.x tree.  The
  ``ocv5m-patches-applied`` block asserts all three markers landed
  (1 ``cv::MatShape`` overload decl, ``array_ops.h`` in place, 23 layer
  TUs carrying a narrow op header), so a doc edit that breaks the patch
  scripts fails here rather than silently shipping an unpatched tree;
* the doc is the **only** one of the four OpenCV docs that builds 5.0.0
  with ``BUILD_opencv_apps=OFF``.  ``apps/model-diagnostics`` is an
  executable and links CANN's ``libge_compiler.so`` / ``libgraph.so``
  directly -- shared libraries may keep undefined symbols, executables
  may not -- so without that flag the ``all`` target fails with
  hundreds of ``undefined reference`` lines.  The guard inherits the
  flag from the doc and fails loudly if it is ever dropped;
* 5.0's default DNN engine is the **new graph engine**, which ignores
  ``setPreferableBackend(CANN)`` (two WARN lines, zero GE graph
  compiles, everything stays on the CPU).  The doc's inference block
  therefore pins ``OPENCV_FORCE_DNN_ENGINE=1`` and asserts the three
  CANN log markers plus >= 100 converted layers, so a future release
  flipping the default cannot make this guard pass on CPU-only;
* the install prefix is **``/usr/local/opencv-cann-50``**, distinct from
  every other OpenCV guard's prefix (``/usr/local/opencv-cann`` /
  ``-411`` / ``-414`` / ``-5``): all of them may land on the same
  self-hosted runner and must not clobber each other;
* the doc pins OpenCV **5.0.0** itself (no ``<ref>`` placeholder), so
  the thin workflow pins ``fixed_ref: 5.0.0`` and the engine monitors
  that tag's commit instead of the latest 5.x release.

Environment variables (injected by the quick-start engine workflow
``quick-start-template.yml``, triggered by
``opencv-quick-start-minimal.yml``):
    ``MONITORED_DOC_URL``         Required; raw URL of the document under test.
    ``UPSTREAM_REF``              Injected but unused (doc pins 5.0.0).
    ``NPU_READY=true``            Required, otherwise the class is skipped.
                                  End-to-end tests only run on the NPU runner:
                                  local dev machines / normal ubuntu runners
                                  have no ``/dev/davinci*`` device.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from pathlib import Path

from workflows.markdown_doc_test_base import MarkdownDocTestBase


def _is_truthy(value: str | None) -> bool:
    """``'true'`` -> True (case-insensitive); anything else (including unset) -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true`` is set, releasing the skip."""
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscendOpenCV5Minimal(MarkdownDocTestBase, unittest.TestCase):
    """``Quick-start-Ascend-OpenCV5-Minimal.md`` end-to-end test: fetch doc
    -> validate contract -> run ``#test-setup`` / ``#test`` in order ->
    compare against ``#test-result``.

    The doc's heaviest steps are:
        1. ``apt-get install`` build deps (cmake / gcc / libjpeg / libpng /
           libtiff / zlib);
        2. ``git clone --depth 1 --branch 5.0.0`` opencv;
        3. the two patch groups (MatShape overload + narrow CANN op
           headers -- the narrow headers are what keeps ``libopencv_dnn``
           inside the aarch64 +-128 MiB branch-range window: 14.7 MiB of
           ``.text`` instead of 53.5 MiB with ``all_ops.h``);
        4. CANN library layout bridge (3 symlinks into ``<CANN>/lib64``);
        5. ``cmake -DWITH_CANN=ON -DCMAKE_BUILD_TYPE=Release`` configure
           with ``-DBUILD_opencv_apps=OFF``;
        6. ``cmake --build . --target install`` (Release -O3 -- with the
           narrow headers the dnn TUs are light, unlike the ``all_ops.h``
           builds of the sibling guards);
        7. MobileNetV2 ONNX forward with the **classic** engine
           (``OPENCV_FORCE_DNN_ENGINE=1``): asserts the three CANN log
           markers, >= 100 layers converted to CANN ops, top-1 = 372, and
           numeric agreement with a CPU reference net.
    """

    # The build is the long pole: the configure step downloads IPPICV /
    # ADE, the dnn module compiles ~39 CANN TUs and the first ACL graph
    # build is a cold GE compile (0.5-1 min). 9000s per command keeps
    # generous headroom under the engine's job budget.
    DEFAULT_COMMAND_TIMEOUT = 9000

    USER_AGENT = 'cosdt-ci-test/quick-start'  # monitored source is the fork under cosdt-ci-test org
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # CANN toolkit emits this typo (sic) in its Python driver
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
        # OpenCV's DNN/CANN backend silently falls back to the CPU engine
        # when a layer doesn't support CANN. A top-1 of 372 is produced by
        # the CPU engine too, so without this marker the guard cannot tell
        # "ran on the NPU" from "fell back to the CPU".
        'Going back to default backend on CPU target',
    )

    # Install prefix of the doc's source build. Distinct from every other
    # OpenCV guard's prefix so a 5.0.0-minimal install can never clobber
    # (or be clobbered by) another guard's install when they all land on
    # the same self-hosted runner.
    _OPENCV_INSTALL = '/usr/local/opencv-cann-50'
    _OPENCV_PYTHONPATH = f'{_OPENCV_INSTALL}/lib/python3.12/site-packages'

    # CANN's environment script. Sourced into ``os.environ`` by
    # ``prepare_environment`` -- see the method docstring for why.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # Build parallelism for the doc's ``cmake --build`` block. The doc
    # defaults to 2 (safe on a 32 GB cgroup); the CI runner has more
    # RAM, so the guard raises it. The doc reads it as
    # ``${OPENCV_BUILD_JOBS:-2}`` -- an explicit override, not a silent
    # one.
    _BUILD_JOBS = '6'

    # The ONNX model is 13.9 MB and lives in projects/opencv/fixtures/
    # (shared with the other guards -- one copy in the repo). The doc's
    # inference script looks for ``fixtures/mobilenetv2-12.onnx``
    # relative to the working dir, so stage it here.
    _MODEL_NAME = 'mobilenetv2-12.onnx'
    _FIXTURE_SRC = Path(__file__).resolve().parents[2] / 'opencv' / 'fixtures'
    _FIXTURE_DST = Path('fixtures')

    # ----------------------------------------------------------
    # prepare_environment
    # ----------------------------------------------------------

    @classmethod
    def _source_cann_env(cls) -> None:
        """Fold CANN's ``set_env.sh`` into ``os.environ``.

        The doc has its own ``#test-setup`` block running
        ``source .../set_env.sh``, but that block is its own subprocess:
        ``MarkdownDocTestBase.execute`` snapshots ``os.environ`` into one
        shared ``env`` dict and hands that same dict to every block, so a
        ``source`` (or an ``export``) inside one block never reaches the
        next one.  The doc's later blocks -- the GE graph compile and the
        ONNX forward -- need exactly what that ``source`` provides:

        * ``LD_LIBRARY_PATH`` -> ``libacl_op_compiler.so`` / ``libgraph.so``
        * ``PATH``            -> the ascendc / tbe compiler helpers
        * ``PYTHONPATH``      -> CANN's ``tbe`` python modules; without
          them the GE graph compile dies with "There is no valid so about
          OpsKernelInfoStore or GraphOptimizer" (ret=0xFFFFFFFF)

        so we source it once here, into the environment the base class
        snapshots.  Only the *delta* is applied (added / changed keys) so
        a failing ``source`` cannot strip the runner's own variables.
        """
        if not Path(cls._CANN_SET_ENV).is_file():
            print(f'setup: CANN set_env.sh not found at {cls._CANN_SET_ENV} '
                  '- skipping (the doc sources it in its own block)')
            return
        before = dict(os.environ)
        proc = subprocess.run(
            ['bash', '-c', f'source "{cls._CANN_SET_ENV}" >/dev/null 2>&1; env -0'],
            capture_output=True,
        )
        raw = proc.stdout
        if not raw:
            print(f'setup: sourcing {cls._CANN_SET_ENV} produced no env '
                  f'(rc={proc.returncode}) - skipping')
            return
        applied = []
        for entry in raw.split(b'\0'):
            if not entry:
                continue
            key, _, value = entry.decode('utf-8', 'replace').partition('=')
            if before.get(key) != value:
                os.environ[key] = value
                applied.append(key)
        print(f'setup: sourced CANN env from {cls._CANN_SET_ENV} '
              f'({len(applied)} vars changed: {", ".join(sorted(applied)[:12])})')

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN's env + set PYTHONPATH to the source-built cv2 +
        stage the ONNX fixture + raise the build parallelism.

        Each ``#test`` / ``#test-setup`` block is its own
        ``subprocess.run(env=env)`` call and the env dict is captured
        once at the top of ``MarkdownDocTestBase.execute``, so a
        doc-level ``export PYTHONPATH=...`` would only live for that one
        subprocess. We set it once on ``os.environ`` here and the base
        class copies it into every block.
        """
        # 1) CANN environment (LD_LIBRARY_PATH / PATH / PYTHONPATH-tbe).
        cls._source_cann_env()

        # 2) PYTHONPATH: PREPEND the source-built OpenCV site-packages
        # so the doc's Python blocks import the cv2 with the CANN
        # backend. PREPEND, not setdefault: the CANN image itself
        # exports PYTHONPATH=/usr/local/Ascend/cann-9.x/python/
        # site-packages:... (its TBE / ACL python bits), and setdefault
        # would silently keep the image value -- the GE graph compile in
        # the DNN block then dies with "CANN graph check failed,
        # ret = 4294967295" (the TBE python module is missing and the
        # log never mentions it).
        _existing = os.environ.get('PYTHONPATH', '')
        if cls._OPENCV_PYTHONPATH not in _existing:
            os.environ['PYTHONPATH'] = (
                f'{cls._OPENCV_PYTHONPATH}:{_existing}' if _existing
                else cls._OPENCV_PYTHONPATH
            )
        print(
            f'setup: PYTHONPATH -> {os.environ["PYTHONPATH"]} '
            '(prepended source-built cv2 5.0.0; kept image entries)'
        )

        # 3) Build parallelism (see _BUILD_JOBS).
        os.environ['OPENCV_BUILD_JOBS'] = cls._BUILD_JOBS
        print(f'setup: OPENCV_BUILD_JOBS -> {cls._BUILD_JOBS}')

        # 4) Stage the ONNX fixture next to the working dir so the
        # doc's ``fixtures/<model>`` path resolves. The doc falls back
        # to downloading from onnx/models when the file is absent, but
        # github raw is flaky from the CI network -- prefer the
        # in-repo copy.
        src = cls._FIXTURE_SRC / cls._MODEL_NAME
        if not src.is_file():
            raise unittest.SkipTest(
                f'fixture {src} missing -- cannot run the NPU leg offline'
            )
        cls._FIXTURE_DST.mkdir(parents=True, exist_ok=True)
        dst = cls._FIXTURE_DST / cls._MODEL_NAME
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy(src, dst)
        print(f'setup: fixture -> {dst} ({dst.stat().st_size} bytes)')

        # 5) mistune is installed by the engine's test step; nothing to
        # do here. OpenCV itself is intentionally NOT pre-installed:
        # the doc's own blocks clone + configure + build + install it, so
        # a broken source build surfaces as a mismatch in the doc's own
        # blocks rather than being masked by a pre-built copy.

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class.

        ``@unittest.skipIf`` only skips the test *method* -- ``setUpClass``
        itself always runs. The ``if _e2e_enabled()`` body guard below is
        what actually keeps heavy setup from firing when ``NPU_READY`` is
        unset.
        """
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        """Template-method entry point. The base class
        ``run_template()`` runs the full ``pre_process`` -> ``parse`` ->
        ``execute`` -> ``post_process`` flow. ``prepare_environment`` is
        triggered by ``setUpClass`` once, not from ``run_template()``."""

        self.run_template()


if __name__ == '__main__':
    unittest.main()
