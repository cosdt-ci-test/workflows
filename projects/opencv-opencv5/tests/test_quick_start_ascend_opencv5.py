"""Quick-start-Ascend-OpenCV5 documentation test: end-to-end case built on
top of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/opencv/docs/Quick-start-Ascend-OpenCV5.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every ``shell``
code block carries one of the ``#test`` / ``#test-setup`` / ``#test-result``
labels plus ``id=`` / ``store=`` / ``load='x>>y'`` / ``fuzzy='xxx'``
parameters).

Run: ``python -m unittest tests.test_quick_start_ascend_opencv5 -v 2>&1``

What makes this guard different from the other two OpenCV guards:

* it watches the **OpenCV 5.0 feature-tour** doc, whose distinguishing
  claim is the *engine* comparison of §10: 5.0's default ``ENGINE_AUTO``
  picks the new graph engine, which **ignores** ``setPreferableBackend`` /
  ``setPreferableTarget`` (two WARN lines, zero CANN graph compiles) and
  therefore never touches the NPU. The ``ocv5-new-engine-default`` block
  asserts exactly that (2 warnings / 0 ``ge::Graph to OM buffer``), and
  ``ocv5-cann-infer`` asserts the classic engine *does* compile the graph
  and converts >= 100 layers. If a future OpenCV release flips the
  default, this guard fails loudly instead of silently going CPU-only;
* it also builds the ``cannops`` gtest target and runs it
  (``ocv5-cann-run-tests``: 78 cases, 25 excluded by the known
  CANN 9.x / 910B defect list, 53 expected to pass);
* the install prefix is **``/usr/local/opencv-cann-5``**, distinct from
  the other two guards' prefixes (``/usr/local/opencv-cann`` for the
  patched 5.x doc, ``/usr/local/opencv-cann-411`` for the 4.11.0
  zero-patch doc): all three may land on the same self-hosted runner and
  must not clobber each other;
* the doc follows the upstream *latest release* tag via ``<ref>`` (like
  the main OpenCV guard, unlike the 4.11.0 guard which pins
  ``fixed_ref``), and its patch scripts assert on 5.0-only upstream
  strings — a 5.1/6.0 release makes the doc's own patch blocks fail,
  which is the intended "review the patches" signal.

Environment variables (injected by the quick-start engine workflow
``quick-start-template.yml``, triggered by
``opencv-quick-start-opencv5.yml``):
    ``MONITORED_DOC_URL``         Required; raw URL of the document under test.
    ``UPSTREAM_REF``              Injected; the doc consumes it as ``<ref>``.
    ``NPU_READY=true``            Required, otherwise the class is skipped.
                                  End-to-end tests only run on the NPU runner:
                                  local dev machines / normal ubuntu runners
                                  have no ``/dev/davinci*`` device.
"""

from __future__ import annotations

import os
import shutil
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


class TestQuickStartAscendOpenCV5(MarkdownDocTestBase, unittest.TestCase):
    """``Quick-start-Ascend-OpenCV5.md`` end-to-end test: fetch doc ->
    validate contract -> run ``#test-setup`` / ``#test`` in order ->
    compare against ``#test-result``.

    The doc's heaviest steps are:
        1. ``apt-get install`` build deps (cmake / gcc / libjpeg / libpng /
           libtiff / zlib);
        2. ``git clone --depth 1 --branch <ref>`` opencv + opencv_contrib;
        3. three source patches (``cv::MatShape`` forwarding, narrow CANN
           op headers for the aarch64 +-128 MiB branch-range limit, NULL
           stream fallback);
        4. CANN library layout bridge (3 symlinks into ``<CANN>/lib64``);
        5. ``cmake -DWITH_CANN=ON -DCMAKE_BUILD_TYPE=Debug`` configure;
        6. ``cmake --build . --target install`` (Debug -O0 with the
           narrow-header patch; every ``all_ops.h`` TU is heavy: ~3 GB RSS);
        7. ``opencv_test_cannops`` gtest (53 of 78 cases expected to pass);
        8. Python binding smoke + the 5.0-only ``transposeND`` / ``flipND``
           core ops;
        9. MobileNetV2 ONNX forward twice: once with 5.0's default
           (new) graph engine — asserted to *not* reach the NPU — and once
           with the classic engine (``OPENCV_FORCE_DNN_ENGINE=1``) —
           asserted to compile the GE graph and return top-1 = 372.
    """

    # The build is the long pole: Debug -O0 with CANN's ``all_ops.h``
    # (~1500 ops) in the dnn layer TUs costs ~3 GB RSS each. 9000s per
    # command keeps generous headroom under the engine's job budget for
    # the configure (IPPICV / ADE downloads), the build itself and the
    # first ACL graph build.
    DEFAULT_COMMAND_TIMEOUT = 9000

    USER_AGENT = 'cosdt-ci-test/quick-start'  # monitored source is the fork under cosdt-ci-test org
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # CANN toolkit emits this typo (sic) in its Python driver
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # Install prefix of the doc's source build. Distinct from the other
    # two OpenCV guards' prefixes so a 5.0 install can never clobber (or
    # be clobbered by) the patched 5.x / 4.11.0 installs when all three
    # guards land on the same self-hosted runner.
    _OPENCV_INSTALL = '/usr/local/opencv-cann-5'
    _OPENCV_PYTHONPATH = f'{_OPENCV_INSTALL}/lib/python3.12/site-packages'

    # Build parallelism for the doc's ``cmake --build`` block. The doc
    # defaults to 2 (safe on a 32 GB cgroup); the CI runner has more
    # RAM, so the guard raises it. The doc reads it as
    # ``${OPENCV_BUILD_JOBS:-2}`` — an explicit override, not a silent
    # one.
    _BUILD_JOBS = '6'

    # The ONNX model is 13.9 MB and lives in projects/opencv/fixtures/
    # (shared with the other guards — one copy in the repo). The doc's
    # inference script looks for ``fixtures/mobilenetv2-12.onnx``
    # relative to the working dir, so stage it here.
    _MODEL_NAME = 'mobilenetv2-12.onnx'
    _FIXTURE_SRC = Path(__file__).resolve().parents[2] / 'opencv' / 'fixtures'
    _FIXTURE_DST = Path('fixtures')

    # ----------------------------------------------------------
    # prepare_environment
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Set PYTHONPATH to the source-built cv2 + stage the ONNX
        fixture + raise the build parallelism.

        Each ``#test`` / ``#test-setup`` block is its own
        ``subprocess.run(env=env)`` call and the env dict is captured
        once at the top of ``MarkdownDocTestBase.execute``, so a
        doc-level ``export PYTHONPATH=...`` would only live for that one
        subprocess. We set it once on ``os.environ`` here and the base
        class copies it into every block.
        """
        # 1) PYTHONPATH: PREPEND the source-built OpenCV site-packages
        # so the doc's Python blocks import the cv2 with the CANN
        # backend. PREPEND, not setdefault: the CANN image itself
        # exports PYTHONPATH=/usr/local/Ascend/cann-9.x/python/
        # site-packages:... (its TBE / ACL python bits), and setdefault
        # would silently keep the image value — the GE graph compile in
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

        # 2) Build parallelism (see _BUILD_JOBS).
        os.environ['OPENCV_BUILD_JOBS'] = cls._BUILD_JOBS
        print(f'setup: OPENCV_BUILD_JOBS -> {cls._BUILD_JOBS}')

        # 3) Stage the ONNX fixture next to the working dir so the
        # doc's ``fixtures/<model>`` path resolves. The doc falls back
        # to downloading from onnx/models when the file is absent, but
        # github raw is flaky from the CI network — prefer the
        # in-repo copy.
        src = cls._FIXTURE_SRC / cls._MODEL_NAME
        if not src.is_file():
            raise unittest.SkipTest(
                f'fixture {src} missing — cannot run the NPU leg offline'
            )
        cls._FIXTURE_DST.mkdir(parents=True, exist_ok=True)
        dst = cls._FIXTURE_DST / cls._MODEL_NAME
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy(src, dst)
        print(f'setup: fixture -> {dst} ({dst.stat().st_size} bytes)')

        # 4) mistune is installed by the engine's test step; nothing to
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

        ``@unittest.skipIf`` only skips the test *method* — ``setUpClass``
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
