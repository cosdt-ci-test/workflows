"""Entry-point audit for the MiniOneRec NPU change.

Document under test: ``projects/minionerec/docs/NPU-entrypoints.md``.

``Quick-start-Ascend.md`` only executes the SFT and evaluation paths. This
audit covers the remaining entry scripts (GPR variants, the RL pair, SASRec
and the RQ preprocessing scripts) and proves, for each of the 13 entry
points, that

* the file imports ``npu_support``;
* ``import npu_support`` resolves from that script's own working directory
  -- ``python <script>`` puts the script directory first on ``sys.path`` and
  *not* the repository root, which the scripts under ``rq/`` rely on (they
  are launched from their own directory, e.g. ``cd rq && bash rqvae.sh``);
* the module body imports cleanly on NPU, i.e. module-level ``torch.cuda``
  use (``evaluate.py`` picks its device at import time) still works under
  ``transfer_to_npu``.

No model or dataset is downloaded: the run is clone + pip + import only, so
it takes minutes while the quick start takes hours.

Run: ``python -m unittest tests.test_npu_entrypoints -v 2>&1``

Environment variables (injected by GitHub workflow
``minionerec-npu-entrypoints.yml``):
    ``MONITORED_DOC_URL``   Required; URL of the audit document under test.
    ``UPSTREAM_REF``        Required; target ref, substituted where
                            ``<UPSTREAM_REF>`` appears in the doc.
    ``NPU_READY=true``      Required, otherwise the class is skipped.
"""

from __future__ import annotations

import tests.test_quick_start_ascend as quick_start


class TestNpuEntrypoints(quick_start.TestQuickStartAscend):
    """``NPU-entrypoints.md`` audit.

    Derives from the quick-start test so the CANN env merge, the CUDA
    exclusion list (``PIP_CONSTRAINT``) and the torch/torch_npu probe are
    shared -- only the document and the timeout differ.

    The base class is reached through its module (not imported by name):
    ``unittest`` collects every ``TestCase`` subclass that ends up in a
    module's namespace, and importing the class directly would make this
    module run the quick-start flow a second time.
    """

    DEFAULT_COMMAND_TIMEOUT = 1200  # 27 short import subprocesses; no training
