#!/usr/bin/env python3
"""One-leg NPU+CPU opencv_version smoke — `samples/python/opencv_version.py`
headless variant.

Why this exists as a separate self-hosted wrapper (mirrors the
test_npu_dnn.py pattern):

  projects/opencv/examples_manifest.yaml 的 "cpu_only"段登记了
  `samples/python/opencv_version.py` 为 unsupported — 注释明说"纯版本
  打印 ... CANN enum 断言已并入 setup_example.sh 第 5 步 sanity"。
  这是 setup 层已有的 DNN_BACKEND_CANN != 0 探针的 redundancy，但
  `opencv_version.py` 本身在 BUILD_LIST 不编 highgui 的 CI 镜像下会
  因末尾无条件 `cv.destroyAllWindows()` 抛 cv2.error。把这条腿单
  独挂在 supported 段，等同于用一个清晰的 self-hosted entry 把
  同一探针价值以"examples 矩阵"的形式落地——零 NPU 算子，但
  `cv2.dnn.DNN_BACKEND_CANN != 0` 在 wrapper 自身 import 时就被
  assert，行为与 setup_example.sh step 5 一致。

不 patch 上游 `samples/python/opencv_version.py` 的关键动作：
monkey-patch `cv2.destroyAllWindows` 为 no-op。其余逻辑全部走
`opencv_version.py` 原文件 exec。

不要打 patch：脚本自包含，零非 cv2 第三方依赖。
"""
from __future__ import annotations

import os
import sys


# Source-built cv2 must be importable for the script to do anything
# useful (DNN_BACKEND_CANN == 0 means the pip wheel was loaded instead).
import cv2  # noqa: E402
assert cv2.dnn.DNN_BACKEND_CANN != 0, (
    "cv2 has no CANN backend; source build not in PYTHONPATH?"
)

# Monkey-patch before importing the sample so the unconditional
# cv.destroyAllWindows() at the end of opencv_version.py doesn't
# fail on a CI build with BUILD_LIST not including highgui.
cv2.destroyAllWindows = lambda *args, **kwargs: None  # type: ignore[assignment]


SAMPLES_DIR = os.environ.get(
    "OPENCV_SAMPLES_PYTHON_DIR",
    os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "opencv", "samples", "python")),
)
VERSION_PATH = os.path.join(SAMPLES_DIR, "opencv_version.py")
if not os.path.isfile(VERSION_PATH):
    print(
        f"test_npu_opencv_version: opencv_version.py not found at {VERSION_PATH}",
        file=sys.stderr,
    )
    sys.exit(2)


# Force the script to print the build info even when no flag is passed,
# so we exercise the cv.getBuildInformation() path (which is the most
# useful artefact for CI log inspection). --help is also accepted and
# turns this into a no-op smoke.
argv = sys.argv[1:]
new_argv = ["opencv_version.py", "--build"]
new_argv.extend(argv)
sys.argv = new_argv

# Need samples/python on sys.path so the script's top-level imports
# resolve (it only imports numpy + cv2, but it's a script-style file).
sys.path.insert(0, SAMPLES_DIR)
with open(VERSION_PATH, encoding="utf-8") as fh:
    code = fh.read()
exec(compile(code, VERSION_PATH, "exec"), {"__name__": "__main__", "__file__": VERSION_PATH})

sys.exit(0)