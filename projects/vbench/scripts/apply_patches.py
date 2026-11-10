"""Apply the reviewed Ascend patch to an upstream VBench checkout.

All hunks are checked before applying. Unknown upstream layouts fail instead
of silently rewriting or replacing upstream implementations.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def apply_patches(target: Path) -> None:
    target = target.resolve()
    patch = Path(__file__).resolve().parents[1] / "patches/0001-single-npu-human-action.patch"
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=target, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=target, check=True)
    print(f"Applied: {patch.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    apply_patches(parser.parse_args().target)
