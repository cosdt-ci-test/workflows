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
    patches = Path(__file__).resolve().parents[1] / "patches"
    selected = [patches / "0001-single-npu-human-action.patch"]
    for name in ("0002-cli-v0.1.4.patch", "0002-cli-master.patch"):
        candidate = patches / name
        result = subprocess.run(
            ["git", "apply", "--check", str(candidate)], cwd=target,
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            selected.append(candidate)
            break
    else:
        raise RuntimeError("Upstream CLI changed: review the VBench Ascend patch")
    if "def check_torch_version(" in (target / "setup.py").read_text():
        selected.append(patches / "0003-npu-install-check.patch")
    for path in selected:
        subprocess.run(["git", "apply", "--check", str(path)], cwd=target, check=True)
    for path in selected:
        subprocess.run(["git", "apply", str(path)], cwd=target, check=True)
        print(f"Applied: {path.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    apply_patches(parser.parse_args().target)
