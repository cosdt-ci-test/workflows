"""Install declared non-CUDA dependencies without replacing the NPU stack."""
import pathlib
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement

root = pathlib.Path("upstream")
unsloth = tomllib.loads((root / "unsloth/pyproject.toml").read_text())
zoo = tomllib.loads((root / "unsloth-zoo/pyproject.toml").read_text())
requirements = (
    unsloth["project"]["dependencies"]
    + unsloth["project"]["optional-dependencies"]["huggingfacenotorch"]
    + zoo["project"]["dependencies"]
)
# CUDA Triton and quantization extras are not used by this BF16 example.
# triton-ascend supplies the triton import; the two source trees install below.
excluded = {"torch", "triton", "torchao", "cut-cross-entropy", "unsloth-zoo"}
selected = []
for raw in requirements:
    req = Requirement(raw)
    name = req.name.lower().replace("_", "-")
    if name not in excluded and (req.marker is None or req.marker.evaluate()):
        selected.append(raw)
constraints = pathlib.Path("output/constraints.txt")
constraints.write_text(
    "torch==2.9.0\ntorch_npu==2.9.0.post2\ntransformers==4.57.6\n"
    "trl==0.22.2\npeft==0.18.0\ndatasets==4.3.0\n"
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-c", str(constraints),
     *sorted(set(selected)), "modelscope==1.37.0"],
    check=True,
)
