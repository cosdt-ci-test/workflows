"""Install declared non-CUDA dependencies without replacing the NPU stack."""
import pathlib
import subprocess
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version

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
common_pins = "transformers==4.57.6\ntrl==0.22.2\npeft==0.18.0\ndatasets==4.3.0\n"
if sys.argv[1:] == ["--vllm-image"]:
    # Keep the matched vLLM image's binary stack, including its NumPy ABI.
    stack = {name: version(name) for name in (
        "torch", "torch-npu", "triton-ascend", "numpy", "vllm", "vllm-ascend",
    )}
    for name in ("triton", "torchvision", "torchaudio"):
        try:
            stack[name] = version(name)
        except PackageNotFoundError:
            pass
    # The Ascend image builds vLLM with VLLM_TARGET_DEVICE=empty.
    # Accept that build but keep its full version in the pip constraints.
    if stack["vllm"] not in {"0.23.0", "0.23.0+empty"} or stack["vllm-ascend"] != "0.23.0":
        raise SystemExit(f"Unexpected vLLM-Ascend image versions: {stack}")
    stack_pins = "".join(f"{name}=={installed}\n" for name, installed in stack.items())
    stack_requirements = ["vllm", "vllm-ascend"]
elif not sys.argv[1:]:
    # Triton-Ascend 3.2.2 requires NumPy 1.26.4. Arrow 21 satisfies
    # datasets 4.3.0's >=21 floor while retaining NumPy 1.x support.
    stack_pins = "torch==2.9.0\ntorch_npu==2.9.0.post2\nnumpy==1.26.4\npyarrow==21.0.0\n"
    stack_requirements = []
else:
    raise SystemExit("Expected no arguments or --vllm-image")
constraints.write_text(common_pins + stack_pins)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-c", str(constraints),
     *sorted(set(selected)), *stack_requirements, "modelscope==1.37.0"],
    check=True,
)
# The non-quantized loader still imports Linear4bit for type checks.
# Its base dependencies are already installed; preserve the NPU torch stack.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-deps", "bitsandbytes==0.50.2"],
    check=True,
)
subprocess.run(
    [sys.executable, "-c",
     "from bitsandbytes.nn import Linear4bit; "
     "from peft.tuners.lora import Linear4bit as PeftLinear4bit; "
     "from importlib.metadata import version; "
     "print('Quantization type imports OK:', version('bitsandbytes'), version('peft'))"],
    check=True,
)
# A successful pip exit does not prove binary imports work together.
# The training image needs the exact NumPy/Arrow pair; the vLLM image keeps
# its own NumPy and still has to import Arrow and construct a Dataset.
data_version_check = (
    "assert numpy.__version__ == '1.26.4'; assert pyarrow.__version__ == '21.0.0'; "
    if not stack_requirements else ""
)
subprocess.run(
    [sys.executable, "-c",
     "import numpy, pyarrow, datasets; " + data_version_check +
     "assert datasets.Dataset.from_dict({'text': ['dependency check']})[0]['text'] == 'dependency check'; "
     "print('Data dependencies OK:', numpy.__version__, pyarrow.__version__, datasets.__version__)"],
    check=True,
)
