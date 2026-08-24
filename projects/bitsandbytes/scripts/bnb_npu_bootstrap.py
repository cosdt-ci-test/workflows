"""Load torch_npu into the pytest process.

Upstream tests/ and tests/conftest.py do not import torch_npu, so torch.npu
does not exist unless this plugin runs first. Loading via pytest -p avoids
patching the target tree.
"""

import torch_npu  # noqa: F401
