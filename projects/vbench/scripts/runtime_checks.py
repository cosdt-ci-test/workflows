"""Test-side checkpoint and device probes; leave VBench scoring unchanged."""

from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch


def check_checkpoint(model, state):
    expected = model.state_dict()
    # The official UMT checkpoint omits the sinusoidal positions generated
    # for the requested frame count. All trained weights must be present.
    missing = expected.keys() - state.keys() - {"pos_embed"}
    unexpected = state.keys() - expected.keys()
    mismatched = [name for name in expected.keys() & state.keys()
                  if expected[name].shape != state[name].shape]
    if missing or unexpected or mismatched:
        raise RuntimeError(f"UMT checkpoint mismatch: missing={sorted(missing)}, "
                           f"unexpected={sorted(unexpected)}, shapes={mismatched}")


def check_forward(model, inputs, output, expected_device):
    assert all(p.device == expected_device for p in model.parameters()), "model device mismatch"
    assert inputs[0].device == expected_device, "input device mismatch"
    assert output.device == expected_device, "output device mismatch"
    assert torch.isfinite(output).all(), "UMT logits must be finite"


@contextmanager
def watch_human_action(module, checkpoint, device="npu:0"):
    """Observe real UMT forwards through the API, restoring all hooks on exit."""
    expected_device = torch.device(device)
    observed = []
    handles = []
    original = module.create_model

    def record(model, inputs, output):
        check_forward(model, inputs, output, expected_device)
        observed.append(str(output.device))

    def create_model(*args, **kwargs):
        model = original(*args, **kwargs)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        check_checkpoint(model, state)
        handles.append(model.register_forward_hook(record))
        return model

    module.create_model = create_model
    try:
        yield observed
    finally:
        module.create_model = original
        for handle in handles:
            handle.remove()


@contextmanager
def check_api(module, checkpoint, device="npu:0", output_root="output/vbench"):
    """Keep acceptance assertions around, rather than inside, the user's example."""
    with watch_human_action(module, checkpoint, device) as observed:
        yield
        if torch.device(device).type == "npu":
            torch.npu.synchronize()
    assert observed == [str(torch.device(device))] * 5, "Expected five forwards on the target device"
    for entry in ("cli", "api"):
        files = list((Path(output_root) / entry).glob("*_eval_results.json"))
        assert len(files) == 1, (entry, files)
        score, videos = json.loads(files[0].read_text())["human_action"]
        assert math.isfinite(score) and 0 <= score <= 1, "Invalid human_action score"
        assert len(videos) == 5
        assert len({video["video_path"] for video in videos}) == 5
        assert all(set(video) == {"video_path", "video_results", "cor_num_per_video"}
                   for video in videos)
        assert all(isinstance(video["video_results"], bool) for video in videos)
