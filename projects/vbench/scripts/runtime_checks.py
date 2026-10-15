"""Test-side checkpoint and device probes; leave VBench scoring unchanged."""

from contextlib import contextmanager

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
