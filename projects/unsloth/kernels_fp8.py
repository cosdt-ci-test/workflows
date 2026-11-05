"""Numerical checks of pinned Unsloth kernels and explicit FP8 device gates."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import traceback

CASES = ("rmsnorm", "rope", "swiglu", "cross-entropy", "lora-qkv",
         "fp8-row", "fp8-block", "fp8-training")
OUTPUT = Path("output")
DETAILS = OUTPUT / "kernels_fp8"


def compare(torch, actual, reference, dtype, label, evidence):
    actual, reference = actual.detach(), reference.detach()
    assert actual.device.type == reference.device.type == "npu", label
    assert actual.shape == reference.shape, label
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all(), label
    atol, rtol = (3e-2, 3e-2) if dtype == torch.bfloat16 else (2e-4, 2e-4)
    entry = dict(label=label, max_abs_error=float((actual.float() - reference.float()).abs().max()),
                 atol=atol, rtol=rtol)
    evidence.append(entry)
    torch.testing.assert_close(actual.float(), reference.float(), atol=atol, rtol=rtol)


def worker(case):
    outcome = dict(case=case, status="running", comparisons=[])
    path = DETAILS / f"{case}.json"
    try:
        import torch
        import torch_npu  # noqa: F401 - registers NPU
        import unsloth  # noqa: F401 - imports device-aware kernels before HF/PEFT
        from unsloth.device_type import DEVICE_TYPE

        assert DEVICE_TYPE == "npu" and torch.npu.is_available()
        torch.npu.set_device(0)
        outcome["device"] = torch.npu.get_device_name(0)
        outcome["versions"] = {name: version(name) for name in (
            "torch", "torch-npu", "triton-ascend", "unsloth", "unsloth-zoo",
        )}
        if case.startswith("fp8-"):
            if case == "fp8-training":
                from unsloth.models.float8_training import _check_hardware
                def call():
                    return _check_hardware(torch.device("npu:0"), "tensorwise")
                expected_type = RuntimeError
                expected_text = "float8_training needs the model on an NVIDIA GPU"
                outcome["entry"] = "float8_training._check_hardware"
            else:
                from unsloth.models.loader_utils import _get_fp8_mode_and_check_settings
                mode = True if case == "fp8-row" else "block"
                def call():
                    return _get_fp8_mode_and_check_settings(
                        mode, fast_inference=False, full_finetuning=False,
                        load_in_4bit=False, load_in_8bit=False, load_in_16bit=False,
                    )
                expected_type = ValueError
                expected_text = "load_in_fp8` requires H100 GPUs or after"
                outcome["entry"] = "loader_utils._get_fp8_mode_and_check_settings"
            try:
                call()
            except expected_type as exc:
                if expected_text not in str(exc):
                    raise
                outcome.update(status="unsupported", reason=str(exc),
                               scope="Upstream device gate only; no FP8 quantization or training executed")
            else:
                raise AssertionError("FP8 gate accepted NPU: a full FP8 test is now required")
            return

        import torch.nn.functional as F
        rng = torch.Generator(device="cpu").manual_seed(3407)
        for dtype in (torch.float32, torch.bfloat16):
            outcome["dtype"] = str(dtype)
            evidence = outcome["comparisons"]

            def sample(shape, train=True):
                return (torch.randn(shape, generator=rng) * 0.2).to(
                    device="npu", dtype=dtype
                ).requires_grad_(train)

            def parity(outputs, refs, inputs, ref_inputs):
                outputs = (outputs,) if torch.is_tensor(outputs) else outputs
                refs = (refs,) if torch.is_tensor(refs) else refs
                seeds = [sample(output.shape, False) for output in outputs]
                for i, (output, ref) in enumerate(zip(outputs, refs)):
                    compare(torch, output, ref, dtype, f"{dtype}/forward/{i}", evidence)
                actual_grads = torch.autograd.grad(outputs, inputs, grad_outputs=seeds)
                ref_grads = torch.autograd.grad(refs, ref_inputs, grad_outputs=seeds)
                for i, (gradient, ref) in enumerate(zip(actual_grads, ref_grads)):
                    compare(torch, gradient, ref, dtype, f"{dtype}/gradient/{i}", evidence)

            if case == "rmsnorm":
                from types import SimpleNamespace
                from unsloth.kernels import fast_rms_layernorm
                x, w = sample((2, 7, 128)), sample((128,))
                xr, wr = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
                out = fast_rms_layernorm(SimpleNamespace(weight=w, eps=1e-6), x)
                normalized = xr.float() * torch.rsqrt(xr.float().square().mean(-1, keepdim=True) + 1e-6)
                ref = (normalized * wr.float()).to(dtype)
                parity(out, ref, (x, w), (xr, wr))
            elif case == "rope":
                from unsloth.kernels import fast_rope_embedding
                q, k = sample((2, 4, 7, 32)), sample((2, 2, 7, 32))
                qr, kr = q.detach().clone().requires_grad_(), k.detach().clone().requires_grad_()
                angles = sample((7, 16), False).float()
                cos = torch.cat((angles.cos(), angles.cos()), -1).to(dtype)
                sin = torch.cat((angles.sin(), angles.sin()), -1).to(dtype)

                def rotate(x):
                    half = x.shape[-1] // 2
                    return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin

                parity(fast_rope_embedding(q, k, cos, sin), (rotate(qr), rotate(kr)),
                       (q, k), (qr, kr))
            elif case == "swiglu":
                from unsloth.kernels import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel
                e, g = sample((2, 7, 128)), sample((2, 7, 128))
                ref = F.silu(e.float()).to(dtype) * g
                out = swiglu_fg_kernel(e.detach(), g.detach())
                compare(torch, out, ref, dtype, f"{dtype}/forward", evidence)
                dy = sample(e.shape, False)
                de, dg = torch.autograd.grad(ref, (e, g), grad_outputs=dy)
                # The kernel overwrites its buffers and returns (forward, d_g, d_e).
                h, kernel_dg, kernel_de = swiglu_DWf_DW_dfg_kernel(
                    dy.reshape(-1, 128).clone(), e.detach().reshape(-1, 128).clone(),
                    g.detach().reshape(-1, 128).clone(),
                )
                compare(torch, h.view_as(ref), ref, dtype, f"{dtype}/backward_recomputed_output", evidence)
                compare(torch, kernel_de.view_as(de), de, dtype, f"{dtype}/gradient/e", evidence)
                compare(torch, kernel_dg.view_as(dg), dg, dtype, f"{dtype}/gradient/g", evidence)
            elif case == "cross-entropy":
                from unsloth.kernels import fast_cross_entropy_loss
                for vocab in (257, 65537):
                    logits = sample((1, 3, vocab))
                    reference = logits.detach().clone().requires_grad_()
                    labels = torch.tensor([[1, -100, 17]], device="npu")
                    loss = fast_cross_entropy_loss(logits, labels)
                    ref = F.cross_entropy(reference.float().reshape(-1, vocab), labels.reshape(-1))
                    compare(torch, loss, ref, dtype, f"{dtype}/{vocab}/loss", evidence)
                    gradients = torch.autograd.grad(loss, (logits,))[0]
                    ref_grad = torch.autograd.grad(ref, (reference,))[0]
                    compare(torch, gradients, ref_grad, dtype, f"{dtype}/{vocab}/gradient", evidence)
                    assert torch.count_nonzero(gradients[:, 1]).item() == 0
            elif case == "lora-qkv":
                from unsloth.kernels.fast_lora import LoRA_QKV
                x = sample((2, 7, 64))
                xr = x.detach().clone().requires_grad_()
                args, inputs, ref_inputs, refs = [x], [x], [xr], []
                for width in (64, 32, 32):
                    w, a, b = sample((width, 64), False), sample((4, 64)), sample((width, 4))
                    ar, br = a.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
                    args.extend((w, None, a, b, 1.0))
                    inputs.extend((a, b))
                    ref_inputs.extend((ar, br))
                    refs.append(F.linear(xr, w) + F.linear(F.linear(xr, ar), br))
                outputs = LoRA_QKV.apply(*args, False, None, None, None, None)
                assert "LoRA_QKV" in type(outputs[0].grad_fn).__name__
                parity(outputs, tuple(refs), tuple(inputs), tuple(ref_inputs))
            torch.npu.synchronize()
        outcome["status"] = "passed"
    except Exception as exc:
        outcome.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        path.write_text(json.dumps(outcome, indent=2))
        print(json.dumps(outcome), flush=True)


def summarize(checks):
    if not checks or any(check["status"] not in {"passed", "unsupported", "failed"} for check in checks):
        return "failed"
    if any(check["status"] == "failed" for check in checks):
        return "failed"
    if any(check["status"] == "unsupported" for check in checks):
        return "completed_with_gaps"
    return "passed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    DETAILS.mkdir(parents=True, exist_ok=True)
    if args.case:
        worker(args.case)
        state = json.loads((DETAILS / f"{args.case}.json").read_text())["status"]
        return 1 if state == "failed" else 0
    result = {
        "status": "running", "stage": "kernel_fp8_audit", "completed": [],
        "scenario": "kernels-fp8", "checks": [],
        "scope": "Five common kernel entry points and three FP8 device guards; no performance claim",
    }
    result_path = OUTPUT / "result.json"
    for case in CASES:
        result["stage"] = case
        result_path.write_text(json.dumps(result, indent=2))
        detail_path = DETAILS / f"{case}.json"
        detail_path.unlink(missing_ok=True)
        try:
            process = subprocess.run([sys.executable, __file__, "--case", case],
                                     capture_output=True, text=True, timeout=240)
            log = process.stdout + process.stderr
            if detail_path.is_file():
                check = json.loads(detail_path.read_text())
            else:
                check = dict(case=case, status="failed", error="Worker exited without a result")
            if process.returncode != 0:
                check.update(status="failed", worker_exit_code=process.returncode)
            if check["status"] not in {"passed", "unsupported", "failed"}:
                check.update(status="failed", error="Worker did not finish its check")
        except subprocess.TimeoutExpired as exc:
            def as_text(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
            log = as_text(exc.stdout) + as_text(exc.stderr)
            check = dict(case=case, status="failed", error="Worker timed out after 240 seconds")
        (DETAILS / f"{case}.log").write_text(log)
        print(f"=== {case} ===\n{log}", flush=True)
        result["checks"].append(check)
        if check["status"] == "passed":
            result["completed"].append(case)
        result_path.write_text(json.dumps(result, indent=2))
    result.update(status=summarize(result["checks"]), stage="complete",
                  counts={state: sum(check["status"] == state for check in result["checks"])
                          for state in ("passed", "unsupported", "failed")})
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
