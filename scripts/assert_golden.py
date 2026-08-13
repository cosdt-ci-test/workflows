#!/usr/bin/env python3
"""Assert or calibrate golden metrics from Megatron logging.jsonl."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

RTOL_LIMIT = 1e-2
EPS = 1e-12
ABS_TOL = 1e-12


def is_finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def parse_iteration(value) -> int:
    if isinstance(value, int):
        return value
    text = str(value)
    if '/' in text:
        text = text.split('/', 1)[0]
    return int(text)


def load_loss_steps(logging_path: Path) -> list[dict]:
    steps: list[dict] = []
    with logging_path.open(encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if 'loss' not in record:
                continue
            steps.append({
                'iteration': parse_iteration(record.get('iteration', len(steps) + 1)),
                'loss': float(record['loss']),
                'grad_norm': record.get('grad_norm'),
            })
    return steps


def load_golden(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), EPS)


def exceeds_rtol(actual: float, expected: float, rtol: float) -> bool:
    if abs(actual - expected) <= ABS_TOL:
        return False
    return relative_error(actual, expected) > rtol


def format_steps_table(runs: list[list[dict]], golden: dict | None = None) -> str:
    lines = ['| iteration | loss | grad_norm | rel_err |', '| --- | --- | --- | --- |']
    steps = runs[0] if runs else []
    expected = {item['iteration']: item for item in (golden or {}).get('steps', [])}
    for step in steps:
        expected_step = expected.get(step['iteration'])
        rel = ''
        if expected_step is not None:
            rel = f"{relative_error(step['loss'], float(expected_step['loss'])):.6g}"
        gn = step.get('grad_norm')
        gn_text = '' if gn is None else f'{float(gn):.8g}'
        lines.append(f"| {step['iteration']} | {step['loss']:.8g} | {gn_text} | {rel} |")
    return '\n'.join(lines)


def append_summary(text: str) -> None:
    sys.stdout.write(text if text.endswith('\n') else text + '\n')
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if not summary:
        return
    with open(summary, 'a', encoding='utf-8') as handle:
        handle.write(text if text.endswith('\n') else text + '\n')


def assert_structure(steps: list[dict], expected_iterations: int, source: str) -> None:
    if len(steps) != expected_iterations:
        raise AssertionError(f'{source}: expected {expected_iterations} loss rows, got {len(steps)}')
    for step in steps:
        if not is_finite(step['loss']):
            raise AssertionError(f'{source}: non-finite loss at iteration {step["iteration"]}: {step["loss"]}')
        if not is_finite(step.get('grad_norm')):
            raise AssertionError(
                f'{source}: non-finite grad_norm at iteration {step["iteration"]}: {step.get("grad_norm")}')


def assert_against_golden(steps: list[dict], golden: dict) -> list[str]:
    failures: list[str] = []
    expected_steps = golden.get('steps') or []
    if len(expected_steps) != len(steps):
        failures.append(f'golden steps ({len(expected_steps)}) != logging steps ({len(steps)})')
        return failures
    rtol = float(golden['rtol'])
    for actual, expected in zip(steps, expected_steps):
        err = relative_error(actual['loss'], float(expected['loss']))
        if exceeds_rtol(actual['loss'], float(expected['loss']), rtol):
            failures.append(
                f'iteration {actual["iteration"]} loss rel_err {err:.6g} > rtol {rtol:.6g} '
                f'(actual={actual["loss"]}, expected={expected["loss"]})')
    return failures


def calibrate(runs: list[list[dict]], golden: dict) -> dict:
    if len(runs) != 3:
        raise AssertionError(f'calibration requires 3 logging.jsonl files, got {len(runs)}')
    expected_iterations = int(golden['expected_iterations'])
    for index, steps in enumerate(runs, start=1):
        assert_structure(steps, expected_iterations, f'run{index}')
    dispersions: list[float] = []
    mean_steps: list[dict] = []
    for offset in range(expected_iterations):
        losses = [run[offset]['loss'] for run in runs]
        grad_norms = [float(run[offset]['grad_norm']) for run in runs]
        mean_loss = round(sum(losses) / len(losses), 8)
        mean_gn = round(sum(grad_norms) / len(grad_norms), 8)
        dispersion = (max(losses) - min(losses)) / max(abs(mean_loss), EPS)
        dispersions.append(dispersion)
        mean_steps.append({
            'iteration': runs[0][offset]['iteration'],
            'loss': mean_loss,
            'grad_norm': mean_gn,
        })
    max_dispersion = max(dispersions)
    rtol = 3.0 * max_dispersion
    append_summary(
        '\n'.join([
            '## golden calibration',
            f'- runs: {len(runs)}',
            f'- max per-step loss relative range: {max_dispersion:.8g}',
            f'- rtol = 3 * max range: {rtol:.8g}',
            f'- rtol limit: {RTOL_LIMIT}',
            '',
        ]))
    if rtol > RTOL_LIMIT:
        raise AssertionError(
            f'calibration rtol {rtol:.8g} > {RTOL_LIMIT}; stopping without changing attention_backend')
    golden = dict(golden)
    golden['calibrated'] = True
    golden['rtol'] = rtol
    golden['steps'] = mean_steps
    return golden


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logging', nargs='+', required=True, help='One or more logging.jsonl paths')
    parser.add_argument('--golden', required=True)
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--image', default='', help='Locked container image written during calibration')
    args = parser.parse_args()
    logging_paths = [Path(p) for p in args.logging]
    golden_path = Path(args.golden)
    golden = load_golden(golden_path)
    runs = [load_loss_steps(path) for path in logging_paths]
    expected_iterations = int(golden['expected_iterations'])

    conclusion = 'pass'
    detail = ''
    try:
        if args.calibrate:
            updated = calibrate(runs, golden)
            if args.image:
                updated['image'] = args.image
            golden_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            golden = updated
            detail = f'wrote calibrated golden rtol={golden["rtol"]:.8g}'
        else:
            if len(runs) != 1:
                raise AssertionError('assert mode expects exactly one logging.jsonl')
            steps = runs[0]
            assert_structure(steps, expected_iterations, str(logging_paths[0]))
            if golden.get('calibrated'):
                failures = assert_against_golden(steps, golden)
                if failures:
                    raise AssertionError('\n'.join(failures))
            detail = 'uncalibrated structural checks only' if not golden.get('calibrated') else 'within rtol'
    except AssertionError as exc:
        conclusion = 'fail'
        detail = str(exc)
        append_summary(f'## golden assert\n\n- conclusion: **{conclusion}**\n- detail: `{detail}`\n\n')
        append_summary(format_steps_table(runs, golden) + '\n')
        raise SystemExit(1) from exc

    append_summary(
        '\n'.join([
            '## golden assert',
            f'- conclusion: **{conclusion}**',
            f'- calibrated: {golden.get("calibrated")}',
            f'- rtol: {golden.get("rtol")}',
            f'- detail: {detail}',
            '',
            format_steps_table(runs, golden),
            '',
        ]))


if __name__ == '__main__':
    main()
