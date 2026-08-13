#!/usr/bin/env python3
"""Scan a target ms-swift tree and write examples_manifest.yaml.

Only examples/ascend/train/qwen3/qwen3_lora_megatron.sh is supported.
Every other scanned example is written as unsupported. That classification is
a task rule, not a community judgment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

SUPPORTED_PATH = 'examples/ascend/train/qwen3/qwen3_lora_megatron.sh'
INCLUDE_EXTENSIONS = ('.sh', '.py', '.yaml')
SUPPORTED_ENTRY = {
    'path': SUPPORTED_PATH,
    'runner': 'linux-aarch64-a2-2',
    'npu_devices': '0,1',
    'overlay': 'overlays/qwen3_lora_megatron.args',
    'timeout_minutes': 180,
}


def scan_examples(target_root: Path) -> list[str]:
    examples_root = target_root / 'examples'
    if not examples_root.is_dir():
        raise SystemExit(f'examples/ not found under {target_root}')
    found: list[str] = []
    for path in sorted(examples_root.rglob('*')):
        if path.is_file() and path.suffix in INCLUDE_EXTENSIONS:
            rel = path.relative_to(target_root).as_posix()
            found.append(rel)
    return found


def render_manifest(paths: list[str]) -> str:
    if SUPPORTED_PATH not in paths:
        raise SystemExit(f'supported example missing from scan: {SUPPORTED_PATH}')
    unsupported = [p for p in paths if p != SUPPORTED_PATH]
    lines = [
        'version: 1',
        'scan:',
        '  root: examples',
        "  include_extensions: ['.sh', '.py', '.yaml']",
        'supported:',
        f"  - path: {SUPPORTED_ENTRY['path']}",
        f"    runner: {SUPPORTED_ENTRY['runner']}",
        f"    npu_devices: '{SUPPORTED_ENTRY['npu_devices']}'",
        f"    overlay: {SUPPORTED_ENTRY['overlay']}",
        f"    timeout_minutes: {SUPPORTED_ENTRY['timeout_minutes']}",
        'unsupported:',
    ]
    for path in unsupported:
        lines.append(f'  - {path}')
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-root', required=True, help='Checkout of the target ms-swift tree')
    parser.add_argument('--output', required=True, help='Path to write examples_manifest.yaml')
    args = parser.parse_args()
    target_root = Path(args.target_root).resolve()
    output = Path(args.output)
    text = render_manifest(scan_examples(target_root))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding='utf-8')
    print(f'wrote {output} ({text.count(chr(10))} lines)')


if __name__ == '__main__':
    main()
