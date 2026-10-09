"""Exercise the project launcher with recording executables, without NPU jobs.

Only the Ascend environment source line is bypassed in a temporary copy. The
real shell argument expansion, entry dispatch, device checks and config
generation execute; recorded commands never claim to run training.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT / "scripts" / "run_example.sh"
HF = "training/tensor_parallel/hf_integration/train.py"
PRMOE = "training/cifar/run_ds_prmoe.sh"
VARIABLE = "training/data_efficiency/variable_batch_size_and_lr/variable_batch_size_and_lr_example.py"
ZENFLOW = "training/DeepSpeed-ZenFlow/benchmark/zf_benchmark.py"
PRUNE = "compression/reasoning_aware_compression/prune.py"


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"),
                     "launcher subprocess tests require a POSIX bash runtime")
class ExampleLauncherTests(unittest.TestCase):
    def launch(self, entry: str, overlays: list[str], *, devices: str | None = None,
               missing_model: bool = False, missing_cache: bool = False
               ) -> tuple[subprocess.CompletedProcess, list[dict], dict | None]:
        with tempfile.TemporaryDirectory(prefix="ds launcher ") as tmp:
            root = Path(tmp)
            examples = root / "examples"
            target = root / "target"
            output = root / "output"
            binaries = root / "bin"
            for path in (examples, target, output, binaries):
                path.mkdir()
            for name in (entry, "training/cifar/cifar10_deepspeed.py"):
                script = examples / name
                script.parent.mkdir(parents=True, exist_ok=True)
                script.touch()
            template = examples / "training/tensor_parallel/hf_integration/configs/ds_config_temp.json"
            template.parent.mkdir(parents=True, exist_ok=True)
            template_text = (
                '{"zero_optimization":{"stage":${zero_stage}},'
                '"tensor_parallel":{"autotp_size":${autotp_size}},'
                '"bf16":{"enabled":"auto"}}\n'
            )
            template.write_text(template_text)
            model = root / "model with spaces"
            if not missing_model:
                model.mkdir()
            fixture = root / "fixture with spaces.jsonl"
            fixture.write_text('{"prompt":"local data"}\n')
            if not missing_cache:
                cache = output / "hf-autotp-work" / "dataset_dict.pkl"
                cache.parent.mkdir()
                cache.write_bytes(b"setup-generated test cache")
            record = root / "commands.jsonl"
            recorder = """
import json, os, sys
from pathlib import Path
command = Path(sys.argv[0]).name
args = sys.argv[1:]
if command == 'python3' and (not args or args[0] != '-c'):
    os.execv(REAL_PYTHON, [REAL_PYTHON, *args])
with open(os.environ['COMMAND_RECORD'], 'a') as handle:
    handle.write(json.dumps({
        'command': command, 'args': args, 'cwd': os.getcwd(),
        'devices': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        'compile_disable': os.environ.get('TORCH_COMPILE_DISABLE'),
    }) + '\\n')
"""
            recorder = f"#!{sys.executable}\nREAL_PYTHON = {sys.executable!r}\n" + recorder
            for binary in ("deepspeed", "python3"):
                path = binaries / binary
                path.write_text(recorder)
                path.chmod(0o755)
            shell = RUN_SCRIPT.read_text(encoding="utf-8")
            source = "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
            self.assertEqual(shell.count(source), 1)
            script = root / "runner.sh"
            script.write_text(shell.replace(source, ": # mock Ascend environment only"))
            env = dict(os.environ)
            env.pop("EXEC", None)
            env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
            env.pop("TORCH_COMPILE_DISABLE", None)
            env.update(
                PATH=f"{binaries}:{env['PATH']}",
                TARGET_ROOT=str(target), EXAMPLES_ROOT=str(examples),
                CI_OUTPUT_DIR=str(output), OVERLAY_ARGS=json.dumps(overlays),
                MODEL_PATH=str(model), FIXTURE_PATH=str(fixture),
                COMMAND_RECORD=str(record), GITHUB_RUN_ID="1234",
            )
            if devices is not None:
                env["ASCEND_RT_VISIBLE_DEVICES"] = devices
            result = subprocess.run(["bash", str(script), entry], env=env,
                                    capture_output=True, text=True, timeout=30)
            commands = [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []
            generated = output / "hf-autotp-config.json"
            config = json.loads(generated.read_text()) if generated.exists() else None
            self.assertEqual(template.read_text(), template_text,
                             "launcher modified the upstream config template")
            return result, commands, config

    def test_prmoe_retains_residual_and_two_expert_layers(self) -> None:
        result, commands, _ = self.launch(PRMOE, ["--epochs 1", "--dtype bf16"], devices="3,6,7")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(commands), 1)
        command = commands[0]
        args = command["args"]
        self.assertEqual(args[args.index("--num_gpus") + 1], "2")
        self.assertEqual(args[args.index("--ep-world-size") + 1], "2")
        self.assertEqual(args[args.index("--num-experts") + 1:args.index("--num-experts") + 3], ["2", "4"])
        self.assertEqual(args[args.index("--mlp-type") + 1], "residual")
        self.assertEqual(command["devices"], "3,6")
        self.assertEqual(command["compile_disable"], "1")

    def test_hf_autotp_generates_config_and_passes_local_paths(self) -> None:
        result, commands, config = self.launch(HF, [
            '--model_name_or_path "${MODEL_PATH}"',
            '--data_path "${FIXTURE_PATH}"', '--max_steps 3',
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(config["zero_optimization"]["stage"], 0)
        self.assertEqual(config["tensor_parallel"]["autotp_size"], 8)
        args = commands[0]["args"]
        self.assertIn("--no_local_rank", args)
        self.assertEqual(args[args.index("--num_gpus") + 1], "8")
        self.assertIn("model with spaces", args[args.index("--model_name_or_path") + 1])
        self.assertIn("fixture with spaces.jsonl", args[args.index("--data_path") + 1])
        self.assertIn("/output/", args[args.index("--deepspeed") + 1])
        self.assertTrue(commands[0]["cwd"].endswith("/output/hf-autotp-work"))
        self.assertIsNone(commands[0]["compile_disable"])

    def test_insufficient_cards_fail_before_launch(self) -> None:
        result, commands, _ = self.launch(HF, [], devices="0,1,2,3")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required=8", result.stderr)
        self.assertEqual(commands, [])

    def test_missing_local_model_fails_before_launch(self) -> None:
        result, commands, _ = self.launch(HF, [
            '--model_name_or_path "${MODEL_PATH}"', '--data_path "${FIXTURE_PATH}"',
        ], missing_model=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing local directory", result.stderr)
        self.assertEqual(commands, [])

    def test_hf_workers_require_setup_generated_cache(self) -> None:
        result, commands, _ = self.launch(HF, [
            '--model_name_or_path "${MODEL_PATH}"', '--data_path "${FIXTURE_PATH}"',
        ], missing_cache=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("shared tokenization cache missing", result.stderr)
        self.assertEqual(commands, [])

    def test_variable_batch_avoids_injecting_unknown_argument(self) -> None:
        result, commands, _ = self.launch(VARIABLE, ["--pipeline-num-stages 0"], devices="4,5")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = commands[0]["args"]
        self.assertIn("--no_local_rank", args)
        self.assertEqual(args[args.index("--num_gpus") + 1], "1")
        self.assertEqual(commands[0]["devices"], "4")
        self.assertIsNone(commands[0]["compile_disable"])

    def test_zenflow_uses_two_ranks_and_unmodified_cli(self) -> None:
        result, commands, _ = self.launch(ZENFLOW, [
            "--iteration 3", "--update_intervals 2", "--pin_memory_opts 0",
            "--topk_ratios 0.1", "--overlap_steps 0",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        args = commands[0]["args"]
        self.assertEqual(args[args.index("--num_gpus") + 1], "2")
        self.assertNotIn("--no_local_rank", args)
        self.assertEqual(args[args.index("--update_intervals") + 1], "2")
        self.assertIsNone(commands[0]["compile_disable"])

    def test_prune_bootstrap_registers_npu_and_preserves_arguments(self) -> None:
        result, commands, _ = self.launch(PRUNE, [
            '--model "${MODEL_PATH}"', '--dataset "${FIXTURE_PATH}"',
            "--device npu", "--pruning-method wanda",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        command = commands[0]
        self.assertEqual(command["command"], "python3")
        self.assertIn("import torch_npu", command["args"][1])
        self.assertIn("runpy.run_path", command["args"][1])
        self.assertEqual(command["args"][command["args"].index("--device") + 1], "npu")
        self.assertIsNone(command["compile_disable"])


if __name__ == "__main__":
    unittest.main()
