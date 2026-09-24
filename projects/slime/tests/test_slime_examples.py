from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "examples_manifest.yaml"

EXPECTED_SUPPORTED = {
    "examples/fully_async/run-qwen2.5-0.5B-fully_async.sh": {
        "profile": "slime_fully_async",
        "runner": "linux-aarch64-a2-4",
        "image": (
            "swr.cn-south-1.myhuaweicloud.com/ascendhub/"
            "cann:9.1.0-910b-ubuntu22.04-py3.12"
        ),
        "timeout_minutes": 300,
    },
    "examples/on_policy_distillation/run-qwen3-8B-opd.sh": {
        "profile": "slime_opd",
        "runner": "linux-aarch64-a2-8",
        "image": (
            "swr.cn-south-1.myhuaweicloud.com/ascendhub/"
            "cann:9.1.0-910b-ubuntu22.04-py3.12"
        ),
        "timeout_minutes": 300,
    },
    "examples/retool/retool_qwen3_4b_rl.sh": {
        "profile": "slime_retool",
        "runner": "linux-aarch64-a2-4",
        "image": (
            "swr.cn-south-1.myhuaweicloud.com/ascendhub/"
            "cann:9.1.0-910b-ubuntu22.04-py3.12"
        ),
        "timeout_minutes": 300,
    },
}

# Entries whose execution recipe needs a minimum visible-device count.
ENTRY_DEVICE_REQUIREMENTS = {
    "examples/fully_async/run-qwen2.5-0.5B-fully_async.sh": 4,
    "examples/on_policy_distillation/run-qwen3-8B-opd.sh": 8,
    "examples/retool/retool_qwen3_4b_rl.sh": 4,
}

# Engine-call metadata that the engine cannot pass through; each
# supported entry must have a per-entry mapping in run_example.sh.
ENTRY_MODEL_TYPES = {
    "examples/fully_async/run-qwen2.5-0.5B-fully_async.sh": "qwen2.5-0.5B",
    "examples/on_policy_distillation/run-qwen3-8B-opd.sh": "qwen2.5-0.5B",
    "examples/retool/retool_qwen3_4b_rl.sh": "qwen3-4B-Instruct-2507",
}
ENTRY_TRAIN_SCRIPTS = {
    "examples/fully_async/run-qwen2.5-0.5B-fully_async.sh": "train_async.py",
    "examples/on_policy_distillation/run-qwen3-8B-opd.sh": "train.py",
    "examples/retool/retool_qwen3_4b_rl.sh": "train.py",
}

# Pin table copied from the fork's Dockerfile ARGs and quick_install.sh.
FORK_PINS = {
    "sglang_ref": "v0.5.13",
    "megatron_commit_prefix": "1dcf0dafa",
    "mbridge_commit_prefix": "89eb1088",
    "megatron_adaptor_commit": "f707a3b6",
    "transformer_engine_npu_commit": "47d60449",
    "torch": "2.10.0",
    "torch_npu": "2.10.0",
    "torchvision": "0.25.0",
    "triton_ascend": "3.2.1",
    "python_tag": "cp312",
    "cann_tag": "cann9.1.0",
    "board_tag": "910b",
}


def load_manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def unsupported_paths(manifest: dict) -> list[str]:
    return [entry for entry in manifest.get("unsupported") or []]


def test_supported_entries_match_plan() -> None:
    manifest = load_manifest()
    supported = {entry["path"]: entry for entry in manifest["supported"]}
    assert set(supported) == set(EXPECTED_SUPPORTED), (
        f"supported set drifted: {sorted(supported)}")
    for path, expectations in EXPECTED_SUPPORTED.items():
        entry = supported[path]
        for field, expected in expectations.items():
            assert entry[field] == expected, (
                f"{path}: {field}={entry[field]!r} != {expected!r}")


def test_supported_and_unsupported_do_not_overlap() -> None:
    manifest = load_manifest()
    supported_paths = {entry["path"] for entry in manifest["supported"]}
    unsupported = unsupported_paths(manifest)
    overlap = supported_paths & set(unsupported)
    assert not overlap, f"entries listed in both sections: {overlap}"
    assert len(unsupported) == len(set(unsupported)), (
        "duplicate unsupported entries")


def test_unsupported_entries_carry_inline_comments() -> None:
    manifest = load_manifest()
    raw_lines = MANIFEST_PATH.read_text(encoding="utf-8").splitlines()

    def commented_mentions(path: str) -> bool:
        """A comment line references the path (full or basename form)."""
        for line in raw_lines:
            stripped = line.lstrip()
            if not stripped.startswith("#"):
                continue
            if path in stripped:
                return True
            # Group comments may reference only the basename (library
            # notes mention e.g. base_env.py without its directory).
            if Path(path).name in stripped:
                return True
        return False

    for path in unsupported_paths(manifest):
        assert commented_mentions(path), f"no ledger comment mentions {path}"


def test_scan_config_targets_fork_examples_tree() -> None:
    manifest = load_manifest()
    scan = manifest["scan"]
    assert scan["root"] == "examples"
    assert ".sh" in scan["include_extensions"]
    assert ".py" in scan["include_extensions"]
    # scan.exclude was retired 2026-09-20; the package marker is now a
    # ledger entry in `unsupported`.
    assert "exclude" not in scan
    unsupported = unsupported_paths(manifest)
    assert "examples/__init__.py" in unsupported
    assert "examples/tau-bench/token_delta.py" in unsupported


def test_fixture_rows_match_dapo_schema() -> None:
    fixture = PROJECT_ROOT / "fixtures" / "ci_dapo_16.jsonl"
    rows = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 16
    for row in rows:
        assert set(row) == {"prompt", "label"}
        assert isinstance(row["prompt"], list) and row["prompt"]
        assert all(set(message) == {"content", "role"} for message in row["prompt"])
        assert isinstance(row["label"], str)


def test_retool_fixture_requests_tool_calls() -> None:
    fixture = PROJECT_ROOT / "fixtures" / "ci_retool_math_8.jsonl"
    rows = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 8
    for row in rows:
        assert set(row) == {"prompt", "label"}
        assert len(row["prompt"]) == 1
        prompt = row["prompt"][0]
        assert prompt["role"] == "user"
        assert "code_interpreter" in prompt["content"]
        assert "\\boxed{" in prompt["content"]
        assert row["label"].isdigit()


def test_setup_pins_match_fork_recipe() -> None:
    setup = (PROJECT_ROOT / "scripts" / "setup_example.sh").read_text(encoding="utf-8")
    for needle in (
        f"{FORK_PINS['megatron_commit_prefix']}",
        f"{FORK_PINS['mbridge_commit_prefix']}",
        FORK_PINS["megatron_adaptor_commit"],
        FORK_PINS["transformer_engine_npu_commit"],
        f"torch=={FORK_PINS['torch']}",
        f"torch_npu=={FORK_PINS['torch_npu']}",
        f"torchvision=={FORK_PINS['torchvision']}",
        f"triton-ascend=={FORK_PINS['triton_ascend']}",
        f"{FORK_PINS['sglang_ref']}",
        FORK_PINS["cann_tag"],
        FORK_PINS["python_tag"],
        FORK_PINS["board_tag"],
    ):
        assert needle in setup, f"setup_example.sh missing fork pin: {needle}"
    assert '"TransformerEngineNPU:transformer_engine_npu"' in setup
    assert '"Megatron-LM:megatron"' in setup
    assert '"Megatron-Bridge:megatron-bridge"' in setup
    assert 'required NPU patch directory missing:' in setup
    assert 'setup_slime_opd()' in setup
    assert 'check_npu_devices 8' in setup
    assert 'prepare_qwen25_assets' in setup
    assert 'setup_slime_retool()' in setup
    assert 'Qwen/Qwen3-4B-Instruct-2507' in setup
    assert 'source scripts/models/qwen3-4B-Instruct-2507.sh' in setup
    assert 'jinja2 psutil' in setup
    assert 'SLIME_FIXTURE_JSONL=$FIXTURE_DIR/ci_retool_math_8.jsonl' in setup


def test_run_example_exports_npu_contract() -> None:
    run_script = (PROJECT_ROOT / "scripts" / "run_example.sh").read_text(encoding="utf-8")
    for needle in (
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1",
        "CUDA_DEVICE_MAX_CONNECTIONS=1",
        "HCCL_HOST_SOCKET_PORT_RANGE",
        "HCCL_NPU_SOCKET_PORT_RANGE",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "WANDB_MODE=offline",
        "require_visible_devices 4",
        "require_visible_devices 8",
        "SLIME_OPD_TEACHER_DEVICE",
        "before_ray_job_submit",
    ):
        assert needle in run_script, f"run_example.sh missing: {needle}"




# Entries whose CI train recipe is carried by manifest overlay_args.
# Sources are the fork's NPU tests or the entry's upstream launcher.
RECIPE_OVERLAY_REQUIRED = {
    "examples/fully_async/run-qwen2.5-0.5B-fully_async.sh": [
        "--hf-checkpoint ${SLIME_MODEL_PATH}",
        "--ref-load ${SLIME_TORCH_DIST_PATH}",
        "--prompt-data ${SLIME_FIXTURE_JSONL}",
        "--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async",
        "--num-rollout 2",
        "--rollout-max-response-len 1024",
        "--sglang-device npu",
        "--ci-test",
    ],
    "examples/on_policy_distillation/run-qwen3-8B-opd.sh": [
        "--hf-checkpoint ${SLIME_MODEL_PATH}",
        "--ref-load ${SLIME_TORCH_DIST_PATH}",
        "--prompt-data ${SLIME_FIXTURE_JSONL}",
        "--custom-rm-path slime.rollout.on_policy_distillation.reward_func",
        "--custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards",
        "--rm-url ${SLIME_TEACHER_URL}",
        "--use-opd",
        "--opd-type sglang",
        "--num-rollout 2",
        "--actor-num-gpus-per-node 4",
        "--rollout-num-gpus 3",
        "--sglang-device npu",
        "--ci-test",
    ],
    "examples/retool/retool_qwen3_4b_rl.sh": [
        "--hf-checkpoint ${SLIME_MODEL_PATH}",
        "--ref-load ${SLIME_TORCH_DIST_PATH}",
        "--prompt-data ${SLIME_FIXTURE_JSONL}",
        "--custom-generate-function-path generate_with_retool.generate",
        "--custom-rm-path generate_with_retool.reward_func",
        "--reward-key score",
        "--colocate",
        "--actor-num-gpus-per-node 4",
        "--num-gpus-per-node 4",
        "--tensor-model-parallel-size 2",
        "--rollout-num-gpus-per-engine 2",
        "--num-rollout 2",
        "--sglang-device npu",
        "--ci-test",
    ],
}


def test_manifest_overlay_carries_full_recipe() -> None:
    manifest = load_manifest()
    for entry in manifest["supported"]:
        required = RECIPE_OVERLAY_REQUIRED[entry["path"]]
        overlay = entry.get("overlay_args") or []
        missing = [token for token in required if token not in overlay]
        assert not missing, (
            f"{entry['path']}: overlay_args missing recipe tokens: {missing}")


def test_run_example_maps_engine_call_metadata() -> None:
    run_script = (PROJECT_ROOT / "scripts" / "run_example.sh").read_text(
        encoding="utf-8")
    assert "ci_train_driver" not in run_script, (
        "driver script was removed; run_example.sh must call execute_train "
        "inline via the fork's command_utils")
    for path, model_type in ENTRY_MODEL_TYPES.items():
        assert f"MODEL_TYPE={model_type}" in run_script, (
            f"run_example.sh missing MODEL_TYPE mapping for {path}")
    for path, train_script in ENTRY_TRAIN_SCRIPTS.items():
        assert f"TRAIN_SCRIPT={train_script}" in run_script, (
            f"run_example.sh missing TRAIN_SCRIPT mapping for {path}")
    # The execute_train invocation must stay generic (one heredoc, no
    # per-entry branches after the metadata case).
    assert run_script.count("module.execute_train(") == 1
    assert "fork_root" in run_script and "command_utils.py" in run_script


def test_inline_launcher_python_is_syntactically_valid() -> None:
    run_script = (PROJECT_ROOT / "scripts" / "run_example.sh").read_text(
        encoding="utf-8")
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", run_script, re.DOTALL)
    assert len(blocks) == 2
    for block in blocks:
        compile(block, "run_example.sh:<inline Python>", "exec")

def test_workflow_registers_engine_call() -> None:
    workflow = (
        PROJECT_ROOT.parents[1] / ".github" / "workflows" / "slime-examples.yml"
    ).read_text(encoding="utf-8")
    assert "name: slime-examples" in workflow
    assert "uses: ./.github/workflows/examples-template.yml" in workflow
    assert "upstream_repo: THUDM/slime" in workflow
    assert "max_parallel: 1" in workflow
    # Bring-up phase: the cron line stays commented out.
    assert "  schedule:" not in workflow
    assert "#   - cron:" in workflow


def test_projects_yaml_registers_examples_workflow() -> None:
    registry = (
        PROJECT_ROOT.parents[1] / "projects.yaml"
    ).read_text(encoding="utf-8")
    assert "examples: .github/workflows/slime-examples.yml" in registry


def test_device_requirements_cover_all_supported_entries() -> None:
    manifest = load_manifest()
    for entry in manifest["supported"]:
        assert entry["path"] in ENTRY_DEVICE_REQUIREMENTS, (
            f"missing device requirement for {entry['path']}")
        assert entry["path"] in ENTRY_MODEL_TYPES, (
            f"missing model type mapping for {entry['path']}")
        assert entry["path"] in ENTRY_TRAIN_SCRIPTS, (
            f"missing train script mapping for {entry['path']}")
