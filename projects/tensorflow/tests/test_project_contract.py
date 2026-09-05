"""Static contract for the TensorFlow 2.6.5 + TF Adapter 9.1.0 guard."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
for _path in (_SRC, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from workflows.markdown_doc_test_base import MarkdownDocTestBase  # noqa: E402


class _Parser(MarkdownDocTestBase):
    pass


class TestTensorFlowProjectContract(unittest.TestCase):
    def test_project_registry_points_to_the_quick_start_workflow(self) -> None:
        registry = yaml.safe_load(
            (_REPO_ROOT / "projects.yaml").read_text(encoding="utf-8")
        )
        projects = [
            item for item in registry["projects"] if item["name"] == "tensorflow"
        ]
        self.assertEqual(len(projects), 1)
        project = projects[0]
        self.assertEqual(project["upstream_repo"], "tensorflow/tensorflow")
        self.assertEqual(project["runner"], "linux-aarch64-a2-1")
        self.assertEqual(
            project["workflows"],
            {"quick_start": ".github/workflows/tensorflow-quick-start.yml"},
        )

    def test_workflow_uses_fixed_tf265_ref_on_one_a2_npu(self) -> None:
        workflow = (
            _REPO_ROOT / ".github" / "workflows" / "tensorflow-quick-start.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("uses: ./.github/workflows/quick-start-template.yml", workflow)
        self.assertIn("project: tensorflow", workflow)
        self.assertIn("test_runner: '[\"linux-aarch64-a2-1\"]'", workflow)
        self.assertIn("fixed_ref: v2.6.5", workflow)
        self.assertNotIn("fixed_ref: v1.15.0", workflow)
        self.assertIn("upstream_repo: tensorflow/tensorflow", workflow)
        self.assertIn(
            "cann:9.1.0-910b-ubuntu22.04-py3.12-devel",
            workflow,
        )
        self.assertIn("container_options: ''", workflow)
        self.assertNotIn("--device=/dev/davinci", workflow)
        self.assertNotIn("/usr/local/Ascend/driver", workflow)

    def test_shared_engine_supports_an_optional_fixed_ref(self) -> None:
        template = (
            _REPO_ROOT / ".github" / "workflows" / "quick-start-template.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("fixed_ref:", template)
        self.assertIn("FIXED_REF: ${{ inputs.fixed_ref }}", template)
        self.assertIn('if [ -n "$FIXED_REF" ]; then', template)
        self.assertIn('ref="$FIXED_REF"', template)
        self.assertIn("/commits/$FIXED_REF", template)

    def test_doc_pins_the_official_compatible_stack_and_npu_flow(self) -> None:
        doc_path = (
            _REPO_ROOT
            / "projects"
            / "tensorflow"
            / "docs"
            / "Quick-start-Ascend.md"
        )
        text = doc_path.read_text(encoding="utf-8")
        commands, results = _Parser().parse(text)
        test_ids = {command.id for command in commands if hasattr(command, "id")}
        self.assertEqual(
            test_ids,
            {
                "check-python",
                "install-tensorflow",
                "install-tf-adapter",
                "run-npu-device",
            },
        )
        self.assertEqual(set(results), test_ids)
        test_commands = {
            command.id: command.cmd
            for command in commands
            if hasattr(command, "id")
        }
        self.assertNotIn("pip install", test_commands["install-tensorflow"])
        self.assertNotIn("curl ", test_commands["install-tensorflow"])
        self.assertNotIn("pip install", test_commands["install-tf-adapter"])
        self.assertNotIn("curl ", test_commands["install-tf-adapter"])

        required_fragments = (
            "| Python | 3.9.25 |",
            "| TensorFlow | 2.6.5 (`v2.6.5`) |",
            "| CANN | 9.1.0 |",
            "| TF Adapter branch | 9.1.0 |",
            "| TF Adapter wheel release | `tfa_v0.0.49_9.1.0` |",
            "| npu_device | 2.6.5 |",
            "| HDF5 | 1.10.5 |",
            "| h5py | 3.1.0 |",
            "| numpy | 1.19.5 |",
            "| protobuf | 3.19.6 |",
            "tensorflow-2.6.5-cp39-cp39-linux_aarch64.whl",
            "npu_device-2.6.5-py3-none-manylinux2014_aarch64.whl",
            "import npu_device as npu",
            "npu.open().as_default()",
            "@tf.function",
            "ASCEND_DEVICE_ID=0",
            "TensorFlow 2.6.5 Ascend Quick Start PASSED",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)

        official_paths = (
            "docs/zh/tfadapter_2/installation/tensorflow-2-6-5_install.md",
            "docs/zh/tfadapter_2/installation/tfadapter_install.md",
            "docs/zh/tfadapter_2/migration/script_migration/manual_porting.md",
        )
        for path in official_paths:
            with self.subTest(path=path):
                self.assertIn(path, text)

    def test_python39_is_installed_side_by_side_and_pip3_is_used(self) -> None:
        doc = (
            _REPO_ROOT
            / "projects"
            / "tensorflow"
            / "docs"
            / "Quick-start-Ascend.md"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "https://repo.huaweicloud.com/python/3.9.25/Python-3.9.25.tgz",
            doc,
        )
        self.assertIn(
            "a7438eabd3a48139f42d4e058096af8d880b0bb6e8fb8c78838892e4ce5583f2",
            doc,
        )
        self.assertIn("--prefix=/usr/local/python3.9.25", doc)
        self.assertIn("make altinstall", doc)
        self.assertIn("export PATH=/usr/local/python3.9.25/bin:$PATH", doc)
        self.assertNotIn("/usr/local/python3.7.10", doc)
        self.assertIn("pip3 install", doc)
        self.assertNotIn("uv pip install", doc)
        self.assertNotIn("UV_PYTHON_DOWNLOADS", doc)
        self.assertNotIn("micro.mamba.pm", doc)
        self.assertNotIn("micromamba", doc.lower())
        self.assertNotIn("conda-forge", doc)

    def test_hdf5_setup_keeps_the_upstream_aarch64_commands(self) -> None:
        doc = (
            _REPO_ROOT
            / "projects"
            / "tensorflow"
            / "docs"
            / "Quick-start-Ascend.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/"
            "hdf5-1.10.5/src/hdf5-1.10.5.tar.gz",
            doc,
        )
        self.assertIn("./configure --prefix=/usr/local/hdf5", doc)
        self.assertIn("make -j16 && make install", doc)
        self.assertIn(
            "export CPATH=/usr/local/hdf5/include/:/usr/local/hdf5/lib/",
            doc,
        )
        self.assertIn(
            "export LD_LIBRARY_PATH=/usr/local/hdf5/lib/:${LD_LIBRARY_PATH:-}",
            doc,
        )

        test_file = (
            _REPO_ROOT
            / "projects"
            / "tensorflow"
            / "tests"
            / "test_quick_start_ascend.py"
        ).read_text(encoding="utf-8")
        self.assertIn("/usr/local/hdf5/lib", test_file)

    def test_slow_bootstrap_phases_are_separate_setup_commands(self) -> None:
        doc_path = (
            _REPO_ROOT
            / "projects"
            / "tensorflow"
            / "docs"
            / "Quick-start-Ascend.md"
        )
        commands, _ = _Parser().parse(doc_path.read_text(encoding="utf-8"))
        markers = (
            "apt-get update",
            "Python-3.9.25.tgz",
            "hdf5-1.10.5.tar.gz",
            'pip3 install "Cython<3"',
        )
        phase_indices = [
            next(i for i, command in enumerate(commands) if marker in command.cmd)
            for marker in markers
        ]
        self.assertEqual(phase_indices, sorted(phase_indices))
        self.assertEqual(len(set(phase_indices)), len(phase_indices))
        for command in commands:
            with self.subTest(command=command.cmd[:80]):
                self.assertLessEqual(
                    sum(marker in command.cmd for marker in markers),
                    1,
                )

    def test_h5py_install_is_an_explicit_step_after_hdf5(self) -> None:
        doc_path = (
            _REPO_ROOT
            / "projects"
            / "tensorflow"
            / "docs"
            / "Quick-start-Ascend.md"
        )
        text = doc_path.read_text(encoding="utf-8")
        commands, results = _Parser().parse(text)
        hdf5_index = next(
            i for i, command in enumerate(commands)
            if "export LD_LIBRARY_PATH=/usr/local/hdf5/lib/" in command.cmd
        )
        h5py_deps_index = next(
            i for i, command in enumerate(commands)
            if 'pip3 install "Cython<3"' in command.cmd
        )
        h5py_install_index = next(
            i for i, command in enumerate(commands)
            if 'pip3 install "h5py==3.1.0"' in command.cmd
        )
        self.assertEqual(h5py_deps_index, hdf5_index + 1)
        self.assertEqual(h5py_install_index, h5py_deps_index + 1)
        h5py_deps_command = commands[h5py_deps_index].cmd
        for dependency in (
            'pip3 install "Cython<3"',
            "pip3 install wheel",
            'pip3 install "numpy==1.19.5"',
        ):
            with self.subTest(dependency=dependency):
                self.assertIn(dependency, h5py_deps_command)
        self.assertIn(
            'pip3 install "h5py==3.1.0"',
            commands[h5py_install_index].cmd,
        )
        tensorflow_command = next(
            command.cmd for command in commands
            if 'pip3 install "$TF_WHEEL"' in command.cmd
        )
        self.assertNotIn("h5py==3.1.0", tensorflow_command)
        self.assertNotIn('Cython<3', tensorflow_command)
        self.assertIn("numpy 1.19.5", results["check-python"].body)
        self.assertIn("h5py 3.1.0", results["check-python"].body)
        self.assertIn("#### 安装 h5py 3.1.0", text)

    def test_tensorflow_aarch64_downloads_the_pinned_wheel(self) -> None:
        doc = (
            _REPO_ROOT
            / "projects"
            / "tensorflow"
            / "docs"
            / "Quick-start-Ascend.md"
        ).read_text(encoding="utf-8")
        for fragment in (
            "## 安装 TensorFlow 2.6.5",
            "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/"
            "OpenSource/packages/"
            "tensorflow-2.6.5-cp39-cp39-linux_aarch64.whl",
            "be1c8f52d6a72cc0db5826605f61c196777f5939441b7e87442688a5d1866bd0",
            'pip3 install "protobuf==3.19.6"',
            'pip3 install "$TF_WHEEL"',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, doc)
        for source_build_fragment in (
            "--branch v2.6.5",
            "nsync-1.22.0.tar.gz",
            "bazel-0.26.1-dist.zip",
            "openjdk-8-jdk-headless",
            "| GCC | linux_gcc7.3.0 |",
            "//tensorflow/tools/pip_package:build_pip_package",
        ):
            with self.subTest(source_build_fragment=source_build_fragment):
                self.assertNotIn(source_build_fragment, doc)

        readme = (
            _REPO_ROOT / "projects" / "tensorflow" / "README.md"
        ).read_text(encoding="utf-8")
        self.assertIn("TensorFlow 2.6.5", readme)
        self.assertIn("npu_device", readme)
        self.assertIn("aarch64 wheel", readme)
        self.assertNotIn("TensorFlow 1.15", readme)

    def test_tf_adapter_uses_the_documented_pip3_target_install(self) -> None:
        doc = (
            _REPO_ROOT
            / "projects"
            / "tensorflow"
            / "docs"
            / "Quick-start-Ascend.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'pip3 install "$ADAPTER_WHEEL" --force-reinstall '
            '-t "$TFPLUGIN_INSTALL_PATH"',
            doc,
        )
        self.assertIn(
            'export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH',
            doc,
        )
        self.assertIn(
            "https://gitcode.com/cann/tensorflow/releases/download/"
            "tfa_v0.0.49_9.1.0/"
            "npu_device-2.6.5-py3-none-manylinux2014_aarch64.whl",
            doc,
        )
        self.assertIn(
            "68a14762b24ebfafe554c2a29406be2932b82a1950938d1de97a2cc0909d73fc",
            doc,
        )
        self.assertNotIn("npu_bridge", doc)
        self.assertNotIn("uv pip install", doc)

    def test_wheel_install_uses_the_standard_quick_start_timeouts(self) -> None:
        workflow = (
            _REPO_ROOT / ".github" / "workflows" / "tensorflow-quick-start.yml"
        ).read_text(encoding="utf-8")
        test_file = (
            _REPO_ROOT
            / "projects"
            / "tensorflow"
            / "tests"
            / "test_quick_start_ascend.py"
        ).read_text(encoding="utf-8")
        self.assertIn("timeout_minutes: 120", workflow)
        self.assertIn("DEFAULT_COMMAND_TIMEOUT = 3600", test_file)

    def test_install_sections_follow_the_upstream_document_structure(self) -> None:
        doc = (
            _REPO_ROOT
            / "projects"
            / "tensorflow"
            / "docs"
            / "Quick-start-Ascend.md"
        ).read_text(encoding="utf-8")
        headings = (
            "## 安装前准备",
            "### 编译安装 HDF5 1.10.5",
            "### 安装 h5py",
            "#### 安装 h5py 依赖包",
            "#### 安装 h5py 3.1.0",
            "## 安装 TensorFlow 2.6.5",
            "## 安装框架插件包 TF Adapter",
            "### 安装插件包",
            "#### 1. 获取 TF Adapter 安装包",
            "#### 2. 安装 TF Adapter",
            "#### 3. 设置 TF Adapter 环境变量",
        )
        positions = []
        for heading in headings:
            with self.subTest(heading=heading):
                self.assertIn(heading, doc)
            if heading in doc:
                positions.append(doc.index(heading))
        if len(positions) == len(headings):
            self.assertEqual(positions, sorted(positions))

    def test_apt_uses_the_existing_arm64_mirror_pattern(self) -> None:
        doc = (
            _REPO_ROOT
            / "projects"
            / "tensorflow"
            / "docs"
            / "Quick-start-Ascend.md"
        ).read_text(encoding="utf-8")
        mirror = "https://mirrors.aliyun.com/ubuntu-ports/"
        self.assertIn(mirror, doc)
        self.assertLess(doc.index(mirror), doc.index("apt-get update"))

    def test_apt_update_and_install_are_separate_verbose_steps(self) -> None:
        doc_path = (
            _REPO_ROOT
            / "projects"
            / "tensorflow"
            / "docs"
            / "Quick-start-Ascend.md"
        )
        text = doc_path.read_text(encoding="utf-8")
        commands, _ = _Parser().parse(text)
        update_index = next(
            i for i, command in enumerate(commands)
            if "apt-get update" in command.cmd
        )
        install_index = next(
            i for i, command in enumerate(commands)
            if "apt-get install" in command.cmd
        )
        self.assertEqual(install_index, update_index + 1)
        self.assertNotIn("apt-get install", commands[update_index].cmd)
        self.assertNotIn("apt-get update", commands[install_index].cmd)
        self.assertNotIn("-qq", text)

    def test_long_running_steps_emit_progress_boundaries(self) -> None:
        doc_path = (
            _REPO_ROOT
            / "projects"
            / "tensorflow"
            / "docs"
            / "Quick-start-Ascend.md"
        )
        commands, _ = _Parser().parse(doc_path.read_text(encoding="utf-8"))
        boundaries = (
            ("apt-get update", "Updating package indexes", "Package indexes updated"),
            (
                "apt-get install",
                "Installing build dependencies",
                "Build dependencies installed",
            ),
            ("Python-3.9.25.tgz", "Preparing Python 3.9.25", "Python 3.9.25 is ready"),
            (
                "hdf5-1.10.5.tar.gz",
                "Downloading HDF5 1.10.5",
                "HDF5 1.10.5 source is ready",
            ),
            ("make -j16", "Building HDF5 1.10.5", "HDF5 1.10.5 is ready"),
            (
                'pip3 install "h5py==3.1.0"',
                "Installing h5py 3.1.0",
                "h5py 3.1.0 is installed",
            ),
            (
                'pip3 install "$TF_WHEEL"',
                "Installing TensorFlow 2.6.5",
                "TensorFlow 2.6.5 is installed",
            ),
            (
                "tfa_v0.0.49_9.1.0/npu_device",
                "Downloading TF Adapter 9.1.0",
                "TF Adapter wheel is ready",
            ),
            (
                'pip3 install "$ADAPTER_WHEEL"',
                "Installing npu_device 2.6.5",
                "npu_device 2.6.5 is installed",
            ),
        )
        for command_marker, before, after in boundaries:
            command = next(
                command.cmd for command in commands
                if command_marker in command.cmd
            )
            with self.subTest(command_marker=command_marker):
                self.assertIn(before, command)
                self.assertIn(after, command)
                if before in command and after in command:
                    self.assertLess(command.index(before), command.index(command_marker))
                    self.assertLess(command.index(command_marker), command.index(after))

    def test_quick_start_explains_user_actions_not_ci_internals(self) -> None:
        doc = (
            _REPO_ROOT
            / "projects"
            / "tensorflow"
            / "docs"
            / "Quick-start-Ascend.md"
        ).read_text(encoding="utf-8")
        for internal_phrase in (
            "CI Runner",
            "文档测试框架",
            "工作流提供",
            "与仓库中的 OpenCV Quick Start",
            "CI 输出稳定",
            "这里看护的",
        ):
            with self.subTest(internal_phrase=internal_phrase):
                self.assertNotIn(internal_phrase, doc)
        self.assertIn("如果已经配置了可用的 Ubuntu 软件源", doc)

    def test_e2e_test_sources_cann_and_is_npu_gated(self) -> None:
        test_file = (
            _REPO_ROOT
            / "projects"
            / "tensorflow"
            / "tests"
            / "test_quick_start_ascend.py"
        ).read_text(encoding="utf-8")
        self.assertIn("MarkdownDocTestBase", test_file)
        self.assertIn("sys.path.insert", test_file)
        self.assertIn("NPU_READY", test_file)
        self.assertIn("/usr/local/Ascend/cann/set_env.sh", test_file)
        self.assertIn("TFPLUGIN_INSTALL_PATH", test_file)
        self.assertIn("ASCEND_DEVICE_ID", test_file)
        self.assertIn("_PYTHON39_BIN", test_file)
        self.assertNotIn("NpuOptimizer init failed", test_file)
        self.assertIn("ERR99999", test_file)


if __name__ == "__main__":
    unittest.main()
