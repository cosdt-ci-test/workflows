"""Static contract for the TensorFlow 1.15 + TF Adapter 9.1.0 guard."""

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

    def test_workflow_uses_fixed_tf115_ref_on_one_a2_npu(self) -> None:
        workflow = (
            _REPO_ROOT / ".github" / "workflows" / "tensorflow-quick-start.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("uses: ./.github/workflows/quick-start-template.yml", workflow)
        self.assertIn("project: tensorflow", workflow)
        self.assertIn("test_runner: '[\"linux-aarch64-a2-1\"]'", workflow)
        self.assertIn("fixed_ref: v1.15.0", workflow)
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
                "run-npu-optimizer",
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
            "| Python | 3.7.10 |",
            "| TensorFlow | 1.15.0 (`v1.15.0`) |",
            "| CANN | 9.1.0 |",
            "| TF Adapter branch | 9.1.0 |",
            "| TF Adapter wheel release | `tfa_v0.0.49_9.1.0` |",
            "| npu_bridge | 1.15.0 |",
            "| HDF5 | 1.10.5 |",
            "| h5py | 2.8.0 |",
            "tensorflow-1.15.0-*.whl",
            "npu_bridge-1.15.0-py3-none-manylinux2014_aarch64.whl",
            "from npu_bridge.npu_init import *",
            'custom_op.name = "NpuOptimizer"',
            "RewriterConfig.OFF",
            "ASCEND_DEVICE_ID=0",
            "TensorFlow Ascend Quick Start PASSED",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)

        official_paths = (
            "docs/zh/tfadapter_1/installation/tensorflow-1-15_install.md",
            "docs/zh/tfadapter_1/installation/tfadapter_install.md",
            "docs/zh/tfadapter_1/quick_start.md",
        )
        for path in official_paths:
            with self.subTest(path=path):
                self.assertIn(path, text)

    def test_python37_is_installed_side_by_side_and_pip3_is_used(self) -> None:
        doc = (
            _REPO_ROOT
            / "projects"
            / "tensorflow"
            / "docs"
            / "Quick-start-Ascend.md"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "https://repo.huaweicloud.com/python/3.7.10/Python-3.7.10.tgz",
            doc,
        )
        self.assertIn(
            "c9649ad84dc3a434c8637df6963100b2e5608697f9ba56d82e3809e4148e0975",
            doc,
        )
        self.assertIn("--prefix=/usr/local/python3.7.10", doc)
        self.assertIn("make altinstall", doc)
        self.assertIn("export PATH=/usr/local/python3.7.10/bin:$PATH", doc)
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
            "Python-3.7.10.tgz",
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
            if "pip3 install h5py==2.8.0" in command.cmd
        )
        self.assertEqual(h5py_deps_index, hdf5_index + 1)
        self.assertEqual(h5py_install_index, h5py_deps_index + 1)
        h5py_deps_command = commands[h5py_deps_index].cmd
        for dependency in (
            'pip3 install "Cython<3"',
            "pip3 install wheel",
            "pip3 install numpy",
        ):
            with self.subTest(dependency=dependency):
                self.assertIn(dependency, h5py_deps_command)
        self.assertIn(
            "pip3 install h5py==2.8.0",
            commands[h5py_install_index].cmd,
        )
        tensorflow_command = next(
            command.cmd for command in commands
            if "pip3 install tensorflow-1.15.0-*.whl" in command.cmd
        )
        self.assertNotIn("h5py==2.8.0", tensorflow_command)
        self.assertNotIn('Cython<3', tensorflow_command)
        self.assertIn("h5py 2.8.0", results["check-python"].body)
        self.assertIn("#### 安装 h5py 2.8.0", text)

    def test_tensorflow_aarch64_follows_the_v115_source_build_flow(self) -> None:
        doc = (
            _REPO_ROOT
            / "projects"
            / "tensorflow"
            / "docs"
            / "Quick-start-Ascend.md"
        ).read_text(encoding="utf-8")
        for fragment in (
            "--branch v1.15.0",
            "nsync-1.22.0.tar.gz",
            "#define ATM_CB_() __sync_synchronize()",
            "sha256sum /tmp/nsync-1.22.0.tar.gz",
            '"file:///tmp/nsync-1.22.0.tar.gz"',
            "bazel-0.26.1-dist.zip",
            "openjdk-8-jdk-headless",
            "JAVA_HOME=/usr/lib/jvm/java-8-openjdk-arm64",
            "| GCC | linux_gcc7.3.0 |",
            "TensorFlow 1.15 requires linux_gcc7.3.0",
            "-D_GLIBCXX_USE_CXX11_ABI=0",
            "//tensorflow/tools/pip_package:build_pip_package",
            "tensorflow-1.15.0-*.whl",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, doc)
        self.assertNotIn(
            "ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/"
            "python/packages/tensorflow-1.15.0",
            doc,
        )

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
        self.assertNotIn("uv pip install", doc)

    def test_source_build_timeouts_cover_the_official_flow(self) -> None:
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
        self.assertIn("timeout_minutes: 360", workflow)
        self.assertIn("DEFAULT_COMMAND_TIMEOUT = 14400", test_file)

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
            "#### 安装 h5py 2.8.0",
            "## 安装 TensorFlow",
            "### 1. 下载 nsync 1.22.0",
            "### 2. 修改 nsync 1.22.0",
            "### 3. 重新压缩 nsync 1.22.0",
            "### 4. 生成 sha256sum 校验码",
            "### 5. 修改 sha256sum 和 urls",
            "### 6. 编译 TensorFlow",
            "### 7. 安装编译好的 TensorFlow",
            "### 8. 验证 TensorFlow",
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
        self.assertIn("ERR99999", test_file)


if __name__ == "__main__":
    unittest.main()
