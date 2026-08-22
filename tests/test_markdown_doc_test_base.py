"""``markdown_doc_test_base`` 的单元测试。

覆盖契约规则的解析 / 校验 / 折叠 / 占位符替换 / 正则比对五条核心路径：

* ``parse`` 切代码块（普通块跳过、HTML 注释内 setup 识别、mistune AST 行为）
* ``_parse_block`` 解析 info string（id / store / load / fuzzy）
* ``_validate`` 规则 2/5/7/10/11 违规抛 ``LabelSpecError``
* ``_fold`` 生成 ``SetupCommand`` / ``TestCommand`` / ``TestExpectedOutput``
* ``substitute_placeholders`` 纯函数（缺 store 时保留字面占位符）
* ``compare_output`` 默认 ``...`` 与 ``fuzzy='xxx'`` 的非贪婪跨行匹配

跑法：``python -m unittest tests.test_markdown_doc_test_base -v 2>&1``

源码位置：``src/workflows/markdown_doc_test_base.py``，所有项目共享。本
测试文件通过 ``sys.path`` 注入仓库根 + ``src/`` 后按 ``workflows.*`` 包
导入，避免在测试目录重复放置同一份源码。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# 把仓库根加进 sys.path，使 ``workflows.*`` 可解析为 ``src/workflows/*``。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # tests/ -> workflows/
_SRC = _REPO_ROOT / 'src'
for _p in (_SRC, _REPO_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from workflows.markdown_doc_test_base import (  # noqa: E402
    LabelSpecError,
    MarkdownDocTestBase,
    SetupCommand,
    TestCommand,
    TestExpectedOutput,
    _rescan_fences,
)


class _Bare(MarkdownDocTestBase):
    """不带 pre/post 钩子的最小子类,只暴露基类方法供测试直接调用。"""

    def pre_process(self) -> str:
        raise NotImplementedError

    def post_process(self) -> None:
        return None


def _parse(text: str) -> tuple[list, dict]:
    """薄封装:跳过 pre_process,直接调基类 ``parse``。"""
    return _Bare().parse(text)


class TestScanBlocks(unittest.TestCase):
    """``_scan_blocks`` + ``_parse_block`` 路径：无标签块跳过、HTML 注释识别。"""

    def test_plain_block_skipped(self):
        """无标签的 ```python``` 不进 commands / results。"""
        text = (
            '```python\n'
            'print(1)\n'
            '```\n'
        )
        commands, results = _parse(text)
        self.assertEqual(commands, [])
        self.assertEqual(results, {})

    def test_test_block_in_main_sequence(self):
        """```shell #test id=\"x\"``` 进 commands 不进 results。"""
        text = (
            '```shell #test id="x"\n'
            'echo hi\n'
            '```\n'
            '\n'
            '```shell #test-result id="x"\n'
            'hi\n'
            '```\n'
        )
        commands, results = _parse(text)
        self.assertEqual(len(commands), 1)
        self.assertIsInstance(commands[0], TestCommand)
        self.assertEqual(commands[0].id, 'x')
        self.assertEqual(results.keys(), {'x'})

    def test_setup_block_with_store(self):
        """```shell #test-setup store=\"x\"``` 进 commands。"""
        text = (
            '```shell #test-setup store="x"\n'
            'echo captured\n'
            '```\n'
        )
        commands, results = _parse(text)
        self.assertEqual(len(commands), 1)
        self.assertIsInstance(commands[0], SetupCommand)
        self.assertEqual(commands[0].store, 'x')
        self.assertEqual(results, {})

    def test_setup_in_html_comment(self):
        """HTML 注释内的 #test-setup 仍切出,hidden=True。"""
        text = (
            '<!-- hidden setup\n'
            '```shell #test-setup store="hidden_var"\n'
            'echo from_comment\n'
            '```\n'
            '-->\n'
        )
        commands, results = _parse(text)
        self.assertEqual(len(commands), 1)
        self.assertTrue(commands[0].hidden)
        self.assertEqual(commands[0].store, 'hidden_var')


class TestRescanFences(unittest.TestCase):
    """``_rescan_fences`` 从 ``block_html.raw`` 救出注释内 fence。"""

    def test_rescues_single_fence(self):
        raw = (
            '<!-- \n'
            '```shell #test-setup\n'
            'cmd\n'
            '```\n'
            '-->'
        )
        out = _rescan_fences(raw)
        self.assertEqual(len(out), 1)
        info, body = out[0]
        self.assertEqual(info, 'shell #test-setup')
        self.assertEqual(body, 'cmd')

    def test_rescues_multiple_fences(self):
        raw = (
            '<!--\n'
            '```shell #test-setup store="a"\n'
            'A\n'
            '```\n'
            '\n'
            '```shell #test-setup store="b"\n'
            'B\n'
            '```\n'
            '-->'
        )
        out = _rescan_fences(raw)
        self.assertEqual([info for info, _ in out],
                         ['shell #test-setup store="a"',
                          'shell #test-setup store="b"'])

    def test_unclosed_fence_raises(self):
        raw = '<!--\n```shell #test-setup\nstill running\n'
        with self.assertRaises(LabelSpecError):
            _rescan_fences(raw)

    def test_empty_html_block_no_fence(self):
        """无 fence 的注释块返回空列表（不抛错）。"""
        self.assertEqual(_rescan_fences('<!-- just a comment -->'), [])
        self.assertEqual(_rescan_fences('<!-- \nmulti\nline\ncomment\n-->'), [])


class TestValidateRules(unittest.TestCase):
    """``_validate``：规则 2/5/7/10/11。"""

    def test_rule2_duplicate_id(self):
        """同 type 重复 id 抛 ``LabelSpecError``。"""
        text = (
            '```shell #test id="dup"\necho a\n```\n'
            '```shell #test id="dup"\necho b\n```\n'
            '```shell #test-result id="dup"\nb\n```\n'
        )
        with self.assertRaises(LabelSpecError) as cm:
            _parse(text)
        msg = str(cm.exception)
        self.assertIn('duplicate', msg)
        self.assertIn('dup', msg)

    def test_rule5_missing_pair(self):
        """#test 缺同 id 的 #test-result 抛错。"""
        text = '```shell #test id="lonely"\necho a\n```\n'
        with self.assertRaises(LabelSpecError) as cm:
            _parse(text)
        self.assertIn('matching', str(cm.exception))

    def test_rule7_test_without_language(self):
        """规则 7:``#test`` / ``#test-setup`` 必须有 language,且必须在契约
        白名单内(当前只支持 shell)。"""
        # 缺 language
        text = '```#test id="x"\necho\n```\n'
        with self.assertRaises(LabelSpecError) as cm:
            _parse(text)
        msg = str(cm.exception)
        self.assertIn('#test', msg)
        self.assertIn('echo', msg)
        # #test-setup 缺 language 也抛错
        text_setup = '```#test-setup store="x"\necho\n```\n'
        with self.assertRaises(LabelSpecError) as cm:
            _parse(text_setup)
        msg = str(cm.exception)
        self.assertIn('#test-setup', msg)
        self.assertIn('echo', msg)
        # 不支持的语言(text / python / console 等)也抛错
        for lang in ('python', 'text', 'console'):
            text_bad = f'```{lang} #test id="x"\necho\n```\n'
            with self.assertRaises(LabelSpecError) as cm:
                _parse(text_bad)
            msg = str(cm.exception)
            self.assertIn(lang, msg)
            self.assertIn('not supported', msg)
        # shell 通过校验
        text_ok = (
            '```shell #test id="x"\necho\n```\n'
            '```shell #test-result id="x"\nhi\n```\n'
        )
        commands, _results = _parse(text_ok)
        self.assertEqual(commands[0].language, 'shell')

    def test_rule10_non_setup_in_comment(self):
        """HTML 注释内出现 #test / #test-result 抛错。"""
        text = (
            '<!-- bad\n'
            '```shell #test id="x"\necho a\n```\n'
            '-->\n'
            '```shell #test-result id="x"\na\n```\n'
        )
        with self.assertRaises(LabelSpecError) as cm:
            _parse(text)
        self.assertIn('HTML comment', str(cm.exception))

    def test_rule11_load_before_store(self):
        """load 引用的 store 在文档中还没出现就报错。"""
        text = (
            '```shell #test id="x" load="missing>>local"\n'
            'echo <local>\n```\n'
            '```shell #test-result id="x"\nsomething\n```\n'
        )
        with self.assertRaises(LabelSpecError) as cm:
            _parse(text)
        self.assertIn("load=", str(cm.exception))
        self.assertIn('earlier', str(cm.exception))

    def test_rule12_fuzzy_only_on_test_result(self):
        """``fuzzy=`` 只允许出现在 ``#test-result``:``#test`` / ``#test-setup``
        块带 fuzzy 直接报错。"""
        for label in ('#test', '#test-setup'):
            text = (
                f'```shell {label} id="x" fuzzy="xxx"\n'
                'echo\n```\n'
                f'```shell #test-result id="x"\nhi\n```\n'
            )
            with self.assertRaises(LabelSpecError) as cm:
                _parse(text)
            self.assertIn('fuzzy', str(cm.exception))
            self.assertIn(label, str(cm.exception))

    def test_rule13_multi_fuzzy_parsed(self):
        """``#test-result`` 上 ``fuzzy='xxx' fuzzy='yyy'`` 解析为 tuple,
        ``xxx`` / ``yyy`` 都是非贪婪通配。"""
        text = (
            '```shell #test-result id="x" fuzzy="xxx" fuzzy="yyy"\n'
            'a xxx b yyy c\n'
            '```\n'
        )
        commands, results = _parse(text)
        self.assertEqual(results['x'].fuzzy, ('xxx', 'yyy'))
        # 实际匹配也走通:xxx 和 yyy 各自按通配处理
        actual = 'a 1 b 2 c\n'
        self.assertTrue(_Bare().compare_output(actual, results['x'].body,
                                               fuzzy=results['x'].fuzzy))

    def test_rule14_duplicate_fuzzy_rejected(self):
        """同一个 placeholder 写两次算违规:大概率笔误。"""
        text = (
            '```shell #test-result id="x" fuzzy="xxx" fuzzy="xxx"\n'
            'a\n```\n'
        )
        with self.assertRaises(LabelSpecError) as cm:
            _parse(text)
        self.assertIn('duplicate fuzzy', str(cm.exception))

    def test_rule15_disable_fuzzy_parsed(self):
        """``disable_fuzzy`` 出现在 ``#test-result`` 上被解析为 ``True``。"""
        text = (
            '```shell #test-result id="x" disable_fuzzy\n'
            'hello ... world\n'
            '```\n'
        )
        commands, results = _parse(text)
        self.assertTrue(results['x'].disable_fuzzy)
        # 也验证 compare_output 真按字面匹配
        self.assertTrue(_Bare().compare_output(
            'hello ... world\n', results['x'].body,
            disable_fuzzy=results['x'].disable_fuzzy))

    def test_rule16_disable_fuzzy_conflicts_with_fuzzy(self):
        """``disable_fuzzy`` 与 ``fuzzy=`` 互斥,一起写报错。"""
        text = (
            '```shell #test-result id="x" disable_fuzzy fuzzy="xxx"\n'
            'a\n```\n'
        )
        with self.assertRaises(LabelSpecError) as cm:
            _parse(text)
        self.assertIn('conflicts', str(cm.exception))

    def test_rule17_disable_fuzzy_only_on_test_result(self):
        """``disable_fuzzy`` 只允许出现在 ``#test-result``。"""
        for label in ('#test', '#test-setup'):
            text = (
                f'```shell {label} id="x" disable_fuzzy\n'
                'echo\n```\n'
            )
            with self.assertRaises(LabelSpecError) as cm:
                _parse(text)
            self.assertIn('disable_fuzzy', str(cm.exception))

    def test_rule11_load_after_store_ok(self):
        """load 引用的 store 在更早位置出现则通过。"""
        text = (
            '```shell #test-setup store="x"\necho captured\n```\n'
            '```shell #test id="y" load="x>>local"\necho <local>\n```\n'
            '```shell #test-result id="y"\ncaptured\n```\n'
        )
        commands, results = _parse(text)
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[1].load, (('x', 'local'),))

    def test_rule11_store_in_hidden_setup_counts(self):
        """HTML 注释内的 store 也算 seen_stores（先 load 后 store 仍会报错）。"""
        text = (
            '<!--\n'
            '```shell #test-setup store="hidden_store"\n'
            'echo a\n```\n'
            '-->\n'
            '```shell #test id="y" load="hidden_store>>h"\n'
            'echo <h>\n```\n'
            '```shell #test-result id="y"\nx\n```\n'
        )
        # hidden store 在前,load 在后,应该通过（hidden 一样计入）。
        commands, _ = _parse(text)
        self.assertEqual(len(commands), 2)
        self.assertTrue(commands[0].hidden)
        self.assertEqual(commands[1].load, (('hidden_store', 'h'),))


class TestFold(unittest.TestCase):
    """``_fold``：生成正确的 dataclass。"""

    def test_setup_command_fields(self):
        text = (
            '```shell #test-setup store="abc"\n'
            'echo line1\n'
            'echo line2\n'
            '```\n'
        )
        commands, _ = _parse(text)
        self.assertEqual(len(commands), 1)
        cmd = commands[0]
        self.assertEqual(cmd.cmd, 'echo line1\necho line2')
        self.assertEqual(cmd.store, 'abc')
        self.assertFalse(cmd.hidden)

    def test_test_command_fields(self):
        text = (
            '```shell #test-setup store="s"\necho captured\n```\n'
            '```shell #test id="abc" load="s>>p"\n'
            'echo <p>\n'
            '```\n'
            '```shell #test-result id="abc" fuzzy="xxx"\n'
            'captured\n'
            '```\n'
        )
        commands, results = _parse(text)
        self.assertEqual(len(commands), 2)
        cmd = commands[1]
        self.assertEqual(cmd.id, 'abc')
        self.assertEqual(cmd.language, 'shell')
        self.assertEqual(cmd.load, (('s', 'p'),))
        self.assertIsInstance(results['abc'], TestExpectedOutput)
        self.assertEqual(results['abc'].fuzzy, ('xxx',))


class TestSubstitutePlaceholders(unittest.TestCase):
    """``substitute_placeholders`` 纯函数。"""

    def setUp(self):
        self.base = _Bare()

    def test_single_load_replaced(self):
        out = self.base.substitute_placeholders(
            'echo <ckpt>', (('checkpoint', 'ckpt'),),
            {'checkpoint': '/path/to/ckpt'},
        )
        self.assertEqual(out, 'echo /path/to/ckpt')

    def test_multi_load_replaced(self):
        out = self.base.substitute_placeholders(
            '[<u>] <cwd>',
            (('workdir', 'cwd'), ('user', 'u')),
            {'workdir': '/hdc', 'user': 'hdc'},
        )
        self.assertEqual(out, '[hdc] /hdc')

    def test_missing_store_preserves_placeholder(self):
        """captures 缺 store_var 时保留 <local> 字面（不静默替换成空）。"""
        out = self.base.substitute_placeholders(
            'echo <ckpt>', (('checkpoint', 'ckpt'),), {},
        )
        self.assertEqual(out, 'echo <ckpt>')

    def test_empty_load_noop(self):
        out = self.base.substitute_placeholders(
            'echo plain', (), {'checkpoint': 'x'},
        )
        self.assertEqual(out, 'echo plain')


class TestCompareOutput(unittest.TestCase):
    """``compare_output`` 默认 ``...`` / ``fuzzy`` 多占位符非贪婪跨行匹配。"""

    def setUp(self):
        self.base = _Bare()

    def test_default_placeholder_multiline(self):
        """``fuzzy=('...',)`` 跨行匹配 (DOTALL)。"""
        actual = 'run sh: /bin/bash\ninit banner\nloss=2.5\nfinal line\n'
        expected = 'run sh: ...\nfinal line\n'
        self.assertTrue(
            self.base.compare_output(actual, expected, fuzzy=('...',))
        )

    def test_fuzzy_overrides_default(self):
        """``fuzzy='xxx'`` 时 ``xxx`` 是占位符;不传 ``...`` 时默认不再内置。"""
        actual = 'Python 3.12.5\n'
        expected = 'Python 3.xxx'
        # 仅 'xxx' 是占位符,``...`` 不内置 -> 'xxx' 通配匹配 12.5
        self.assertTrue(
            self.base.compare_output(actual, expected, fuzzy='xxx')
        )
        # 不传 fuzzy 时,expected 字面 (含 xxx) 必须字面出现
        self.assertFalse(self.base.compare_output(actual, expected))

    def test_fuzzy_multiple_placeholders(self):
        """``fuzzy=('xxx', 'yyy')``:每种 placeholder 各自按非贪婪通配。"""
        actual = 'step 1/5 loss=2.5\nstep 5/5 loss=0.1\n'
        expected = 'step xxx/5 loss=yyy\nstep 5/5 loss=yyy\n'
        self.assertTrue(
            self.base.compare_output(actual, expected, fuzzy=('xxx', 'yyy'))
        )
        # 单独只用 yyy:xxx 应按字面匹配
        actual2 = 'step 1/5 loss=2.5\n'
        expected2 = 'step xxx/5 loss=yyy\n'
        self.assertTrue(
            self.base.compare_output(actual2, expected2, fuzzy=('xxx', 'yyy'))
        )

    def test_fuzzy_default_alongside_custom(self):
        """``fuzzy=('...', 'xxx')``:两种占位符并存,调用方显式声明。"""
        actual = 'header\n...\nbody\n'
        expected = 'header\n...\nbody\n'
        self.assertTrue(
            self.base.compare_output(actual, expected, fuzzy=('...', 'xxx'))
        )

    def test_literal_mismatch(self):
        """期望含字面 token 与实际不一致 -> False。"""
        actual = 'hello world\n'
        expected = 'hello there\n'
        self.assertFalse(
            self.base.compare_output(actual, expected, fuzzy=('...',))
        )

    def test_literal_match(self):
        actual = 'exact line\n'
        expected = 'exact line\n'
        self.assertTrue(self.base.compare_output(actual, expected))

    def test_disable_fuzzy_literal_dots(self):
        """``disable_fuzzy=True``:所有 placeholder 按字面匹配。

        验证:
          1. 不传 fuzzy 时(空 tuple)compare_output 走字面匹配;
             字面相等 -> True,字面不等 -> False。
          2. ``disable_fuzzy=True`` 后即便显式传 fuzzy 也按字面。
        """
        actual_dots = 'hello ... world\n'
        actual_no_dots = 'hello world\n'
        # 1a:不传 fuzzy + literal 一致 -> 字面匹配 True
        self.assertTrue(self.base.compare_output(actual_dots, 'hello ... world'))
        # 1b:不传 fuzzy + literal 不等 -> False
        self.assertFalse(self.base.compare_output(
            actual_no_dots, 'hello ... world'))
        # 1c:传 fuzzy=('...',) 后 ``...`` 是通配,actual=literal 也匹配
        self.assertTrue(self.base.compare_output(
            actual_dots, 'hello ... world', fuzzy=('...',)))
        # 2:disable_fuzzy=True + 字面不等 -> False
        self.assertFalse(self.base.compare_output(
            actual_no_dots, 'hello ... world', disable_fuzzy=True))


class TestEndToEndV2Doc(unittest.TestCase):
    """拿 v2 doc 真实跑解析,验证契约回归(不执行 subprocess)。"""

    DOC_PATH = (
        Path(__file__).resolve().parent.parent
        / 'projects' / 'ms-swift' / 'docs' / 'Quick-start-Ascend.md'
    )

    def test_v2_doc_parses(self):
        if not self.DOC_PATH.exists():
            self.skipTest(f'{self.DOC_PATH} not present')
        with open(self.DOC_PATH) as f:
            text = f.read()
        commands, results = _Bare().parse(text)
        # v2 文档包含 9 个 commands + 6 个 expected outputs
        self.assertEqual(len(commands), 9)
        self.assertEqual(len(results), 6)
        # 两个 hidden setup（uninstall 无 store + upstream_ref 有 store）
        hidden_setups = [
            c for c in commands
            if isinstance(c, SetupCommand) and c.hidden
        ]
        self.assertEqual(len(hidden_setups), 2)
        # 注释内的 upstream_ref 应被 load 引用
        source_cmd = next(
            c for c in commands
            if isinstance(c, TestCommand) and c.id == 'swift-install-source'
        )
        self.assertEqual(source_cmd.load, (('upstream_ref', 'ref'),))
        # infer 引用 checkpoint
        infer_cmd = next(
            c for c in commands
            if isinstance(c, TestCommand) and c.id == 'infer'
        )
        self.assertEqual(infer_cmd.load, (('checkpoint', 'ckpt'),))


if __name__ == '__main__':
    unittest.main()