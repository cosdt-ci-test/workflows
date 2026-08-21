"""Markdown 文档标签测试基类：模板方法 pre_process -> parse -> execute -> post_process。

契约定义在 docs/markdown_doc_test_label.md：每个代码块 info string 上挂
``#test`` / ``#test-result`` / ``#test-setup`` 标签，加 ``id=`` / ``store=`` /
``load='x>>y'`` / ``fuzzy='xxx'`` 参数。本模块把这条契约落成可执行框架：

* 解析（``parse`` -> mistune AST + 内层 fence 回扫 + ``_parse_block`` ->
  ``_fold``）返回 ``SetupCommand`` / ``TestCommand`` 主序列 +
  ``TestExpectedOutput`` 注册表；
* 校验（``_validate``）嵌入解析，规则 2/5/7/10/11 违规抛 ``LabelSpecError``；
* 执行（``execute``）按文档顺序跑命令，``SetupCommand`` 捕获 stdout 进
  ``captures``，``TestCommand`` 替换 ``<local>`` 占位符后跑，再按 id 查
  ``TestExpectedOutput`` 比对；
* 日志（``log`` / ``log_block``）统一格式，失败时 dump 失败命令本身 + 实际输出。

解析器借助 mistune v3 的 AST 处理"外层 fence + HTML 注释范围"，对
``block_html.raw`` 内部的 fence 再做一次行扫描以救回被 CommonMark HTML block
折叠掉的注释内 setup（v2 契约里 ``<!-- ```shell #test-setup ... ``` -->``
是被支持的形态，但任何标准 markdown 库都不会单独把内部 fence 切出来）。

子类只需实现 ``pre_process``（取 markdown 文本）和 ``post_process``（清理），
可覆盖 ``DEFAULT_COMMAND_TIMEOUT``（所有 subprocess 共用的超时秒数，默认 1800）。
"""

from __future__ import annotations

import os
import re
import subprocess
import time
import unittest
import urllib.error
import urllib.request
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mistune

# ============================================================
# 数据结构：每个标签一个 schema
# ============================================================


@dataclass(frozen=True)
class SetupCommand:
    """#test-setup 命令：跑 + 捕获 stdout。

    ``hidden=True`` 表示该 setup 块在 HTML 注释内（页面上不渲染），但仍参与执行
    与 store 链（契约规则 10）。

    ``__post_init__`` 在构造期校验字段:cmd 非空、store 非空字符串。这是
    "不可变契约"——runner 拿到 dataclass 时所有字段必然合法,无须再防御。
    """

    cmd: str
    store: str | None
    hidden: bool

    def __post_init__(self) -> None:
        if not self.cmd:
            raise LabelSpecError(
                'SetupCommand.cmd must be a non-empty string'
            )
        if self.store is not None and not self.store:
            raise LabelSpecError(
                'SetupCommand.store must be None or a non-empty string'
            )


@dataclass(frozen=True)
class TestCommand:
    """#test 命令：跑 + 比对期望。注意：不携带 expected。

    比对时由 runner 按 ``id`` 到 ``TestExpectedOutput`` 注册表查期望输出。
    ``__post_init__`` 校验必填字段 + load 元组形态。
    """

    id: str
    cmd: str
    language: str
    load: tuple = ()  # ((store_var, local_name), ...)

    def __post_init__(self) -> None:
        if not self.id:
            raise LabelSpecError('TestCommand.id must be non-empty')
        if not self.cmd:
            raise LabelSpecError('TestCommand.cmd must be non-empty')
        if not self.language:
            raise LabelSpecError('TestCommand.language must be non-empty')
        # load 是 ((store_var, local_name), ...) 形态
        for i, item in enumerate(self.load):
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not all(isinstance(x, str) and x for x in item)
            ):
                raise LabelSpecError(
                    f'TestCommand.load[{i}] must be a (str, str) tuple; '
                    f'got {item!r}'
                )


@dataclass(frozen=True)
class TestExpectedOutput:
    """#test-result 命令：期望输出，放在注册表，不进主序列。

    ``body`` 里的 ``<local>`` 占位符在比对前由 ``substitute_placeholders``
    替换（用同一份 captures）；``fuzzy`` 是非贪婪匹配占位符集合（默认
    ``...``）。支持多个：每个 placeholder 都是"非贪婪通配"的同义词，
    任意一个出现在 expected 里都按通配处理。
    ``disable_fuzzy=True`` 时所有占位符(含默认 ``...``)按字面匹配。
    ``__post_init__`` 校验:必填字段非空、fuzzy 项为非空字符串、
    ``disable_fuzzy=True`` 时 fuzzy 必空(解析期规则 16 已挡,这里是防御性兜底)。
    """

    id: str
    body: str
    fuzzy: tuple = ()  # placeholder 字符串元组;空表示只用默认 '...'
    disable_fuzzy: bool = False  # True 时关闭所有非贪婪匹配
    load: tuple = ()  # ((store_var, local_name), ...)

    def __post_init__(self) -> None:
        if not self.id:
            raise LabelSpecError('TestExpectedOutput.id must be non-empty')
        if not self.body:
            raise LabelSpecError('TestExpectedOutput.body must be non-empty')
        for i, p in enumerate(self.fuzzy):
            if not isinstance(p, str) or not p:
                raise LabelSpecError(
                    f'TestExpectedOutput.fuzzy[{i}] must be a non-empty '
                    f'string; got {p!r}'
                )
        if self.disable_fuzzy and self.fuzzy:
            raise LabelSpecError(
                'TestExpectedOutput.disable_fuzzy=True conflicts with '
                f'non-empty fuzzy={self.fuzzy!r}'
            )
        for i, item in enumerate(self.load):
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not all(isinstance(x, str) and x for x in item)
            ):
                raise LabelSpecError(
                    f'TestExpectedOutput.load[{i}] must be a (str, str) '
                    f'tuple; got {item!r}'
                )


class LabelSpecError(Exception):
    """契约违规。错误信息里包含足够的上下文（id / load value / 当前已知
    store 集合）以便在文档里直接定位出错的代码块。"""

# ============================================================
# 模块级工具
# ============================================================


def _rescan_fences(raw: str) -> list[tuple[str, str]]:
    """从 ``block_html.raw`` 里切出所有 ``` 围栏，返回 ``[(info, body), ...]``。

    例：

        输入 raw（mistune 给的 ``block_html.raw`` 字段）::

            <!--
            ```shell #test-setup store="x"
            echo captured
            ```
            some prose
            ```shell #test-setup store="y"
            echo twice
            ```
            -->

        返回 ``[
            ('shell #test-setup store="x"', 'echo captured'),
            ('shell #test-setup store="y"', 'echo twice'),
        ]`` —— 把注释内被吃掉的两个 fence 拆出来。

    未闭合抛 ``LabelSpecError``（保留契约内的错误类型，避免文档作者看一堆不同的异常类）。
    """
    out: list[tuple[str, str]] = []
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith('```'):
            info = lines[i].lstrip()[3:].strip()
            j = i + 1
            body_lines: list[str] = []
            closed = False
            while j < len(lines):
                if lines[j].lstrip().startswith('```'):
                    out.append((info, '\n'.join(body_lines)))
                    i = j + 1
                    closed = True
                    break
                body_lines.append(lines[j])
                j += 1
            if not closed:
                raise LabelSpecError(
                    f'unclosed fence inside HTML comment: '
                    f'info={info!r} body_head={body_lines[:1]!r}'
                )
        else:
            i += 1
    return out

# ============================================================
# 基类：模板方法模式
# ============================================================


# mistune 模块单例：renderer='ast' 输出 dict 流；plugins=[] 禁掉所有扩展以免
# 改变 fence 切分行为。测试运行一个 doc 只调用一次，重复实例化浪费。
_MD_AST = mistune.create_markdown(renderer='ast', plugins=[])


class MarkdownDocTestBase(ABC):
    """抽象基类：模板方法 ``pre_process`` -> ``parse`` -> ``execute`` -> ``post_process``。

    子类可以覆盖实现：
        ``pre_process()`` -> ``str``    取 markdown 文本
        ``post_process()`` -> ``None``   清理 / 上报

    子类可覆盖：
        ``DEFAULT_COMMAND_TIMEOUT``      所有 subprocess 共用的超时（秒），默认 1800

    用法（在 unittest TestCase 子类里）：
        ``@unittest.skipIf(...)``        按项目门控
        ``def test_runs_doc(self):``
            ``self.run_template()``      模板方法入口
    """

    DEFAULT_COMMAND_TIMEOUT: int = 1800  # 30 分钟;训练类长命令子类需覆盖。

    # ============================================================
    # 私有：解析器内部
    # ============================================================

    _LABEL_TEST = '#test'
    _LABEL_TEST_RESULT = '#test-result'
    _LABEL_TEST_SETUP = '#test-setup'
    _KNOWN_LABELS = (_LABEL_TEST, _LABEL_TEST_RESULT, _LABEL_TEST_SETUP)
    # 契约承认的参数名(typo 早死):fuzzy / disable_fuzzy 仅 #test-result 允许,
    # 但 _KNOWN_PARAMS 不区分标签——标签限制在 _parse_block 里另行校验。
    _KNOWN_PARAMS = frozenset({'id', 'store', 'load', 'fuzzy', 'disable_fuzzy'})
    # 默认非贪婪占位符:不指定 fuzzy= 时,该 placeholder 永远生效。
    # _parse_block 在 fuzzies 为空时自动把这一项放进 fuzzy 字段。
    _DEFAULT_FUZZY_PLACEHOLDER = '...'
    # 契约当前只支持 shell。其它语言(text / console / python 等)直接报
    # 规则 7 违规。要新增语言时,先在执行器侧落地 selector,再加进 tuple。
    _KNOWN_LANGUAGES = ('shell',)

    # 无值 flag 参数（不带 ``=value``）。识别后值用 ``['1']`` 占位,
    # 实际语义在 _parse_block / compare_output 里按 key 名判定。
    _FLAG_PARAMS = ('disable_fuzzy',)

    @staticmethod
    def _parse_params(param_strs: list[str]) -> dict[str, list[str]]:
        """解析 ``key='value'`` / ``key="value"`` tokens 成多值 dict。
        也接受无值 flag（如 ``disable_fuzzy``），值用 ``['1']`` 占位。"""
        params: dict[str, list[str]] = {}
        for tok in param_strs:
            if '=' not in tok:
                if tok in MarkdownDocTestBase._FLAG_PARAMS:
                    params.setdefault(tok, ['1'])
                    continue
                raise LabelSpecError(
                    f"invalid parameter (no '='): {tok!r}"
                )
            key, _, value = tok.partition('=')
            # len(value) < 2 意味着只有引号
            if len(value) < 2 or not (
                (value.startswith("'") and value.endswith("'"))
                or (value.startswith('"') and value.endswith('"'))
            ):
                raise LabelSpecError(
                    f'parameter value must be single- or double-quoted: {tok!r}'
                )
            params.setdefault(key, []).append(value[1:-1])
        return params

    @staticmethod
    def _parse_load_value(value: str) -> tuple[str, str]:
        """``xxx>>yyy`` -> ``(xxx, yyy)``。"""
        if '>>' not in value:
            raise LabelSpecError(
                f"load= value must be in xxx>>yyy form: {value!r}"
            )
        parts = value.split('>>')
        if len(parts) != 2 or not all(parts):
            raise LabelSpecError(
                f"load= value must be exactly 'store>>placeholder': {value!r}"
            )
        return parts[0], parts[1]
    
    def _scan_blocks(self, text: str) -> list[dict]:
        """识别代码块，或者html注释（<!-- -->）里的代码块
        """
        # mistune 的 Markdown.__call__ 没有精确的类型注解（返回 list[dict]），
        # 静态检查工具看不到节点字段，这里用 Any 接收后按 dict 访问。
        ast: Any = _MD_AST(text)
        blocks: list[dict] = []
        for node in ast:
            if node['type'] == 'block_html':
                raw = node['raw']
                if not raw.lstrip().startswith('<!--'):
                    continue
                # raw 字段保留尾部换行（mistune 直接复制原文本段），
                # bash -c 执行多一个 \n 无影响，但保持 body 不带尾换行
                # 让单元测试断言更直观（cmd=='<expected lines>'）。
                for info, body in _rescan_fences(raw):
                    blocks.append({
                        'info': info,
                        'body': body.rstrip('\n'),
                        'hidden': True,
                    })
            elif node['type'] == 'block_code':
                attrs = node.get('attrs') or {}
                info = attrs.get('info', '') or ''
                blocks.append({
                    'info': info,
                    'body': node['raw'].rstrip('\n'),
                    'hidden': False,
                })
        return blocks

    def _parse_block(self, block: dict) -> dict | None:
        """解析单个代码块的 info string。返回 ``None`` 表示无标签（规则 9 跳过）。"""
        info = block['info']
        parts = info.split()
        if not parts:
            return None

        label_idx = -1
        label: str | None = None
        for k, p in enumerate(parts):
            if p in self._KNOWN_LABELS:
                label_idx = k
                label = p
                break
        if label_idx < 0:
            return None

        # language：标签前的第一个 token（若标签不在第一位）
        language = parts[0] if label_idx > 0 else None

        param_strs = parts[label_idx + 1:]
        params = self._parse_params(param_strs)

        ids = params.get('id', [])
        if len(ids) > 1:
            raise LabelSpecError(
                'duplicate id= parameter on the same block'
            )
        block_id = ids[0] if ids else None

        stores = params.get('store', [])
        if len(stores) > 1:
            raise LabelSpecError(
                'duplicate store= parameter on the same block'
            )
        store = stores[0] if stores else None
        if store is not None and label != self._LABEL_TEST_SETUP:
            raise LabelSpecError(
                f"store= is only valid on #test-setup, got {label}"
            )

        fuzzies = params.get('fuzzy', [])
        # fuzzy= 仅 #test-result 允许:#test 是命令体,#test-setup 是 setup
        # 命令,body 不参与模糊匹配。
        if fuzzies and label != self._LABEL_TEST_RESULT:
            raise LabelSpecError(
                f"fuzzy= is only valid on #test-result, got {label}"
            )
        # 同一个 placeholder 重复出现算契约违规:用户多半是笔误。
        if len(fuzzies) != len(set(fuzzies)):
            raise LabelSpecError(
                f'duplicate fuzzy placeholder: {fuzzies!r}'
            )

        disable_fuzzy = bool(params.get('disable_fuzzy'))
        # fuzzy= 与 disable_fuzzy 互斥:前者要占位符,后者明确取消,
        # 文档规则 3 明确写一起就报错。
        if disable_fuzzy and fuzzies:
            raise LabelSpecError(
                "disable_fuzzy conflicts with fuzzy=: pick one"
            )
        # disable_fuzzy 仅 #test-result 允许(沿用 fuzzy 的标签限制)。
        if disable_fuzzy and label != self._LABEL_TEST_RESULT:
            raise LabelSpecError(
                f"disable_fuzzy is only valid on #test-result, got {label}"
            )

        loads: list[tuple[str, str]] = []
        for raw in params.get('load', []):
            loads.append(self._parse_load_value(raw))

        # 拒绝未知参数（typo 早死）
        unknown = set(params) - self._KNOWN_PARAMS
        if unknown:
            raise LabelSpecError(
                f'unknown parameter(s): {sorted(unknown)}'
            )

        # 注入默认 placeholder:不写 fuzzy= 且没 disable_fuzzy 的 #test-result,
        # fuzzy 字段自动含 '...'。把 '...' 当成"占位符集合的一员"统一处理,
        # dataclass 自描述一个块生效的全部 placeholder,compare_output 拿到
        # 这个字段就能直接跑（不必再内置默认）。
        # 注意:fuzzy= 与 disable_fuzzy 互斥(规则 16),此处 disable_fuzzy 为真
        # 时 fuzzies 必空,所以"fuzzies 空 && disable_fuzzy 真"等价于"关闭",
        # 不补默认。
        if (
            label == self._LABEL_TEST_RESULT
            and not fuzzies
            and not disable_fuzzy
        ):
            fuzzies = [self._DEFAULT_FUZZY_PLACEHOLDER]

        return {
            'label': label,
            'id': block_id,
            'language': language,
            'load': tuple(loads),
            'store': store,
            'fuzzy': tuple(fuzzies),
            'disable_fuzzy': disable_fuzzy,
            'body': block['body'],
            'hidden': block['hidden'],
        }

    def _validate(self, parsed: list[dict]) -> None:
        """规则 2/5/7/10/11 校验。任一违规抛 ``LabelSpecError``。"""
        # 规则 10：HTML 注释内仅允许 #test-setup
        for p in parsed:
            if p['hidden'] and p['label'] != self._LABEL_TEST_SETUP:
                raise LabelSpecError(
                    f'HTML comment can only contain #test-setup, '
                    f'got {p["label"]}'
                )

        # 规则 7:#test / #test-setup 必须指定 language,且必须在契约白名单
        # 内(当前只支持 shell)
        for p in parsed:
            if p['label'] not in (self._LABEL_TEST, self._LABEL_TEST_SETUP):
                continue
            if not p['language']:
                raise LabelSpecError(
                    f"{p['label']} block must specify a language (rule 7); "
                    f'block body={p["body"]!r}'
                )
            if p['language'] not in self._KNOWN_LANGUAGES:
                raise LabelSpecError(
                    f"{p['label']} block language {p['language']!r} not "
                    f"supported (rule 7); supported={self._KNOWN_LANGUAGES}; "
                    f'block body={p["body"]!r}'
                )

        # 规则 2：同 type id 唯一
        seen_ids: dict[str, set[str]] = {
            label: set() for label in self._KNOWN_LABELS
        }
        for p in parsed:
            if not p['id']:
                continue
            bucket = seen_ids[p['label']]
            if p['id'] in bucket:
                raise LabelSpecError(
                    f"duplicate id {p['id']!r} in {p['label']} blocks"
                )
            bucket.add(p['id'])

        # 规则 5：#test 与 #test-result 按 id 配对
        result_ids = {
            p['id'] for p in parsed
            if p['label'] == self._LABEL_TEST_RESULT and p['id']
        }
        for p in parsed:
            if p['label'] == self._LABEL_TEST:
                if not p['id']:
                    raise LabelSpecError('#test block must have id=')
                if p['id'] not in result_ids:
                    raise LabelSpecError(
                        f"#test id={p['id']!r} has no matching #test-result"
                    )
        for p in parsed:
            if p['label'] == self._LABEL_TEST_RESULT and not p['id']:
                raise LabelSpecError('#test-result block must have id=')

        # 规则 11：load 引用必须晚于 store（按文档顺序）
        # HTML 注释内的 #test-setup 同样计入 seen_stores，因为它们执行时
        # 仍会写 captures。
        seen_stores: set[str] = set()
        for p in parsed:
            if p['label'] == self._LABEL_TEST_SETUP and p['store']:
                seen_stores.add(p['store'])
            elif p['label'] in (self._LABEL_TEST, self._LABEL_TEST_RESULT):
                for store_var, _local in p['load']:
                    if store_var not in seen_stores:
                        raise LabelSpecError(
                            f"load={store_var!r} references store that "
                            f"hasn't appeared earlier in document "
                            f'(seen_stores so far: {sorted(seen_stores)})'
                        )

    def _fold(
        self, parsed: list[dict]
    ) -> tuple[list, dict]:
        """按 type 生成 SetupCommand / TestCommand / TestExpectedOutput。

        ``TestExpectedOutput`` 进 dict 备查，不进主序列。
        """
        commands: list = []
        results: dict = {}
        for p in parsed:
            if p['label'] == self._LABEL_TEST_SETUP:
                commands.append(SetupCommand(
                    cmd=p['body'],
                    store=p['store'],
                    hidden=p['hidden'],
                ))
            elif p['label'] == self._LABEL_TEST:
                commands.append(TestCommand(
                    id=p['id'],
                    cmd=p['body'],
                    language=p['language'],
                    load=p['load'],
                ))
            elif p['label'] == self._LABEL_TEST_RESULT:
                if p['id'] in results:
                    # 规则 2 已挡，这里防御性
                    raise LabelSpecError(
                        f"duplicate #test-result id={p['id']!r}"
                    )
                results[p['id']] = TestExpectedOutput(
                    id=p['id'],
                    body=p['body'],
                    fuzzy=p['fuzzy'],
                    disable_fuzzy=p['disable_fuzzy'],
                    load=p['load'],
                )
        return commands, results

    # ============================================================
    # 私有：单步执行细节
    # ============================================================

    def _run_one(self, cmd, results, env, cwd, timeout, idx):
        if isinstance(cmd, SetupCommand):
            rc, out = self.run_command(cmd.cmd, env, cwd, timeout)
            if rc != 0:
                raise AssertionError(
                    f'setup command failed (rc={rc}); see CMD stderr above'
                )
            if cmd.store:
                self._captures[cmd.store] = out
                self.log(
                    f'[Step {idx}] captured {cmd.store!r} ({len(out)}B)'
                )
            return

        if isinstance(cmd, TestCommand):
            expected_obj = results.get(cmd.id)
            if expected_obj is None:
                # _validate 已挡；这里是防御性
                raise AssertionError(
                    f'no #test-result for id={cmd.id!r}'
                )
            actual_cmd = self.substitute_placeholders(
                cmd.cmd, cmd.load, self._captures
            )
            expected_body = self.substitute_placeholders(
                expected_obj.body, expected_obj.load, self._captures
            )
            rc, actual = self.run_command(actual_cmd, env, cwd, timeout)
            if rc != 0:
                raise AssertionError(
                    f'test command failed (rc={rc}); see CMD stderr above'
                )
            if self.compare_output(
                actual, expected_body,
                fuzzy=expected_obj.fuzzy,
                disable_fuzzy=expected_obj.disable_fuzzy,
            ):
                self.log(f'[Step {idx}] test id={cmd.id!r}: OK')
                self.log_block('expected', expected_body.splitlines())
                self.log_block('actual', actual.splitlines())
                return

            self.log(f'[Step {idx}] test id={cmd.id!r}: MISMATCH')
            self.log_block('expected', expected_body.splitlines(), cap=0)
            self.log_block('actual', actual.splitlines(), cap=0)
            raise AssertionError(
                f'test id={cmd.id!r} output mismatch; see summary above'
            )

        raise AssertionError(f'unknown command type: {type(cmd).__name__}')

    @staticmethod
    def _cmd_label(cmd) -> str:
        if isinstance(cmd, SetupCommand):
            return f'setup store={cmd.store!r}'
        if isinstance(cmd, TestCommand):
            return f'test id={cmd.id!r}'
        return f'unknown:{type(cmd).__name__}'

    # ============================================================
    # 公开：模板方法入口 + 子类钩子 + 框架实现
    # ============================================================
    def pre_process(self) -> str:
        """从 ``MONITORED_DOC_URL`` 拉 doc 文本。

        失败抛 ``RuntimeError``（不是 ``SkipTest``），让 CI 显式失败。无本地
        fallback：stale 本地副本会与触发源失同步。
        """
        url = os.environ.get('MONITORED_DOC_URL')
        if not url:
            raise RuntimeError(
                'MONITORED_DOC_URL unset - test must run inside the '
                'workflow which sets it; no local fallback by design.'
            )

        # urllib 默认无 timeout：网络抖动可能挂死。每 30s timeout 重试 1 次。
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'cosdt-ci-test/quick-start-v2'},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return resp.read().decode('utf-8')
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = e
                self.log(
                    f'fetch attempt {attempt+1}/2 failed for {url}: {e!r}'
                )
                time.sleep(2)
        raise RuntimeError(
            f'failed to fetch {url} after 2 attempts: {last_err!r}'
        )

    def post_process(self) -> None:
        """清理临时文件 / 上传 artifact / 关闭连接。"""
        return

    def execute(self, commands: list, results: dict) -> None:
        """按顺序跑命令。失败抛 unittest 断言异常 + dump 失败命令本身 + 实际输出。"""
        self._captures: dict = {}
        env = os.environ.copy()
        cwd = Path.cwd()
        timeout = self.DEFAULT_COMMAND_TIMEOUT

        for i, cmd in enumerate(commands):
            label = self._cmd_label(cmd)
            self.log(
                f'[Step {i}/{len(commands)-1}] {label} timeout={timeout}s'
            )
            try:
                self._run_one(cmd, results, env, cwd, timeout, i)
            except unittest.SkipTest:
                raise
            except Exception as e:
                # 失败时 dump 失败命令本身 + 实际输出
                self.log(f'[Step {i}] FAILED: {e}')
                self.log_block('cmd', cmd.cmd.splitlines(), cap=0)
                raise

    def run_template(self) -> None:
        """``pre_process`` -> ``parse`` -> ``execute`` -> ``post_process``。

        ``post_process`` 在 ``finally`` 里调用，确保 ``execute`` 抛错时也跑清理。
        注意：``run_template`` 不负责环境准备——子类自行负责。
        """
        text = self.pre_process()
        commands, test_expected_results = self.parse(text)
        self.log(
            f'parsed {len(commands)} commands, '
            f'{len(test_expected_results)} #test-result blocks'
        )
        try:
            self.execute(commands, test_expected_results)
        finally:
            self.post_process()

    def run_command(
        self, cmd: str, env: dict, cwd, timeout: int
    ) -> tuple[int, str]:
        """``bash -c`` + 强制 flush + 错误时 dump stderr 全量（<= 256 KB）。

        stdout 错误路径 dump 头 2000 + 尾 2000；stderr 含错误标记（``[ERROR]`` /
        ``Traceback (most recent call last)`` / ``applicaiton exception`` /
        ``ERR99999``）时 dump 全部（<= 256 KB），因为错误标记常在 traceback 中段。
        """
        self.log(f'CMD start (timeout={timeout}s): {cmd[:2000]}')
        t0 = time.time()
        proc = subprocess.run(
            ['bash', '-c', cmd],
            env=env,
            cwd=cwd,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        out = proc.stdout.decode('utf-8', errors='replace')
        err = proc.stderr.decode('utf-8', errors='replace')
        elapsed = time.time() - t0
        self.log(
            f'CMD done in {elapsed:.1f}s rc={proc.returncode} '
            f'(stdout={len(out)}B stderr={len(err)}B)'
        )

        error_markers = (
            '[ERROR]', 'Traceback (most recent call last)',
            'applicaiton exception', 'ERR99999',
        )
        stderr_has_error = any(m in err for m in error_markers)
        if proc.returncode != 0 or stderr_has_error \
                or (out.strip() == '' and err.strip() != ''):
            head_err = err[:2000]
            tail_err = err[-2000:] if len(err) > 2000 else ''
            if stderr_has_error and len(err) <= 256_000:
                self.log(
                    f'CMD stderr (full, {len(err)}B):\n{err.rstrip()}'
                )
            else:
                if head_err:
                    self.log(f'CMD stderr (head):\n{head_err.rstrip()}')
                if tail_err and tail_err != head_err:
                    self.log(f'CMD stderr (tail):\n{tail_err.rstrip()}')
            head_out = out[:2000]
            tail_out = out[-2000:] if len(out) > 2000 else ''
            if head_out:
                self.log(f'CMD stdout (head):\n{head_out.rstrip()}')
            if tail_out and tail_out != head_out:
                self.log(f'CMD stdout (tail):\n{tail_out.rstrip()}')

        return proc.returncode, out

    def parse(self, text: str) -> tuple[list, dict]:
        """解析 + 校验。

        失败抛 ``LabelSpecError``；返回 ``(主命令序列, TestExpectedOutput 注册表)``。
        主序列只含 ``SetupCommand`` / ``TestCommand``；``TestExpectedOutput`` 进
        dict，runner 执行 ``TestCommand`` 时按 ``id`` 取对应期望。
        """
        raw_blocks = self._scan_blocks(text)
        parsed = [self._parse_block(b) for b in raw_blocks]
        # 过滤无标签的普通块（规则 9）
        parsed = [p for p in parsed if p is not None]
        self._validate(parsed)
        return self._fold(parsed)
    
    def substitute_placeholders(
        self, text: str, load: tuple, captures: dict
    ) -> str:
        """把 ``<local>`` 替换为 ``captures[store_var]``。

        ``store_var`` 不在 ``captures`` 时保留占位符不动（规则 11 已在解析期校验
        load 引用必须晚于 store，这里理论上必然命中；保留字面让 bash 报错比静默
        替换成空字符串更易诊断）。
        """
        for store_var, local in load:
            if store_var in captures:
                text = text.replace(f'<{local}>', captures[store_var])
        return text

    def compare_output(
        self, actual: str, expected: str,
        fuzzy: str | tuple[str, ...] = (),
        disable_fuzzy: bool = False,
    ) -> bool:
        """``actual`` vs ``expected`` 正则匹配；``fuzzy`` 列出所有非贪婪
        placeholder，每个出现于 expected 时按跨行非贪婪通配（``re.DOTALL``）。

        调用方负责提供完整 placeholder 集合——runner 从 ``TestExpectedOutput.fuzzy``
        直接拿，单元测试调用方自行决定要传什么。常见用法:

            fuzzy=('...',)           # 默认,空 fuzzies 已自动注入
            fuzzy=('xxx', 'yyy')     # 多种自定义占位符并用
            disable_fuzzy=True       # 关闭所有非贪婪匹配,按字面匹配
        """
        if disable_fuzzy:
            # 完全字面匹配:不对 expected 做任何 placeholder 切分。
            return re.search(re.escape(expected), actual, re.DOTALL) is not None
        # 把 expected 按所有 placeholder 切成段;段与段之间用非贪婪跨行匹配
        # 连接。``str.split(sep)`` 只支持单 sep,所以用正则 split 一次搞定
        # 多种 placeholder。注意 placeholder 顺序无关:split 按出现位置切。
        placeholders: list[str]
        if isinstance(fuzzy, str):
            placeholders = [fuzzy]
        else:
            placeholders = list(fuzzy)
        if not placeholders:
            # 兜底:空 fuzzy + 非 disable_fuzzy = 字面匹配,等价于 disable_fuzzy
            return re.search(re.escape(expected), actual, re.DOTALL) is not None
        sep_pattern = '|'.join(re.escape(p) for p in placeholders)
        parts = re.split(sep_pattern, expected)
        pattern = r'.*?'.join(re.escape(part) for part in parts)
        return re.search(pattern, actual, re.DOTALL) is not None

    def log(self, msg: str) -> None:
        """``[HH:MM:SS.mmm] {msg}``，``flush=True``。"""
        ms = int(time.time() * 1000) % 1000
        ts = time.strftime('%H:%M:%S') + f'.{ms:03d}'
        print(f'[{ts}] {msg}', flush=True)

    def log_block(self, label: str, lines, cap: int = 30) -> None:
        """块日志：OK 路径 ``cap`` 行 head+tail，MISMATCH 路径 ``cap=0`` 不截断。

        每行带行号（``1.`` / ``2.`` / ``154.``）便于对照；超长输出 dump 头
        ``cap/2`` + 尾 ``cap/2``，中间 ``... [N line(s) elided] ...``。
        """
        lines = list(lines)
        self.log(f'  --- {label} (head + tail if huge) ---')
        if cap and len(lines) > cap:
            half = cap // 2
            for i, ln in enumerate(lines[:half], 1):
                self.log(f'  {i:>3}. {ln}')
            elided = len(lines) - 2 * half
            self.log(f'  ... [{elided} line(s) elided] ...')
            tail_start = len(lines) - half + 1
            for offset, ln in enumerate(lines[-half:]):
                self.log(f'  {tail_start + offset:>3}. {ln}')
            return
        for i, ln in enumerate(lines, 1):
            self.log(f'  {i:>3}. {ln}')

