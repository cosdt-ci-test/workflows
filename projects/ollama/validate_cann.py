"""Exercise the fork's CANN path on a real runner without a backend override."""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener, urlopen


FORK = 'https://github.com/lltiaor/ollama.git'
MODEL_URL = (
    'https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/master/'
)
MODELS = {
    'q4_0': (
        428730208,
        '7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed',
    ),
    'q8_0': (
        675710816,
        'ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e',
    ),
    'fp16': (
        1266425696,
        '8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc',
    ),
}


def run(argv, *, timeout=120, **kwargs):
    print('+', *argv, flush=True)
    return subprocess.run(argv, check=True, timeout=timeout, **kwargs)


def valid_file(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest() == digest


def download(url, target, size, digest):
    if not valid_file(target, size, digest):
        with tempfile.TemporaryDirectory(dir=target.parent) as temp:
            partial = Path(temp) / target.name
            with urlopen(url, timeout=120) as source, partial.open('wb') as output:
                shutil.copyfileobj(source, output)
            if not valid_file(partial, size, digest):
                raise RuntimeError(f'Checksum or size mismatch: {target.name}')
            partial.replace(target)
    return target


def go_toolchain(source, cache, work, env):
    arch = {'aarch64': 'arm64', 'x86_64': 'amd64'}[platform.machine()]
    match = re.search(
        r'^go (\d+\.\d+(?:\.\d+)?)$', (source / 'go.mod').read_text(), re.M
    )
    if match is None:
        raise RuntimeError('Cannot read Go version from go.mod')
    version = 'go' + match[1]
    if version.count('.') == 1:
        version += '.0'
    filename = f'{version}.linux-{arch}.tar.gz'
    archive = cache / filename
    with urlopen('https://go.dev/dl/?mode=json&include=all', timeout=60) as response:
        package = next(
            entry
            for release in json.load(response)
            if release['version'] == version
            for entry in release['files']
            if entry['filename'] == filename
        )
    download(
        'https://go.dev/dl/' + filename, archive, package['size'], package['sha256']
    )
    run(['tar', '-xzf', str(archive), '-C', str(work)])
    env['PATH'] = str(work / 'go/bin') + os.pathsep + env['PATH']
    run(['go', 'version'], env=env)


def model_runner_mask(server_pid):
    """Read the actual model runner's mask, not a discovery child's mask."""
    children = set()
    for task in Path(f'/proc/{server_pid}/task').glob('*/children'):
        with contextlib.suppress(FileNotFoundError, ProcessLookupError):
            children.update(task.read_text().split())
    matches = []
    for pid in children:
        proc = Path('/proc') / pid
        with contextlib.suppress(FileNotFoundError, ProcessLookupError):
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            if (
                args
                and Path(os.fsdecode(args[0])).name == 'llama-server'
                and b'--model' in args
            ):
                variables = dict(
                    entry.split(b'=', 1)
                    for entry in (proc / 'environ').read_bytes().split(b'\0')
                    if b'=' in entry
                )
                matches.append(
                    variables.get(b'ASCEND_RT_VISIBLE_DEVICES', b'').decode()
                )
    if len(matches) != 1:
        raise RuntimeError(f'Expected one live model runner, found {len(matches)}')
    return matches[0]


def run_case(
    source, work, reports, env, commit, name, model, visible, expected, spread=False
):
    case_dir = work / name
    case_dir.mkdir()
    modelfile = case_dir / 'Modelfile'
    modelfile.write_text(f'FROM {model}\n', encoding='utf-8')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    host = f'http://127.0.0.1:{port}'
    env = dict(
        env,
        OLLAMA_HOST=host,
        OLLAMA_MODELS=str(case_dir / 'models'),
        OLLAMA_DEBUG='1',
        OLLAMA_KEEP_ALIVE='5m',
        OLLAMA_NUM_PARALLEL='1',
        OLLAMA_SCHED_SPREAD='1' if spread else '0',
        ASCEND_RT_VISIBLE_DEVICES=visible,
    )
    opener = build_opener(ProxyHandler({}))

    def request(path, payload=None, timeout=5):
        data = None if payload is None else json.dumps(payload).encode()
        req = Request(
            host + path, data=data, headers={'Content-Type': 'application/json'}
        )
        with opener.open(req, timeout=timeout) as response:
            return json.load(response)

    binary = str(source / 'ollama')
    log_path = reports / f'{name}-server.log'
    with log_path.open('wb') as output:
        server = subprocess.Popen(
            [binary, 'serve'],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline and server.poll() is None:
                try:
                    version = request('/api/version')['version']
                except (URLError, TimeoutError):
                    time.sleep(1)
                    continue
                if version != commit:
                    raise RuntimeError(
                        f'Server version {version} differs from build {commit}'
                    )
                break
            else:
                raise RuntimeError('Ollama did not become ready')

            run(
                [binary, 'create', 'cann-check', '-f', str(modelfile)],
                env=env,
                timeout=300,
            )
            options = {'num_ctx': 1024, 'num_predict': 32, 'temperature': 0}
            if name == 'a2-auto':
                # Select the second visible NPU; it becomes CANN0 in the child.
                options['main_gpu'] = 1
            response = request(
                '/api/chat',
                {
                    'model': 'cann-check',
                    'stream': False,
                    'options': options,
                    'messages': [
                        {
                            'role': 'user',
                            'content': 'Introduce yourself in one sentence.',
                        }
                    ],
                },
                timeout=300,
            )
            loaded = request('/api/ps')
            for suffix, data in [('chat', response), ('ps', loaded)]:
                (reports / f'{name}-{suffix}.json').write_text(
                    json.dumps(data, indent=2)
                )

            mask = model_runner_mask(server.pid)
            log = log_path.read_text(encoding='utf-8', errors='replace')
            latest = re.split(r'load_tensors:\s+loading model tensors', log)[-1]
            offload = re.findall(r'offloaded\s+(\d+)/(\d+)\s+layers to GPU', latest)
            buffers = {
                int(card): float(size)
                for card, size in re.findall(
                    r'\bCANN(\d+)\s+model buffer size\s*=\s*([\d.]+)\s*MiB', latest
                )
            }
            evidence = {
                'case': name,
                'model_file': model.name,
                'model_sha256': MODELS[model.stem.split('-')[-1]][1],
                'ollama_backend_override': None,
                'server_mask': visible,
                'model_runner_mask': mask,
                'offloaded_layers': offload[-1] if offload else None,
                'cann_buffers_mib': buffers,
                'eval_count': response.get('eval_count', 0),
                'size_vram': next(
                    (
                        m.get('size_vram', 0)
                        for m in loaded['models']
                        if m['name'] in {'cann-check', 'cann-check:latest'}
                    ),
                    0,
                ),
            }
            (reports / f'{name}-evidence.json').write_text(
                json.dumps(evidence, indent=2)
            )
            print(json.dumps(evidence), flush=True)
            if (
                response.get('done') is not True
                or not response['message']['content'].strip()
                or evidence['eval_count'] <= 0
                or evidence['size_vram'] <= 0
            ):
                raise RuntimeError(
                    'Expected a completed chat and positive model VRAM usage'
                )
            if sorted(mask.split(',')) != sorted(expected.split(',')):
                raise RuntimeError(
                    f'Model runner selected {mask!r}, expected {expected!r}'
                )
            if (
                not offload
                or offload[-1][0] != offload[-1][1]
                or int(offload[-1][1]) == 0
            ):
                raise RuntimeError('Expected all reported model layers to offload')
            if {i for i, size in buffers.items() if size > 0} != set(
                range(len(expected.split(',')))
            ):
                raise RuntimeError(
                    'Expected a positive model buffer on each selected NPU'
                )
        except Exception:
            print(log_path.read_text(encoding='utf-8', errors='replace')[-16000:])
            raise
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            with contextlib.suppress(ProcessLookupError):
                os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', choices=['a2', 'a2-formats', '310p'])
    parser.add_argument('reports', type=Path)
    args = parser.parse_args()
    reports = args.reports.resolve()
    reports.mkdir(parents=True, exist_ok=True)
    is_310p = args.profile == '310p'
    hardware = (reports / 'npu-smi-before.log').read_text(encoding='utf-8')
    chip = re.search(
        r'\b(?:Ascend)?(' + ('310P' if is_310p else '910B') + r'\d*)\b', hardware, re.I
    )
    if not chip:
        raise RuntimeError(f'Unexpected hardware for {args.profile}:\n{hardware}')
    soc = 'Ascend' + chip[1].upper()

    cache = reports.parent.parent / 'downloads'
    cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in (
        'OLLAMA_LLM_LIBRARY',
        'GGML_BACKEND_PATH',
        'OLLAMA_LLAMA_CPP_SOURCE',
        'ASCEND_RT_VISIBLE_DEVICES',
        'LLAMA_ARG_DEVICE',
        'LLAMA_ARG_SPLIT_MODE',
    ):
        env.pop(key, None)
    env.setdefault('GOPROXY', 'https://goproxy.cn,direct')
    env.setdefault('GOSUMDB', 'sum.golang.org https://sum.golang.google.cn')
    with tempfile.TemporaryDirectory(prefix='ollama-cann-') as temp:
        work = Path(temp)
        source = work / 'ollama'
        run(['git', 'clone', '--depth', '1', FORK, str(source)], env=env, timeout=300)
        commit = run(
            ['git', 'rev-parse', 'HEAD'], cwd=source, capture_output=True, text=True
        ).stdout.strip()
        llama_ref = (source / 'LLAMA_CPP_VERSION').read_text().strip()
        print(f'Ollama commit: {commit}; llama.cpp: {llama_ref}')
        (reports / 'source.json').write_text(
            json.dumps(
                {
                    'source': FORK,
                    'commit': commit,
                    'llama_cpp': llama_ref,
                    'profile': args.profile,
                    'soc': soc,
                },
                indent=2,
            )
        )
        go_toolchain(source, cache, work, env)
        build = source / 'build'
        run(
            [
                'cmake',
                '-S',
                str(source),
                '-B',
                str(build),
                '-DCMAKE_BUILD_TYPE=Release',
                '-DOLLAMA_LLAMA_BACKENDS=cann',
                '-DGGML_CPU_ALL_VARIANTS=OFF',
                '-DOLLAMA_BUILD_PARALLEL=8',
                f'-DOLLAMA_VERSION={commit}',
                f'-DSOC_TYPE={soc}',
            ],
            env=env,
        )
        with (reports / 'build.log').open('w') as log:
            try:
                run(
                    ['cmake', '--build', str(build), '--parallel', '8'],
                    env=env,
                    timeout=7200,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                print(Path(log.name).read_text(errors='replace')[-16000:])
                raise
        payload = build / 'lib/ollama'
        for product in (
            source / 'ollama',
            payload / 'llama-server',
            payload / 'cann/libggml-cann.so',
        ):
            if not product.is_file():
                raise RuntimeError(f'Missing build product: {product}')

        # Preflight the CANN runtime separately. The Ollama process in each
        # case below still has to find and select this library on its own.
        probe_env = dict(env, GGML_BACKEND_PATH=str(payload / 'cann/libggml-cann.so'))
        probe_env['LD_LIBRARY_PATH'] = ':'.join(
            filter(
                None,
                (
                    str(payload),
                    str(payload / 'cann'),
                    env.get('LD_LIBRARY_PATH'),
                ),
            )
        )
        probe = subprocess.run(
            [str(payload / 'llama-server'), '--list-devices', '--offline'],
            env=probe_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
        (reports / 'devices.log').write_text(probe.stdout, encoding='utf-8')
        print(probe.stdout, flush=True)
        devices = set(re.findall(r'^\s*CANN(\d+):', probe.stdout, re.M))
        expected_devices = {'0', '1'} if args.profile == 'a2' else {'0'}
        if probe.returncode or not expected_devices <= devices:
            raise RuntimeError(
                f'Expected CANN devices {sorted(expected_devices)}, found {sorted(devices)}'
            )

        cases = {
            'a2': [
                ('a2-auto', 'q4_0', '0,1', '1', False),
                ('a2-dual', 'q4_0', '0,1', '0,1', True),
            ],
            'a2-formats': [
                ('a2-fp16', 'fp16', '0', '0', False),
                ('a2-q8', 'q8_0', '0', '0', False),
            ],
            '310p': [('310p-fp16', 'fp16', '0', '0', False)],
        }[args.profile]
        for name, quant, visible, expected, spread in cases:
            filename = f'qwen2.5-0.5b-instruct-{quant}.gguf'
            model = download(MODEL_URL + filename, cache / filename, *MODELS[quant])
            run_case(
                source,
                work,
                reports,
                env,
                commit,
                name,
                model,
                visible,
                expected,
                spread,
            )


if __name__ == '__main__':
    main()
