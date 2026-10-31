#!/usr/bin/env python3
"""Build both DSL add fixtures in a new directory and verify direct NPU and GE execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True, help='New directory on the NPU machine')
    parser.add_argument('--device', type=int, required=True, help='Confirmed logical ACL device ID')
    parser.add_argument('--tilelang-root', type=Path, required=True)
    parser.add_argument('--arch', required=True, help='Confirmed target Bisheng NPU architecture')
    parser.add_argument('--cxx11-abi', type=int, choices=[0, 1], required=True,
                        help='C++ standard-library ABI confirmed against the target GE build')
    parser.add_argument('--build-jobs', type=int,
                        default=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1,
                        help='Host build parallelism; defaults to available CPU count')
    parser.add_argument('--stage-timeout', type=int, default=600,
                        help='Per-stage subprocess timeout in seconds; slow or loaded machines may need more for the build')
    args = parser.parse_args()
    if args.build_jobs < 1:
        parser.error('--build-jobs must be positive')
    if args.stage_timeout < 1:
        parser.error('--stage-timeout must be positive')
    if not re.fullmatch(r'[A-Za-z0-9._-]+', args.arch):
        parser.error('Invalid --arch')
    if args.device < 0 or not os.environ.get('ASCEND_HOME_PATH'):
        parser.error('Use a non-negative device ID and source the target CANN set_env.sh first')
    skill = Path(__file__).resolve().parents[1]
    work = args.work_dir.resolve()
    root = args.tilelang_root.resolve(strict=True)
    work.mkdir(parents=True, exist_ok=False)
    for template in ['ge-common', 'ge-smoke']:
        for source in (skill / 'assets' / template).iterdir():
            if source.is_file():
                shutil.copy2(source, work / source.name)
    report = {'status': 'running', 'device': args.device, 'arch': args.arch,
              'cxx11_abi': args.cxx11_abi, 'build_jobs': args.build_jobs, 'stage_timeout': args.stage_timeout, 'work_dir': str(work),
              'scope': 'fixed FP32 ND add; direct NPU and native GE static graph; six input cases per path',
              'tests': [], 'stages': []}

    def save():
        (work / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    def run(name, command, env=None):
        print(f'[STAGE] {name}', flush=True)
        log = work / (name + '.log')
        try:
            with log.open('w') as stream:
                process = subprocess.run([str(v) for v in command], cwd=work, env=env,
                                         stdout=stream, stderr=subprocess.STDOUT, timeout=args.stage_timeout)
        except subprocess.TimeoutExpired as error:
            report['stages'].append({'name': name, 'returncode': None, 'status': 'timeout',
                                     'timeout_seconds': args.stage_timeout, 'log': log.name})
            save()
            raise RuntimeError(f'{name} timed out after {args.stage_timeout}s; inspect {log}') from error
        report['stages'].append({'name': name, 'returncode': process.returncode, 'log': log.name})
        save()
        if process.returncode:
            raise RuntimeError(f'{name} failed; inspect {log}')
        return log.read_text()

    save()
    try:
        run('environment', [sys.executable, skill / 'scripts/probe_env.py'])
        run('device', ['npu-smi', 'info'])
        run('export-cannbot', [sys.executable, skill / 'scripts/export_cannbot.py',
            '--driver', work / 'cannbot_add.py', '--output', work / 'cannbot-export',
            '--arch', args.arch, '--compile-so'])
        run('export-tilelang', [sys.executable, skill / 'scripts/export_tilelang.py',
            '--tilelang-root', root, '--factory', str(root / 'examples/ascend/example_vecadd.py') + ':vector_add',
            '--args', '[524288, "simt"]', '--arch', args.arch,
            '--output', work / 'tilelang-export', '--compile-aibin'])
        run('prepare', [sys.executable, work / 'prepare.py'])
        run('configure', ['cmake', '-S', work, '-B', work / 'build', '-DCMAKE_BUILD_TYPE=Release',
                          f'-DGE_CXX11_ABI={args.cxx11_abi}'])
        run('build', ['cmake', '--build', work / 'build', '--parallel', args.build_jobs])
        run('check-opp', [sys.executable, skill / 'scripts/check_opp.py', work / 'build/opp'])
        run('ldd-runner', ['ldd', work / 'build/smoke_runner'])
        env = os.environ.copy()
        old = env.get('ASCEND_CUSTOM_OPP_PATH', '')
        env['ASCEND_CUSTOM_OPP_PATH'] = str(work / 'build/opp') + (':' + old if old else '')
        for backend in ['cannbot', 'tilelang']:
            for mode, output in [('direct', 'external'), ('ge', 'external'), ('ge', 'auto')]:
                name = f'{backend}-{mode}-{output}'
                command = [work / 'build/smoke_runner', backend, mode, str(args.device), output]
                text = run(name, command, env=env)
                line = next((v for v in text.splitlines() if v.startswith('[RESULT]')), '')
                result = dict(re.findall(r'(\w+)=([^\s]+)', line))
                cases = [v for v in text.splitlines() if v.startswith('[CASE]')]
                expected = 6 * (256 if backend == 'cannbot' else 524288)
                good = (result.get('status') == 'PASS' and int(result.get('checked', 0)) == expected
                        and len(cases) == 6 and '[CLEANUP] status=PASS' in text)
                if mode == 'ge':
                    good = good and int(result.get('ge_execute_callbacks', 0)) > 0
                    good = good and all(f'[GRAPH] {stage}=PASS' in text for stage in
                                        ['AddGraph', 'CompileGraph', 'LoadGraph'])
                    good = good and 'nodes=3' in text
                report['tests'].append({'name': name, 'passed': good, **result})
                save()
                if not good:
                    raise RuntimeError(f'Incomplete verification evidence: {name}')
        # Actual link resolution and hashes matter when local GE has patched components.
        home = Path(os.environ['ASCEND_HOME_PATH'])
        paths = [home / 'lib64' / name for name in
                 ['libge_compiler.so', 'libge_runner.so', 'libgert.so', 'liblowering.so']]
        paths += [work / 'build/smoke_runner', work / 'cannbot-export/add_host.so',
                  work / 'tilelang-export/kernel.aibin']
        report['files'] = [{'path': str(path.resolve()),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
        report['status'] = 'passed'
        report['checked_elements'] = sum(int(test['checked']) for test in report['tests'])
        report['max_abs'] = max(float(test['max_abs']) for test in report['tests'])
        report['limits'] = ['fixed shapes/dtype/layout', 'one device/context, serial execution',
                            'no TorchAir/ONNX/OM or concurrent-model verification',
                            'guard checks apply only to caller-provided output buffers']
        save()
        print(f'[PASS] six paths, {report["checked_elements"]} elements, max_abs={report["max_abs"]}; {work / "results.json"}')
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        save()
        raise


if __name__ == '__main__':
    main()
