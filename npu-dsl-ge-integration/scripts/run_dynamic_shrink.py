#!/usr/bin/env python3
"""Verify dynamic Add with non-increasing shapes and caller/kernel address evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

SHAPES = [4096, 4096, 4096, 2049, 2049, 1025, 257, 1, 1]


def validate_schedule(shapes):
    if not shapes or any(n < 1 or n > 4096 for n in shapes):
        raise ValueError('SHRINK_ONLY: dimensions must be in [1,4096]')
    if any(b > a for a, b in zip(shapes, shapes[1:])):
        raise ValueError('SHRINK_ONLY: every upward transition is disabled, including regrowth below the first maximum')


def records(text, prefix):
    return [dict(re.findall(r'(\w+)=([^\s]+)', line))
            for line in text.splitlines() if line.startswith(prefix)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True, help='New directory on the NPU machine')
    parser.add_argument('--device', type=int, required=True)
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
    validate_schedule(SHAPES)
    if args.device < 0 or not os.environ.get('ASCEND_HOME_PATH'):
        parser.error('Use a non-negative device ID and source the target CANN set_env.sh')
    skill = Path(__file__).resolve().parents[1]
    tilelang = args.tilelang_root.resolve(strict=True)
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    for template in ['ge-common', 'ge-dynamic-shrink']:
        for source in (skill / 'assets' / template).iterdir():
            if source.is_file():
                shutil.copy2(source, work / source.name)
    report = {
        'status': 'running', 'device': args.device, 'arch': args.arch,
              'cxx11_abi': args.cxx11_abi, 'build_jobs': args.build_jobs, 'stage_timeout': args.stage_timeout, 'shapes': SHAPES,
        'growth_executed': False, 'growth_status': 'not included in this decreasing-shape fixture',
        'physical_external_capacity_elements': 4096, 'guard_bytes_per_side': 4096,
        'scope': 'one runtime-dimension artifact per DSL; one GE model per scenario, repeated execution',
        'work_dir': str(work), 'tests': [], 'stages': [],
    }

    def save():
        (work / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    def run(name, command, env=None):
        print(f'[STAGE] {name}', flush=True)
        log = work / (name + '.log')
        try:
            with log.open('w') as stream:
                result = subprocess.run([str(value) for value in command], cwd=work, env=env,
                                         stdout=stream, stderr=subprocess.STDOUT, timeout=args.stage_timeout)
        except subprocess.TimeoutExpired as error:
            report['stages'].append({'name': name, 'returncode': None, 'status': 'timeout',
                                     'timeout_seconds': args.stage_timeout, 'log': log.name})
            save()
            raise RuntimeError(f'{name} timed out after {args.stage_timeout}s; inspect {log}') from error
        report['stages'].append({'name': name, 'returncode': result.returncode, 'log': log.name})
        save()
        if result.returncode:
            raise RuntimeError(f'{name} failed; inspect {log}')
        return log.read_text()

    save()
    try:
        run('environment', [sys.executable, skill / 'scripts/probe_env.py'])
        run('device', ['npu-smi', 'info'])
        run('export-cannbot', [sys.executable, skill / 'scripts/export_cannbot.py',
            '--driver', work / 'cannbot_dynamic.py', '--output', work / 'cannbot-export',
            '--arch', args.arch, '--compile-so'])
        run('export-tilelang', [sys.executable, skill / 'scripts/export_tilelang.py',
            '--tilelang-root', tilelang, '--factory', str(work / 'tilelang_dynamic.py') + ':dynamic_add',
            '--arch', args.arch, '--output', work / 'tilelang-export', '--compile-aibin'])
        run('prepare', [sys.executable, work / 'prepare.py'])
        run('configure', ['cmake', '-S', work, '-B', work / 'build', '-DCMAKE_BUILD_TYPE=Release',
                          f'-DGE_CXX11_ABI={args.cxx11_abi}'])
        run('build', ['cmake', '--build', work / 'build', '--parallel', args.build_jobs])
        run('check-opp', [sys.executable, skill / 'scripts/check_opp.py', work / 'build/opp'])
        run('ldd-runner', ['ldd', work / 'build/dynamic_runner'])
        env = os.environ.copy()
        old = env.get('ASCEND_CUSTOM_OPP_PATH', '')
        env['ASCEND_CUSTOM_OPP_PATH'] = str(work / 'build/opp') + (':' + old if old else '')
        for backend in ['cannbot', 'tilelang']:
            for mode in ['direct', 'reuse', 'rotate', 'auto', 'alias']:
                name = f'{backend}-{mode}'
                text = run(name, [work / 'build/dynamic_runner', backend, mode, args.device], env=env)
                summary = records(text, '[RESULT]')
                steps = records(text, '[STEP]')
                callbacks = records(text, '[GE_DYNAMIC_EXECUTE]')
                if len(summary) != 1:
                    raise RuntimeError(f'Missing result evidence for {name}')
                result = summary[0]
                good = (result.get('status') == 'PASS' and result.get('growth_executed') == '0'
                        and int(result.get('checked', 0)) == sum(SHAPES)
                        and [int(step['n']) for step in steps] == SHAPES
                        and '[CLEANUP] status=PASS' in text)
                good = good and all(step.get('status') == 'PASS' and step.get('input_guards') == 'PASS'
                                    and step.get('retired_buffers') == 'PASS' for step in steps)
                if mode != 'auto':
                    good = good and all(step.get('output_guards') == 'PASS' for step in steps)
                unique_kernel = {key: len({callback[key] for callback in callbacks}) for key in ['x', 'y', 'z']}
                kernel_matches_caller = (bool(callbacks) and len(callbacks) == len(steps)
                    and all(all(callback[key] == step[key] for key in ['x', 'y', 'z'])
                            for callback, step in zip(callbacks, steps)))
                if mode != 'direct':
                    good = good and result.get('same_model') == '1' and result.get('graphs_compiled') == '1'
                    good = good and [int(callback['n']) for callback in callbacks] == SHAPES
                    good = good and all(int(step['callback_delta']) == 1 for step in steps)
                    good = good and len({callback['node_ctx'] for callback in callbacks}) == 1
                    good = good and len({callback['op'] for callback in callbacks}) == 1
                    good = good and all(f'[GRAPH] {stage}=PASS' in text for stage in
                                        ['AddGraph', 'CompileGraph', 'LoadGraph'])
                    if mode == 'alias':
                        good = good and kernel_matches_caller and all(count == len(SHAPES) for count in unique_kernel.values())
                report['tests'].append({
                    'name': name, 'passed': good, **result, 'steps': steps,
                    'callbacks': callbacks, 'unique_kernel_addresses': unique_kernel,
                    'kernel_addresses_equal_caller': kernel_matches_caller,
                })
                save()
                if not good:
                    raise RuntimeError(f'Incomplete dynamic/address verification evidence for {name}')
        home = Path(os.environ['ASCEND_HOME_PATH'])
        files = [home / 'lib64' / name for name in
                 ['libge_compiler.so', 'libge_runner.so', 'libgert.so', 'liblowering.so']]
        files += [work / 'cannbot-export/dynamic_add_host.so', work / 'tilelang-export/kernel.aibin']
        report['files'] = [{'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                           for path in files]
        report['status'] = 'passed'
        report['checked_elements'] = sum(int(test['checked']) for test in report['tests'])
        report['max_abs'] = max(float(test['max_abs']) for test in report['tests'])
        report['limits'] = [
            'no upward shape transition or growth fix verification',
            'external physical buffers retain maximum capacity; logical tails are checked',
            'GE auto/internal output buffer has no external guard coverage',
            'alias case uses an explicit destination input/output reference contract',
            'dynamic executor address refresh is not static sink ArgsUpdater/AnnotatedArgs verification',
        ]
        save()
        print(f'[PASS] ten paths, {report["checked_elements"]} elements; growth skipped; {work / "results.json"}')
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        save()
        raise


if __name__ == '__main__':
    main()
