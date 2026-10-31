#!/usr/bin/env python3
"""Run both DSL TorchAir frontend suites against explicitly supplied fixture OPPs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def records(text, prefix):
    return [dict(re.findall(r'(\w+)=([^\s]+)', line)) for line in text.splitlines() if line.startswith(prefix)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-opp-root', type=Path, required=True)
    parser.add_argument('--dynamic-opp-root', type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--stage-timeout', type=int, default=900)
    args = parser.parse_args()
    if args.device < 0 or args.stage_timeout < 1 or not os.environ.get('ASCEND_HOME_PATH'):
        parser.error('Confirm device, positive stage timeout and the target CANN environment')
    roots = {scope: getattr(args, scope + '_opp_root').resolve(strict=True) for scope in ['fixed', 'dynamic']}
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    skill = Path(__file__).resolve().parents[1]
    shutil.copytree(skill / 'assets/ge-torchair', work / 'fixture', ignore=shutil.ignore_patterns('__pycache__'))
    report = {'status': 'running', 'device': args.device, 'stage_timeout': args.stage_timeout,
              'scope': 'both DSLs; fixed shape, shrinking shape, stable/rotated caller I/O and mutable output refs',
              'phases': [], 'growth_executed': False,
              'singleton_note': 'PyTorch specializes size one into a second graph; shrinking sizes >=2 reuse one graph'}

    def save():
        (work / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    save()
    try:
        for scope, opp in roots.items():
            command = [sys.executable, work / 'fixture/verify.py', '--scope', scope, '--opp-root', opp,
                       '--work-dir', work / scope, '--device', args.device]
            log = work / (scope + '.log')
            print('[TORCHAIR_PHASE]', scope, flush=True)
            try:
                with log.open('w') as stream:
                    process = subprocess.run([str(v) for v in command], cwd=work, stdout=stream,
                                             stderr=subprocess.STDOUT, timeout=args.stage_timeout)
            except subprocess.TimeoutExpired as error:
                report['phases'].append({'scope': scope, 'status': 'timeout', 'log': log.name})
                save()
                raise RuntimeError(f'{scope} timed out; inspect {log}') from error
            if process.returncode:
                report['phases'].append({'scope': scope, 'status': 'failed', 'returncode': process.returncode, 'log': log.name})
                save()
                raise RuntimeError(f'{scope} failed; inspect {log}')
            result = json.loads((work / scope / 'results.json').read_text())
            assert result['status'] == 'passed'
            text = log.read_text()
            for case in result['cases']:
                begin = '[TORCHAIR_CASE_BEGIN] ' + json.dumps({'name': case['name'], 'ge_type': case['ge_type']})
                section = text.split(begin, 1)[1].split('[TORCHAIR_CASE_END]', 1)[0]
                events = records(section, '[GE_CUSTOM_EXECUTE]' if scope == 'fixed' else '[GE_DYNAMIC_EXECUTE]')
                assert events, case['name']
                case['ge_callbacks'] = events
                if scope == 'dynamic':
                    events = [event for event in events if event['n'] != '1']
                    steps = [step for step in case['steps'] if step['n'] > 1]
                    assert len(events) == len(steps)
                    assert [int(e['n']) for e in events] == [s['n'] for s in steps]
                    assert len({e['node_ctx'] for e in events}) == len({e['op'] for e in events}) == 1
                    unique = {key: len({e[key] for e in events}) for key in ['x', 'y', 'z']}
                    case['nonunit_kernel_unique_addresses'] = unique
                    case['kernel_matches_caller'] = {key: all(e[key] == s[key] for e, s in zip(events, steps)) for key in ['x', 'y', 'z']}
                    if not case['reuse']:
                        assert unique['x'] == unique['y'] == len(steps), (case['name'], unique)
                        if case['out']:
                            assert unique['z'] == len(steps), (case['name'], unique)
            report['phases'].append({'scope': scope, 'status': 'passed', 'command': list(map(str, command)),
                                     'opp_root': str(opp), 'log': log.name, 'result': result,
                                     'opp_files': {str(p.relative_to(opp)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                   for p in opp.rglob('*.so') if p.is_file()}})
            save()
            print('[TORCHAIR_PASS]', scope, 'calls=', result['calls'], flush=True)
        cases = [case for phase in report['phases'] for case in phase['result']['cases']]
        report.update(status='passed', calls=sum(c['calls'] for c in cases), checked=sum(c['checked'] for c in cases),
                      max_abs=max(c['max_abs'] for c in cases), max_rmse=max(c['rmse'] for c in cases),
                      ge_compilations=sum(len(c['compile_events']) for c in cases))
        save()
        print('[TORCHAIR_RESULT]', json.dumps({key: report[key] for key in ['status', 'calls', 'checked', 'max_abs', 'max_rmse']}), flush=True)
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        save()
        raise


if __name__ == '__main__':
    main()
