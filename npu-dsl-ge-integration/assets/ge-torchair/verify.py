"""Verify fixture TorchAir graphs, precision, output ownership and address changes."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import traceback

SHAPES = [4096, 4096, 4096, 2049, 2049, 1025, 257, 3, 2, 2, 1, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=['fixed', 'dynamic'], required=True)
    parser.add_argument('--opp-root', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    args = parser.parse_args()
    if args.device < 0:
        parser.error('--device must be non-negative')
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    opp = args.opp_root.resolve(strict=True)
    os.environ['ASCEND_CUSTOM_OPP_PATH'] = str(opp)  # This verification process only.
    os.chdir(work)
    import torch
    import torch_npu
    import torchair
    import torch_add
    library = torch_add.load_opp(opp)
    torch_add.register_converter()
    torch.npu.set_device(args.device)
    device = f'npu:{args.device}'
    report = {'status': 'running', 'scope': args.scope, 'device': args.device,
              'probe': torch_add.probe(), 'cases': [], 'growth_executed': False,
              'limitations': ['input values/logical tails checked; no external guards on rotated input allocations',
                             'functional GE internal output has no external guard',
                             'mutable output uses TorchAir functionalization; kernel target may be a temporary']}

    def save():
        (work / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    class Model(torch.nn.Module):
        def __init__(self, operation, out):
            super().__init__()
            self.operation, self.out = operation, out

        def forward(self, x, y, out_buffer=None):
            if self.out:
                self.operation(x, y, out_buffer)
                return out_buffer
            return self.operation(x, y)

    save()
    try:
        for backend in ['cannbot', 'tilelang']:
            modes = ['reuse', 'rotate'] if args.scope == 'fixed' else ['reuse', 'rotate', 'out_reuse', 'out_rotate']
            for mode in modes:
                out = mode.startswith('out_')
                reuse = mode.endswith('reuse')
                name = f'{backend}_{args.scope}_{mode}'
                case_dir = work / name
                case_dir.mkdir()
                torch._dynamo.reset()
                gc.collect()
                torch_add.set_case_tag(name)
                spec = torch_add.get_spec(backend, args.scope, out=out)
                shapes = [spec['elements']] * 6 if args.scope == 'fixed' else SHAPES
                if any(b > a for a, b in zip(shapes, shapes[1:])):
                    raise ValueError('Decreasing-shape fixture requires a non-increasing sequence')
                config = torchair.CompilerConfig()
                config.debug.graph_dump.type = 'pbtxt'
                config.debug.graph_dump.path = str(case_dir)
                inner = torchair.get_npu_backend(compiler_config=config)
                compile_events = []

                def counting(gm, inputs):
                    index = len(compile_events) + 1
                    (case_dir / f'fx_{index}.py').write_text(gm.code)
                    compile_events.append({'index': index, 'input_shapes': [[str(d) for d in v.shape] for v in inputs if isinstance(v, torch.Tensor)]})
                    print('[TORCHAIR_COMPILE]', name, index, flush=True)
                    return inner(gm, inputs)

                operation = torch_add.get_op(backend, args.scope, out=out)
                compiled = torch.compile(Model(operation, out), backend=counting, fullgraph=True,
                                         dynamic=args.scope == 'dynamic')
                retained, returned = [], []
                steps = []
                shared_x = shared_y = shared_output = None
                capacity = max(shapes)
                if reuse:
                    shared_x = torch.empty(capacity, dtype=torch.float32, device=device)
                    shared_y = torch.empty_like(shared_x)
                    if out:
                        shared_output = torch.empty(capacity + 2048, dtype=torch.float32, device=device)
                print('[TORCHAIR_CASE_BEGIN]', json.dumps({'name': name, 'ge_type': spec['ge_type']}), flush=True)
                for step, n in enumerate(shapes):
                    generator = torch.Generator(device='cpu').manual_seed(7101 + step)
                    a = torch.randn(n, generator=generator, dtype=torch.float32) + step * .25
                    b = torch.randn(n, generator=generator, dtype=torch.float32) - step * .125
                    reference = (a.double() + b.double()).float()
                    raw_x = shared_x if reuse else torch.empty(n, dtype=torch.float32, device=device)
                    raw_y = shared_y if reuse else torch.empty_like(raw_x)
                    raw_x.fill_(12345.25)
                    raw_y.fill_(12345.25)
                    x, y = raw_x[:n], raw_y[:n]
                    x.copy_(a)
                    y.copy_(b)
                    target = raw_output = None
                    if out:
                        raw_output = shared_output if reuse else torch.empty(capacity + 2048, dtype=torch.float32, device=device)
                        raw_output.fill_(12345.25)
                        target = raw_output[1024:1024+n]
                        target.fill_(float('nan'))
                    if args.scope == 'dynamic' and n > 1:
                        for tensor in [x, y] + ([target] if out else []):
                            torch._dynamo.mark_dynamic(tensor, 0, min=2, max=4096)
                    before = library.SmokeExecuteCount()
                    with torch.inference_mode():
                        z = compiled(x, y, target) if out else compiled(x, y)
                    torch.npu.synchronize()
                    delta = library.SmokeExecuteCount() - before
                    actual = z.cpu()
                    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
                    torch.testing.assert_close(x.cpu(), a, atol=0, rtol=0)
                    torch.testing.assert_close(y.cpu(), b, atol=0, rtol=0)
                    if reuse:
                        assert bool(torch.all(raw_x[n:] == 12345.25)) and bool(torch.all(raw_y[n:] == 12345.25))
                    if out:
                        assert z.data_ptr() == target.data_ptr()
                        assert bool(torch.all(raw_output[:1024] == 12345.25))
                        assert bool(torch.all(raw_output[1024+n:] == 12345.25))
                    for tensors, snapshots in retained:
                        for tensor, snapshot in zip(tensors, snapshots):
                            torch.testing.assert_close(tensor.cpu(), snapshot, atol=0, rtol=0)
                    for previous, expected in returned:
                        torch.testing.assert_close(previous.cpu(), expected, atol=1e-6, rtol=1e-6)
                    if not reuse:
                        tensors = [raw_x, raw_y] + ([raw_output] if out else [])
                        retained.append((tensors, [t.cpu().clone() for t in tensors]))
                    if not out or not reuse:
                        returned.append((z, reference.clone()))
                    error = actual.double() - reference.double()
                    record = {'step': step, 'n': n, 'x': hex(x.data_ptr()), 'y': hex(y.data_ptr()),
                              'z': hex(z.data_ptr()), 'callback_delta': delta, 'compile_count': len(compile_events),
                              'checked': n, 'max_abs': float(error.abs().max()), 'square_error': float((error*error).sum()),
                              'inputs_unchanged': True, 'retired_buffers_unchanged': True,
                              'retained_outputs_unchanged': True,
                              'output_guards': 'passed' if out else 'not_applicable'}
                    steps.append(record)
                    print('[TORCHAIR_STEP]', json.dumps(record), flush=True)
                expected_compiles = 1 if args.scope == 'fixed' else 2
                assert len(compile_events) == expected_compiles, compile_events
                dumps = sorted(case_dir.glob('*optimized*.pbtxt'))
                assert len(dumps) == expected_compiles, [p.name for p in dumps]
                for dump in dumps:
                    text = dump.read_text()
                    assert len(re.findall(r'^\s*op:\s*"' + re.escape(spec['ge_type']) + r'"\s*$', text, re.M)) == 1
                    if args.scope == 'dynamic' and not out:
                        assert re.search(r'^\s*op:\s*"TensorMove"\s*$', text, re.M)
                x_count = len({s['x'] for s in steps})
                y_count = len({s['y'] for s in steps})
                z_count = len({s['z'] for s in steps})
                assert x_count == y_count == (1 if reuse else len(shapes))
                assert z_count == (1 if out and reuse else len(shapes))
                if args.scope == 'dynamic':
                    nonunit = [s for s in steps if s['n'] > 1]
                    assert all(s['compile_count'] == 1 and s['callback_delta'] == 1 for s in nonunit)
                    assert all(s['compile_count'] == 2 for s in steps if s['n'] == 1)
                assert sum(s['callback_delta'] for s in steps) > 0
                checked = sum(s['checked'] for s in steps)
                result = {'name': name, 'backend': backend, 'out': out, 'reuse': reuse, 'ge_type': spec['ge_type'],
                          'calls': len(steps), 'checked': checked, 'max_abs': max(s['max_abs'] for s in steps),
                          'rmse': (sum(s['square_error'] for s in steps)/checked)**.5,
                          'compile_events': compile_events, 'steps': steps,
                          'caller_unique_addresses': {'x': x_count, 'y': y_count, 'z': z_count},
                          'graph_dumps': [{'file': str(p.relative_to(work)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in dumps],
                          'status': 'passed'}
                report['cases'].append(result)
                save()
                print('[TORCHAIR_CASE_END]', json.dumps({'name': name, 'status': 'passed'}), flush=True)
                del compiled
                torch._dynamo.reset()
                gc.collect()
        report['status'] = 'passed'
        report['calls'] = sum(c['calls'] for c in report['cases'])
        report['checked'] = sum(c['checked'] for c in report['cases'])
        report['max_abs'] = max(c['max_abs'] for c in report['cases'])
        save()
        print('[TORCHAIR_RESULT]', json.dumps({'status': 'passed', 'calls': report['calls'], 'checked': report['checked']}), flush=True)
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        report['traceback'] = traceback.format_exc()
        save()
        raise


if __name__ == '__main__':
    main()
