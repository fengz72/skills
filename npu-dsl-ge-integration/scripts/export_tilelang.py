#!/usr/bin/env python3
"""Export a TileLang PrimFunc as AscendC plus launch IR and matching headers."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factory', required=True, help='Python factory file followed by :function returning PrimFunc')
    parser.add_argument('--args', default='[]', help='JSON positional arguments to the factory')
    parser.add_argument('--kwargs', default='{}', help='JSON keyword arguments to the factory')
    parser.add_argument('--tilelang-root', type=Path, help='Explicit TileLang source checkout')
    parser.add_argument('--arch', required=True, help='Confirmed Bisheng NPU architecture')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--compile-aibin', action='store_true', help='Also compile executable CCE ELF')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9._-]+', args.arch):
        parser.error('Invalid --arch')
    filename, separator, name = args.factory.rpartition(':')
    if not separator or not name.isidentifier():
        parser.error('--factory must be file.py:function')
    factory_file = Path(filename).resolve(strict=True)
    positional, keywords = json.loads(args.args), json.loads(args.kwargs)
    if not isinstance(positional, list) or not isinstance(keywords, dict):
        parser.error('--args must be an array and --kwargs must be an object')
    root = args.tilelang_root.resolve(strict=True) if args.tilelang_root else None
    if root:
        sys.path.insert(0, str(root))
    import tilelang
    from tilelang import tvm
    from tilelang.engine import lower
    from tilelang.engine.param import dump_kernel_params
    from tilelang.env import TILELANG_TEMPLATE_PATH
    if root and not Path(tilelang.__file__).resolve().is_relative_to(root):
        raise RuntimeError(f'TileLang import does not match --tilelang-root: {tilelang.__file__}')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(factory_file.parent))
    spec = importlib.util.spec_from_file_location('dsl_ge_export_factory', factory_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    prim_func = getattr(module, name)(*positional, **keywords)
    target = tvm.target.Target({'kind': 'ascend', 'arch': args.arch})
    # This checkout requires a current target during Ascend layout inference.
    with target, tvm.transform.PassContext(opt_level=3):
        artifact = lower(prim_func, target=target, enable_host_codegen=False, enable_device_compile=False)
    source = artifact.kernel_source
    if not source.strip() or '__global__' not in source:
        raise RuntimeError('No AscendC device kernel generated; inspect factory/target/backend version')
    (output / 'kernel.asc').write_text(source)
    (output / 'host_ir.txt').write_text(artifact.host_mod.script())
    (output / 'device_ir.txt').write_text(artifact.device_mod.script())
    (output / 'params.tvm.json').write_text(dump_kernel_params(artifact.params))
    template_source = Path(TILELANG_TEMPLATE_PATH).resolve() / 'tl_templates' / 'ascend'
    include_dir = output / 'include'
    shutil.copytree(template_source, include_dir / 'tl_templates' / 'ascend')
    header_hashes = {str(p.relative_to(include_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(include_dir.rglob('*')) if p.is_file()}
    commit, dirty = None, None
    if root and (root / '.git').exists():
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip())
    manifest = {
        'dsl': 'tilelang', 'distribution_version': importlib.metadata.version('tilelang'),
        'module': tilelang.__file__, 'checkout_commit': commit, 'checkout_dirty': dirty,
        'python': sys.executable, 'target': str(target), 'arch': args.arch,
        'factory': str(factory_file) + ':' + name,
        'factory_sha256': hashlib.sha256(factory_file.read_bytes()).hexdigest(),
        'args': positional, 'kwargs': keywords, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'kernel_symbols': re.findall(r'extern\s+"C"\s+__global__.*?\bvoid\s+(\w+)\s*\(', source, re.S),
        'template_source': str(template_source), 'header_sha256': header_hashes,
        'validation': 'source-export',
    }
    if args.compile_aibin:
        from tilelang.contrib.bisheng import compile_ascend
        binary = compile_ascend(source, target_format='aibin', npu_arch=args.arch,
                                options=['-I' + str(include_dir)], path_target=str(output / 'kernel.aibin'))
        if bytes(binary[:4]) != b'\x7fELF':
            raise RuntimeError('Compiler did not return an ELF binary')
        manifest['binary_sha256'] = hashlib.sha256(binary).hexdigest()
        manifest['validation'] = 'source-export-and-recompile'
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'Exported AscendC, launch IR and template headers: {output}')


if __name__ == '__main__':
    main()
