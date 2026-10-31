#!/usr/bin/env python3
"""Export cannbot-dsl AscendC using an AOT-only driver in a fresh process."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver', type=Path, required=True,
                        help='Python file calling cb.compile(@cb.host, TensorSpec, ...)')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--arch', required=True, help='Confirmed Bisheng NPU architecture')
    parser.add_argument('--compile-so', action='store_true', help='Also rebuild each exported .asc')
    parser.add_argument('driver_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    driver = args.driver.resolve(strict=True)
    if not re.fullmatch(r'[A-Za-z0-9._-]+', args.arch):
        parser.error('Invalid --arch')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CANNBOTDSL_PIPE_STAGE='translate', CANNBOTDSL_DUMP_ASCENDC='1',
               CANNBOTDSL_DUMP_DIR=str(output), CANNBOTDSL_NPU_ARCH=args.arch,
               CANNBOTDSL_REUSE_ASCENDC='0')
    extra = args.driver_args
    if extra[:1] == ['--']:
        extra = extra[1:]
    command = [sys.executable, str(driver), *extra]
    with (output / 'codegen.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    sources = sorted(output.glob('*.asc'))
    if not sources:
        raise RuntimeError(f'No .asc exported. Inspect {output / "codegen.log"}; the driver must call cb.compile.')
    manifest = {
        'dsl': 'cannbot-dsl', 'version': importlib.metadata.version('cannbot-dsl'),
        'module': importlib.util.find_spec('cannbotdsl').origin,
        'python': sys.executable, 'arch': args.arch, 'driver': str(driver),
        'driver_sha256': hashlib.sha256(driver.read_bytes()).hexdigest(),
        'command': command, 'validation': 'source-export', 'sources': [],
    }
    compiler = None
    if args.compile_so:
        # Version-dependent internal API; see references/cannbot-dsl.md.
        os.environ['CANNBOTDSL_NPU_ARCH'] = args.arch
        from cannbotdsl.core.toolchain.ascendc import AscendCCompiler
        compiler = AscendCCompiler()
    for source in sources:
        text = source.read_text()
        marker = re.search(r'CANNBOTDSL_HOST_ABI_VERSION=(\d+)', text)
        symbols = sorted(set(re.findall(
            r'\b\w+__cannbot_(?:schema|invoke|invoke_profile)_v\d+\b', text)))
        if not marker or not any('__cannbot_schema_v' in s for s in symbols) or not any('__cannbot_invoke_v' in s for s in symbols):
            raise RuntimeError(f'Missing host ABI marker/schema/invoke in {source}; inspect installed compiler version')
        record = {'file': source.name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'host_abi_version': int(marker.group(1)), 'exported_symbols': symbols}
        if compiler is not None:
            library = source.with_suffix('.so')
            compiler.compile_ascendc_to_so(text, output_path=str(library), exported_symbols=symbols)
            record['library'] = library.name
        manifest['sources'].append(record)
    if compiler is not None:
        manifest['validation'] = 'source-export-and-recompile'
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'Exported {len(sources)} AscendC source(s): {output}')


if __name__ == '__main__':
    main()
