#!/usr/bin/env python3
"""Check a trusted, locally built OPP layout and shared-library dependencies."""
import argparse
from pathlib import Path
import platform
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path)
parser.add_argument('--arch', default=platform.machine())
args = parser.parse_args()
directory = args.root.resolve() / 'op_graph' / 'lib' / 'linux' / args.arch
libraries = sorted(directory.glob('*.so'))
if not libraries:
    parser.error(f'No GE shared libraries in {directory}')
failed = False
for library in sorted(directory.rglob('*.so')):
    if library.read_bytes()[:4] != b'\x7fELF':
        print(f'FAIL: not ELF: {library}')
        failed = True
        continue
    result = subprocess.run(['ldd', str(library)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    missing = [line.strip() for line in result.stdout.splitlines() if 'not found' in line]
    if result.returncode or missing:
        print(f'FAIL: {library}\n{result.stdout}')
        failed = True
    else:
        print(f'PASS: {library}')
raise SystemExit(1 if failed else 0)
