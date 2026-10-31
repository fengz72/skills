#!/usr/bin/env python3
"""Read compiler/package locations without initializing an NPU or changing settings."""
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys

result = {'python': sys.executable, 'python_version': platform.python_version(),
          'host_arch': platform.machine(), 'packages': {}, 'tools': {}, 'environment': {}}
for distribution, module in [('cannbot-dsl', 'cannbotdsl'), ('tilelang', 'tilelang'),
                             ('torch', 'torch'), ('torch-npu', 'torch_npu')]:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    spec = importlib.util.find_spec(module)
    result['packages'][distribution] = {'distribution_version': version, 'module': spec.origin if spec else None}
for tool in ['bisheng', 'cmake', 'g++', 'nm', 'ldd', 'atc', 'npu-smi']:
    result['tools'][tool] = shutil.which(tool)
for key in ['ASCEND_HOME_PATH', 'ASCEND_OPP_PATH', 'ASCEND_CUSTOM_OPP_PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH',
            'NPU_ARCH', 'CANNBOTDSL_NPU_ARCH', 'ASCEND_NPU_ARCH', 'DEVICE_ID',
            'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES', 'CXX11_ABI']:
    result['environment'][key] = os.environ.get(key)
cann_home = os.environ.get('ASCEND_HOME_PATH')
home = Path(cann_home).resolve() if cann_home else None
result['cann_home_resolved'] = str(home) if home else None
version_file = home / 'version.cfg' if home else None
result['cann_version_cfg'] = version_file.read_text() if version_file and version_file.is_file() else None
header = home / 'include/graph/custom_op.h' if home else None
result['custom_op_header'] = str(header) if header and header.is_file() else None
if header and header.is_file():
    text = header.read_text()
    result['custom_op_interfaces'] = {name: name in text for name in
        ['EagerExecuteOp', 'ShapeInferOp', 'CompilableOp', 'PortableOp', 'ArgsUpdater', 'ResourceUsageReporter']}
print(json.dumps(result, ensure_ascii=False, indent=2))
