"""Bundle one already validated fixed Add export into a self-contained GE SO."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-fixture',type=Path,required=True)
    parser.add_argument('--backend',choices=['cannbot','tilelang'],required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    fixture=args.native_fixture.resolve(strict=True)
    lines=['#pragma once','#include <cstdint>']
    metadata={'backend':args.backend,'packaging':'kernel bytes and launch metadata embedded in implementation SO; GE SO_BINS carries that SO'}
    if args.backend=='cannbot':
        from cannbot_abi import prepare_cannbot_abi,ADAPTERS
        shutil.copytree(fixture/'cannbot-export',root/'cannbot-export')
        prepare_cannbot_abi(root)
        abi=json.loads((root/'cannbot_abi.json').read_text())
        binary=(root/'cannbot-export/add_host.so').read_bytes()
        defined=subprocess.check_output(['nm','-D','--defined-only',str(root/'cannbot-export/add_host.so')],text=True)
        for key in ['schema_symbol','invoke_symbol']:
            if abi[key] not in {s.split()[-1] for s in defined.splitlines() if s.split()}:raise ValueError('Host binary does not export manifest symbols')
        header,launch=ADAPTERS[abi['host_abi_version'],abi['schema_version']]
        lines += [f'#define CANNBOT_HOST_ABI_VERSION {abi["host_abi_version"]}',f'#define CANNBOT_SCHEMA_VERSION {abi["schema_version"]}',f'#define CANNBOT_ABI_ADAPTER_HEADER "{header}"',f'#define CANNBOT_FRAME_LAUNCH {launch}',f'static constexpr const char *kSchemaSymbol={json.dumps(abi["schema_symbol"])};',f'static constexpr const char *kInvokeSymbol={json.dumps(abi["invoke_symbol"])};']
        elements=256
        metadata['host_abi']=abi
        metadata['binary_source']='cannbot-export/add_host.so'
    else:
        manifest=json.loads((fixture/'tilelang-export/manifest.json').read_text())
        if manifest['args'] != [524288,'simt']:raise ValueError('Expected validated fixed SIMT vecadd specialization')
        text=(fixture/'tilelang-export/host_ir.txt').read_text()
        match=re.search(r'T\.call_packed\("main_kernel", A, B, C, (\d+), T\.int64\((\d+)\)\)',text)
        if not match:raise ValueError('Unrecognized launch IR; inspect before adapting')
        grid,ubuf=map(int,match.groups())
        if min(grid,ubuf)<=0:raise ValueError('Invalid launch metadata')
        binary=(fixture/'tilelang-export/kernel.aibin').read_bytes()
        elements=524288
        lines += ['static constexpr const char *kKernelName="main_kernel";',f'static constexpr uint32_t kGrid={grid};',f'static constexpr uint32_t kDynUbuf={ubuf};']
        metadata.update(kernel='main_kernel',grid=grid,dyn_ubuf_bytes=ubuf,binary_source='tilelang-export/kernel.aibin',host_ir_sha256=hashlib.sha256(text.encode()).hexdigest())
    if not binary.startswith(b'\x7fELF'):raise ValueError('Kernel artifact must be ELF')
    digest=hashlib.sha256(binary).hexdigest()
    lines += [f'static constexpr int64_t kElements={elements};',f'static constexpr const char *kKernelSha256="{digest}";','alignas(16) static const uint8_t kKernelBinary[]={',','.join(str(b) for b in binary),'};']
    (root/'kernel_bundle.h').write_text('\n'.join(lines)+'\n')
    metadata.update(elements=elements,binary_sha256=digest,binary_bytes=len(binary))
    (root/'bundle.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata))


if __name__=='__main__':main()
