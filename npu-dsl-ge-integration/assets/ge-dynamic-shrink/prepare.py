"""Generate artifact-specific declarations and embed the exported TileLang CCE ELF."""
import json
import re
from pathlib import Path
root=Path(__file__).resolve().parent
from cannbot_abi import prepare_cannbot_abi
prepare_cannbot_abi(root)
tile_meta=json.loads((root/'tilelang-export/manifest.json').read_text())
device_source=(root/'tilelang-export/kernel.asc').read_text()
if not re.search(r'void main_kernel\(__gm__ float\* A, __gm__ float\* B, __gm__ float\* C, int64_t N\)',device_source):
    raise ValueError('This fixture requires device arguments A,B,C,N:int64 in that order')
host_ir=(root/'tilelang-export/host_ir.txt').read_text()
launch=re.search(r'T\.call_packed\("main_kernel", A, B, C, N, (\d+), T\.int64\((\d+)\)\)',host_ir)
if not launch:
    raise ValueError('Unrecognized host launch IR; inspect the TileLang version before adapting this fixture')
grid,ubuf=map(int,launch.groups())
if grid<=0 or ubuf<=0:
    raise ValueError('Invalid grid/dynamic UBUF size')
(root/'tile_launch_metadata.json').write_text(json.dumps({'kernel':'main_kernel','grid':grid,'dyn_ubuf_bytes':ubuf,'runtime_dimension':'N:int64'},indent=2)+'\n')
binary=(root/'tilelang-export/kernel.aibin').read_bytes()
(root/'tile_binary.h').write_text('#pragma once\n#include <cstdint>\nstatic constexpr uint32_t kTileGrid='+str(grid)+';\nstatic constexpr uint32_t kTileDynUbuf='+str(ubuf)+';\nalignas(16) static const uint8_t kTileBinary[]={\n'+','.join(str(b) for b in binary)+'\n};\n')
