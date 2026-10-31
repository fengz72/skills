"""Select an implemented wire-format decoder from the freshly exported artifact."""
import json
from pathlib import Path
import re

# Each entry represents a decoded binary layout, not a required DSL installation version.
ADAPTERS = {(9, 2): ('cannbot_frame_v9_schema_v2.h', 'cannbot_frame_v9_schema_v2::Launch')}


def prepare_cannbot_abi(root):
    root = Path(root)
    exported = root / 'cannbot-export'
    manifest = json.loads((exported / 'manifest.json').read_text())
    if len(manifest['sources']) != 1:
        raise ValueError('Add fixture expects one exported host specialization')
    source = manifest['sources'][0]
    markers = set(map(int, re.findall(r'CANNBOTDSL_HOST_ABI_VERSION=(\d+)',
                                     (exported / source['file']).read_text())))
    if len(markers) != 1 or source['host_abi_version'] not in markers:
        raise ValueError('Exported source ABI marker and manifest disagree')
    abi = source['host_abi_version']
    symbols = source['exported_symbols']
    invokes = [s for s in symbols if re.fullmatch(r'\w+__cannbot_invoke_v\d+', s)]
    if len(invokes) != 1:
        raise ValueError('Expected one unambiguous host invoke symbol')
    invoke = invokes[0]
    prefix, invoke_abi = invoke.rsplit('__cannbot_invoke_v', 1)
    if int(invoke_abi) != abi:
        raise ValueError('Host invoke symbol and source ABI marker disagree')
    schemas = [s for s in symbols if re.fullmatch(re.escape(prefix) + r'__cannbot_schema_v\d+', s)]
    if len(schemas) != 1:
        raise ValueError('Expected one schema symbol for the selected host specialization')
    schema = schemas[0]
    schema_version = int(schema.rsplit('_v', 1)[1])
    adapter = ADAPTERS.get((abi, schema_version))
    if adapter is None:
        raise ValueError(f'No frame decoder for detected host ABI {abi}/schema {schema_version}; '
                         'implement the matching wire layout before building this fixture')
    header, launch = adapter
    if not (root / header).is_file():
        raise ValueError(f'Missing selected ABI adapter: {header}')
    (root / 'artifact_symbols.h').write_text(
        '#pragma once\n#include <cstdint>\n'
        f'extern "C" const void *{schema}() noexcept;\n'
        f'extern "C" int32_t {invoke}(void *, const void *, uint64_t) noexcept;\n'
        f'#define CANNBOT_SCHEMA {schema}\n#define CANNBOT_INVOKE {invoke}\n'
        f'#define CANNBOT_HOST_ABI_VERSION {abi}\n#define CANNBOT_SCHEMA_VERSION {schema_version}\n'
        f'#define CANNBOT_ABI_ADAPTER_HEADER "{header}"\n#define CANNBOT_FRAME_LAUNCH {launch}\n')
    (root / 'cannbot_abi.json').write_text(json.dumps({
        'host_abi_version': abi, 'schema_version': schema_version,
        'schema_symbol': schema, 'invoke_symbol': invoke, 'adapter': header,
        'frame_capacity_and_alignment': 'read from runtime schema',
    }, indent=2) + '\n')
