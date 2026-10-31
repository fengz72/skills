"""TorchAir adapters for the fixed/dynamic GE Add fixtures.

OPP location and device are configured by the caller before TorchAir initializes.
The shapes below are operator specializations, not hardware settings. Each
process loads one fixture OPP: the static/dynamic native libraries share C symbols.
"""
import ctypes
import importlib
import inspect
from pathlib import Path
import platform
import re
import torch

DOMAIN = "dsl_ge"
SPECS = {}
for _backend, _fixed in [("cannbot", 256), ("tilelang", 524288)]:
    _title = "Cannbot" if _backend == "cannbot" else "Tilelang"
    for _scope, _out in [("fixed", False), ("dynamic", False), ("dynamic", True)]:
        _name = "add" if (_backend, _scope) == ("cannbot", "fixed") else f"{_backend}_{_scope}_add"
        if _out:
            _name += "_out"
        SPECS[_name] = {"name": _name, "backend": _backend, "scope": _scope, "out": _out,
                        "elements": _fixed if _scope == "fixed" else None,
                        "ge_type": _title + ("AddSmoke" if _scope == "fixed" else "DynamicAdd" + ("Out" if _out else ""))}

_lib = torch.library.Library(DOMAIN, "DEF")
_case_tag = "model"
_native_library = None
_loaded_root = None
_converters_registered = False


def _check_tensor(x):
    if x.dtype != torch.float32 or x.layout != torch.strided or not x.is_contiguous() or x.dim() != 1:
        raise ValueError("GE Add fixtures require contiguous rank-one FP32 tensors")


def _meta(spec):
    def implementation(x, y, out_buffer=None):
        _check_tensor(x)
        _check_tensor(y)
        torch._check(x.shape[0] == y.shape[0])
        n = x.shape[0]
        if spec["scope"] == "fixed":
            torch._check(n == spec["elements"])
        else:
            torch._check(n >= 1)
            torch._check(n <= 4096)
        if spec["out"]:
            _check_tensor(out_buffer)
            torch._check(out_buffer.shape[0] == n)
            return None
        return torch.empty_like(x)
    return implementation


def _no_eager(*args, **kwargs):
    raise NotImplementedError("This fixture requires its TorchAir converter; no eager fallback is provided")


for _name, _spec in SPECS.items():
    _schema = "(Tensor x, Tensor y, Tensor(a!) out_buffer) -> ()" if _spec["out"] else "(Tensor x, Tensor y) -> Tensor"
    _lib.define(_name + _schema)
    _lib.impl(_name, _meta(_spec), "Meta")
    _lib.impl(_name, _no_eager, "PrivateUse1")


def get_spec(backend, scope, *, out=False):
    return next(dict(s) for s in SPECS.values() if (s["backend"], s["scope"], s["out"]) == (backend, scope, out))


def get_op(backend, scope, *, out=False):
    return getattr(getattr(torch.ops, DOMAIN), get_spec(backend, scope, out=out)["name"]).default


def set_case_tag(value):
    global _case_tag
    if not re.fullmatch(r"[A-Za-z0-9_]+", value):
        raise ValueError("Case tag must contain only letters, digits and underscores")
    _case_tag = value


def load_opp(root):
    """Load this fixture's GE prototype/implementation before frontend lowering."""
    global _native_library, _loaded_root
    root = Path(root).resolve(strict=True)
    if _loaded_root is not None and root != _loaded_root:
        raise RuntimeError("Static/dynamic fixture libraries have overlapping C symbols; use separate processes")
    if _native_library is None:
        library = root / "op_graph" / "lib" / "linux" / platform.machine() / "libcust_opapi.so"
        _native_library = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        _native_library.SmokeExecuteCount.restype = ctypes.c_uint
        _loaded_root = root
    return _native_library


def _converter(spec, ge):
    if spec["out"]:
        def convert(x, y, out_buffer, meta_outputs=None):
            ge.custom_op(spec["ge_type"], inputs={"x": x, "y": y, "out_buffer": out_buffer},
                         outputs=["out_buffer"], node_name=spec["ge_type"] + "_" + _case_tag)
            # The torch schema returns nothing; functionalization carries the
            # mutated destination through the GE input/output ref relation.
            return None
    else:
        def convert(x, y, meta_outputs=None):
            result = ge.custom_op(spec["ge_type"], inputs={"x": x, "y": y}, outputs=["z"],
                                  node_name=spec["ge_type"] + "_" + _case_tag)
            # Dynamic EagerExecuteOp may expose its cached output to TorchAir.
            # Explicit GE TensorMove materializes an owned graph output; this
            # is different from aten.clone, which may lower to Identity.
            return ge.TensorMove(result) if spec["scope"] == "dynamic" else result
    return convert


def register_converter():
    """Register all fixture converters once; importing/probing does not register."""
    global _converters_registered
    if _converters_registered:
        return
    torchair = importlib.import_module("torchair")
    ge = importlib.import_module("torchair.ge")
    register = getattr(torchair, "register_fx_node_ge_converter", None)
    if not callable(register) or not callable(getattr(ge, "custom_op", None)) or not callable(getattr(ge, "TensorMove", None)):
        raise RuntimeError("Installed TorchAir lacks required public APIs; inspect probe() before adapting")
    for spec in SPECS.values():
        register(getattr(getattr(torch.ops, DOMAIN), spec["name"]).default)(_converter(spec, ge))
    _converters_registered = True


def _api_description(value):
    if not callable(value):
        return {"callable": False}
    try:
        signature = str(inspect.signature(value))
    except (TypeError, ValueError):
        signature = None
    return {"callable": True, "signature": signature}


def probe():
    report = {"torch_version": torch.__version__, "torch_cxx11_abi": getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None)}
    try:
        torchair = importlib.import_module("torchair")
        ge = importlib.import_module("torchair.ge")
    except ImportError as error:
        report["torchair_error"] = str(error)
        return report
    report["torchair_module"] = getattr(torchair, "__file__", None)
    for name in ["register_fx_node_ge_converter", "get_npu_backend", "CompilerConfig"]:
        report[name] = _api_description(getattr(torchair, name, None))
    for name in ["custom_op", "TensorMove"]:
        report[name] = _api_description(getattr(ge, name, None))
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(probe(), ensure_ascii=False, indent=2, default=str))
