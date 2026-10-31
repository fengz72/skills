# TileLang → AscendC

## 后端与导入来源

通过 `importlib.metadata.version('tilelang')`、实际模块位置和源码信息探测当前安装，确认选中的实现包含目标芯片的 Ascend 后端。本指南使用 Ascend codegen 路径；其他后端需核对对应实现。

默认 Python 导入 site-packages。为使用用户指定的源码树，脚本在 import 前将 `--tilelang-root` 放到 `sys.path` 首位，并检查最终 `tilelang.__file__`。源码树与安装包的 `ascend/codegen.py` 可能不同，版本号相同也不能视为同一构建。使用源码树时还须保证其 `build/` 下 native 库与 Python 源码配套，不混入其他 TVM/TileLang 动态库。

## 首选：只 lowering/codegen，不生成运行时模块

算子 factory 返回 `PrimFunc`；已有 `@tilelang.jit` 对象可在核实本版本 API 后用 `get_tir(...)` 取出 PrimFunc，再提供简单 factory。本脚本的 `--args`/`--kwargs` 为 JSON，复杂对象在 factory 内显式构造。

```python
import os
from tilelang import tvm
from tilelang.engine import lower

target = tvm.target.Target({"kind": "ascend", "arch": os.environ["NPU_ARCH"]})
with target, tvm.transform.PassContext(opt_level=3):
    artifact = lower(
        prim_func, target=target,
        enable_host_codegen=False,
        enable_device_compile=False,
    )
ascendc = artifact.kernel_source
```

**显式 target context 是本 checkout 实测需要的。** 仅向 `lower()` 传 `target='ascend'` 时，SIMT vector-add 在 `AscendLayoutInference` 内报 `Target context required`；外层 `with target` 后通过。避免在 skill 中修补 TileLang 的用户仓库。

## 导出命令示例

变量设置见 [environment.md](environment.md)。示例使用源码树中的 vecadd factory，运行前核对其存在及签名。

```bash
source "${CANN_ENV_SCRIPT:?set CANN_ENV_SCRIPT}"
python "$SKILL_DIR/scripts/export_tilelang.py" \
  --tilelang-root "${TILELANG_ROOT:?set TILELANG_ROOT}" \
  --factory "$TILELANG_ROOT/examples/ascend/example_vecadd.py:vector_add" \
  --args '[524288, "simt"]' \
  --arch "${NPU_ARCH:?set NPU_ARCH}" --output "${WORK_DIR:?set WORK_DIR}" --compile-aibin
```

`example_vecadd.py` 的 `vector_add` 支持 `simt`/`simd`，要求 N 是 `8192*64` 的正整数倍。换成 `simd` 使用另一个新输出目录。这里选择最小的合法 N，不运行示例默认的大规模性能测试。

导出内容：

| 文件 | 接入用途 |
|---|---|
| `kernel.asc` | 设备 kernel 与相关 VF 实现 |
| `host_ir.txt` | kernel 调用顺序、实参、block/grid、workspace 等启动信息 |
| `device_ir.txt` | 设备侧 lowering 结果及入口属性 |
| `params.tvm.json` | 参数顺序、dtype、shape 的 TVM JSON，可用本版本 `load_kernel_params` 读取 |
| `include/tl_templates/ascend/` | 随本次导出复制的匹配模板头文件 |
| `manifest.json` | 实际导入路径、commit/dirty、target、factory 参数、源码和头文件哈希 |
| `kernel.aibin` | 仅在 `--compile-aibin` 时产生的 executable CCE ELF |

## 模板依赖和 binary 生成

导出的 AscendC 含 `#include <tl_templates/ascend/common.h>` 等 include。必须同时交付匹配的模板头文件；本脚本复制 `TILELANG_TEMPLATE_PATH/tl_templates/ascend` 并使用导出目录的 `include/` 重编译。CANN 的 `c_api`/`simt_api`/ACL 头文件仍来自工具链，并非完全独立于 CANN。

本版本的重编译入口：

```python
import os
from pathlib import Path
from tilelang.contrib.bisheng import compile_ascend
export_dir = Path(os.environ["WORK_DIR"])
binary = compile_ascend(
    ascendc, target_format="aibin", npu_arch=os.environ["NPU_ARCH"],
    options=[f"-I{export_dir / 'include'}"],
    path_target=str(export_dir / "kernel.aibin"),
)
```

`aibin` 经过 bisheng 设备编译与 aicore ELF 链接，可用于相应 ACL binary-load 路径。`target_format='o'`、`'so'`、`'aibin'` 的含义不同；重定位 `.o` 不能直接当作可加载执行的 `.aibin`。相关命令以 `tilelang/contrib/bisheng.py` 为准。

## GE launcher 的下一步

`kernel.asc` 通常只有设备入口，**没有 cannbot-dsl 的 host frame ABI**。GE 层需额外封装加载、函数查找、参数布局及 launch，或从该源构建带明确 C ABI 的 host wrapper。

从 host/device IR 提取入口名、实参顺序、标量和动态维、block 数及临时空间需求。首次验证的 SIMT vecadd host IR 调用为 `main_kernel(A, B, C, 64, 196608)`；64 为 grid，196608 为 **dynamic UBUF bytes**。后者通过 `ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE`、`attr.value.dynUBufSize`、`aclrtLaunchKernelCfg` 配置到 `aclrtLaunchKernelWithHostArgs`，不是用 `MallocWorkSpace` 分配的全局设备内存。设备函数签名只有 A/B/C，不能追加这两个元数据为普通参数。此处数值只对这个特化有效。

上述 binary-load、function lookup、grid 和动态 UBUF 配置已在 [GE smoke 工程](ge-smoke.md)中完成真实 NPU 和 GE 静态图验证；工程按 context 隔离句柄，在模型卸载后、GEFinalize 前显式卸载 binary。其 `.aibin` 嵌入插件编译单元，不依赖启动目录或 Python JIT 缓存。

动态维示例见 [dynamic-shrink.md](dynamic-shrink.md)：使用 `T.dynamic('N', dtype='int64')` 后，设备入口增加实际的 `int64_t N` 参数，位于三个指针之后；grid 和 UBUF 仍是 launch 元数据。必须重新检查设备签名/host IR，不能复用固定 shape 的三指针参数包。当前模板使用单调不增的 shape，不执行输出扩容。

使用 GE stream，binary/函数句柄按实际 context 和设备隔离并管理生命周期；输出和临时空间由 GE context 分配。离线 OM 可将 binary 与启动元数据嵌入实现 SO，随 SO_BINS 打包；运行时生成的实例状态可通过 PortableOp 序列化。加载后重新建立运行时句柄，不保存地址。固定 Add 的前一种方式已完成隔离验证，见 [onnx-om.md](onnx-om.md)。

已有 JIT 编译结果也可通过 `get_kernel_source()` 取得源码；这是补充路径，可能先触发 binary 编译或后端初始化。只需要 codegen 时优先用 `lower()`，不必创建 NPU 张量。

## 版本变动时的源码定位

- `examples/ascend/example_vecadd.py`：最小真实 factory 及形状条件。
- `tilelang/engine/lower.py`、`param.py`：源码返回字段、IR 和参数序列化。
- `tilelang/ascend/codegen.py`、`backend.py`：plain AscendC 与 PTO 后端选择。
- `tilelang/contrib/bisheng.py`：架构选项、编译格式和链接行为。
- `src/tl_templates/ascend/`：导出源码的配套头文件。

PTO 和 plain Ascend 可能共用 target kind，但产物不同。本脚本明确使用 plain Ascend target，未验证 PTO 路径；不得把 `.pto`/PTO DSL 改扩展名伪装成 AscendC。
