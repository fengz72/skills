# cannbot-dsl → AscendC

## 探测安装与编译入口

distribution 名为 `cannbot-dsl`，Python 导入名是 `cannbotdsl`。通过 `importlib.metadata.version('cannbot-dsl')` 和 `importlib.util.find_spec('cannbotdsl').origin` 探测实际版本与源码位置，再核对编译入口。

本适配路径使用的 `cb.compile()` 接受 `@cb.host` 函数/绑定方法和 `TensorSpec`、标量、constexpr 等参数，完成编译但不执行 kernel。源码导出使用：

```bash
export CANNBOTDSL_PIPE_STAGE=translate
export CANNBOTDSL_DUMP_ASCENDC=1
export CANNBOTDSL_DUMP_DIR="$PWD/export-fp32"
export CANNBOTDSL_NPU_ARCH="${NPU_ARCH:?set NPU_ARCH}"
python aot_driver.py
```

把配置限制在导出子进程内更合适：本 skill 的 `scripts/export_cannbot.py` 已这样处理，同时关闭与 DUMP 冲突的 `REUSE_ASCENDC`。检查实际安装实现中 translate/dump 与编译缓存的关系，确认本次确实重新导出；排查过程无需清理全局缓存。

## 可运行的源码导出示例

先按 [environment.md](environment.md) 设置环境变量。`WORK_DIR` 为本次新输出目录：

```bash
source "${CANN_ENV_SCRIPT:?set CANN_ENV_SCRIPT}"
python "$SKILL_DIR/scripts/export_cannbot.py" \
  --driver "$SKILL_DIR/assets/cannbot_copy.py" \
  --arch "${NPU_ARCH:?set NPU_ARCH}" --output "${WORK_DIR:?set WORK_DIR}" --compile-so
```

driver 的实质：

```python
spec = cb.TensorSpec((256,), cb.dtypes.float32)
cb.compile(copy_host, spec, spec)
```

示例是固定 256 元素 FP32 copy，仅验证 codegen/重编译机制。替换算子时写独立 AOT driver，复用真实 `@host` 入口并按其签名传入描述符；不要为了导出运行整个模型。

需要实际接入参考时使用 [GE smoke 工程](ge-smoke.md) 的 `assets/ge-smoke/cannbot_add.py`，它使用本安装包支持的 `vf(mode='simt', thread=256)` 实现 256 元素 FP32 Add，已通过直接 NPU 与 GE 静态模型执行。它是为接入验证编写的独立最小算子，并非从官方 samples 复制；浏览[官方 samples](https://gitcode.com/cann/cannbot-dsl/tree/master/samples)选择其他算子时仍需核对当前安装版本的 API，不能因 distribution 版本号相同就跳过核实。

动态维与地址变化见 [dynamic-shrink.md](dynamic-shrink.md)。AOT 用 `Dim` 描述维度，host frame 每轮填写实际 shape；动态 shape 的读取要放在 SIMT VF 外，否则当前版本报 `cannir.extract_layout` 不允许在 VF 内。不要把“每个固定 shape 导出一份 binary”误报为同一 kernel 的动态 shape 验证。

`translate` 阶段返回的 callable 是早停占位结果，不能用来做数值验证。driver 应只定义/导入 kernel 并调用 `cb.compile`，不要在后面执行、读取输出或声称结果正确。导入模块本身如果会执行 NPU 操作，先把入口与顶层执行分开。

产物为每个 host 函数的 `.asc`、导出日志、manifest；`--compile-so` 额外从 `.asc` 重新构建 `.so`。同一个 host 函数的多个特化可能写同名文件，因此每个 shape/dtype/constexpr 特化分别导出到新目录。

## GE launcher 所需契约

完整 `.asc` 包含 device 实现与 host ABI。导出脚本读取 `CANNBOTDSL_HOST_ABI_VERSION` marker，记录带版本后缀的 schema/invoke 符号。GE 示例的准备阶段再核对 marker、manifest 和同一特化的符号，按实际版本选择 `assets/ge-common/` 中的解码器，写出 `cannbot_abi.json`。

符号含特化 hash，重新 codegen 后从产物提取，不手写旧符号。脚本把实际导出符号和 ABI 版本记入 manifest。

ABI 处理、已实现的协议范围和配置分类见 [environment.md](environment.md)。GE 层按 schema 填充 frame：参数 slot 的 kind、dtype、offset、size、alignment，tensor 的 pointer、shape、stride，以及 header 的版本/大小/digest，均来自匹配的产物契约。通用填充入口为 `cannbot_frame_v9_schema_v2::LaunchTensors`，支持 identity 布局下任意槽数、rank 和 dtype 的 tensor slot；编码布局的 physical shape/stride 尚未实现，和标量等非 tensor slot 一样显式拒绝，需从产物核对布局后扩展。`LaunchTensors` 只填充描述符，不转换数据 dtype 或存储布局，调用方仍须提供与 schema 一致的实际数据。Add 的 `Launch(x,y,z,n,stream)` 包装保留三个 FP32/rank-one slot 的检查；其他 rank/dtype 使用通用入口。frame 使用本次 GE 输入/输出/workspace 地址，stream 使用 `ctx->GetStream()`。不要只从 kernel 参数的 C++ 拼写猜测 ABI。

本 skill 的可选重编译调用 `cannbotdsl.core.toolchain.ascendc.AscendCCompiler.compile_ascendc_to_so`，这是内部 API，使用前检查实际安装的模块和签名。它用于导出阶段，部署时可将记录的 bisheng 编译步骤固化到构建工程，避免让 GE 执行依赖 Python/cannbotdsl。

当前 CMake 对没有 `lib` 前缀的导出库名可能生成不可解析的 `-l<name>`。GE smoke 工程把导出库复制为 `libcannbot_add.so` 并配置相对 RPATH，已核对实际 DT_NEEDED。构建与部署不能只凭导出 `.so` 存在就认定依赖可加载。

如果算子含 AICPU planner，主 `.asc` 并不能替代 planner 源码。对实际 `cannbotdsl.aicpu.Kernel` 对象调用 `generate_sources(launch_mode='interface')` 获取源码映射，保留编译所需 kernel/body/args/shim 文件。现有 PrefixAttention 脚本排除 `_x86.cpp` 是其部署选择；该扩展路径已阅读示例，本轮未额外执行验证。

## 版本变动时的源码定位

相对 `cannbotdsl` 包目录检查：

- `aot/compiler.py`：公共 AOT 入口和 early-exit callable。
- `core/compiler/builder.py`：translate、dump 和编译顺序。
- `core/compiler/cache.py`：dump/pipe stage 是否绕过缓存。
- `core/utils/env.py`：环境变量解析。
- `core/toolchain/ascendc.py`：目标架构、编译与链接选项。
- `core/compiler/abi.py`、`host_schema.py`：当前 ABI/契约实现。

常见定位：没有 `.asc` 时确认 driver 真正触发编译；只有导入不会 codegen。遇到旧缓存/复用源码时先核对版本与本次 dump 配置，不删除全局缓存。编译成功但 GE 执行失败时优先核对产物、schema、frame、特化约束是否配套。
