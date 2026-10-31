# 环境探测与路径配置

## 任务参数

从用户指定的执行环境、现有项目配置和工具链中确定下列参数。路径、主机和容器名称由当前任务提供；skill 不保存个人环境配置。

| 参数 | 用途与来源 |
|---|---|
| `SKILL_DIR` | 当前 skill 的实际目录 |
| `WORK_DIR` | 本轮新建的独立工作目录；示例脚本要求尚不存在 |
| `CANN_ENV_SCRIPT` | 目标 CANN 安装提供的环境初始化脚本 |
| `ASCEND_HOME_PATH` | 初始化后使用的 CANN 根目录，供头文件与库定位 |
| `TILELANG_ROOT` | 选择源码模式时使用的 TileLang checkout，需包含配套 native 库 |
| `SOC_VERSION` | ATC 接受的完整目标 SoC 名，经 `--soc-version` 传入；可用目标设备的 ACL/runtime 探测，不能直接照抄简化显示名 |
| `NATIVE_FIXTURE_DIR` | ONNX/OM 复用的已验证固定 Add 导出/构建目录，经 `--native-fixture` 传入 |
| `NPU_ARCH` | 由目标芯片和当前 DSL/编译器支持范围确认，经 `--arch` 传入 |
| `DEVICE_ID` | 本次使用的逻辑 ACL 设备 ID，经 `--device` 传入 |
| `FIXED_OPP_ROOT` / `DYNAMIC_OPP_ROOT` | TorchAir 验证所用的两套已构建示例 OPP；经运行器对应参数传入，在独立进程加载 |
| `CXX11_ABI` | 与目标 GE 构建一致的 C++ 标准库 ABI，经 `--cxx11-abi` 传入 |
| `--stage-timeout` | 验证脚本每个子进程阶段的超时秒数；记录在报告中，超时时保留失败阶段与日志 |

下列命令假设上述变量已按当前环境设置；供 Python 示例读取的变量需通过 `export` 导出。

```bash
source "${CANN_ENV_SCRIPT:?set CANN_ENV_SCRIPT}"
python "${SKILL_DIR:?set SKILL_DIR}/scripts/probe_env.py"
```

SSH 和容器进入方式沿用当前任务配置；进入执行环境后再探测 Python、工具链和可用设备。工作目录与现有 DSL/GE 仓库分开。

## 芯片、设备与架构

用执行环境中的 `npu-smi info` 检查设备型号和可见设备，再核对 ACL 的逻辑设备映射与容器可见设备配置。设备号通过 `--device` 显式传入，不默认使用第一张卡，也不直接把物理设备编号当作逻辑 ACL 编号。

芯片型号与编译器架构名称分别确认。`--arch` 使用当前后端支持的目标名称，原样传到两个 DSL 的导出和重编译步骤，并写入 manifest。脚本不内置型号到架构的固定映射；交叉编译时使用明确指定的目标架构。

主机架构由 `platform.machine()` 或 CMake 的 `CMAKE_SYSTEM_PROCESSOR` 获取；OPP 检查脚本的 `--arch` 表示主机架构，与导出/验证脚本的 NPU `--arch` 含义不同。交叉构建时由目标 CMake toolchain 和 OPP 检查参数确定主机架构。

示例中的 tile 大小、线程数、grid 和 UBUF 属于算子实现及 launch 契约，不用于识别芯片。TileLang launcher 的 grid/UBUF 从本次 host IR 提取；使用其他硬件或后端时，按其能力核对这些配置并重新做数值验证。

## ABI、核数与线程配置的来源

| 配置 | 属性 | 处理方式 |
|---|---|---|
| C++ 标准库 ABI | GE/工具链构建环境 | 根据 GE 构建配置或实际链接验证确认，经 `--cxx11-abi` 传给 CMake；不沿用编译器默认值或示例历史值 |
| cannbot host ABI/schema | 本次产物的调用协议 | 从源码 marker、manifest 和导出符号识别并交叉核对，选择匹配的 frame 解码器 |
| frame 容量、对齐、slot offset | 产物 schema | 执行时读取 schema，按实际容量和对齐分配并填写 frame |
| 线程数、tile、算法 block 数 | 算子实现/特化条件 | 保留示例自身的数值配置；换算子或调优时作为特化参数重新编译 |
| TileLang grid、动态 UBUF | codegen 产物的启动契约 | 从本次 host IR 提取，不以设备总核数代替 grid，也不覆盖为历史值 |
| 主机编译并行数 | 构建资源 | 默认探测可用 CPU 数，可用 `--build-jobs` 覆盖；与 NPU 核数无关 |

`assets/ge-common/` 目前提供 host ABI v9/schema v2 的具体 wire-format 解码器，按 schema 填充 identity 布局的 tensor slot（不限槽数、rank、dtype）。编码存储布局需要额外的 physical shape/stride，当前与非 tensor slot（标量等）一样显式拒绝；扩展时从真实产物核对相应布局。版本号、magic 和结构大小属于该协议实现的常量；准备阶段先识别实际产物，再选择实现。遇到未支持的协议时需补充对应解码器，不能仅把版本号作为参数改掉而继续解释旧布局。C++ 标准库 ABI 与 cannbot host ABI 是两层不同的契约。

通用 frame 填充已有 CPU stub 验证，覆盖四个 slot、rank 0–3 与不同 dtype 的元数据，以及 Add 包装的字节等价性；实际 NPU/GE 验证仍只覆盖三个 slot、rank-one FP32 的 Add。多维、多输出或其他 dtype 的设备调用尚未验证，不能把通用填充能力当作这些算子的接入通过结论。

示例的 256 线程与 256 元素 tile、单 block launch，以及固定输入尺寸，是验证算子的实现选择；它们不表示设备拥有这些数量的线程或核心。保留这些约束有助于复现最小算子，移植时先核对后端能力，再调整实现并验证。运行时 frame 的三个 slot、FP32、连续一维布局也属于该 Add 示例的接口契约。

## 按实际安装探测

`probe_env.py` 只读记录 Python/主机架构、distribution 版本、模块位置、编译工具、CANN 环境和 `graph/custom_op.h` 中可见的接口。未配置 CANN 根目录时返回空值，不猜测安装目录。

cannbot-dsl 与 TileLang 不要求固定版本号。适配前检查实际安装的 API 签名、生成格式、目标后端及 ABI；使用源码树时额外记录 commit、dirty 状态和实际 native 库。包版本相同也可能来自不同构建。

TileLang 的源码与 wheel 可能有不同的 codegen 实现。`--tilelang-root` 用于显式选择源码树，导出脚本核对最终导入位置。头文件存在某个接口仅证明声明可见，还需核对已加载 GE Compiler/Executor 库与实现是否配套。

## 记录与复现

每次运行在任务工作目录保存环境日志、产物 manifest、已加载库路径/哈希、数值和 GE 执行结果。源码导出与重编译阶段可使用交叉编译，实际 NPU 运行需要单独记录设备及执行证据。

TorchAir 的 `probe()` 另行记录 torch ABI 和实际公开 API；GE ABI 与 torch ABI 分别确认，不要求两者相同。

固定 shape 参考工程见 [ge-smoke.md](ge-smoke.md)；动态缩小及地址变化见 [dynamic-shrink.md](dynamic-shrink.md)。其中的历史通过结果属于对应示例，不替代新环境中的适配验证。
