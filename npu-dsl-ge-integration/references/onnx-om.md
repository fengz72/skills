# ONNX 解析、OM 导出与自包含交付

## 已验证的链路与范围

`assets/ge-onnx-om/` 和 `scripts/run_onnx_om.py` 已完成两个 DSL 的 ONNX → ATC → GE 自定义节点 → OM → ACL/NPU 链路，支持范围为连续一维 FP32 固定 Add：

| DSL | ONNX 标识 | GE type | shape |
|---|---|---|---|
| cannbot-dsl | `com.dsl_ge::1::CannbotAdd` | `CannbotAddOm` | `[256]` |
| TileLang | `com.dsl_ge::1::TilelangAdd` | `TilelangAddOm` | `[524288]` |

两套模型分别在隔离的新进程中加载一次，各执行 6 组数据，共 12 次、3,147,264 个元素，最大绝对误差和 RMSE 均为 0；输入未改变，输入输出前后哨兵通过。模型的实现 SO 和 kernel 原始字节完整嵌入 OM，自定义 OPP、原始源码、导出目录和用户缓存被屏蔽后仍通过。此结论限于实际验证的同机兼容环境、固定 shape 与顺序执行，不代表动态 shape、跨芯片/架构/运行时版本或并发已验证。

本轮复用此前通过验证的固定 Add DSL 导出件，重新构建 ONNX 插件、GE 实现和 OM；不是重新执行 DSL codegen。需要验证新 DSL 代码时先完成 [ge-smoke.md](ge-smoke.md) 的导出、重编译及数值流程。

## 运行参数与入口

初始化目标 CANN 环境；确认 ONNX、ATC、C++ 编译器及 ACL 可用。`NATIVE_FIXTURE_DIR` 指向已验证的固定 Add 工作目录，含 `cannbot-export/`、`tilelang-export/` 的 manifest、源码、host IR 和重编译 binary。该入口复用这些 binary，报告记录哈希及未重新导出的事实。

`SOC_VERSION` 应取目标 ACL/runtime 支持的完整 SoC 名称，经 `--soc-version` 传入，不能直接照抄 `npu-smi` 的简化显示名。可在目标设备上用配套 Python ACL 的 `init`、`rt.set_device(DEVICE_ID)`、`get_soc_name()` 获取，并检查返回码、最后 reset/finalize；交叉编译则显式提供已确认的目标值。C++ ABI、设备、构建并行数均沿用 [environment.md](environment.md) 的参数纪律。

```bash
source "${CANN_ENV_SCRIPT:?set CANN_ENV_SCRIPT}"
python "${SKILL_DIR:?set SKILL_DIR}/scripts/run_onnx_om.py" \
  --native-fixture "${NATIVE_FIXTURE_DIR:?set NATIVE_FIXTURE_DIR}" \
  --device "${DEVICE_ID:?set DEVICE_ID}" \
  --soc-version "${SOC_VERSION:?set SOC_VERSION}" \
  --cxx11-abi "${CXX11_ABI:?set CXX11_ABI}" \
  --build-jobs "${BUILD_JOBS:?set BUILD_JOBS}" \
  --work-dir "${WORK_DIR:?set a new WORK_DIR}" \
  --stage-timeout "${STAGE_TIMEOUT:-600}"
```

运行器的强隔离验证使用 Linux `unshare --mount` 和私有挂载，需目标容器允许这些操作。开始隔离后先将挂载传播改为 private，避免影响其他进程。环境不支持时保留失败阶段；可另用干净部署容器完成等价验证，但不能将仅清空环境变量的测试标记为同等隔离通过。这是本自包含验证夹具的实现条件，不是 ONNX/OM 推理本身的要求。

## ONNX → GE 的解析证据

插件位于独立 OPP 的 `framework/onnx/`，GE 实现位于 `op_graph/lib/linux/<host-arch>/`。`parser.cc` 用 `REGISTER_CUSTOM_OP` 注册 `domain::version::OpType`，通过 `ParseParamsByOperatorFn` 与 `AutoMappingByOpFn` 完成当前无自定义属性的 Add 映射，无需直接依赖 ONNX protobuf C++ 类型。换成有属性的算子时逐项核对 int/float/string/list、GE 类型及双方默认值；不能用这个无属性示例代替属性转换验证。

运行器在 ATC 初始化前为各阶段设置自定义 OPP。先暂时移开本轮新建的 parser 目录，确认 ATC 因对应自定义类型缺少解析插件而失败，再恢复插件重编译；恢复采用 finally 路径。随后同时检查：

- `onnx.checker.check_model(..., check_custom_domain=False)` 结构检查通过。示例 domain opset 1、标准 opset 13、IR 8 是模型的协议选择，不是安装版本要求；未提供自定义 schema，所以 checker 不证明语义已被识别。
- ATC 日志确有解析回调成功，正向编译生成 OM。
- 用 `atc --mode=1 --om=... --json=...` 从 OM 导出 JSON，断言恰有一个目标 GE type，保留解析后属性、I/O 描述及 GE dump。
- ACL 执行时确有对应后端的自定义 Execute 日志，kernel 哈希匹配打包记录；不是映射成内置 Add 后仅看结果。

本例要求 ND，ATC 命令显式传 `--input_format=ND`。ONNX 本身不携带 GE format，默认入口可能赋成 NCHW；实际遇到 rank/元素数正确而 Execute 因 format 契约不符拒绝加载的情况。应对齐前端格式与 kernel 支持范围，不应简单删除检查。

## 本例的自包含方式：SO_BINS

GE 可将已注册的 C++ 自定义算子实现 SO 打包进 OM，即使该类没有 `PortableOp`。但这不表示 ELF 的 `DT_NEEDED` 依赖会递归打包。原固定 fixture 的 GE 库依赖外部 `libcannbot_add.so`；直接导出 OM 成功，独立加载却因这个库缺失失败，连使用同一 GE 库的 TileLang 模型也受到影响。

离线模板改为每个后端独立构建一个实现 SO：

- cannbot-dsl：完整 host/device SO 字节嵌入实现库，加载 OM 后经 memfd/dlopen 恢复，按实际导出符号和 host ABI/schema 解码 frame。没有外部 kernel 文件路径或 DSL JIT 调用。
- TileLang：原始 `.aibin` 字节和 host IR 提取的 kernel 名、grid、dynamic UBUF 嵌入实现库，按当前 context 载入 binary，并使用 GE 提供的 stream。
- 实现库链接目标 GE 的 `custom_op_registry_static`，导出 pull creator ABI；本模板显式保留静态注册对象并绑定库内函数。准备后检查三个 `GetRegisteredCustomOpCreator*` 入口，不能只凭编译成功推断 OM 可恢复注册。

环境 ABI 显式传给 CMake；cannbot wire-format decoder 从产物识别，TileLang grid/UBUF 从真实 host IR 提取。固定尺寸、后端编号和 Add 参数布局属于算子示例契约。

## 隔离与数值验收

`results.json` 同时保存以下证据：

1. kernel 哈希匹配本轮所用导出 binary；该 binary 是实现 SO 的完整字节子串，实现 SO 又是 OM 的完整字节子串。记录 binary、实现库与 OM 的 SHA-256，而不是仅搜索某个名称。
2. 检查实现库 `ldd` 无缺失依赖、不再链接外部 Add kernel。部署目录仅包含两份 OM 与只链接 ACL/runtime 的 C++ 执行器。
3. 新进程的私有挂载命名空间以只读方式暴露部署目录，屏蔽开发工作区所在的用户目录以及用户和临时缓存，取消自定义 OPP、Python 路径与预加载变量；保留声明的 CANN/ACL、驱动和基础系统库。该步骤不重命名或删除共享源码仓库。
4. 从新路径加载每个 OM，校验 I/O 数量、shape、dtype、字节数；重复执行零值、常量、规律值、随机、抵消及较大幅值六组输入。独立 CPU FP64 加法转 FP32 作为参考，容差为 `1e-6 + 1e-6 * abs(reference)`。同步执行后检查所有元素、输入与前后 guard，最后卸载模型并检查设备/ACL 清理。

“自包含”指自定义实现和 kernel 不需要外部交付件；仍依赖兼容的 CANN/GE/ACL runtime、驱动、基础系统库、host 架构和 NPU。内置运行时/算子包仍可见；没有验证完全移除 CANN 或跨机器兼容性。运行器保存阶段超时/失败、日志、ONNX、OM JSON、GE dump 和独立部署目录，所有实际环境路径与版本留在任务报告。

## 何时需要 PortableOp

若 kernel 是 Compile 阶段生成并只存在对象成员中的数据，或希望把多特化 binary 与 launch 元数据作为实例状态随 OM 保存，使用 `PortableOp::Serialize/Deserialize`；本轮没有实现或执行该分支，不能把 SO_BINS 的通过结论写成 PortableOp 回调通过。

设计格式时保存 magic/version、条目数、key/长度边界、binary、grid/UBUF/参数布局与特化元数据；拒绝截断、重复 key、未知版本及尾随数据。运行时指针、stream 和句柄不序列化；恢复后按 context 重建。外置 artifacts 也是可交付方式，但必须列明依赖及发现路径，并单独验证迁移后的完整部署包。

接口依据为目标安装的 GE 头文件与[官方开发指南](https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/custom_op/custom_op_v2/custom_op_development_guide.md)，checker 范围参见 [ONNX 官方实现](https://github.com/onnx/onnx/blob/main/onnx/checker.py)。错误案例见 [troubleshooting.md](troubleshooting.md)。
