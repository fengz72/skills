# 分层验证与交付证据

## 源码导出

- 使用新输出目录；导出命令退出成功，文件非空，确实包含目标后端的设备入口。
- cannbot-dsl 保存 ABI marker、schema/invoke 符号及 manifest；TileLang 保存源码、参数、host/device IR 和匹配模板头文件。
- 记录 Python 导入路径，避免源码与 wheel 混用。记录目标架构、所有特化参数和实际 commit/dirty。
- 只有代码生成结果时明确报告到此层，不宣称数值正确或模型接入通过。

## 从源码重编译

- 在当前 CANN 下重新编译导出源码，不复用 DSL JIT 缓存中的既有 binary。
- cannbot-dsl 的 `.so` 用 `nm -D --defined-only` 确认导出的 schema/invoke 符号匹配 manifest，并用 `ldd` 检查依赖。
- TileLang 用导出目录的模板头文件构建 `.aibin`，检查 executable CCE ELF，不能只检查文件扩展名。
- 更换工作目录后仍能定位交付文件。依赖头文件/工具链的重编译与部署是否需要 Python DSL 分别说明。

## kernel 数值与语义

按算子语义使用独立参考实现和合理精度标准；比较绝对/相对误差，记录 dtype、seed、shape 与失败样本。不要只以两个共享同一 kernel 的包装路径互相比对来证明正确。

覆盖所声称支持的 dtype/layout/shape 范围、边界/尾块、重复执行和数值变化。按本次任务选择 shape 重复、增大、缩小或回升的序列，分别记录各场景结果和未覆盖项。动态输出越界的定位方法见[错误排查案例](troubleshooting.md)。有数据驱动 planner 时保持 shape 不变而改变数据内容，验证 metadata 确实刷新。

## GE 与模型链路

先单算子 GE 图，再目标模型子图/端到端。确认图中实际使用目标自定义 type；检查 OPP 库加载和 converter 属性，避免 eager fallback 掩盖未入图。

需要时覆盖：

- 多次模型执行、实际 I/O 地址变化、workspace 复用。
- 对地址变化分别记录调用方与 kernel 实际地址。保留旧分配保证地址真正不同，比较旧缓冲区快照；外部输出逻辑尾部设哨兵，避免较大物理容量掩盖按旧 shape 继续写的问题。
- 动态输出容量、输入未被改写、外部缓冲区边界。
- 多模型/节点与 context 资源隔离、模型卸载后的清理。
- 新进程加载交付件；离线 OM 在关闭原始源码/缓存依赖后加载执行，验证产物是否真正自包含。

这些检查按真实功能选择。未声称并发/多 context 支持时不以单模型串行测试推断这些能力。

TorchAir 两个 DSL 的固定/动态 Add、输出引用与地址变化已完成实测，运行器与第④层证据见 [torchair.md](torchair.md)。特别区分单位维特化产生的新图、普通动态输出所有权，以及 functionalization 临时地址。ONNX/OM 的两个固定 Add 已通过解析插件正反对照、OM GE 节点断言、隔离加载及 NPU 精度，见 [onnx-om.md](onnx-om.md)。其自包含通过依据是 SO_BINS 内嵌实现及 kernel，不包含 PortableOp 序列化回调的验证结论。

## 配置参数化后的回归

在新目录分别运行固定 shape 工程和动态缩小/地址变化工程，从 DSL 源码重新导出并重编译，再完成直接 NPU 与 GE 图数值比较。C++ 编译链接和 CPU frame 测试与实际 NPU/GE 执行分别记录结果。

对照运行命令与 `results.json` 核对设备、目标架构、C++ ABI 和构建并行数；进一步检查 CMake cache 与编译 flags、导出 manifest 的目标及模块来源、`cannbot_abi.json` 的解码器选择，以及 `tile_launch_metadata.json` 的实际 grid/UBUF。运行日志记录实际设备号和调用次数，确认参数进入构建及执行环节。

动态场景检查同一模型多次执行、每步实际 shape、节点/实现对象复用及输入输出地址证据。外部输出轮换与显式 AddOut 引用分别判断；AddOut 路径逐步核对 kernel 与调用方指针，并检查旧地址快照和当前逻辑尾部。

## 报告内容

交付位置和复现命令、环境/commit、支持范围、产物 manifest、每层结果、误差指标和已知限制。失败时区分 codegen、ABI/链接、注册/加载、shape/属性、launch/stream、数值、模型集成，保留最小失败样本。

环境探测与路径配置见 [environment.md](environment.md)，各示例的历史验证结果见对应参考文档。

完整可复现实例见 [ge-smoke.md](ge-smoke.md) 与 `scripts/run_ge_smoke.py`。它从新目录重新导出、重编译和执行两个 DSL Add；通过依据包含 GE 三个阶段、实际回调、六组数值比较和清理。静态 Session 输出描述符初始化、host 回调仅在加载时录制、dynamic UBUF launch 属性均有实测修正。

动态 shrink-only 入口见 [dynamic-shrink.md](dynamic-shrink.md) 与 `scripts/run_dynamic_shrink.py`。输出引用场景有明确的 AddOut 语义，不能与普通 Add 的外部输出拷贝混为同一种地址更新；GE 自动分配的内部输出没有外部哨兵覆盖。
