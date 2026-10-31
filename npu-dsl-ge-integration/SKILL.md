---
name: npu-dsl-ge-integration
description: 将 cannbot-dsl、TileLang 等 DSL 算子接入 Ascend GE custom_op_v2 模型，覆盖 PyTorch/TorchAir converter 与 ONNX parser 等前端映射；用于导出 AscendC 源码、确认 kernel ABI 与 launch 契约、实现 GE 注册、构建 OPP 交付件并验证模型执行。仅需源码导出时执行对应分支，不要求完成整条模型链路。
---

# DSL 算子接入 GE

把 DSL 算子的计算实现接入 GE，保留其真实的 shape、dtype、layout、特化条件和执行语义。说明与产物默认使用中文，代码标识符沿用项目习惯。

## 先确定本次交付范围

从用户给出的代码和环境识别：DSL/版本、目标 SoC、输入输出及属性、静态或动态 shape、是否包含 planner/多 kernel、目标前端，以及在线执行或离线 OM。能从源码、环境和模型定位的信息先自行检查。只询问会影响实现且尚不能确认的条件。

区分四层结果：①源码已导出；②源码可重新编译；③kernel 数值正确；④GE/目标模型接入通过。按本次任务做到所需层级，报告实际完成的层级；前一层通过不代表后一层。

## 工作流程与按需参考

1. **环境与契约**：在独立工作目录记录版本、源码 commit/改动状态、实际 Python 导入路径、CANN/GE Compiler/Executor、主机架构和目标 SoC。运行 `python scripts/probe_env.py` 获取只读基线；路径参数与环境准备见 [environment.md](references/environment.md)。源码导出允许交叉编译，但要求显式指定已确认的目标架构，不能从容器名推断芯片。
2. **导出和重编译**：
   - cannbot-dsl：读 [cannbot-dsl.md](references/cannbot-dsl.md)，用 `scripts/export_cannbot.py` 运行只做 AOT 编译的 driver，导出完整 host/device AscendC。
   - TileLang：读 [tilelang.md](references/tilelang.md)，用 `scripts/export_tilelang.py` 从返回 PrimFunc 的 factory 导出 AscendC、host/device IR、参数和模板头文件。
   - 其他 DSL：核实本版本的真实导出入口，确认得到的是 AscendC、PTO、IR 还是设备二进制。没有验证过的 DSL 不套用上述命令。
3. **制定 GE 适配**：读 [ge-integration.md](references/ge-integration.md)。先选执行链路，再选 `EagerExecuteOp`、shape/dtype 推导、`CompilableOp`、`PortableOp` 等所需接口。为输出和 workspace 使用 GE context，kernel 使用 GE 提供的 stream。从产物和 host IR 提取参数顺序、grid/block、workspace、特化键和 stream 依赖。
4. **构建和接入**：产出独立 OPP 根目录，保留已有环境变量。按目标前端实现 GE 构图、TorchAir converter（已验证的 Add 模板、运行器及适用范围见 [torchair.md](references/torchair.md)）或 ONNX parser 插件与离线 OM（实测模板见 [onnx-om.md](references/onnx-om.md)）；只实现本次需要的前端。`python scripts/check_opp.py <opp-root>` 检查布局及动态库依赖，这不会证明 GE 已加载注册。
5. **验证并交付**：按 [validation.md](references/validation.md) 验证实际范围。固定 shape 简单算子接入参考 [ge-smoke.md](references/ge-smoke.md) 与 `scripts/run_ge_smoke.py`；动态 shape 缩小及地址变化参考 [dynamic-shrink.md](references/dynamic-shrink.md) 与 `scripts/run_dynamic_shrink.py`，模板位于对应 assets 子目录。TorchAir 固定/动态及可变输出验证用 `scripts/run_torchair.py`，先读 [torchair.md](references/torchair.md) 确认 OPP 与图特化边界。ONNX/OM 固定 Add 的解析、打包与隔离执行用 `scripts/run_onnx_om.py`，先读 [onnx-om.md](references/onnx-om.md) 的范围及部署依赖。记录可复现命令、日志、版本、导出 manifest、构建产物、数值/模型结果及未验证项。远程日志和大产物留在任务目录，不塞入 skill。遇到动态输出越界或地址踩踏时，读 [troubleshooting.md](references/troubleshooting.md) 的排查案例。

## 必须保留的接入约束

- GE op type、实现注册名、前端 converter 的 type 一致；输入顺序、属性类型和默认值三侧一致。
- 环境配置与算子特化分别处理：C++ ABI 显式传参，host ABI 从产物识别，frame 容量/对齐从 schema 读取；核数、线程和 tile 按算子设计核对，详见 [environment.md](references/environment.md)。
- AscendC 源码并不一定包含可从 GE 调用的 host launcher。cannbot-dsl 的 frame/schema ABI 与 TileLang 的设备入口/host IR 是两种契约，不能互换。
- 导出每个特化到新目录，记录 shape/dtype/stride/constexpr、目标架构、源文件及头文件哈希。固定特化不能声称支持任意动态 shape。
- 不在 GE 下沉执行路径中隐式调用 Python JIT、把数据回读 CPU，或依赖临时编译缓存。确有运行时编译/回读需求时明确其生命周期与模型执行限制。
- planner、metadata 初始化、主辅流事件及资源清理属于算子语义。先判断算子是否需要，再实现；不要把多流复杂度强加给简单算子。
- 区分 host IR 中的 grid、dynamic UBUF 与全局 workspace；TileLang 的 dynamic UBUF 经 launch 属性配置，不能机械转成 workspace 指针。静态模型的 `Execute` 可能仅在加载时录制设备任务；模型重放的次数与 host 回调次数分别验证。
- Session 输出 Tensor 的描述字段与算子内部输出分配分开处理；本版本静态图在调用前需初始化输出 shape/format/dtype，地址可为 nullptr 以请求 GE 分配。不能假设默认构造的 Tensor 会得到完整元信息。
- 分开记录调用方 I/O 与 kernel 实际 I/O 地址；外部输出地址轮换可能只触发 GE 拷贝。若采用输出引用，先明确算子别名语义；本版本 `MakeOutputRefInput` 要求原型中对应输入与输出同名。
- OM 自包含部署需保存 binary 和 launch 元数据，或明确外部交付依赖。没有 `PortableOp` 不等于一定不能生成 OM；需核实本版本 GE 的打包行为。
- 示例中的 dtype、head 数、host ABI、核数和 AICPU 辅流配置均有具体适用范围，见 [prefix-attention-example.md](references/prefix-attention-example.md)。

## 资料优先级

以目标环境的头文件、源码和实际验证为接口依据；使用[官方开发指南](https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/custom_op/custom_op_v2/custom_op_development_guide.md)理解接口生命周期，以 [PrefixAttention 示例](https://github.com/fengz72/ascend-ops/tree/main/prefix-attention/cannbot-dsl/ge)参考已完成的工程。记录版本差异，不把某个 master 文档的接口直接当成所有 CANN 版本都具备的能力。
