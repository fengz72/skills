# GE custom_op_v2 接入决策

本文件是实现路线，接口签名和能力以目标 CANN 头文件为准。[官方开发指南](https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/custom_op/custom_op_v2/custom_op_development_guide.md)为概念与生命周期依据。

## 接入前写清契约

在项目中记录：每个输入/输出的名称、顺序、rank、逻辑/storage shape、dtype、format、stride、是否连续、别名/in-place；每个属性的类型和默认值；所有 constexpr、特化范围、workspace、planner、stream 及目标架构。

动态维分清由编译器泛化、由多份 kernel 分派、还是仅允许固定值。缓存键至少包含影响产物的 shape/特化、dtype、layout、属性、DSL/编译器版本和目标架构。数据内容变化不得被错误缓存为固定 planner 结果。

## 按场景选择能力

| 场景 | 要实现/核实的内容 |
|---|---|
| 已编译 kernel 接入 GE | `EagerExecuteOp::Execute`；shape/dtype 推导按图编译需要提供 |
| GE 编译时生成 binary | 增加 `CompilableOp::Compile`，此阶段以编译元信息选择/编译产物 |
| OM 携带编译产物 | 已嵌入实现 SO 的 binary/元数据可随 SO_BINS 打包；对象运行时生成的状态用 `PortableOp` 保存/恢复，或明确外置依赖。实测方式见 [onnx-om.md](onnx-om.md) |
| 下沉后输入输出地址改变 | 核实 `ArgsUpdater` 或本版本提供的参数标注机制及回调时机 |
| 需要 GE 管理辅流 | 核实 `ResourceUsageReporter`、attached stream 接口和模型重放约束 |
| 数据依赖输出 shape | 编译阶段声明可推导信息/上界；执行期分配、同步及实际 shape 更新按本版本契约实现 |

shape/dtype 推导可采用实现类 `ShapeInferOp` 或版本支持的 `IMPL_OP` 注册路径。PrefixAttention 使用 `IMPL_OP`，不要为贴合文档机械改成另一种实现。读取本版本官方样例核实生命周期。

## GE 层实现

- 用 `REG_OP` 定义原型，`REG_AUTO_MAPPING_OP` 映射实现；原型、实现类、前端 type 名一致。TensorFlow 的原型生成有独立流程，不能一概套用其他前端。
- `Execute` 核对输入和属性契约，从 `ctx->GetInputTensor` 取本次输入；输出用 `MallocOutputTensor`，临时空间用 `MallocWorkSpace`。检查错误与空指针。
- 输出的逻辑 shape 与 storage shape/format 均要正确；地址来自 GE，不把 Python tensor 或历史执行地址固化到 launcher。
- 动态 shape/地址验证时区分调用方地址与 Execute 真正接收的 kernel 地址。普通外部输出轮换可能只是 GE 拷贝目的地改变，内部输出仍复用；输出引用须明确别名契约，当前 `MakeOutputRefInput` 要求被引用的输入与输出原型名称相同。
- TileLang host IR 的动态 UBUF 大小要配置到 ACL launch 属性，不同于全局 workspace；只有实际设备参数含全局临时空间时才分配并传该指针。
- 在 `ctx->GetStream()` 发起 launch；二进制加载/句柄缓存按目标设备/context 管理。多 kernel 则保留执行顺序和 workspace 生命周期。
- 需要异步 planner 时明确 metadata 初始化、主辅流双向依赖、重复执行的事件重置和模型卸载后的清理。不要默认用 host 同步消除问题。
- `Compile` 使用编译上下文元信息。若编译结果依赖输入数值，先确定本场景允许访问什么数据及缓存规则。
- `Serialize/Deserialize` 保存有版本和长度边界的 binary/元数据格式，验证数据完整性；运行时指针、stream 和句柄在恢复后重建。

## OPP 与链接

```text
<opp-root>/
└── op_graph/
    ├── include/<op_proto>.h
    └── lib/linux/<host-arch>/
        ├── lib<custom-op>.so
        └── artifacts/                 # 如果依赖独立 kernel 库/binary
```

根据目标 GE loader 的真实规则确定库名/布局；`libcust_opapi.so` 是官方/参考例子的做法，并不支持推断任意命名都能被发现。host 架构来自 `uname -m`，NPU arch 是另一维度。

```bash
export ASCEND_CUSTOM_OPP_PATH="$PWD/output${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
```

在 GE 初始化前设置，指向 OPP 根，不是 `.so` 所在目录。优先安装到任务独立 OPP 根，避免照搬示例覆盖工具链自带 OPP。对依赖 `.so` 设置可迁移 RUNPATH（例如 `$ORIGIN/artifacts`），并核对 `DT_NEEDED` 和目标机器上 `ldd` 的结果。

构建时使用当前 CANN 头文件/库。Torch 扩展单独核对 `_GLIBCXX_USE_CXX11_ABI`、torch/torch_npu 路径和版本，不无条件链接文档列出的所有库。GE Compiler 与 Executor 若来自本地修复，应保持同源配套。

## 前端接入分支

**GE 原生构图**：构图程序能看到原型/注册，创建匹配的 type，连接输入并明确输出。先单节点图验证，再替换模型子图。

**PyTorch/TorchAir**：定义 torch schema、Meta/Fake 推导和实际需要的 eager 实现，注册 FX→GE converter。converter 的 inputs/outputs/attrs 对齐 GE 原型，特别检查 schema 默认值被 export 省略后的行为。确认图 dump 中有预期 GE 节点，而非仅 eager 结果正确。两个 DSL 的固定/动态 Add 与可变输出已完成前端图及 NPU 验证，模板、运行器和适用范围见 [torchair.md](torchair.md)。

**ONNX**：对 parser 不识别的 type，提供版本配套的解析插件、domain/opset/type 映射与属性转换；同时交付 GE 实现。解析插件与 GE 计算库职责不同，目录按该版本 loader 规则验证。两个 DSL 的固定 Add 已完成 ONNX 解析、GE 节点、OM 导出/隔离加载及 NPU 验证，见 [onnx-om.md](onnx-om.md)。

**TensorFlow**：按该版本 Adapter 的算子声明、支持清单和自动原型映射流程接入；不将 TorchAir 注册要求套入此分支。

原生 GE 的两个简单 Add 已在当前容器完成实际 NPU、图编译/加载/重复执行与精度验证，见 [ge-smoke.md](ge-smoke.md)。其中静态模型 `Execute` 在 LoadGraph 录制一次；Session 输出描述符需预先初始化，shape 正确而地址为 nullptr 时可让 GE 分配地址。不要将静态调用契约与动态输出推导混用。TorchAir 的通过范围另见 [torchair.md](torchair.md)；ONNX/OM 的固定 Add 与 SO_BINS 自包含通过范围见 [onnx-om.md](onnx-om.md)；TensorFlow 与 PortableOp 实例状态序列化仍没有执行结论。

动态 shape 缩小与真实 kernel 地址变化见 [dynamic-shrink.md](dynamic-shrink.md)。在本环境动态模型中，Execute 每次调用，节点 context/实现对象复用；不要因此推断静态下沉模型的 ArgsUpdater/AnnotatedArgs 也已验证。动态输出越界及容量复用问题见[错误排查案例](troubleshooting.md)。
