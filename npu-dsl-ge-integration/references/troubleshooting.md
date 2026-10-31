# 错误排查案例

## 动态 shape 增大后输出越界或地址踩踏

**现象与状态**：已有问题反馈显示，同一 GE 模型在 shape 增大后发生地址踩踏。当前验证记录仅覆盖 shape 重复、缩小和地址变化，未复验增长路径，尚无目标环境的修复验证结论。

**定位方法**：

1. 记录相邻执行的 shape、dtype、所需字节数、输出实际分配容量和 kernel 接收的地址。重点检查 `MallocOutputTensor` 复用输出时是否检查容量、容量不足时是否重新分配，以及 launcher 是否刷新地址与动态维参数。
2. 分别追踪调用方输出 Tensor 与 kernel 输出指针。普通 Add 可能写入 GE 内部缓存后再拷贝到外部输出，调用方地址变化本身不能证明 kernel 的目标地址已更新。
3. 检查外部缓冲区容量、前后哨兵、当前逻辑尾部和旧缓冲区快照，区分容量不足、沿用旧 shape 写入和写入旧地址。数值比较与边界检查分别保留证据。
4. 核对实际加载的 GE Compiler/Executor 版本、动态库路径及哈希。[PrefixAttention 参考工程](prefix-attention-example.md)记载过容量检查/重分配修复依赖，需结合目标版本源码和实际结果判断修复状态。

**已有验证与结果解释**：[动态缩小工程](dynamic-shrink.md)对两个 DSL 使用 `4096 → 2049 → 1025 → 257 → 1`（含重复步骤），外部缓冲区固定预留 4096 个元素容量，并检查逻辑尾部与旧地址。本次缩小及地址变化检查通过。最大容量预分配可以绕开容量不足，但没有验证 GE 在增长时的扩容行为；框架修复的依据应包含容量不足分支的实际执行、足够的输出容量、正确地址、边界检查和数值结果。

## 设备初始化失败与算子执行失败的区分

若日志在 `aclrtSetDevice` 或 stream 创建阶段失败，先保留 ACL 错误码、`aclGetRecentErrMsg()`、设备状态和进程占用，确认是否已经进入自定义 Execute/kernel。已有测试记录出现过设置设备或 stream 创建失败，而部分独立进程能够正常执行；仅凭错误码不能认定是 frame 或算子数值问题。

环境状态恢复后在新目录复测，并保留首次失败与后续结果。验证脚本不自动吞掉或重试设备初始化错误；超时也记录具体阶段、时限和部分日志。

## TorchAir 当前输出正确，但持有的旧输出被覆盖

动态普通 Add 直接返回 GE 自定义节点时，每次的即时精度检查通过，但 kernel 复用内部输出，已返回 Tensor 的值随下一次执行改变。编译图内追加 `clone()` 仍出现覆盖；检查本次安装的 `aten.clone.default` converter，实际映射为 `ge.Identity`。

在自定义 converter 后显式加入 `ge.TensorMove`，并在优化后 GE 图断言该节点存在。两个 DSL 在默认优化配置下的缩小序列及旧输出持有检查均通过。不能仅凭函数名判断是否物化；按目标版本检查 lowering 和输出生命周期。

## TorchAir 多模型共享实现对象导致 last-shape 误判

测试一个模型缩小后，再编译另一个从初始尺寸开始的模型，共享 C++ 实现对象上的 `last_n` 会把新模型误判为增长。同一实现对象可对应不同 node context。修正为 Execute 每次读取当前 context、校验本算子 shape 范围，不保存在对象级别跨模型的形状状态；缩小序列由测试运行器检查。需要模型级状态时按真实 context 生命周期管理。这项修正解决的是状态归属，不证明 GE 增长扩容问题已修复。

## TorchAir 注册、原型查询与 fixture 符号冲突

先初始化 OPP 环境并加载本示例注册库，再通过 TorchAir backend 实际降低节点。GE Session 初始化前调用内部原型查询可能报 `Session is not initialized`；这不能直接证明 OPP 缺失，应区分初始化顺序与注册错误。

固定/动态 fixture 都导出 `SmokeLaunch`，但签名不同。全局加载两套库可能发生符号串用；验证器以独立子进程隔离。生产库可改用独立符号，不能把这条示例限制推广为所有 OPP 必须分进程。

## 可变输出 schema 和单位维特化

可变示例 schema 是 `Tensor(a!) out_buffer -> ()`，Meta 与 converter 均返回 None；functionalization 负责传递修改，GE 原型按同名输入/输出建立引用。用户 Tensor 别名正确不代表 kernel 直接写用户地址，应同时检查回调地址与旧缓冲区。

长度 1 在本次 TorchDynamo 中触发独立图。记录编译次数、FX 与 GE dump；同一模型的缩小结论仅覆盖没有重新编译的段。不要为了维持预期图数而把符号维转为 Python int，或把实现对象共享误当作模型复用。

## ONNX/OM 导出成功，独立加载找不到 kernel SO

原固定 Add 的 GE 实现依赖外部 `libcannbot_add.so`。ATC 成功生成 OM，但移除自定义 OPP 后 ACL 报 `dlopen custom op so ... libcannbot_add.so: cannot open shared object file`。同库中的 TileLang 实现也被这个 ELF 依赖阻断。

GE 打包实现 SO 不等于递归打包 `DT_NEEDED`。按 [ONNX/OM 实测](onnx-om.md) 将每个后端独立构建，kernel 与 launch 元数据嵌入实现库，再确认完整 kernel/实现字节存在于 OM，并在开发目录不可见的新进程中验证。若选择外置依赖，交付完整依赖包并记录其发现方式；不能把只设置 LD_LIBRARY_PATH 后成功加载称为 OM 单文件自包含。

## ATC SoC 名和 ONNX 格式与显示/默认值不同

`npu-smi` 的简化芯片名传给 ATC 曾触发 `rtSetSocVersion` 不支持；用目标 ACL 返回的完整 SoC 名后编译通过。记录探测值并参数化传入，不将本次具体型号写进模板。

ONNX 的一维输入在默认入口被赋成 NCHW，元素数正确但自定义 Execute 因 ND 契约不符拒绝加载。日志应分别显示 rank、元素数、dtype、format；本例改用 `--input_format=ND` 后加载与精度通过。形状相同不能证明 format 相容。

## OM 打包与 PortableOp 的边界

本次固定 Add 不实现 PortableOp，将常量 binary 和 launch 元数据嵌入实现 SO，由 SO_BINS 携带后可以自包含执行。PortableOp 解决实例状态的保存/恢复，两种机制可组合；不要因缺少 PortableOp 就拒绝验证已有完整嵌入的离线产物，也不要把 SO_BINS 验证通过写成 Serialize/Deserialize 回调通过。当前离线模板同时静态链接注册支持并检查 pull creator ABI，保证 OM 内实现能恢复注册。
