# PrefixAttention 示例的可复用经验与边界

来源：[fengz72/ascend-ops/prefix-attention/cannbot-dsl/ge](https://github.com/fengz72/ascend-ops/tree/main/prefix-attention/cannbot-dsl/ge)。此处基于 2026-10-27 读取的文件总结；使用时重新记录具体 commit，main 可能变化。

## 应阅读哪些文件

- `scripts/dump_kernel_src.py`：导出主 kernel 和 AICPU planner，提取特化符号生成 `artifacts.h`。
- `CMakeLists.txt`：从签入的 codegen 源码构建 kernel 库、GE 交付件，以及辅流接口能力检查。
- `src/kernel_launcher.cc`：按 ABI schema 构造 frame，使用本次 tensor 地址/shape/stride。
- `src/prefix_attention_op.cc`：原型、`IMPL_OP` 推导、`Execute`、GE attached stream、event 生命周期。
- `prefix_attn_torch/__init__.py`：torch schema、Meta、eager bridge、OPP 定位和 converter 默认值一致性。
- `README.md`：经过验证的环境、特化范围和动态输出容量修复依赖。

## 适合复用的工程决策

1. DSL 只参与源码生成；部署交付件从 AscendC 重建，让 GE 执行不再依赖 Python DSL。
2. 从生成产物提取真实 host ABI 符号，检查 schema/ABI 版本，而不是固定粘贴历史 hash。
3. 分开审视 eager bridge 和 GE 图执行，复用 kernel launcher，但验证各自的加载和调用路径。
4. 需要时使用 GE 管理的辅流，把初始化、planner 和计算之间的依赖写成模型可以重复重放的任务。
5. 动态 shape 验证覆盖增大、缩小、再增大；输出容量不足必须扩容，不能只改逻辑描述符。

## 只属于该示例的条件

- 交付件 FP16；N1=14、N2=2、D=64 固定，T/B 动态。
- GE 路径含固定核数配置，planner 和 attention 双 launch；复用时按算子设计及设备能力重新确定核数。
- host ABI v9/schema v2；升级 DSL 时必须重查。
- 示例的编译目标按当前任务传入的架构重新配置，核对后端及设备能力。
- 示例安装脚本写当前 `$ASCEND_OPP_PATH`，本 skill 的新任务优先使用独立 OPP 根。

示例的 event key 包含 ACL context、模型辅流和节点名；wait 后 reset，避免重放误用旧信号。清理通过显式 `prefix_attn_release_attached_events()`，调用时机在相关模型执行完成/卸载之后、`GEFinalize` 之前。该函数名仅属于这个示例；其他算子按资源设计提供自己的释放入口。

README 记录动态输出增大依赖 GE `MallocOutputTensor` 容量检查/重分配修复，以及配套 Compiler/Executor。不要把这个历史环境的测试结果当成本次环境已通过；本轮没有重跑 PrefixAttention 完整数值和模型测试。

动态缩小与地址变化的复现实例见 [dynamic-shrink.md](dynamic-shrink.md)；增长后的输出越界定位见[错误排查案例](troubleshooting.md)。
